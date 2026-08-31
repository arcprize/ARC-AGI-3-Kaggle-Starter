"""A student policy distilled from the searcher's own solutions.

WHY THIS EXISTS
---------------
The Go-Explore search in ``explore.py`` picks actions uniformly at random from
the untried set. That is enough to clear level 0 on the games where a short
random line happens to work, and hopeless everywhere else: the number of
distinct action sequences of length L grows like b**L, so a game needing a
precise 40-action opening is unreachable by chance no matter how long we run.

The fix is the second half of Go-Explore, the part usually called
*robustification*: once search has found solutions, train a policy to imitate
them, then put that policy back into the search loop as an action prior. Search
generates data -> data trains the policy -> the policy makes search reach
further -> which generates harder data. That loop is the only thing here that
compounds.

This is also the answer to "what if the LLM fails". It does fail: experiment 11
scored 2.68 locally and 0.60 on Kaggle because vLLM prefill timed out on a
shared GPU. This model has no such failure mode - it is ~2 MB of numpy floats,
runs on CPU in ~0.1 ms, and cannot time out, rate-limit, or hallucinate.

WHY AN MLP AND NOT A CONVNET
----------------------------
ARC-AGI-3 renders to a fixed 64x64 grid with a fixed camera. Absolute position
is meaningful (the HUD is always in row 1, the play area is always centred), so
the translation invariance a convnet buys is not worth its cost, and there is
no autograd here to hide the cost. Instead:

  frame (64x64, values 0..15)
    -> 4x4 max-pool                       (16x16, the games' own render grid)
    -> one-hot over colours                (16 planes x 16 x 16 = 4096)
    -> dense 4096 x 256, ReLU
    -> dense 256 x 261  = 5 simple actions + 256 coarse click cells
    -> softmax RESTRICTED TO THE LEGAL ACTIONS

That last line is what makes such a small model useful. We never ask it "what
is the best action in the abstract"; we ask "of these 7 legal moves, which one
did the search take in states that looked like this". Masking the softmax to
the legal set removes the entire burden of learning legality and turns a
261-way problem into a ~7-way one.

Click actions are bucketed to a 4x4 pixel grid (256 cells) because ARC games
place interactive elements on a coarse grid; predicting an exact pixel would
split near-identical examples across neighbouring outputs.

WHAT IT TRAINS ON
-----------------
Compressed winning plans, replayed in the twin. Compression matters: the raw
search plan wanders, and imitating a wander teaches wandering. The compressed
plan is close to shortest, so every (state, action) pair in it is a step that
provably had to happen. Labels are free and exactly correct - no reward
shaping, no human labels, no LLM.

NO PER-GAME KNOWLEDGE
---------------------
One set of weights for all games. Nothing keyed on game id, and the input is
only the rendered frame, so a game the model has never seen still gets a
prior. That is not a nicety, it is the whole reason this file exists: the
scored set is **110 private games the agent has never seen**, and the 25 games
that ship with the dataset are a development instrument, not families the
scored set is drawn from.

That fact is what rules out the alternative. Go-Explore in ``explore.py`` needs
a local engine file to ``deepcopy``; hidden games have none, and the gateway
runs ``environments_dir=""`` and cannot be cloned or rewound. So free search
cannot run on a scored game at all, and a searched plan cannot be replayed into
one either. Search's only route to the leaderboard is as a **teacher** whose
free, provably-optimal traces train these weights - which then transfer,
because they read pixels rather than game ids.
"""

from __future__ import annotations

import copy
import json
import pathlib
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from arc3x.twin import Act, Twin

# -- geometry ---------------------------------------------------------------

GRID = 64
POOL = 4
CELLS = GRID // POOL          # 16 -> a 16x16 coarse grid
N_COLOR = 16
N_SIMPLE = 5                  # ACTION1..ACTION5
N_CLICK = CELLS * CELLS       # 256 coarse click targets
N_OUT = N_SIMPLE + N_CLICK    # 261
N_IN = N_COLOR * CELLS * CELLS  # 4096


def featurise(frame: np.ndarray) -> np.ndarray:
    """frame (64,64) ints -> flat one-hot of the 4x4-max-pooled colour grid.

    Max-pool (not mean) because these frames are flat colour blocks; averaging
    invents colours that are not in the palette, while max keeps a real one.
    """
    idx = feature_idx(frame)
    x = np.zeros(N_IN, dtype=np.float32)
    x[idx] = 1.0
    return x


def feature_idx(frame: np.ndarray) -> np.ndarray:
    """The 256 nonzero positions of ``featurise``, as indices.

    The one-hot input has exactly one active colour per coarse cell, so
    ``x @ w1`` is a sum of 256 rows of ``w1`` - an embedding-bag lookup, 16x
    cheaper than the dense matmul and numerically identical. Inference happens
    inside the search's hot loop, so this matters.
    """
    f = np.asarray(frame)
    if f.shape != (GRID, GRID):
        out = np.zeros((GRID, GRID), dtype=np.int16)
        h, w = min(GRID, f.shape[0]), min(GRID, f.shape[1])
        out[:h, :w] = f[:h, :w]
        f = out
    blocks = f.reshape(CELLS, POOL, CELLS, POOL).max(axis=(1, 3))
    blocks = np.clip(blocks, 0, N_COLOR - 1).astype(np.intp)
    return (blocks.ravel() * (CELLS * CELLS) + _CELL_OFFSET).astype(np.intp)


_CELL_OFFSET = np.arange(CELLS * CELLS, dtype=np.intp)


def slot_of(a: Act) -> int:
    """Map an action to one of the 261 output slots."""
    if a.is_click:
        r = min(CELLS - 1, max(0, int(a.y) // POOL))
        c = min(CELLS - 1, max(0, int(a.x) // POOL))
        return N_SIMPLE + r * CELLS + c
    return min(N_SIMPLE - 1, max(0, int(a.aid) - 1))


# -- model ------------------------------------------------------------------


@dataclass
class Student:
    """4096 -> 256 -> 261 MLP. Explicit forward/backward; no autograd needed."""

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    hidden: int = 256
    trained_on: int = 0
    games: tuple[str, ...] = ()

    @classmethod
    def new(cls, hidden: int = 256, seed: int = 0, n_in: int = N_IN) -> "Student":
        """``n_in`` is the width of the input encoding.

        Defaults to the 4096 of ``featurise``, but ``arc3x/features.py`` offers
        narrower palette-invariant encodings, and the whole point of comparing
        them is that only the input meaning changes - so the width has to be a
        parameter rather than a constant.
        """
        rng = np.random.default_rng(seed)
        # He init on layer 1 (ReLU), small init on the output layer so the
        # initial prior is near-uniform and cannot hurt the search.
        return cls(
            w1=(rng.standard_normal((n_in, hidden)) * np.sqrt(2.0 / n_in)).astype(np.float32),
            b1=np.zeros(hidden, dtype=np.float32),
            w2=(rng.standard_normal((hidden, N_OUT)) * 0.01).astype(np.float32),
            b2=np.zeros(N_OUT, dtype=np.float32),
            hidden=hidden,
        )

    # -- inference ---------------------------------------------------------

    def logits(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(x @ self.w1 + self.b1, 0.0)
        return h @ self.w2 + self.b2

    def prior(self, frame: np.ndarray, actions: Sequence[Act]) -> np.ndarray:
        """P(action | frame), normalised over ONLY the given legal actions.

        Returns a uniform distribution if ``actions`` is empty or degenerate,
        so a caller can always use the result without special-casing.
        """
        n = len(actions)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        if n == 1:
            return np.ones(1, dtype=np.float32)
        z = self.logits(featurise(frame))
        sel = np.array([slot_of(a) for a in actions], dtype=np.intp)
        v = z[sel]
        # Several legal clicks can land in one coarse cell; that is intended -
        # they share a prior, and the search breaks the tie at random.
        v -= v.max()
        p = np.exp(v)
        s = p.sum()
        if not np.isfinite(s) or s <= 0:
            return np.full(n, 1.0 / n, dtype=np.float32)
        return (p / s).astype(np.float32)

    # -- training ----------------------------------------------------------

    def fit(
        self,
        x: np.ndarray,
        slots: np.ndarray,
        masks: list[np.ndarray],
        *,
        epochs: int = 30,
        lr: float = 0.05,
        batch: int = 64,
        wd: float = 1e-5,
        seed: int = 0,
        val_frac: float = 0.15,
        log: bool = True,
    ) -> dict[str, list[float]]:
        """Masked-softmax cross-entropy by SGD with momentum.

        ``masks[i]`` holds the legal output slots for example ``i``. Loss is
        computed only over those slots, so the model is never penalised for
        putting mass on an action that was not offered - it only has to rank
        the real choices.
        """
        rng = np.random.default_rng(seed)
        n = len(slots)
        idx = rng.permutation(n)
        n_val = max(1, int(n * val_frac)) if n > 20 else 0
        val, tr = idx[:n_val], idx[n_val:]

        m1 = np.zeros_like(self.w1)
        m1b = np.zeros_like(self.b1)
        m2 = np.zeros_like(self.w2)
        m2b = np.zeros_like(self.b2)
        mom = 0.9
        hist: dict[str, list[float]] = {"loss": [], "train_acc": [], "val_acc": []}

        for ep in range(epochs):
            rng.shuffle(tr)
            tot = 0.0
            for s in range(0, len(tr), batch):
                bi = tr[s : s + batch]
                xb = x[bi]
                hpre = xb @ self.w1 + self.b1
                h = np.maximum(hpre, 0.0)
                z = h @ self.w2 + self.b2

                # Masked softmax + gradient, per example (masks vary in size).
                dz = np.zeros_like(z)
                for j, i in enumerate(bi):
                    mk = masks[i]
                    zz = z[j, mk]
                    zz = zz - zz.max()
                    p = np.exp(zz)
                    p /= p.sum()
                    hit = int(np.searchsorted(mk, slots[i]))
                    if hit >= len(mk) or mk[hit] != slots[i]:
                        continue  # label not legal (should not happen)
                    tot -= float(np.log(max(p[hit], 1e-9)))
                    p[hit] -= 1.0
                    dz[j, mk] = p
                dz /= max(1, len(bi))

                gw2 = h.T @ dz + wd * self.w2
                gb2 = dz.sum(axis=0)
                dh = dz @ self.w2.T
                dh[hpre <= 0] = 0.0
                gw1 = xb.T @ dh + wd * self.w1
                gb1 = dh.sum(axis=0)

                m2 = mom * m2 + gw2
                m2b = mom * m2b + gb2
                m1 = mom * m1 + gw1
                m1b = mom * m1b + gb1
                self.w2 -= lr * m2
                self.b2 -= lr * m2b
                self.w1 -= lr * m1
                self.b1 -= lr * m1b

            hist["loss"].append(tot / max(1, len(tr)))
            hist["train_acc"].append(self._acc(x, slots, masks, tr))
            hist["val_acc"].append(self._acc(x, slots, masks, val) if n_val else float("nan"))
            if log and (ep % 5 == 4 or ep == epochs - 1):
                print(
                    f"  epoch {ep + 1:3d}  loss {hist['loss'][-1]:.4f}  "
                    f"train {hist['train_acc'][-1]:.3f}  val {hist['val_acc'][-1]:.3f}"
                )

        self.trained_on = len(tr)
        return hist

    def _acc(
        self, x: np.ndarray, slots: np.ndarray, masks: list[np.ndarray], which: np.ndarray
    ) -> float:
        if len(which) == 0:
            return float("nan")
        ok = 0
        for i in which:
            z = self.logits(x[i])
            mk = masks[i]
            if z[mk].argmax() == int(np.searchsorted(mk, slots[i])):
                ok += 1
        return ok / len(which)

    def baseline_acc(self, masks: list[np.ndarray], which: Iterable[int] | None = None) -> float:
        """Accuracy of picking uniformly at random from the legal set.

        This is the number the model has to beat; without it a 0.4 accuracy is
        uninterpretable, since a game with 2 legal moves gives 0.5 for free.
        """
        ws = list(range(len(masks))) if which is None else list(which)
        if not ws:
            return float("nan")
        return float(np.mean([1.0 / max(1, len(masks[i])) for i in ws]))

    # -- persistence -------------------------------------------------------

    def save(self, path: str | pathlib.Path) -> None:
        np.savez_compressed(
            path,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            meta=np.array(
                json.dumps({"hidden": self.hidden, "trained_on": self.trained_on,
                            "games": list(self.games)})
            ),
        )

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "Student":
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        return cls(
            w1=d["w1"], b1=d["b1"], w2=d["w2"], b2=d["b2"],
            hidden=int(meta["hidden"]), trained_on=int(meta["trained_on"]),
            games=tuple(meta.get("games", ())),
        )


# -- data harvesting --------------------------------------------------------


def harvest_plan(
    root: Any, plan: Sequence[Act], frame0: np.ndarray | None = None
) -> list[tuple[np.ndarray, int, np.ndarray]]:
    """Replay one plan in the twin, emitting (features, label, legal-mask).

    The state is recorded *before* each action, which is what a policy sees at
    decision time. Examples where only one action is legal are dropped: they
    carry no preference information and would dominate the accuracy figure.

    ``frame0`` is the frame at ``root``. The engine only renders as a side
    effect of ``perform_action``, so there is no way to read a clone's frame
    without spending a (free, simulated) action; pass it in if you have it,
    otherwise the first action of the plan simply yields no training example.
    """
    g = copy.deepcopy(root)
    valid = Twin.valid_actions(g)
    frame = frame0
    out: list[tuple[np.ndarray, int, np.ndarray]] = []
    for a in plan:
        if frame is not None and valid and len(valid) > 1:
            slots = sorted({slot_of(v) for v in valid})
            lab = slot_of(a)
            if lab in slots:
                out.append((featurise(frame), lab, np.array(slots, dtype=np.intp)))
        obs = Twin.step_game(g, a)
        if obs.terminal:
            break
        frame = obs.frame
        valid = obs.valid or valid
    return out
