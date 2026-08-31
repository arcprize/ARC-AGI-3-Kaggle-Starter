"""Self-imitation data: learn from the whole search, not just the wins.

WHY
---
Training only on winning plans gave 223 examples across 25 games - 14 games
times ~16 actions. That is nowhere near enough to make a policy sharp, and it
throws away almost everything the search learned. A 300-second search visits
100,000+ states per game; only a dozen of them end up in the final plan.

The signal being discarded: **every action that discovered a new archive cell
is a verified example of a move that made progress.** Not "progress" by a
hand-written heuristic - progress by the search's own state-abstraction, which
already excludes clock/HUD pixels. That is the self-imitation signal Go-Explore
normally uses for its robustification phase, and it is three orders of magnitude
more data than the plans alone.

Three label classes are recorded, all free and all exactly correct:

  ``new``   the action reached a cell never seen before  -> imitate
  ``level`` the action completed a level                 -> imitate, weighted up
  ``dead``  the action ended the game                    -> avoid

``dead`` matters more than it looks. tn36 dies at exactly step 61 every run, and
several games are 0/N purely because rollouts keep walking into deaths. A policy
that only knows what to *do* cannot help there; one that also knows what kills
extends effective search depth directly.

MEMORY
------
Frames are stored as their 256 feature indices (int16), not as 64x64 arrays:
512 bytes per example instead of 4 KB, so 200,000 examples fit in ~100 MB. The
one-hot is reconstructed at training time.

WEIGHTING
---------
Level completions are rare and worth far more than novelty (game score is a
weighted mean with 1-indexed level weights, so level 5 is worth 6x level 0), so
they carry a larger sample weight. Deaths enter as negative examples via a
separate head-free trick: the label is *every legal action except the fatal
one*, spread uniformly, which pushes mass away from the killer without needing
a second output head.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from arc3x.student import CELLS, N_IN, feature_idx, slot_of
from arc3x.twin import Act

W_NEW = 1.0
W_LEVEL = 6.0
W_DEAD = 2.0


@dataclass
class Recorder:
    """Collects (frame, action, legal-set, weight, kind) during a search.

    Capped and reservoir-sampled so a long search cannot exhaust memory and so
    the kept sample stays representative of the whole run rather than only its
    first minutes.
    """

    cap: int = 200_000
    seed: int = 0
    idx: list[np.ndarray] = field(default_factory=list)
    slot: list[int] = field(default_factory=list)
    legal: list[np.ndarray] = field(default_factory=list)
    weight: list[float] = field(default_factory=list)
    kind: list[str] = field(default_factory=list)
    seen: int = 0
    _rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def _store(
        self, at: int, fi: np.ndarray, slot: int, legal: np.ndarray, w: float, kind: str
    ) -> None:
        if at == len(self.idx):
            self.idx.append(fi)
            self.slot.append(slot)
            self.legal.append(legal)
            self.weight.append(w)
            self.kind.append(kind)
        else:
            self.idx[at] = fi
            self.slot[at] = slot
            self.legal[at] = legal
            self.weight[at] = w
            self.kind[at] = kind

    def add(
        self,
        frame: np.ndarray,
        action: Act,
        legal_actions: tuple[Act, ...],
        kind: str,
    ) -> None:
        """Record one decision. ``kind`` is 'new', 'level', or 'dead'."""
        if frame is None or len(legal_actions) < 2:
            return  # a forced move teaches nothing and inflates accuracy
        slots = sorted({slot_of(a) for a in legal_actions})
        if len(slots) < 2:
            return
        lab = slot_of(action)
        if lab not in slots:
            return

        if kind == "dead":
            # Push mass away from the fatal action: relabel to the *other* legal
            # actions. Implemented as one example per alternative, which keeps a
            # single softmax head and needs no extra machinery.
            others = [s for s in slots if s != lab]
            if not others:
                return
            w = W_DEAD / len(others)
            fi = feature_idx(frame).astype(np.int16)
            arr = np.array(slots, dtype=np.int16)
            for s in others:
                self._offer(fi, s, arr, w, kind)
            return

        w = W_LEVEL if kind == "level" else W_NEW
        self._offer(feature_idx(frame).astype(np.int16), lab, np.array(slots, dtype=np.int16), w, kind)

    def _offer(
        self, fi: np.ndarray, slot: int, legal: np.ndarray, w: float, kind: str
    ) -> None:
        """Reservoir sampling so the kept set represents the whole search."""
        assert self._rng is not None
        self.seen += 1
        if len(self.idx) < self.cap:
            self._store(len(self.idx), fi, slot, legal, w, kind)
            return
        j = int(self._rng.integers(self.seen))
        if j < self.cap:
            self._store(j, fi, slot, legal, w, kind)

    # -- export ------------------------------------------------------------

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for k in self.kind:
            out[k] = out.get(k, 0) + 1
        return out

    def to_npz(self, path: str) -> None:
        """Ragged legal-sets are flattened with an offsets array."""
        if not self.idx:
            np.savez_compressed(path, empty=np.array([1]))
            return
        lens = np.array([len(m) for m in self.legal], dtype=np.int32)
        np.savez_compressed(
            path,
            idx=np.stack(self.idx).astype(np.int16),
            slot=np.array(self.slot, dtype=np.int16),
            legal=np.concatenate(self.legal).astype(np.int16),
            legal_len=lens,
            weight=np.array(self.weight, dtype=np.float32),
            seen=np.array([self.seen], dtype=np.int64),
        )


def load_npz(paths: list[str]) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    """Load and concatenate recorder dumps into (idx, slot, legal, weight)."""
    idx_l, slot_l, legal_l, w_l = [], [], [], []
    for p in paths:
        d = np.load(p)
        if "idx" not in d:
            continue
        idx_l.append(d["idx"])
        slot_l.append(d["slot"])
        w_l.append(d["weight"])
        flat, lens = d["legal"], d["legal_len"]
        off = 0
        for n in lens:
            legal_l.append(flat[off : off + n].astype(np.intp))
            off += int(n)
    if not idx_l:
        return np.zeros((0, 256), np.int16), np.zeros(0, np.int16), [], np.zeros(0, np.float32)
    return (
        np.concatenate(idx_l),
        np.concatenate(slot_l),
        legal_l,
        np.concatenate(w_l),
    )


def onehot_batch(idx: np.ndarray) -> np.ndarray:
    """Rebuild the dense one-hot input from stored feature indices."""
    n = len(idx)
    x = np.zeros((n, N_IN), dtype=np.float32)
    rows = np.repeat(np.arange(n), idx.shape[1])
    x[rows, idx.ravel().astype(np.intp)] = 1.0
    return x
