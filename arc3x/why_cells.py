"""Which state abstraction makes "have I been here before?" mean anything.

WHY THIS EXISTS
---------------
``relive.py`` archives cells and rewards landing in a new one. That only works if
many action sequences collapse to the same cell. Measured after wiring it in:

    ka59  informative=514   128 cells in ~430 actions
    bp35  informative=1542  227 cells in ~430 actions
    cn04  informative=1890  295 cells in ~430 actions

Nearly one new cell per action, i.e. the key is effectively bijective and novelty
is not a signal - the search degenerates into a depth-first walk that pays for
restarts and learns nothing from them. ``cell.py`` hit exactly this with raw
frames (60 distinct keys in 60 steps on tn36) and fixed it by dropping clocks;
dropping clocks is evidently not enough on these games.

The question this file answers is *which* coarsening to use, by counting distinct
keys per abstraction over the same walk. Fewer keys is not automatically better -
an abstraction that maps everything to one cell is useless in the other
direction - so the number to read is the **collapse ratio**, keys per step, with
an eye on whether the abstraction still separates states the agent must
distinguish. Two states that differ only in which way a sprite is facing should
collapse; two states with the avatar in different rooms must not.

The candidates, in order of increasing coarseness:

  raw        every pixel                       (the baseline that fails)
  informative
             clock-free pixels                 (what relive does today)
  tile       informative, max-pooled by the
             game's own step size              (removes sub-tile rendering)
  pos        the avatar's box alone            (pure position)
  poscount   position plus the per-colour
             pixel census                      (position plus what is left)
  census     the per-colour pixel census alone (what is left, ignoring where)

``tile`` is the interesting one because the pool size is *learned* - ``Mechanics.tile``
is the gcd of the observed move deltas, so an 8px-grid game pools by 8 and a
1px game pools by 1, with no constant to tune.

Run:
    .venv/Scripts/python.exe arc3x/why_cells.py --steps 200
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from collections import Counter

import numpy as np

from arc3x.mind import Mechanics
from arc3x.relive import Clockless
from arc3x.twin import Twin, default_env_dir

GAMES = [
    "ka59",
    "bp35",
    "cn04",
    "dc22",
    "g50t",
    "sk48",
    "m0r0",
    "tn36",
    "su15",
    "sb26",
]


def _h(payload: bytes) -> bytes:
    return hashlib.blake2b(payload, digest_size=8).digest()


def census(frame: np.ndarray, mask: np.ndarray) -> bytes:
    vals, counts = np.unique(frame[mask], return_counts=True)
    return b"".join(int(v).to_bytes(1, "big") + int(n).to_bytes(3, "big")
                    for v, n in zip(vals, counts))


def pooled(frame: np.ndarray, mask: np.ndarray, p: int) -> bytes:
    """Max-pool the informative pixels, with everything else blanked to -1.

    Blanking first is what makes this a *coarsening of the informative mask*
    rather than a coarsening of the raw frame: a clock inside a block would
    otherwise dominate the max and reintroduce exactly what the mask removed.
    """
    f = frame.astype(np.int8).copy()
    f[~mask] = -1
    if p <= 1:
        return f.tobytes()
    h, w = f.shape
    hh, ww = h // p * p, w // p * p
    blocks = f[:hh, :ww].reshape(hh // p, p, ww // p, p).max(axis=(1, 3))
    return blocks.tobytes()


def survey(game_id: str, env_dir: str, *, steps: int, seed: int) -> dict:
    """One random walk, every abstraction counted over the identical states."""
    tw = Twin(game_id, env_dir)
    root = copy.deepcopy(tw.game)
    g = copy.deepcopy(root)
    rng = np.random.default_rng(seed)
    valid = Twin.valid_actions(g)

    ck = Clockless()
    m = Mechanics()
    frames: list[np.ndarray] = []
    boxes: list[tuple | None] = []
    prev: np.ndarray | None = None
    for i in range(steps):
        if not valid:
            break
        a = valid[int(rng.integers(len(valid)))]
        obs = Twin.step_game(g, a)
        if obs.terminal:
            break
        if prev is not None:
            m.observe(a.aid, prev, obs.frame, level_up=False, died=False)
        if i % 10 == 9:
            m.settle()
        ck.feed(obs.frame)
        frames.append(obs.frame)
        prev = obs.frame
        valid = obs.valid or valid
    m.settle()
    ck.freeze()
    mask = ck.mask
    assert mask is not None
    for f in frames:
        boxes.append(m.where(f))

    tile = max(1, int(m.tile))
    full = np.ones_like(mask, dtype=bool)
    keys: dict[str, set[bytes]] = {k: set() for k in
                                  ("raw", "informative", f"tile({tile})", "pos",
                                   "poscount", "census")}
    for f, box in zip(frames, boxes):
        keys["raw"].add(_h(pooled(f, full, 1)))
        keys["informative"].add(_h(pooled(f, mask, 1)))
        keys[f"tile({tile})"].add(_h(pooled(f, mask, tile)))
        pos = b"none" if box is None else bytes((box[0] & 0xFF, box[1] & 0xFF))
        keys["pos"].add(_h(pos))
        cen = census(f, mask)
        keys["poscount"].add(_h(pos + cen))
        keys["census"].add(_h(cen))
    return {
        "game": game_id,
        "steps": len(frames),
        "informative": int(mask.sum()),
        "tile": tile,
        "avatar": int(m.avatar),
        "keys": {k: len(v) for k, v in keys.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--games", default=",".join(GAMES))
    a = ap.parse_args()
    env = str(default_env_dir())

    rows = []
    for gid in a.games.split(","):
        try:
            rows.append(survey(gid, env, steps=a.steps, seed=a.seed))
        except Exception as exc:  # a broken game must not sink the survey
            print(f"{gid:6s} FAILED {type(exc).__name__}: {exc}")
    if not rows:
        return 1

    names = list(rows[0]["keys"].keys())
    print(f"\ndistinct keys per {a.steps}-step walk  (lower = more collapse)")
    print(f"{'game':7s}{'steps':>6s}{'info':>6s}{'tile':>5s}" +
          "".join(f"{n:>14s}" for n in names))
    for r in rows:
        ks = r["keys"]
        print(
            f"{r['game']:7s}{r['steps']:>6d}{r['informative']:>6d}{r['tile']:>5d}"
            + "".join(f"{ks[n]:>14d}" for n in names)
        )

    print("\nkeys per step, averaged over games  (1.00 = every state unique)")
    for n in names:
        tot = sum(r["keys"][n] for r in rows)
        stp = sum(r["steps"] for r in rows)
        print(f"  {n:14s} {tot / max(1, stp):.3f}")
    print(
        "\nan abstraction is usable when this is well under 1.00 *and* it still\n"
        "separates positions - 'pos' collapses hardest and is blind to everything\n"
        "the avatar is carrying or has changed, so read it as the floor, not the goal."
    )
    lost = [r["game"] for r in rows if r["avatar"] < 0]
    if lost:
        print(f"no avatar found, so 'pos'/'poscount' are meaningless on: {', '.join(lost)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
