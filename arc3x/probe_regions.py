"""Do the click-only games show a reference panel and a canvas? Measure before building.

WHY ASK
-------
Eleven of the 25 dev games have no steerable avatar, so a move-and-plan agent
scores zero on 44% of the set. Since the leaderboard is a mean, that caps any
possible score at 14/25 = 56% of the per-game average, which puts 10+ out of
reach arithmetically no matter how well the walking games are played. The click
games therefore have to be covered.

But "cover the click games" is not a plan until we know what they *are*. The
cheap hypothesis is that many of them are the copy-the-pattern genre: the board
shows you the answer somewhere and asks you to reproduce it elsewhere. If that is
true it is worth a lot, because mismatch between two regions is a scalar that
ratchets, so it drops straight into the objective machinery that already exists
and needs no new search, no new planner and no avatar.

If it is false, this script costs ten minutes and saves ten hours.

WHAT IT MEASURES
----------------
For every canonical way of cutting the board into two equal panels - left/right,
top/bottom, and the same two after ignoring a border - how similar are the two
halves? The signature of a reference-and-canvas pair is *high but imperfect*
agreement: identical halves are decoration, unrelated halves are two different
things, and 60-95% agreement is one panel that is most of the way to matching the
other. Reflections are tested too, because "mirror this" is as common as "copy
this".

    .venv/Scripts/python.exe arc3x/probe_regions.py         # the 11 click games
    .venv/Scripts/python.exe arc3x/probe_regions.py all     # all 25, as a control
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from arc3x.explore import discover_games
from arc3x.graded import GradedRun
from arc3x.twin import default_env_dir

# Measured 2026-08-22 by arc3x/measure_dream.py: wiggling every button produces no
# rigid translation on any of these, so there is nothing to steer.
NO_AVATAR = [
    "cd82", "ft09", "lf52", "lp85", "r11l",
    "s5i5", "sb26", "su15", "tn36", "tr87", "vc33",
]


def _agree(a: np.ndarray, b: np.ndarray) -> float:
    return float((a == b).mean()) if a.size and a.shape == b.shape else 0.0


def splits(frame: np.ndarray, inset: int = 0) -> dict[str, float]:
    """Agreement between the two halves of the board, several ways.

    ``inset`` trims a border first: most of these games draw a frame around the
    play area, and a shared border inflates agreement without meaning anything.
    """
    f = frame[inset : frame.shape[0] - inset, inset : frame.shape[1] - inset] if inset else frame
    h, w = f.shape
    out: dict[str, float] = {}
    if w >= 2:
        l, r = f[:, : w // 2], f[:, w - w // 2 :]
        out["l|r"] = _agree(l, r)
        out["l|mirror(r)"] = _agree(l, r[:, ::-1])
    if h >= 2:
        t, b = f[: h // 2, :], f[h - h // 2 :, :]
        out["t|b"] = _agree(t, b)
        out["t|mirror(b)"] = _agree(t, b[::-1, :])
    return out


def probe(game_id: str, env_dir: Path) -> dict:
    run = GradedRun(game_id, env_dir)
    obs = run.reset()
    frame = obs.frame
    best: tuple[float, str, int] = (0.0, "-", 0)
    for inset in (0, 1, 2, 4):
        for name, score in splits(frame, inset).items():
            # Perfect agreement is decoration (two identical empty halves), not a
            # puzzle. What we are hunting is a panel most of the way to matching.
            if score >= 0.999:
                continue
            if score > best[0]:
                best = (score, name, inset)
    ncol = len(np.unique(frame))
    return {
        "game": game_id.split("-")[0],
        "best": best[0],
        "how": best[1],
        "inset": best[2],
        "ncol": ncol,
        "acts": sorted(a for a in obs.available_actions),
    }


def main() -> int:
    env_dir = default_env_dir()
    games = discover_games(env_dir)
    want = NO_AVATAR if "all" not in sys.argv[1:] else None
    if want is not None:
        games = [g for g in games if g.split("-")[0] in want]

    rows = []
    for gid in games:
        try:
            r = probe(gid, env_dir)
        except Exception as exc:
            print(f"{gid.split('-')[0]:6s} {type(exc).__name__}: {exc}", flush=True)
            continue
        rows.append(r)
        flag = "PANEL?" if 0.55 <= r["best"] <= 0.97 else "      "
        print(
            f"{r['game']:6s} best-agreement={r['best']:6.1%} via {r['how']:12s} "
            f"inset={r['inset']} colours={r['ncol']:2d} actions={r['acts']} {flag}",
            flush=True,
        )

    hot = [r for r in rows if 0.55 <= r["best"] <= 0.97]
    print(
        f"\n{len(hot)}/{len(rows)} games look like a reference panel plus a canvas: "
        f"{[r['game'] for r in hot]}"
    )
    print(
        "read: >=0.55 and <=0.97 is the band where one half is most of the way to\n"
        "matching the other. Below that the halves are unrelated; above it they are\n"
        "the same decoration twice. A hit here means mismatch-count is a usable\n"
        "objective for that game, with no avatar and no new planner."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
