"""Print what ``moved_objects`` failed to match, button by button. A debugging aid.

Two dev games have four working movement buttons and the model only ever finds
two of them. The suspected cause is that ``moved_objects`` demands an *exact*
rigid shift - same pixel count, same bounding box, same mask - so a sprite that
**turns to face** the way it is walking never matches itself and is reported as
one object vanishing plus another appearing. No displacement, no vote, and the
button looks dead.

That is a guess. This prints the evidence: for every transition, the pairs that
matched exactly, and then every unmatched departure and arrival with its size,
bounding box shape, bounding-box centre and mass centroid. If the guess is right
the departures and arrivals come in same-colour pairs with equal size and
different shape, and the four numbers say which displacement measure would have
recovered the step.

    .venv/Scripts/python.exe arc3x/why_moves.py wa30 --presses 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc3x.explore import discover_games
from arc3x.percept import background, moved_objects
from arc3x.twin import Act, Twin, default_env_dir


def _shape(frame: np.ndarray, color: int, top: int, left: int, size: int) -> str:
    """Describe the component of ``color`` whose bbox starts at (top,left)."""
    return f"c{color} n{size} @({top},{left})"


def _describe(frame: np.ndarray, m, tag: str) -> str:
    """Size, bbox and both centres of one reported object."""
    from arc3x.percept import _component

    mask, t, l, n = _component(frame, m.top, m.left, limit=512)
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    cy = t + float(ys.mean()) if n else -1.0
    cx = l + float(xs.mean()) if n else -1.0
    return (
        f"{tag} c{m.color:2d} n{n:3d} bbox {h}x{w} @({t},{l})"
        f"  bcentre ({t + (h - 1) / 2:.1f},{l + (w - 1) / 2:.1f})"
        f"  centroid ({cy:.1f},{cx:.1f})"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("--presses", type=int, default=3, help="times to press each button")
    a = ap.parse_args(argv)

    env = default_env_dir()
    gid = next(g for g in discover_games(env) if g.startswith(a.game))
    twin = Twin(gid, env)
    obs = twin.current()
    # One reset so the board is the real level 0 rather than the title card.
    obs = Twin.step_game(twin.game, Act(5))
    bg = background(obs.frame)
    print(f"{gid}  background={bg}  declared={[x.aid for x in obs.valid]}")

    for aid in sorted({x.aid for x in obs.valid}):
        if aid == 6:
            continue
        print(f"\n=== button {aid} " + "=" * 50)
        for k in range(a.presses):
            before = obs.frame
            obs = Twin.step_game(twin.game, Act(aid))
            after = obs.frame
            if not (before != after).any():
                print(f"  press {k}: no change")
                continue
            mv, van, app = moved_objects(before, after)
            mv = [m for m in mv if m.color != bg]
            van = [m for m in van if m.color != bg]
            app = [m for m in app if m.color != bg]
            print(f"  press {k}: changed {int((before != after).sum())} px")
            for m in mv:
                print(f"    MATCHED c{m.color:2d} n{m.size:3d} d=({m.dy:+d},{m.dx:+d})")
            for m in van:
                print("    " + _describe(before, m, "LEFT   "))
            for m in app:
                print("    " + _describe(after, m, "ARRIVED"))
            if obs.terminal:
                obs = Twin.step_game(twin.game, Act(5))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
