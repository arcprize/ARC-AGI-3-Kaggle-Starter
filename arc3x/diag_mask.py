"""Is the cell mask calibrated on level 0 still valid on later levels?

The mask is ``varies & ~clock``, learned from probe walks at the level-0 root.
A pixel that never moved during those walks is excluded from the archive key
permanently. If later levels animate a different region of the screen, the key
goes blind there and the search degenerates - which would explain why games
clear level 0 in a handful of actions and then stall.

Run: python arc3x/diag_mask.py cd82 sp80 vc33 ar25
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc3x.cell import calibrate
from arc3x.explore import Explorer, discover_games
from arc3x.twin import Act, Twin, default_env_dir


def main() -> None:
    wanted = sys.argv[1:] or ["cd82", "sp80", "vc33", "ar25"]
    env_dir = default_env_dir()
    by_prefix = {g.split("-")[0]: g for g in discover_games(env_dir)}

    print(f"{'game':6s} {'level':>5s} {'informative':>11s} {'varying':>8s} "
          f"{'clock':>6s} {'new vs L0':>10s} {'blind px':>9s}")
    print("-" * 62)
    for pre in wanted:
        gid = by_prefix.get(pre)
        if gid is None:
            print(f"{pre}: not found")
            continue
        twin = Twin(gid, env_dir)
        root = twin.snapshot()
        Twin.step_game(root, Act(0))

        c0 = calibrate(root, seed=0)
        base0 = c0.mask.copy()
        print(f"{pre:6s} {0:5d} {c0.n_informative:11d} {c0.n_varying:8d} "
              f"{c0.n_clock:6d} {'-':>10s} {'-':>9s}")

        # Walk the game forward one level at a time using the searcher itself,
        # then re-calibrate at each new root and compare.
        ex = Explorer(twin, cell=c0, seed=0, verbose=False)
        cur = root
        for lvl in range(1, 4):
            res = ex.solve_level(cur, lvl - 1, 100, 45.0)
            if res.plan is None:
                print(f"{pre:6s} {lvl:5d}  (level {lvl - 1} not solved in 45s, stop)")
                break
            g = copy.deepcopy(cur)
            for a in res.plan:
                Twin.step_game(g, a)
            cur = g
            cl = calibrate(cur, seed=0)
            new = int((cl.mask & ~base0).sum())
            blind = int((cl.mask & ~base0).sum())
            print(f"{pre:6s} {lvl:5d} {cl.n_informative:11d} {cl.n_varying:8d} "
                  f"{cl.n_clock:6d} {new:10d} {blind:9d}")


if __name__ == "__main__":
    main()
