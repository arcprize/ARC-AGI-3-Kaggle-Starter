"""Harvest (state, action) pairs from solved plans and train the student.

    python arc3x/train_student.py --plans sweep25_v2.json --out arc3x/student.npz

Reads a sweep's plans, replays each in its own twin to recover the frames the
searcher saw, then fits one shared policy over all games. Prints held-out
accuracy against the random-choice baseline, which is the only honest way to
read the number: a game offering two legal moves gives 0.50 for free.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc3x.explore import discover_games
from arc3x.student import Student, harvest_plan
from arc3x.twin import Act, Twin, default_env_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", default="sweep25_v2.json")
    ap.add_argument("--out", default="arc3x/student.npz")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    blob = json.load(open(args.plans))
    results = blob["results"] if isinstance(blob, dict) else blob
    solved = [r for r in results if r.get("plan") and r.get("levels_solved", 0) > 0]
    print(f"plans file  : {args.plans}")
    print(f"games solved: {len(solved)}/{len(results)}")
    if not solved:
        print("nothing solved -> nothing to imitate. Run a sweep first.")
        return

    env_dir = default_env_dir()
    games = discover_games(env_dir)
    by_prefix = {gid.split("-")[0]: gid for gid in games}

    xs: list[np.ndarray] = []
    ys: list[int] = []
    ms: list[np.ndarray] = []
    used: list[str] = []
    t0 = time.perf_counter()
    for r in solved:
        pre = r["game_id"].split("-")[0]
        gid = by_prefix.get(pre)
        if gid is None:
            print(f"  {pre}: not found locally, skipped")
            continue
        twin = Twin(gid, env_dir)
        root = twin.snapshot()
        # Reproduce solve_game's setup exactly: plans are relative to the state
        # *after* the opening RESET, and do not contain that RESET themselves.
        frame0 = Twin.step_game(root, Act(0)).frame
        plan = [Act(int(a), int(x), int(y)) for a, x, y in r["plan"]]
        pairs = harvest_plan(root, plan, frame0)
        for x, y, m in pairs:
            xs.append(x)
            ys.append(y)
            ms.append(m)
        used.append(pre)
        print(
            f"  {pre:6s} plan {len(plan):4d} actions -> {len(pairs):4d} examples "
            f"(levels {r['levels_solved']}/{r['n_levels']})"
        )

    if not xs:
        print("no usable examples")
        return

    x = np.stack(xs)
    ys_a = np.array(ys, dtype=np.intp)
    print(
        f"\ndataset: {len(ys_a):,} examples x {x.shape[1]:,} features "
        f"({x.nbytes / 1e6:.0f} MB)  harvested in {time.perf_counter() - t0:.1f}s"
    )
    branch = float(np.mean([len(m) for m in ms]))
    print(f"mean legal actions per state: {branch:.2f}")

    model = Student.new(hidden=args.hidden, seed=args.seed)
    rnd = model.baseline_acc(ms)
    print(f"random-choice accuracy      : {rnd:.3f}   <- the bar to beat\n")
    hist = model.fit(
        x, ys_a, ms, epochs=args.epochs, lr=args.lr, seed=args.seed, log=True
    )
    model.games = tuple(used)
    model.save(args.out)

    val = hist["val_acc"][-1]
    print(
        f"\nheld-out accuracy {val:.3f} vs random {rnd:.3f} "
        f"= {val / max(rnd, 1e-9):.2f}x  ->  {args.out}"
    )
    if val <= rnd:
        print("NOT better than random: do not enable the prior yet.")


if __name__ == "__main__":
    main()
