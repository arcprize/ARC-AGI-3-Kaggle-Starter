"""Search every game in parallel and report the estimated leaderboard score.

Each game is fully independent - separate engine object, separate archive - so
games parallelise perfectly across processes. The graded Kaggle run gets ~12
hours and ~4 cores, which at ~550 simulated actions/sec/core is on the order of
75 million simulated actions: roughly 3 million per game. For comparison, the
LLM agent at 17.6 s/action could afford about 2,400 actions in total.

The plans this writes are replayed verbatim against the graded environment, so
nothing here needs the LLM at all.

    .venv/Scripts/python.exe arc3x/sweep.py --budget 300 --workers 4 --out plans.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc3x.explore import discover_games, game_score, solve_game
from arc3x.twin import default_env_dir


def _one(args: tuple[str, str, float, int]) -> dict:
    """Worker entry point: search one game and return a plain-dict result."""
    game_id, env_dir, budget_s, seed = args
    t0 = time.perf_counter()
    try:
        sol = solve_game(
            game_id,
            env_dir=Path(env_dir),
            budget_s=budget_s,
            seed=seed,
            verbose=False,
        )
        return {
            "game_id": sol.game_id,
            "plan": [[a.aid, a.x, a.y] for a in sol.plan],
            "actions_per_level": sol.actions_per_level,
            "baselines": sol.baselines,
            "levels_solved": sol.levels_solved,
            "n_levels": len(sol.baselines),
            "est_score": sol.est_score,
            "steps": sol.steps,
            "seconds": sol.seconds,
        }
    except Exception as exc:  # a single broken game must not sink the sweep
        return {
            "game_id": game_id,
            "plan": [],
            "actions_per_level": [],
            "baselines": [],
            "levels_solved": 0,
            "n_levels": 0,
            "est_score": 0.0,
            "steps": 0,
            "seconds": time.perf_counter() - t0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=300.0, help="seconds per game")
    ap.add_argument("--workers", type=int, default=0, help="0 = cpu_count-1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--games", default="", help="comma-separated id prefixes")
    ap.add_argument("--out", default="plans.json")
    args = ap.parse_args()

    env_dir = default_env_dir()
    ids = discover_games(env_dir)
    if args.games:
        wanted = [w.strip() for w in args.games.split(",") if w.strip()]
        ids = [g for g in ids if any(g.startswith(w) for w in wanted)]
    if not ids:
        print("no matching games")
        return 1

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"env_dir : {env_dir}")
    print(f"games   : {len(ids)}   budget {args.budget:.0f}s each   workers {workers}")
    print(f"wall-clock estimate: {len(ids) * args.budget / workers / 60:.0f} min\n")

    t0 = time.perf_counter()
    payload = [(g, str(env_dir), args.budget, args.seed) for g in ids]
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_one, p): p[0] for p in payload}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            done = len(results)
            tag = f"[{done:2d}/{len(ids)}]"
            if r.get("error"):
                print(f"{tag} {r['game_id']:16s} ERROR {r['error'][:60]}")
            else:
                per = ",".join(str(n) for n in r["actions_per_level"]) or "-"
                print(
                    f"{tag} {r['game_id']:16s} {r['levels_solved']}/{r['n_levels']} levels  "
                    f"score {r['est_score']:6.2f}  actions/level [{per}]  "
                    f"{r['steps']:,} sim steps"
                )

    results.sort(key=lambda r: -r["est_score"])
    mean = sum(r["est_score"] for r in results) / len(results)
    solved = sum(1 for r in results if r["levels_solved"] > 0)
    lvls = sum(r["levels_solved"] for r in results)

    print(f"\n{'=' * 64}")
    print(f"games with >=1 level solved : {solved}/{len(results)}")
    print(f"total levels completed      : {lvls}")
    print(f"MEAN ESTIMATED SCORE        : {mean:.3f}")
    print(f"wall clock                  : {(time.perf_counter() - t0) / 60:.1f} min")
    print(f"{'=' * 64}\n")
    print(f"{'game':16s} {'lvls':>6s} {'score':>7s}")
    for r in results:
        print(
            f"{r['game_id']:16s} {r['levels_solved']}/{r['n_levels']:<4} "
            f"{r['est_score']:7.2f}"
        )

    Path(args.out).write_text(
        json.dumps({"mean_est_score": mean, "results": results}, indent=1),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
