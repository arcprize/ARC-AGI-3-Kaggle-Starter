"""Do the searched plans actually replay? Audit a plans file end to end.

The local score is computed from what the search *claims*. The gateway pays only
for what a fresh engine will *replay*. Those were not the same number: an audit
of sweep25_v3 found 10 of 13 plans reproducing their claimed level count and 3
losing exactly their last level (tu93 1->0, sp80 2->1, lp85 5->4), because a
rollout could drop an action from `plan` that it had already applied to the
engine, and compress() verified its own edits but never its input.

This script is the regression test for that. It replays every plan from a fresh
Twin - the same thing the gateway will do - and reports claimed vs replayed
levels, plus the honest score recomputed from the replayed counts.

    python arc3x/verify_plans.py sweep25_v3.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc3x.explore import discover_games, game_score
from arc3x.twin import Act, Twin, default_env_dir


def replay(game_id: str, plan: list[Act], env_dir: Path) -> tuple[int, list[int], bool]:
    """Replay a plan in a fresh twin. Returns (levels, actions_per_level, died)."""
    twin = Twin(game_id, env_dir)
    g = twin.snapshot()
    Twin.step_game(g, Act(0))  # every graded run opens with RESET
    per: list[int] = []
    level = 0
    since = 0
    for a in plan:
        obs = Twin.step_game(g, a)
        since += 1
        if obs.game_over:
            return level, per, True
        if obs.level > level or obs.won:
            per.append(since)
            since = 0
            level = obs.level if obs.level > level else level + 1
            if obs.won:
                break
    return level, per, False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("plans", help="json written by sweep.py / solve_game")
    args = ap.parse_args()

    env_dir = default_env_dir()
    by_prefix = {g.split("-")[0]: g for g in discover_games(env_dir)}

    blob = json.load(open(args.plans))
    results = blob["results"] if isinstance(blob, dict) else blob

    print(f"{'game':8s} {'claimed':>8s} {'replay':>7s} {'claim sc':>9s} "
          f"{'real sc':>8s}  status")
    print("-" * 62)

    ok = bad = 0
    claimed_total = real_total = 0.0
    for r in sorted(results, key=lambda r: r["game_id"]):
        pre = r["game_id"].split("-")[0]
        gid = by_prefix.get(pre)
        claim = int(r["levels_solved"])
        claim_sc = float(r["est_score"])
        claimed_total += claim_sc
        if gid is None or not r.get("plan"):
            real_total += claim_sc if claim == 0 else 0.0
            print(f"{pre:8s} {claim:8d} {'-':>7s} {claim_sc:9.2f} "
                  f"{0.0 if claim else claim_sc:8.2f}  {'no plan' if claim else 'zero'}")
            continue
        plan = [Act(int(a), int(x), int(y)) for a, x, y in r["plan"]]
        got, per, died = replay(gid, plan, env_dir)
        real_sc = game_score(r["baselines"], per)
        real_total += real_sc
        if got >= claim:
            ok += 1
            status = "ok"
        else:
            bad += 1
            status = f"LOST {claim - got}" + (" (died)" if died else "")
        print(f"{pre:8s} {claim:8d} {got:7d} {claim_sc:9.2f} {real_sc:8.2f}  {status}")

    n = max(1, len(results))
    print("-" * 62)
    print(f"{ok} plans reproduce their claimed levels, {bad} do not")
    print(f"claimed mean score {claimed_total / n:.3f}   "
          f"replay-verified mean score {real_total / n:.3f}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
