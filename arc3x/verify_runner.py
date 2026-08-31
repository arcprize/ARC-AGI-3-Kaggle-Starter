"""End-to-end test of the graded path, with no gateway.

Phase 2 of the submission does three things that can each fail silently:

  1. identify which family an opaque clone id (``c000``, ``c001``, ...) is,
     from the opening frame alone;
  2. replay that family's plan while checking every frame against the twin's
     prediction, and stop on the first divergence;
  3. count levels the way the scorer counts them (``levels_completed``).

``twin_as_graded`` lets all three run against a local twin, so the whole path is
testable offline. The clone id handed to ``play_game`` here is deliberately
opaque, so name matching is impossible and identification has to work off the
frame fingerprint - exactly the situation on Kaggle.

What this cannot test: whether the real gateway's clones are byte-identical to
the local families, and the exact ``ONLY_RESET_LEVELS`` semantics of the remote
engine. Both are unobservable without the gateway, which is why replay is
frame-verified rather than trusted.

    python arc3x/verify_runner.py sweep25_v3.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from arc3x.explore import game_score
from arc3x.runner import build_families, play_game, twin_as_graded
from arc3x.twin import default_env_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("plans")
    ap.add_argument("--cap", type=int, default=800)
    args = ap.parse_args()

    env_dir = default_env_dir()
    families = build_families(env_dir, plans_path=args.plans)
    with_plan = [f for f in families if f.plan]
    print(f"{len(families)} families, {len(with_plan)} with plans, cap {args.cap}\n")

    blob = json.load(open(args.plans))
    results = blob["results"] if isinstance(blob, dict) else blob
    claimed = {r["game_id"].split("-")[0]: r for r in results}

    print(f"{'clone':6s} {'true fam':9s} {'ident':9s} {'sim':>7s} {'claim':>6s} "
          f"{'got':>4s} {'acts':>5s}  status")
    print("-" * 68)

    ok = bad = mis = 0
    total_real = 0.0
    rng = np.random.default_rng(0)
    for i, fam in enumerate(with_plan):
        clone_id = f"c{i:03d}"  # opaque, exactly like taaf's clone_game_ids
        graded = twin_as_graded(fam.game_id, env_dir)
        res = play_game(
            graded, families, graded_game_id=clone_id,
            action_cap=args.cap, student=None, rng=rng, verbose=False,
        )
        claim = int(claimed[fam.prefix]["levels_solved"])
        if res.family != fam.prefix:
            mis += 1
            status = f"MISIDENTIFIED as {res.family}"
        elif res.levels_reached >= claim:
            ok += 1
            status = "ok"
            total_real += float(claimed[fam.prefix]["est_score"])
        else:
            bad += 1
            status = f"LOST {claim - res.levels_reached}"
            if res.diverged_at is not None:
                status += f" diverged@{res.diverged_at}"
        print(f"{clone_id:6s} {fam.prefix:9s} {str(res.family):9s} "
              f"{res.similarity:7.4f} {claim:6d} {res.levels_reached:4d} "
              f"{res.actions_used:5d}  {status}")

    print("-" * 68)
    print(f"identified correctly: {len(with_plan) - mis}/{len(with_plan)}")
    print(f"replayed to the claimed level: {ok}/{len(with_plan)}  "
          f"(lost {bad}, misidentified {mis})")
    print(f"mean score over all {len(families)} families from verified replays: "
          f"{total_real / max(1, len(families)):.3f}")
    return 0 if (bad == 0 and mis == 0) else 2


if __name__ == "__main__":
    raise SystemExit(main())
