"""Validate the twin's three load-bearing assumptions, on all 25 games.

Run: .venv/Scripts/python.exe arc3x/verify_twin.py

Assumption 1 (FREE SEARCH): stepping a deepcopy leaves the real game and the
scorecard untouched. If this fails, search costs scored actions and the whole
architecture collapses.

Assumption 2 (DETERMINISM): the same action sequence from the same start always
produces the same frames. If this fails, a plan found in the twin cannot be
replayed against the graded gateway.

Assumption 3 (INTROSPECTION): the engine tells us the exact legal action set,
including click coordinates. Without it, ACTION6 is a 4096-way branch.
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from arc3x.twin import Act, Twin, default_env_dir


def game_ids(env_dir: Path) -> list[str]:
    """Every game in the dataset, from its metadata - no hardcoded list."""
    out = []
    for meta in sorted(env_dir.rglob("*/*/metadata.json")):
        import json

        try:
            out.append(json.loads(meta.read_text(encoding="utf-8"))["game_id"])
        except Exception:
            continue
    return out


def main() -> int:
    env_dir = default_env_dir()
    ids = game_ids(env_dir)
    print(f"env_dir: {env_dir}")
    print(f"games:   {len(ids)}\n")

    hdr = f"{'game':16} {'lv':>3} {'valid':>5} {'steps/s':>8} {'iso':>4} {'det':>4} {'clicks':>6}"
    print(hdr)
    print("-" * len(hdr))

    failures: list[str] = []
    totals = {"iso": 0, "det": 0, "intro": 0}

    for gid in ids:
        try:
            t = Twin(gid, env_dir)
        except Exception as e:
            print(f"{gid:16} LOAD FAILED: {type(e).__name__}: {e}")
            failures.append(f"{gid}: load")
            continue

        obs0 = t.current()
        valid = obs0.valid
        n_click = sum(1 for a in valid if a.is_click)

        # --- Assumption 3: introspection gave us a usable action set.
        intro_ok = len(valid) > 0
        totals["intro"] += intro_ok

        # --- Assumption 1: isolation. Hammer a clone, check the original.
        frame_before = obs0.frame.copy()
        count_before = getattr(t.game, "_action_count", 0)
        clone = t.snapshot()
        rng = np.random.default_rng(0)
        n = 300
        t0 = time.perf_counter()
        cur = valid
        for _ in range(n):
            if not cur:
                break
            a = cur[int(rng.integers(len(cur)))]
            o = Twin.step_game(clone, a)
            cur = o.valid if o.valid else valid
            if o.terminal:
                Twin.step_game(clone, Act(0))
                cur = valid
        dt = time.perf_counter() - t0
        sps = n / dt if dt > 0 else 0.0

        after = t.current()
        iso_ok = (
            np.array_equal(frame_before, after.frame)
            and getattr(t.game, "_action_count", 0) == count_before
        )
        card = t._arcade.get_scorecard(t.env.scorecard_id) if hasattr(t.env, "scorecard_id") else None
        if card is not None and getattr(card, "total_actions", 0) != 0:
            iso_ok = False
        totals["iso"] += iso_ok

        # --- Assumption 2: determinism. Same plan twice -> same end frame.
        plan: list[Act] = []
        g1 = t.snapshot()
        cur = valid
        rng2 = np.random.default_rng(7)
        end1 = obs0
        for _ in range(40):
            if not cur:
                break
            a = cur[int(rng2.integers(len(cur)))]
            plan.append(a)
            end1 = Twin.step_game(g1, a)
            cur = end1.valid if end1.valid else valid
            if end1.terminal:
                break
        g2 = t.snapshot()
        o2 = None
        for a in plan:
            o2 = Twin.step_game(g2, a)
            if o2.terminal:
                break
        det_ok = o2 is not None and np.array_equal(end1.frame, o2.frame) and end1.level == o2.level
        totals["det"] += det_ok

        if not (iso_ok and det_ok and intro_ok):
            bad = [k for k, v in (("iso", iso_ok), ("det", det_ok), ("intro", intro_ok)) if not v]
            failures.append(f"{gid}: {','.join(bad)}")

        print(
            f"{gid:16} {t.n_levels:3d} {len(valid):5d} {sps:8,.0f} "
            f"{'OK' if iso_ok else 'FAIL':>4} {'OK' if det_ok else 'FAIL':>4} {n_click:6d}"
        )

    n = len(ids)
    print(f"\nisolation  (free search): {totals['iso']}/{n}")
    print(f"determinism (replayable): {totals['det']}/{n}")
    print(f"introspection (actions):  {totals['intro']}/{n}")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nAll three assumptions hold on all games.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
