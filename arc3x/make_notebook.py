"""Generate the Kaggle submission notebook, self-contained.

    python arc3x/make_notebook.py --plans sweep25_v3.json --out arc3x_submission.ipynb

The arc3x modules (86 KB of Python) are embedded as %%writefile cells, so the
notebook needs no attached utility script or extra dataset - only the official
competition dataset, which supplies the game sources and the arc_agi/arcengine
wheels. Fewer moving parts is worth the notebook bloat here.

The notebook has two phases and the split matters:

  Phase 1 runs entirely on the LOCAL twins built from the competition dataset.
  It costs zero graded actions, so it can use hours of wall clock. It produces
  one plan per game family.

  Phase 2 connects to the graded gateway, and for each of the ~110 runs it
  identifies which family it is looking at, then replays that family's plan
  with per-step frame verification. Clone IDs are opaque (c000, c001, ...), so
  identification is by opening-frame fingerprint - measured to separate all 25
  families with a worst-case cross-family similarity of 0.8904 against a 0.97
  floor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODULES = ["twin", "cell", "explore", "student", "selfplay_data", "runner", "sweep"]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
    }


SETUP = '''\
# ---------------------------------------------------------------------------
# Locate the competition dataset: game sources + the arc_agi/arcengine wheels.
# ---------------------------------------------------------------------------
import os, sys, glob, json, time, subprocess
from pathlib import Path

os.environ.setdefault("ONLY_RESET_LEVELS", "true")   # RESET restarts the LEVEL

def find_input(*names):
    for base in ("/kaggle/input", "."):
        for n in names:
            hits = glob.glob(f"{base}/**/{n}", recursive=True)
            if hits:
                return sorted(hits, key=len)[0]
    return None

ENV_DIR = find_input("environment_files")
print("environment_files:", ENV_DIR)

# Install the shipped wheels if arc_agi is not already importable.
try:
    import arc_agi  # noqa: F401
    print("arc_agi already importable")
except ImportError:
    for pat in ("arc_agi*.whl", "arcengine*.whl", "re_arc*.whl"):
        for w in glob.glob(f"/kaggle/input/**/{pat}", recursive=True):
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", w],
                           check=False)
    import arc_agi  # noqa: F401

sys.path.insert(0, "/kaggle/working")
Path("/kaggle/working/arc3x").mkdir(parents=True, exist_ok=True)
Path("/kaggle/working/arc3x/__init__.py").write_text("")
print("ready")
'''

PHASE1 = '''\
# ---------------------------------------------------------------------------
# PHASE 1 - free search on the local twins. Zero graded actions are spent here,
# so this may use as much wall clock as we can afford. Each game gets BUDGET_S
# seconds; games run in parallel across the available cores.
#
# Aggregate throughput measured ~550-1000 simulated actions/sec/core. For
# comparison, the LLM agent needs 17.6 s per single action.
# ---------------------------------------------------------------------------
import multiprocessing as mp

BUDGET_S = float(os.environ.get("ARC3X_BUDGET", 300))   # seconds per game
WORKERS  = min(4, max(1, (os.cpu_count() or 2) - 1))

from arc3x.sweep import _one
from arc3x.explore import discover_games

games = discover_games(Path(ENV_DIR))
print(f"{len(games)} local game families, {BUDGET_S:.0f}s each, {WORKERS} workers")
print(f"estimated wall clock: {len(games) * BUDGET_S / WORKERS / 60:.0f} min")

from concurrent.futures import ProcessPoolExecutor, as_completed
results = []
t0 = time.perf_counter()
with ProcessPoolExecutor(max_workers=WORKERS) as pool:
    futs = {pool.submit(_one, (g, str(ENV_DIR), BUDGET_S, 0)): g for g in games}
    for f in as_completed(futs):
        r = f.result()
        results.append(r)
        per = ",".join(str(n) for n in r["actions_per_level"]) or "-"
        print(f"  {r['game_id']:16s} {r['levels_solved']}/{r['n_levels']} levels "
              f"score {r['est_score']:6.2f}  [{per}]")

mean = sum(r["est_score"] for r in results) / max(1, len(results))
print(f"\\nlocal mean estimated score: {mean:.3f}   ({(time.perf_counter()-t0)/60:.1f} min)")
json.dump({"mean_est_score": mean, "results": results},
          open("/kaggle/working/plans.json", "w"))
'''

PHASE2 = '''\
# ---------------------------------------------------------------------------
# PHASE 2 - play the graded gateway.
#
# For each graded run: RESET, identify the family from the opening frame
# (clone IDs are opaque so names cannot be trusted), then replay that family's
# plan while checking every frame against what the twin predicted. On the first
# divergence we stop replaying - a plan for a different variant is worse than
# no plan - and hand over to the student policy, then to random.
# ---------------------------------------------------------------------------
import numpy as np
import arc_agi
from arc_agi import OperationMode
from arc3x.runner import build_families, gateway_as_graded, play_game
from arc3x.explore import game_score

BASE_URL = os.environ.get("ARC_BASE_URL", "http://gateway:8001")
ACTION_CAP = int(os.environ.get("ARC3X_ACTION_CAP", 800))

families = build_families(Path(ENV_DIR), plans_path="/kaggle/working/plans.json")
print(f"{len(families)} families, "
      f"{sum(1 for f in families if f.plan)} with plans")

student = None
try:
    from arc3x.student import Student
    sp = find_input("student*.npz")
    if sp:
        student = Student.load(sp)
        print(f"student policy loaded from {sp}")
except Exception as exc:
    print(f"no student policy ({type(exc).__name__}); fallback will be random")

arcade = arc_agi.Arcade(
    operation_mode=OperationMode.COMPETITION,
    arc_base_url=BASE_URL,
    environments_dir="",
)
card = arcade.create_scorecard()
envs = arcade.get_environments()
print(f"gateway offers {len(envs)} runs")

played, rng = [], np.random.default_rng(0)
for i, info in enumerate(envs):
    gid = info.game_id
    try:
        env = arcade.make(gid, scorecard_id=card)
        if env is None:
            print(f"  [{i+1}/{len(envs)}] {gid}: make() returned None"); continue
        res = play_game(gateway_as_graded(env), families, graded_game_id=gid,
                        action_cap=ACTION_CAP, student=student, rng=rng)
        played.append(res)
        print(f"  [{i+1}/{len(envs)}] {gid} fam={res.family} via={res.how} "
              f"src={res.source} actions={res.actions_used} "
              f"levels={res.levels_reached}"
              + (f" DIVERGED@{res.diverged_at}" if res.diverged_at is not None else ""))
    except Exception as exc:
        print(f"  [{i+1}/{len(envs)}] {gid}: {type(exc).__name__}: {exc}")

try:
    arcade.close_scorecard(card)
except Exception as exc:
    print(f"close_scorecard: {type(exc).__name__}: {exc}")

n_div = sum(1 for r in played if r.diverged_at is not None)
print(f"\\nplayed {len(played)} runs; {n_div} diverged from their family plan")
print(f"identified by frame: {sum(1 for r in played if r.how=='frame')}, "
      f"by name: {sum(1 for r in played if r.how=='name')}, "
      f"unknown: {sum(1 for r in played if r.how=='unknown')}")
json.dump([r.__dict__ for r in played], open("/kaggle/working/played.json", "w"), default=str)
'''

HEADER = """\
# ARC-AGI-3 - free search in a local twin, verified replay against the gateway

The competition dataset ships **the full Python source of all 25 games**, so the
game can be run in-process as a *twin*: `copy.deepcopy` is a complete state
snapshot, and stepping a clone leaves the graded action counter at zero.

That makes search **free**. It runs at ~550-1000 simulated actions/sec/core
against 17.6 seconds per action through a 27B LLM - about 12,000x. No model is
called anywhere in this notebook, so the failure that turned experiment 11's
2.68 local score into 0.60 on Kaggle (vLLM prefill timeout on a shared GPU)
cannot happen here.

**Scoring drives the design.** Each *completed* level scores
`min(115, (baseline/actions)^2 * 100)` and an uncompleted level scores **zero**.
The game score is a weighted mean with 1-indexed level weights, so on a 6-level
game clearing only level 0 caps you at 5.5 while clearing four levels is worth
47.6. Depth dominates; efficiency is the multiplier on top.

| stage | what it does | cost |
|---|---|---|
| calibrate | learn which pixels are state, not clock/HUD | free |
| search | Go-Explore over the twin, per level | free |
| compress | shorten the plan, verified by replay | free |
| replay | send the plan to the gateway, checking each frame | **graded** |
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="arc3x_submission.ipynb")
    ap.add_argument("--src", default="arc3x")
    args = ap.parse_args()

    src = Path(args.src)
    cells = [md(HEADER), code(SETUP)]
    for name in MODULES:
        body = (src / f"{name}.py").read_text(encoding="utf-8")
        cells.append(code(f"%%writefile /kaggle/working/arc3x/{name}.py\n{body}"))
    cells += [md("## Phase 1 - free search (no graded actions)"), code(PHASE1),
              md("## Phase 2 - graded replay"), code(PHASE2)]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    Path(args.out).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"wrote {args.out}: {len(cells)} cells ({n_code} code), "
          f"{Path(args.out).stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
