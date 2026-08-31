"""Run the graded suite and say where it is losing. The measurement of record.

    .venv/Scripts/python.exe arc3x/suite.py                     # all 25, parallel
    .venv/Scripts/python.exe arc3x/suite.py --budget 400        # quick pass
    .venv/Scripts/python.exe arc3x/suite.py lp85 cd82 -w 1      # debuggable
    .venv/Scripts/python.exe arc3x/suite.py --json out.json     # keep the detail

WHY A SCRIPT AND NOT A ONE-LINER
--------------------------------
A serial pass over 25 games at a 3000-action budget takes the better part of an
hour, and a measurement that slow gets run once instead of after every change -
which is how a regression survives to the end of a session. Every game is fully
independent (its own ``Twin``, its own engine, no shared state, no writes), so
the parallel result is exact rather than approximate, and on a 12-core box the
same pass takes minutes.

WHAT TO READ IN THE OUTPUT
--------------------------
The mean is the leaderboard statistic and it is the least informative line here,
because a mean of 0.142 over 25 games is 23 zeros with one game working. Read
the level histogram and the stall reasons instead: score follows from levels
cleared, and the scoring formula caps a level-0-only run at 3.52 over this set no
matter how efficient it becomes. Two levels per game is 10.57. So "reached L1+"
is the number that matters.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc3x.agent import play_agent
from arc3x.explore import discover_games
from arc3x.graded import run_suite, stall_summary
from arc3x.twin import default_env_dir

# The evaluation set is 110 private games nobody has looked at; these 25 are a
# development set. So the danger is not that a rule fails - it is that a rule
# succeeds here for a reason that does not exist there. A fixed alphabetical
# split is the cheapest guard: tune only against TUNE, and treat HOLD as if it
# were the private set. If a change helps TUNE and not HOLD, it is a coincidence
# that has been fitted, and it must not ship.
#
# Fixed, not random, and not re-drawn: a split that gets re-rolled after a
# disappointing result is not a holdout, it is a lottery. Every third game, so
# the two halves are interleaved rather than clumped by name.
_ALL = [
    "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52",
    "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26", "sc25", "sk48",
    "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30",
]
HOLD = [g for i, g in enumerate(_ALL) if i % 3 == 2]   # 8 games, never tuned on
TUNE = [g for i, g in enumerate(_ALL) if i % 3 != 2]   # 17 games, fair game


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    ap.add_argument("games", nargs="*", help="short ids, e.g. lp85 cd82; default all")
    ap.add_argument("--budget", type=int, default=3000, help="action cap per game")
    ap.add_argument(
        "--seconds",
        type=float,
        default=900.0,
        help="wall clock per game; a safety valve only - if it binds, the run is "
        "not a comparable measurement and the summary says so",
    )
    ap.add_argument("-w", "--workers", type=int, default=10, help="1 for tracebacks")
    ap.add_argument("--json", type=Path, default=None, help="write full reports here")
    ap.add_argument(
        "--split",
        choices=("all", "tune", "hold", "both"),
        default="all",
        help="'tune' = the 17 games it is legitimate to iterate on; 'hold' = the "
        "8 kept clean; 'both' = run each and print the gap, which is the "
        "overfitting readout",
    )
    a = ap.parse_args(argv)

    env = default_env_dir()
    ids = discover_games(env)
    short = {g.split("-")[0]: g for g in ids}

    def pick(names: list[str]) -> list[str]:
        return [short[n] for n in names if n in short]

    if a.games:
        want = set(a.games)
        chosen = [g for g in ids if g.split("-")[0] in want]
        missing = want - {g.split("-")[0] for g in chosen}
        if missing:
            print(f"unknown games: {sorted(missing)}", file=sys.stderr)
            return 2
        groups = [("chosen", chosen)]
    elif a.split == "tune":
        groups = [("tune", pick(TUNE))]
    elif a.split == "hold":
        groups = [("hold", pick(HOLD))]
    elif a.split == "both":
        groups = [("tune", pick(TUNE)), ("hold", pick(HOLD))]
    else:
        groups = [("all", ids)]

    if not any(g for _n, g in groups):
        print("no games found", file=sys.stderr)
        return 2

    means: dict[str, float] = {}
    everything: list[dict] = []
    for name, gids in groups:
        print(f"\n=== {name}: {len(gids)} games ===", flush=True)
        t = time.perf_counter()
        out = run_suite(
            play_agent,
            gids,
            env_dir=env,
            workers=a.workers,
            budget=a.budget,
            seconds=a.seconds,
        )
        means[name] = out["mean_score"]
        everything.extend(out["reports"])
        print(f"({time.perf_counter() - t:.0f}s, {a.workers} workers, budget {a.budget})")

    if "tune" in means and "hold" in means:
        t, h = means["tune"], means["hold"]
        print(
            f"\nOVERFIT CHECK   tune {t:.3f}   hold {h:.3f}   "
            f"ratio {h / t if t else float('nan'):.2f}"
        )
        print(
            "  a ratio near 1 means the mechanism is general. hold well below\n"
            "  tune means it was fitted to games it was allowed to see, and the\n"
            "  110 private games will behave like hold, not like tune."
        )

    if a.json:
        a.json.write_text(json.dumps(everything, indent=1), encoding="utf-8")
        print(f"detail -> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
