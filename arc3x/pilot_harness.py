"""Run the Kaggle ``Pilot`` policy against local twins, without an LLM.

This is a diagnostic, not a leaderboard proxy.  The production notebook gives
the pilot Qwen's action/frame history and falls back to Qwen whenever the pilot
abstains.  This harness instead gives it only its own history, which is useful
for answering narrowly testable questions:

* Does the observer learn enough mechanics to make a world-model prediction?
* Does the mental loop become confident, find a simulated improvement, and emit
  a route?
* How many proposed actions are actually executed before an observed divergence?

It never exposes ``Twin.snapshot`` or ``Twin.valid_actions`` to the Pilot.  The
only inputs it supplies are the same grid history and declared action names the
Kaggle wrapper sees.  Results must not be compared directly to Qwen/Kaggle runs.

    .venv/Scripts/python.exe arc3x/pilot_harness.py tn36 --budget 300
    .venv/Scripts/python.exe arc3x/pilot_harness.py --split both --budget 600
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc3x.explore import discover_games
from arc3x.graded import GradedRun
from arc3x.mindgraft import AID_LABEL
from arc3x.pilot import Pilot
from arc3x.suite import HOLD, TUNE
from arc3x.twin import default_env_dir


def _entry(action: str, obs) -> SimpleNamespace:
    """Adapt a local observation to the framework history shape."""
    return SimpleNamespace(
        action=action,
        frame=SimpleNamespace(
            grid=tuple(tuple(int(v) for v in row) for row in obs.frame),
            # ``Pilot`` uses the framework's 1-based display level.
            level=int(obs.levels_completed) + 1,
        ),
    )


def _spent_on_level(history: list[SimpleNamespace], level: int) -> int:
    """Count actions since the latest 1-based level boundary."""
    total = 0
    for item in reversed(history):
        if int(item.frame.level) != level:
            break
        total += 1
    # The opening/level-transition frame is evidence, not an action charged to
    # the level currently displayed.
    return max(0, total - 1)


def _name(press) -> str:
    if press.is_click:
        return f"MOUSE(row={press.row}, col={press.col})"
    return press.name or AID_LABEL.get(press.aid, f"ACTION{press.aid}")


def play(game_id: str, *, budget: int, turns: int) -> dict:
    """Let the autonomous Pilot act until it abstains, ends, or hits a cap."""
    run = GradedRun(game_id, env_dir=default_env_dir())
    pilot = Pilot()
    obs = run.reset()
    history = [_entry("RESET", obs)]
    decisions = 0
    executed = 0
    stopped = "budget"

    while run.actions < budget and decisions < turns and not obs.won:
        pilot.observe(history)
        level = int(obs.levels_completed) + 1
        valid = [f"ACTION{aid}" for aid in obs.available_actions if int(aid) > 0]
        plan = pilot.decide(
            obs.frame,
            valid,
            level,
            spent_on_level=_spent_on_level(history, level),
        )
        if plan is None:
            stopped = "pilot-abstained"
            break
        decisions += 1
        level_before = obs.levels_completed
        for press in plan.presses:
            if run.actions >= budget:
                break
            # Local ``GradedRun`` uses x/y whereas the pilot uses row/col.
            obs = run.step(press.aid, x=press.col, y=press.row)
            history.append(_entry(_name(press), obs))
            executed += 1
            if obs.game_over:
                obs = run.reset()
                history.append(_entry("RESET", obs))
                break
            if obs.won or obs.levels_completed != level_before:
                # A new board invalidates the remainder of a simulated route.
                break
    else:
        stopped = "won" if obs.won else "turn-cap" if decisions >= turns else "budget"

    report = run.report()
    report.update(
        {
            "game": game_id.split("-")[0],
            "stopped": stopped,
            "decisions": decisions,
            "pilot_actions": executed,
            "mental_checks": pilot.imagine_checks,
            "mental_plans": pilot.imagine_plans,
            "mental_rejections": dict(pilot.imagine_rejections),
            "pilot_summary": pilot.summary(),
        }
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("games", nargs="*", help="short ids; defaults to the selected split")
    parser.add_argument("--budget", type=int, default=600)
    parser.add_argument("--turns", type=int, default=80)
    parser.add_argument("--split", choices=("tune", "hold", "both", "all"), default="tune")
    parser.add_argument("--json", type=Path, help="optional report path")
    args = parser.parse_args(argv)

    by_short = {gid.split("-")[0]: gid for gid in discover_games(default_env_dir())}
    if args.games:
        names = args.games
    elif args.split == "tune":
        names = TUNE
    elif args.split == "hold":
        names = HOLD
    elif args.split == "both":
        names = TUNE + HOLD
    else:
        names = sorted(by_short)

    reports: list[dict] = []
    for name in names:
        game_id = by_short.get(name)
        if game_id is None:
            print(f"unknown game: {name}", file=sys.stderr)
            continue
        try:
            report = play(game_id, budget=max(1, args.budget), turns=max(1, args.turns))
        except Exception as exc:
            print(f"{name:6s} ERROR {type(exc).__name__}: {exc}", flush=True)
            continue
        reports.append(report)
        print(
            f"{report['game']:6s} levels={report['levels_completed']}/{report['n_levels']} "
            f"score={report['score']:.3f} actions={report['actions']} "
            f"mental={report['mental_plans']}/{report['mental_checks']} "
            f"stop={report['stopped']}",
            flush=True,
        )

    if reports:
        mean = sum(float(r["score"]) for r in reports) / len(reports)
        plans = sum(int(r["mental_plans"]) for r in reports)
        checks = sum(int(r["mental_checks"]) for r in reports)
        print(f"mean={mean:.3f}  mental plans={plans}/{checks}  n={len(reports)}")
    if args.json:
        args.json.write_text(json.dumps(reports, indent=1), encoding="utf-8")
        print(f"detail -> {args.json}")
    return 0 if reports else 2


if __name__ == "__main__":
    raise SystemExit(main())

