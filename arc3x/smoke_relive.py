"""Does the archive search actually collapse states, and does it cost what it should?

WHY THIS EXISTS
---------------
``relive.py`` has three claims in it that are arithmetic, not opinion, and none of
them can be settled by reading the code:

  1. **The cell key collapses.** Cells created per action spent should fall well
     below 1.0. At 1.0 the key is bijective, novelty carries no signal, and the
     search is a paid random walk - which is the measured failure that
     ``coarsen`` exists to fix. Read ``cells/action`` and ``coarsen=``.
  2. **The rewind is affordable.** ``restarts`` should be a small fraction of the
     actions spent, and ``drift`` should be near zero. Each restart bills
     ``1 + depth``, so a search that spends most of its budget returning to cells
     has bought nothing. The first smoke test read 481 restarts in 500 actions.
  3. **It runs at all.** The self-calibrating key and the ``Agent.act``-side route
     recording have never executed.

AND ONE CLAIM THAT MATTERS MORE THAN ALL THREE
---------------------------------------------
Measured across six notebook runs on tn36: identical depth (2 of 7 levels), scores
**0.00, 0.00, 0.10, 1.36, 5.28, 10.71**. And the run that scored 10.71 spent 467
actions where the run that scored 1.36 spent 329 - **more actions, 7.9x the score**.
Its per-action event log splits **31 / 69 / 483** over levels 0 / 1 / 2: it cleared
the first two off 100 actions and then spent 83% of the run failing level 2, for
nothing. Which is the rule:

    An action spent on a level you go on to clear costs score quadratically.
    An action spent on a level you never clear costs nothing but budget.

m0r0 says it from the other side - level 0 *cleared* in 477 actions, scoring
**0.02 of an available 4.76**, 99.6% discarded precisely because it cleared the
level it dawdled on.

So "relive cleared level 0" is *not* the success condition, and neither is "relive
used few actions". A level 0 cleared by 400 actions of archive search scores about
zero on its own - and that is fine, because level 0 carries weight 1 of 21 to 55, at
most 4.8% of the game. What has to be true is that the search leaves behind a
**model** good enough to take level 1 at =<1.4x baseline, where the points actually
are. That is the carry-over ``Agent.on_new_level`` was built for and it has never
been exercised.

Hence the ``x base`` column below: actions charged to each level over that level's
human baseline, **per level, never averaged**. Level 0 is allowed to be terrible.
Level 1 is not. And a level the agent never clears does not appear at all, because
what it spent there did not cost anything.

The games below are chosen to cover the three action-space shapes, because the
search behaves differently in each and an average over them hides all three:

    ka59, dc22, tn36   movement + click
    cn04, bp35         movement only
    su15               click only

Deliberately *not* a score measurement. Six games at a 1500-action budget is not
the suite; ``suite.py --split both`` is, and it is the only thing allowed to make
a claim about score. This file answers "is the mechanism doing what it says", and
a mechanism that is not can be rejected in four minutes instead of forty.

Run:
    PYTHONPATH=. .venv/Scripts/python.exe arc3x/smoke_relive.py
"""

from __future__ import annotations

import argparse
import time
import traceback

from arc3x.agent import Agent
from arc3x.graded import GradedRun, default_env_dir

GAMES = ["ka59", "cn04", "su15", "bp35", "tn36", "dc22"]


def _speed(rep: dict) -> str:
    """Actions charged per cleared level, over that level's human baseline.

    The only column that predicts score, because level i's score is
    ``(baseline/actions)^2 x 100`` - so 2x baseline keeps a quarter of the points
    and 5x keeps 4%. Two deliberate choices, both following from the measured rule
    in the module docstring:

    * **Only cleared levels are listed.** An uncleared level scores 0 whatever was
      spent on it, so its action count is not an efficiency figure at all - it is
      just budget. tn36's best run spent 83% of its actions on a level it failed
      and still took the full cap.
    * **Per level, never averaged.** Level 0 is allowed to be terrible (weight 1 of
      21-55) and level 1 is not, and a mean hides exactly that distinction.

    Prints the level's own score alongside the ratio, because that is the quantity
    the ratio is a proxy for and ``report()`` already computes it exactly.
    """
    charged = rep["level_actions"]
    base = rep["baselines"]
    per = rep["level_scores"]
    done = rep["levels_completed"]
    if not done:
        return f"cleared nothing; {charged[0] if charged else 0} actions on L0"
    bits = []
    for i in range(min(done, len(charged), len(base))):
        b = max(1, base[i])
        s = f" ({per[i]:.0f}pts)" if i < len(per) else ""
        bits.append(f"L{i} {charged[i]}/{b}={charged[i] / b:.1f}x{s}")
    return "  ".join(bits)


def _check_baselines(rep: dict) -> str:
    """Is the ``x base`` column real, or is it the fallback?

    ``GradedRun.report`` falls back to ``[100] * n_levels`` when the engine hands
    over no ``baseline_actions``. That fallback is silent, and it turns every ratio
    in this file into a fabrication - a level cleared in 100 actions would read
    ``1.0x`` regardless of what a human needs. Worth one line to rule out, because
    the whole test hangs on that column.
    """
    base = rep["baselines"]
    if not base:
        return "  !! NO BASELINES - every x base figure below is meaningless"
    if all(b == 100 for b in base):
        return "  !! baselines are all exactly 100 - almost certainly the fallback"
    return ""


def one(game_id: str, env_dir, *, budget: int, seconds: float) -> dict:
    """One game, played to exhaustion, with the relive counters attached.

    A crash is printed with its traceback and then swallowed, rather than raised.
    That is the opposite of the usual preference, and it is deliberate: this module
    exists to find out whether six differently-shaped games survive the search at
    all, and a first game that dies would otherwise hide the five behind it. The
    report still comes back (``GradedRun`` keeps its own tally), so a crashed game
    shows up as a row with a ``crash:`` note instead of as a missing row.
    """
    run = GradedRun(game_id, env_dir)
    ag = Agent(run, budget=budget, seconds=seconds)
    t0 = time.perf_counter()
    try:
        ag.play()
    except Exception as exc:
        run.note(f"crash:{type(exc).__name__}")
        print(f"  !! {game_id} raised {type(exc).__name__}: {exc}")
        traceback.print_exc()
    rep = run.report()
    rep["_wall"] = time.perf_counter() - t0
    # Defaults first, then overwrite: after a crash mid-``play`` any of these may be
    # the thing that is broken, and a smoke test that dies while reporting a death
    # tells you strictly less than one that prints zeros and moves on.
    rep.update({"_relive": "", "_cells": 0, "_restarts": 0, "_drift": 0, "_coarsen": 0, "_pool": 0})
    try:
        rel = ag.relive
        rep["_relive"] = rel.summary()
        rep["_cells"] = rel.cells
        rep["_restarts"] = rel.restarts
        rep["_drift"] = rel.drift
        rep["_coarsen"] = rel.coarsenings
        rep["_pool"] = rel.ck.pool
    except Exception as exc:
        rep["_relive"] = f"counters unreadable: {type(exc).__name__}: {exc}"
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("--seconds", type=float, default=240.0)
    ap.add_argument("--games", default=",".join(GAMES))
    a = ap.parse_args()
    env = default_env_dir()

    rows = []
    for gid in a.games.split(","):
        # ``one`` already swallows a crash inside ``play``; this catches the outer
        # ring - a bad game id, an engine that will not construct, a report that
        # cannot be built. Same reason: six games' worth of information per run
        # matters more than a clean traceback, because the run is the scarce thing.
        try:
            rep = one(gid, env, budget=a.budget, seconds=a.seconds)
        except Exception as exc:
            print(f"{gid:6s} !! could not run at all: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue
        rows.append(rep)
        short = rep["game_id"].split("-")[0]
        print(
            f"{short:6s} levels={rep['levels_completed']}/{rep['n_levels']} "
            f"score={rep['score']:6.2f} actions={rep['actions']:5d} "
            f"{rep['_wall']:5.1f}s"
        )
        print(f"    {rep['_relive']}")
        print(f"    {_speed(rep)}")
        warn = _check_baselines(rep)
        if warn:
            print(warn)
        stall = rep["notes"].get(rep["stalled_on"], {})
        why = ",".join(
            f"{k}x{v}" for k, v in sorted(stall.items(), key=lambda kv: -kv[1])[:6]
        )
        print(f"    L{rep['stalled_on']}: {why}")

    print("\n" + "=" * 68)
    if not rows:
        print("no game produced a report - nothing to summarise")
        return 1
    print(f"{'game':7s}{'act':>6s}{'cells':>7s}{'/act':>7s}{'restart':>8s}"
          f"{'drift':>7s}{'pool':>6s}{'coarse':>7s}{'lvls':>6s}")
    for r in rows:
        act = max(1, r["actions"])
        print(
            f"{r['game_id'].split('-')[0]:7s}{r['actions']:>6d}{r['_cells']:>7d}"
            f"{r['_cells'] / act:>7.2f}{r['_restarts']:>8d}{r['_drift']:>7d}"
            f"{r['_pool']:>6d}{r['_coarsen']:>7d}{r['levels_completed']:>6d}"
        )
    tot_a = sum(max(1, r["actions"]) for r in rows)
    print(
        f"\ncells per action, all games: {sum(r['_cells'] for r in rows) / tot_a:.3f}"
        f"   (>=0.7 means the key still carries no signal)"
    )
    print(
        f"restarts per action:        {sum(r['_restarts'] for r in rows) / tot_a:.3f}"
        f"   (high means the budget went on rewinding, not searching)"
    )
    print(f"levels cleared:             {sum(r['levels_completed'] for r in rows)}/{len(rows)} games")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
