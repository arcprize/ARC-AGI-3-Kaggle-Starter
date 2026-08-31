"""Measure the mind before letting it act: does it predict what it has not seen?

WHY THIS EXISTS
---------------
``docs/EXPERIMENT_LOG.md`` records eleven leaderboard experiments and eleven
results at or below the baseline. The common factor is that each mechanism was
plausible and unmeasured. So the mind gets a number before it gets to spend a
single real action, and this is that number.

WHAT IS BEING MEASURED
----------------------
For each dev game: drive it forward with a seeded random button policy, keep the
``(action, frame)`` history in the framework's own ``HistoryEntry`` objects, then
ask ``arc3x.mindgraft.backtest`` to induce a model from the first 70% of those
transitions and predict the last 30% it has never seen.

WHAT MAKES IT HONEST
--------------------
* **The engine is a data generator, never an oracle.** It is stepped forward one
  action at a time. Nothing here snapshots, searches, deep-copies or rolls back,
  because none of that exists on the 110 remote games. The mind sees exactly what
  it will see there: past frames and the actions that produced them.
* **The frames are the real ones.** ``inference/framework/solver.py`` builds the
  agent's grid with ``_grid_from_state`` = ``state.frame.data``, the raw engine
  grid with no rendering, so what this harness feeds the mind is byte-identical in
  format to what the notebook feeds it.
* **The click policy is deliberately blind.** ``Twin.valid_actions`` will hand
  over the engine's exact legal click coordinates, and using them would make the
  local number better than anything achievable at the gateway, which does not
  offer them. Clicks here get a uniformly random cell, which is what an agent that
  can only look at the board has to do.
* **Random play is the pessimistic proxy.** The real corpus is written by an LLM
  that presses buttons deliberately, so its transitions should be no *less*
  informative than these. A model that cannot learn from random play has not
  been given an unfair test.

READING THE OUTPUT
------------------
``spoke``    - of the held-out transitions, how many the model would predict at
               all. Declining is not an error; a model with nothing learned about
               a button should say so.
``exact``    - the entire 64x64 grid was right. One HUD clock pixel holds this at
               zero forever, so a low number here is a finding, not a failure.
``place``    - the sprite ended where the model said. **This is the planning
               gate**: a route is correct exactly when this is.
``movecall`` - moved-versus-blocked was called right. What a route needs in order
               not to walk into a wall.

Run:
    .venv/Scripts/python.exe tools/mind_backtest.py
    .venv/Scripts/python.exe tools/mind_backtest.py --steps 400 --games ls20 sk48
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratch" / "archive2_extracted" / "src" / "ARC3-Inference"))

import numpy as np  # noqa: E402

# The framework's own state objects, so this tests the real data path rather than
# a duck-typed imitation of it. smoke_grafts.py establishes that
# inference.agent.* imports here with no stubbing at all.
from inference.agent.runtime_state import Frame, HistoryEntry  # noqa: E402

from arc3x.mindgraft import AID_LABEL, Report, aggregate, backtest  # noqa: E402
from arc3x.twin import Act, Twin, default_env_dir  # noqa: E402

# aid 0 is RESET: Twin.valid_actions skips it, and the mind's codec drops it,
# because a reset teleports the sprite instead of moving it.
RESET = Act(0)


def _frame(grid: np.ndarray, step: int, completed: int, n_levels: int) -> Frame:
    """Mirror solver.py exactly: raw grid, action count, 1-based clamped level."""
    return Frame(
        grid=tuple(tuple(int(c) for c in row) for row in grid),
        step=step,
        level=max(1, min(int(n_levels), int(completed) + 1)),
    )


def _display(act: Act) -> str:
    name = AID_LABEL.get(act.aid, f"ACTION{act.aid}")
    if act.aid == 6:
        return f"{name}(row={act.y}, col={act.x})"
    return name


def collect(game_id: str, *, steps: int, seed: int, env_dir: Path) -> list[HistoryEntry]:
    """Play one game with a seeded random policy; return the framework history.

    Click coordinates are drawn at random rather than taken from the engine's
    introspection, and the grid is read back after every single action - the same
    one-way stream a live run produces.
    """
    rng = random.Random(f"{game_id}:{seed}")
    twin = Twin(game_id, env_dir=env_dir)
    obs = twin.current()
    H, W = obs.frame.shape

    # solver.seed_initial_history: one entry with an empty action for frame 0.
    entries = [HistoryEntry(action="", frame=_frame(obs.frame, 0, obs.level, twin.n_levels))]

    for step in range(1, steps + 1):
        # What the framework tells the agent: which buttons exist, not where to
        # click. Collapse the engine's concrete clicks down to the bare button.
        aids = sorted({a.aid for a in obs.valid if a.aid != 0})
        if not aids:
            break
        aid = rng.choice(aids)
        act = Act(6, rng.randrange(W), rng.randrange(H)) if aid == 6 else Act(aid)

        obs = Twin.step_game(twin.game, act)
        entries.append(
            HistoryEntry(
                action=_display(act),
                frame=_frame(obs.frame, step, obs.level, twin.n_levels),
            )
        )
        if obs.terminal:
            if obs.won:
                break
            # A death is not the end of a run: the real agent resets and carries
            # on, and the RESET entry is exactly the kind of non-transition the
            # mind has to drop rather than learn from.
            obs = Twin.step_game(twin.game, RESET)
            entries.append(
                HistoryEntry(
                    action="RESET",
                    frame=_frame(obs.frame, step, obs.level, twin.n_levels),
                )
            )
    return entries


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", nargs="*", default=None, help="game ids (default: all)")
    ap.add_argument("--steps", type=int, default=250, help="actions of random play per game")
    ap.add_argument("--holdout", type=float, default=0.3, help="fraction predicted, never trained on")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--verbose", action="store_true", help="print each game's learned model")
    args = ap.parse_args()

    env_dir = default_env_dir()
    if args.games:
        ids = list(args.games)
    else:
        from arc3x.explore import discover_games

        ids = discover_games(env_dir)

    print(
        f"mind backtest: {len(ids)} games, {args.steps} random actions each, "
        f"predicting the last {args.holdout:.0%} of transitions\n"
    )
    reports: list[Report] = []
    for gid in ids:
        try:
            entries = collect(gid, steps=args.steps, seed=args.seed, env_dir=env_dir)
            rep = backtest(entries, holdout=args.holdout, game=gid.split("-")[0])
        except Exception as exc:  # noqa: BLE001 — one broken game must not hide the rest
            print(f"{gid.split('-')[0]:>6}  FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        reports.append(rep)
        print(rep.line() if rep.n else f"{rep.game:>6}  too few transitions to score", flush=True)
        if args.verbose and rep.n:
            print()

    print()
    print(aggregate(reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
