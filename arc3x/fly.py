"""Fly the pilot against the real engine, billed exactly the way Kaggle bills it.

WHY THIS EXISTS
---------------
:mod:`arc3x.pilot` decides batches from frames alone, which is the same power it
will have against a remote gateway. This file is the only place that difference is
allowed to disappear: it hands the pilot real frames from a local engine, executes
its presses one at a time, and charges every one of them.

Four rules keep the measurement honest, and they are why this is not just a call
into :mod:`arc3x.graded`:

* **No snapshots.** ``Twin.snapshot`` exists and costs 14 ms, and using it here
  would measure an agent that can rewind - which the gateway cannot.
* **No privileged action list.** ``Twin.valid_actions`` returns concrete ACTION6
  click coordinates, i.e. which of 4,096 clicks are live. The gateway gives the
  agent only coarse names, so that is all the pilot is handed - see
  ``_names`` below. Getting this wrong would flatter every click game.
* **Every action is charged**, including the ones inside a batch and including
  RESET, and charged to the level it was spent on. That is what
  ``score_from_card`` needs to reproduce the leaderboard.
* **Turns are counted separately from actions.** A turn is one LLM round-trip in
  the notebook and costs ~26 s of a 7,920 s clock; an extra action inside a batch
  costs no time at all. Reporting both is the point, because the pilot's whole
  claim is that it converts turns into actions at better than 10:1.

Run::

    python -m arc3x.fly --games 6 --turns 60
    python -m arc3x.fly --game tn36 --verbose
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from arc3x.explore import discover_games
from arc3x.graded import score_from_card
from arc3x.mindgraft import AID_LABEL
from arc3x.pilot import Pilot
from arc3x.twin import ACTION_BY_ID, Act, Obs, Twin, default_env_dir

RESET = Act(0)


# `HistoryEntry`/`Frame` shaped just enough for `Mind.absorb` and `CellSense.fit`,
# which reach for `.action`, `.frame.grid` and `.frame.level` and nothing else.
# Rebuilt here rather than imported so this runs without the inference package
# installed - the pilot must never depend on the framework to be measurable.


@dataclass(frozen=True)
class _Frame:
    grid: tuple[tuple[int, ...], ...]
    step: int
    level: int


@dataclass(frozen=True)
class _Entry:
    action: str
    frame: _Frame


def _entry(action: str, grid: np.ndarray, step: int, completed: int, n_levels: int) -> _Entry:
    """Mirror ``solver.py:128 _level_number`` exactly: 1-based, clamped both ends.

    The clamp is not cosmetic. ``Pilot._lab`` asks whether ``level <= FIRST_LEVEL``,
    so an unclamped 0 would make the pilot treat level 1 as a free laboratory and
    spend a level's worth of points learning what it already knew.
    """
    return _Entry(
        action=action,
        frame=_Frame(
            grid=tuple(tuple(int(v) for v in row) for row in grid),
            step=step,
            level=max(1, min(int(n_levels), int(completed) + 1)) if n_levels else max(1, completed + 1),
        ),
    )


def _label(act: Act) -> str:
    """The display string the framework would have written for this action.

    Must match what ``mindgraft.parse_press`` reads, or the mind learns nothing:
    clicks carry their coordinates because the same button at two points is two
    different experiments.
    """
    if act.aid == 0:
        return "RESET"
    name = AID_LABEL.get(act.aid, f"ACTION{act.aid}")
    if act.aid == 6:
        return f"{name}(row={act.y}, col={act.x})"
    return name


def _names(game) -> list[str]:
    """The coarse action names the gateway would advertise - no click coordinates.

    Mirrors ``solver.py:136 _engine_action_names``, which reads
    ``game.current_state.available_actions`` and emits ``ACTION1``-style names with
    RESET excluded. ``_available_actions`` is the same declared set one level up
    (``graded.py`` reads it). Click coordinates never appear here: the gateway
    withholds which of the 4,096 clicks are live, and so does this list.
    """
    declared = list(getattr(game, "_available_actions", []) or [])
    out: list[str] = []
    for a in declared:
        aid = int(a)
        if aid and aid in ACTION_BY_ID:
            out.append(ACTION_BY_ID[aid].name)
    return out or ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"]


@dataclass
class Flight:
    """One graded flight of one game."""

    game_id: str
    env_dir: Path | None = None
    max_turns: int = 60
    max_actions: int = 3000
    verbose: bool = False

    actions: int = 0
    turns: int = 0
    handoffs: int = 0
    resets: int = 0
    #: ``(levels_completed, cumulative actions)`` at each level change - the exact
    #: shape ``Card.set_levels_completed`` records and ``score_from_card`` bills.
    cleared: list[tuple[int, int]] = field(default_factory=list)
    state: str = "NOT_PLAYED"
    seconds: float = 0.0
    pilot: Pilot = field(default_factory=Pilot)
    baselines: list[int] = field(default_factory=list)
    n_levels: int = 0

    def fly(self) -> "Flight":
        started = time.time()
        twin = Twin(self.game_id, self.env_dir)
        game = twin.game
        self.n_levels = int(twin.n_levels)
        self.baselines = list(twin.baselines or []) or [100] * max(1, self.n_levels)
        names = _names(game)

        obs = twin.current()
        done = int(obs.level)  # levels_completed, 0-based
        # solver.seed_initial_history: one entry with an empty action for frame 0.
        history = [_entry("", obs.frame, 0, done, self.n_levels)]

        while self.turns < self.max_turns and self.actions < self.max_actions:
            if obs.won:
                break
            if obs.game_over:
                # Faithful to `solver._execute_auto_reset`: a death costs one
                # billed action and restores the current level. Learning what
                # killed us is why `Mechanics.fatal` is cheap to fill.
                obs, done = self._charge(twin, game, RESET, history, done)
                self.resets += 1
                continue

            self.pilot.observe(history)
            plan = self.pilot.decide(
                obs.frame, names, done + 1, spent_on_level=self._on_level()
            )
            self.turns += 1

            if plan is None:
                # The pilot declined. In the notebook this is where the language
                # model earns its 26 seconds; here it is one arbitrary legal press,
                # so every number this file prints is a *floor* on what the pilot
                # is worth inside the real agent, never a claim about the agent.
                self.handoffs += 1
                # One legal press, chosen without looking at which clicks are
                # live - `obs.valid` carries concrete ACTION6 coordinates and
                # using them here would be the privileged information this file
                # exists to withhold.
                fallback = next(
                    (a for a in obs.valid if a.aid and a.aid != 6),
                    next((a for a in obs.valid if a.aid), None),
                )
                if fallback is None:
                    break
                obs, done = self._charge(twin, game, fallback, history, done)
                continue

            for press in plan.presses:
                if self.actions >= self.max_actions:
                    break
                act = _act(press)
                if act is None:
                    break
                obs, moved_on = self._charge(twin, game, act, history, done)
                # `step_env` breaks a batch on exactly these three, and nothing
                # else - notably not on an unchanged board (`solver.py:683-776`).
                if moved_on != done or obs.terminal:
                    done = moved_on
                    break
                done = moved_on

        self.state = str(obs.state)
        self.seconds = time.time() - started
        return self

    # -- billing --------------------------------------------------------------

    def _charge(
        self, twin: Twin, game, act: Act, history: list[_Entry], done: int
    ) -> tuple[Obs, int]:
        obs = Twin.step_game(game, act)
        self.actions += 1
        now = int(obs.level)
        history.append(_entry(_label(act), obs.frame, self.actions, now, self.n_levels))
        if now != done:
            self.cleared.append((now, self.actions))
            if self.verbose:
                print(f"    ! level {done} cleared at action {self.actions}")
        return obs, now

    def _on_level(self) -> int:
        """Actions charged to the level we are standing on right now."""
        return self.actions - (self.cleared[-1][1] if self.cleared else 0)

    # -- reporting ------------------------------------------------------------

    @property
    def levels(self) -> int:
        return len(self.cleared)

    def report(self) -> dict:
        score, per_level, charged = score_from_card(
            self.baselines, self.cleared, self.actions
        )
        show = max(1, min(len(self.baselines), self.levels + 1))
        return {
            "game": self.game_id.split("-")[0],
            "score": round(score, 3),
            "levels": f"{self.levels}/{len(self.baselines)}",
            "actions": self.actions,
            "turns": self.turns,
            "per_turn": round(self.actions / self.turns, 1) if self.turns else 0.0,
            "handoffs": self.handoffs,
            "resets": self.resets,
            "charged": charged[:show],
            "base": self.baselines[:show],
            "state": self.state,
            "s": round(self.seconds, 1),
        }

    def line(self) -> str:
        r = self.report()
        return (
            f"{r['game']:>6} score={r['score']:>8} lv={r['levels']:>5} "
            f"act={r['actions']:>5} turns={r['turns']:>4} a/turn={r['per_turn']:>5} "
            f"hand={r['handoffs']:>4} charged={r['charged']} base={r['base']} "
            f"{r['s']:>6}s"
        )


def _act(press) -> Act | None:
    """A pilot press -> the twin's action object.

    Click coordinates are passed through unchecked on purpose. The engine treats
    a dead cell as a no-op, which is exactly what the gateway does, and filtering
    against ``Twin.valid_actions`` here would hand the pilot the answer.
    """
    if press.aid not in ACTION_BY_ID or press.aid == 0:
        return None
    if press.is_click:
        if press.row < 0 or press.col < 0:
            return None
        return Act(aid=press.aid, x=int(press.col), y=int(press.row))
    return Act(aid=press.aid)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    ap.add_argument("--game", default=None, help="substring of one game id")
    ap.add_argument("--games", type=int, default=0, help="how many games (0 = all)")
    ap.add_argument("--turns", type=int, default=60, help="LLM-turn budget per game")
    ap.add_argument("--actions", type=int, default=3000, help="hard action stop")
    ap.add_argument("--lab", type=int, default=None, help="override Pilot.lab_actions")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    env = default_env_dir()
    ids = discover_games(env)
    if args.game:
        ids = [g for g in ids if args.game in g]
    if args.games:
        ids = ids[: args.games]
    if not ids:
        print(f"no games found under {env}")
        return 1

    flights: list[Flight] = []
    for gid in ids:
        pilot = Pilot()
        if args.lab is not None:
            pilot.lab_actions = args.lab
        flight = Flight(
            gid,
            env_dir=env,
            max_turns=args.turns,
            max_actions=args.actions,
            verbose=args.verbose,
            pilot=pilot,
        )
        try:
            flight.fly()
        except Exception as exc:  # a crash on one game must not hide the other 24
            print(f"{gid.split('-')[0]:>6} CRASHED {type(exc).__name__}: {exc}")
            if args.verbose:
                raise
            continue
        flights.append(flight)
        print(flight.line())
        if args.verbose:
            print(f"       {flight.pilot.summary()}")
            for entry in flight.pilot.log[:50]:
                print(f"       {entry}")

    if not flights:
        return 1
    rows = [f.report() for f in flights]
    got = sum(1 for f in flights if f.levels)
    deep = sum(1 for f in flights if f.levels >= 2)
    print("-" * 96)
    print(
        f"{len(rows)} games  mean={sum(r['score'] for r in rows) / len(rows):.3f}  "
        f"cleared>=1 level={got}/{len(rows)}  >=2 levels={deep}/{len(rows)}  "
        f"actions/turn={sum(r['actions'] for r in rows) / max(1, sum(r['turns'] for r in rows)):.1f}  "
        f"handoff rate={sum(r['handoffs'] for r in rows) / max(1, sum(r['turns'] for r in rows)):.0%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
