"""Win-then-replay banking solver (battle-plan STEP 4b / week-plan Delta 3).

Engine facts this graft is built on (all verified against the vendored
sources and ``arc_agi`` 0.9.8 / ``arcengine`` 0.9.3):

- A scorecard card's score is the MAX over its plays
  (``arc_agi.scorecard.EnvironmentScoreList.score``).
- RESET issued while the engine is in the WIN state performs a *full*
  reset even under ``ONLY_RESET_LEVELS=true`` (``arcengine.base_game
  .handle_reset``), and a full reset opens a NEW play on the SAME card
  (``Scorecard.update_scorecard`` -> ``new_play``).
- ``taaf.game.Game.execute_action`` refuses to run once the ``GameRun``
  is ``"won"``, so the replay must drive the underlying
  ``arc_agi.EnvironmentWrapper`` (``GameAPI.env``) directly. That keeps
  the framework-side win record untouched while the engine/gateway
  scores the second play.

Strategy: once a session's WIN is fully recorded, prune the winning
trace per level (drop actions that changed neither the frame nor
``levels_completed``; a RESET voids everything since the level started)
and replay it on a fresh play of the same card. Every replayed action is
checked against the recorded frame and level count; any divergence
aborts the replay immediately. Aborting is free — the recorded win still
owns the card max — so every guard in here fails toward "do nothing".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, fields
from typing import Any

import arcengine

from inference.framework.solver import (
    HarnessSolver,
    _grid_from_state,
    _HarnessGameSession,
)

from taaf_grafts.solver_base import SessionSeamMixin

Grid = tuple[tuple[int, ...], ...]


class BankingPlanError(ValueError):
    """The recorded trace cannot be turned into a trustworthy replay plan."""


@dataclass(frozen=True)
class TraceStep:
    """One executed engine action with the state it produced."""

    action_id: arcengine.GameAction
    action_data: dict[str, Any]
    grid: Grid
    levels_completed: int
    state: arcengine.GameState


def prune_winning_trace(
    trace: list[TraceStep],
    initial_grid: Grid,
    number_of_levels: int,
) -> list[TraceStep]:
    """Return the pruned replay plan for a recorded winning trace.

    Per level, only the segment after the last RESET (mid-level RESETs —
    including the harness auto-RESET after GAME_OVER — restart the level,
    voiding everything before them) survives, minus actions that neither
    changed the visible frame nor advanced ``levels_completed``. An
    action that advances ``levels_completed`` is always kept.

    Raises :class:`BankingPlanError` when the trace does not describe a
    clean win (defensive: replaying a garbled plan risks nothing score-
    wise, but there is no point sending actions we cannot verify).
    """
    if not trace:
        raise BankingPlanError("empty trace")
    if trace[-1].state != arcengine.GameState.WIN:
        raise BankingPlanError(f"trace ends in {trace[-1].state.name}, not WIN")

    plan: list[TraceStep] = []
    pending: list[TraceStep] = []  # kept actions of the level in progress
    prev_grid = initial_grid
    prev_levels = 0
    for step in trace:
        if step.action_id == arcengine.GameAction.RESET:
            pending = []
            prev_grid = step.grid
            continue
        if step.levels_completed > prev_levels:
            pending.append(step)
            plan.extend(pending)
            pending = []
            prev_levels = step.levels_completed
        elif step.grid != prev_grid:
            pending.append(step)
        # else: no frame change, no level progress — prunable no-op.
        prev_grid = step.grid
    if pending:
        raise BankingPlanError("trailing actions after the last level advance")
    if prev_levels != number_of_levels:
        raise BankingPlanError(
            f"trace covers {prev_levels}/{number_of_levels} levels"
        )
    return plan


def _grid_from_frame_raw(resp: Any) -> Grid:
    """Final visible frame of a ``FrameDataRaw`` as a hashable grid."""
    data = resp.frame[-1]
    rows = data.tolist() if hasattr(data, "tolist") else data
    return tuple(tuple(int(cell) for cell in row) for row in rows)


@dataclass
class _BankingGameSession(_HarnessGameSession):
    """Harness session that records a replayable trace and, on WIN, banks
    a pruned replay as a second play of the same card before the game is
    finished (``finish_game`` must run last: in competition mode it fires
    the shared scorecard's ``finish_run``, and the LAST one closes the
    card)."""

    _trace: list[TraceStep] = field(default_factory=list, init=False, repr=False)
    _banking_attempted: bool = field(default=False, init=False, repr=False)

    # -- trace recording ---------------------------------------------------

    def _execute_action(
        self,
        action: arcengine.ActionInput,
        *,
        batch_index: int,
        batch_size: int,
        generated_tokens: int | None = None,
        flush_viewer_payload: bool = True,
    ) -> dict[str, Any]:
        payload = super()._execute_action(
            action,
            batch_index=batch_index,
            batch_size=batch_size,
            generated_tokens=generated_tokens,
            flush_viewer_payload=flush_viewer_payload,
        )
        # game.execute_action raised => nothing recorded on either side,
        # so trace and game_run.history stay 1:1.
        state = self.game.current_state
        self._trace.append(
            TraceStep(
                action_id=action.id,
                action_data=dict(action.data),
                grid=_grid_from_state(state),
                levels_completed=int(state.levels_completed),
                state=state.raw.state,
            )
        )
        return payload

    # -- banking -------------------------------------------------------------

    def _finish_if_needed(self) -> None:
        try:
            self._maybe_bank_win()
        except Exception as exc:  # noqa: BLE001 — banking must never block completion
            self._note_banking(f"error {type(exc).__name__}: {exc}")
        super()._finish_if_needed()

    def _maybe_bank_win(self) -> None:
        if self._banking_attempted:
            return
        self._banking_attempted = True

        solver = self.solver
        if not getattr(solver, "banking_enabled", False):
            return
        run = self.game.game_run
        # Bank strictly AFTER a fully recorded WIN and strictly BEFORE
        # finish_game (final_score is None until then).
        if run is None or run.state != "won" or run.final_score is not None:
            return
        if self.stop_event.is_set():
            return
        env = getattr(self.game, "env", None)
        if env is None:  # not an engine-backed game
            return
        if len(self._trace) != len(run.history) or not self.history_entries:
            self._note_banking("skip: trace/history misaligned")
            return

        try:
            plan = prune_winning_trace(
                self._trace,
                self.history_entries[0].frame.grid,
                int(self.game.number_of_levels),
            )
        except BankingPlanError as exc:
            self._note_banking(f"skip: {exc}")
            return

        original = sum(
            1 for s in self._trace if s.action_id != arcengine.GameAction.RESET
        )
        if len(plan) >= original:
            self._note_banking(f"skip: nothing to prune ({original} actions)")
            return
        max_replay = getattr(solver, "banking_max_replay_actions", None)
        if max_replay is not None and len(plan) > int(max_replay):
            self._note_banking(
                f"skip: plan {len(plan)} > banking_max_replay_actions {max_replay}"
            )
            return

        budget = self._replay_budget_seconds()
        needed = len(plan) * float(solver.banking_seconds_per_action) + float(
            solver.banking_finish_margin_s
        )
        if budget is not None and budget < needed:
            self._note_banking(
                f"skip: budget {budget:.0f}s < estimated {needed:.0f}s"
            )
            return

        self._replay(env, plan, original, budget)

    def _replay_budget_seconds(self) -> float | None:
        candidates: list[float] = []
        remaining = self.timing_payload()["time_remaining_seconds"]
        if remaining is not None:
            candidates.append(float(remaining))
        soft_remaining = self.solver.soft_time_remaining_seconds()
        if soft_remaining is not None:
            candidates.append(float(soft_remaining))
        if not candidates:
            return None
        return min(candidates)

    def _replay(
        self,
        env: Any,
        plan: list[TraceStep],
        original_actions: int,
        budget: float | None,
    ) -> None:
        margin = float(self.solver.banking_finish_margin_s)
        deadline = (
            None if budget is None else time.monotonic() + max(0.0, budget - margin)
        )
        try:
            # RESET in WIN state = full reset = new play on the same card.
            resp = env.step(arcengine.GameAction.RESET, data={})
            if resp is None or not resp.frame:
                self._note_banking("abort: RESET rejected")
                return
            if int(resp.levels_completed) != 0 or resp.state == arcengine.GameState.WIN:
                self._note_banking("abort: RESET did not open a fresh play")
                return

            for index, step in enumerate(plan, start=1):
                if self.stop_event.is_set():
                    self._note_banking(f"abort: stop requested at {index}/{len(plan)}")
                    return
                if deadline is not None and time.monotonic() >= deadline:
                    self._note_banking(f"abort: budget exhausted at {index}/{len(plan)}")
                    return
                resp = env.step(step.action_id, data=dict(step.action_data))
                if resp is None or not resp.frame:
                    self._note_banking(f"abort: engine refused step {index}/{len(plan)}")
                    return
                if _grid_from_frame_raw(resp) != step.grid:
                    self._note_banking(f"abort: frame divergence at {index}/{len(plan)}")
                    return
                if int(resp.levels_completed) != step.levels_completed:
                    self._note_banking(f"abort: level divergence at {index}/{len(plan)}")
                    return

            if resp.state != arcengine.GameState.WIN:
                self._note_banking(
                    f"abort: replay ended in {resp.state.name}, not WIN"
                )
                return
            self._note_banking(
                f"banked: replayed win in {len(plan)} actions (original {original_actions})"
            )
        except Exception as exc:  # noqa: BLE001 — a broken replay must not touch the win
            self._note_banking(f"abort: {type(exc).__name__}: {exc}")

    def _note_banking(self, message: str) -> None:
        text = f"[banking] {message}"
        run = self.game.game_run
        if run is not None:
            run.solver_note = f"{run.solver_note}; {text}" if run.solver_note else text
        try:
            with open(self.transcript_path, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except OSError:
            pass


@dataclass
class BankingHarnessSolver(SessionSeamMixin, HarnessSolver):
    """``HarnessSolver`` with win-then-replay banking.

    Construct it in notebook cell 12 via :meth:`from_solver` on the
    unpickled solver — the class itself is never pickled (the pickles
    must keep unpickling against the stock framework), and
    ``Benchmark.run`` deepcopies whatever solver instance is present, so
    replacing the instance wholesale is fully supported.

    The banking session is grafted purely by the ``session_class`` seam
    (design module 1): ``SessionSeamMixin._play_one`` constructs it, so
    there is no per-graft ``_play_one`` copy to drift against stock.
    """

    session_class = _BankingGameSession
    label: str = "BankingHarnessSolver"
    banking_enabled: bool = True
    # Conservative per-action estimate for the budget check: local
    # engine steps are ~0.1s, gateway HTTP steps a few hundred ms.
    banking_seconds_per_action: float = 2.0
    banking_finish_margin_s: float = 30.0
    banking_max_replay_actions: int | None = None

    @classmethod
    def from_solver(cls, base: HarnessSolver, **overrides: Any) -> "BankingHarnessSolver":
        """Build a banking solver carrying every configured field of ``base``."""
        kwargs = {f.name: getattr(base, f.name) for f in fields(HarnessSolver) if f.init}
        kwargs.update(overrides)
        return cls(**kwargs)
