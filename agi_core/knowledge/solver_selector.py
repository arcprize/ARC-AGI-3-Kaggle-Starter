from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agi_core.knowledge.puzzle_memory import (
    PuzzleBoardState,
    initialize_puzzle_schema,
    store_board_state,
    update_solver_outcome,
)
from agi_core.knowledge.solvers import (
    DynamicsSolver,
    SolverContext,
    get_solver,
)
from agi_core.types import ActionDecision, ObservedTransition

if TYPE_CHECKING:
    from agi_core.knowledge.hyperon_bridge import HyperonBridge
    from agi_core.orch_or.game_model import GameModel
    from agi_core.types import ExecutiveContext

logger = logging.getLogger(__name__)


class SolverSelector:
    """Selects and manages the appropriate dynamics solver for the current board.

    Queries the hyperon bridge for solver recommendations, checks memory
    for better alternatives, instantiates the solver, and records outcomes.
    """

    def __init__(
        self,
        hyperon: HyperonBridge,
        memory_db_path: str | Path | None = None,
    ) -> None:
        self._hyperon = hyperon
        self._memory_db_path = Path(memory_db_path) if memory_db_path else None
        self._active_solver: DynamicsSolver | None = None
        self._active_name: str = ""
        self._solver_history: list[tuple[str, str]] = []
        self._pending_feedback: tuple[
            DynamicsSolver,
            SolverContext,
            GameModel,
            ActionDecision,
        ] | None = None
        self._last_retention_reason: str = ""
        self._attempt_steps = 0
        self._attempt_game_id = ""
        self._attempt_dynamics = ""
        self._attempt_solver_name = ""

    @property
    def active_solver(self) -> DynamicsSolver | None:
        return self._active_solver

    @property
    def active_attempt(self) -> bool:
        """Whether the retained solver owns an in-progress execution attempt."""
        value = getattr(self._active_solver, "has_active_attempt", False)
        return value if isinstance(value, bool) else False

    @property
    def last_retention_reason(self) -> str:
        return self._last_retention_reason

    def _replace_active_solver(
        self,
        solver_cls: type[DynamicsSolver],
        solver_name: str,
    ) -> DynamicsSolver:
        displaced = self._active_solver
        if displaced is not None:
            try:
                displaced.reset_level()
            except Exception:
                logger.debug("active solver reset failed during replacement", exc_info=True)
        self._active_solver = solver_cls(memory_db_path=self._memory_db_path)
        self._active_name = solver_name
        return self._active_solver

    def _should_retain_active_solver(
        self,
        *,
        solver_name: str,
        context: ExecutiveContext,
    ) -> bool:
        if self.active_attempt:
            self._last_retention_reason = "active-attempt"
            return True
        # Beam-coverage owns AR25 beam boards even when perception briefly
        # downgrades them to generic beam dynamics between mirror and shape
        # control frames. Keep that solver until an explicit boundary resets it.
        if (
            self._active_name == "beam-coverage"
            and solver_name == "beam"
            and getattr(context.observation, "last_action", None) != "ACTION5"
        ):
            self._last_retention_reason = "beam-coverage-ownership"
            return True
        self._last_retention_reason = ""
        return False

    def select_solver(
        self,
        game_model: GameModel,
        context: ExecutiveContext,
    ) -> DynamicsSolver | None:
        """Select the best solver for the current board state."""
        dynamics = game_model.features.primary_dynamics

        # 1. Query hyperon for solver recommendation
        solver_name = "navigation"  # default
        try:
            result = self._hyperon.run_rules("solver-select")
            if result and "solver" in result[0]:
                solver_name = str(result[0]["solver"])
            logger.debug("SELECTOR: dynamics=%s hyperon_solver=%s", dynamics, solver_name)
        except Exception:
            pass

        # 2. Check if memory has a better solver for this board state
        memory_solver = self._query_memory_for_solver(game_model)
        if memory_solver and memory_solver != solver_name:
            solver_name = memory_solver

        # 3. Instantiate solver
        solver_cls = get_solver(solver_name)
        if solver_cls is None:
            from agi_core.knowledge.solvers.navigation_solver import NavigationSolver
            solver_cls = NavigationSolver

        # Reuse existing solver of the same type to preserve state across frames.
        # Allow switching when the solver name genuinely changes (level transition).
        if self._active_solver is not None:
            if type(self._active_solver) is solver_cls:
                self._active_name = solver_name
                self._last_retention_reason = "same-class"
                return self._active_solver
            # Keep cross-frame ownership only for solvers that explicitly hold
            # an in-progress attempt. Stateless solvers should yield as soon as
            # the board reclassifies to a different dynamics family.
            if self._should_retain_active_solver(
                solver_name=solver_name,
                context=context,
            ):
                logger.debug(
                    "SELECTOR: retaining active solver %s over mid-level recommendation %s reason=%s",
                    self._active_name,
                    solver_name,
                    self._last_retention_reason,
                )
                return self._active_solver
            self._last_retention_reason = ""
            logger.debug(
                "SELECTOR: switching active solver %s -> %s active_attempt=%s last_action=%s",
                self._active_name,
                solver_name,
                self.active_attempt,
                getattr(context.observation, "last_action", None),
            )
            return self._replace_active_solver(solver_cls, solver_name)
        return self._replace_active_solver(solver_cls, solver_name)

    def build_solver_context(
        self,
        game_model: GameModel,
        plan_context: dict[str, Any],
        context: ExecutiveContext,
    ) -> SolverContext:
        """Build a SolverContext from the current game state."""
        episode = {}
        memory = {}
        try:
            episode = dict(context.hyperon.facts.get("episode", {}))
        except Exception:
            pass
        try:
            memory = dict(context.hyperon.facts.get("memory", {}))
        except Exception:
            pass
        return SolverContext(
            game_model=game_model,
            plan_context=plan_context,
            episode=episode,
            memory=memory,
        )

    def record_outcome(
        self,
        context: ExecutiveContext,
        game_model: GameModel,
        decision: ActionDecision,
        plan_context: dict[str, Any],
    ) -> None:
        """Register a solver decision for feedback from the next observation."""
        if self._active_solver is None:
            return
        solver_context = self.build_solver_context(game_model, plan_context, context)
        self._pending_feedback = (
            self._active_solver,
            solver_context,
            game_model,
            decision,
        )
        self._attempt_steps += 1
        self._attempt_game_id = str(context.observation.game_id or "")
        self._attempt_dynamics = str(
            getattr(self._active_solver, "dynamics", "unknown") or "unknown"
        )
        self._attempt_solver_name = self._active_name or self._attempt_dynamics

    def note_transition(self, transition: ObservedTransition) -> None:
        """Apply one registered solver decision's observed result."""
        pending = self._pending_feedback
        if pending is not None:
            solver, solver_context, game_model, decision = pending
            matches = (
                (not transition.action_name or transition.action_name == decision.action_name)
                and (
                    not transition.decision_source
                    or transition.decision_source == decision.source
                )
            )
            outcome = str(transition.local_outcome or "unknown")
            if matches and outcome != "unknown":
                try:
                    solver.learn(solver_context, outcome)
                except Exception:
                    logger.debug("solver learning failed", exc_info=True)
                self._store_observed_board_state(
                    transition,
                    game_model,
                    decision,
                    outcome,
                )
            if matches or transition.boundary is not None:
                self._pending_feedback = None

        if transition.boundary in {"win", "level"}:
            self._close_attempt(succeeded=True, game_id=transition.game_id)
        elif transition.boundary in {"game_over", "restart", "game"}:
            self._close_attempt(succeeded=False, game_id=transition.game_id)

    def on_boundary(self, boundary: str, *, game_id: str = "") -> None:
        if boundary in {"win", "level"}:
            self._close_attempt(succeeded=True, game_id=game_id)
        elif boundary in {"game_over", "restart", "game", "stalled"}:
            self._close_attempt(succeeded=False, game_id=game_id)
        self._pending_feedback = None
        self.reset_level()

    def reset_level(self) -> None:
        if self._active_solver is not None:
            try:
                self._active_solver.reset_level()
            except Exception:
                logger.debug("active solver reset failed", exc_info=True)
        self._active_solver = None
        self._active_name = ""
        self._last_retention_reason = ""

    def _store_observed_board_state(
        self,
        transition: ObservedTransition,
        game_model: GameModel,
        decision: ActionDecision,
        outcome: str,
    ) -> None:
        if self._memory_db_path is None:
            return
        try:
            initialize_puzzle_schema(self._memory_db_path)
            player = game_model.player_pos
            store_board_state(
                self._memory_db_path,
                PuzzleBoardState(
                    game_id=transition.game_id,
                    step=(
                        int(transition.record.step_index)
                        if transition.record is not None
                        else 0
                    ),
                    frame_signature=(
                        str(transition.record.prior_frame_signature or "")
                        if transition.record is not None
                        else ""
                    ),
                    player_x=player[0] if player else -1,
                    player_y=player[1] if player else -1,
                    goal_positions=list(game_model.goal_positions),
                    goals_lit=game_model.features.goals_lit_by_beam,
                    action_taken=decision.action_name,
                    outcome=outcome,
                    progress_delta=(
                        int(transition.record.progress_delta or 0)
                        if transition.record is not None
                        else 0
                    ),
                ),
            )
        except Exception:
            logger.debug("solver board-state persistence failed", exc_info=True)

    def _close_attempt(self, *, succeeded: bool, game_id: str) -> None:
        if self._memory_db_path is None or self._attempt_steps <= 0:
            self._attempt_steps = 0
            return
        try:
            initialize_puzzle_schema(self._memory_db_path)
            update_solver_outcome(
                self._memory_db_path,
                game_id=game_id or self._attempt_game_id,
                dynamics=self._attempt_dynamics or "unknown",
                solver_name=self._attempt_solver_name or self._attempt_dynamics or "unknown",
                succeeded=succeeded,
                steps_taken=self._attempt_steps,
            )
        except Exception:
            logger.debug("solver outcome persistence failed", exc_info=True)
        finally:
            self._attempt_steps = 0

    def _query_memory_for_solver(self, game_model: GameModel) -> str | None:
        """Check memory for a solver that succeeded on similar boards."""
        if self._memory_db_path is None or not self._memory_db_path.exists():
            return None
        try:
            import sqlite3
            from contextlib import closing
            with closing(sqlite3.connect(self._memory_db_path)) as conn:
                row = conn.execute(
                    """SELECT solver_name FROM solver_outcomes
                       WHERE dynamics = ? AND success_count > attempt_count * 0.3
                       ORDER BY success_count DESC, attempt_count DESC LIMIT 1""",
                    [game_model.features.primary_dynamics],
                ).fetchone()
                if row:
                    return str(row[0])
        except Exception:
            pass
        return None
