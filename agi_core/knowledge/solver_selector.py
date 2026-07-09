from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agi_core.knowledge.puzzle_memory import (
    PuzzleBoardState,
    store_board_state,
)
from agi_core.knowledge.solvers import (
    DynamicsSolver,
    SolverContext,
    get_solver,
)
from agi_core.types import ActionDecision

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

    @property
    def active_solver(self) -> DynamicsSolver | None:
        return self._active_solver

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

        # Reuse existing solver of the same type to preserve state across frames
        if self._active_solver is not None and type(self._active_solver) is solver_cls:
            self._active_name = solver_name
            return self._active_solver
        # Don't switch a MirrorSolver away mid-level — dynamics can oscillate
        # between "puzzle", "sliding", "beam" etc. within the same level.
        if self._active_solver is not None:
            from agi_core.knowledge.solvers.mirror_solver import MirrorSolver
            from agi_core.knowledge.solvers.navigation_solver import NavigationSolver
            if isinstance(self._active_solver, (MirrorSolver, NavigationSolver)):
                self._active_name = solver_name
                return self._active_solver
        self._active_solver = solver_cls(memory_db_path=self._memory_db_path)
        self._active_name = solver_name
        return self._active_solver

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
        """Record solver outcome for learning."""
        if self._active_solver is None:
            return
        try:
            if self._memory_db_path is not None:
                self._memory_db_path.parent.mkdir(parents=True, exist_ok=True)
                player = game_model.player_pos
                state = PuzzleBoardState(
                    game_id=context.observation.game_id,
                    step=context.observation.step_index,
                    player_x=player[0] if player else -1,
                    player_y=player[1] if player else -1,
                    goal_positions=game_model.goal_positions,
                    beam_cells=[],
                    mirror_x=player[0] if player else -1,
                    goals_lit=game_model.features.goals_lit_by_beam,
                    action_taken=decision.action_name,
                    outcome=self._active_name,
                )
                store_board_state(self._memory_db_path, state)
        except Exception:
            pass

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
