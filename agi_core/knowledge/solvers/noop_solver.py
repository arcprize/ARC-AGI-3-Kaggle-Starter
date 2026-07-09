from __future__ import annotations

from agi_core.knowledge.solvers import DynamicsSolver, SolverContext, register_solver
from agi_core.types import ActionDecision


class NoopSolver(DynamicsSolver):
    """Solver for static boards — no player action needed."""

    dynamics = "static"

    def should_activate(self, ctx: SolverContext) -> bool:
        gm = ctx.game_model
        return gm.player_pos is None or not gm.features.goal_detectable

    def decide(self, ctx: SolverContext) -> ActionDecision | None:
        return None


register_solver("static", NoopSolver)
