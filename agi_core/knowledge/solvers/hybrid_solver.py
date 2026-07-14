from __future__ import annotations

import logging
from typing import Any

from agi_core.knowledge.solvers import (
    DynamicsSolver,
    SolverContext,
    get_solver,
    get_all_solvers,
    register_solver,
)
from agi_core.types import ActionDecision

logger = logging.getLogger(__name__)


class HybridSolver(DynamicsSolver):
    """Solver for hybrid boards — multiple dynamics detected.

    Delegates to individual solvers in priority order. First solver
    that returns a valid decision wins.
    """

    dynamics = "hybrid"

    # Priority-ordered list of dynamics to try
    SOLVER_PRIORITY = [
        "beam",
        "signal",
        "push-pull",
        "multi-agent",
        "gravity",
        "sliding",
        "inventory",
        "navigation",
        "mirror",
        "puzzle",
        "static",
        "cellular-automata",
        "pattern-transform",
    ]

    def __init__(self, memory_db_path=None):
        super().__init__(memory_db_path)
        self._sub_solvers: dict[str, DynamicsSolver] = {}

    def reset_level(self):
        for s in self._sub_solvers.values():
            s.reset_level()
        self._sub_solvers.clear()

    def should_activate(self, ctx: SolverContext) -> bool:
        # Activate for any board that has at least one sub-solver
        gm = ctx.game_model
        return gm.player_pos is not None

    def decide(self, ctx: SolverContext) -> ActionDecision | None:
        gm = ctx.game_model
        detected = set(gm.features.dynamics) if gm.features.dynamics else set()
        if not detected:
            detected.add(gm.features.primary_dynamics)

        for dyn in self.SOLVER_PRIORITY:
            if dyn not in detected:
                continue
            if dyn not in self._sub_solvers:
                cls = get_solver(dyn)
                if cls is None:
                    continue
                self._sub_solvers[dyn] = cls(memory_db_path=self._memory_db_path)
            try:
                sub = self._sub_solvers[dyn]
                sub_ctx = SolverContext(
                    game_model=ctx.game_model,
                    plan_context=ctx.plan_context,
                    episode=ctx.episode,
                    memory=ctx.memory,
                )
                if sub.should_activate(sub_ctx):
                    result = sub.decide(sub_ctx)
                    if result is not None:
                        return result
            except Exception as exc:
                logger.debug("hybrid sub-solver %s failed: %s", dyn, exc)
                continue

        return None

    def learn(self, ctx: SolverContext, outcome: str) -> None:
        for s in self._sub_solvers.values():
            try:
                s.learn(ctx, outcome)
            except Exception:
                pass


register_solver("hybrid", HybridSolver)
