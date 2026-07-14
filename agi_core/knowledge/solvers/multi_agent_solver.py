from __future__ import annotations

import logging
from typing import Any

from agi_core.knowledge.solvers import DynamicsSolver, SolverContext, register_solver
from agi_core.types import ActionDecision

logger = logging.getLogger(__name__)


class MultiAgentSolver(DynamicsSolver):
    """Solver for multi-agent boards — switch between entities with ACTION5."""

    dynamics = "multi-agent"

    def __init__(self, memory_db_path=None):
        super().__init__(memory_db_path)
        self._switch_count: int = 0
        self._last_entity_count: int = 0
        self.reset_level()

    def reset_level(self):
        self._switch_count = 0
        self._last_entity_count = 0

    def should_activate(self, ctx: SolverContext) -> bool:
        gm = ctx.game_model
        return gm.player_pos is not None and len(self._get_entities(gm)) >= 2

    def decide(self, ctx: SolverContext) -> ActionDecision | None:
        gm = ctx.game_model
        entities = self._get_entities(gm)
        if len(entities) <= 1:
            return None
        player_color = gm.features.player_color
        # If current entity can reach goals directly, let navigation handle it
        if gm.features.goal_detectable and gm.goal_positions:
            return None  # fall through to navigation
        # Try switching to another entity
        if self._switch_count < len(entities):
            self._switch_count += 1
            return ActionDecision(
                action_name="ACTION5",
                source="solver:multi_agent",
                confidence=0.60,
                reason="switch_entity",
            )
        return None

    def _get_entities(self, gm):
        bg = gm.features.background_color
        return [c for c in gm.components
                if not c.touches_border
                and c.dominant_color not in (bg, -1, 0, 4, 11)
                and c.size >= 4]

    def learn(self, ctx: SolverContext, outcome: str) -> None:
        pass


register_solver("multi-agent", MultiAgentSolver)
