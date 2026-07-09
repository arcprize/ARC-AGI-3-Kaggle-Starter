from __future__ import annotations

import logging
from typing import Any

from agi_core.knowledge.solvers import DynamicsSolver, SolverContext, register_solver
from agi_core.orch_or.game_model import DIR_TO_ACTION
from agi_core.types import ActionDecision

logger = logging.getLogger(__name__)


class GravitySolver(DynamicsSolver):
    """Solver for gravity / platformer boards.

    Detects gravity direction, platform edges, and plans jump arcs
    to navigate the player toward goals.
    """

    dynamics = "gravity"

    def __init__(self, memory_db_path=None):
        super().__init__(memory_db_path)
        self._last_pos: tuple[int, int] | None = None
        self._falling: bool = False
        self._jump_cooldown: int = 0

    def reset_level(self):
        self._last_pos = None
        self._falling = False
        self._jump_cooldown = 0

    def should_activate(self, ctx: SolverContext) -> bool:
        gm = ctx.game_model
        return gm.player_pos is not None and gm.features.has_gravity

    def decide(self, ctx: SolverContext) -> ActionDecision | None:
        gm = ctx.game_model
        px, py = gm.player_pos

        # Detect falling — player moved down without input
        if self._last_pos is not None:
            dy = py - self._last_pos[1]
            self._falling = dy > 0
        self._last_pos = (px, py)

        if self._jump_cooldown > 0:
            self._jump_cooldown -= 1

        # Check if standing on solid ground (walkable cell below)
        below_walkable = (px, py + 1) in gm.walkable
        if not below_walkable and not self._falling:
            # At edge — jump
            if self._jump_cooldown <= 0:
                self._jump_cooldown = 3
                return ActionDecision(
                    action_name="ACTION1",
                    source="solver:gravity",
                    confidence=0.55,
                    reason="jump",
                )

        # Navigate toward goal
        if gm.features.goal_detectable and gm.goal_positions:
            return None  # let navigation handle

        # Default: move right (typical platformer scroll direction)
        return ActionDecision(
            action_name="ACTION4",
            source="solver:gravity",
            confidence=0.45,
            reason="move_right",
        )


register_solver("gravity", GravitySolver)
