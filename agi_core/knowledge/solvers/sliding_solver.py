from __future__ import annotations

import logging
from typing import Any

from agi_core.knowledge.solvers import DynamicsSolver, SolverContext, register_solver
from agi_core.orch_or.game_model import DIR_TO_ACTION
from agi_core.types import ActionDecision

logger = logging.getLogger(__name__)


class SlidingSolver(DynamicsSolver):
    """Solver for sliding-physics boards — momentum compensation.

    Detects slide direction from frame differencing and plans movements
    that account for continued motion after input stops.
    """

    dynamics = "sliding"

    def __init__(self, memory_db_path=None):
        super().__init__(memory_db_path)
        self._last_pos: tuple[int, int] | None = None
        self._slide_dx: int = 0
        self._slide_dy: int = 0
        self._slide_frames: int = 0
        self.reset_level()

    def reset_level(self):
        self._last_pos = None
        self._slide_dx = 0
        self._slide_dy = 0
        self._slide_frames = 0

    def should_activate(self, ctx: SolverContext) -> bool:
        return ctx.game_model.player_pos is not None

    def decide(self, ctx: SolverContext) -> ActionDecision | None:
        gm = ctx.game_model
        px, py = gm.player_pos

        # Detect slide from movement between frames
        if self._last_pos is not None:
            dx = px - self._last_pos[0]
            dy = py - self._last_pos[1]
            if dx != 0 or dy != 0:
                self._slide_dx = dx
                self._slide_dy = dy
                self._slide_frames += 1
            else:
                self._slide_frames = 0
        self._last_pos = (px, py)

        # If sliding, try to compensate or use momentum
        if self._slide_frames >= 2:
            return self._compensate_slide(ctx, px, py)
        return None  # fall through to navigation

    def _compensate_slide(
        self, ctx: SolverContext, px: int, py: int
    ) -> ActionDecision | None:
        gm = ctx.game_model
        # Predict next position based on current slide
        next_px = px + self._slide_dx
        next_py = py + self._slide_dy
        expanded = self._build_walkable(gm)
        if (next_px, next_py) not in expanded:
            # About to slide into wall — don't input, let it stop
            return ActionDecision(
                action_name="ACTION1",  # safe no-op direction
                source="solver:sliding",
                confidence=0.50,
                reason="slide_brake",
            )
        # If goal is in slide direction, let it carry us
        if gm.features.goal_detectable and gm.goal_positions:
            nearest = min(gm.goal_positions, key=lambda g: abs(g[0]-px) + abs(g[1]-py))
            goal_dx = nearest[0] - px
            goal_dy = nearest[1] - py
            if (goal_dx > 0 and self._slide_dx > 0) or (goal_dy > 0 and self._slide_dy > 0):
                return None  # let navigation handle
        return ActionDecision(
            action_name=DIR_TO_ACTION.get("up" if self._slide_dy > 0 else "down", "ACTION1"),
            source="solver:sliding",
            confidence=0.50,
            reason="slide_compensate",
        )

    def _build_walkable(self, gm):
        w = set(gm.walkable)
        for c in gm.components:
            if not c.touches_border:
                for cx, cy in c.cells:
                    w.add((cx, cy))
        if gm.player_pos:
            w.add(gm.player_pos)
        for g in gm.goal_positions:
            w.add(g)
        return w


register_solver("sliding", SlidingSolver)
