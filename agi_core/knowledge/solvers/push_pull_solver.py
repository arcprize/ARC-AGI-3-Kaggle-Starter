from __future__ import annotations

import logging
from typing import Any

from agi_core.knowledge.solvers import DynamicsSolver, SolverContext, register_solver
from agi_core.orch_or.game_model import DIR_TO_ACTION
from agi_core.types import ActionDecision

logger = logging.getLogger(__name__)


class PushPullSolver(DynamicsSolver):
    """Solver for push/pull (Sokoban-like) boards.

    Identifies movable blocks adjacent to the player and plans
    pushes toward goal areas.
    """

    dynamics = "push-pull"

    def __init__(self, memory_db_path=None):
        super().__init__(memory_db_path)
        self._pushed_blocks: set[tuple[int, int, int, int]] = set()
        self.reset_level()

    def reset_level(self):
        self._pushed_blocks.clear()

    def should_activate(self, ctx: SolverContext) -> bool:
        gm = ctx.game_model
        return gm.player_pos is not None and len(self._find_movable_blocks(gm, *gm.player_pos)) > 0

    def decide(self, ctx: SolverContext) -> ActionDecision | None:
        gm = ctx.game_model
        px, py = gm.player_pos
        blocks = self._find_movable_blocks(gm, px, py)
        if not blocks:
            return None

        block = blocks[0]
        # Determine push direction: from player through block
        bx, by = block[0], block[1]
        dx = bx - px
        dy = by - py
        push_dir = "right" if dx > 0 else ("left" if dx < 0 else ("down" if dy > 0 else "up"))
        vec = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

        # Check if push destination is clear
        dvx, dvy = vec[push_dir]
        push_to = (bx + dvx, by + dvy)
        expanded = self._build_walkable(gm)
        if push_to in expanded:
            # Move toward block to push it
            return ActionDecision(
                action_name=DIR_TO_ACTION.get(push_dir, "ACTION4"),
                source="solver:push_pull",
                confidence=0.65,
                reason=f"push_{push_dir}",
            )

        # Blocked — try pulling (move away from block while adjacent)
        pull_dir = {"up": "down", "down": "up", "left": "right", "right": "left"}[push_dir]
        return ActionDecision(
            action_name=DIR_TO_ACTION.get(pull_dir, "ACTION2"),
            source="solver:push_pull",
            confidence=0.50,
            reason=f"pull_{pull_dir}",
        )

    def _find_movable_blocks(self, gm, px, py):
        """Find non-player, non-background components adjacent to player."""
        bg = gm.features.background_color
        neighbors = {(px - 1, py), (px + 1, py), (px, py - 1), (px, py + 1)}
        blocks = []
        for c in gm.components:
            if c.touches_border:
                continue
            if c.dominant_color in (bg, -1, 0, 4, 11):
                continue
            if c.size > 20 or c.size < 2:
                continue
            for cx, cy in c.cells:
                if (cx, cy) in neighbors:
                    blocks.append((cx, cy, c.size, c.dominant_color))
                    break
        blocks.sort(key=lambda b: -b[2])
        return blocks

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


register_solver("push-pull", PushPullSolver)
