from __future__ import annotations

import logging
from typing import Any

from agi_core.knowledge.solvers import DynamicsSolver, SolverContext, register_solver
from agi_core.orch_or.game_model import DIR_TO_ACTION
from agi_core.orch_or.search import astar
from agi_core.types import ActionDecision

logger = logging.getLogger(__name__)


class InventorySolver(DynamicsSolver):
    """Solver for inventory/collection boards.

    Identifies collectible items, plans a route through all of them,
    then navigates to the exit.
    """

    dynamics = "inventory"

    def __init__(self, memory_db_path=None):
        super().__init__(memory_db_path)
        self._collected: set[tuple[int, int]] = set()
        self._visited: set[tuple[int, int]] = set()
        self.reset_level()

    def reset_level(self):
        self._collected.clear()
        self._visited.clear()

    def should_activate(self, ctx: SolverContext) -> bool:
        gm = ctx.game_model
        return gm.player_pos is not None and len(self._find_collectibles(gm)) >= 2

    def decide(self, ctx: SolverContext) -> ActionDecision | None:
        gm = ctx.game_model
        px, py = gm.player_pos

        # Track visited cells to detect collected items
        self._visited.add((px, py))
        if py < len(gm.board) and px < len(gm.board[py]):
            cell_color = gm.board[py][px]
            if cell_color not in (gm.features.background_color, 0):
                self._collected.add((px, py))

        # Find remaining collectibles
        remaining = [c for c in self._find_collectibles(gm)
                     if not any(abs(c[0]-cx)+abs(c[1]-cy) <= 2 for cx, cy in self._collected)]

        if not remaining:
            return None  # all collected, let navigation take to exit/goal

        # Route to nearest uncollected item
        nearest = min(remaining, key=lambda c: abs(c[0]-px)+abs(c[1]-py))
        expanded = self._build_walkable(gm)
        path = astar((px, py), nearest, expanded, gm.width, gm.height, max_steps=5000)
        if path and len(path) > 1:
            nx, ny = path[1]
            d = "right" if nx > px else ("left" if nx < px else ("down" if ny > py else "up"))
            return ActionDecision(
                action_name=DIR_TO_ACTION.get(d, "ACTION4"),
                source="solver:inventory",
                confidence=0.65,
                reason=f"collect_{d}",
            )
        return None

    def _find_collectibles(self, gm) -> list[tuple[int, int]]:
        bg = gm.features.background_color
        items = []
        for c in gm.components:
            if c.touches_border:
                continue
            if c.dominant_color in (bg, -1, 0, 4, 11, gm.features.player_color):
                continue
            if 1 <= c.size <= 12:
                for cx, cy in c.cells:
                    items.append((cx, cy))
        return items

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


register_solver("inventory", InventorySolver)
