from __future__ import annotations

import logging
from typing import Any

from agi_core.knowledge.solvers import DynamicsSolver, SolverContext, register_solver
from agi_core.orch_or.game_model import DIR_TO_ACTION
from agi_core.orch_or.search import astar
from agi_core.types import ActionDecision

logger = logging.getLogger(__name__)


class NavigationSolver(DynamicsSolver):
    """Solver for navigation / pathfinding boards.

    Uses A* to navigate the player toward the nearest goal. Falls back to
    exploration when no goal is detectable. Handles interactive obstacles
    via ACTION6 clicks.
    """

    dynamics = "navigation"

    def __init__(self, memory_db_path: str | Path | None = None) -> None:
        super().__init__(memory_db_path)
        self._last_pos: tuple[int, int] | None = None
        self._stall_frames: int = 0
        self._last_dir: str = ""
        self._stal_dirs: list[str] = ["down", "right", "up", "left"]
        self.reset_level()

    def reset_level(self) -> None:
        self._last_pos = None
        self._stall_frames = 0
        self._last_dir = ""

    def should_activate(self, ctx: SolverContext) -> bool:
        gm = ctx.game_model
        return gm.player_pos is not None

    def decide(self, ctx: SolverContext) -> ActionDecision | None:
        gm = ctx.game_model
        px, py = gm.player_pos
        if self._last_pos == (px, py):
            self._stall_frames += 1
        else:
            self._stall_frames = 0
        self._last_pos = (px, py)

        if gm.features.goal_detectable and gm.goal_positions:
            return self._navigate_to_goal(ctx, px, py)
        return self._explore(ctx, px, py)

    def _navigate_to_goal(
        self, ctx: SolverContext, px: int, py: int
    ) -> ActionDecision | None:
        gm = ctx.game_model
        goals = gm.goal_positions
        if not goals:
            return self._explore(ctx, px, py)

        # Pick nearest goal
        nearest = min(goals, key=lambda g: abs(g[0] - px) + abs(g[1] - py))
        gx, gy = nearest

        # Check proximity
        proximity = 2
        try:
            prox_rows = ctx.plan_context.get("board_goal_proximity", [])
            if prox_rows:
                proximity = int(prox_rows[0].get("proximity", 2))
        except Exception:
            pass
        exit_type = "position"
        try:
            ext_rows = ctx.plan_context.get("board_exit_type", [])
            if ext_rows:
                exit_type = str(ext_rows[0].get("exit", "position"))
        except Exception:
            pass
        if exit_type != "action" and abs(gx - px) + abs(gy - py) <= proximity:
            return None  # already at goal

        # Build expanded walkable set
        expanded = self._build_walkable(gm)
        path = astar((px, py), (gx, gy), expanded, gm.width, gm.height, max_steps=5000)
        if path and len(path) > 1:
            nx, ny = path[1]
            d = "right" if nx > px else ("left" if nx < px else ("down" if ny > py else "up"))
            self._last_dir = d
            return ActionDecision(
                action_name=DIR_TO_ACTION.get(d, "ACTION1"),
                source="solver:navigation",
                confidence=0.75,
                reason=f"navigate_{d}",
            )

        # A* failed — try heuristic movement
        failed = ctx.plan_context.get("failed_directions", set())
        dx, dy = gx - px, gy - py
        candidates = []
        if dx > 0: candidates.append("right")
        elif dx < 0: candidates.append("left")
        if dy > 0: candidates.append("down")
        elif dy < 0: candidates.append("up")
        for d in candidates:
            if d in failed:
                continue
            vec = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
            dvx, dvy = vec[d]
            nx, ny = px + dvx, py + dvy
            if not expanded or (nx, ny) in expanded:
                return ActionDecision(
                    action_name=DIR_TO_ACTION.get(d, "ACTION1"),
                    source="solver:navigation",
                    confidence=0.55,
                    reason=f"heuristic_{d}",
                )
        return None

    def _explore(
        self, ctx: SolverContext, px: int, py: int
    ) -> ActionDecision | None:
        gm = ctx.game_model
        if self._stall_frames >= 3:
            self._stal_dirs = self._stal_dirs[1:] + self._stal_dirs[:1]
        d = self._stal_dirs[0]
        vec = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
        dvx, dvy = vec[d]
        nx, ny = px + dvx, py + dvy
        expanded = self._build_walkable(gm)
        if not expanded or (nx, ny) in expanded:
            return ActionDecision(
                action_name=DIR_TO_ACTION.get(d, "ACTION2"),
                source="solver:navigation",
                confidence=0.50,
                reason=f"explore_{d}",
            )
        return None

    def _build_walkable(self, gm) -> set[tuple[int, int]]:
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


register_solver("navigation", NavigationSolver)
