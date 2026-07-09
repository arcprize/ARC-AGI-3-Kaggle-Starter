from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

from agi_core.knowledge.puzzle_memory import (
    initialize_puzzle_schema,
    load_puzzle_solution,
    query_puzzle_solution_fuzzy,
    query_similar_state,
    store_board_state,
)
from agi_core.knowledge.solvers import DynamicsSolver, SolverContext, register_solver
from agi_core.orch_or.analyzer import _component_center
from agi_core.orch_or.game_model import DIR_TO_ACTION
from agi_core.orch_or.search import astar
from agi_core.types import ActionDecision

logger = logging.getLogger(__name__)


class MirrorSolver(DynamicsSolver):
    """Solver for mirror reflection puzzles (color-10 mirror, no beam).

    Detects whether the player entity IS the mirror (Level 2+) or controls
    a separate shape piece (Level 1). Navigates the correct entity to the
    computed reflection target position.
    Handles mirror reflection puzzles (color-10 mirror, no beam).
    """

    dynamics = "mirror"

    def __init__(self, memory_db_path: str | Path | None = None) -> None:
        super().__init__(memory_db_path)
        self._last_move_target: tuple[int, int] | None = None
        self._impassable_cells: set[tuple[int, int]] = set()
        self._cache_game_id: str = ""
        self._expanded_walkable_cache: set[tuple[int, int]] | None = None
        self._cached_mirror_gx: int = -1
        if self._memory_db_path:
            self._memory_db_path.parent.mkdir(parents=True, exist_ok=True)
            self.reset_level()
            initialize_puzzle_schema(self._memory_db_path)

    def reset_level(self) -> None:
        self._last_move_target = None
        self._impassable_cells.clear()
        self._cache_game_id = ""
        self._expanded_walkable_cache = None
        self._cached_mirror_gx = -1

    def should_activate(self, ctx: SolverContext) -> bool:
        gm = ctx.game_model
        if gm.player_pos is None or not gm.goal_positions:
            return False
        if gm.features.has_beam:
            return False
        # Check: mirror (color 10) with goals on opposite side
        mirror_comp = next((c for c in gm.components if c.dominant_color == 10), None)
        if mirror_comp is None:
            return False
        px, py = gm.player_pos
        mirror_px = sum(cx for cx, _ in mirror_comp.cells) // len(mirror_comp.cells)
        goal_xs = [gx for gx, _ in gm.goal_positions]
        if not goal_xs:
            return False
        return (px < mirror_px < min(goal_xs)) or (px > mirror_px > max(goal_xs))

    def decide(self, ctx: SolverContext) -> ActionDecision | None:
        gm = ctx.game_model
        player = gm.player_pos
        goals = gm.goal_positions
        if player is None or not goals:
            return None

        player = gm.player_pos
        px, py = player
        gs = gm.features.grid_scale
        if gs <= 1:
            return None

        # Find mirror component
        all_c10 = [c for c in gm.components if c.dominant_color == 10]
        if not all_c10:
            return None
        # Infer mirror position from color-10 cells or reflection cells
        mirr_xs = [cx for c in all_c10 for cx, _ in c.cells]
        if mirr_xs:
            mirror_px = sum(mirr_xs) // len(mirr_xs)
            mirror_gx = mirror_px // gs
        else:
            # Fall back to reflection-based inference
            beam_refs = [c for c in gm.components if c.dominant_color == 4 and not c.touches_border]
            shape_comps = [c for c in gm.components
                           if c.dominant_color in (5, 0) and not c.touches_border and 1 <= c.size <= 100]
            if not beam_refs or not shape_comps:
                return None
            best_shape = max(shape_comps, key=lambda c: c.size)
            spx, _ = _component_center(best_shape)
            all_ref = set()
            for br in beam_refs:
                all_ref.update(br.cells)
            votes = Counter()
            for sx, sy in best_shape.cells:
                for rx, ry in all_ref:
                    if abs(ry - sy) <= gs * 2:
                        votes[sx + rx] += 1
            if not votes:
                return None
            mirror_gx = votes.most_common(1)[0][0] // (2 * gs)

        # Cache mirror position
        if self._cached_mirror_gx < 0:
            self._cached_mirror_gx = mirror_gx
        if abs(px // gs - mirror_gx) > 1:
            mirror_gx = self._cached_mirror_gx

        player_gx = px // gs

        # Find shape piece for target computation
        shape_comps = [c for c in gm.components
                       if c.dominant_color in (5, 0) and not c.touches_border and 1 <= c.size <= 100]
        has_shape = bool(shape_comps)
        spx = px
        spy = py
        if shape_comps:
            best_shape = max(shape_comps, key=lambda c: c.size)
            spx, spy = _component_center(best_shape)

        # Find goal component (exclude energy bar at rightmost column)
        goal_comp = next((c for c in gm.components
                          if c.dominant_color == 11
                          and any(cx < gm.width - 1 for cx, _ in c.cells)), None)
        if goal_comp is None:
            return None
        goal_cells = [c for c in goal_comp.cells
                      if c[0] < gm.width - 1 and c[1] < gm.height - 1]
        if not goal_cells:
            return None
        min_gpx = min(c[0] for c in goal_cells)
        min_gpy = min(c[1] for c in goal_cells)
        max_gpx = max(c[0] for c in goal_cells)
        min_ggx = min_gpx // gs
        max_ggx = max_gpx // gs
        min_ggy = min_gpy // gs

        # Determine if player IS the mirror by checking:
        # 1. Player grid x is near mirror grid x, OR
        # 2. Player is at a color-0 cell at the mirror's position
        player_is_mirror = abs(player_gx - mirror_gx) <= 1
        if not player_is_mirror:
            mirror_px = mirror_gx * gs + 1
            for c in gm.components:
                if c.dominant_color == 0 and not c.touches_border and 1 <= c.size <= 30:
                    cx, _ = _component_center(c)
                    if abs(cx - mirror_px) <= gs:
                        player_is_mirror = True
                        px = cx
                        player_gx = px // gs
                        break


        if player_is_mirror and has_shape:
            # Player IS the mirror — compute mirror target from shape + goals
            best_shape_comp = max(shape_comps, key=lambda c: c.size)
            shape_right_gx = best_shape_comp.bbox[2] // gs
            target_mgx = (min_ggx + shape_right_gx) // 2
            dx = target_mgx - mirror_gx
            if dx < 0:
                return ActionDecision(
                    action_name="ACTION3",
                    source="solver:mirror",
                    confidence=0.70,
                    reason="mirror_nav_left",
                )
            elif dx > 0:
                return ActionDecision(
                    action_name="ACTION4",
                    source="solver:mirror",
                    confidence=0.70,
                    reason="mirror_nav_right",
                )
            # At correct x — wait
            return ActionDecision(
                action_name="ACTION6",
                source="solver:mirror",
                confidence=0.60,
                reason="mirror_positioned_wait",
            )

        # Player is NOT the mirror — standard shape piece reflection
        if not ((player_gx < mirror_gx < min_ggx) or
                (player_gx > mirror_gx > max_ggx)):
            return None

        target_gx = 2 * mirror_gx - max_ggx
        target_gy = min_ggy
        max_gx = (gm.width - 1) // gs
        target_gx = max(0, min(max_gx, target_gx))
        target_px = target_gx * gs + 1
        target_py = target_gy * gs + 1

        if abs(px - target_px) + abs(py - target_py) <= 1:
            return ActionDecision(
                action_name="ACTION6",
                source="solver:mirror",
                confidence=0.60,
                reason="mirror_placed_wait",
            )

        expanded = self._build_expanded_walkable(gm, ctx)
        path = astar(player, (target_px, target_py), expanded,
                     gm.width, gm.height, max_steps=5000)
        if path and len(path) > 1:
            nx, ny = path[1]
            d = "right" if nx > px else ("left" if nx < px else ("down" if ny > py else "up"))
            self._last_move_target = (nx, ny)
            return ActionDecision(
                action_name=DIR_TO_ACTION.get(d, "ACTION1"),
                source="solver:mirror",
                confidence=0.75,
                reason=f"mirror_placement_{d}",
            )

        # Heuristic fallback
        failed = ctx.plan_context.get("failed_directions", set())
        dx = target_px - px
        dy = target_py - py
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
                    source="solver:mirror",
                    confidence=0.55,
                    reason=f"mirror_heuristic_{d}",
                )
        return None

    def _build_expanded_walkable(self, gm, ctx):
        gid = ctx.episode.get("game_id", "")
        if self._expanded_walkable_cache is not None and self._cache_game_id == gid:
            w = set(self._expanded_walkable_cache)
        else:
            w = set(gm.walkable)
            for c in gm.components:
                if not c.touches_border:
                    for cx, cy in c.cells:
                        w.add((cx, cy))
            self._expanded_walkable_cache = set(w)
            self._cache_game_id = gid
        if gm.player_pos:
            w.add(gm.player_pos)
        for gx, gy in gm.goal_positions:
            w.add((gx, gy))
        w -= self._impassable_cells
        return w


register_solver("puzzle", MirrorSolver)
register_solver("mirror", MirrorSolver)
