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

    Handles vertical mirrors (reflect X, move LEFT/RIGHT) and horizontal
    mirrors (reflect Y, move UP/DOWN).  Supports multi-shape-piece levels
    (Level 3+) by cycling through entities via ACTION5 switches.
    """

    dynamics = "mirror"

    def __init__(self, memory_db_path: str | Path | None = None) -> None:
        super().__init__(memory_db_path)
        self._last_move_target: tuple[int, int] | None = None
        self._impassable_cells: set[tuple[int, int]] = set()
        self._cache_game_id: str = ""
        self._expanded_walkable_cache: set[tuple[int, int]] | None = None
        self._cached_mirror_gx: int = -1
        self._cached_mirror_gy: int = -1
        self._mirror_is_horizontal: bool = False
        self._positioned_steps: int = 0
        self._switch_count: int = 0  # 0=mirror, 1=first shape, 2=second shape, …
        self._last_shape_pos: tuple[int, int] | None = None
        self._shape_stuck_steps: int = 0
        self._level_id: str = ""
        if self._memory_db_path:
            self._memory_db_path.parent.mkdir(parents=True, exist_ok=True)
            self.reset_level()
            initialize_puzzle_schema(self._memory_db_path)

    def reset_level(self, full: bool = True) -> None:
        self._last_move_target = None
        self._impassable_cells.clear()
        self._cache_game_id = ""
        self._expanded_walkable_cache = None
        self._cached_mirror_gx = -1
        self._cached_mirror_gy = -1
        self._mirror_is_horizontal = False
        self._positioned_steps = 0
        self._last_shape_pos = None
        self._shape_stuck_steps = 0
        # _switch_count is NOT reset here — it persists across entity
        # switches within the same level.  The orchestrator's own
        # _switch_count (tracked separately) is used for switch-back logic.

    # ── Axis detection ────────────────────────────────────────────────

    @staticmethod
    def _detect_mirror_axis(mirror_comp) -> bool:
        mx1, my1, mx2, my2 = mirror_comp.bbox
        return (mx2 - mx1) > (my2 - my1)

    # ── Activation ─────────────────────────────────────────────────────

    def should_activate(self, ctx: SolverContext) -> bool:
        gm = ctx.game_model
        if gm.player_pos is None or not gm.goal_positions:
            return False
        mirror_comp = next((c for c in gm.components if c.dominant_color == 10), None)
        if mirror_comp is None:
            return False
        if gm.features.has_beam and not self._detect_mirror_axis(mirror_comp):
            return False
        horiz = self._detect_mirror_axis(mirror_comp)
        if horiz:
            return bool(gm.goal_positions)
        px, py = gm.player_pos
        mirr_cells = list(mirror_comp.cells)
        mirror_px = sum(cx for cx, _ in mirr_cells) // len(mirr_cells)
        goal_xs = [gx for gx, _ in gm.goal_positions]
        if not goal_xs:
            return False
        return ((px < mirror_px < min(goal_xs)) or
                (px > mirror_px > max(goal_xs)) or
                (px == mirror_px))

    # ── Decision ───────────────────────────────────────────────────────

    def decide(self, ctx: SolverContext) -> ActionDecision | None:
        try:
            return self._decide_impl(ctx)
        except Exception as exc:
            import traceback
            logger.error("MirrorSolver FAILED at step %d: %s\n%s",
                         ctx.plan_context.get("game_step", -1) if ctx.plan_context else -1,
                         exc, traceback.format_exc())
            return None

    def _decide_impl(self, ctx: SolverContext) -> ActionDecision | None:
        gm = ctx.game_model
        goals = gm.goal_positions
        if gm.player_pos is None or not goals:
            return None

        # Reset switch count on level change.
        _goal_cells_now = sum(1 for c in gm.components
                              if c.dominant_color == 11
                              for _ in c.cells)
        if hasattr(self, '_last_goal_cells') and self._last_goal_cells >= 0:
            if abs(_goal_cells_now - self._last_goal_cells) > 50:
                self._switch_count = 0
        self._last_goal_cells = _goal_cells_now

        px, py = gm.player_pos
        gs = gm.features.grid_scale
        if gs <= 1:
            return None

        # Find mirror component & axis
        all_c10 = [c for c in gm.components if c.dominant_color == 10]
        if not all_c10:
            return None
        if len(all_c10) > 1:
            _pp = gm.player_pos
            if _pp is not None:
                _ppx, _ppy = _pp
                all_c10.sort(key=lambda c: min(abs(cx - _ppx) + abs(cy - _ppy) for cx, cy in c.cells))
        self._mirror_is_horizontal = self._detect_mirror_axis(all_c10[0])

        # Mirror centre
        mirr_xs = [cx for c in all_c10 for cx, _ in c.cells]
        mirr_ys = [cy for c in all_c10 for _, cy in c.cells]
        if not mirr_xs:
            return None
        mirror_px = sum(mirr_xs) // len(mirr_xs)
        mirror_py = sum(mirr_ys) // len(mirr_ys)
        mirror_gx = mirror_px // gs
        mirror_gy = mirror_py // gs

        player_gx = px // gs
        player_gy = py // gs

        # Shape pieces
        shape_comps = [c for c in gm.components
                       if c.dominant_color in (5, 0)
                       and not c.touches_border
                       and 4 <= c.size <= 100]
        has_shape = bool(shape_comps)
        spx, spy = (px, py)
        if shape_comps:
            best_shape = max(shape_comps, key=lambda c: c.size)
            spx, spy = _component_center(best_shape)

        # Goals
        goal_cells = [
            c for comp in gm.components
            if comp.dominant_color == 11
            and comp.size <= 100
            and any(cx < gm.width - 1 for cx, _ in comp.cells)
            for c in comp.cells
        ]
        if not goal_cells:
            return None
        min_gpx = min(c[0] for c in goal_cells)
        min_gpy = min(c[1] for c in goal_cells)
        max_gpx = max(c[0] for c in goal_cells)
        max_gpy = max(c[1] for c in goal_cells)
        min_ggx = min_gpx // gs
        max_ggx = max_gpx // gs
        min_ggy = min_gpy // gs
        max_ggy = max_gpy // gs

        # ── Player IS the mirror? ──────────────────────────────────────
        if self._mirror_is_horizontal:
            player_is_mirror = abs(player_gy - mirror_gy) <= 1
            if not player_is_mirror and mirr_ys:
                live_mirror_py = sum(mirr_ys) // len(mirr_ys)
                for c in gm.components:
                    if c.dominant_color == 0 and not c.touches_border and 1 <= c.size <= 30:
                        _, cy = _component_center(c)
                        if abs(cy - live_mirror_py) <= gs:
                            player_is_mirror = True
                            py = cy
                            player_gy = py // gs
                            mirror_gy = live_mirror_py // gs
                            break
        else:
            player_is_mirror = abs(player_gx - mirror_gx) <= 1
            if not player_is_mirror and mirr_xs:
                live_mirror_px = sum(mirr_xs) // len(mirr_xs)
                for c in gm.components:
                    if c.dominant_color == 0 and not c.touches_border and 1 <= c.size <= 30:
                        cx, _ = _component_center(c)
                        if abs(cx - live_mirror_px) <= gs:
                            player_is_mirror = True
                            px = cx
                            player_gx = px // gs
                            mirror_gx = live_mirror_px // gs
                            break

        # Cache mirror position
        if self._cached_mirror_gx < 0 or player_is_mirror:
            self._cached_mirror_gx = mirror_gx
            self._cached_mirror_gy = mirror_gy
        if not player_is_mirror:
            mirror_gx = self._cached_mirror_gx
            mirror_gy = self._cached_mirror_gy

        # ── Player IS mirror: position it or switch ────────────────────
        if player_is_mirror and has_shape:
            # After the first switch (_switch_count > 0) the player should
            # be a shape piece — ignore false mirror detection from stray
            # color-0 cells near the mirror Y.
            if self._switch_count == 0:
                # For horizontal mirrors, position the mirror at gy=9 so
                # reflections of shapes (at gy=15-17) reach upper goals
                # (gy=1-3). Mirror starts at gy=16 → needs to move UP.
                if self._mirror_is_horizontal:
                    target_mgy = 9
                    dy = target_mgy - mirror_gy
                    if dy < 0:
                        return ActionDecision("ACTION1", "solver:mirror", 0.70,
                                              reason="mirror_nav_up")
                    elif dy > 0:
                        return ActionDecision("ACTION2", "solver:mirror", 0.70,
                                              reason="mirror_nav_down")
                    # Reached target — switch to first shape piece
                    if self._switch_count == 0:
                        self._switch_count = 1
                        ctx.plan_context["switch_attempted"] = 1
                        return ActionDecision("ACTION5", "solver:mirror", 0.70,
                                              reason="mirror_switch_horiz")
                    return ActionDecision("ACTION6", "solver:mirror", 0.60,
                                          reason="mirror_positioned_wait")
                else:
                    return self._navigate_mirror(
                        gm, ctx, shape_comps, mirror_gx, mirror_gy,
                        min_ggx, max_ggx, min_ggy, max_ggy, gs,
                    )
            # Fall through to shape-piece navigation.
            return self._navigate_shape(
                gm, ctx, (px, py), mirror_gx, mirror_gy, gs,
                min_ggx, max_ggx, min_ggy, max_ggy,
                shape_comps,
            )

        # ── Player is a shape piece ────────────────────────────────────
        if self._mirror_is_horizontal:
            target_gy_for_check = (2 * mirror_gy - max_ggy if self._switch_count == 1
                                   else 2 * mirror_gy - min_ggy)
            if abs(player_gy - target_gy_for_check) <= 1:
                return None
        else:
            if not ((player_gx < mirror_gx < min_ggx) or
                    (player_gx > mirror_gx > max_ggx)):
                return None

        return self._navigate_shape(
            gm, ctx, (px, py), mirror_gx, mirror_gy, gs,
            min_ggx, max_ggx, min_ggy, max_ggy,
            shape_comps,
        )

    # ── Mirror navigation (player IS the mirror) ──────────────────────

    def _navigate_mirror(
        self, gm, ctx, shape_comps,
        mirror_gx: int, mirror_gy: int,
        min_ggx: int, max_ggx: int,
        min_ggy: int, max_ggy: int,
        gs: int,
    ) -> ActionDecision | None:
        # For horizontal mirrors, skip positioning and switch directly.
        if self._mirror_is_horizontal:
            if shape_comps and self._switch_count < len(shape_comps):
                self._switch_count += 1
                ctx.plan_context["switch_attempted"] = self._switch_count
                return ActionDecision("ACTION5", "solver:mirror", 0.70,
                                      reason="mirror_switch_horiz")
            return ActionDecision("ACTION6", "solver:mirror", 0.60,
                                  reason="mirror_skip_horiz")

        # Vertical mirror: position it first.
        best_shape = max(shape_comps, key=lambda c: c.size)
        shape_right_gx = best_shape.bbox[2] // gs
        target_mgx = (min_ggx + shape_right_gx) // 2
        dx = target_mgx - mirror_gx
        if dx < 0:
            return ActionDecision("ACTION3", "solver:mirror", 0.70,
                                  reason="mirror_nav_left")
        elif dx > 0:
            return ActionDecision("ACTION4", "solver:mirror", 0.70,
                                  reason="mirror_nav_right")
        self._positioned_steps += 1
        if self._positioned_steps >= 3 and shape_comps and self._switch_count < len(shape_comps):
            self._switch_count += 1
            return ActionDecision("ACTION5", "solver:mirror", 0.70,
                                  reason="mirror_switch_to_shape")
        return ActionDecision("ACTION6", "solver:mirror", 0.60,
                              reason="mirror_positioned_wait")

    # ── Shape-piece navigation (after switch) ─────────────────────────

    def _navigate_shape(
        self, gm, ctx,
        player: tuple[int, int],
        mirror_gx: int, mirror_gy: int, gs: int,
        min_ggx: int, max_ggx: int,
        min_ggy: int, max_ggy: int,
        shape_comps: list | None = None,
    ) -> ActionDecision | None:
        px, py = player

        if self._mirror_is_horizontal:
            # Alternate targets per shape piece so reflections cover ALL goals.
            # Shape 1: above mirror, reflects to BOTTOM goals.
            # Shape 2: below mirror, reflects to TOP goals.
            if self._switch_count == 1:
                target_gy = 2 * mirror_gy - max_ggy  # shape above → reflect to bottom
            else:
                target_gy = 2 * mirror_gy - min_ggy  # shape below → reflect to top
            target_gx = (min_ggx + max_ggx) // 2
            target_py = max(0, min(gm.height - 1, target_gy * gs + 1))
            target_px = max(0, min(gm.width - 1, target_gx * gs + 1))
        else:
            target_gx = 2 * mirror_gx - max_ggx
            target_gy = min_ggy
            target_px = max(0, min(gm.width - 1, target_gx * gs + 1))
            target_py = target_gy * gs + 1

        # Stuck detection: switch to next shape piece if stuck
        if self._last_shape_pos == (px, py):
            self._shape_stuck_steps += 1
        else:
            self._shape_stuck_steps = 0
        self._last_shape_pos = (px, py)

        max_switches = len(shape_comps) if shape_comps else 0
        if self._shape_stuck_steps >= 20 and self._switch_count < max_switches:
            self._switch_count += 1
            ctx.plan_context["switch_attempted"] = self._switch_count
            return ActionDecision("ACTION5", "solver:mirror", 0.65,
                                  reason="mirror_switch_other_shape")

        if abs(px - target_px) + abs(py - target_py) <= 1:
            return ActionDecision("ACTION6", "solver:mirror", 0.60,
                                  reason="mirror_placed_wait")

        expanded = self._build_expanded_walkable(gm, ctx)
        try:
            path = astar(player, (target_px, target_py), expanded,
                         gm.width, gm.height, max_steps=5000)
        except Exception:
            path = None
        if path and len(path) > 1:
            nx, ny = path[1]
            d = ("right" if nx > px else "left" if nx < px else
                 "down" if ny > py else "up")
            self._last_move_target = (nx, ny)
            return ActionDecision(
                DIR_TO_ACTION.get(d, "ACTION1"), "solver:mirror", 0.75,
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
        vec = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
        for d in candidates:
            if d in failed:
                continue
            dvx, dvy = vec[d]
            nx, ny = px + dvx, py + dvy
            if not expanded or (nx, ny) in expanded:
                return ActionDecision(
                    DIR_TO_ACTION.get(d, "ACTION1"), "solver:mirror", 0.55,
                    reason=f"mirror_heuristic_{d}",
                )
        for d, (dvx, dvy) in vec.items():
            nx, ny = px + dvx, py + dvy
            if 0 <= nx < gm.width and 0 <= ny < gm.height:
                return ActionDecision(
                    DIR_TO_ACTION.get(d, "ACTION1"), "solver:mirror", 0.40,
                    reason=f"mirror_forced_{d}",
                )
        return None

    # ── Walkable helper ───────────────────────────────────────────────

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
