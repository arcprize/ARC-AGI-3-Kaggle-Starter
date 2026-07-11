from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Any

from agi_core.knowledge.solvers import DynamicsSolver, SolverContext, register_solver
from agi_core.orch_or.game_model import DIR_TO_ACTION
from agi_core.orch_or.search import astar
from agi_core.types import ActionDecision

logger = logging.getLogger(__name__)

_MAX_REFLECT_DEPTH = 12


def _find_mirror_info(
    components: list,
    width: int,
    height: int,
) -> dict[str, Any] | None:
    c10 = [c for c in components if c.dominant_color == 10]
    if not c10:
        return None
    comp = max(c10, key=lambda c: c.size)
    x1, y1, x2, y2 = comp.bbox
    horizontal = (x2 - x1) > (y2 - y1)
    if horizontal:
        return {"horizontal": True, "axis_px": 0, "axis_py": (y1 + y2) // 2}
    return {"horizontal": False, "axis_px": (x1 + x2) // 2, "axis_py": 0}


def _get_shape_pixels(
    components: list,
    width: int,
    height: int,
) -> list[list[tuple[int, int]]]:
    # Collect color-5 interior components (the actual shape pieces)
    color5_shapes = []
    for c in components:
        if c.dominant_color != 5:
            continue
        if c.touches_border:
            continue
        if not (4 <= c.size <= 100):
            continue
        x1, y1, x2, y2 = c.bbox
        if x1 <= 1 or y1 <= 1 or x2 >= width - 2 or y2 >= height - 2:
            continue
        color5_shapes.append(c)

    # Merge color-0 overlays into their parent color-5 shape
    result = []
    for c5 in color5_shapes:
        cells = set(c5.cells)
        x1, y1, x2, y2 = c5.bbox
        # Find any color-0 component overlapping this shape
        for c0 in components:
            if c0.dominant_color != 0:
                continue
            if c0.touches_border:
                continue
            # Check bounding box overlap
            ox1, oy1, ox2, oy2 = c0.bbox
            if ox1 > x2 or ox2 < x1 or oy1 > y2 or oy2 < y1:
                continue
            # Check cell overlap
            if any(cell in cells for cell in c0.cells):
                cells.update(c0.cells)
        result.append(list(cells))
    return result


def get_goal_cells(components: list, width: int) -> list[tuple[int, int]]:
    return [
        cell
        for comp in components
        if comp.dominant_color == 11
        and any(cx < width - 1 for cx, _ in comp.cells)
        for cell in comp.cells
    ]


def predict_beam_coverage(
    shape_cells: list[tuple[int, int]],
    mirror_info: dict[str, Any],
    board_width: int,
    board_height: int,
    gs: int = 3,
) -> set[tuple[int, int]]:
    """Simulate beam propagation in GRID coordinates (0-20).

    The game engine's beam grid is indexed by GRID coordinates.
    All inputs (shape_cells, mirror axis) can be in pixel coordinates;
    they are converted to grid coordinates internally.
    Returns covered grid cells as pixel positions (the grid cell center).
    """
    grid_w = board_width // gs
    grid_h = board_height // gs
    if grid_w <= 0 or grid_h <= 0:
        return set()
    # Convert pixel coords to grid coords
    grid_cells = {(x // gs, y // gs) for x, y in shape_cells
                  if 0 <= x // gs < grid_w and 0 <= y // gs < grid_h}
    if not grid_cells:
        return set()
    # Mirror axis in grid coordinates
    horizontal = mirror_info.get("horizontal", True)
    axis_gx = mirror_info.get("axis_px", 0) // gs
    axis_gy = mirror_info.get("axis_py", 0) // gs
    covered: set[tuple[int, int]] = set(grid_cells)
    q: deque[tuple[int, int, int]] = deque()
    visited: set[tuple[int, int]] = set(grid_cells)
    for gx, gy in grid_cells:
        q.append((gx, gy, 0))
    while q:
        gx, gy, depth = q.popleft()
        if depth >= _MAX_REFLECT_DEPTH:
            continue
        if horizontal:
            rgx, rgy = gx, 2 * axis_gy - gy
        else:
            rgx, rgy = 2 * axis_gx - gx, gy
        if 0 <= rgx < grid_w and 0 <= rgy < grid_h and (rgx, rgy) not in visited:
            visited.add((rgx, rgy))
            covered.add((rgx, rgy))
            q.append((rgx, rgy, depth + 1))
    # Convert grid cells to ALL pixel positions within each cell
    # so that pixel-level goal checks match correctly.
    pixel_covered = set()
    for gx, gy in covered:
        base_x = gx * gs
        base_y = gy * gs
        for dx in range(gs):
            for dy in range(gs):
                pixel_covered.add((base_x + dx, base_y + dy))
    return pixel_covered


def find_best_placement(
    shape_cells: list[tuple[int, int]],
    remaining_goals: set[tuple[int, int]],
    mirror_info: dict[str, Any],
    board_width: int,
    board_height: int,
    current_center: tuple[int, int] | None = None,
    forbidden_cells: set[tuple[int, int]] | None = None,
    gs: int = 3,
) -> tuple[int, int] | None:
    if not shape_cells or not remaining_goals:
        return None
    xs = [x for x, _ in shape_cells]
    ys = [y for _, y in shape_cells]
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    shape_w = x2 - x1 + 1
    shape_h = y2 - y1 + 1
    if shape_w <= 0 or shape_h <= 0:
        return None

    rel_set = {(x - x1, y - y1) for x, y in shape_cells}
    best_score: tuple[int, int] | None = None
    best_pos: tuple[int, int] | None = None

    for tx in range(0, max(1, board_width - shape_w + 1)):
        for ty in range(0, max(1, board_height - shape_h + 1)):
            placed = {(tx + dx, ty + dy) for dx, dy in rel_set}
            # Skip if any placed cell overlaps the mirror
            if forbidden_cells and placed & forbidden_cells:
                continue
            covered = predict_beam_coverage(
                list(placed), mirror_info, board_width, board_height, gs=gs,
            )
            hits = len(covered & remaining_goals)
            if hits == 0:
                continue
            cx = sum(x for x, _ in placed) // len(placed)
            cy = sum(y for _, y in placed) // len(placed)
            dist = abs(cx - current_center[0]) + abs(cy - current_center[1]) if current_center else 0
            score = (hits, -dist)
            if best_score is None or score > best_score:
                best_score = score
                best_pos = (tx, ty)

    return best_pos


def compute_coverage_for_all_shapes(
    shapes: list[list[tuple[int, int]]],
    goal_cells: list[tuple[int, int]],
    mirror_info: dict[str, Any],
    board_width: int,
    board_height: int,
    forbidden_cells: set[tuple[int, int]] | None = None,
    gs: int = 3,
) -> list[dict[str, Any]]:
    remaining = set(goal_cells)
    stage_targets: list[dict[str, Any]] = []

    for idx, shape_cells in enumerate(shapes):
        if not shape_cells:
            continue
        # Use current position as center preference (no perturbation on first pass)
        current_center = (
            sum(x for x, _ in shape_cells) // len(shape_cells),
            sum(y for _, y in shape_cells) // len(shape_cells),
        ) if shape_cells else None
        best_pos = find_best_placement(
            shape_cells, remaining, mirror_info,
            board_width, board_height,
            current_center=current_center,
            forbidden_cells=forbidden_cells,
            gs=gs,
        )
        if best_pos is None:
            continue
        tx, ty = best_pos
        xs = [x for x, _ in shape_cells]
        ys = [y for _, y in shape_cells]
        x1, y1 = min(xs), min(ys)
        rel_set = {(x - x1, y - y1) for x, y in shape_cells}
        placed = {(tx + dx, ty + dy) for dx, dy in rel_set}
        covered = predict_beam_coverage(
            list(placed), mirror_info, board_width, board_height, gs=gs,
        )
        newly_covered = covered & remaining
        remaining -= newly_covered

        cx = sum(x for x, _ in placed) // len(placed) if placed else tx
        cy = sum(y for _, y in placed) // len(placed) if placed else ty
        placed_xs = [x for x, _ in placed]
        placed_ys = [y for _, y in placed]

        stage_targets.append({
            "shape_index": idx,
            "target_anchor": [cx, cy],
            "target_bbox": [
                min(placed_xs) if placed_xs else tx,
                min(placed_ys) if placed_ys else ty,
                max(placed_xs) if placed_xs else tx,
                max(placed_ys) if placed_ys else ty,
            ],
            "covered_goals": sorted(newly_covered),
        })

    return stage_targets


class BeamCoverageSolver(DynamicsSolver):
    """Solver for beam-coverage puzzles where shape pieces emit beam
    that reflects off fixed mirrors to cover goals.

    Paradigm: shape pieces are beam sources, mirrors are passive reflectors.
    """

    dynamics = "beam-coverage"

    def __init__(self, memory_db_path: str | Path | None = None) -> None:
        super().__init__(memory_db_path)
        self._cached_targets: list[dict[str, Any]] | None = None
        self._cached_frame: int = -1
        self._switch_count: int = 0
        self._last_player_pos: tuple[int, int] | None = None
        self._shape_stuck_steps: int = 0
        self._move_history: list[str] = []
        self._last_levels_completed: int | None = None
        self._switched: bool = False
        self._last_switch_step: int = -1
        self._board_sig: tuple = ()
        self._retry_count: int = 0
        self._both_placed_steps: int = 0

    def reset_level(self) -> None:
        self._cached_targets = None
        self._cached_frame = -1
        self._switch_count = 0
        self._last_player_pos = None
        self._shape_stuck_steps = 0
        self._move_history = []
        self._last_levels_completed = None
        self._switched = False
        self._last_switch_step = -1

    # ── Activation ─────────────────────────────────────────────────────

    def should_activate(self, ctx: SolverContext) -> bool:
        gm = ctx.game_model
        if gm.player_pos is None or not gm.goal_positions:
            return False
        mirror_info = _find_mirror_info(gm.components, gm.width, gm.height)
        if mirror_info is None:
            return False
        shapes = _get_shape_pixels(gm.components, gm.width, gm.height)
        if not shapes:
            return False
        goals = get_goal_cells(gm.components, gm.width)
        if not goals:
            return False
        primary = gm.features.primary_dynamics
        if primary == "beam":
            return False
        # Build a board signature to detect level transitions
        # Use stable features that don't change within a level.
        sig = (mirror_info["horizontal"], mirror_info.get("axis_py"), mirror_info.get("axis_px"),
               len(gm.goal_positions), len(shapes))
        # Reset on any change in board configuration (new level)
        if sig != self._board_sig:
            self.reset_level()
            self._board_sig = sig
        if mirror_info["horizontal"]:
            return True
        return len(shapes) >= 2

    # ── Decision ───────────────────────────────────────────────────────

    def decide(self, ctx: SolverContext) -> ActionDecision | None:
        gm = ctx.game_model
        if gm.player_pos is None:
            return None

        mirror_info = _find_mirror_info(gm.components, gm.width, gm.height)
        if mirror_info is None:
            return None
        shapes = _get_shape_pixels(gm.components, gm.width, gm.height)
        if not shapes:
            return None
        goals = get_goal_cells(gm.components, gm.width)
        if not goals:
            return None

        px, py = gm.player_pos
        gs = max(1, gm.features.grid_scale) if hasattr(gm.features, 'grid_scale') else 3

        # Reset on level transition (MUST be before cache check)
        try:
            current_level = int(ctx.plan_context.get("levels_completed", 0))
        except (TypeError, ValueError):
            current_level = 0
        if self._last_levels_completed is None or current_level > self._last_levels_completed:
            self.reset_level()
        self._last_levels_completed = current_level

        # Collect forbidden cells: mirror strip + border-touching obstacles
        forbidden: set[tuple[int, int]] = set()
        for c in gm.components:
            if c.touches_border and c.dominant_color not in (5, 11):
                forbidden.update(c.cells)
        # Add a thin buffer around the mirror axis to prevent shape overlap.
        if mirror_info["horizontal"]:
            my = mirror_info["axis_py"]
            for x in range(gm.width):
                for dy in range(-1, 2):
                    forbidden.add((x, my + dy))
        else:
            mx = mirror_info["axis_px"]
            for y in range(gm.height):
                for dx in range(-1, 2):
                    forbidden.add((mx + dx, y))

        # Cache targets — recompute on level change or retry
        if self._cached_targets is None or self._cached_frame != current_level:
            self._cached_targets = compute_coverage_for_all_shapes(
                shapes, goals, mirror_info, gm.width, gm.height,
                forbidden_cells=forbidden, gs=gs,
            )
            self._cached_frame = current_level

        if not self._cached_targets:
            return None

        # Apply retry offset to nudge shapes to slightly different positions
        if self._retry_count > 0:
            shift = (self._retry_count % 3) * gs  # shift by 0, 3, or 6 pixels
            for stage in self._cached_targets:
                anc = stage["target_anchor"]
                anc[0] = max(0, min(gm.width - 1, anc[0] + shift))
                anc[1] = max(0, min(gm.height - 1, anc[1] + shift))

        player_near_mirror = False
        if mirror_info["horizontal"]:
            player_near_mirror = abs(py - mirror_info["axis_py"]) <= 3
        else:
            player_near_mirror = abs(px - mirror_info["axis_px"]) <= 3

        active_target_idx = min(self._switch_count, len(self._cached_targets) - 1)
        stage = self._cached_targets[active_target_idx]
        target_cx, target_cy = stage["target_anchor"]
        # Grid-align target: snap to nearest grid cell's pixel center
        target_cx = (target_cx // gs) * gs + (gs // 2)
        target_cy = (target_cy // gs) * gs + (gs // 2)

        if self._last_player_pos == (px, py):
            self._shape_stuck_steps += 1
        else:
            self._shape_stuck_steps = 0
        self._last_player_pos = (px, py)

        # On first activation, do the initial switch to the first shape piece.
        if self._switch_count == 0 and not self._switched:
            self._switched = True
            if len(shapes) > 0:
                return ActionDecision("ACTION5", "solver:beam-coverage", 0.70,
                                      reason="coverage_switch_to_shape")
            return None

        if self._shape_stuck_steps >= 30 and self._switch_count < len(shapes):
            self._switch_count += 1
            self._shape_stuck_steps = 0
            if self._switch_count < len(shapes):
                return ActionDecision("ACTION5", "solver:beam-coverage", 0.65,
                                      reason="coverage_stuck_switch")
            return None

        # Convert to grid coordinates for navigation (game engine moves in grid steps)
        gx, gy = px // gs, py // gs
        target_gx = target_cx // gs
        target_gy = target_cy // gs
        g_w = gm.width // gs
        g_h = gm.height // gs
        arrival_dist = abs(gx - target_gx) + abs(gy - target_gy)
        # Also compute pixel distance for comparison
        pixel_dist = abs(px - target_cx) + abs(py - target_cy)
        final_stage = self._switch_count >= len(shapes)

        # Track how long both shapes have been placed
        if self._switch_count >= len(shapes):
            self._both_placed_steps += 1
        else:
            self._both_placed_steps = 0

        # Emergency switch if close enough in pixel or grid space.
        # Skip when both shapes have been placed for >5 frames (to allow retry).
        if self._both_placed_steps <= 5 and (arrival_dist <= 1 or pixel_dist <= gs * 2):
            if self._switch_count < len(shapes):
                self._switch_count += 1
                self._shape_stuck_steps = 0
                if self._switch_count < len(shapes):
                    return ActionDecision("ACTION5", "solver:beam-coverage", 0.68,
                                          reason="coverage_switch_next")
                return ActionDecision("ACTION6", "solver:beam-coverage", 0.60,
                                      reason="coverage_all_placed")
            return ActionDecision("ACTION6", "solver:beam-coverage", 0.60,
                                  reason="coverage_wait")

        # Emergency switch after too many stuck steps or both placed for a while
        if self._shape_stuck_steps >= 8 or self._both_placed_steps >= 8:
            if self._switch_count < len(shapes):
                self._switch_count += 1
                self._shape_stuck_steps = 0
                if self._switch_count < len(shapes):
                    return ActionDecision("ACTION5", "solver:beam-coverage", 0.65,
                                          reason="coverage_early_switch")
                return None
            # Both shapes placed but level not complete — invalidate cache
            # to try different targets.  Keep _switch_count as-is and navigate
            # the current entity to the recomputed target.
            self._shape_stuck_steps = 0
            self._both_placed_steps = 0
            self._cached_targets = None
            self._retry_count += 1
            # Don't change _switch_count — retry with current entity
            return None  # fall through to let frame advance

        # Direct axis-priority navigation: prioritize the axis with larger delta
        # to avoid A* oscillation between directions.
        dx = target_gx - gx
        dy = target_gy - gy
        candidates = []
        if abs(dx) >= abs(dy):
            if dx > 0: candidates.append("right")
            elif dx < 0: candidates.append("left")
            if dy > 0: candidates.append("down")
            elif dy < 0: candidates.append("up")
        else:
            if dy > 0: candidates.append("down")
            elif dy < 0: candidates.append("up")
            if dx > 0: candidates.append("right")
            elif dx < 0: candidates.append("left")
        vec = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
        expanded = self._build_expanded_walkable(gm)
        grid_walkable = set()
        for wx, wy in expanded:
            grid_walkable.add((wx // gs, wy // gs))
        failed = ctx.plan_context.get("failed_directions", set())
        for d in candidates:
            if d in failed:
                continue
            dvx, dvy = vec[d]
            ngx, ngy = gx + dvx, gy + dvy
            if (ngx, ngy) in grid_walkable:
                return ActionDecision(
                    DIR_TO_ACTION.get(d, "ACTION1"), "solver:beam-coverage", 0.65,
                    reason=f"coverage_nav_{d}",
                    metadata={"target_anchor": [target_cx, target_cy],
                              "shape_stage": self._switch_count},
                )
        # Forced fallback
        for d, (dvx, dvy) in vec.items():
            ngx, ngy = gx + dvx, gy + dvy
            if 0 <= ngx < g_w and 0 <= ngy < g_h:
                return ActionDecision(
                    DIR_TO_ACTION.get(d, "ACTION1"), "solver:beam-coverage", 0.40,
                    reason=f"coverage_forced_{d}",
                    metadata={"target_anchor": [target_cx, target_cy],
                              "shape_stage": self._switch_count},
                )
        return None

    def _build_expanded_walkable(self, gm) -> set[tuple[int, int]]:
        # In beam-coverage puzzles, shape pieces can move across all
        # board cells (the game engine's sprite-based collision model
        # allows traversal through most terrain).  Use the full board.
        w = set()
        for y in range(gm.height):
            for x in range(gm.width):
                w.add((x, y))
        # Exclude only border-touching walls (immovable obstacles)
        for c in gm.components:
            if c.touches_border and c.dominant_color != 11:
                for cx, cy in c.cells:
                    w.discard((cx, cy))
        return w


register_solver("beam-coverage", BeamCoverageSolver)
