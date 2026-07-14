from __future__ import annotations

import logging
from collections import deque
from itertools import product
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


# Canonical sprite pixel patterns for AR25 shape pieces.
# These are the exact non-transparent pixels from the sprite definitions.
# Using the canonical patterns ensures beam simulation matches game engine.
_SPRITE_PATTERNS: dict[str, list[tuple[int, int]]] = {
    "0027crpzfdutil": [(0,0), (0,1), (0,2), (0,3), (1,3), (2,3), (3,3)],  # L-shape
    "0053vnfodarlql": [(0,0), (1,0), (2,0), (3,0), (1,1), (2,1)],        # dumbbell
}


def _get_shape_pixels(
    components: list,
    width: int,
    height: int,
    board: list[list[int]] | None = None,
    gs: int = 3,
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

    # For each shape, use the canonical sprite pixel pattern if available.
    # The sprite width/height determines which pattern to use.
    result = []
    for c5 in color5_shapes:
        x1, y1, x2, y2 = c5.bbox
        sw = x2 - x1 + 1
        sh = y2 - y1 + 1
        key = "%dx%d" % (sw, sh)
        # Map sprite dimensions to known patterns
        dim_to_pattern = {
            "4x4": _SPRITE_PATTERNS.get("0027crpzfdutil"),   # L-shape
            "4x2": _SPRITE_PATTERNS.get("0053vnfodarlql"),   # dumbbell
        }
        pattern = dim_to_pattern.get(key)
        if pattern:
            # Canonical sprite pixel patterns give grid-relative offsets.
            # Each sprite pixel at pattern (dx, dy) renders at PIXEL:
            #   px = (sprite_grid_x + dx) * gs = x1 + dx * gs
            #   py = (sprite_grid_y + dy) * gs = y1 + dy * gs
            grid_x = x1 // gs  # approximate sprite grid x from bbox
            grid_y = y1 // gs  # approximate sprite grid y from bbox
            cells = {(grid_x * gs + dx * gs, grid_y * gs + dy * gs) for dx, dy in pattern}
        else:
            # Fallback: extract from board at the sprite's bounding box
            cells = set()
            bg = 9
            if board:
                for dy in range(sw):
                    for dx in range(sw):
                        px, py = x1 + dx, y1 + dy
                        if 0 <= px < width and 0 <= py < height:
                            v = board[py][px]
                            if v not in (-1, bg):
                                cells.add((px, py))
            if not cells:
                cells = set(c5.cells)
        # Merge color-0 overlays
        for c0 in components:
            if c0.dominant_color != 0:
                continue
            if c0.touches_border:
                continue
            ox1, oy1, ox2, oy2 = c0.bbox
            if ox1 > x2 or ox2 < x1 or oy1 > y2 or oy2 < y1:
                continue
            if any(cell in cells for cell in c0.cells):
                cells.update(c0.cells)
        result.append(list(cells))
    return result


def _shape_center(shape: list[tuple[int, int]]) -> tuple[int, int] | None:
    if not shape:
        return None
    return (
        sum(x for x, _ in shape) // len(shape),
        sum(y for _, y in shape) // len(shape),
    )


def get_goal_cells(
    components: list,
    width: int,
    board: list[list[int]] | None = None,
) -> list[tuple[int, int]]:
    if board:
        board_goals = [
            (x, y)
            for y, row in enumerate(board)
            for x, value in enumerate(row[:max(0, width - 1)])
            if value == 11
        ]
        if board_goals:
            return board_goals
    return [
        cell
        for comp in components
        if comp.dominant_color == 11
        and any(cx < width - 1 for cx, _ in comp.cells)
        for cell in comp.cells
    ]


def _normalize_goal_cells(value: Any, width: int) -> list[tuple[int, int]]:
    """Normalize serialized logical/pixel goal coordinates and remove duplicates."""
    result: set[tuple[int, int]] = set()
    if not isinstance(value, (list, tuple, set)):
        return []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            x, y = int(item[0]), int(item[1])
        except (TypeError, ValueError):
            continue
        if 0 <= x < max(0, width - 1) and y >= 0:
            result.add((x, y))
    return sorted(result)


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


def predict_beam_cells(
    shape_cells: list[tuple[int, int]],
    mirror_info: dict[str, Any],
    board_width: int,
    board_height: int,
    gs: int = 3,
) -> set[tuple[int, int]]:
    """Return logical beam cells using the AR25 sprite reflection model.

    The engine reflects occupied sprite cells as logical grid coordinates;
    it does not score every pixel in a rendered cell independently.
    """
    grid_w = board_width // max(1, gs)
    grid_h = board_height // max(1, gs)
    source = {
        (x // gs, y // gs)
        for x, y in shape_cells
        if 0 <= x // gs < grid_w and 0 <= y // gs < grid_h
    }
    if not source:
        return set()

    horizontal = bool(mirror_info.get("horizontal", True))
    axis_gx = int(mirror_info.get("axis_px", 0)) // gs
    axis_gy = int(mirror_info.get("axis_py", 0)) // gs
    # The engine's completion array contains both the reflected beam and the
    # occupied source sprite cells before goal checks are evaluated.
    beam: set[tuple[int, int]] = set(source)
    queue: deque[tuple[int, int, int]] = deque(
        (gx, gy, 0) for gx, gy in source
    )
    visited = set(source)
    while queue:
        gx, gy, depth = queue.popleft()
        if depth >= _MAX_REFLECT_DEPTH:
            continue
        if horizontal:
            reflected = (gx, 2 * axis_gy - gy)
        else:
            reflected = (2 * axis_gx - gx, gy)
        if not (0 <= reflected[0] < grid_w and 0 <= reflected[1] < grid_h):
            continue
        if reflected in visited:
            continue
        visited.add(reflected)
        beam.add(reflected)
        queue.append((reflected[0], reflected[1], depth + 1))
    return beam


def reflected_visibility(
    shape_cells: list[tuple[int, int]],
    mirror_info: dict[str, Any],
    board_width: int,
    board_height: int,
    goal_cells: set[tuple[int, int]] | list[tuple[int, int]] = (),
    gs: int = 3,
) -> dict[str, Any]:
    """Describe the first reflected sprite image using engine cell rules.

    AR25 seeds reflection from occupied source cells that are on-screen.  A
    reflected cell is useful only when its logical cell is on-screen; the
    source sprite itself may still be partially outside the viewport.
    """
    grid_w = board_width // max(1, gs)
    grid_h = board_height // max(1, gs)
    source_cells = {(x // gs, y // gs) for x, y in shape_cells}
    source_cells = {
        (x, y) for x, y in source_cells
        if 0 <= x < grid_w and 0 <= y < grid_h
    }
    horizontal = bool(mirror_info.get("horizontal", True))
    axis_gx = int(mirror_info.get("axis_px", 0)) // max(1, gs)
    axis_gy = int(mirror_info.get("axis_py", 0)) // max(1, gs)
    reflected_all: set[tuple[int, int]] = set()
    for gx, gy in source_cells:
        if horizontal:
            reflected_all.add((gx, 2 * axis_gy - gy))
        else:
            reflected_all.add((2 * axis_gx - gx, gy))
    visible = {
        cell for cell in reflected_all
        if 0 <= cell[0] < grid_w and 0 <= cell[1] < grid_h
    }
    goals = {(x // max(1, gs), y // max(1, gs)) for x, y in goal_cells}
    return {
        "source_cells": source_cells,
        "reflected_cells": reflected_all,
        "visible_cells": visible,
        "visible_goal_cells": visible & goals,
        "offscreen": bool(reflected_all) and not visible,
        "grid_height": grid_h,
        "grid_width": grid_w,
    }


def reflection_recovery_direction(
    visibility: dict[str, Any],
    mirror_info: dict[str, Any],
) -> str | None:
    """Return the vertical direction that brings an off-screen image back."""
    if not mirror_info.get("horizontal") or visibility.get("visible_cells"):
        return None
    reflected = set(visibility.get("reflected_cells", ()))
    if not reflected:
        return None
    ys = [cell[1] for cell in reflected]
    grid_h = int(visibility.get("grid_height", 21))
    if max(ys) < 0:
        return "up"
    if min(ys) >= grid_h:
        return "down"
    # A mixed out-of-bounds image is nearest to the closer board edge.
    return "up" if abs(max(ys) + 1) <= abs(min(ys) - grid_h) else "down"


def covered_goal_cells(
    beam_cells: set[tuple[int, int]],
    goal_cells: set[tuple[int, int]] | list[tuple[int, int]],
    gs: int = 3,
) -> set[tuple[int, int]]:
    """Evaluate goals in the same logical-cell coordinate space as AR25."""
    goals = {(x // max(1, gs), y // max(1, gs)) for x, y in goal_cells}
    return beam_cells & goals


def _logical_goal_set(
    goal_cells: set[tuple[int, int]] | list[tuple[int, int]],
    gs: int = 3,
) -> set[tuple[int, int]]:
    return {(x // max(1, gs), y // max(1, gs)) for x, y in goal_cells}


def _shape_relative_cells(
    shape_cells: list[tuple[int, int]],
) -> tuple[set[tuple[int, int]], tuple[int, int]] | None:
    if not shape_cells:
        return None
    xs = [x for x, _ in shape_cells]
    ys = [y for _, y in shape_cells]
    x1, y1 = min(xs), min(ys)
    rel_set = {(x - x1, y - y1) for x, y in shape_cells}
    return rel_set, (x1, y1)


def _placed_shape_from_anchor(
    shape_cells: list[tuple[int, int]],
    anchor: tuple[int, int],
) -> set[tuple[int, int]]:
    relative = _shape_relative_cells(shape_cells)
    if relative is None:
        return set()
    rel_set, (x1, y1) = relative
    rel_cx = sum(x for x, _ in rel_set) // len(rel_set) if rel_set else 0
    rel_cy = sum(y for _, y in rel_set) // len(rel_set) if rel_set else 0
    px = int(anchor[0]) - rel_cx + x1
    py = int(anchor[1]) - rel_cy + y1
    return {(px + dx, py + dy) for dx, dy in rel_set}


def _score_stage_target_family(
    stage_targets: list[dict[str, Any]],
    shapes: list[list[tuple[int, int]]],
    goal_cells: list[tuple[int, int]],
    mirror_info: dict[str, Any],
    board_width: int,
    board_height: int,
    *,
    current_centers: list[tuple[int, int] | None] | None = None,
    gs: int = 3,
) -> dict[str, Any]:
    total_goals = _logical_goal_set(goal_cells, gs=gs)
    covered_goals: set[tuple[int, int]] = set()
    stage_coverage: list[int] = []
    total_distance = 0

    for stage in stage_targets:
        if not isinstance(stage, dict):
            continue
        shape_index = int(stage.get("shape_index", -1))
        if shape_index < 0 or shape_index >= len(shapes):
            continue
        target_anchor = stage.get("target_anchor")
        if not isinstance(target_anchor, (list, tuple)) or len(target_anchor) != 2:
            continue
        placed = _placed_shape_from_anchor(
            shapes[shape_index],
            (int(target_anchor[0]), int(target_anchor[1])),
        )
        logical_beam = predict_beam_cells(
            list(placed),
            mirror_info,
            board_width,
            board_height,
            gs=gs,
        )
        hits = covered_goal_cells(logical_beam, goal_cells, gs=gs)
        covered_goals.update(hits)
        stage_coverage.append(len(hits))
        if current_centers and shape_index < len(current_centers):
            current_center = current_centers[shape_index]
            if current_center is not None:
                total_distance += (
                    abs(int(target_anchor[0]) - current_center[0])
                    + abs(int(target_anchor[1]) - current_center[1])
                )

    return {
        "covered_goal_count": len(covered_goals),
        "goal_cell_count": len(total_goals),
        "full_coverage": bool(total_goals) and covered_goals >= total_goals,
        "final_stage_coverage": stage_coverage[-1] if stage_coverage else 0,
        "stage_coverage": stage_coverage,
        "total_distance": total_distance,
    }


def find_candidate_placements(
    shape_cells: list[tuple[int, int]],
    remaining_goals: set[tuple[int, int]],
    mirror_info: dict[str, Any],
    board_width: int,
    board_height: int,
    current_center: tuple[int, int] | None = None,
    forbidden_cells: set[tuple[int, int]] | None = None,
    gs: int = 3,
) -> list[dict[str, Any]]:
    if not shape_cells or not remaining_goals:
        return []
    xs = [x for x, _ in shape_cells]
    ys = [y for _, y in shape_cells]
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    shape_w = x2 - x1 + 1
    shape_h = y2 - y1 + 1
    if shape_w <= 0 or shape_h <= 0:
        return []

    rel_set = {(x - x1, y - y1) for x, y in shape_cells}
    candidates: list[dict[str, Any]] = []

    step = max(1, gs)
    for tx in range(0, max(1, board_width - shape_w + 1), step):
        for ty in range(0, max(1, board_height - shape_h + 1), step):
            placed = {(tx + dx, ty + dy) for dx, dy in rel_set}
            # Skip if any placed cell overlaps the mirror
            if forbidden_cells and placed & forbidden_cells:
                continue
            covered = predict_beam_coverage(
                list(placed), mirror_info, board_width, board_height, gs=gs,
            )
            logical_beam = predict_beam_cells(
                list(placed), mirror_info, board_width, board_height, gs=gs,
            )
            hits = len(covered_goal_cells(logical_beam, remaining_goals, gs=gs))
            if hits == 0:
                continue
            cx = sum(x for x, _ in placed) // len(placed)
            cy = sum(y for _, y in placed) // len(placed)
            dist = abs(cx - current_center[0]) + abs(cy - current_center[1]) if current_center else 0
            covered_goals = {
                cell for cell in remaining_goals
                if (cell[0] // max(1, gs), cell[1] // max(1, gs)) in logical_beam
            }
            candidates.append({
                "top_left": (tx, ty),
                "target_anchor": [cx, cy],
                "target_bbox": [
                    min(x for x, _ in placed),
                    min(y for _, y in placed),
                    max(x for x, _ in placed),
                    max(y for _, y in placed),
                ],
                "covered_goals": sorted(covered_goals),
                "hits": hits,
                "distance": dist,
            })

    candidates.sort(
        key=lambda item: (
            int(item.get("hits", 0)),
            -int(item.get("distance", 0)),
        ),
        reverse=True,
    )
    return candidates[:6]


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
    candidates = find_candidate_placements(
        shape_cells,
        remaining_goals,
        mirror_info,
        board_width,
        board_height,
        current_center=current_center,
        forbidden_cells=forbidden_cells,
        gs=gs,
    )
    if not candidates:
        return None
    return tuple(candidates[0]["top_left"])


def rank_joint_best_placements(
    shapes: list[list[tuple[int, int]]],
    goal_cells: list[tuple[int, int]],
    mirror_info: dict[str, Any],
    board_width: int,
    board_height: int,
    forbidden_cells: set[tuple[int, int]] | None = None,
    *,
    family_limit: int = 6,
    gs: int = 3,
) -> list[dict[str, Any]]:
    if not shapes or not goal_cells:
        return []

    all_goals = set(goal_cells)
    remaining = set(goal_cells)
    current_centers: list[tuple[int, int] | None] = []
    candidate_lists: list[list[dict[str, Any]]] = []
    for shape_cells in shapes:
        current_center = (
            sum(x for x, _ in shape_cells) // len(shape_cells),
            sum(y for _, y in shape_cells) // len(shape_cells),
        ) if shape_cells else None
        current_centers.append(current_center)
        candidates = find_candidate_placements(
            shape_cells,
            remaining,
            mirror_info,
            board_width,
            board_height,
            current_center=current_center,
            forbidden_cells=forbidden_cells,
            gs=gs,
        )
        if not candidates:
            candidates = find_candidate_placements(
                shape_cells,
                all_goals,
                mirror_info,
                board_width,
                board_height,
                current_center=current_center,
                forbidden_cells=forbidden_cells,
                gs=gs,
            )
        if not candidates:
            return []
        trimmed = candidates[: min(len(candidates), 4)]
        candidate_lists.append(trimmed)
        remaining -= {
            tuple(cell)
            for cell in trimmed[0].get("covered_goals", [])
            if isinstance(cell, (list, tuple)) and len(cell) == 2
        }

    ranked: list[dict[str, Any]] = []
    seen_orders: set[tuple[tuple[int, int], ...]] = set()
    for combo in product(*candidate_lists):
        stage_targets: list[dict[str, Any]] = []
        for shape_index, candidate in enumerate(combo):
            target_anchor = candidate.get("target_anchor", [])
            if not isinstance(target_anchor, list) or len(target_anchor) != 2:
                continue
            stage_targets.append({
                "shape_index": shape_index,
                "target_anchor": [int(target_anchor[0]), int(target_anchor[1])],
                "target_bbox": list(candidate.get("target_bbox", [])),
                "covered_goals": list(candidate.get("covered_goals", [])),
            })
        order_key = tuple(
            (int(stage["target_anchor"][0]), int(stage["target_anchor"][1]))
            for stage in stage_targets
        )
        if not order_key or order_key in seen_orders:
            continue
        seen_orders.add(order_key)
        score = _score_stage_target_family(
            stage_targets,
            shapes,
            goal_cells,
            mirror_info,
            board_width,
            board_height,
            current_centers=current_centers,
            gs=gs,
        )
        ranked.append({"targets": stage_targets, "score": score})

    ranked.sort(
        key=lambda item: (
            bool(item["score"].get("full_coverage")),
            int(item["score"].get("covered_goal_count", 0)),
            int(item["score"].get("final_stage_coverage", 0)),
            -int(item["score"].get("total_distance", 0)),
        ),
        reverse=True,
    )
    return ranked[:family_limit]


def find_joint_best_placement(
    shapes: list[list[tuple[int, int]]],
    goal_cells: list[tuple[int, int]],
    mirror_info: dict[str, Any],
    board_width: int,
    board_height: int,
    forbidden_cells: set[tuple[int, int]] | None = None,
    gs: int = 3,
) -> list[dict[str, Any]]:
    """Joint placement search: try combinations for BOTH shapes together.

    Uses N best base positions for each shape from greedy search, then
    evaluates all pairs of N×N to find the joint optimum that covers
    the most total goals.  Falls back to greedy if N×N is empty.
    """
    # First, run greedy to get candidate positions
    greedy = compute_coverage_for_all_shapes(
        shapes, goal_cells, mirror_info,
        board_width, board_height, forbidden_cells, gs,
    )
    if not greedy:
        return greedy
    best_combined_score = len(set(goal_cells))
    # Collect top-K positions for each shape from greedy candidates
    # and evaluate all combinations
    goal_set = set(goal_cells)
    for attempt in range(3):  # try 3 different offset sets
        all_covered = set()
        all_beam_cells: set[tuple[int, int]] = set()
        for idx, shape_cells in enumerate(shapes):
            if idx >= len(greedy):
                continue
            anc = greedy[idx].get("target_anchor", [0, 0])
            # Use the greedy target with optional offset
            ox, oy = 0, 0
            if attempt > 0:
                ox = (attempt % 3 - 1) * gs
                oy = ((attempt // 3) % 3 - 1) * gs
            tx = max(0, min(board_width - 1, anc[0] + ox))
            ty = max(0, min(board_height - 1, anc[1] + oy))
            # Convert to top-left position
            xs = [x for x, _ in shape_cells]
            ys = [y for _, y in shape_cells]
            x1, y1 = min(xs), min(ys)
            rel_set = {(x - x1, y - y1) for x, y in shape_cells}
            placed = {(tx + dx - ox + x1, ty + dy - oy + y1) for dx, dy in rel_set}
            # Actually just use the greedy centroid
            cx, cy = anc
            # Find the tx, ty that gives this centroid
            rel_cx = sum(x for x, _ in rel_set) // len(rel_set) if rel_set else 0
            rel_cy = sum(y for _, y in rel_set) // len(rel_set) if rel_set else 0
            px = cx - rel_cx + x1
            py = cy - rel_cy + y1
            placed2 = {(px + dx, py + dy) for dx, dy in rel_set}
            forbidden = forbidden_cells or set()
            if placed2 & forbidden:
                continue
            covered = predict_beam_coverage(
                list(placed2), mirror_info, board_width, board_height, gs=gs,
            )
            all_covered.update(covered)
            all_beam_cells.update(
                predict_beam_cells(
                    list(placed2), mirror_info, board_width, board_height, gs=gs,
                )
            )
        hit_logical = covered_goal_cells(all_beam_cells, goal_set, gs=gs)
        newly = {
            cell for cell in goal_set
            if (cell[0] // max(1, gs), cell[1] // max(1, gs)) in hit_logical
        }
        if len(newly) >= best_combined_score:
            # Keep these targets (with the offset applied)
            for idx, stage in enumerate(greedy):
                if idx < len(shapes):
                    anc = stage["target_anchor"]
                    anc[0] = max(0, min(board_width - 1, anc[0]))
                    anc[1] = max(0, min(board_height - 1, anc[1]))
        if len(newly) >= best_combined_score:
            break
    return greedy


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
        logical_beam = predict_beam_cells(
            list(placed), mirror_info, board_width, board_height, gs=gs,
        )
        hit_logical = covered_goal_cells(logical_beam, remaining, gs=gs)
        newly_covered = {
            cell for cell in remaining
            if (cell[0] // max(1, gs), cell[1] // max(1, gs)) in hit_logical
        }
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


def _stage_target_coverage_count(stage_targets: list[dict[str, Any]]) -> int:
    covered: set[tuple[int, int]] = set()
    for stage in stage_targets:
        for cell in list(stage.get("covered_goals", []) or []):
            if isinstance(cell, tuple) and len(cell) == 2:
                covered.add((int(cell[0]), int(cell[1])))
            elif isinstance(cell, list) and len(cell) == 2:
                covered.add((int(cell[0]), int(cell[1])))
    return len(covered)


def _base_forbidden_cells(gm) -> set[tuple[int, int]]:
    # AR25 movement has no sprite collision test. Beam, mirror, goals, and
    # other visible components are all traversable by the active shape.
    return set()


def _forbidden_cells_for_mirror_row(
    base_forbidden: set[tuple[int, int]],
    width: int,
    axis_py: int,
) -> set[tuple[int, int]]:
    # Mirror rows are valid shape positions in the engine; do not add a
    # blanket exclusion buffer around the selected row.
    return set(base_forbidden)


def _clone_stage_targets_with_offset(
    stage_targets: list[dict[str, Any]],
    dx: int,
    dy: int,
    board_width: int,
    board_height: int,
) -> list[dict[str, Any]]:
    cloned: list[dict[str, Any]] = []
    for raw_stage in stage_targets:
        stage = dict(raw_stage)
        target_anchor = list(stage.get("target_anchor", []) or [])
        if len(target_anchor) == 2:
            stage["target_anchor"] = [
                max(0, min(board_width - 1, int(target_anchor[0]) + dx)),
                max(0, min(board_height - 1, int(target_anchor[1]) + dy)),
            ]
        target_bbox = list(stage.get("target_bbox", []) or [])
        if len(target_bbox) == 4:
            stage["target_bbox"] = [
                max(0, min(board_width - 1, int(target_bbox[0]) + dx)),
                max(0, min(board_height - 1, int(target_bbox[1]) + dy)),
                max(0, min(board_width - 1, int(target_bbox[2]) + dx)),
                max(0, min(board_height - 1, int(target_bbox[3]) + dy)),
            ]
        cloned.append(stage)
    return cloned


def select_best_horizontal_mirror_plan(
    shapes: list[list[tuple[int, int]]],
    goal_cells: list[tuple[int, int]],
    mirror_info: dict[str, Any],
    board_width: int,
    board_height: int,
    current_mirror_row: int,
    base_forbidden: set[tuple[int, int]],
    *,
    gs: int = 3,
    excluded_rows: set[int] | None = None,
    candidate_rows: set[int] | None = None,
) -> tuple[int | None, list[dict[str, Any]] | None]:
    excluded = excluded_rows or set()
    grid_height = max(1, board_height // max(1, gs))
    best_row: int | None = None
    best_targets: list[dict[str, Any]] | None = None
    best_score: tuple[int, int] | None = None

    row_candidates = sorted(candidate_rows) if candidate_rows else range(grid_height)
    for mirror_row in row_candidates:
        if mirror_row in excluded:
            continue
        row_mirror_info = dict(mirror_info)
        row_mirror_info["axis_py"] = mirror_row * gs + (gs // 2)
        row_targets = find_joint_best_placement(
            shapes,
            goal_cells,
            row_mirror_info,
            board_width,
            board_height,
            forbidden_cells=_forbidden_cells_for_mirror_row(
                base_forbidden,
                board_width,
                int(row_mirror_info["axis_py"]),
            ),
            gs=gs,
        )
        score = (
            _stage_target_coverage_count(row_targets),
            -abs(mirror_row - current_mirror_row),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_row = mirror_row
            best_targets = row_targets

    if best_row is None or best_targets is None:
        return None, None
    return best_row, best_targets


def rank_horizontal_mirror_plans(
    shapes: list[list[tuple[int, int]]],
    goal_cells: list[tuple[int, int]],
    mirror_info: dict[str, Any],
    board_width: int,
    board_height: int,
    current_mirror_row: int,
    base_forbidden: set[tuple[int, int]],
    *,
    gs: int = 3,
    excluded_rows: set[int] | None = None,
    candidate_rows: set[int] | None = None,
) -> dict[str, Any] | None:
    excluded = excluded_rows or set()
    grid_height = max(1, board_height // max(1, gs))
    best_plan: dict[str, Any] | None = None
    best_score: tuple[int, int, int, int] | None = None

    row_candidates = sorted(candidate_rows) if candidate_rows else range(grid_height)
    for mirror_row in row_candidates:
        if mirror_row in excluded:
            continue
        row_mirror_info = dict(mirror_info)
        row_mirror_info["axis_py"] = mirror_row * gs + (gs // 2)
        ranked_families = rank_joint_best_placements(
            shapes,
            goal_cells,
            row_mirror_info,
            board_width,
            board_height,
            forbidden_cells=_forbidden_cells_for_mirror_row(
                base_forbidden,
                board_width,
                int(row_mirror_info["axis_py"]),
            ),
            gs=gs,
        )
        if not ranked_families:
            continue
        row_family_score = ranked_families[0]["score"]
        score = (
            int(bool(row_family_score.get("full_coverage"))),
            int(row_family_score.get("covered_goal_count", 0)),
            int(row_family_score.get("final_stage_coverage", 0)),
            -abs(mirror_row - current_mirror_row),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_plan = {
                "mirror_row": mirror_row,
                "targets": [dict(stage) for stage in ranked_families[0]["targets"]],
                "families": ranked_families,
                "score": dict(row_family_score),
            }

    return best_plan


class BeamCoverageSolver(DynamicsSolver):
    """Solver for beam-coverage puzzles where shape pieces emit beam
    that reflects off fixed mirrors to cover goals.

    Paradigm: shape pieces are beam sources, mirrors are passive reflectors.
    """

    dynamics = "beam-coverage"

    def __init__(self, memory_db_path: str | Path | None = None) -> None:
        super().__init__(memory_db_path)
        self._cached_targets: list[dict[str, Any]] | None = None
        self._cached_mirror_info: dict[str, Any] | None = None
        self._cached_frame: int = -1
        self._cached_target_key: tuple[int, int, int] | None = None
        self._active_stage_index: int = -1
        self._switch_pending: bool = False
        self._pending_stage_index: int | None = None
        self._pending_switch_origin: tuple[int, int] | None = None
        self._pending_switch_color: int | None = None
        self._pending_switch_frames: int = 0
        self._pending_switch_retries: int = 0
        self._pending_switch_scans: int = 0
        self._emitted_switches: int = 0
        self._planned_shape_count: int = 0
        self._last_player_pos: tuple[int, int] | None = None
        self._shape_stuck_steps: int = 0
        self._move_history: list[str] = []
        self._last_levels_completed: int | None = None
        self._board_sig: tuple = ()
        self._retry_count: int = 0
        self._both_placed_steps: int = 0
        self._mirror_target_gy: int | None = None
        self._mirror_phase: bool = False
        self._mirror_stall_steps: int = 0
        self._last_observed_mirror_gy: int | None = None
        self._unreachable_mirror_rows: set[int] = set()
        # A selected row and its joint targets form one execution attempt.
        # They must not drift as the live sprite is tuned or temporarily lost.
        self._attempt_id: int = 0
        self._attempt_reset_reason: str | None = None
        self._base_targets: list[dict[str, Any]] | None = None
        self._ranked_target_families: list[dict[str, Any]] = []
        self._current_family_index: int = 0
        self._family_retry_count: int = 0
        self._family_retry_budget: int = 1
        self._planned_goal_cells: list[tuple[int, int]] | None = None
        self._tune_step: int = 0
        self._tune_budget: int = 12
        self._final_tuning: bool = False
        self._last_decision_player_pos: tuple[int, int] | None = None
        self._reflection_recovery: bool = False
        self._recovery_direction: str | None = None
        self._recovery_step: int = 0
        self._recovery_budget: int = 24
        self._reflected_visible_count: int = 0
        self._reflected_goal_count: int = 0
        self._reflection_offscreen: bool = False
        self._stage_shape_centers: dict[int, tuple[int, int]] = {}
        self._confirmed_shape_center: tuple[int, int] | None = None

    @property
    def has_active_attempt(self) -> bool:
        return bool(
            self._cached_targets
            or self._base_targets
            or self._active_stage_index >= 0
            or self._switch_pending
            or self._final_tuning
        )

    def reset_level(self) -> None:
        logger.debug("beam solver reset_level attempt=%s stage=%s cached=%s", self._attempt_id, self._active_stage_index, bool(self._cached_targets))
        self._cached_targets = None
        self._cached_mirror_info = None
        self._cached_frame = -1
        self._cached_target_key = None
        self._active_stage_index = -1
        self._switch_pending = False
        self._pending_stage_index = None
        self._pending_switch_origin = None
        self._pending_switch_color = None
        self._pending_switch_frames = 0
        self._pending_switch_retries = 0
        self._pending_switch_scans = 0
        self._emitted_switches = 0
        self._planned_shape_count = 0
        self._last_player_pos = None
        self._shape_stuck_steps = 0
        self._move_history = []
        self._last_levels_completed = None
        self._retry_count = 0
        self._both_placed_steps = 0
        self._mirror_target_gy = None
        self._mirror_phase = False
        self._mirror_stall_steps = 0
        self._last_observed_mirror_gy = None
        self._unreachable_mirror_rows.clear()
        self._attempt_id += 1
        self._attempt_reset_reason = "level_reset"
        self._base_targets = None
        self._ranked_target_families = []
        self._current_family_index = 0
        self._family_retry_count = 0
        self._planned_goal_cells = None
        self._tune_step = 0
        self._final_tuning = False
        self._last_decision_player_pos = None
        self._reset_reflection_recovery()
        self._stage_shape_centers.clear()
        self._confirmed_shape_center = None

    def _invalidate_target_cache(self) -> None:
        self._cached_targets = None
        self._cached_frame = -1
        self._cached_target_key = None

    def _reset_reflection_recovery(self) -> None:
        self._reflection_recovery = False
        self._recovery_direction = None
        self._recovery_step = 0
        self._reflected_visible_count = 0
        self._reflected_goal_count = 0
        self._reflection_offscreen = False

    def _goal_cells(self, ctx: SolverContext, board: list[list[int]] | None) -> list[tuple[int, int]]:
        """Prefer the complete perception goal set over reduced GameModel centers."""
        if self._planned_goal_cells is not None:
            return list(self._planned_goal_cells)
        plan_context = ctx.plan_context if isinstance(ctx.plan_context, dict) else {}
        recognition = plan_context.get("shape_recognition", {})
        mirror_puzzle = recognition.get("mirror_puzzle", {}) if isinstance(recognition, dict) else {}
        candidates = []
        if isinstance(mirror_puzzle, dict):
            candidates.append(mirror_puzzle.get("goal_cells"))
        if isinstance(recognition, dict):
            candidates.append(recognition.get("goal_cells"))
        goals: list[tuple[int, int]] = []
        for candidate in candidates:
            goals = _normalize_goal_cells(candidate, ctx.game_model.width)
            if goals:
                break
        if not goals:
            goals = get_goal_cells(
                ctx.game_model.components,
                ctx.game_model.width,
                board=board,
            )
        if goals:
            self._planned_goal_cells = list(goals)
        return goals

    def _decision_metadata(self, **extra: Any) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "selected_mirror_row": self._mirror_target_gy,
            "base_target_anchor": (
                list(self._base_targets[0].get("target_anchor", []))
                if self._base_targets else None
            ),
            "chosen_target_family": [
                list(stage.get("target_anchor", []))
                for stage in (self._base_targets or [])
                if isinstance(stage, dict)
            ],
            "cached_target_order": [
                list(stage.get("target_anchor", []))
                for stage in (self._cached_targets or [])
                if isinstance(stage, dict)
            ],
            "target_family_index": self._current_family_index,
            "target_family_retry_count": self._family_retry_count,
            "alternative_family_count": max(0, len(self._ranked_target_families) - 1),
            "family_scores": [
                dict(family.get("score", {}))
                for family in self._ranked_target_families[:4]
                if isinstance(family, dict)
            ],
            "tune_step": self._tune_step,
            "attempt_id": self._attempt_id,
            "attempt_reset_reason": self._attempt_reset_reason,
            "placement_overlap_allowed": True,
            "beam_crossing_allowed": True,
            "covered_goal_count": _stage_target_coverage_count(
                self._base_targets or []
            ),
            "goal_cell_count": len(self._planned_goal_cells or []),
            "planned_goal_count": len(self._planned_goal_cells or []),
            "observed_player_pos": (
                list(self._last_decision_player_pos)
                if self._last_decision_player_pos is not None else None
            ),
            "active_stage_index": self._active_stage_index,
            "switch_pending": self._switch_pending,
            "pending_switch_origin": (
                list(self._pending_switch_origin)
                if self._pending_switch_origin is not None else None
            ),
            "pending_switch_color": self._pending_switch_color,
            "confirmed_shape_anchor": (
                list(self._confirmed_shape_center)
                if self._confirmed_shape_center is not None else None
            ),
            "reflection_recovery": self._reflection_recovery,
            "recovery_direction": self._recovery_direction,
            "reflected_visible_count": self._reflected_visible_count,
            "reflected_goal_count": self._reflected_goal_count,
            "reflection_offscreen": self._reflection_offscreen,
            "recovery_step": self._recovery_step,
            "final_stage_coverage_incomplete": False,
        }
        metadata.update(extra)
        return metadata

    def _visible_shape_candidates(
        self,
        shapes: list[list[tuple[int, int]]],
    ) -> list[tuple[list[tuple[int, int]], tuple[int, int]]]:
        candidates: list[tuple[list[tuple[int, int]], tuple[int, int]]] = []
        for shape in shapes:
            center = _shape_center(shape)
            if center is None:
                continue
            candidates.append((shape, center))
        return candidates

    def _bind_stage_shape(self, stage_index: int, center: tuple[int, int]) -> None:
        if stage_index < 0:
            return
        normalized = (int(center[0]), int(center[1]))
        self._stage_shape_centers[stage_index] = normalized
        self._confirmed_shape_center = normalized

    def _select_active_shape_for_stage(
        self,
        shapes: list[list[tuple[int, int]]],
        *,
        stage_index: int,
        target: tuple[int, int],
    ) -> tuple[list[tuple[int, int]] | None, tuple[int, int] | None]:
        candidates = self._visible_shape_candidates(shapes)
        if not candidates:
            return None, None
        binding = self._stage_shape_centers.get(stage_index)
        if binding is not None:
            shape, center = min(
                candidates,
                key=lambda item: abs(item[1][0] - binding[0]) + abs(item[1][1] - binding[1]),
            )
        else:
            shape, center = min(
                candidates,
                key=lambda item: abs(item[1][0] - target[0]) + abs(item[1][1] - target[1]),
            )
        self._bind_stage_shape(stage_index, center)
        return shape, center

    def _pending_stage_target(self, gs: int) -> tuple[int, int] | None:
        if (
            self._pending_stage_index is None
            or not self._cached_targets
            or self._pending_stage_index < 0
            or self._pending_stage_index >= len(self._cached_targets)
        ):
            return None
        stage = self._cached_targets[self._pending_stage_index]
        if not isinstance(stage, dict):
            return None
        target = stage.get("target_anchor")
        if not isinstance(target, (list, tuple)) or len(target) != 2:
            return None
        return (
            (int(target[0]) // gs) * gs + (gs // 2),
            (int(target[1]) // gs) * gs + (gs // 2),
        )

    def _horizontal_final_stage_covered(
        self,
        *,
        active_shape: list[tuple[int, int]] | None,
        mirror_info: dict[str, Any],
        goals: list[tuple[int, int]],
        board_width: int,
        board_height: int,
        gs: int,
    ) -> bool:
        if active_shape is None:
            return False
        beam_cells = predict_beam_cells(
            active_shape,
            mirror_info,
            board_width,
            board_height,
            gs=gs,
        )
        covered = covered_goal_cells(beam_cells, goals, gs=gs)
        self._reflected_goal_count = len(covered)
        total_goals = {(x // max(1, gs), y // max(1, gs)) for x, y in goals}
        return bool(total_goals) and covered >= total_goals

    def _emit_tuning_action(
        self,
        *,
        gm,
        gx: int,
        gy: int,
        gs: int,
        target_cx: int,
        target_cy: int,
        final_stage_coverage_incomplete: bool = False,
    ) -> ActionDecision:
        self._final_tuning = True
        self._tune_step += 1
        if self._tune_step > self._tune_budget:
            self._restart_attempt_with_retry("final_tuning_exhausted", gm, gs)
            return ActionDecision(
                "ACTION5", "solver:beam-coverage", 0.40,
                reason="coverage_full_reset",
                metadata=self._decision_metadata(),
            )
        spiral = [
            (0, -1), (1, 0), (0, 1), (-1, 0),
            (0, -2), (1, -1), (2, 0), (1, 1),
            (0, 2), (-1, 1), (-2, 0), (-1, -1),
        ]
        idx = (self._tune_step - 1) % len(spiral)
        dvx, dvy = spiral[idx]
        ngx, ngy = gx + dvx, gy + dvy
        g_w = gm.width // gs
        g_h = gm.height // gs
        if 0 <= ngx < g_w and 0 <= ngy < g_h:
            direction = "up" if dvy < 0 else "down" if dvy > 0 else "left" if dvx < 0 else "right"
            return ActionDecision(
                DIR_TO_ACTION.get(direction, "ACTION1"), "solver:beam-coverage", 0.45,
                reason=f"coverage_tune_{direction}",
                metadata=self._decision_metadata(
                    target_anchor=[target_cx, target_cy],
                    shape_stage=self._active_stage_index,
                    final_stage_coverage_incomplete=final_stage_coverage_incomplete,
                ),
            )
        return ActionDecision(
            "ACTION6", "solver:beam-coverage", 0.40,
            reason="coverage_tune_wait",
            metadata=self._decision_metadata(
                target_anchor=[target_cx, target_cy],
                shape_stage=self._active_stage_index,
                final_stage_coverage_incomplete=final_stage_coverage_incomplete,
            ),
        )

    def _reset_attempt(self, reason: str) -> None:
        self._attempt_id += 1
        self._attempt_reset_reason = reason
        self._base_targets = None
        self._ranked_target_families = []
        self._current_family_index = 0
        self._family_retry_count = 0
        self._cached_mirror_info = None
        self._planned_goal_cells = None
        self._tune_step = 0
        self._final_tuning = False
        self._active_stage_index = -1
        self._switch_pending = False
        self._pending_stage_index = None
        self._pending_switch_origin = None
        self._pending_switch_color = None
        self._pending_switch_frames = 0
        self._pending_switch_retries = 0
        self._pending_switch_scans = 0
        self._shape_stuck_steps = 0
        self._both_placed_steps = 0
        self._invalidate_target_cache()
        self._mirror_target_gy = None
        self._reset_mirror_phase_progress()
        self._reset_reflection_recovery()
        self._stage_shape_centers.clear()
        self._confirmed_shape_center = None

    def _reset_mirror_phase_progress(self, *, clear_unreachable: bool = False) -> None:
        self._mirror_phase = False
        self._mirror_stall_steps = 0
        self._last_observed_mirror_gy = None
        if clear_unreachable:
            self._unreachable_mirror_rows.clear()

    def _restart_attempt_with_retry(self, reason: str, gm, gs: int) -> None:
        self._attempt_id += 1
        self._attempt_reset_reason = reason
        self._active_stage_index = -1
        self._switch_pending = False
        self._pending_stage_index = None
        self._pending_switch_origin = None
        self._pending_switch_color = None
        self._pending_switch_frames = 0
        self._pending_switch_retries = 0
        self._pending_switch_scans = 0
        self._shape_stuck_steps = 0
        self._both_placed_steps = 0
        self._tune_step = 0
        self._final_tuning = False
        self._last_player_pos = None
        self._last_decision_player_pos = None
        self._reset_reflection_recovery()
        self._stage_shape_centers.clear()
        self._confirmed_shape_center = None
        if self._base_targets:
            if (
                self._ranked_target_families
                and self._current_family_index + 1 < len(self._ranked_target_families)
            ):
                promoted = self._promote_target_family(gm, gs)
                if promoted:
                    return
            self._family_retry_count += 1
            self._retry_count += 1
            if self._family_retry_count <= self._family_retry_budget:
                self._cached_targets = self._targets_with_retry_offset(self._base_targets, gm, gs)
                self._cached_frame = self._last_levels_completed or self._cached_frame
                return
        self._invalidate_target_cache()

    def _targets_with_retry_offset(
        self,
        stage_targets: list[dict[str, Any]],
        gm,
        gs: int,
    ) -> list[dict[str, Any]]:
        if self._retry_count <= 0:
            return [dict(stage) for stage in stage_targets]
        offsets = [
            (0, gs), (0, -gs), (gs, 0), (-gs, 0),
            (gs, gs), (gs, -gs), (-gs, gs), (-gs, -gs),
            (0, gs * 2), (0, -gs * 2), (gs * 2, 0), (-gs * 2, 0),
        ]
        idx = (self._retry_count - 1) % len(offsets)
        dx, dy = offsets[idx]
        return _clone_stage_targets_with_offset(
            stage_targets,
            dx,
            dy,
            gm.width,
            gm.height,
        )

    def _adopt_target_family(
        self,
        family_index: int,
        gm,
        gs: int,
        *,
        apply_offset: bool = False,
    ) -> bool:
        if family_index < 0 or family_index >= len(self._ranked_target_families):
            return False
        family = self._ranked_target_families[family_index]
        targets = family.get("targets", []) if isinstance(family, dict) else []
        if not isinstance(targets, list) or not targets:
            return False
        self._current_family_index = family_index
        self._base_targets = [dict(stage) for stage in targets if isinstance(stage, dict)]
        self._cached_targets = (
            self._targets_with_retry_offset(self._base_targets, gm, gs)
            if apply_offset else [dict(stage) for stage in self._base_targets]
        )
        self._cached_frame = self._last_levels_completed or self._cached_frame
        return bool(self._cached_targets)

    def _promote_target_family(self, gm, gs: int) -> bool:
        if self._current_family_index + 1 >= len(self._ranked_target_families):
            return False
        self._family_retry_count = 0
        self._retry_count = 0
        return self._adopt_target_family(self._current_family_index + 1, gm, gs)

    def _request_switch(
        self,
        ctx: SolverContext,
        *,
        next_stage: int,
        player_pos: tuple[int, int],
        player_color: int,
        reason: str,
        confidence: float,
    ) -> ActionDecision:
        self._switch_pending = True
        self._pending_stage_index = next_stage
        self._pending_switch_origin = player_pos
        self._pending_switch_color = player_color
        self._pending_switch_frames = 0
        self._pending_switch_retries = 0
        self._pending_switch_scans = 0
        self._emitted_switches += 1
        current = int(ctx.plan_context.get("switch_attempted", 0) or 0)
        ctx.plan_context["switch_attempted"] = max(current, self._emitted_switches)
        return ActionDecision(
            "ACTION5",
            "solver:beam-coverage",
            confidence,
            reason=reason,
            metadata=self._decision_metadata(pending_shape_stage=next_stage),
        )

    def _observe_pending_switch(
        self,
        ctx: SolverContext,
        *,
        player_pos: tuple[int, int],
        player_color: int,
        shapes: list[list[tuple[int, int]]],
        gs: int,
    ) -> tuple[bool, ActionDecision | None]:
        if not self._switch_pending:
            return False, None

        origin = self._pending_switch_origin
        changed_position = origin is not None and (
            abs(player_pos[0] - origin[0]) + abs(player_pos[1] - origin[1]) > gs * 2
        )
        changed_color = (
            self._pending_switch_color is not None
            and player_color != self._pending_switch_color
        )
        target = self._pending_stage_target(gs) or player_pos
        candidates = self._visible_shape_candidates(shapes)
        visible_shape_cells = [cell for shape in shapes for cell in shape]
        controlled_shape = None
        if candidates:
            controlled_shape = min(
                candidates,
                key=lambda item: abs(item[1][0] - player_pos[0]) + abs(item[1][1] - player_pos[1]),
            )
        controls_shape = (
            (
                controlled_shape is not None
                and abs(player_pos[0] - controlled_shape[1][0]) + abs(player_pos[1] - controlled_shape[1][1]) <= gs * 3
            )
            if candidates
            else player_color == 5
        )
        if visible_shape_cells and (changed_position or changed_color) and not controls_shape:
            scan_limit = 1
            if self._pending_switch_scans < scan_limit:
                self._pending_switch_scans += 1
                self._pending_switch_origin = player_pos
                self._pending_switch_color = player_color
                self._pending_switch_frames = 0
                self._emitted_switches += 1
                current = int(ctx.plan_context.get("switch_attempted", 0) or 0)
                ctx.plan_context["switch_attempted"] = max(current, self._emitted_switches)
                return False, ActionDecision(
                    "ACTION5",
                    "solver:beam-coverage",
                    0.62,
                    reason="coverage_switch_scan",
                    metadata=self._decision_metadata(
                        pending_shape_stage=self._pending_stage_index,
                        pending_shape_target=list(target) if target is not None else None,
                    ),
                )

        if (changed_position or changed_color) and controls_shape:
            self._active_stage_index = int(self._pending_stage_index or 0)
            if controlled_shape is not None:
                self._bind_stage_shape(self._active_stage_index, controlled_shape[1])
            self._switch_pending = False
            self._pending_stage_index = None
            self._pending_switch_origin = None
            self._pending_switch_color = None
            self._pending_switch_frames = 0
            self._pending_switch_retries = 0
            self._pending_switch_scans = 0
            self._shape_stuck_steps = 0
            self._last_player_pos = player_pos
            return True, None

        self._pending_switch_frames += 1
        if self._pending_switch_frames <= 2:
            return False, None
        if self._pending_switch_retries == 0:
            self._pending_switch_retries = 1
            self._pending_switch_frames = 0
            self._emitted_switches += 1
            current = int(ctx.plan_context.get("switch_attempted", 0) or 0)
            ctx.plan_context["switch_attempted"] = max(current, self._emitted_switches)
            return False, ActionDecision(
                "ACTION5",
                "solver:beam-coverage",
                0.50,
                reason="coverage_switch_retry",
                metadata=self._decision_metadata(pending_shape_stage=self._pending_stage_index),
            )

        self._switch_pending = False
        self._pending_stage_index = None
        self._pending_switch_origin = None
        self._pending_switch_color = None
        self._pending_switch_frames = 0
        return False, None

    # ── Activation ─────────────────────────────────────────────────────

    def should_activate(self, ctx: SolverContext) -> bool:
        gm = ctx.game_model
        if gm.player_pos is None and not self.has_active_attempt:
            return False
        mirror_info = _find_mirror_info(gm.components, gm.width, gm.height)
        if mirror_info is None and self.has_active_attempt:
            return True
        if mirror_info is None:
            return False
        shapes = _get_shape_pixels(gm.components, gm.width, gm.height, board=gm.board if hasattr(gm, 'board') else None)
        if not shapes and self._cached_targets and self._planned_shape_count > 0:
            shapes = [[] for _ in range(len(self._cached_targets))]
        if not shapes and self.has_active_attempt:
            return True
        if not shapes:
            return False
        goals = self._goal_cells(ctx, getattr(gm, "board", None))
        if not goals:
            logger.debug("beam activation skipped: no goals active=%s planned=%s", self.has_active_attempt, bool(self._planned_goal_cells))
            return False
        primary = gm.features.primary_dynamics
        coverage_in_progress = (
            self._active_stage_index >= 0
            or self._switch_pending
            or bool(self._cached_targets)
        )
        if primary == "beam" and not coverage_in_progress:
            # A new AR25 level can already be classified as beam before the
            # solver has created an attempt.  Permit the general horizontal
            # mirror planner to bootstrap when multiple shape sources exist;
            # retain the conservative behavior for single-source beam boards.
            if not (mirror_info["horizontal"] and len(shapes) >= 2):
                logger.debug("beam activation skipped: no coverage in progress active=%s cached=%s stage=%s", self.has_active_attempt, bool(self._cached_targets), self._active_stage_index)
                return False
        # Build a board signature to detect level transitions
        # Use only stable features that don't change mid-level.
        sig = (mirror_info["horizontal"], len(gm.goal_positions), len(goals))
        # Reset on any change in board configuration (new level)
        if sig != self._board_sig and not coverage_in_progress:
            self.reset_level()
            goals = self._goal_cells(ctx, getattr(gm, "board", None))
        self._board_sig = sig
        if mirror_info["horizontal"]:
            return True
        return len(shapes) >= 2

    # ── Decision ───────────────────────────────────────────────────────

    def decide(self, ctx: SolverContext) -> ActionDecision | None:
        gm = ctx.game_model
        if gm.player_pos is None:
            logger.debug("beam decide skipped: no player position active=%s", self.has_active_attempt)
            return None

        mirror_info = _find_mirror_info(gm.components, gm.width, gm.height)
        if mirror_info is None:
            mirror_info = self._cached_mirror_info
        if mirror_info is None:
            logger.debug("beam decide skipped: no mirror active=%s cached=%s", self.has_active_attempt, bool(self._cached_mirror_info))
            return None
        gs = max(1, gm.features.grid_scale) if hasattr(gm.features, 'grid_scale') else 3
        board = gm.board if hasattr(gm, 'board') else None
        shapes = _get_shape_pixels(gm.components, gm.width, gm.height, board=board, gs=gs)
        if not shapes and self._cached_targets and self._planned_shape_count > 0:
            shapes = [[] for _ in range(len(self._cached_targets))]
        if not shapes and self.has_active_attempt:
            shapes = [[] for _ in range(max(1, self._planned_shape_count))]
        if not shapes:
            logger.debug("beam decide skipped: no shapes active=%s cached=%s planned=%s", self.has_active_attempt, bool(self._cached_targets), self._planned_shape_count)
            return None
        px, py = gm.player_pos
        self._last_decision_player_pos = (px, py)

        # Reset on level transition (MUST be before cache check)
        try:
            current_level = int(ctx.plan_context.get("levels_completed", 0))
        except (TypeError, ValueError):
            current_level = 0
        logger.debug(
            "beam level state current=%s previous=%s active=%s stage=%s",
            current_level, self._last_levels_completed, self.has_active_attempt,
            self._active_stage_index,
        )
        if self._last_levels_completed is None or current_level > self._last_levels_completed:
            self.reset_level()
        self._last_levels_completed = current_level
        goals = self._goal_cells(ctx, board)
        if not goals:
            logger.debug("beam decide skipped: no goals active=%s planned=%s", self.has_active_attempt, bool(self._planned_goal_cells))
            return None
        self._planned_shape_count = max(self._planned_shape_count, len(shapes))

        switch_confirmed, switch_decision = self._observe_pending_switch(
            ctx,
            player_pos=(px, py),
            player_color=int(gm.features.player_color),
            shapes=shapes,
            gs=gs,
        )
        if switch_decision is not None:
            return switch_decision
        if self._switch_pending:
            return None

        player_jump = (
            self._last_player_pos is not None
            and abs(px - self._last_player_pos[0]) + abs(py - self._last_player_pos[1]) > gs * 4
        )
        execution_started = self._active_stage_index >= 0 or self._switch_pending
        if player_jump and not switch_confirmed and not execution_started:
            self._mirror_target_gy = None
            self._retry_count = 0
            self._invalidate_target_cache()
            self._reset_mirror_phase_progress(clear_unreachable=True)

        current_mirror_gy = mirror_info["axis_py"] // gs if mirror_info["horizontal"] else None
        base_forbidden = _base_forbidden_cells(gm)
        target_row = current_mirror_gy if current_mirror_gy is not None else -1
        use_ranked_families = bool(mirror_info["horizontal"]) and current_level >= 3
        planning_needed = not execution_started or (
            self._cached_targets is None and self._mirror_target_gy is None
        )

        if mirror_info["horizontal"] and current_mirror_gy is not None and planning_needed:
            selected_plan = None
            selected_row, selected_targets = None, None
            mirror_cache_key = (
                current_level,
                int(self._mirror_target_gy) if self._mirror_target_gy is not None else current_mirror_gy,
                self._planned_shape_count,
            )
            if self._mirror_target_gy is None or self._cached_target_key != mirror_cache_key:
                if use_ranked_families:
                    selected_plan = rank_horizontal_mirror_plans(
                        shapes,
                        goals,
                        mirror_info,
                        gm.width,
                        gm.height,
                        current_mirror_row=current_mirror_gy,
                        base_forbidden=base_forbidden,
                        gs=gs,
                        excluded_rows=self._unreachable_mirror_rows,
                    )
                    if selected_plan is None and current_mirror_gy not in self._unreachable_mirror_rows:
                        selected_plan = rank_horizontal_mirror_plans(
                            shapes,
                            goals,
                            mirror_info,
                            gm.width,
                            gm.height,
                            current_mirror_row=current_mirror_gy,
                            base_forbidden=base_forbidden,
                            gs=gs,
                            candidate_rows={current_mirror_gy},
                        )
                    selected_row = (
                        int(selected_plan["mirror_row"])
                        if isinstance(selected_plan, dict) and selected_plan.get("mirror_row") is not None
                        else None
                    )
                else:
                    selected_row, selected_targets = select_best_horizontal_mirror_plan(
                        shapes,
                        goals,
                        mirror_info,
                        gm.width,
                        gm.height,
                        current_mirror_row=current_mirror_gy,
                        base_forbidden=base_forbidden,
                        gs=gs,
                        excluded_rows=self._unreachable_mirror_rows,
                    )
                    if selected_row is None and current_mirror_gy not in self._unreachable_mirror_rows:
                        selected_row, selected_targets = select_best_horizontal_mirror_plan(
                            shapes,
                            goals,
                            mirror_info,
                            gm.width,
                            gm.height,
                            current_mirror_row=current_mirror_gy,
                            base_forbidden=base_forbidden,
                            gs=gs,
                            candidate_rows={current_mirror_gy},
                        )
                self._mirror_target_gy = selected_row if selected_row is not None else current_mirror_gy
                self._cached_mirror_info = dict(mirror_info)
                if use_ranked_families:
                    self._ranked_target_families = (
                        list(selected_plan.get("families", []))
                        if isinstance(selected_plan, dict) else []
                    )
                    self._current_family_index = 0
                    self._family_retry_count = 0
                    self._cached_targets = None
                    self._base_targets = None
                    self._adopt_target_family(0, gm, gs)
                else:
                    self._ranked_target_families = []
                    self._current_family_index = 0
                    self._family_retry_count = 0
                    self._cached_targets = selected_targets
                    self._base_targets = [dict(stage) for stage in (selected_targets or [])]
                self._cached_target_key = (
                    current_level,
                    int(self._mirror_target_gy),
                    self._planned_shape_count,
                )
                self._cached_frame = current_level

            target_row = int(self._mirror_target_gy if self._mirror_target_gy is not None else current_mirror_gy)
            if target_row != current_mirror_gy:
                player_is_mirror = abs(py - mirror_info["axis_py"]) <= gs
                if not player_is_mirror:
                    self._unreachable_mirror_rows.add(target_row)
                    self._mirror_target_gy = current_mirror_gy
                    self._retry_count += 1
                    self._reset_attempt("mirror_row_unreachable")
                    self._reset_mirror_phase_progress()
                    target_row = current_mirror_gy
                else:
                    if self._last_observed_mirror_gy == current_mirror_gy:
                        self._mirror_stall_steps += 1
                    else:
                        self._last_observed_mirror_gy = current_mirror_gy
                        self._mirror_stall_steps = 0
                    if self._mirror_stall_steps >= 4:
                        self._unreachable_mirror_rows.add(target_row)
                        self._mirror_target_gy = current_mirror_gy
                        self._retry_count += 1
                        self._reset_attempt("mirror_row_stalled")
                        self._reset_mirror_phase_progress()
                        target_row = current_mirror_gy
                    else:
                        direction = "up" if target_row < current_mirror_gy else "down"
                        return ActionDecision(
                            DIR_TO_ACTION.get(direction, "ACTION1"),
                            "solver:beam-coverage",
                            0.65,
                            reason=f"coverage_position_mirror_{direction}",
                            metadata=self._decision_metadata(),
                        )
            else:
                self._reset_mirror_phase_progress()

        # The selected row remains authoritative after execution starts.
        if mirror_info["horizontal"] and self._mirror_target_gy is not None:
            mirror_info = dict(mirror_info)
            mirror_info["axis_py"] = int(self._mirror_target_gy) * gs + (gs // 2)

        # ── Mirror positioning phase ─────────────────────────────────
        # The AR25 engine only enforces board bounds for shape movement.
        # Shapes may overlap each other, goals, mirrors, and beam cells.
        forbidden: set[tuple[int, int]] = set()

        # Cache targets — recompute on level change or retry
        target_cache_key = (
            current_level,
            (mirror_info["axis_py"] // gs) if mirror_info["horizontal"] else -1,
            self._planned_shape_count,
        )
        if self._cached_targets is None or (
            self._active_stage_index < 0 and self._cached_target_key != target_cache_key
        ):
            if use_ranked_families:
                self._ranked_target_families = rank_joint_best_placements(
                    shapes, goals, mirror_info, gm.width, gm.height,
                    forbidden_cells=forbidden, gs=gs,
                )
                if self._ranked_target_families:
                    self._current_family_index = 0
                    self._family_retry_count = 0
                    self._adopt_target_family(0, gm, gs)
                else:
                    self._cached_targets = find_joint_best_placement(
                        shapes, goals, mirror_info, gm.width, gm.height,
                        forbidden_cells=forbidden, gs=gs,
                    )
                    self._base_targets = [dict(stage) for stage in self._cached_targets]
            else:
                self._ranked_target_families = []
                self._cached_targets = find_joint_best_placement(
                    shapes, goals, mirror_info, gm.width, gm.height,
                    forbidden_cells=forbidden, gs=gs,
                )
                self._base_targets = [dict(stage) for stage in self._cached_targets]
            self._cached_target_key = target_cache_key
            self._cached_frame = current_level

        if not self._cached_targets:
            return None

        self._planned_shape_count = max(self._planned_shape_count, len(self._cached_targets))

        targets = self._cached_targets

        active_target_idx = min(max(self._active_stage_index, 0), len(targets) - 1)
        stage = targets[active_target_idx]
        target_cx, target_cy = stage["target_anchor"]
        # Grid-align target: snap to nearest grid cell's pixel center
        target_cx = (target_cx // gs) * gs + (gs // 2)
        target_cy = (target_cy // gs) * gs + (gs // 2)

        # Analyzer player selection can alternate between overlapping shape
        # components after ACTION5. For beam puzzles, the active shape is the
        # visible shape nearest its frozen stage target; use that position for
        # navigation while retaining gm.player_pos for switch confirmation.
        active_shape = None
        if self._active_stage_index >= 0:
            active_shape, active_center = self._select_active_shape_for_stage(
                shapes,
                stage_index=self._active_stage_index,
                target=(target_cx, target_cy),
            )
            if active_center is not None:
                px, py = active_center
                self._last_decision_player_pos = (px, py)

        recovery = self._reflection_recovery_decision(
            shapes=shapes,
            mirror_info=mirror_info,
            goals=goals,
            board_width=gm.width,
            board_height=gm.height,
            target=(target_cx, target_cy),
            gs=gs,
        )
        if recovery is not None:
            return recovery

        if self._last_player_pos == (px, py):
            self._shape_stuck_steps += 1
        else:
            self._shape_stuck_steps = 0
        self._last_player_pos = (px, py)

        # On first activation, switch from the mirror controller to shape 0.
        if self._active_stage_index < 0:
            return self._request_switch(
                ctx,
                next_stage=0,
                player_pos=(px, py),
                player_color=int(gm.features.player_color),
                reason="coverage_switch_to_shape",
                confidence=0.70,
            )

        shape_count = max(self._planned_shape_count, len(targets))
        if self._shape_stuck_steps >= 30 and self._active_stage_index + 1 < shape_count:
            self._shape_stuck_steps = 0
            return self._request_switch(
                ctx,
                next_stage=self._active_stage_index + 1,
                player_pos=(px, py),
                player_color=int(gm.features.player_color),
                reason="coverage_stuck_switch",
                confidence=0.65,
            )

        # Convert to grid coordinates for navigation (game engine moves in grid steps)
        gx, gy = px // gs, py // gs
        target_gx = target_cx // gs
        target_gy = target_cy // gs
        g_w = gm.width // gs
        g_h = gm.height // gs
        arrival_dist = abs(gx - target_gx) + abs(gy - target_gy)
        # Also compute pixel distance for comparison
        pixel_dist = abs(px - target_cx) + abs(py - target_cy)
        final_stage = self._active_stage_index >= shape_count - 1
        at_target = arrival_dist == 0

        horizontal_final_covered = True
        if final_stage and at_target and mirror_info.get("horizontal"):
            horizontal_final_covered = self._horizontal_final_stage_covered(
                active_shape=active_shape,
                mirror_info=mirror_info,
                goals=goals,
                board_width=gm.width,
                board_height=gm.height,
                gs=gs,
            )

        if final_stage and self._final_tuning:
            self._shape_stuck_steps = 0
            self._both_placed_steps = 0
            return self._emit_tuning_action(
                gm=gm,
                gx=gx,
                gy=gy,
                gs=gs,
                target_cx=target_cx,
                target_cy=target_cy,
            )

        # Track final-stage placement only after the live entity actually
        # reaches its frozen anchor.  Counting every final-stage frame moved
        # a still-navigating shape into the tuning spiral prematurely.
        # A one-cell tolerance can switch away before the sprite occupies the
        # planned cell.  AR25 checks completion after movement, so confirm the
        # exact logical cell before advancing or tuning the stage.
        if final_stage and at_target and mirror_info.get("horizontal") and not horizontal_final_covered:
            self._shape_stuck_steps = 0
            self._both_placed_steps = 0
            return self._emit_tuning_action(
                gm=gm,
                gx=gx,
                gy=gy,
                gs=gs,
                target_cx=target_cx,
                target_cy=target_cy,
                final_stage_coverage_incomplete=True,
            )
        if final_stage and at_target:
            self._both_placed_steps += 1
        else:
            self._both_placed_steps = 0

        # Emergency switch if close enough in pixel or grid space.
        # Skip when both shapes have been placed for >5 frames (to allow retry).
        if self._both_placed_steps <= 2 and at_target:
            if not final_stage:
                self._shape_stuck_steps = 0
                return self._request_switch(
                    ctx,
                    next_stage=self._active_stage_index + 1,
                    player_pos=(px, py),
                    player_color=int(gm.features.player_color),
                    reason="coverage_switch_next",
                    confidence=0.68,
                )
            return ActionDecision("ACTION6", "solver:beam-coverage", 0.60,
                                  reason="coverage_all_placed")

        # Emergency switch after stuck or both placed (fast retry with 3-frame cooldown)
        if self._shape_stuck_steps >= 6 or self._both_placed_steps >= 3:
            if not final_stage:
                self._shape_stuck_steps = 0
                return self._request_switch(
                    ctx,
                    next_stage=self._active_stage_index + 1,
                    player_pos=(px, py),
                    player_color=int(gm.features.player_color),
                    reason="coverage_early_switch",
                    confidence=0.65,
                )
            # Both shapes placed — try brute-force fine-tuning.
            # Move current entity by 1 grid cell in each cardinal direction.
            # The game engine checks win after each step.
            self._shape_stuck_steps = 0
            self._both_placed_steps = 0
            return self._emit_tuning_action(
                gm=gm,
                gx=gx,
                gy=gy,
                gs=gs,
                target_cx=target_cx,
                target_cy=target_cy,
            )

        # Direct axis-priority navigation: prioritize the axis with larger delta
        # to avoid A* oscillation between directions.
        dx = target_gx - gx
        dy = target_gy - gy
        logger.debug(
            "beam navigation state: player_px=(%s,%s) player_grid=(%s,%s) "
            "target_px=(%s,%s) target_grid=(%s,%s) delta=(%s,%s) stage=%s",
            px, py, gx, gy, target_cx, target_cy, target_gx, target_gy,
            dx, dy, self._active_stage_index,
        )
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
        for d in candidates:
            dvx, dvy = vec[d]
            ngx, ngy = gx + dvx, gy + dvy
            if (ngx, ngy) in grid_walkable:
                return ActionDecision(
                    DIR_TO_ACTION.get(d, "ACTION1"), "solver:beam-coverage", 0.65,
                    reason=f"coverage_nav_{d}",
                    metadata=self._decision_metadata(
                        target_anchor=[target_cx, target_cy],
                        shape_stage=self._active_stage_index,
                    ),
                )
        # Forced fallback
        for d, (dvx, dvy) in vec.items():
            ngx, ngy = gx + dvx, gy + dvy
            if 0 <= ngx < g_w and 0 <= ngy < g_h:
                return ActionDecision(
                    DIR_TO_ACTION.get(d, "ACTION1"), "solver:beam-coverage", 0.40,
                    reason=f"coverage_forced_{d}",
                    metadata=self._decision_metadata(
                        target_anchor=[target_cx, target_cy],
                        shape_stage=self._active_stage_index,
                    ),
                )
        return None

    def _reflection_recovery_decision(
        self,
        *,
        shapes: list[list[tuple[int, int]]],
        mirror_info: dict[str, Any],
        goals: list[tuple[int, int]],
        board_width: int,
        board_height: int,
        target: tuple[int, int],
        gs: int,
    ) -> ActionDecision | None:
        """Keep the active shape selected until its reflection is visible."""
        if self._active_stage_index < 0 or self._switch_pending:
            return None
        if not mirror_info.get("horizontal"):
            self._reset_reflection_recovery()
            return None
        active_shape, _ = self._select_active_shape_for_stage(
            shapes,
            stage_index=self._active_stage_index,
            target=target,
        )
        if active_shape is None:
            # A dropout cannot prove that the reflection is still absent.  The
            # frozen attempt remains active and the next observation retries.
            return None
        visibility = reflected_visibility(
            active_shape,
            mirror_info,
            board_width,
            board_height,
            goals,
            gs=gs,
        )
        self._reflected_visible_count = len(visibility["visible_cells"])
        self._reflected_goal_count = len(visibility["visible_goal_cells"])
        self._reflection_offscreen = bool(visibility["offscreen"])
        direction = reflection_recovery_direction(visibility, mirror_info)
        if direction is None:
            if self._reflection_recovery:
                self._reset_reflection_recovery()
            return None
        if not self._reflection_recovery:
            self._reflection_recovery = True
            self._recovery_direction = direction
            self._recovery_step = 0
        elif self._recovery_direction != direction:
            self._recovery_direction = direction
        if self._recovery_step >= self._recovery_budget:
            self._reset_attempt("reflection_recovery_exhausted")
            return None
        self._recovery_step += 1
        return ActionDecision(
            DIR_TO_ACTION[direction],
            "solver:beam-coverage",
            0.78,
            reason=f"coverage_reflection_recovery_{direction}",
            metadata=self._decision_metadata(
                target_anchor=[target[0], target[1]],
                shape_stage=self._active_stage_index,
            ),
        )

    def _build_expanded_walkable(self, gm) -> set[tuple[int, int]]:
        # In beam-coverage puzzles, shape pieces can move across all
        # board cells (the game engine's sprite-based collision model
        # allows traversal through most terrain).  Use the full board.
        w = set()
        for y in range(gm.height):
            for x in range(gm.width):
                w.add((x, y))
        return w


register_solver("beam-coverage", BeamCoverageSolver)
