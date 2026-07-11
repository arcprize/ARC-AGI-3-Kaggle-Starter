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
    return [
        list(c.cells)
        for c in components
        if c.dominant_color in (0, 5)
        and 4 <= c.size <= 100
        and not c.touches_border
    ]


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
) -> set[tuple[int, int]]:
    """Simulate beam propagation: each non-transparent pixel reflects
    across the mirror up to _MAX_REFLECT_DEPTH times, matching the
    game engine's frckrbsfcm() logic.
    """
    covered: set[tuple[int, int]] = set()
    q: deque[tuple[int, int, int]] = deque()
    visited: set[tuple[int, int]] = set()

    for px, py in shape_cells:
        covered.add((px, py))
        if (px, py) not in visited:
            visited.add((px, py))
            q.append((px, py, 0))

    horizontal = mirror_info.get("horizontal", True)
    axis_px = mirror_info.get("axis_px", 0)
    axis_py = mirror_info.get("axis_py", 0)

    while q:
        px, py, depth = q.popleft()
        if depth >= _MAX_REFLECT_DEPTH:
            continue
        if horizontal:
            rx, ry = px, 2 * axis_py - py
        else:
            rx, ry = 2 * axis_px - px, py
        if 0 <= rx < board_width and 0 <= ry < board_height and (rx, ry) not in visited:
            visited.add((rx, ry))
            covered.add((rx, ry))
            q.append((rx, ry, depth + 1))

    return covered


def find_best_placement(
    shape_cells: list[tuple[int, int]],
    remaining_goals: set[tuple[int, int]],
    mirror_info: dict[str, Any],
    board_width: int,
    board_height: int,
    current_center: tuple[int, int] | None = None,
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
            covered = predict_beam_coverage(
                list(placed), mirror_info, board_width, board_height,
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
) -> list[dict[str, Any]]:
    remaining = set(goal_cells)
    stage_targets: list[dict[str, Any]] = []

    for idx, shape_cells in enumerate(shapes):
        if not shape_cells:
            continue
        current_center = (
            sum(x for x, _ in shape_cells) // len(shape_cells),
            sum(y for _, y in shape_cells) // len(shape_cells),
        )
        best_pos = find_best_placement(
            shape_cells, remaining, mirror_info,
            board_width, board_height,
            current_center=current_center,
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
            list(placed), mirror_info, board_width, board_height,
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

    def reset_level(self) -> None:
        self._cached_targets = None
        self._cached_frame = -1
        self._switch_count = 0
        self._last_player_pos = None
        self._shape_stuck_steps = 0

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

        frame_key = len(gm.components) * 1000 + (gm.player_pos[0] * gm.width + gm.player_pos[1])
        if self._cached_targets is None or self._cached_frame != frame_key:
            self._cached_targets = compute_coverage_for_all_shapes(
                shapes, goals, mirror_info, gm.width, gm.height,
            )
            self._cached_frame = frame_key

        if not self._cached_targets:
            return None

        px, py = gm.player_pos

        player_near_mirror = False
        if mirror_info["horizontal"]:
            player_near_mirror = abs(py - mirror_info["axis_py"]) <= 3
        else:
            player_near_mirror = abs(px - mirror_info["axis_px"]) <= 3

        active_target_idx = min(self._switch_count, len(self._cached_targets) - 1)
        stage = self._cached_targets[active_target_idx]
        target_cx, target_cy = stage["target_anchor"]

        if self._last_player_pos == (px, py):
            self._shape_stuck_steps += 1
        else:
            self._shape_stuck_steps = 0
        self._last_player_pos = (px, py)

        if player_near_mirror and self._switch_count == 0:
            if len(shapes) > 0:
                self._switch_count = 1
                return ActionDecision("ACTION5", "solver:beam-coverage", 0.70,
                                      reason="coverage_switch_to_shape")

        if self._switch_count == 0:
            return None

        if self._shape_stuck_steps >= 30 and self._switch_count < len(shapes):
            self._switch_count += 1
            self._shape_stuck_steps = 0
            if self._switch_count < len(shapes):
                return ActionDecision("ACTION5", "solver:beam-coverage", 0.65,
                                      reason="coverage_stuck_switch")
            return None

        arrival_dist = abs(px - target_cx) + abs(py - target_cy)
        final_stage = self._switch_count >= len(shapes)

        if arrival_dist <= (1 if final_stage else 3):
            if self._switch_count < len(shapes):
                self._switch_count += 1
                if self._switch_count < len(shapes):
                    return ActionDecision("ACTION5", "solver:beam-coverage", 0.68,
                                          reason="coverage_switch_next")
                return ActionDecision("ACTION6", "solver:beam-coverage", 0.60,
                                      reason="coverage_all_placed")
            return ActionDecision("ACTION6", "solver:beam-coverage", 0.60,
                                  reason="coverage_wait")

        expanded = self._build_expanded_walkable(gm)
        try:
            path = astar(
                (px, py), (target_cx, target_cy), expanded,
                gm.width, gm.height, max_steps=5000,
            )
        except Exception:
            path = None

        if path and len(path) > 1:
            nx, ny = path[1]
            d = ("right" if nx > px else "left" if nx < px else
                 "down" if ny > py else "up")
            return ActionDecision(
                DIR_TO_ACTION.get(d, "ACTION1"), "solver:beam-coverage", 0.75,
                reason=f"coverage_nav_{d}",
                metadata={"target_anchor": [target_cx, target_cy],
                          "shape_stage": self._switch_count},
            )

        failed = ctx.plan_context.get("failed_directions", set())
        dx = target_cx - px
        dy = target_cy - py
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
                    DIR_TO_ACTION.get(d, "ACTION1"), "solver:beam-coverage", 0.55,
                    reason=f"coverage_heuristic_{d}",
                    metadata={"target_anchor": [target_cx, target_cy],
                              "shape_stage": self._switch_count},
                )
        for d, (dvx, dvy) in vec.items():
            nx, ny = px + dvx, py + dvy
            if 0 <= nx < gm.width and 0 <= ny < gm.height:
                return ActionDecision(
                    DIR_TO_ACTION.get(d, "ACTION1"), "solver:beam-coverage", 0.40,
                    reason=f"coverage_forced_{d}",
                    metadata={"target_anchor": [target_cx, target_cy],
                              "shape_stage": self._switch_count},
                )
        return None

    def _build_expanded_walkable(self, gm) -> set[tuple[int, int]]:
        w = set(gm.walkable)
        for c in gm.components:
            if not c.touches_border:
                for cx, cy in c.cells:
                    w.add((cx, cy))
        if gm.player_pos:
            w.add(gm.player_pos)
        for gx, gy in gm.goal_positions:
            w.add((gx, gy))
        return w


register_solver("beam-coverage", BeamCoverageSolver)
