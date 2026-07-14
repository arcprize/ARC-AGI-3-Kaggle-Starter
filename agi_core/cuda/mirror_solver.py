"""GPU-accelerated reflection placement and scoring helpers."""

from __future__ import annotations

import logging

import numpy as np

import agi_core.cuda as cuda_runtime
from agi_core.cuda import get_cupy, kernel_available, note_kernel_stat
from agi_core.cuda.kernels import (
    _launch_mirror_placement_kernel,
    _launch_reflection_score_kernel,
)

logger = logging.getLogger(__name__)


def _shape_mask_from_component(
    component_cells: list[tuple[int, int]],
    component_bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, int, int] | None:
    if not component_cells:
        return None
    x1, y1, x2, y2 = component_bbox
    shape_w = x2 - x1 + 1
    shape_h = y2 - y1 + 1
    if shape_w <= 0 or shape_h <= 0:
        return None
    shape_np = np.zeros((shape_h, shape_w), dtype=np.uint8)
    for px, py in component_cells:
        dx = px - x1
        dy = py - y1
        if 0 <= dx < shape_w and 0 <= dy < shape_h:
            shape_np[dy, dx] = 1
    return shape_np, shape_w, shape_h


def _relative_shape_center(
    component_cells: list[tuple[int, int]],
    component_bbox: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    if not component_cells:
        return None
    x1, y1, _, _ = component_bbox
    rel_x = sum(px - x1 for px, _ in component_cells) // len(component_cells)
    rel_y = sum(py - y1 for _, py in component_cells) // len(component_cells)
    return rel_x, rel_y


def score_reflection_candidates_gpu(
    *,
    width: int,
    height: int,
    component_cells: list[tuple[int, int]],
    component_bbox: tuple[int, int, int, int],
    goal_cells: set[tuple[int, int]],
    horizontal: bool,
    axis: int,
    config: object | None = None,
) -> list[dict[str, int]] | None:
    """Score all valid top-left placements against reflected goal coverage."""
    if not goal_cells:
        return None
    if not kernel_available("reflection_scoring", config) or not cuda_runtime.CUDA_AVAILABLE:
        return None

    shape_payload = _shape_mask_from_component(component_cells, component_bbox)
    if shape_payload is None:
        return None
    shape_np, shape_w, shape_h = shape_payload

    try:
        cp = get_cupy()
        goal_np = np.zeros((height, width), dtype=np.uint8)
        for gx, gy in goal_cells:
            if 0 <= gx < width and 0 <= gy < height:
                goal_np[gy, gx] = 1

        d_goal = cp.asarray(goal_np, dtype=cp.uint8)
        d_shape = cp.asarray(shape_np, dtype=cp.uint8)
        d_scores = cp.zeros((height, width), dtype=cp.int32)

        _launch_reflection_score_kernel(
            d_goal,
            d_shape,
            shape_w,
            shape_h,
            horizontal,
            axis,
            width,
            height,
            d_scores,
        )
        note_kernel_stat("reflection_scoring")

        host_scores = cp.asnumpy(d_scores)
        results: list[dict[str, int]] = []
        for ty in range(max(0, height - shape_h + 1)):
            for tx in range(max(0, width - shape_w + 1)):
                hits = int(host_scores[ty, tx])
                if hits <= 0:
                    continue
                results.append(
                    {
                        "top_left_x": tx,
                        "top_left_y": ty,
                        "center_x": tx + shape_w // 2,
                        "center_y": ty + shape_h // 2,
                        "hits": hits,
                    }
                )
        return results
    except Exception as exc:
        logger.debug("CUDA reflection scoring failed: %s", exc)
        return None


def best_reflection_placement_gpu(
    *,
    width: int,
    height: int,
    component_cells: list[tuple[int, int]],
    component_bbox: tuple[int, int, int, int],
    remaining_goals: set[tuple[int, int]],
    horizontal: bool,
    axis: int,
    config: object | None = None,
) -> tuple[int, int, int] | None:
    """Find the best reflection placement on the GPU."""
    if not remaining_goals:
        return None
    if not kernel_available("mirror_placement", config) or not cuda_runtime.CUDA_AVAILABLE:
        return None

    try:
        cp = get_cupy()
        goal_np = np.zeros((height, width), dtype=np.uint8)
        for gx, gy in remaining_goals:
            if 0 <= gx < width and 0 <= gy < height:
                goal_np[gy, gx] = 1

        shape_payload = _shape_mask_from_component(component_cells, component_bbox)
        if shape_payload is None:
            return None
        shape_np, shape_w, shape_h = shape_payload
        rel_center = _relative_shape_center(component_cells, component_bbox)
        if rel_center is None:
            return None
        rel_cx, rel_cy = rel_center
        d_goal = cp.asarray(goal_np, dtype=cp.uint8)
        d_shape = cp.asarray(shape_np, dtype=cp.uint8)
        d_scores = cp.zeros((height, width), dtype=cp.int32)

        _launch_reflection_score_kernel(
            d_goal,
            d_shape,
            shape_w,
            shape_h,
            horizontal,
            axis,
            width,
            height,
            d_scores,
        )
        note_kernel_stat("mirror_placement")

        host_scores = cp.asnumpy(d_scores)
        if not component_cells:
            return None
        current_center = (
            sum(px for px, _ in component_cells) // len(component_cells),
            sum(py for _, py in component_cells) // len(component_cells),
        )
        current_side = (
            0
            if (current_center[1] if horizontal else current_center[0]) == axis
            else (1 if (current_center[1] if horizontal else current_center[0]) > axis else -1)
        )
        best_score: tuple[int, int] | None = None
        best_result: tuple[int, int, int] | None = None
        for ty in range(max(0, height - shape_h + 1)):
            for tx in range(max(0, width - shape_w + 1)):
                hits = int(host_scores[ty, tx])
                if hits <= 0:
                    continue
                if current_side != 0:
                    off_axis = False
                    for py in range(shape_h):
                        for px in range(shape_w):
                            if shape_np[py, px] == 0:
                                continue
                            coord = ty + py if horizontal else tx + px
                            side = 0 if coord == axis else (1 if coord > axis else -1)
                            if side not in (0, current_side):
                                off_axis = True
                                break
                        if off_axis:
                            break
                if off_axis:
                    continue
                cx = tx + rel_cx
                cy = ty + rel_cy
                score = (
                    hits,
                    -(abs(cx - current_center[0]) + abs(cy - current_center[1])),
                )
                if best_score is not None and score <= best_score:
                    continue
                best_score = score
                best_result = (hits, cx, cy)
        return best_result
    except Exception as exc:
        logger.debug("CUDA mirror placement failed: %s", exc)
        return None
