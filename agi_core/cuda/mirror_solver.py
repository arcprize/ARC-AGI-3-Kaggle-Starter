"""GPU-accelerated reflection placement and scoring helpers."""

from __future__ import annotations

import logging

import numpy as np

from agi_core.cuda import CUDA_AVAILABLE, kernel_available
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
    if not kernel_available("reflection_scoring", config) or not CUDA_AVAILABLE:
        return None

    shape_payload = _shape_mask_from_component(component_cells, component_bbox)
    if shape_payload is None:
        return None
    shape_np, shape_w, shape_h = shape_payload

    try:
        from numba import cuda

        goal_np = np.zeros((height, width), dtype=np.uint8)
        for gx, gy in goal_cells:
            if 0 <= gx < width and 0 <= gy < height:
                goal_np[gy, gx] = 1

        d_goal = cuda.to_device(goal_np)
        d_shape = cuda.to_device(shape_np)
        d_scores = cuda.device_array((height, width), dtype=np.int32)
        d_scores[:] = 0

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

        host_scores = d_scores.copy_to_host()
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
    if not kernel_available("mirror_placement", config) or not CUDA_AVAILABLE:
        return None

    try:
        from numba import cuda

        goal_np = np.zeros((height, width), dtype=np.uint8)
        for gx, gy in remaining_goals:
            if 0 <= gx < width and 0 <= gy < height:
                goal_np[gy, gx] = 1

        shape_payload = _shape_mask_from_component(component_cells, component_bbox)
        if shape_payload is None:
            return None
        shape_np, shape_w, shape_h = shape_payload

        d_goal = cuda.to_device(goal_np)
        d_shape = cuda.to_device(shape_np)
        d_best = cuda.device_array(4, dtype=np.int32)
        d_best[0] = 0
        d_best[1] = 0
        d_best[2] = 0
        d_best[3] = 0

        _launch_mirror_placement_kernel(
            d_goal,
            d_shape,
            shape_w,
            shape_h,
            horizontal,
            axis,
            width,
            height,
            d_best,
        )

        best = d_best.copy_to_host()
        hits = int(best[3])
        if hits <= 0:
            return None
        return (hits, int(best[1]), int(best[2]))
    except Exception as exc:
        logger.debug("CUDA mirror placement failed: %s", exc)
        return None
