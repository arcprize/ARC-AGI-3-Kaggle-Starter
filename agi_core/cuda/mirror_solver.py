"""GPU-accelerated mirror reflection placement with CPU fallback.

The ``best_reflection_placement_gpu`` function replaces the CPU brute-force
search in ``shape_recognition._best_reflection_placement``, testing
thousands of (tx, ty) positions in parallel on the GPU.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from agi_core.cuda import CUDA_AVAILABLE, is_enabled
from agi_core.cuda.kernels import _launch_mirror_placement_kernel

logger = logging.getLogger(__name__)


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
    """Find the best reflection placement on the GPU.

    Parameters
    ----------
    width, height : board dimensions (typically 64×64).
    component_cells : pixel cells of the shape component.
    component_bbox : (x1, y1, x2, y2) bounding box.
    remaining_goals : set of (x, y) goal cells still uncovered.
    horizontal : True if the mirror reflects the Y axis.
    axis : pixel coordinate of the mirror axis.
    config : optional config with ``cuda_enabled``.

    Returns
    -------
    ``(fresh_hits, center_x, center_y)`` or *None* if CUDA is unavailable or
    no valid placement exists (caller falls back to CPU).
    """
    if not remaining_goals:
        return None
    if not is_enabled(config) or not CUDA_AVAILABLE:
        return None

    try:
        from numba import cuda

        # Build goal bitmask (64×64 uint8 = 512 bytes)
        goal_np = np.zeros((height, width), dtype=np.uint8)
        for gx, gy in remaining_goals:
            if 0 <= gx < width and 0 <= gy < height:
                goal_np[gy, gx] = 1

        # Build shape mask (normalised)
        x1, y1, x2, y2 = component_bbox
        sw = x2 - x1 + 1
        sh = y2 - y1 + 1
        shape_np = np.zeros((sh, sw), dtype=np.uint8)
        for px, py in component_cells:
            dx = px - x1
            dy = py - y1
            if 0 <= dx < sw and 0 <= dy < sh:
                shape_np[dy, dx] = 1

        # Upload to GPU
        d_goal = cuda.to_device(goal_np)
        d_shape = cuda.to_device(shape_np)
        d_best = cuda.device_array(4, dtype=np.int32)
        d_best[0] = 0  # packed score
        d_best[1] = 0  # center_x
        d_best[2] = 0  # center_y
        d_best[3] = 0  # hit count

        _launch_mirror_placement_kernel(
            d_goal, d_shape, sw, sh,
            horizontal, axis, width, height, d_best,
        )

        best = d_best.copy_to_host()
        hits = int(best[3])
        if hits == 0:
            return None
        return (hits, int(best[1]), int(best[2]))

    except Exception as exc:
        logger.debug("CUDA mirror placement failed: %s", exc)
        return None
