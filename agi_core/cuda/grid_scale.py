"""GPU-accelerated grid scale detection with CPU fallback.

The grid scale is the pixel-to-tile ratio (e.g. 3 for AR25's 21×21 grid
in a 64×64 viewport).  The GPU computes block uniformity and edge
periodicity in parallel for all candidate scales 2–16.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

import agi_core.cuda as cuda_runtime
from agi_core.cuda import get_cupy, is_enabled, note_kernel_stat
from agi_core.cuda.kernels import _launch_uniformity_kernel, _launch_edge_kernel

logger = logging.getLogger(__name__)


def _best_scale(
    uniformity: np.ndarray,
    edge_scores: np.ndarray,
    total_cells: int = 64 * 64,
) -> int:
    """Combine uniformity (70 %) and edge periodicity (30 %) to pick the
    best scale.  Returns 1 if no scale exceeds 50 % uniformity."""
    if (
        float(np.max(edge_scores[2:17])) <= 0.0
        and float(np.max(uniformity[2:17])) >= 0.999
    ):
        return 1
    best_scale = 1
    best_score = 0.0
    for scale in range(2, 17):
        uniform_pct = float(uniformity[scale])
        edge_bonus = float(edge_scores[scale]) / max(1, total_cells // 100)
        combined = uniform_pct * 0.7 + min(edge_bonus, 0.3)
        if combined > best_score and uniform_pct > 0.50:
            best_score = combined
            best_scale = scale
    return best_scale


def detect_grid_scale_gpu(
    board: list[list[int]],
    config: object | None = None,
) -> int | None:
    """Detect the grid scale factor on the GPU.

    Parameters
    ----------
    board : 64×64 int grid.
    config : optional config with ``cuda_enabled`` attribute.

    Returns
    -------
    int (the detected scale) or *None* if CUDA is unavailable / fails.
    """
    if not is_enabled(config) or not cuda_runtime.CUDA_AVAILABLE:
        return None

    try:
        height = len(board)
        width = len(board[0]) if height and board[0] else 0
        if height == 0 or width == 0:
            return None

        cp = get_cupy()
        board_np = np.asarray(board, dtype=np.int8)
        d_board = cp.asarray(board_np, dtype=cp.int8)

        d_uniformity = cp.zeros(17, dtype=cp.float32)
        d_edge_scores = cp.zeros(17, dtype=cp.int32)

        _launch_uniformity_kernel(d_board, d_uniformity, width, height)
        note_kernel_stat("grid_scale.uniformity")
        _launch_edge_kernel(d_board, d_edge_scores, width, height)
        note_kernel_stat("grid_scale.edge")

        uniformity = cp.asnumpy(d_uniformity)
        edge_scores = cp.asnumpy(d_edge_scores)

        return _best_scale(uniformity, edge_scores, total_cells=width * height)

    except Exception as exc:
        logger.debug("CUDA grid scale detection failed: %s", exc)
        return None
