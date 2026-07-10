"""GPU-accelerated salience map computation with CPU fallback."""

import logging
from typing import Any

import numpy as np

from agi_core.cuda import CUDA_AVAILABLE, is_enabled
from agi_core.cuda.kernels import _launch_salience_kernel

logger = logging.getLogger(__name__)


def compute_salience_gpu(
    board: list[list[int]],
    player_pos: tuple[int, int] | None,
    goal_positions: list[tuple[int, int]],
    prev_board: list[list[int]] | None = None,
    config: object | None = None,
) -> list[list[float]] | None:
    """Compute a 64×64 salience map on the GPU.
    
    Returns a 2-D list[list[float]] or None if CUDA is unavailable / fails
    (caller falls back to CPU).
    """
    if not is_enabled(config) or not CUDA_AVAILABLE:
        return None

    try:
        h = len(board)
        w = len(board[0]) if board else 0
        if h == 0 or w == 0:
            return None

        board_np = np.asarray(board, dtype=np.int8)
        prev_np = np.asarray(prev_board, dtype=np.int8) if prev_board is not None else None

        px = int(player_pos[0]) if player_pos is not None else 0
        py = int(player_pos[1]) if player_pos is not None else 0
        if player_pos is None:
            px = py = 0

        goal_xs = np.array([int(g[0]) for g in goal_positions], dtype=np.int32) if goal_positions else np.zeros(0, dtype=np.int32)
        goal_ys = np.array([int(g[1]) for g in goal_positions], dtype=np.int32) if goal_positions else np.zeros(0, dtype=np.int32)

        out_np = np.zeros((h, w), dtype=np.float32)

        # Launch kernel (now with prev_board for motion detection)
        _launch_salience_kernel(board_np, px, py, goal_xs, goal_ys, out_np, prev_board=prev_np)

        return out_np.tolist()

    except Exception as exc:
        logger.debug("CUDA salience failed: %s", exc)
        return None
