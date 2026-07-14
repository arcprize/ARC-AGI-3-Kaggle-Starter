"""GPU-accelerated connected-component labeling with CPU fallback.

Provides a single unified entry point ``detect_components_gpu`` that both
perception pipelines (``shape_recognition.py`` and ``analyzer.py``) call
instead of their respective CPU flood-fills.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

import agi_core.cuda as cuda_runtime
from agi_core.cuda import get_cupy, is_enabled, note_kernel_stat
from agi_core.cuda.kernels import (
    _compact_labels_cpu,
    _launch_compact_labels,
    _launch_init_labels,
    _launch_propagate_labels,
)

logger = logging.getLogger(__name__)


def detect_components_gpu(
    board: list[list[int]],
    bg_color: int = -1,
    config: object | None = None,
) -> dict[str, Any] | None:
    """Unified GPU connected-component labeling.

    Parameters
    ----------
    board : 64×64 int grid (list[list[int]]).
    bg_color : background colour (cells equal to this get label -1).
               Pass -1 to include all non-negative cells.
    config : optional config object with a ``cuda_enabled`` attribute.

    Returns
    -------
    dict with keys:
        ``"components"``
            list of dicts — one per component.  Each dict has:
            ``label`` (int), ``color`` (int), ``cells`` (list[tuple]),
            ``bbox`` (4-tuple), ``size`` (int), ``center`` (2-tuple),
            ``touches_border`` (bool).
        ``"color_frequencies"``
            dict[int, int] — count of cells per colour (excluding bg and -1).

    Returns *None* if CUDA is unavailable or the kernel fails (caller falls
    back to the respective CPU implementation).
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
        if bg_color >= 0:
            board_np = board_np.copy()
            board_np[board_np == int(bg_color)] = -1
        d_board = cp.asarray(board_np, dtype=cp.int8)
        d_labels = cp.empty((height, width), dtype=cp.int32)

        # Phase 1: initialise labels
        _launch_init_labels(d_board, d_labels, width, height)
        note_kernel_stat("components.init")

        # Phase 2: iterative propagation (max 64 iterations)
        for _ in range(64):
            note_kernel_stat("components.propagate")
            if _launch_propagate_labels(d_board, d_labels, width, height):
                break

        # Phase 3: GPU label compaction (collect label/x/y/color tuples on GPU,
        # then sort and build component metadata on CPU from compacted data)
        raw = _launch_compact_labels(d_board, d_labels, width, height)
        note_kernel_stat("components.compact")
        if not raw:
            # Fall back to CPU compaction if GPU returned empty
            labels = cp.asnumpy(d_labels)
            raw = _compact_labels_cpu(labels, board, width, height)

        # Build colour frequencies (excluding bg_color and negative values)
        color_freq: dict[int, int] = {}
        for entry in raw:
            c = entry["color"]
            if c != bg_color and c >= 0:
                color_freq[c] = color_freq.get(c, 0) + entry["size"]

        return {
            "components": raw,
            "color_frequencies": color_freq,
        }

    except Exception as exc:
        logger.debug("CUDA component detection failed: %s", exc)
        return None
