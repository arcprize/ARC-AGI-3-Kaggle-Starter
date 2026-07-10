"""Unified GPU-accelerated perception pipeline.

Uploads the board to GPU once, runs connected-component labeling and
grid scale detection in a single round-trip, then returns results in the
formats expected by both ``analyzer.py`` and ``shape_recognition.py``.

Every function has a CPU fallback — CUDA is only used when both the
``AGI_CUDA_ENABLE`` env var is set AND a compatible GPU is detected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agi_core.cuda import (
    CUDA_AVAILABLE,
    is_enabled,
    detect_components_gpu,
    detect_grid_scale_gpu,
)
from agi_core.orch_or.game_model import Component

logger = logging.getLogger(__name__)


def _component_confidence_from_gpu_entry(
    entry: dict[str, Any],
    total_pixels: int,
    color_freq: dict[int, int],
) -> float:
    score = 1.0
    size = int(entry["size"])
    if size < 3:
        score -= 0.2
    elif size > 500:
        score -= 0.3
    if bool(entry["touches_border"]):
        score -= 0.1
    x1, y1, x2, y2 = entry["bbox"]
    width = x2 - x1 + 1
    height = y2 - y1 + 1
    if width * height > 0 and size / (width * height) < 0.5:
        score -= 0.15
    freq = int(color_freq.get(int(entry["color"]), 0))
    rarity = 1.0 - min(1.0, freq / max(1, total_pixels))
    score *= 0.5 + 0.5 * rarity
    return max(0.1, min(1.0, score))


@dataclass(slots=True)
class PerceptionResult:
    """Unified result from the GPU perception pipeline.

    Both ``analyzer.py`` and ``shape_recognition.py`` can consume this.
    """
    analyzer_components: list[Component] = field(default_factory=list)
    grid_scale: int = 1
    color_frequencies: dict[int, int] = field(default_factory=dict)


def gpu_perceive(
    board: list[list[int]],
    bg_color: int = -1,
    config: object | None = None,
) -> PerceptionResult | None:
    """Run the entire perception pipeline on the GPU in one round-trip.

    Parameters
    ----------
    board : 64×64 int grid.
    bg_color : background colour to exclude from colour frequencies.
    config : optional config with a ``cuda_enabled`` attribute.

    Returns
    -------
    A ``PerceptionResult`` or *None* if CUDA is unavailable / fails.
    """
    if not is_enabled(config) or not CUDA_AVAILABLE:
        return None

    try:
        # 1. Component detection (GPU CCL) — already unified
        comp_result = detect_components_gpu(board, bg_color=bg_color)
        if comp_result is None:
            return None

        raw_components = comp_result["components"]
        color_freq = comp_result["color_frequencies"]
        total_pixels = len(board) * (len(board[0]) if board and board[0] else 0)

        # 2. Build analyzer-style Component list
        analyzer_comps = [
            Component(
                cells=entry["cells"],
                bbox=entry["bbox"],
                size=entry["size"],
                dominant_color=entry["color"],
                touches_border=entry["touches_border"],
                confidence=_component_confidence_from_gpu_entry(
                    entry,
                    total_pixels=total_pixels,
                    color_freq=color_freq,
                ),
            )
            for entry in raw_components
            if entry["color"] != bg_color and entry["color"] >= 0
        ]

        # 3. Grid scale detection (GPU)
        grid_scale = detect_grid_scale_gpu(board)
        if grid_scale is None:
            from agi_core.orch_or.analyzer import _detect_grid_scale
            grid_scale = _detect_grid_scale(board)

        return PerceptionResult(
            analyzer_components=analyzer_comps,
            grid_scale=grid_scale,
            color_frequencies=color_freq,
        )

    except Exception as exc:
        logger.debug("GPU perception pipeline failed: %s", exc)
        return None
