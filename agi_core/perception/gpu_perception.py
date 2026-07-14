"""Unified GPU-accelerated perception pipeline."""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from agi_core.cuda import (
    CUDA_AVAILABLE,
    detect_components_gpu,
    detect_grid_scale_gpu,
    kernel_available,
)
from agi_core.orch_or.game_model import Component

logger = logging.getLogger(__name__)
_PERCEPTION_CACHE_LIMIT = 4
_PERCEPTION_CACHE: OrderedDict[tuple[str, int, int, int], "PerceptionResult"] = OrderedDict()


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


def _board_cache_key(board: list[list[int]], bg_color: int) -> tuple[str, int, int, int]:
    height = len(board)
    width = len(board[0]) if board and board[0] else 0
    payload = bytearray()
    for row in board:
        for value in row:
            payload.extend(int(value).to_bytes(2, "little", signed=True))
    digest = hashlib.blake2b(payload, digest_size=16).hexdigest()
    return (digest, width, height, int(bg_color))


def _remember_perception(key: tuple[str, int, int, int], result: "PerceptionResult") -> "PerceptionResult":
    _PERCEPTION_CACHE[key] = result
    _PERCEPTION_CACHE.move_to_end(key)
    while len(_PERCEPTION_CACHE) > _PERCEPTION_CACHE_LIMIT:
        _PERCEPTION_CACHE.popitem(last=False)
    return result


def get_cached_perception(
    board: list[list[int]],
    *,
    bg_color: int = -1,
) -> "PerceptionResult | None":
    if not CUDA_AVAILABLE:
        return None
    key = _board_cache_key(board, bg_color)
    cached = _PERCEPTION_CACHE.get(key)
    if cached is None:
        return None
    _PERCEPTION_CACHE.move_to_end(key)
    return cached


@dataclass(slots=True)
class PerceptionResult:
    analyzer_components: list[Component] = field(default_factory=list)
    grid_scale: int = 1
    color_frequencies: dict[int, int] = field(default_factory=dict)
    raw_components: list[dict[str, Any]] = field(default_factory=list)


def gpu_perceive(
    board: list[list[int]],
    bg_color: int = -1,
    config: object | None = None,
) -> PerceptionResult | None:
    """Run the perception pipeline on the GPU in one round-trip."""
    if not CUDA_AVAILABLE or not kernel_available("components", config):
        return None

    cached = get_cached_perception(board, bg_color=bg_color)
    if cached is not None:
        return cached

    try:
        comp_result = detect_components_gpu(board, bg_color=bg_color, config=config)
        if comp_result is None:
            return None

        raw_components = list(comp_result["components"])
        color_freq = dict(comp_result["color_frequencies"])
        total_pixels = len(board) * (len(board[0]) if board and board[0] else 0)

        analyzer_components = [
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

        grid_scale = detect_grid_scale_gpu(board, config=config)
        if grid_scale is None:
            from agi_core.orch_or.analyzer import _detect_grid_scale

            grid_scale = _detect_grid_scale(board)

        result = PerceptionResult(
            analyzer_components=analyzer_components,
            grid_scale=grid_scale,
            color_frequencies=color_freq,
            raw_components=raw_components,
        )
        return _remember_perception(_board_cache_key(board, bg_color), result)
    except Exception as exc:
        logger.debug("GPU perception pipeline failed: %s", exc)
        return None
