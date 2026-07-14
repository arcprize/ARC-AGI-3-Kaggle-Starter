"""CUDA-accelerated computation module for ARC-AGI-3.

The runtime uses CuPy when available and falls back to CPU behavior otherwise.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from pathlib import Path
import tempfile
from typing import Any

logger = logging.getLogger(__name__)

CUDA_AVAILABLE: bool = False
CUDA_DEVICE_NAME: str = ""
CUDA_DEVICE_ID: int = 0
_CUPY_MODULE: Any | None = None
_KERNEL_STATS: Counter[str] = Counter()
_KERNEL_CAPABILITY_NAMES = frozenset(
    {
        "salience",
        "astar",
        "components",
        "grid_scale",
        "mirror_placement",
        "reflection_scoring",
    }
)


def note_kernel_stat(name: str, value: int = 1) -> None:
    if not name or value == 0:
        return
    _KERNEL_STATS[name] += int(value)


def snapshot_kernel_stats(reset: bool = False) -> dict[str, int]:
    snapshot = dict(_KERNEL_STATS)
    if reset:
        _KERNEL_STATS.clear()
    return snapshot


def format_kernel_stats(reset: bool = False) -> str:
    snapshot = snapshot_kernel_stats(reset=reset)
    if not snapshot:
        return "none"
    return ", ".join(f"{name}={snapshot[name]}" for name in sorted(snapshot))


def get_cupy() -> Any:
    global _CUPY_MODULE
    if _CUPY_MODULE is not None:
        return _CUPY_MODULE
    cache_dir = Path.cwd() / ".cupy_cache"
    temp_dir = Path.cwd() / ".cupy_tmp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUPY_CACHE_DIR", str(cache_dir))
    os.environ.setdefault("CUPY_CACHE_IN_MEMORY", "1")
    os.environ.setdefault("TMP", str(temp_dir))
    os.environ.setdefault("TEMP", str(temp_dir))
    os.environ.setdefault("TMPDIR", str(temp_dir))
    import cupy as cp

    _CUPY_MODULE = cp
    return cp


def init_cuda(device_id: int = 0) -> bool:
    """Probe for a CUDA-capable GPU and initialise CuPy."""
    global CUDA_AVAILABLE, CUDA_DEVICE_NAME, CUDA_DEVICE_ID
    try:
        cp = get_cupy()
        cp.cuda.Device(device_id).use()
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        raw_name = props.get("name", "")
        if isinstance(raw_name, bytes):
            CUDA_DEVICE_NAME = raw_name.decode("utf-8", "ignore")
        else:
            CUDA_DEVICE_NAME = str(raw_name)
        CUDA_DEVICE_ID = int(device_id)
        CUDA_AVAILABLE = bool(cp.cuda.is_available())
        if CUDA_AVAILABLE:
            logger.info("CUDA initialised via CuPy: device=%s id=%d", CUDA_DEVICE_NAME, CUDA_DEVICE_ID)
    except ImportError:
        logger.debug("CUDA unavailable: cupy not installed")
        CUDA_AVAILABLE = False
    except Exception as exc:
        logger.debug("CUDA unavailable: %s", exc)
        CUDA_AVAILABLE = False
    return CUDA_AVAILABLE


def is_enabled(config: object | None = None) -> bool:
    """Check whether CUDA is available AND enabled via config / env."""
    if not CUDA_AVAILABLE:
        return False
    if config is not None:
        return bool(getattr(config, "cuda_enabled", False))
    return os.environ.get("AGI_CUDA_ENABLE", "1").strip().lower() in {"1", "true", "yes", "on"}


def kernel_available(
    capability: str,
    config: object | None = None,
) -> bool:
    """Return whether a specific CUDA runtime capability should be used."""
    if capability not in _KERNEL_CAPABILITY_NAMES:
        return False
    return is_enabled(config)


def detect_components_gpu(
    board: list[list[int]],
    bg_color: int = -1,
    config: object | None = None,
) -> dict | None:
    from agi_core.cuda.components import detect_components_gpu as _dcg

    return _dcg(board, bg_color=bg_color, config=config)


def detect_grid_scale_gpu(
    board: list[list[int]],
    config: object | None = None,
) -> int | None:
    from agi_core.cuda.grid_scale import detect_grid_scale_gpu as _dgsg

    return _dgsg(board, config=config)


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
) -> tuple | None:
    from agi_core.cuda.mirror_solver import best_reflection_placement_gpu as _brpg

    return _brpg(
        width=width,
        height=height,
        component_cells=component_cells,
        component_bbox=component_bbox,
        remaining_goals=remaining_goals,
        horizontal=horizontal,
        axis=axis,
        config=config,
    )
