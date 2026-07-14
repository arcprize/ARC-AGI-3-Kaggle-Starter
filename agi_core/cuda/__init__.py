"""CUDA-accelerated computation module for ARC-AGI-3.

Provides GPU kernels for:
- **Salience map** computation (attention system)
- **Batch A* pathfinding** (multi-agent navigation)
- **Connected-component labeling** (perception — ``detect_components_gpu``)

Every function has a CPU fallback — CUDA is only used when both the
``AGI_CUDA_ENABLE`` environment flag is set AND a compatible GPU is detected.
"""

import logging
import os

logger = logging.getLogger(__name__)

CUDA_AVAILABLE: bool = False
CUDA_DEVICE_NAME: str = ""
CUDA_DEVICE_ID: int = 0
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


def detect_components_gpu(
    board: list[list[int]],
    bg_color: int = -1,
    config: object | None = None,
) -> dict | None:
    """Re-exported for convenience; see ``components.detect_components_gpu``."""
    from agi_core.cuda.components import detect_components_gpu as _dcg
    return _dcg(board, bg_color=bg_color, config=config)


def detect_grid_scale_gpu(
    board: list[list[int]],
    config: object | None = None,
) -> int | None:
    """Re-exported for convenience; see ``grid_scale.detect_grid_scale_gpu``."""
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
    """Re-exported for convenience; see ``mirror_solver.best_reflection_placement_gpu``."""
    from agi_core.cuda.mirror_solver import best_reflection_placement_gpu as _brpg
    return _brpg(width=width, height=height,
                 component_cells=component_cells,
                 component_bbox=component_bbox,
                 remaining_goals=remaining_goals,
                 horizontal=horizontal, axis=axis,
                 config=config)


def init_cuda(device_id: int = 0) -> bool:
    """Probe for a CUDA-capable GPU and initialise numba.cuda.
    
    Returns True if CUDA is available and ready.
    """
    global CUDA_AVAILABLE, CUDA_DEVICE_NAME, CUDA_DEVICE_ID
    try:
        from numba import cuda
        cuda.select_device(device_id)
        CUDA_DEVICE_ID = device_id
        CUDA_DEVICE_NAME = cuda.get_current_device().name
        # Warm-up: touch the device so import-time errors surface immediately
        cuda.is_available()
        CUDA_AVAILABLE = True
        logger.info("CUDA initialised: device=%s id=%d", CUDA_DEVICE_NAME, CUDA_DEVICE_ID)
    except ImportError:
        logger.debug("CUDA unavailable: numba.cuda not installed")
    except Exception as exc:
        logger.debug("CUDA unavailable: %s", exc)
    return CUDA_AVAILABLE


def is_enabled(config: object | None = None) -> bool:
    """Check whether CUDA is available AND enabled via config / env."""
    if not CUDA_AVAILABLE:
        return False
    if config is not None:
        return bool(getattr(config, 'cuda_enabled', False))
    return os.environ.get("AGI_CUDA_ENABLE", "1").strip().lower() in {"1", "true", "yes", "on"}


def kernel_available(
    capability: str,
    config: object | None = None,
) -> bool:
    """Return whether a specific CUDA runtime capability should be used.

    This is intentionally conservative: any unknown capability name returns
    False so runtime callers can fall back to CPU without widening failures.
    """
    if capability not in _KERNEL_CAPABILITY_NAMES:
        return False
    return is_enabled(config)
