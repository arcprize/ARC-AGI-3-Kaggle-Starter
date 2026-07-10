"""CUDA-accelerated computation module for ARC-AGI-3.

Provides GPU kernels for:
- **Salience map** computation (attention system)
- **Batch A* pathfinding** (multi-agent navigation)

Every function has a CPU fallback — CUDA is only used when both the
``AGI_CUDA_ENABLE`` environment flag is set AND a compatible GPU is detected.
"""

import logging
import os

logger = logging.getLogger(__name__)

CUDA_AVAILABLE: bool = False
CUDA_DEVICE_NAME: str = ""
CUDA_DEVICE_ID: int = 0


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
    return os.environ.get("AGI_CUDA_ENABLE", "0").strip().lower() in {"1", "true", "yes", "on"}
