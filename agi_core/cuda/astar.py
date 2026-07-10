"""GPU-accelerated batch A* pathfinding with CPU fallback."""

import logging
from typing import Any

import numpy as np

from agi_core.cuda import CUDA_AVAILABLE, is_enabled
from agi_core.cuda.kernels import _launch_astar_kernel

logger = logging.getLogger(__name__)


def batch_astar_gpu(
    walkable_sets: list[set[tuple[int, int]]],
    start_positions: list[tuple[int, int]],
    goal_positions: list[tuple[int, int]],
    width: int = 64,
    height: int = 64,
    config: object | None = None,
) -> list[list[tuple[int, int]] | None]:
    """Run batch A* on the GPU for multiple agents.
    
    Each entry in ``walkable_sets``, ``start_positions``, ``goal_positions``
    represents one search.  Returns a list of paths (list of (x, y) tuples)
    or None for searches that found no path.
    
    Falls back to CPU if CUDA is unavailable.
    """
    if not is_enabled(config) or not CUDA_AVAILABLE:
        return _fallback_cpu(walkable_sets, start_positions, goal_positions)

    n = len(start_positions)
    if n == 0:
        return []

    try:
        # Build binary walkable raster for each search
        # (we use a single walkable raster for all — the union)
        walkable_np = np.zeros((height, width), dtype=np.uint8)
        for ws in walkable_sets:
            for x, y in ws:
                if 0 <= x < width and 0 <= y < height:
                    walkable_np[y, x] = 1

        starts_x = np.array([s[0] for s in start_positions], dtype=np.int32)
        starts_y = np.array([s[1] for s in start_positions], dtype=np.int32)
        goals_x = np.array([g[0] for g in goal_positions], dtype=np.int32)
        goals_y = np.array([g[1] for g in goal_positions], dtype=np.int32)

        paths_out = np.zeros((n, 256), dtype=np.int32)
        path_lens = np.zeros(n, dtype=np.int32)

        _launch_astar_kernel(walkable_np, starts_x, starts_y, goals_x, goals_y, paths_out, path_lens)

        # Reconstruct Python paths
        results: list[list[tuple[int, int]] | None] = []
        for i in range(n):
            length = int(path_lens[i])
            if length == 0:
                results.append(None)
            else:
                path = []
                for j in range(length):
                    idx = int(paths_out[i, j])
                    if idx < 0:
                        break
                    x = idx % width
                    y = idx // width
                    path.append((x, y))
                results.append(path)
        return results

    except Exception as exc:
        logger.debug("CUDA A* failed: %s", exc)
        return _fallback_cpu(walkable_sets, start_positions, goal_positions)


def _fallback_cpu(
    walkable_sets: list[set[tuple[int, int]]],
    start_positions: list[tuple[int, int]],
    goal_positions: list[tuple[int, int]],
) -> list[list[tuple[int, int]] | None]:
    """CPU fallback: run iterative A* for each search."""
    from agi_core.orch_or.search import astar
    results: list[list[tuple[int, int]] | None] = []
    for ws, start, goal in zip(walkable_sets, start_positions, goal_positions):
        path = astar(start, goal, ws, 64, 64)
        results.append(path)
    return results
