"""GPU-accelerated batch A* pathfinding with CPU fallback."""

import logging

import numpy as np

import agi_core.cuda as cuda_runtime
from agi_core.cuda import kernel_available, note_kernel_stat
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
    """Run batch A* on the GPU for multiple agents."""
    if not kernel_available("astar", config) or not cuda_runtime.CUDA_AVAILABLE:
        return _fallback_cpu(
            walkable_sets,
            start_positions,
            goal_positions,
            width=width,
            height=height,
        )

    n = len(start_positions)
    if n == 0:
        return []

    try:
        walkable_np = np.zeros((n, height, width), dtype=np.uint8)
        for index, walkable in enumerate(walkable_sets):
            for x, y in walkable:
                if 0 <= x < width and 0 <= y < height:
                    walkable_np[index, y, x] = 1

        starts_x = np.array([start[0] for start in start_positions], dtype=np.int32)
        starts_y = np.array([start[1] for start in start_positions], dtype=np.int32)
        goals_x = np.array([goal[0] for goal in goal_positions], dtype=np.int32)
        goals_y = np.array([goal[1] for goal in goal_positions], dtype=np.int32)
        paths_out = np.full((n, 256), -1, dtype=np.int32)
        path_lens = np.zeros(n, dtype=np.int32)

        _launch_astar_kernel(
            walkable_np,
            starts_x,
            starts_y,
            goals_x,
            goals_y,
            paths_out,
            path_lens,
        )
        note_kernel_stat("astar")

        results: list[list[tuple[int, int]] | None] = []
        for index in range(n):
            path_len = int(path_lens[index])
            if path_len <= 0:
                results.append(None)
                continue
            path: list[tuple[int, int]] = []
            for step in range(path_len):
                cell_index = int(paths_out[index, step])
                if cell_index < 0:
                    break
                path.append((cell_index % width, cell_index // width))
            results.append(path or None)
        return results
    except Exception as exc:
        logger.debug("CUDA A* failed: %s", exc)
        return _fallback_cpu(
            walkable_sets,
            start_positions,
            goal_positions,
            width=width,
            height=height,
        )


def _fallback_cpu(
    walkable_sets: list[set[tuple[int, int]]],
    start_positions: list[tuple[int, int]],
    goal_positions: list[tuple[int, int]],
    *,
    width: int,
    height: int,
) -> list[list[tuple[int, int]] | None]:
    """CPU fallback: run iterative A* for each search."""
    from agi_core.orch_or.search import _astar_impl

    results: list[list[tuple[int, int]] | None] = []
    for walkable, start, goal in zip(walkable_sets, start_positions, goal_positions):
        if start not in walkable or goal not in walkable:
            results.append(None)
            continue
        if start == goal:
            results.append([start])
            continue
        results.append(_astar_impl(start, goal, walkable, width, height))
    return results
