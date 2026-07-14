from __future__ import annotations

import heapq
import logging
from concurrent.futures import Future

from agi_core.cuda import kernel_available

logger = logging.getLogger(__name__)

# Threshold: problems with more passable cells than this run in the process
# pool (true parallelism, bypassing the GIL). Smaller problems run inline.
_PROCESS_POOL_THRESHOLD = 5000
_CUDA_ASTAR_MIN_PASSABLE = 768

# A* Future cache: {(start, goal) -> Future}.
# Populated by orchestrator's background pre-computation (process pool).
# astar() pops the Future and awaits it with a short timeout.
# Only Futures are stored (not results) to avoid stale inline results.
path_cache: dict[tuple[tuple[int, int], tuple[int, int]], Future] = {}


def should_use_process_pool(passable_count: int) -> bool:
    return int(passable_count) > _PROCESS_POOL_THRESHOLD


def astar(
    start: tuple[int, int],
    goal: tuple[int, int],
    passable: set[tuple[int, int]],
    width: int,
    height: int,
    max_steps: int = 500,
) -> list[tuple[int, int]] | None:
    if start not in passable or goal not in passable:
        return None
    if start == goal:
        return [start]

    key = (start, goal)
    cached = path_cache.pop(key, None)
    if cached is not None:
        try:
            return cached.result(timeout=0.5)
        except Exception:
            pass

    cuda_path = _astar_via_cuda(start, goal, passable, width, height)
    if cuda_path is not None:
        return cuda_path

    if should_use_process_pool(len(passable)):
        try:
            from agi_core.orch_or.parallel import ProcessPoolManager

            future = ProcessPoolManager.submit(
                _astar_impl,
                start,
                goal,
                passable,
                width,
                height,
                max_steps,
            )
            return future.result()
        except Exception:
            pass

    return _astar_impl(start, goal, passable, width, height, max_steps)


def _astar_via_cuda(
    start: tuple[int, int],
    goal: tuple[int, int],
    passable: set[tuple[int, int]],
    width: int,
    height: int,
) -> list[tuple[int, int]] | None:
    if not kernel_available("astar"):
        return None
    if width != 64 or height != 64:
        return None
    if len(passable) < _CUDA_ASTAR_MIN_PASSABLE:
        return None
    try:
        from agi_core.cuda.astar import batch_astar_gpu

        results = batch_astar_gpu(
            [passable],
            [start],
            [goal],
            width=width,
            height=height,
        )
    except Exception as exc:
        logger.debug("A* CUDA fast path unavailable: %s", exc)
        return None
    if not results:
        return None
    return results[0]


def _astar_impl(
    start: tuple[int, int],
    goal: tuple[int, int],
    passable: set[tuple[int, int]],
    width: int,
    height: int,
    max_steps: int = 500,
) -> list[tuple[int, int]] | None:
    open_set = [(0, 0, start)]
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    g_score: dict[tuple[int, int], float] = {start: 0}

    gx, gy = goal
    sx, sy = start
    dx_goal = abs(gx - sx)
    dy_goal = abs(gy - sy)
    prefer_right = gx > sx and dx_goal >= dy_goal
    prefer_left = gx < sx and dx_goal >= dy_goal
    prefer_down = gy > sy and dy_goal > dx_goal
    prefer_up = gy < sy and dy_goal > dx_goal

    while open_set:
        _, _, current = heapq.heappop(open_set)
        if current == goal:
            return _reconstruct_path(came_from, current)
        if len(g_score) > max_steps:
            return None
        cx, cy = current
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = (cx + dx, cy + dy)
            if neighbor[0] < 0 or neighbor[0] >= width or neighbor[1] < 0 or neighbor[1] >= height:
                continue
            if neighbor not in passable:
                continue
            tentative = g_score[current] + 1
            if neighbor not in g_score or tentative < g_score[neighbor]:
                g_score[neighbor] = tentative
                h = _manhattan(neighbor, goal)
                priority = tentative + h
                pref = 0
                if dx == 1 and prefer_right:
                    pref = -1
                elif dx == -1 and prefer_left:
                    pref = -1
                elif dy == 1 and prefer_down:
                    pref = -1
                elif dy == -1 and prefer_up:
                    pref = -1
                heapq.heappush(open_set, (priority, pref, neighbor))
                came_from[neighbor] = current
    return None


def _reconstruct_path(
    came_from: dict[tuple[int, int], tuple[int, int] | None],
    current: tuple[int, int],
) -> list[tuple[int, int]]:
    path = [current]
    while came_from.get(current) is not None:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
