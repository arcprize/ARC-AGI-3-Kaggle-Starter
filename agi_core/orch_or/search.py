from __future__ import annotations

import heapq


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
    open_set = [(0, 0, start)]
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    g_score: dict[tuple[int, int], float] = {start: 0}

    gx, gy = goal
    sx, sy = start
    dx_goal = abs(gx - sx)
    dy_goal = abs(gy - sy)
    # Prefer the axis with larger distance
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
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
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
                # Tiebreaker: prefer direction toward goal's primary axis
                pref = 0
                if dx == 1 and prefer_right: pref = -1
                elif dx == -1 and prefer_left: pref = -1
                elif dy == 1 and prefer_down: pref = -1
                elif dy == -1 and prefer_up: pref = -1
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
