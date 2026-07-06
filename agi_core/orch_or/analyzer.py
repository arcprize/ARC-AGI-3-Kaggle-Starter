from __future__ import annotations

from collections import Counter, deque

from agi_core.orch_or.game_model import Component, GameFeatures, GameModel
from agi_core.orch_or.search import astar

_PLATFORMER_THRESHOLD = 5


def build_game_model(board: list[list[int]]) -> GameModel:
    height = len(board)
    width = len(board[0]) if height > 0 else 0
    bg_color = _detect_background_color(board, width, height)
    raw_components = _find_components(board, width, height, bg_color)
    components = _merge_nearby_same_color(raw_components)
    color_freq = _compute_color_frequencies(board, bg_color)
    player_pos = _find_player(components, height, color_freq)
    goal_positions = _find_goals(components, player_pos, board, color_freq)
    obstacles = _find_obstacles(components, player_pos, goal_positions)
    walkable = _find_walkable(board, width, height, bg_color, obstacles, player_pos, goal_positions, components)
    features = _extract_features(board, width, height, bg_color, components, player_pos, goal_positions, walkable)
    return GameModel(
        board=board,
        height=height,
        width=width,
        player_pos=player_pos,
        goal_positions=goal_positions,
        obstacles=obstacles,
        components=components,
        walkable=walkable,
        features=features,
    )


def _detect_background_color(board: list[list[int]], width: int, height: int) -> int:
    border_pixels: list[int] = []
    if height > 0:
        if width > 0:
            border_pixels.extend(board[0])
            border_pixels.extend(board[-1])
    for y in range(1, height - 1):
        if width > 0:
            border_pixels.append(board[y][0])
            border_pixels.append(board[y][-1])
    if not border_pixels:
        return -1
    counts = Counter(border_pixels)
    return counts.most_common(1)[0][0]



_GOAL_COLOR_CANDIDATES: set[int] = {11}
_PLATFORMER_COMPONENT_THRESHOLD: int = 5


def _find_components(
    board: list[list[int]], width: int, height: int, bg_color: int,
) -> list[Component]:
    visited: set[tuple[int, int]] = set()
    components: list[Component] = []
    for y in range(height):
        for x in range(width):
            if (x, y) in visited:
                continue
            val = board[y][x]
            if val == bg_color or val < 0:
                visited.add((x, y))
                continue
            cells: list[tuple[int, int]] = []
            queue = deque([(x, y)])
            visited.add((x, y))
            min_x, min_y, max_x, max_y = x, y, x, y
            while queue:
                cx, cy = queue.popleft()
                cells.append((cx, cy))
                min_x = min(min_x, cx)
                min_y = min(min_y, cy)
                max_x = max(max_x, cx)
                max_y = max(max_y, cy)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                        nv = board[ny][nx]
                        if nv != bg_color and nv >= 0 and nv == val:
                            visited.add((nx, ny))
                            queue.append((nx, ny))
            touches_border = (min_x == 0 or min_y == 0 or max_x == width - 1 or max_y == height - 1)
            dominant_color = val
            components.append(Component(
                cells=cells,
                bbox=(min_x, min_y, max_x, max_y),
                size=len(cells),
                dominant_color=dominant_color,
                touches_border=touches_border,
            ))
    return components


def _compute_color_frequencies(board: list[list[int]], bg_color: int) -> dict[int, int]:
    freq: dict[int, int] = {}
    for row in board:
        for v in row:
            if v != bg_color and v >= 0:
                freq[v] = freq.get(v, 0) + 1
    return freq


def _merge_nearby_same_color(components: list[Component]) -> list[Component]:
    if not components:
        return []
    width = max(c.bbox[2] for c in components) + 1 if components else 0
    height = max(c.bbox[3] for c in components) + 1 if components else 0
    threshold = max(3, min(width, height) // 12)
    by_color: dict[int, list[Component]] = {}
    for c in components:
        by_color.setdefault(c.dominant_color, []).append(c)
    result: list[Component] = []
    for color, group in by_color.items():
        if len(group) <= 1:
            result.extend(group)
            continue
        merged = []
        used = [False] * len(group)
        for i in range(len(group)):
            if used[i]:
                continue
            cxi = sum(p[0] for p in group[i].cells) // len(group[i].cells)
            cyi = sum(p[1] for p in group[i].cells) // len(group[i].cells)
            merged_cells = set(group[i].cells)
            used[i] = True
            for j in range(i + 1, len(group)):
                if used[j]:
                    continue
                cxj = sum(p[0] for p in group[j].cells) // len(group[j].cells)
                cyj = sum(p[1] for p in group[j].cells) // len(group[j].cells)
                if abs(cxi - cxj) + abs(cyi - cyj) <= threshold:
                    merged_cells.update(group[j].cells)
                    used[j] = True
            cells = sorted(merged_cells)
            min_x = min(x for x, y in cells)
            min_y = min(y for x, y in cells)
            max_x = max(x for x, y in cells)
            max_y = max(y for x, y in cells)
            color_counts: dict[int, int] = {}
            for cx, cy in cells:
                found = False
                for c in group:
                    if (cx, cy) in c.cells:
                        found = True
                        break
                color_counts[color] = color_counts.get(color, 0) + 1
            merged.append(Component(
                cells=cells,
                bbox=(min_x, min_y, max_x, max_y),
                size=len(cells),
                dominant_color=color,
                touches_border=(
                    min_x == 0 or min_y == 0
                    or max_x >= width - 1 or max_y >= height - 1
                ),
            ))
        result.extend(merged)
    return result


def _find_player(
    components: list[Component],
    height: int,
    color_freq: dict[int, int] | None = None,
) -> tuple[int, int] | None:
    candidates = [c for c in components if not c.touches_border and 1 <= c.size <= 80]
    if not candidates:
        return None
    total_pixels = sum(color_freq.values()) if color_freq else 1
    if not color_freq:
        lower_half = [c for c in candidates if c.bbox[1] >= height // 2]
        if lower_half:
            result = max(lower_half, key=lambda c: c.bbox[3])
            return _component_center(result)
        result = max(candidates, key=lambda c: c.size)
        cx, cy = _component_center(result)
        return (int(cx), int(cy))
    color_0 = color_freq.get(0, 0)
    if color_0 > 0:
        color_0_comps = [c for c in candidates if c.dominant_color == 0]
        if color_0_comps:
            return _component_center(max(color_0_comps, key=lambda c: c.size))
    best_score = -1.0
    best_comp = candidates[0]
    for c in candidates:
        cy = c.bbox[1] + (c.bbox[3] - c.bbox[1]) // 2
        freq = color_freq.get(c.dominant_color, 1)
        rarity = max(0.01, 1.0 - freq / total_pixels)
        cy_norm = cy / max(height, 1)
        size_norm = min(c.size / 80.0, 1.0)
        score = rarity * 2.0 + cy_norm * 1.5 + size_norm * 0.5
        if score > best_score:
            best_score = score
            best_comp = c
    cx, cy = _component_center(best_comp)
    return (int(cx), int(cy))


def _find_goals(
    components: list[Component],
    player_pos: tuple[int, int] | None,
    board: list[list[int]] | None = None,
    color_freq: dict[int, int] | None = None,
) -> list[tuple[int, int]]:
    candidates = [c for c in components if not c.touches_border and c.size >= 1]
    if player_pos is not None:
        px, py = player_pos
        candidates = [c for c in candidates if not _in_bbox(player_pos, c.bbox)]
    if not candidates:
        return []
    player_color = -1
    if player_pos is not None and board is not None:
        px, py = player_pos
        if 0 <= py < len(board) and 0 <= px < len(board[0]):
            player_color = board[py][px]
    total_non_bg = sum(color_freq.values()) if color_freq else 1
    scored: list[tuple[float, Component]] = []
    for c in candidates:
        cx, cy = _component_center(c)
        score = 0.0
        if c.dominant_color != player_color:
            score += 2.0
        else:
            score -= 2.0
        if 3 <= c.size <= 20:
            score += 1.5
        elif c.size <= 60:
            score += 1.0
        else:
            score -= 1.0
        if player_pos is not None:
            px, py = player_pos
            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            score += dist / 100.0
        if color_freq:
            freq = color_freq.get(c.dominant_color, 0)
            freq_ratio = freq / total_non_bg if total_non_bg > 0 else 0
            if freq_ratio < 0.02:
                score -= 2.0
            elif freq_ratio < 0.15:
                score += 0.5
            elif freq_ratio < 0.30:
                score += 1.0
            else:
                score -= 0.5
        scored.append((score, c))
    scored.sort(key=lambda x: -x[0])
    n_goals = max(1, min(len(scored) // 2, 3))
    primary_goals = [_component_center(c) for _, c in scored[:n_goals]]
    goal_color_candidates = _GOAL_COLOR_CANDIDATES
    goal_colored_in_top = [c for _, c in scored[:n_goals] if c.dominant_color in goal_color_candidates]
    if goal_colored_in_top:
        primary_goals = [_component_center(c) for c in goal_colored_in_top]
    elif player_pos is not None and board and board[0]:
        h, w = len(board), len(board[0])
        bg = _detect_background_color(board, w, h)
        simple_walk = {(x, y) for y in range(h) for x in range(w) if board[y][x] == bg}
        px, py = player_pos
        reachable = []
        for gx, gy in primary_goals:
            passable = simple_walk | {player_pos, (gx, gy)}
            path = astar(player_pos, (gx, gy), passable, w, h, max_steps=1000)
            if path is not None:
                reachable.append((gx, gy))
        if not reachable and scored:
            fallback = sorted([c for _, c in scored if not (_in_bbox(player_pos, c.bbox) if player_pos else False)], key=lambda c: abs(c.bbox[0] - px) + abs(c.bbox[1] - py))
            if fallback:
                return [_component_center(fallback[0])]
    return primary_goals


def _in_bbox(pos: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    x, y = pos
    min_x, min_y, max_x, max_y = bbox
    return min_x <= x <= max_x and min_y <= y <= max_y


def _component_center(c: Component) -> tuple[int, int]:
    xs = [p[0] for p in c.cells]
    ys = [p[1] for p in c.cells]
    return (sum(xs) // len(xs), sum(ys) // len(ys))


def _find_obstacles(
    components: list[Component],
    player_pos: tuple[int, int] | None = None,
    goal_positions: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int, str]]:
    skip_positions: set[tuple[int, int]] = set()
    if player_pos is not None:
        skip_positions.add(player_pos)
    if goal_positions:
        skip_positions.update(goal_positions)
    obstacles: list[tuple[int, int, str]] = []
    player_comps = set()
    if player_pos is not None:
        for c in components:
            if _in_bbox(player_pos, c.bbox) and not c.touches_border:
                player_comps.add(id(c))
    for c in components:
        cx, cy = _component_center(c)
        if (cx, cy) in skip_positions:
            continue
        if id(c) in player_comps:
            continue
        obs_type = "wall" if c.touches_border else "interactive"
        obstacles.append((cx, cy, obs_type))
    return obstacles


def _find_walkable(
    board: list[list[int]],
    width: int, height: int,
    bg_color: int,
    obstacles: list[tuple[int, int, str]],
    player_pos: tuple[int, int] | None = None,
    goal_positions: list[tuple[int, int]] | None = None,
    components: list[Component] | None = None,
) -> set[tuple[int, int]]:
    passable: set[tuple[int, int]] = set()
    obs_set = set((ox, oy) for ox, oy, otype in obstacles if otype == "wall")
    player_color = board[player_pos[1]][player_pos[0]] if player_pos else -1
    goal_colors = set()
    if goal_positions:
        for gx, gy in goal_positions:
            if 0 <= gy < height and 0 <= gx < width:
                goal_colors.add(board[gy][gx])
    walkable_colors = {bg_color, 0}
    if player_color >= 0:
        walkable_colors.add(player_color)
    walkable_colors.update(goal_colors)
    for y in range(height):
        for x in range(width):
            val = board[y][x]
            if val in walkable_colors and (x, y) not in obs_set:
                passable.add((x, y))
    return passable


def _detect_beam(
    components: list[Component],
    goal_positions: list[tuple[int, int]],
    board: list[list[int]],
    player_pos: tuple[int, int] | None,
    bg_color: int,
) -> tuple[bool, int, list[tuple[int, int]], int, float, int]:
    beam_color_candidates = [4]
    beam_comp = None
    for c in components:
        if c.dominant_color in beam_color_candidates and not c.touches_border:
            beam_comp = c
            break
    if beam_comp is None:
        return (False, -1, [], 0, float('inf'), -1)
    beam_cells_set = set(beam_comp.cells)
    goals_lit = 0
    for gx, gy in goal_positions:
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = gx + dx, gy + dy
            if (nx, ny) in beam_cells_set:
                goals_lit += 1
                break
    beam_centers = []
    min_dist = float('inf')
    mirror_axis = -1
    if beam_comp.cells:
        n = max(1, len(beam_comp.cells) // 3)
        step = max(1, len(beam_comp.cells) // n)
        for i in range(0, len(beam_comp.cells), step):
            beam_centers.append(beam_comp.cells[i])
        if goal_positions:
            min_dist = min(
                min(abs(bx - gx) + abs(by - gy) for gx, gy in goal_positions)
                for bx, by in beam_comp.cells
            )
        mirror_axis = _infer_mirror_axis(beam_comp, player_pos, board)
    return (True, beam_comp.dominant_color, beam_centers[:5], goals_lit, min_dist, mirror_axis)


def _infer_mirror_axis(
    beam_comp: Component,
    player_pos: tuple[int, int] | None,
    board: list[list[int]],
) -> int:
    if player_pos is None:
        return -1
    px, py = player_pos
    best_dist = float('inf')
    best_bx = -1
    for bx, by in beam_comp.cells:
        d = abs(bx - px) + abs(by - py)
        if d < best_dist:
            best_dist = d
            best_bx = bx
    if best_bx < 0:
        return -1
    axis = (px + best_bx + 1) // 2
    if 0 <= axis < (len(board[0]) if board else 64):
        return axis
    return -1


def _extract_features(
    board: list[list[int]],
    width: int, height: int,
    bg_color: int,
    components: list[Component],
    player_pos: tuple[int, int] | None,
    goal_positions: list[tuple[int, int]],
    walkable: set[tuple[int, int]] | None = None,
) -> GameFeatures:
    path_possible = False
    if player_pos is not None and goal_positions and walkable:
        a_star_passable = walkable | {player_pos} | set(goal_positions)
        path = astar(player_pos, goal_positions[0], a_star_passable, width, height)
        path_possible = path is not None
    board_type = _classify_board(components, player_pos)
    has_beam, beam_color, beam_positions, goals_lit, beam_min_dist, mirror_axis = _detect_beam(
        components, goal_positions, board, player_pos, bg_color,
    )
    features = GameFeatures(
        background_color=bg_color,
        player_detectable=player_pos is not None,
        goal_detectable=len(goal_positions) > 0,
        board_type=board_type,
        tile_size=_detect_tile_size(board, width, height, bg_color),
        path_to_goal_possible=path_possible,
        has_beam=has_beam,
        beam_color=beam_color,
        beam_positions=beam_positions,
        goals_lit_by_beam=goals_lit,
        beam_min_dist_to_goal=beam_min_dist,
        mirror_axis=mirror_axis,
    )
    features.has_gravity = board_type == "platformer"
    if player_pos is not None:
        player_comp = _find_player_component(components, player_pos)
        if player_comp is not None and player_comp.dominant_color != 0:
            features.player_color = player_comp.dominant_color
        else:
            features.player_color = board[player_pos[1]][player_pos[0]]
    for gx, gy in goal_positions:
        if 0 <= gy < height and 0 <= gx < width:
            features.goal_color = board[gy][gx]
            break
    if player_pos is not None and goal_positions:
        px, py = player_pos
        all_y = [gy for _, gy in goal_positions]
        all_x = [gx for gx, _ in goal_positions]
        axes: set[str] = set()
        if any(abs(gx - px) > 0 for gx in all_x):
            axes.add("x")
        if any(abs(gy - py) > 0 for gy in all_y):
            axes.add("y")
        if axes:
            features.movement_axes = axes
    player_region = [c for c in components if not c.touches_border and 1 <= c.size <= 80]
    top_colors = set()
    for c in player_region:
        if player_pos is None or not _in_bbox(player_pos, c.bbox):
            top_colors.add(c.dominant_color)
    if len(top_colors) >= 2:
        features.has_click_interaction = True
        features.has_breakable_blocks = True
    return features


def _find_player_component(components: list[Component], player_pos: tuple[int, int]) -> Component | None:
    candidates = [c for c in components if _in_bbox(player_pos, c.bbox)]
    if not candidates:
        return None
    non_zero = [c for c in candidates if c.dominant_color != 0]
    if non_zero:
        return max(non_zero, key=lambda c: c.size)
    return max(candidates, key=lambda c: c.size)


def _classify_board(components: list[Component], player_pos: tuple[int, int] | None = None) -> str:
    non_border = [c for c in components if not c.touches_border]
    if player_pos is not None:
        non_border = [c for c in non_border if not _in_bbox(player_pos, c.bbox)]
    if len(non_border) <= _PLATFORMER_THRESHOLD:
        return "platformer"
    return "puzzle"


def _detect_tile_size(board: list[list[int]], width: int, height: int, bg_color: int) -> int:
    detected = _detect_scale_from_frame(board, width, height, bg_color)
    if detected and detected > 1:
        return detected
    candidates = [6, 8, 10, 12, 16, 32]
    for tile_size in candidates:
        if width % tile_size == 0 and height % tile_size == 0:
            return tile_size
    return 6


def _detect_scale_from_frame(board: list[list[int]], width: int, height: int, bg_color: int) -> int:
    step_counts: dict[int, int] = {}
    for y in range(height):
        prev = -1
        for x in range(width):
            if board[y][x] != bg_color and board[y][x] >= 0:
                if prev >= 0:
                    gap = x - prev
                    if 2 <= gap <= 8:
                        step_counts[gap] = step_counts.get(gap, 0) + 1
                prev = x
    for x in range(width):
        prev = -1
        for y in range(height):
            if board[y][x] != bg_color and board[y][x] >= 0:
                if prev >= 0:
                    gap = y - prev
                    if 2 <= gap <= 8:
                        step_counts[gap] = step_counts.get(gap, 0) + 1
                prev = y
    if not step_counts:
        return 1
    best_gap = max(step_counts, key=step_counts.get)
    best_count = step_counts[best_gap]
    total = sum(step_counts.values())
    if best_count < 5 or best_count < 0.5 * total:
        return 1
    return best_gap
