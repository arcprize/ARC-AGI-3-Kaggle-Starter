from __future__ import annotations

from collections import Counter, deque

from agi_core.orch_or.game_model import Component, DynamicsCategory, GameFeatures, GameModel, SpriteInfo
from agi_core.orch_or.search import astar

_PLATFORMER_THRESHOLD = 5


def build_game_model(board: list[list[int]], attention=None) -> GameModel:
    height = len(board)
    width = len(board[0]) if height > 0 else 0
    bg_color = _detect_background_color(board, width, height)
    # Phase 1: parallel detection — components, grid scale, color frequencies all independent
    try:
        from agi_core.orch_or.parallel import ThreadPoolManager
        pool = ThreadPoolManager.executor()
        comp_future = pool.submit(_find_components, board, width, height, bg_color)
        scale_future = pool.submit(_detect_grid_scale, board, width, height)
        freq_future = pool.submit(_compute_color_frequencies, board, bg_color)
        raw_components = comp_future.result()
        grid_scale = scale_future.result()
        color_freq = freq_future.result()
    except Exception:
        raw_components = _find_components(board, width, height, bg_color)
        grid_scale = _detect_grid_scale(board, width, height)
        color_freq = _compute_color_frequencies(board, bg_color)

    components = _merge_nearby_same_color(raw_components)
    player_pos = _find_player(components, height, color_freq)
    goal_positions = _find_goals(components, player_pos, board, color_freq)
    obstacles = _find_obstacles(components, player_pos, goal_positions)

    # Phase 2: walkable + sprites + tile_size are independent — run in parallel
    # Use attention salience to skip low-salience regions in walkable scan.
    try:
        pool = ThreadPoolManager.executor()
        walk_future = pool.submit(
            _find_walkable, board, width, height, bg_color, obstacles,
            player_pos, goal_positions, components, attention=attention,
        )
        if grid_scale > 1:
            sprite_future = pool.submit(
                _detect_sprites, board, grid_scale, bg_color,
                player_pos, goal_positions, components, width, height,
            )
            sprites = sprite_future.result()
        else:
            sprites = []
        tile_future = pool.submit(
            _detect_tile_size, board, width, height, bg_color,
        )
        walkable = walk_future.result()
        tile_size = tile_future.result()
    except Exception:
        walkable = _find_walkable(board, width, height, bg_color, obstacles,
                                  player_pos, goal_positions, components, attention=attention)
        if grid_scale > 1:
            sprites = _detect_sprites(board, grid_scale, bg_color,
                                      player_pos, goal_positions, components, width, height)
        else:
            sprites = []
        tile_size = _detect_tile_size(board, width, height, bg_color)

    features = _extract_features(board, width, height, bg_color, components, player_pos, goal_positions, walkable)
    features.tile_size = tile_size
    features.grid_scale = grid_scale
    if grid_scale > 1:
        features.grid_width = width // grid_scale
        features.grid_height = height // grid_scale
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
        sprites=sprites,
    )


def _detect_background_color(board: list[list[int]], width: int, height: int) -> int:
    """Detect background color, excluding letterbox padding and UI elements.
    
    Samples the top row, bottom row, and left column to find the most common
    color, EXCLUDING the rightmost column (often energy bar/step counter) and
    bottom-right corner (padding artifact)."""
    border_pixels: list[int] = []
    if height <= 0 or width <= 0:
        return -1
    # Top row: exclude last column (energy bar)
    for x in range(width - 1):
        border_pixels.append(board[0][x])
    # Bottom row: exclude last column
    for x in range(width - 1):
        border_pixels.append(board[-1][x])
    # Left column: exclude first and last row (already counted)
    for y in range(1, height - 1):
        border_pixels.append(board[y][0])
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
        # Fallback: look for components with color 5 (shape piece) even if border-touching
        shape = [c for c in components if c.dominant_color == 5 and 1 <= c.size <= 80]
        if shape:
            from collections import Counter as _C
            _c = _C()
            for c in shape:
                _c[c.dominant_color] += 1
            result = max(shape, key=lambda c: c.size)
            cx, cy = _component_center(result)
            return (int(cx), int(cy))
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
        if c.dominant_color in (11,):
            score += 3.0
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
    n_goals = min(max(2, len(scored)), 4)
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
        if c.dominant_color == 4 or c.dominant_color == 10:
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
    attention=None,
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
            if attention is not None and not attention.is_relevant(x, y):
                continue
            val = board[y][x]
            if val in walkable_colors and (x, y) not in obs_set:
                passable.add((x, y))
    # Always include player and goal cells regardless of salience.
    if player_pos is not None:
        passable.add(player_pos)
    if goal_positions:
        passable.update(goal_positions)
    # Include non-obstacle component cells (skip mirrors/beams/walls).
    if components:
        for c in components:
            if c.touches_border or c.dominant_color in (4, 10):
                continue
            for cx, cy in c.cells:
                passable.add((cx, cy))
    return passable


def _detect_beam(
    components: list[Component],
    goal_positions: list[tuple[int, int]],
    board: list[list[int]],
    player_pos: tuple[int, int] | None,
    bg_color: int,
    width: int = 64,
    height: int = 64,
) -> tuple[bool, int, list[tuple[int, int]], int, float, int]:
    beam_color_candidates = [4]
    beam_cells: list[tuple[int, int]] = []
    beam_color = -1
    for c in components:
        if c.dominant_color in beam_color_candidates:
            beam_color = c.dominant_color
            beam_cells.extend(c.cells)
    if not beam_cells:
        return (False, -1, [], 0, float('inf'), -1)

    # Distinguish real beam cells from mirror-reflection artifacts.
    # Mirror reflections are rendered as color-4 cells. Check if ALL color-4
    # cells are reflections by testing mirror axis:
    #   - Vertical mirror (tall>wide): reflect X → beam cells on opposite X side
    #   - Horizontal mirror (wide>tall): reflect Y → beam cells on opposite Y side
    if player_pos is not None:
        mirror_comp = next((c for c in components if c.dominant_color == 10), None)
        if mirror_comp is not None and len(beam_cells) > 0:
            mx1, my1, mx2, my2 = mirror_comp.bbox
            horiz = (mx2 - mx1) > (my2 - my1)
            px, py = player_pos
            if horiz:
                mirror_py = sum(cy for _, cy in mirror_comp.cells) // len(mirror_comp.cells)
                y_sides = set()
                for bx, by in beam_cells:
                    if by < mirror_py:
                        y_sides.add("above")
                    elif by > mirror_py:
                        y_sides.add("below")
                if len(y_sides) == 1:
                    beam_side = next(iter(y_sides))
                    player_side = "above" if py < mirror_py else "below"
                    if beam_side != player_side:
                        return (False, -1, [], 0, float('inf'), -1)
            else:
                mirror_px = sum(cx for cx, _ in mirror_comp.cells) // len(mirror_comp.cells)
                x_sides = set()
                for bx, by in beam_cells:
                    if bx < mirror_px:
                        x_sides.add("left")
                    elif bx > mirror_px:
                        x_sides.add("right")
                if len(x_sides) == 1:
                    beam_side = next(iter(x_sides))
                    player_side = "left" if px < mirror_px else "right"
                    if beam_side != player_side:
                        return (False, -1, [], 0, float('inf'), -1)

    beam_cells_set = set(beam_cells)
    goals_lit = sum(1 for gx, gy in goal_positions if (gx, gy) in beam_cells_set)
    min_dist = float('inf')
    mirror_axis = -1
    beam_direction = "point"
    if goal_positions:
        min_dist = min(
            min(abs(bx - gx) + abs(by - gy) for gx, gy in goal_positions)
            for bx, by in beam_cells
        )
    if player_pos is not None:
        px, py = player_pos
        for bx, by in beam_cells:
            if abs(bx - px) + abs(by - py) <= 6:
                mirror_axis = px
                break
    # Infer beam direction from cell positions
    if len(beam_cells) >= 2:
        xs = [c[0] for c in beam_cells]
        ys = [c[1] for c in beam_cells]
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        if x_range > y_range * 2:
            beam_direction = "horizontal"
        elif y_range > x_range * 2:
            beam_direction = "vertical"
        else:
            beam_direction = "diagonal"
    else:
        beam_direction = "point"
    sample = beam_cells[:5]
    return (True, beam_color, sample, goals_lit, min_dist, mirror_axis)


def _compute_goal_coverage(
    player_pos: tuple[int, int],
    beam_cells: list[tuple[int, int]],
    goal_positions: list[tuple[int, int]],
    mirror_x: int,
    width: int,
    height: int,
) -> int:
    """Compute how many goals are covered by beam + reflection + player."""
    covered = set(beam_cells)
    if player_pos is not None:
        covered.add(player_pos)
    # Add reflected beam cells
    for bx, by in beam_cells:
        rx = 2 * mirror_x - bx
        if 0 <= rx < width:
            covered.add((rx, by))
    return sum(1 for gx, gy in goal_positions if (gx, gy) in covered)


def _compute_board_similarity(
    board1: list[list[int]],
    board2: list[list[int]],
    width: int = 64,
    height: int = 64,
) -> float:
    """Compute similarity between two boards (0.0 to 1.0)."""
    if len(board1) != len(board2) or len(board1[0]) != len(board2[0]):
        return 0.0
    total = width * height
    if total == 0:
        return 0.0
    matching = 0
    for y in range(height):
        for x in range(width):
            if board1[y][x] == board2[y][x]:
                matching += 1
    return matching / total


def _infer_mirror_axis(
    beam_comp: Component,
    player_pos: tuple[int, int] | None,
    board: list[list[int]],
) -> int:
    if player_pos is None:
        return -1
    px, py = player_pos
    for bx, by in beam_comp.cells:
        if abs(bx - px) + abs(by - py) <= 6:
            return px
    return -1


def _extract_features(
    board: list[list[int]],
    width: int, height: int,
    bg_color: int,
    components: list[Component],
    player_pos: tuple[int, int] | None,
    goal_positions: list[tuple[int, int]],
    walkable: set[tuple[int, int]],
) -> GameFeatures:
    path_possible = False
    if player_pos is not None and goal_positions and walkable:
        expanded_walkable = set(walkable)
        for c in components:
            if not c.touches_border:
                for cx, cy in c.cells:
                    expanded_walkable.add((cx, cy))
        a_star_passable = expanded_walkable | {player_pos} | set(goal_positions)
        path = astar(player_pos, goal_positions[0], a_star_passable, width, height)
        path_possible = path is not None
    has_beam, beam_color, beam_positions, goals_lit, beam_min_dist, mirror_axis = _detect_beam(
        components, goal_positions, board, player_pos, bg_color,
    )
    # Compute movement axes before dynamics detection
    movement_axes: set[str] = set()
    if player_pos is not None and goal_positions:
        px, py = player_pos
        all_y = [gy for _, gy in goal_positions]
        all_x = [gx for gx, _ in goal_positions]
        if any(abs(gx - px) > 0 for gx in all_x):
            movement_axes.add("x")
        if any(abs(gy - py) > 0 for gy in all_y):
            movement_axes.add("y")
    dynamics_list, primary_dynamics = _detect_dynamics(
        board, components, player_pos, goal_positions, bg_color,
        has_beam, False, movement_axes, width, height,
    )
    features = GameFeatures(
        background_color=bg_color,
        player_detectable=player_pos is not None,
        goal_detectable=len(goal_positions) > 0,
        board_type=primary_dynamics,
        dynamics=dynamics_list,
        primary_dynamics=primary_dynamics,
        tile_size=_detect_tile_size(board, width, height, bg_color),
        path_to_goal_possible=path_possible,
        has_beam=has_beam,
        beam_color=beam_color,
        beam_positions=beam_positions,
        goals_lit_by_beam=goals_lit,
        beam_min_dist_to_goal=beam_min_dist,
        mirror_axis=mirror_axis,
    )
    features.has_gravity = "gravity" in dynamics_list
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
    if movement_axes:
        features.movement_axes = movement_axes
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


def _detect_dynamics(
    board: list[list[int]],
    components: list[Component],
    player_pos: tuple[int, int] | None,
    goal_positions: list[tuple[int, int]],
    bg_color: int,
    has_beam: bool,
    has_click_interaction: bool,
    movement_axes: set[str],
    board_width: int,
    board_height: int,
) -> tuple[list[str], str]:
    detected: list[str] = []

    if player_pos is None and not goal_positions:
        return (["static"], "static")

    non_border = [c for c in components if not c.touches_border]

    # Multi-agent candidates: non-border components that could be entities
    candidate_entities = [
        c for c in non_border
        if c.dominant_color != bg_color
        and c.dominant_color not in (0, 4, 11)
        and c.size >= 4
    ]
    if player_pos is not None:
        player_comp = _find_player_component(components, player_pos)
        if player_comp is not None:
            candidate_entities = [c for c in candidate_entities if c is not player_comp]
    entity_count = len(candidate_entities)
    if entity_count >= 2:
        detected.append("multi-agent")

    # Beam
    if has_beam:
        detected.append("beam")

    # Mirror reflection puzzle: mirror (color 10) with goals on opposite side from player
    if player_pos is not None and goal_positions:
        mirror_comp = next((c for c in components if c.dominant_color == 10), None)
        if mirror_comp is not None:
            mirr_cells = list(mirror_comp.cells)
            mirror_px = sum(cx for cx, _ in mirr_cells) // len(mirr_cells) if mirr_cells else 0
            mirror_py = sum(cy for _, cy in mirr_cells) // len(mirr_cells) if mirr_cells else 0
            px, py = player_pos
            # Detect axis: horizontal mirror (wide > tall) vs vertical (tall > wide)
            mx1, my1, mx2, my2 = mirror_comp.bbox
            horiz = (mx2 - mx1) > (my2 - my1)
            if horiz:
                # For horizontal mirrors, detect puzzle if goals exist and
                # the player is on the board (not off-screen). The mirror
                # reflects vertically; the shape piece can be on either side.
                goal_ys = [gy for _, gy in goal_positions]
                if goal_ys and player_pos is not None:
                    detected.append("puzzle")
            else:
                goal_xs = [gx for gx, _ in goal_positions]
                if not goal_xs:
                    pass
                elif (px < mirror_px < min(goal_xs)) or (px > mirror_px > max(goal_xs)):
                    detected.append("puzzle")
                elif abs(px - mirror_px) <= 6 and goal_xs:
                    detected.append("puzzle")

    # Push/Pull: manipulable components adjacent to player
    if player_pos is not None:
        px, py = player_pos
        neighbors = {(px - 1, py), (px + 1, py), (px, py - 1), (px, py + 1)}
        adjacent_manipulable = [
            c for c in non_border
            if c.dominant_color != bg_color
            and c.dominant_color not in (0, 4, 11)
            and any(cell in neighbors for cell in c.cells)
            and c.size <= 12
        ]
        if adjacent_manipulable:
            detected.append("push-pull")

    # Gravity: player in bottom third with empty space below, or components stacked at bottom
    if player_pos is not None and board_height > 0:
        px, py = player_pos
        bottom_third = board_height * 2 // 3
        if py >= bottom_third:
            below_empty = False
            for y in range(py + 1, board_height):
                if y < board_height and board[y][px] == bg_color:
                    below_empty = True
                else:
                    break
            if below_empty:
                detected.append("gravity")

    # Inventory: small non-border components (collectible-sized)
    if player_pos is not None:
        small_components = [
            c for c in non_border
            if c.dominant_color != bg_color
            and c.dominant_color not in (0, 4, 11)
            and c.size <= 4
        ]
        if len(small_components) >= 2:
            detected.append("inventory")

    # Sliding: long unobstructed corridors in movement axes
    if player_pos is not None:
        px, py = player_pos
        long_corridor = False
        if not movement_axes or "x" in movement_axes:
            for y in range(max(0, py - 3), min(board_height, py + 4)):
                run = 0
                for x in range(board_width):
                    if board[y][x] == bg_color or board[y][x] == 0:
                        run += 1
                        if run >= 12:
                            long_corridor = True
                            break
                    else:
                        run = 0
                if long_corridor:
                    break
        if not long_corridor and (not movement_axes or "y" in movement_axes):
            for x in range(max(0, px - 3), min(board_width, px + 4)):
                run = 0
                for y in range(board_height):
                    if board[y][x] == bg_color or board[y][x] == 0:
                        run += 1
                        if run >= 12:
                            long_corridor = True
                            break
                    else:
                        run = 0
                if long_corridor:
                    break
        if long_corridor:
            detected.append("sliding")

    # Signal: many interactive-looking non-border components
    interactive_count = sum(
        1 for c in non_border
        if c.dominant_color not in (bg_color, 0, 4, 11)
        and 4 <= c.size <= 20
    )
    if interactive_count >= 3:
        detected.append("signal")

    # Pattern-transform: many distinct colors among non-border components
    distinct_colors = set(c.dominant_color for c in non_border if c.dominant_color != bg_color)
    if len(distinct_colors) >= 6 and len(non_border) >= 10:
        detected.append("pattern-transform")

    # Cellular-automata: many tiny components suggesting grid evolution
    tiny_count = sum(1 for c in non_border if c.size <= 3 and c.dominant_color != bg_color)
    if tiny_count >= 8 and len(non_border) >= 15:
        detected.append("cellular-automata")

    # Navigation: default when player and goals exist
    if player_pos is not None and goal_positions:
        if not detected:
            detected.append("navigation")
    elif not detected:
        detected.append("navigation")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_detected: list[str] = []
    for d in detected:
        if d not in seen:
            seen.add(d)
            unique_detected.append(d)

    primary = _pick_primary_dynamics(unique_detected, has_beam, player_pos is not None, bool(goal_positions))

    if len(unique_detected) >= 2:
        unique_detected.append("hybrid")

    return (unique_detected, primary)


def _pick_primary_dynamics(
    detected: list[str],
    has_beam: bool,
    has_player: bool,
    has_goals: bool,
) -> str:
    priority = [
        "static", "beam", "puzzle", "gravity", "sliding", "push-pull", "signal",
        "cellular-automata", "pattern-transform", "multi-agent",
        "inventory", "navigation", "hybrid",
    ]
    for p in priority:
        if p in detected:
            return p
    return "navigation"


def _detect_grid_scale(board: list[list[int]], width: int = 64, height: int = 64) -> int:
    """Detect the pixel-to-grid scale factor using multi-resolution analysis.
    
    Strategy:
    1. Block uniformity: check % of s×s blocks with uniform color
    2. Edge periodicity: find repeating color transitions
    3. Returns scale with >50% uniform blocks (strict)
    """
    # Quick check: if all cells are same color, scale is 1
    first_color = board[0][0] if height > 0 and width > 0 else -1
    all_same = all(board[y][x] == first_color for y in range(height) for x in range(width))
    if all_same:
        return 1
    scores: dict[int, float] = {}
    for scale in range(2, 17):
        bw = width // scale
        bh = height // scale
        if bw < 4 or bh < 4:
            continue
        uniform = 0
        total = 0
        for by in range(bh):
            for bx in range(bw):
                px = bx * scale
                py = by * scale
                block_color = board[py][px]
                is_uniform = True
                for dy in range(scale):
                    row = board[py + dy]
                    for dx in range(scale):
                        if row[px + dx] != block_color:
                            is_uniform = False
                            break
                    if not is_uniform:
                        break
                if is_uniform:
                    uniform += 1
                total += 1
        if total > 0:
            scores[scale] = uniform / total
    # Also check edge periodicity
    edge_scores: dict[int, int] = {}
    for y in range(height):
        prev = -2
        gap = 0
        for x in range(width):
            val = board[y][x]
            if val != prev and prev >= 0:
                if gap > 0 and gap < 10:
                    edge_scores[gap] = edge_scores.get(gap, 0) + 1
                gap = 0
            gap += 1
            prev = val
    # Combine scores
    best_scale = 1
    best_score = 0.0
    for scale in range(2, 17):
        uniformity = scores.get(scale, 0.0)
        edge_bonus = edge_scores.get(scale, 0) / max(1, width * height // 100)
        combined = uniformity * 0.7 + min(edge_bonus, 0.3)
        if combined > best_score and uniformity > 0.50:
            best_score = combined
            best_scale = scale
    return best_scale


def _detect_sprites(
    board: list[list[int]],
    grid_scale: int,
    bg_color: int,
    player_pos: tuple[int, int] | None,
    goal_positions: list[tuple[int, int]],
    components: list[Component],
    width: int = 64,
    height: int = 64,
) -> list[SpriteInfo]:
    """Detect game sprites from the pixel board at grid resolution.
    
    Uses grid-scale downsampling, flood-fill with 8-connectivity, and shape heuristics.
    Links color-0 selection overlay to underlying entity sprites.
    """
    if grid_scale <= 1:
        return []
    gw = width // grid_scale
    gh = height // grid_scale
    if gw < 4 or gh < 4:
        return []
    # Downsample: sample center of each grid cell
    grid = [[0] * gw for _ in range(gh)]
    for gy in range(gh):
        cy = gy * grid_scale + grid_scale // 2
        for gx in range(gw):
            cx = gx * grid_scale + grid_scale // 2
            if cy < height and cx < width:
                grid[gy][gx] = board[cy][cx]
    # Flood-fill with 8-connectivity (including diagonals)
    visited = [[False] * gw for _ in range(gh)]
    sprite_list: list[SpriteInfo] = []
    color_0_cells: set[tuple[int, int]] = set()
    dirs8 = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    for gy in range(gh):
        for gx in range(gw):
            if visited[gy][gx]:
                continue
            color = grid[gy][gx]
            if color == bg_color or color < 0:
                visited[gy][gx] = True
                continue
            cells: list[tuple[int, int]] = []
            stack = [(gx, gy)]
            visited[gy][gx] = True
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for dx, dy in dirs8:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < gw and 0 <= ny < gh and not visited[ny][nx] and grid[ny][nx] == color:
                        visited[ny][nx] = True
                        stack.append((nx, ny))
            if color == 0:
                color_0_cells.update(cells)
                continue
            min_gx = min(c[0] for c in cells)
            max_gx = max(c[0] for c in cells)
            min_gy = min(c[1] for c in cells)
            max_gy = max(c[1] for c in cells)
            span_x = max_gx - min_gx + 1
            span_y = max_gy - min_gy + 1
            # Classify by shape
            is_strip = (span_x == 1 or span_y == 1) and len(cells) >= 3
            is_mirror = (color == 10 and is_strip)
            is_goal = (color == 11)
            is_entity = (color == 5)
            is_beam = (color == 4)
            sprite_id = f"s{color}_{min_gx}_{min_gy}"
            si = SpriteInfo(
                grid_x=min_gx,
                grid_y=min_gy,
                grid_w=span_x,
                grid_h=span_y,
                pixel_x=min_gx * grid_scale,
                pixel_y=min_gy * grid_scale,
                color=color,
                is_clickable=is_entity or is_mirror,
                is_mirror=is_mirror,
                is_goal=is_goal,
                sprite_id=sprite_id,
            )
            sprite_list.append(si)
    # Link color-0 overlay to player entity
    if color_0_cells and player_pos is not None:
        px, py = player_pos
        if px < width and py < height:
            pgx = px // grid_scale
            pgy = py // grid_scale
            for si in sprite_list:
                if si.color == 5 and si.grid_x <= pgx <= si.grid_x + si.grid_w - 1 and si.grid_y <= pgy <= si.grid_y + si.grid_h - 1:
                    si.is_player = True
                    break
    return sprite_list


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
