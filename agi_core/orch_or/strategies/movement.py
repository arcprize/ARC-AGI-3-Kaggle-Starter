from __future__ import annotations

from typing import TYPE_CHECKING

from agi_core.orch_or.game_model import PlanStep
from agi_core.orch_or.search import astar

if TYPE_CHECKING:
    from agi_core.orch_or.game_model import GameModel


def _direction(from_pos: tuple[int, int], to_pos: tuple[int, int]) -> str:
    fx, fy = from_pos
    tx, ty = to_pos
    if tx > fx:
        return "right"
    if tx < fx:
        return "left"
    if ty > fy:
        return "down"
    if ty < fy:
        return "up"
    return "right"


def _is_beam_controller(game_model: GameModel) -> bool:
    """True when the player entity is a beam-controller (adjacent to beam)."""
    features = game_model.features
    if not features.has_beam or not features.beam_positions or game_model.player_pos is None:
        return False
    px, py = game_model.player_pos
    return any(abs(bx - px) <= 6 and abs(by - py) <= 6
               for bx, by in features.beam_positions)


def movement_strategy(game_model: GameModel, context: dict | None = None) -> list[PlanStep]:
    steps: list[PlanStep] = []
    player = game_model.player_pos
    goals = game_model.goal_positions
    if player is None or not goals:
        return steps
    # Prefer the stable goal from context (persisted from platformer boards).
    stable_goal = (context or {}).get("stable_goal")
    search_goals = [stable_goal] if stable_goal is not None else []
    for g in goals:
        if not search_goals or g != stable_goal:
            search_goals.append(g)
    # Board-type path preference: "safe" avoids interactive obstacles.
    _path_walkable = set(game_model.walkable)
    try:
        pref_rows = (context or {}).get("board_path_preference", [])
        if pref_rows and str(pref_rows[0].get("preference", "balanced")) == "safe":
            obs_set = set((ox, oy) for ox, oy, _ in game_model.obstacles if _ == "interactive")
            for ox, oy in list(obs_set):
                for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                    _path_walkable.discard((ox+dx, oy+dy))
    except Exception:
        pass
    # Board-type danger detection: "avoid" excludes hazard-adjacent cells.
    try:
        danger_rows = (context or {}).get("board_danger_detection", [])
        if danger_rows and str(danger_rows[0].get("detection", "ignore")) == "avoid":
            for c in game_model.components:
                if c.dominant_color in (1, 2) and not c.touches_border:
                    for cx, cy in c.cells:
                        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                            _path_walkable.discard((cx+dx, cy+dy))
    except Exception:
        pass
    for goal in search_goals:
        path = astar(player, goal, _path_walkable, game_model.width, game_model.height)
        if path and len(path) > 1:
            is_close = abs(player[0] - goal[0]) + abs(player[1] - goal[1]) <= 9
            step_size = 1 if is_close else 3
            for i in range(1, len(path), step_size):
                prev = path[i - 1]
                curr = path[i]
                steps.append(_move_step(_direction(prev, curr)))
            return steps
    if game_model.obstacles and game_model.player_pos is not None:
        obs_walkable = set(_path_walkable)
        for c in game_model.components:
            if not c.touches_border:
                for cxx, cyy in c.cells:
                    obs_walkable.add((cxx, cyy))
        px2, py2 = game_model.player_pos
        for ox, oy, otype in sorted(game_model.obstacles, key=lambda o: abs(o[0] - px2) + abs(o[1] - py2)):
            if otype != "interactive":
                continue
            bg = game_model.features.background_color
            comp = next((c for c in game_model.components if c.dominant_color != bg and c.bbox[0] <= ox <= c.bbox[2] and c.bbox[1] <= oy <= c.bbox[3]), None)
            if comp is not None:
                click_target = min(comp.cells, key=lambda cell: abs(cell[0] - px2) + abs(cell[1] - py2))
                cx2, cy2 = click_target
            else:
                cx2, cy2 = ox, oy
            dist = abs(cx2 - px2) + abs(cy2 - py2)
            if dist <= 2:
                steps.append(_interact_step(cx2, cy2))
                return steps
            adj_targets = [(cx2 + dx, cy2 + dy) for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)) if (cx2 + dx, cy2 + dy) in obs_walkable]
            if adj_targets:
                adj_goal = min(adj_targets, key=lambda t: abs(t[0] - px2) + abs(t[1] - py2))
                path_to_obs = astar(player, adj_goal, obs_walkable, game_model.width, game_model.height)
                if path_to_obs and len(path_to_obs) > 1:
                    step_idx = 0
                    while step_idx < len(path_to_obs) - 1:
                        next_idx = min(step_idx + 3, len(path_to_obs) - 1)
                        prev = path_to_obs[step_idx]
                        curr = path_to_obs[next_idx]
                        steps.append(_move_step(_direction(prev, curr)))
                        step_idx = next_idx
                    steps.append(_interact_step(cx2, cy2))
                    return steps
    px, py = player
    non_beam = (context or {}).get("non_beam_controllers", set())
    failed_dirs = (context or {}).get("failed_directions", set())
    if (game_model.features.has_beam and game_model.features.beam_positions
            and game_model.features.player_color not in non_beam
            and _is_beam_controller(game_model)):
        bp = min(game_model.features.beam_positions, key=lambda p: abs(p[0] - px) + abs(p[1] - py))
        bx, by = bp
        dx = bx - px
        dy = by - py
        if abs(dx) > abs(dy):
            nd = "right" if dx > 0 else "left"
            nx, ny = (px + (1 if dx > 0 else -1), py)
        else:
            nd = "down" if dy > 0 else "up"
            nx, ny = (px, py + (1 if dy > 0 else -1))
        if nd not in failed_dirs and (not game_model.walkable or (nx, ny) in game_model.walkable):
            steps.append(_move_step(nd))
            return steps
    target = stable_goal if stable_goal is not None else goals[0]
    gx, gy = target
    dx = gx - px
    dy = gy - py
    if dx == 0 and dy == 0:
        return steps
    _DIR_TO_ACTION = {"up": "ACTION1", "down": "ACTION2", "left": "ACTION3", "right": "ACTION4"}
    action_priors = (context or {}).get("action_priors", {})
    failed_dirs = (context or {}).get("failed_directions", set())
    candidates = []
    if dx > 0: candidates.append("right")
    elif dx < 0: candidates.append("left")
    if dy > 0: candidates.append("down")
    elif dy < 0: candidates.append("up")
    if abs(dx) > abs(dy):
        pass  # horizontal bias from order above
    elif abs(dy) > abs(dx):
        candidates = candidates[::-1]
    else:
        pri_h = action_priors.get(_DIR_TO_ACTION.get(candidates[0], ""), 0.0) if candidates else 0.0
        pri_v = action_priors.get(_DIR_TO_ACTION.get(candidates[-1], ""), 0.0) if len(candidates) > 1 else 0.0
        if pri_v > pri_h and len(candidates) > 1:
            candidates = candidates[::-1]
    for d in candidates:
        if d in failed_dirs:
            continue
        dvx, dvy = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}.get(d, (0, 0))
        nx, ny = px + dvx, py + dvy
        if not game_model.walkable or (nx, ny) in game_model.walkable:
            steps.append(_move_step(d))
            return steps
    return steps


def _move_step(direction: str) -> PlanStep:
    return PlanStep(
        action="move",
        params={"direction": direction},
        source="movement_strategy",
        preconditions=frozenset(),
        expected_effect=f"move_{direction}",
    )


def _interact_step(tx: int, ty: int) -> PlanStep:
    return PlanStep(
        action="click",
        params={"x": tx, "y": ty, "target_gx": tx, "target_gy": ty},
        source="movement_strategy",
        preconditions=frozenset(),
        expected_effect="break_block",
    )
