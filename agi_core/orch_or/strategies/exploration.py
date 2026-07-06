from __future__ import annotations

from typing import TYPE_CHECKING

from agi_core.orch_or.game_model import PlanStep
from agi_core.orch_or.search import astar

if TYPE_CHECKING:
    from agi_core.orch_or.game_model import GameModel


def exploration_strategy(game_model: GameModel, context: dict | None = None) -> list[PlanStep]:
    steps: list[PlanStep] = []
    player = game_model.player_pos
    features = game_model.features
    if player is not None:
        px, py = player
        # When a stable goal is known and the entity is a non-beam-controller
        # (e.g. the shape piece after a switch), navigate toward it.
        # For beam controllers use beam navigation below.
        ctx = context or {}
        stable_goal = ctx.get("stable_goal")
        non_beam_set = ctx.get("non_beam_controllers", set())
        if stable_goal is not None and features.goal_detectable and features.player_color in non_beam_set:
            gx, gy = stable_goal
            candidates = []
            if gx < px:
                candidates.append("left")
            elif gx > px:
                candidates.append("right")
            if gy < py:
                candidates.append("up")
            elif gy > py:
                candidates.append("down")
            # Reorder by hyperon action priors, then filter failed directions.
            priors = ctx.get("action_priors", {})
            candidates = _bias_candidates_by_priors(candidates, priors)
            failed = ctx.get("failed_directions", set())
            candidates = [d for d in candidates if d not in failed]
            for d in candidates:
                dvx, dvy = _direction_vec(d)
                nx, ny = px + dvx, py + dvy
                if not game_model.walkable or (nx, ny) in game_model.walkable:
                    steps.append(_explore_step(d))
                    return steps
            # Direct route blocked — use A* with an expanded walkable set
            # that includes interior (non-border) component cells, so the
            # shape piece can pathfind through terrain of any color.
            expanded_walkable = set(game_model.walkable)
            for c in game_model.components:
                if not c.touches_border:
                    for cx, cy in c.cells:
                        expanded_walkable.add((cx, cy))
            path = astar((px, py), (gx, gy), expanded_walkable, game_model.width, game_model.height)
            if path and len(path) > 1:
                for i in range(1, min(len(path), 5)):
                    nx, ny = path[i]
                    dx, dy = nx - px, ny - py
                    d = {1: "right", -1: "left"}.get(dx, "") or {1: "down", -1: "up"}.get(dy, "")
                    if d and d not in failed:
                        steps.append(_explore_step(d))
                        return steps
        if features.has_beam:
            ctx = context or {}
            non_beam = ctx.get("non_beam_controllers", set())
            if features.player_color not in non_beam:
                # If hyperon has a very high confidence direction, try it first
                # before falling through to beam navigation.
                _AP_TO_DIR = {"ACTION1": "up", "ACTION2": "down", "ACTION3": "left", "ACTION4": "right"}
                ap = ctx.get("action_priors", {})
                hyperon_best_dir = max(((d, ap.get(a, 0.0)) for a, d in _AP_TO_DIR.items()), key=lambda x: x[1], default=(None, 0.0))
                if hyperon_best_dir[0] and hyperon_best_dir[1] > 0.8:
                    dvx, dvy = _direction_vec(hyperon_best_dir[0])
                    nx, ny = px + dvx, py + dvy
                    if not game_model.walkable or (nx, ny) in game_model.walkable:
                        steps.append(_explore_step(hyperon_best_dir[0]))
                        return steps
                mirror_dir = _mirror_axis_direction(game_model)
                if mirror_dir is not None:
                    dx, dy_ = _direction_vec(mirror_dir)
                    nx, ny = px + dx, py + dy_
                    if not game_model.walkable or (nx, ny) in game_model.walkable:
                        steps.append(_explore_step(mirror_dir))
                        return steps
                gradient_dir = _beam_gradient_step(game_model, context)
                if gradient_dir is not None:
                    dx, dy_ = _direction_vec(gradient_dir)
                    nx, ny = px + dx, py + dy_
                    if not game_model.walkable or (nx, ny) in game_model.walkable:
                        steps.append(_explore_step(gradient_dir))
                        return steps
                if features.beam_positions:
                    bp = min(features.beam_positions, key=lambda p: abs(p[0] - px) + abs(p[1] - py))
                    bx, by = bp
                    dx = bx - px
                    dy = by - py
                    if abs(dx) > abs(dy):
                        dir = "right" if dx > 0 else "left"
                    else:
                        dir = "down" if dy > 0 else "up"
                    if features.mirror_axis >= 0:
                        dir = {"left": "right", "right": "left", "up": "up", "down": "down"}.get(dir, dir)
                    nx, ny = (px + (1 if dx > 0 else -1), py) if abs(dx) > abs(dy) else (px, py + (1 if dy > 0 else -1))
                    if not game_model.walkable or (nx, ny) in game_model.walkable:
                        steps.append(_explore_step(dir))
                        return steps
                # Board-type explore bias overrides the default search order.
                _explore_bias = "vertical" if features.has_beam else "horizontal"
                try:
                    board_bias_rows = ctx.get("board_explore_bias", [])
                    if board_bias_rows:
                        _explore_bias = str(board_bias_rows[0].get("axis", _explore_bias))
                except Exception:
                    pass
                if _explore_bias == "vertical":
                    search_dirs = ["down", "up", "right", "left"]
                else:
                    search_dirs = ["right", "left", "down", "up"]
                search_dirs = _bias_candidates_by_priors(search_dirs, ctx.get("action_priors", {}))
                for d in search_dirs:
                    dvx, dvy = _direction_vec(d)
                    nx, ny = px + dvx, py + dvy
                    if not game_model.walkable or (nx, ny) in game_model.walkable:
                        steps.append(_explore_step(d))
                        return steps
                # Last resort: try the stable goal when beam nav gives nothing.
                if stable_goal is not None and features.goal_detectable:
                    gx, gy = stable_goal
                    candidates = []
                    if gx < px:
                        candidates.append("left")
                    elif gx > px:
                        candidates.append("right")
                    if gy < py:
                        candidates.append("up")
                    elif gy > py:
                        candidates.append("down")
                    for d in candidates:
                        dvx, dvy = _direction_vec(d)
                        nx, ny = px + dvx, py + dvy
                        if not game_model.walkable or (nx, ny) in game_model.walkable:
                            steps.append(_explore_step(d))
                            return steps
                return steps
        # Direct navigation toward the nearest goal when the current entity does
        # not control the beam (e.g. a secondary sprite in a puzzle level).
        non_beam_set = (context or {}).get("non_beam_controllers", set())
        if features.goal_detectable and game_model.goal_positions and (features.board_type == "platformer" or (features.board_type == "puzzle" and features.player_color in non_beam_set)):
            stable_goal = (context or {}).get("stable_goal")
            if stable_goal is not None:
                gx, gy = stable_goal
            else:
                nearest_goal = min(game_model.goal_positions, key=lambda g: abs(g[0] - px) + abs(g[1] - py))
                gx, gy = nearest_goal
            candidates = []
            if gx < px:
                candidates.append("left")
            elif gx > px:
                candidates.append("right")
            if gy < py:
                candidates.append("up")
            elif gy > py:
                candidates.append("down")
            # Order by hyperon priors, then push failed directions to back.
            priors = (context or {}).get("action_priors", {})
            candidates = _bias_candidates_by_priors(candidates, priors)
            failed = (context or {}).get("failed_directions", set())
            candidates = [d for d in candidates if d not in failed] + [d for d in candidates if d in failed]
            for d in candidates:
                dvx, dvy = _direction_vec(d)
                nx, ny = px + dvx, py + dvy
                if not game_model.walkable or (nx, ny) in game_model.walkable:
                    steps.append(_explore_step(d))
                    return steps
        if not features.goal_detectable:
            center_x = game_model.width // 2
            if px < center_x:
                steps.append(_explore_step("right"))
            else:
                steps.append(_explore_step("left"))
            return steps
        # Search order fallback for puzzle boards with goal detectable
        if features.board_type == "puzzle":
            _explore_bias = "vertical"
            try:
                board_bias_rows = (context or {}).get("board_explore_bias", [])
                if board_bias_rows:
                    _explore_bias = str(board_bias_rows[0].get("axis", "vertical"))
            except Exception:
                pass
            if _explore_bias == "vertical":
                search_dirs = ["down", "up", "right", "left"]
            else:
                search_dirs = ["right", "left", "down", "up"]
            for d in search_dirs:
                dvx, dvy = _direction_vec(d)
                nx, ny = px + dvx, py + dvy
                if (nx, ny) in game_model.walkable:
                    steps.append(_explore_step(d))
                    return steps
    return steps


def _perpendicular(d: str) -> list[str]:
    return {"up": ["left", "right"], "down": ["left", "right"],
            "left": ["up", "down"], "right": ["up", "down"]}.get(d, ["up", "down", "left", "right"])


def _direction_vec(d: str) -> tuple[int, int]:
    return {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}.get(d, (0, 0))


def _mirror_axis_direction(game_model: GameModel) -> str | None:
    features = game_model.features
    if features.mirror_axis < 0 or not features.has_beam or not game_model.goal_positions:
        return None
    player = game_model.player_pos
    if player is None:
        return None
    px, py = player
    all_beam_cells = _get_beam_cells(game_model)
    if not all_beam_cells:
        return None
    nearest_beam = None
    nearest_goal = None
    min_d = float('inf')
    for bx, by in all_beam_cells:
        for gx, gy in game_model.goal_positions:
            d = abs(bx - gx) + abs(by - gy)
            if d < min_d:
                min_d = d
                nearest_beam = (bx, by)
                nearest_goal = (gx, gy)
    if nearest_beam is None or nearest_goal is None:
        return None
    bx, by = nearest_beam
    gx, gy = nearest_goal
    toward_x = None
    if bx < gx:
        toward_x = "left"
    elif bx > gx:
        toward_x = "right"
    toward_y = None
    if by < gy:
        toward_y = "down"
    elif by > gy:
        toward_y = "up"
    if toward_y:
        ydiff = abs(by - gy)
        if ydiff >= 12:
            return toward_y
        beam_min_y = min(y for _, y in all_beam_cells)
        beam_max_y = max(y for _, y in all_beam_cells)
        goal_min_y = min(g_y for _, g_y in game_model.goal_positions)
        goal_max_y = max(g_y for _, g_y in game_model.goal_positions)
        if beam_max_y < goal_min_y or beam_min_y > goal_max_y:
            return toward_y
        if toward_x and features.mirror_axis >= 0:
            reflection_x = 2 * features.mirror_axis - px
            goal_x = gx
            if reflection_x >= goal_x:
                return toward_y
        if toward_x:
            return toward_x
        return toward_y
    if toward_x:
        return toward_x
    return None


def _get_beam_cells(game_model: GameModel) -> list[tuple[int, int]]:
    comps = [c for c in game_model.components if c.dominant_color == game_model.features.beam_color and not c.touches_border]
    if not comps:
        comps = [c for c in game_model.components if c.dominant_color == game_model.features.beam_color]
    if comps:
        return comps[0].cells
    return game_model.features.beam_positions


def _beam_gradient_step(
    gm: GameModel, ctx: dict | None,
) -> str | None:
    if ctx is None:
        return None
    px, py = gm.player_pos
    if px is None:
        return None
    coverage_change = ctx.get("beam_coverage_change", 0)
    proximity_change = ctx.get("beam_proximity_change", 0)
    last_dir = ctx.get("last_move_direction")
    best_dir = ctx.get("beam_best_direction")
    if coverage_change > 0 and last_dir:
        return last_dir
    if coverage_change < 0:
        if best_dir and best_dir != last_dir:
            return best_dir
        if last_dir:
            for d in _perpendicular(last_dir):
                dx, dy_ = _direction_vec(d)
                nx, ny = px + dx, py + dy_
                if not gm.walkable or (nx, ny) in gm.walkable:
                    return d
    if coverage_change == 0 and proximity_change > 0 and last_dir:
        return last_dir
    if proximity_change < 0:
        if best_dir and best_dir != last_dir:
            return best_dir
        if last_dir:
            opposite = {"up": "down", "down": "up", "left": "right", "right": "left"}.get(last_dir)
            if opposite:
                dx, dy_ = _direction_vec(opposite)
                nx, ny = px + dx, py + dy_
                if not gm.walkable or (nx, ny) in gm.walkable:
                    return opposite
            for d in _perpendicular(last_dir):
                dx, dy_ = _direction_vec(d)
                nx, ny = px + dx, py + dy_
                if not gm.walkable or (nx, ny) in gm.walkable:
                    return d
    if best_dir:
        return best_dir
    if last_dir:
        dx, dy_ = _direction_vec(last_dir)
        nx, ny = px + dx, py + dy_
        if not gm.walkable or (nx, ny) in gm.walkable:
            return last_dir
    return None


def _bias_candidates_by_priors(
    candidates: list[str],
    priors: dict[str, float],
) -> list[str]:
    """Reorder candidate directions so that the one with the highest hyperon
    action prior is tried first."""
    _DIR_TO_ACTION = {"up": "ACTION1", "down": "ACTION2", "left": "ACTION3", "right": "ACTION4"}
    if not priors:
        return candidates
    scored = [(priors.get(_DIR_TO_ACTION.get(d, ""), 0.0), d) for d in candidates]
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored]


def _explore_step(direction: str) -> PlanStep:
    return PlanStep(
        action="move",
        params={"direction": direction},
        source="exploration_strategy",
        preconditions=frozenset(),
        expected_effect=f"explore_{direction}",
    )