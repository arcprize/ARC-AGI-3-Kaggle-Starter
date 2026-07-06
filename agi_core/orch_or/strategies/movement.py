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
    for goal in search_goals:
        path = astar(player, goal, game_model.walkable, game_model.width, game_model.height)
        if path and len(path) > 1:
            for i in range(1, len(path)):
                prev = path[i - 1]
                curr = path[i]
                steps.append(_move_step(_direction(prev, curr)))
            return steps
    px, py = player
    non_beam = (context or {}).get("non_beam_controllers", set())
    if game_model.features.has_beam and game_model.features.beam_positions and game_model.features.player_color not in non_beam:
        bp = min(game_model.features.beam_positions, key=lambda p: abs(p[0] - px) + abs(p[1] - py))
        bx, by = bp
        dx = bx - px
        dy = by - py
        if abs(dx) > abs(dy):
            nx, ny = (px + (1 if dx > 0 else -1), py)
        else:
            nx, ny = (px, py + (1 if dy > 0 else -1))
        if not game_model.walkable or (nx, ny) in game_model.walkable:
            if abs(dx) > abs(dy):
                steps.append(_move_step("right" if dx > 0 else "left"))
            elif dy != 0:
                steps.append(_move_step("down" if dy > 0 else "up"))
            return steps
    # Use stable goal for fallback direction, falling through to goals[0].
    target = stable_goal if stable_goal is not None else goals[0]
    gx, gy = target
    dx = gx - px
    dy = gy - py
    if dx == 0 and dy == 0:
        return steps
    # When both axes need movement, let hyperon action priors break the tie.
    _DIR_TO_ACTION = {"up": "ACTION1", "down": "ACTION2", "left": "ACTION3", "right": "ACTION4"}
    action_priors = (context or {}).get("action_priors", {})
    horiz = "right" if dx > 0 else "left"
    vert = "down" if dy > 0 else "up"
    if abs(dx) > abs(dy):
        primary, secondary = horiz, vert
    elif abs(dy) > abs(dx):
        primary, secondary = vert, horiz
    else:
        pri_h = action_priors.get(_DIR_TO_ACTION.get(horiz, ""), 0.0)
        pri_v = action_priors.get(_DIR_TO_ACTION.get(vert, ""), 0.0)
        primary = horiz if pri_h >= pri_v else vert
        secondary = vert if pri_h >= pri_v else horiz
    for d in (primary, secondary):
        off = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
        dvx, dvy = off.get(d, (0, 0))
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
