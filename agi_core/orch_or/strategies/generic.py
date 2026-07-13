from __future__ import annotations

from agi_core.orch_or.game_model import PlanStep
from agi_core.orch_or.search import astar

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agi_core.orch_or.game_model import GameModel


def generic_strategy(game_model: GameModel, context: dict | None = None) -> list[PlanStep]:
    steps: list[PlanStep] = []
    player = game_model.player_pos
    goals = game_model.goal_positions
    if player is None or not goals:
        return steps

    px, py = player
    # Try A* to the nearest goal
    nearest = min(goals, key=lambda g: abs(g[0] - px) + abs(g[1] - py))
    path = astar(player, nearest, game_model.walkable, game_model.width, game_model.height)
    if path and len(path) > 1:
        nx, ny = path[1]
        dx, dy = nx - px, ny - py
        d = ({1: "right", -1: "left"}.get(dx) or
             {1: "down", -1: "up"}.get(dy) or "right")
        steps.append(PlanStep(
            action="move", params={"direction": d},
            source="generic_strategy",
            preconditions=frozenset(),
            expected_effect=f"move_{d}",
        ))
        return steps

    # Heuristic fallback: move toward nearest goal
    gx, gy = nearest
    dx, dy = gx - px, gy - py
    if abs(dx) >= abs(dy):
        d = "right" if dx > 0 else ("left" if dx < 0 else "down")
    else:
        d = "down" if dy > 0 else ("up" if dy < 0 else "right")
    steps.append(PlanStep(
        action="move", params={"direction": d},
        source="generic_strategy",
        preconditions=frozenset(),
        expected_effect=f"move_{d}",
    ))
    return steps
