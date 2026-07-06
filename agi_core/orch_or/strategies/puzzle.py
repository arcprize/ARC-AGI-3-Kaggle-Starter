from __future__ import annotations

import logging

from typing import TYPE_CHECKING

from agi_core.orch_or.game_model import PlanStep

if TYPE_CHECKING:
    from agi_core.orch_or.game_model import GameModel

logger = logging.getLogger(__name__)


def puzzle_strategy(game_model: GameModel, context: dict | None = None) -> list[PlanStep]:
    context = context or {}
    steps: list[PlanStep] = []
    player = game_model.player_pos
    if player is None:
        return steps
    already_switched = bool(context.get("switch_attempted", False))
    logger.debug("puzzle_strategy: already_switched=%s player=%s player_color=%s", already_switched, player, game_model.features.player_color)
    if already_switched:
        return steps
    ap = context.get("action_priors", {})
    # Skip puzzle-specific actions when board has single-entity focus.
    try:
        ent_rows = context.get("board_entity_count", [])
        focus_rows = context.get("board_focus_mode", [])
        if ent_rows and str(ent_rows[0].get("count", "single")) == "single":
            return steps
        if focus_rows and str(focus_rows[0].get("mode", "single")) == "single":
            return steps
    except Exception:
        pass
    interactive = [(ox, oy) for ox, oy, otype in game_model.obstacles if otype == "interactive"]
    # Skip click on interactive obstacles when a beam is present — the
    # exploration strategy's beam navigation should drive instead.
    if interactive and not game_model.features.has_beam:
        if not ap or ap.get("ACTION6", 0.5) > 0.1:
            px, py = player
            interactive.sort(key=lambda p: abs(p[0] - px) + abs(p[1] - py))
            tx, ty = interactive[0]
            steps.append(_interact_step(tx, ty, game_model))
    if not steps and game_model.walkable:
        # If the current entity controls the beam, let the exploration
        # strategy's beam navigation handle movement instead.
        if game_model.features.has_beam:
            return steps
        px, py = player
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = px + dx, py + dy
            if (nx, ny) in game_model.walkable:
                direction = {1: "right", -1: "left", 0: ""}[dx] or {1: "down", -1: "up"}[dy]
                steps.append(_explore_step(direction))
                break
    return steps


def _find_player_component(gm: GameModel) -> object | None:
    px, py = gm.player_pos
    if px is None or py is None:
        return None
    for c in gm.components:
        min_x, min_y, max_x, max_y = c.bbox
        if min_x <= px <= max_x and min_y <= py <= max_y:
            return c
    return None


def _switch_step() -> PlanStep:
    return PlanStep(
        action="switch",
        params={},
        source="puzzle_strategy",
        preconditions=frozenset(),
        expected_effect="switch_entity",
    )


def _interact_step(tx: int, ty: int, gm: GameModel) -> PlanStep:
    return PlanStep(
        action="click",
        params={"x": tx, "y": ty, "target_gx": tx, "target_gy": ty},
        source="puzzle_strategy",
        preconditions=frozenset(),
        expected_effect="clear_obstacle",
    )


def _explore_step(direction: str) -> PlanStep:
    return PlanStep(
        action="move",
        params={"direction": direction},
        source="puzzle_strategy",
        preconditions=frozenset(),
        expected_effect=f"explore_{direction}",
    )
