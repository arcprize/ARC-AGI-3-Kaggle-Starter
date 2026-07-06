from __future__ import annotations

from typing import TYPE_CHECKING

from agi_core.orch_or.game_model import PlanStep

if TYPE_CHECKING:
    from agi_core.orch_or.game_model import Component, GameModel


def interaction_strategy(game_model: GameModel, context: dict | None = None) -> list[PlanStep]:
    steps: list[PlanStep] = []
    player = game_model.player_pos
    obstacles = game_model.obstacles
    if player is None:
        return steps
    ctx = context or {}
    ap = ctx.get("action_priors", {})
    # Board-type interaction range from MeTTa rules.
    interact_range = 2
    try:
        range_rows = ctx.get("board_interact_range", [])
        if range_rows:
            interact_range = int(range_rows[0].get("range", 2))
    except Exception:
        pass
    px, py = player
    interactive_obstacles = [(ox, oy) for ox, oy, otype in obstacles if otype == "interactive"]
    if not ap or ap.get("ACTION6", 0.5) > 0.1:
        for ox, oy in interactive_obstacles:
            if abs(ox - px) + abs(oy - py) > interact_range:
                continue
            steps.append(_interact_step(ox, oy, game_model))
    player_comp = _find_player_component(game_model)
    if player_comp is not None and player_comp.dominant_color == 0:
        if not ap or ap.get("ACTION5", 0.5) > 0.1:
            has_switch = any(s.action == "switch" for s in steps)
            if not has_switch:
                steps.append(_switch_step())
    return steps


def _find_player_component(gm: GameModel) -> Component | None:
    px, py = gm.player_pos
    if px is None or py is None:
        return None
    for c in gm.components:
        min_x, min_y, max_x, max_y = c.bbox
        if min_x <= px <= max_x and min_y <= py <= max_y:
            return c
    return None


def _interact_step(tx: int, ty: int, gm: GameModel) -> PlanStep:
    return PlanStep(
        action="click",
        params={"x": tx, "y": ty, "target_gx": tx, "target_gy": ty},
        source="interaction_strategy",
        preconditions=frozenset(),
        expected_effect="break_block",
    )


def _switch_step() -> PlanStep:
    return PlanStep(
        action="switch",
        params={},
        source="interaction_strategy",
        preconditions=frozenset(),
        expected_effect="switch_entity",
    )
