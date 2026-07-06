from __future__ import annotations

from typing import TYPE_CHECKING

from agi_core.orch_or.game_model import ADVANCE, ESCALATE, REPLAN, RETRY, SWITCH_ENTITY, Plan

if TYPE_CHECKING:
    from agi_core.orch_or.game_model import GameModel, MonitorStatus


def check_progress(
    plan: Plan,
    current_model: GameModel,
    previous_model: GameModel | None,
) -> MonitorStatus:
    if plan.current_index >= len(plan.steps):
        return ESCALATE
    step = plan.steps[plan.current_index]
    if previous_model is None or current_model.player_pos is None or previous_model.player_pos is None:
        return ADVANCE
    prev_px, prev_py = previous_model.player_pos
    curr_px, curr_py = current_model.player_pos
    moved = (curr_px != prev_px or curr_py != prev_py)
    board_changed = _board_changed(previous_model.board, current_model.board)
    if step.action == "move":
        if moved:
            return ADVANCE
        if step.retry_count >= step.tolerance:
            if _player_on_color_0(current_model):
                return SWITCH_ENTITY
            if not _plan_has_switch(plan):
                return SWITCH_ENTITY
            return REPLAN
        return RETRY
    if step.action == "click":
        if board_changed:
            return ADVANCE
        if step.retry_count >= step.tolerance:
            return REPLAN
        return RETRY
    if step.action == "switch":
        if moved:
            return ADVANCE
        if step.retry_count >= step.tolerance:
            return REPLAN
        return RETRY
    return REPLAN


def _plan_has_switch(plan: Plan) -> bool:
    return any(s.action == "switch" for s in plan.steps)


def _player_on_color_0(gm: GameModel) -> bool:
    pos = gm.player_pos
    if pos is None:
        return False
    px, py = pos
    for c in gm.components:
        min_x, min_y, max_x, max_y = c.bbox
        if min_x <= px <= max_x and min_y <= py <= max_y:
            if c.dominant_color == 0:
                return True
    return False


def _board_changed(prev: list[list[int]], curr: list[list[int]]) -> bool:
    if len(prev) != len(curr):
        return True
    for y in range(len(prev)):
        if len(prev[y]) != len(curr[y]):
            return True
        for x in range(len(prev[y])):
            if prev[y][x] != curr[y][x]:
                return True
    return False
