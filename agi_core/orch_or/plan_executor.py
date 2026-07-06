from __future__ import annotations

from typing import TYPE_CHECKING

from agi_core.types import ActionProposal

if TYPE_CHECKING:
    from agi_core.orch_or.game_model import GameModel, Plan, PlanStep


def plan_step_to_proposal(
    step: PlanStep,
    game_model: GameModel,
    plan: Plan,
) -> ActionProposal | None:
    action_name = _resolve_action_name(step, game_model)
    if action_name is None:
        return None
    return ActionProposal(
        action_name=action_name,
        source=f"orch_or:{plan.objective_id}:{step.source}",
        confidence=0.7,
        reason=step.expected_effect,
        subgoal=plan.objective_id,
        metadata={
            "step_index": plan.current_index,
            "total_steps": len(plan.steps),
            "plan_objective": plan.objective_id,
            "plan_attempt": plan.attempt_count,
            "expected_effect": step.expected_effect,
        },
        expected_outcome=step.expected_effect,
        legal=True,
    )


def action_name_for_direction(direction: str) -> str | None:
    mapping = {
        "left": "ACTION3",
        "right": "ACTION4",
        "up": "ACTION1",
        "down": "ACTION2",
    }
    return mapping.get(direction)


def _resolve_action_name(step: PlanStep, game_model: GameModel) -> str | None:
    if step.action == "move":
        direction = (step.params or {}).get("direction", "")
        return action_name_for_direction(direction)
    if step.action == "click":
        return "ACTION6"
    if step.action == "switch":
        return "ACTION5"
    return None



