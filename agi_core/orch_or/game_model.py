from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Component:
    cells: list[tuple[int, int]]
    bbox: tuple[int, int, int, int]
    size: int
    dominant_color: int
    touches_border: bool


@dataclass(slots=True)
class GameFeatures:
    has_gravity: bool = False
    gravity_dir: int = 0
    has_click_interaction: bool = False
    has_breakable_blocks: bool = False
    has_collectibles: bool = False
    movement_axes: set[str] = field(default_factory=lambda: {"x", "y"})
    player_detectable: bool = False
    goal_detectable: bool = False
    board_type: str = "unknown"
    path_to_goal_possible: bool = False
    tile_size: int = 6
    background_color: int = -1
    player_color: int = -1
    goal_color: int = -1
    wall_color: int = -1
    has_beam: bool = False
    beam_color: int = -1
    beam_positions: list[tuple[int, int]] = field(default_factory=list)
    goals_lit_by_beam: int = 0
    beam_min_dist_to_goal: float = float('inf')
    mirror_axis: int = -1


@dataclass(slots=True)
class GameModel:
    board: list[list[int]]
    height: int
    width: int
    player_pos: tuple[int, int] | None
    goal_positions: list[tuple[int, int]]
    obstacles: list[tuple[int, int, str]]
    components: list[Component]
    walkable: set[tuple[int, int]]
    features: GameFeatures


@dataclass(slots=True)
class PlanStep:
    action: str
    params: dict[str, Any] | None
    source: str
    preconditions: frozenset[str]
    expected_effect: str
    tolerance: int = 3
    retry_count: int = 0


@dataclass(slots=True)
class Plan:
    objective_id: str
    steps: list[PlanStep]
    current_index: int = 0
    created_at_step: int = 0
    attempt_count: int = 0
    confidence: float = 0.0
    is_active: bool = True


@dataclass(slots=True)
class Objective:
    description: str
    priority: float
    sub_objectives: list[Objective] = field(default_factory=list)


MonitorStatus = str
ADVANCE: MonitorStatus = "advance"
RETRY: MonitorStatus = "retry"
REPLAN: MonitorStatus = "replan"
ESCALATE: MonitorStatus = "escalate"
SWITCH_ENTITY: MonitorStatus = "switch_entity"

DIR_TO_ACTION: dict[str, str] = {
    "up": "ACTION1", "down": "ACTION2", "left": "ACTION3", "right": "ACTION4",
}
ACTION_TO_DIR: dict[str, str] = {
    "ACTION1": "up", "ACTION2": "down", "ACTION3": "left", "ACTION4": "right",
}


def serialize_plan(plan: Plan) -> dict[str, Any]:
    return {
        "objective_id": plan.objective_id,
        "current_index": plan.current_index,
        "attempt_count": plan.attempt_count,
        "created_at_step": plan.created_at_step,
        "confidence": plan.confidence,
        "is_active": plan.is_active,
        "steps": [
            {
                "action": s.action,
                "params": s.params,
                "source": s.source,
                "expected_effect": s.expected_effect,
                "tolerance": s.tolerance,
                "retry_count": s.retry_count,
                "preconditions": list(s.preconditions),
            }
            for s in plan.steps
        ],
    }


def deserialize_plan(data: dict[str, Any]) -> Plan:
    steps = [
        PlanStep(
            action=s["action"],
            params=s.get("params"),
            source=s.get("source", "fallback"),
            preconditions=frozenset(s.get("preconditions", [])),
            expected_effect=s.get("expected_effect", ""),
            tolerance=s.get("tolerance", 3),
            retry_count=s.get("retry_count", 0),
        )
        for s in data.get("steps", [])
    ]
    return Plan(
        objective_id=data["objective_id"],
        steps=steps,
        current_index=data.get("current_index", 0),
        created_at_step=data.get("created_at_step", 0),
        attempt_count=data.get("attempt_count", 0),
        confidence=data.get("confidence", 0.0),
        is_active=data.get("is_active", True),
    )
