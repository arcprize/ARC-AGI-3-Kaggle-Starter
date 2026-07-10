from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DynamicsCategory(str, Enum):
    STATIC = "static"
    NAVIGATION = "navigation"
    SLIDING = "sliding"
    GRAVITY = "gravity"
    PUSH_PULL = "push-pull"
    BEAM = "beam"
    SIGNAL = "signal"
    CELLULAR_AUTOMATA = "cellular-automata"
    PATTERN_TRANSFORM = "pattern-transform"
    MULTI_AGENT = "multi-agent"
    INVENTORY = "inventory"
    HYBRID = "hybrid"


@dataclass(slots=True)
class Component:
    cells: list[tuple[int, int]]
    bbox: tuple[int, int, int, int]
    size: int
    dominant_color: int
    touches_border: bool
    confidence: float = 1.0


@dataclass(slots=True)
class SpriteInfo:
    grid_x: int
    grid_y: int
    grid_w: int
    grid_h: int
    pixel_x: int
    pixel_y: int
    color: int
    is_clickable: bool = False
    is_player: bool = False
    is_mirror: bool = False
    is_goal: bool = False
    sprite_id: str = ""


@dataclass(slots=True)
class GameFeatures:
    has_gravity: bool = False
    gravity_dir: int = 0
    has_collectibles: bool = False
    movement_axes: set[str] = field(default_factory=lambda: {"x", "y"})
    player_detectable: bool = False
    goal_detectable: bool = False
    board_type: str = "unknown"
    dynamics: list[str] = field(default_factory=list)
    primary_dynamics: str = "unknown"
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
    beam_direction: str = ""
    grid_scale: int = 0
    dynamics_confidence: dict[str, float] = field(default_factory=dict)
    perception_confidence: float = 1.0
    grid_width: int = 0
    grid_height: int = 0


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
    sprites: list[SpriteInfo] = field(default_factory=list)
    entity_tracks: list[EntityTrack] = field(default_factory=list)


@dataclass(slots=True)
class EntityTrack:
    """Persistent tracking state for one entity across frames.

    Updated each frame by ``EntityTracker`` (in ``tracker.py``).
    """
    entity_id: int
    color: int
    positions: list[tuple[int, int]] = field(default_factory=list)
    bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    age: int = 0
    last_seen: int = 0
    velocity: tuple[int, int] = (0, 0)
    stationary_streak: int = 0
    moved_this_frame: bool = False


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
