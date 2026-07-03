from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True)
class Observation:
    game_id: str
    step_index: int
    game_state: str
    levels_completed: int
    last_action: str | None
    legal_actions: list[str]
    raw_frame_ref: Any = None
    derived_features: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActionProposal:
    action_name: str
    source: str
    confidence: float
    reason: str
    subgoal: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    expected_outcome: str | None = None
    legal: bool = True
    score: float = 0.0


@dataclass(slots=True)
class ActionDecision:
    action_name: str
    source: str
    confidence: float
    reason: str
    subgoal: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    proposals: list[ActionProposal] = field(default_factory=list)
    used_fallback: bool = False


@dataclass(slots=True)
class EpisodeRecord:
    game_id: str
    step_index: int
    action_name: str
    outcome: str
    progress_delta: int
    game_state: str
    prior_game_state: str
    levels_completed: int
    prior_frame_signature: str | None = None
    next_game_state: str = ""
    next_frame_signature: str | None = None
    prior_legal_actions: list[str] = field(default_factory=list)
    stagnation_before: int = 0
    stagnation_after: int = 0
    decision_subgoal: str | None = None
    decision_source: str | None = None


@dataclass(slots=True)
class MemoryBundle:
    recent_records: list[dict[str, Any]] = field(default_factory=list)
    basin_status: dict[str, Any] = field(default_factory=dict)
    basin_action_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    action_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    state_action_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    frame_action_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    subgoal_action_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    trained_prior_bucket: dict[str, int] = field(default_factory=dict)
    recommended_actions: list[str] = field(default_factory=list)
    suppressed_actions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GoalState:
    name: str
    target_state: str | None = None
    priority: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutiveContext:
    observation: Observation
    goals: list[GoalState]
    memory: Any
    adapter: Any
    hyperon: Any
    llm_backend: Any
    config: Any
    episode_id: str
    memory_bundle: MemoryBundle | None = None
    llm_memory_bundle: MemoryBundle | None = None
    contextual_memory_bundles: dict[str, MemoryBundle] = field(default_factory=dict)


@runtime_checkable
class EnvironmentAdapter(Protocol):
    def bind_game(self, game_id: str) -> None:
        ...

    def observe(self, frames: list[Any], latest_frame: Any) -> Observation:
        ...

    def legal_actions(self, observation: Observation) -> list[str]:
        ...

    def fallback_action_name(self, observation: Observation) -> str:
        ...

    def fallback_decision(self, observation: Observation) -> ActionDecision:
        ...

    def default_metadata_for_action(self, action_name: str) -> dict[str, Any]:
        ...

    def to_game_action(self, decision: ActionDecision) -> Any:
        ...


@runtime_checkable
class Specialist(Protocol):
    def propose(self, context: ExecutiveContext) -> list[ActionProposal]:
        ...
