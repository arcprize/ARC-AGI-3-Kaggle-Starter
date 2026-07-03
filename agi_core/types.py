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
    decision_subgoal: str | None = None
    decision_source: str | None = None


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

    def to_game_action(self, decision: ActionDecision) -> Any:
        ...


@runtime_checkable
class Specialist(Protocol):
    def propose(self, context: ExecutiveContext) -> list[ActionProposal]:
        ...
