from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .image_input import BoardImageInput


@dataclass(slots=True)
class Observation:
    game_id: str
    step_index: int
    game_state: str
    levels_completed: int
    last_action: str | None
    legal_actions: list[str]
    image_input: BoardImageInput | None = None
    raw_frame_ref: Any = None
    derived_features: dict[str, Any] = field(default_factory=dict)
    perception_confidence: float = 1.0


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
class AgentProposal:
    """A proposal from a specific agent in the multi-agent system.
    
    Extends ActionProposal with agent metadata for consensus weighting.
    """
    action_name: str
    source: str
    confidence: float
    reason: str
    agent_id: str = ""
    agent_confidence: float = 0.0
    agent_goal_depth: int = 0
    subgoal: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    expected_outcome: str | None = None
    legal: bool = True
    score: float = 0.0


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
    prior_board_height: int = 0
    prior_board_width: int = 0
    prior_board_active_cells: int = 0
    prior_board_source_attr: str | None = None
    prior_board_default_probe: dict[str, int] = field(default_factory=dict)
    recording_guid: str = ""
    cognition_strategy: str = ""
    cognition_confidence: float = 0.0
    wm_load: float = 0.0
    goal_depth: int = 0
    goal_progress: float = 0.0


@dataclass(slots=True)
class MemoryBundle:
    working_memory: dict[str, Any] = field(default_factory=dict)
    episodic_recalls: list[dict[str, Any]] = field(default_factory=list)
    semantic_memories: list[dict[str, Any]] = field(default_factory=list)
    procedural_memories: list[dict[str, Any]] = field(default_factory=list)
    abstract_memories: list[dict[str, Any]] = field(default_factory=list)
    retrieval_summary: dict[str, Any] = field(default_factory=dict)
    board_context_summary: dict[str, Any] = field(default_factory=dict)
    board_context_action_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    pattern_action_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    sequence_action_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    board_cluster_action_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    episode_summaries: list[dict[str, Any]] = field(default_factory=list)
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
    llm_subgoal_hint: str | None = None
    contextual_memory_bundles: dict[str, MemoryBundle] = field(default_factory=dict)
    force_llm_replan: bool = False
    replan_mode: bool = False
    probe_dead_end_replan: bool = False
    score_floor_proposals: list[Any] = field(default_factory=list)
    score_floor_debug: dict[str, Any] = field(default_factory=dict)
    cognition_strategy: str = "exploit"
    goal_stack: Any = None
    working_memory: Any = None
    chunk_library: Any = None
    attention: Any = None
    episodic_memory: Any = None
    multi_agent_frame_cache: dict[str, Any] = field(default_factory=dict)


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
