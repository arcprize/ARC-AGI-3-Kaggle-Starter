from __future__ import annotations

from agi_core.config import RuntimeConfig
from agi_core.memory import EpisodicMemory
from agi_core.types import ActionDecision, ExecutiveContext, GoalState


class Executive:
    def __init__(
        self,
        config: RuntimeConfig,
        reactive,
        symbolic,
        llm,
        critic,
    ) -> None:
        self.config = config
        self.reactive = reactive
        self.symbolic = symbolic
        self.llm = llm
        self.critic = critic

    def decide(self, context: ExecutiveContext) -> ActionDecision:
        context.hyperon.sync(context.observation, context.memory, context.goals)
        reactive_proposals = self.reactive.propose(context)
        symbolic_proposals = self.symbolic.propose(context)
        proposals = list(reactive_proposals) + list(symbolic_proposals)
        best_symbolic_confidence = max(
            (proposal.confidence for proposal in symbolic_proposals),
            default=0.0,
        )
        if best_symbolic_confidence < context.config.symbolic_confidence_gate:
            proposals.extend(self.llm.propose(context))
        context.hyperon.update_proposals(proposals)
        rescored = self.critic.rescore(context, proposals)
        for proposal in rescored:
            if proposal.legal:
                decision = ActionDecision(
                    action_name=proposal.action_name,
                    source=proposal.source,
                    confidence=proposal.confidence,
                    reason=proposal.reason,
                    subgoal=proposal.subgoal,
                    metadata=dict(proposal.metadata),
                    proposals=rescored,
                )
                context.memory.note_decision(decision)
                return decision
        fallback = context.adapter.fallback_decision(context.observation)
        fallback.proposals = rescored
        context.memory.note_decision(fallback)
        return fallback


class ExecutiveRuntime:
    def __init__(
        self,
        adapter,
        executive: Executive,
        memory: EpisodicMemory,
        hyperon,
        llm_backend,
        config: RuntimeConfig,
    ) -> None:
        self.adapter = adapter
        self.executive = executive
        self.memory = memory
        self.hyperon = hyperon
        self.llm_backend = llm_backend
        self.config = config

    def bind_game(self, game_id: str) -> None:
        self.adapter.bind_game(game_id)
        self.memory.bind_game(game_id)

    def decide(self, frames, latest_frame) -> ActionDecision:
        observation = self.adapter.observe(frames, latest_frame)
        self.memory.note_observation(observation)
        context = ExecutiveContext(
            observation=observation,
            goals=self.default_goals(),
            memory=self.memory,
            adapter=self.adapter,
            hyperon=self.hyperon,
            llm_backend=self.llm_backend,
            config=self.config,
            episode_id=observation.game_id,
        )
        return self.executive.decide(context)

    def default_goals(self) -> list[GoalState]:
        return [
            GoalState(
                name="finish-episode",
                target_state=self.config.win_state_name,
                priority=1.0,
            ),
            GoalState(name="avoid-reset-loop", priority=0.8),
            GoalState(name="prefer-forward-progress", priority=0.9),
        ]
