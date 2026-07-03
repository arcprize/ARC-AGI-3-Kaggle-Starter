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
        context.memory_bundle = context.hyperon.current_memory_bundle()
        context.llm_memory_bundle = context.memory_bundle
        context.contextual_memory_bundles = {}
        replan_mode = self._should_enter_replan_mode(context)
        reactive_proposals = [] if replan_mode else self.reactive.propose(context)
        symbolic_proposals = self.symbolic.propose(context)
        self._refresh_llm_memory_bundle(context, symbolic_proposals, reactive_proposals)
        proposals = list(reactive_proposals) + list(symbolic_proposals)
        best_symbolic_confidence = max(
            (proposal.confidence for proposal in symbolic_proposals),
            default=0.0,
        )
        if replan_mode or best_symbolic_confidence < context.config.symbolic_confidence_gate:
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
        fallback = self._memory_aware_fallback(context)
        fallback.proposals = rescored
        context.memory.note_decision(fallback)
        return fallback

    def _should_enter_replan_mode(self, context: ExecutiveContext) -> bool:
        basin_status = {}
        if context.memory_bundle is not None:
            basin_status = dict(context.memory_bundle.basin_status or {})
        basin_exhausted = bool(basin_status.get("is_exhausted")) or context.memory.is_basin_exhausted(
            context.observation
        )
        return (
            (
                context.memory.stagnation_steps
                >= context.config.executive_replan_stagnation_threshold
            )
            or (
                basin_exhausted
                and context.memory.stagnation_steps >= 4
            )
        ) and (
            context.observation.game_state not in context.config.reset_state_names
            and context.observation.game_state != context.config.win_state_name
        )

    def _refresh_llm_memory_bundle(
        self,
        context: ExecutiveContext,
        symbolic_proposals: list,
        reactive_proposals: list,
    ) -> None:
        subgoal = self._select_contextual_subgoal(symbolic_proposals, reactive_proposals)
        if not subgoal or not hasattr(context.hyperon, "contextual_memory_bundle"):
            return
        bundle = context.hyperon.contextual_memory_bundle(
            context.observation,
            subgoal,
            list(context.observation.legal_actions),
        )
        context.contextual_memory_bundles[subgoal] = bundle
        context.llm_memory_bundle = bundle

    def _select_contextual_subgoal(
        self,
        symbolic_proposals: list,
        reactive_proposals: list,
    ) -> str | None:
        ranked = sorted(
            [
                proposal
                for proposal in [*symbolic_proposals, *reactive_proposals]
                if getattr(proposal, "subgoal", None)
            ],
            key=lambda proposal: float(getattr(proposal, "confidence", 0.0)),
            reverse=True,
        )
        if not ranked:
            return None
        return str(ranked[0].subgoal)

    def _memory_aware_fallback(self, context: ExecutiveContext) -> ActionDecision:
        bundle = context.memory_bundle
        legal_actions = list(context.observation.legal_actions)
        legal_actions_known = bool(
            context.observation.derived_features.get("legal_actions_known", True)
        )
        if not legal_actions and not legal_actions_known:
            return ActionDecision(
                action_name="RESET",
                source="fallback",
                confidence=0.01,
                reason="fail-closed unknown legality reset",
                used_fallback=True,
            )
        if bundle is not None:
            suppressed = set(bundle.suppressed_actions or [])
            for action_name in bundle.recommended_actions or []:
                if action_name in legal_actions and action_name not in suppressed:
                    metadata = {}
                    if hasattr(context.adapter, "default_metadata_for_action"):
                        metadata = dict(
                            context.adapter.default_metadata_for_action(action_name) or {}
                        )
                    return ActionDecision(
                        action_name=action_name,
                        source="fallback",
                        confidence=0.08,
                        reason="memory-aware executive fallback",
                        metadata=metadata,
                        used_fallback=True,
                    )
            non_reset_legal = [
                action_name
                for action_name in legal_actions
                if action_name != "RESET" and action_name not in suppressed
            ]
            if non_reset_legal:
                action_name = non_reset_legal[0]
                metadata = {}
                if hasattr(context.adapter, "default_metadata_for_action"):
                    metadata = dict(
                        context.adapter.default_metadata_for_action(action_name) or {}
                    )
                return ActionDecision(
                    action_name=action_name,
                    source="fallback",
                    confidence=0.08,
                    reason="memory-aware legal fallback",
                    metadata=metadata,
                    used_fallback=True,
                )
            if not non_reset_legal and "RESET" in legal_actions:
                return ActionDecision(
                    action_name="RESET",
                    source="fallback",
                    confidence=0.08,
                    reason="memory-aware exhausted fallback reset",
                    used_fallback=True,
                )
        return context.adapter.fallback_decision(context.observation)


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
        if hasattr(self.hyperon, "bind_game"):
            self.hyperon.bind_game(game_id)

    def decide(self, frames, latest_frame) -> ActionDecision:
        observation = self._observe_and_persist(frames, latest_frame)
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

    def note_terminal_observation(self, frames, latest_frame) -> None:
        self._observe_and_persist(frames, latest_frame)

    def _observe_and_persist(self, frames, latest_frame):
        observation = self.adapter.observe(frames, latest_frame)
        self.memory.note_observation(observation)
        for record in self.memory.drain_new_records():
            self.hyperon.record_transition(record)
        return observation

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
