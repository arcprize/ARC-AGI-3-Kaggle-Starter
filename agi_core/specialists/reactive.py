from __future__ import annotations

from collections.abc import Iterable

from agi_core.types import ActionProposal, ExecutiveContext


class ReactiveSpecialist:
    source = "reactive"

    def propose(self, context: ExecutiveContext) -> list[ActionProposal]:
        observation = context.observation
        proposals: list[ActionProposal] = []
        if observation.game_state in context.config.reset_state_names:
            proposals.append(
                ActionProposal(
                    action_name="RESET",
                    source=self.source,
                    confidence=1.0,
                    reason="reactive reset for non-started or game-over state",
                    subgoal="re-enter-level",
                )
            )
            return proposals
        if observation.game_state == context.config.win_state_name:
            return proposals
        if not observation.legal_actions:
            if not bool(observation.derived_features.get("legal_actions_known", True)):
                return proposals
            fallback = context.adapter.fallback_action_name(observation)
            proposals.append(
                ActionProposal(
                    action_name=fallback,
                    source=self.source,
                    confidence=0.20,
                    reason="no explicit legal actions reported",
                )
            )
            return proposals
        if self._is_context_exhausted(context, observation.legal_actions):
            if "RESET" in observation.legal_actions:
                proposals.append(
                    ActionProposal(
                        action_name="RESET",
                        source=self.source,
                        confidence=0.65,
                        reason="reactive context exhausted reset",
                        subgoal="re-enter-level",
                    )
                )
            return proposals
        if context.memory.stagnation_steps >= 2:
            ranked_actions = self._rank_stall_break_actions(context, observation.legal_actions)
            for index, action_name in enumerate(ranked_actions[:3]):
                proposals.append(
                    ActionProposal(
                        action_name=action_name,
                        source=self.source,
                        confidence=self._stall_break_confidence(
                            context.memory.stagnation_steps,
                            index,
                        ),
                        reason="reactive stall breaker",
                        subgoal="break-stall",
                    )
                )
            if proposals:
                return proposals
        memory_baseline = self._memory_informed_baseline(context, observation.legal_actions)
        if memory_baseline is not None:
            proposals.append(memory_baseline)
            return proposals
        fallback = self._deterministic_baseline_action(context, observation.legal_actions)
        metadata = {}
        if hasattr(context.adapter, "default_metadata_for_action"):
            metadata = dict(context.adapter.default_metadata_for_action(fallback) or {})
        proposals.append(
            ActionProposal(
                action_name=fallback,
                source=self.source,
                confidence=0.40,
                reason="deterministic reactive baseline",
                subgoal="maintain-control-loop",
                metadata=metadata,
            )
        )
        return proposals

    def _rank_stall_break_actions(
        self,
        context: ExecutiveContext,
        legal_actions: Iterable[str],
    ) -> list[str]:
        observation = context.observation
        frame_signature = str(observation.derived_features.get("frame_signature", "") or "")
        recent_actions = context.memory.recent_actions()
        most_recent = recent_actions[-1] if recent_actions else None
        recommended_actions: list[str] = []
        suppressed_actions: set[str] = set()
        if context.memory_bundle is not None:
            recommended_actions = list(context.memory_bundle.recommended_actions or [])
            suppressed_actions = set(context.memory_bundle.suppressed_actions or [])
        blocked: list[tuple[tuple[int, int, float, float, float, int, str], str]] = []
        for index, action_name in enumerate(legal_actions):
            if action_name == "RESET" or action_name == most_recent:
                continue
            context_penalty = context.memory.context_stall_penalty(
                action_name,
                state_name=observation.game_state,
                frame_signature=frame_signature,
            )
            repeat_penalty = context.memory.repeated_failure_penalty(action_name)
            loop_penalty = context.memory.loop_penalty(action_name)
            recency_penalty = 1 if action_name in recent_actions[-4:] else 0
            suppressed_penalty = 1 if action_name in suppressed_actions else 0
            recommendation_bonus = (
                recommended_actions.index(action_name)
                if action_name in recommended_actions
                else len(recommended_actions) + 1
            )
            blocked.append(
                (
                    (
                        suppressed_penalty,
                        recommendation_bonus,
                        context_penalty,
                        repeat_penalty,
                        loop_penalty,
                        recency_penalty,
                        str(index).zfill(3),
                    ),
                    action_name,
                )
            )
        blocked.sort(key=lambda item: item[0])
        return [action_name for _, action_name in blocked]

    def _memory_informed_baseline(
        self,
        context: ExecutiveContext,
        legal_actions: Iterable[str],
    ) -> ActionProposal | None:
        if context.memory_bundle is None:
            return None
        legal = [action_name for action_name in legal_actions if action_name != "RESET"]
        if not legal:
            return None
        recommended = [
            action_name
            for action_name in (context.memory_bundle.recommended_actions or [])
            if action_name in legal
        ]
        if not recommended:
            return None
        suppressed = set(context.memory_bundle.suppressed_actions or [])
        for action_name in recommended:
            if action_name in suppressed:
                continue
            metadata = {}
            if hasattr(context.adapter, "default_metadata_for_action"):
                metadata = dict(context.adapter.default_metadata_for_action(action_name) or {})
            return ActionProposal(
                action_name=action_name,
                source=self.source,
                confidence=0.46,
                reason="memory-informed reactive baseline",
                subgoal="reuse-known-good-action",
                metadata=metadata,
            )
        return None

    def _stall_break_confidence(
        self,
        stagnation_steps: int,
        proposal_index: int,
    ) -> float:
        stagnation_decay = min(0.30, max(0, stagnation_steps - 2) * 0.03)
        return max(0.18, 0.55 - stagnation_decay - (proposal_index * 0.05))

    def _deterministic_baseline_action(
        self,
        context: ExecutiveContext,
        legal_actions: Iterable[str],
    ) -> str:
        adapter_choice = context.adapter.fallback_action_name(context.observation)
        legal = list(legal_actions)
        suppressed = set()
        if context.memory_bundle is not None:
            suppressed = set(context.memory_bundle.suppressed_actions or [])
        if adapter_choice in legal and adapter_choice not in suppressed:
            return adapter_choice
        for action_name in legal:
            if action_name == "RESET" or action_name in suppressed:
                continue
            return action_name
        if "RESET" in legal:
            return "RESET"
        return adapter_choice

    def _is_context_exhausted(
        self,
        context: ExecutiveContext,
        legal_actions: Iterable[str],
    ) -> bool:
        observation = context.observation
        candidate_actions = [action_name for action_name in legal_actions if action_name != "RESET"]
        if not candidate_actions:
            return False
        if context.memory_bundle is not None:
            if bool((context.memory_bundle.basin_status or {}).get("is_exhausted")):
                return True
            suppressed_actions = set(context.memory_bundle.suppressed_actions or [])
            if suppressed_actions and all(
                action_name in suppressed_actions for action_name in candidate_actions
            ):
                return True
        frame_signature = str(observation.derived_features.get("frame_signature", "") or "")
        if not frame_signature:
            return False
        exhausted = 0
        for action_name in candidate_actions:
            counts = context.memory.context_outcome_counts(
                action_name,
                state_name=observation.game_state,
                frame_signature=frame_signature,
            )
            if counts["progress"] > 0:
                continue
            if counts["failed"] > 0 or counts["stalled"] > 0:
                exhausted += 1
        return exhausted == len(candidate_actions)
