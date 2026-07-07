from __future__ import annotations

from collections.abc import Iterable

from agi_core.image_input import board_probe_click_target, default_click_target, preferred_board_probe_action
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
            metadata = {}
            if hasattr(context.adapter, "default_metadata_for_action"):
                metadata = dict(context.adapter.default_metadata_for_action(fallback) or {})
            if fallback == "ACTION6" and observation.image_input is not None:
                metadata = self._board_probe_metadata(context)
            proposals.append(
                ActionProposal(
                    action_name=fallback,
                    source=self.source,
                    confidence=0.20,
                    reason="no explicit legal actions reported",
                    subgoal=self._subgoal_for_action(
                        observation.image_input,
                        fallback,
                        default="maintain-control-loop",
                    ),
                    metadata=metadata,
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
                metadata = {}
                if hasattr(context.adapter, "default_metadata_for_action"):
                    metadata = dict(context.adapter.default_metadata_for_action(action_name) or {})
                if action_name == "ACTION6" and observation.image_input is not None:
                    metadata = self._board_probe_metadata(context)
                proposals.append(
                    ActionProposal(
                        action_name=action_name,
                        source=self.source,
                        confidence=self._stall_break_confidence(
                            context.memory.stagnation_steps,
                            index,
                        ),
                        reason="reactive stall breaker",
                        subgoal=self._subgoal_for_action(
                            observation.image_input,
                            action_name,
                            default="break-stall",
                        ),
                        metadata=metadata,
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
        if fallback == "ACTION6" and observation.image_input is not None:
            metadata = self._board_probe_metadata(context)
        proposals.append(
            ActionProposal(
                action_name=fallback,
                source=self.source,
                confidence=0.40,
                reason="deterministic reactive baseline",
                subgoal=self._subgoal_for_action(
                    observation.image_input,
                    fallback,
                    default="maintain-control-loop",
                ),
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
        frame_signature = ""
        if observation.image_input is not None:
            frame_signature = str(observation.image_input.frame_signature or "")
        if not frame_signature:
            frame_signature = str(observation.derived_features.get("frame_signature", "") or "")
        recent_actions = context.memory.recent_actions()
        most_recent = recent_actions[-1] if recent_actions else None
        recommended_actions: list[str] = []
        suppressed_actions: set[str] = set()
        procedural_bias: dict[str, float] = {}
        recent_failure_actions: set[str] = set()
        failure_streak_counts: dict[str, int] = {}
        focus_rank: dict[str, int] = {}
        trained_prior_rank: dict[str, int] = {}
        if context.memory_bundle is not None:
            recommended_actions = list(context.memory_bundle.recommended_actions or [])
            suppressed_actions = set(context.memory_bundle.suppressed_actions or [])
            trained_prior_rank = self._trained_prior_rank(
                context.memory_bundle.trained_prior_bucket,
                legal_actions,
            )
            for row in context.memory_bundle.procedural_memories or []:
                action_name = str(row.get("action_name", "") or "").upper()
                if not action_name:
                    continue
                procedural_bias[action_name] = max(
                    procedural_bias.get(action_name, 0.0),
                    min(0.20, float(row.get("confidence", 0.0) or 0.0) * 0.22),
                )
            working_memory = dict(context.memory_bundle.working_memory or {})
            for row in list(working_memory.get("open_failure_modes", []) or []):
                action_name = str(row.get("action_name", "") or "").upper()
                if action_name:
                    recent_failure_actions.add(action_name)
            for row in list(working_memory.get("recent_failures", []) or []):
                action_name = str(row.get("action_name", "") or "").upper()
                if action_name:
                    recent_failure_actions.add(action_name)
            for row in list(working_memory.get("failure_streaks", []) or []):
                action_name = str(row.get("action_name", "") or "").upper()
                if not action_name:
                    continue
                failure_streak_counts[action_name] = max(
                    failure_streak_counts.get(action_name, 0),
                    int(row.get("count", 0) or 0),
                )
            for index, action_name in enumerate(
                list(working_memory.get("procedural_focus_actions", []) or [])
            ):
                normalized = str(action_name or "").upper()
                if not normalized:
                    continue
                focus_rank[normalized] = min(focus_rank.get(normalized, index), index)
            for index, action_name in enumerate(
                list(working_memory.get("semantic_focus_actions", []) or [])
            ):
                normalized = str(action_name or "").upper()
                if not normalized:
                    continue
                focus_rank[normalized] = min(focus_rank.get(normalized, index), index)
        board_probe_action = preferred_board_probe_action(
            legal_actions,
            observation.image_input,
            suppressed_actions=suppressed_actions,
        )
        blocked: list[tuple[tuple[int, int, int, float, float, float, int, str], str]] = []
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
            trained_prior_bonus = trained_prior_rank.get(
                action_name,
                len(trained_prior_rank) + 1,
            )
            board_probe_penalty = 0 if action_name == board_probe_action else 1
            procedural_bonus = -procedural_bias.get(action_name, 0.0)
            failure_penalty = 1 if action_name in recent_failure_actions else 0
            failure_streak_penalty = failure_streak_counts.get(action_name, 0)
            focus_penalty = focus_rank.get(action_name, len(focus_rank) + 1)
            blocked.append(
                (
                    (
                        suppressed_penalty,
                        recommendation_bonus,
                        trained_prior_bonus,
                        focus_penalty,
                        board_probe_penalty,
                        failure_penalty,
                        failure_streak_penalty,
                        procedural_bonus,
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
        trained_prior_actions = [
            action_name
            for action_name in self._trained_prior_order(
                context.memory_bundle.trained_prior_bucket,
                legal,
            )
            if action_name not in recommended
        ]
        suppressed = set(context.memory_bundle.suppressed_actions or [])
        working_memory = dict(context.memory_bundle.working_memory or {})
        focus_actions = [
            str(action_name)
            for action_name in (
                list(working_memory.get("procedural_focus_actions", []) or [])
                + list(working_memory.get("semantic_focus_actions", []) or [])
            )
            if str(action_name)
        ]
        ranked_candidates = recommended or trained_prior_actions or focus_actions
        if not ranked_candidates:
            return None
        for action_name in recommended:
            if action_name in suppressed:
                continue
            metadata = {}
            if hasattr(context.adapter, "default_metadata_for_action"):
                metadata = dict(context.adapter.default_metadata_for_action(action_name) or {})
            if action_name == "ACTION6" and context.observation.image_input is not None:
                metadata = self._board_probe_metadata(context)
            return ActionProposal(
                action_name=action_name,
                source=self.source,
                confidence=0.46,
                reason="memory-informed reactive baseline",
                subgoal=self._subgoal_for_action(
                    context.observation.image_input,
                    action_name,
                    default="reuse-known-good-action",
                ),
                metadata=metadata,
            )
        for action_name in trained_prior_actions:
            if action_name in suppressed:
                continue
            metadata = {}
            if hasattr(context.adapter, "default_metadata_for_action"):
                metadata = dict(context.adapter.default_metadata_for_action(action_name) or {})
            if action_name == "ACTION6" and context.observation.image_input is not None:
                metadata = self._board_probe_metadata(context)
            return ActionProposal(
                action_name=action_name,
                source=self.source,
                confidence=0.44,
                reason="trained-prior reactive baseline",
                subgoal=self._subgoal_for_action(
                    context.observation.image_input,
                    action_name,
                    default="reuse-known-good-action",
                ),
                metadata=metadata,
            )
        seen_focus: set[str] = set()
        for action_name in focus_actions:
            if action_name in seen_focus:
                continue
            seen_focus.add(action_name)
            if action_name not in legal:
                continue
            if action_name in suppressed:
                continue
            metadata = {}
            if hasattr(context.adapter, "default_metadata_for_action"):
                metadata = dict(context.adapter.default_metadata_for_action(action_name) or {})
            if action_name == "ACTION6" and context.observation.image_input is not None:
                metadata = self._board_probe_metadata(context)
            return ActionProposal(
                action_name=action_name,
                source=self.source,
                confidence=0.42,
                reason="memory-focus reactive baseline",
                subgoal=self._subgoal_for_action(
                    context.observation.image_input,
                    action_name,
                    default="reuse-known-good-action",
                ),
                metadata=metadata,
            )
        return None

    def _trained_prior_order(
        self,
        trained_prior_bucket: dict[str, object] | None,
        legal_actions: Iterable[str],
    ) -> list[str]:
        legal = {str(action_name) for action_name in legal_actions if str(action_name)}
        ranked: list[tuple[int, str]] = []
        for action_name, count in dict(trained_prior_bucket or {}).items():
            normalized = str(action_name or "")
            value = int(count or 0)
            if not normalized or normalized not in legal or value <= 0:
                continue
            ranked.append((-value, normalized))
        ranked.sort()
        return [action_name for _, action_name in ranked]

    def _trained_prior_rank(
        self,
        trained_prior_bucket: dict[str, object] | None,
        legal_actions: Iterable[str],
    ) -> dict[str, int]:
        return {
            action_name: index
            for index, action_name in enumerate(
                self._trained_prior_order(trained_prior_bucket, legal_actions)
            )
        }

    def _subgoal_for_action(
        self,
        image_input,
        action_name: str,
        *,
        default: str,
    ) -> str:
        if action_name == "ACTION6" and image_input is not None:
            return "inspect-active-board"
        return default

    def _stall_break_confidence(
        self,
        stagnation_steps: int,
        proposal_index: int,
    ) -> float:
        stagnation_decay = min(0.30, max(0, stagnation_steps - 2) * 0.03)
        return max(0.18, 0.55 - stagnation_decay - (proposal_index * 0.05))

    def _board_probe_metadata(self, context: ExecutiveContext) -> dict[str, int]:
        image_input = context.observation.image_input
        if image_input is None:
            return default_click_target(image_input)
        frame_signature = str(
            context.observation.derived_features.get("frame_signature", "") or ""
        ) or None
        counts = context.memory.context_outcome_counts(
            "ACTION6",
            state_name=context.observation.game_state,
            frame_signature=frame_signature,
        )
        attempt_index = int(counts.get("failed", 0) or 0) + int(counts.get("stalled", 0) or 0)
        return board_probe_click_target(image_input, attempt_index=attempt_index)

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
        board_probe_action = preferred_board_probe_action(
            legal,
            context.observation.image_input,
            suppressed_actions=suppressed,
        )
        if board_probe_action is not None:
            return board_probe_action
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
        frame_signature = ""
        if observation.image_input is not None:
            frame_signature = str(observation.image_input.frame_signature or "")
        if not frame_signature:
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
