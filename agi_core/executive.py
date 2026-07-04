from __future__ import annotations

from agi_core.config import RuntimeConfig
from agi_core.image_input import (
    active_cell_count,
    board_probe_click_target,
    canonical_click_target_bonus,
    click_target_bias,
    default_click_target,
    preferred_board_probe_action,
)
from agi_core.memory import EpisodicMemory
from agi_core.types import ActionDecision, ExecutiveContext, GoalState, MemoryBundle


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
        context.llm_subgoal_hint = None
        context.contextual_memory_bundles = {}
        pending_reopen = context.memory.pending_score_floor_reopen(context.observation)
        if pending_reopen is not None and "RESET" in context.observation.legal_actions:
            context.score_floor_debug = dict(pending_reopen.get("score_floor_debug") or {})
            context.memory.clear_pending_score_floor_reopen()
            decision = ActionDecision(
                action_name="RESET",
                source="fallback",
                confidence=0.05,
                reason="score-floor pending frontier reopen reset",
                metadata={"score_floor_debug": dict(context.score_floor_debug)},
                used_fallback=True,
            )
            context.memory.note_decision(decision)
            return decision
        context.probe_dead_end_replan = self._should_force_probe_dead_end_llm_replan(context)
        context.force_llm_replan = self._should_force_emergency_llm_replan(context)
        replan_mode = self._should_enter_replan_mode(context)
        context.replan_mode = replan_mode
        reactive_proposals = [] if replan_mode else self.reactive.propose(context)
        symbolic_proposals = self.symbolic.propose(context)
        self._refresh_llm_memory_bundle(context, symbolic_proposals, reactive_proposals)
        proposals = list(reactive_proposals) + list(symbolic_proposals)
        score_floor_fallback_cooldown = context.memory.recent_score_floor_fallback(
            context.observation
        )
        best_symbolic_confidence = max(
            (proposal.confidence for proposal in symbolic_proposals),
            default=0.0,
        )
        skip_early_probe_llm = self._should_skip_early_probe_llm(
            context,
            symbolic_proposals,
        )
        if context.force_llm_replan and not context.llm_subgoal_hint:
            context.llm_subgoal_hint = "replan"
        if (
            (
                replan_mode
                or context.force_llm_replan
                or best_symbolic_confidence < context.config.symbolic_confidence_gate
            )
            and not score_floor_fallback_cooldown
            and not skip_early_probe_llm
        ):
            proposals.extend(self.llm.propose(context))
        context.hyperon.update_proposals(proposals)
        rescored = self.critic.rescore(context, proposals)
        if self._all_legal_proposals_are_score_floor(context, rescored):
            fallback = self._score_floor_fallback(context, rescored)
            fallback.proposals = rescored
            context.memory.note_decision(fallback)
            return fallback
        probe_dead_end_escape = self._probe_dead_end_escape(context, rescored)
        if probe_dead_end_escape is not None:
            probe_dead_end_escape.proposals = rescored
            context.memory.note_decision(probe_dead_end_escape)
            return probe_dead_end_escape
        probe_dead_end_followup_escape = self._probe_dead_end_followup_escape(
            context,
            rescored,
        )
        if probe_dead_end_followup_escape is not None:
            probe_dead_end_followup_escape.proposals = rescored
            context.memory.note_decision(probe_dead_end_followup_escape)
            return probe_dead_end_followup_escape
        for proposal in rescored:
            if proposal.legal:
                metadata = self._finalize_decision_metadata(
                    context,
                    proposal.action_name,
                    dict(proposal.metadata),
                )
                subgoal = self._finalize_decision_subgoal(
                    context,
                    proposal.action_name,
                    proposal.subgoal,
                )
                decision = ActionDecision(
                    action_name=proposal.action_name,
                    source=proposal.source,
                    confidence=proposal.confidence,
                    reason=proposal.reason,
                    subgoal=subgoal,
                    metadata=metadata,
                    proposals=rescored,
                )
                context.memory.note_decision(decision)
                return decision
        fallback = self._memory_aware_fallback(context)
        fallback.proposals = rescored
        context.memory.note_decision(fallback)
        return fallback

    def _all_legal_proposals_are_score_floor(
        self,
        context: ExecutiveContext,
        proposals: list,
    ) -> bool:
        legal = [
            proposal
            for proposal in proposals
            if getattr(proposal, "legal", False)
        ]
        if not legal:
            return False
        if all(float(getattr(proposal, "score", 0.0)) <= -0.95 for proposal in legal):
            return True
        if self._should_force_probe_recovery_score_floor(context, legal):
            return True
        if context.memory.stagnation_steps < 6:
            return False
        recent_actions = [
            action_name
            for action_name in context.memory.recent_actions()[-6:]
            if action_name and action_name != "RESET"
        ]
        diverse_stall_cycle = (
            len(recent_actions) >= 6
            and len(set(recent_actions)) >= 4
        )
        basin_status = {}
        if context.memory_bundle is not None:
            basin_status = dict(context.memory_bundle.basin_status or {})
        basin_exhausted = bool(basin_status.get("is_exhausted")) or context.memory.is_basin_exhausted(
            context.observation
        )
        if not basin_exhausted and not diverse_stall_cycle:
            return False
        if any(str(getattr(proposal, "source", "") or "") == "llm" for proposal in legal):
            return False
        return all(float(getattr(proposal, "score", 0.0)) <= 0.0 for proposal in legal)

    def _should_force_probe_recovery_score_floor(
        self,
        context: ExecutiveContext,
        legal_proposals: list,
    ) -> bool:
        if active_cell_count(context.observation.image_input) <= 0:
            return False
        if context.memory.stagnation_steps < 5:
            return False
        if any(str(getattr(proposal, "source", "") or "") == "llm" for proposal in legal_proposals):
            return False
        if not all(float(getattr(proposal, "score", 0.0)) <= 0.0 for proposal in legal_proposals):
            return False
        recent_actions = [
            action_name
            for action_name in context.memory.recent_actions()[-5:]
            if action_name and action_name != "RESET"
        ]
        if len(recent_actions) < 5:
            return False
        if recent_actions[:2] != ["ACTION6", "ACTION6"]:
            return False
        return len(set(recent_actions[2:])) >= 3

    def _should_skip_early_probe_llm(
        self,
        context: ExecutiveContext,
        symbolic_proposals: list,
    ) -> bool:
        del symbolic_proposals
        if context.force_llm_replan or context.replan_mode:
            return False
        if context.memory.stagnation_steps > 1:
            return False
        if context.memory.most_recent_action() != "ACTION6":
            return False
        if active_cell_count(context.observation.image_input) <= 0:
            return False
        return "ACTION6" in context.observation.legal_actions

    def _should_enter_replan_mode(self, context: ExecutiveContext) -> bool:
        basin_status = {}
        if context.memory_bundle is not None:
            basin_status = dict(context.memory_bundle.basin_status or {})
        basin_exhausted = bool(basin_status.get("is_exhausted")) or context.memory.is_basin_exhausted(
            context.observation
        )
        recent_actions = [
            action_name
            for action_name in context.memory.recent_actions()[-6:]
            if action_name and action_name != "RESET"
        ]
        diverse_stall_cycle = (
            context.memory.stagnation_steps >= 6
            and len(recent_actions) >= 6
            and len(set(recent_actions)) >= 4
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
            or diverse_stall_cycle
        ) and (
            context.observation.game_state not in context.config.reset_state_names
            and context.observation.game_state != context.config.win_state_name
        )

    def _should_force_emergency_llm_replan(self, context: ExecutiveContext) -> bool:
        if context.observation.game_state in context.config.reset_state_names:
            return False
        if context.observation.game_state == context.config.win_state_name:
            return False
        if "RESET" in context.observation.legal_actions:
            return False
        if context.probe_dead_end_replan:
            return True
        if context.memory.can_call_llm(context.config.llm_max_calls_per_episode):
            return False
        basin_status = {}
        if context.memory_bundle is not None:
            basin_status = dict(context.memory_bundle.basin_status or {})
        basin_exhausted = bool(basin_status.get("is_exhausted")) or context.memory.is_basin_exhausted(
            context.observation
        )
        if not basin_exhausted or context.memory.stagnation_steps < 6:
            return False
        recent_actions = [
            action_name
            for action_name in context.memory.recent_actions()[-8:]
            if action_name and action_name != "RESET"
        ]
        if len(recent_actions) < 6:
            return False
        return len(set(recent_actions)) >= 3

    def _should_force_probe_dead_end_llm_replan(
        self,
        context: ExecutiveContext,
    ) -> bool:
        if active_cell_count(context.observation.image_input) <= 0:
            return False
        if context.memory.stagnation_steps < 2:
            return False
        recent_actions = [
            action_name
            for action_name in context.memory.recent_actions()[-2:]
            if action_name
        ]
        if recent_actions != ["ACTION6", "ACTION6"]:
            return False
        return "ACTION6" in context.observation.legal_actions

    def _score_floor_fallback(self, context: ExecutiveContext, rescored: list) -> ActionDecision:
        candidate_actions = self._ordered_score_floor_actions(rescored)
        allow_board_probe = self._should_allow_score_floor_board_probe(
            context,
            candidate_actions,
        )
        if allow_board_probe and "ACTION6" in context.observation.legal_actions and "ACTION6" not in candidate_actions:
            candidate_actions.append("ACTION6")
        context.score_floor_proposals = list(rescored)
        reopen_summary = context.memory.replan_frontier_reopen_summary(
            context.observation,
            action_names=candidate_actions,
        )
        active_reopen_candidates = [
            action_name
            for action_name, count in dict(reopen_summary.get("action_counts") or {}).items()
            if int(count or 0) > 0
        ]
        active_reopen_summary: dict[str, object] = {}
        if 1 < len(active_reopen_candidates) < len(candidate_actions):
            active_reopen_summary = context.memory.replan_frontier_reopen_summary(
                context.observation,
                action_names=active_reopen_candidates,
            )
        context.score_floor_debug = dict(reopen_summary)
        if active_reopen_summary:
            context.score_floor_debug["active_reopen"] = dict(active_reopen_summary)
        reopen_detected = bool(reopen_summary.get("is_reopened")) or bool(
            active_reopen_summary.get("is_reopened")
        )
        if reopen_detected and "RESET" not in context.observation.legal_actions:
            context.score_floor_debug["pending_reset"] = True
            context.memory.mark_pending_score_floor_reopen(
                context.observation,
                candidate_actions=active_reopen_candidates or candidate_actions,
                debug=context.score_floor_debug,
            )
        if (
            "RESET" in context.observation.legal_actions
            and reopen_detected
        ):
            context.memory.clear_pending_score_floor_reopen()
            return ActionDecision(
                action_name="RESET",
                source="fallback",
                confidence=0.05,
                reason="score-floor frontier reopen reset",
                metadata={"score_floor_debug": dict(context.score_floor_debug)},
                used_fallback=True,
            )
        if (
            context.memory.is_replan_frontier_exhausted(
                context.observation,
                action_names=candidate_actions,
            )
            and "RESET" in context.observation.legal_actions
        ):
            return ActionDecision(
                action_name="RESET",
                source="fallback",
                confidence=0.05,
                reason="score-floor frontier exhausted reset",
                metadata={"score_floor_debug": dict(context.score_floor_debug)},
                used_fallback=True,
            )
        if self._should_reset_exhausted_score_floor(context, candidate_actions):
            return ActionDecision(
                action_name="RESET",
                source="fallback",
                confidence=0.05,
                reason="score-floor basin reset",
                metadata={"score_floor_debug": dict(context.score_floor_debug)},
                used_fallback=True,
            )
        memory_escape = self._score_floor_memory_escape(context)
        if memory_escape is not None:
            return memory_escape
        fallback = self._memory_aware_fallback(
            context,
            candidate_actions=candidate_actions,
            exploratory=True,
            respect_suppressed_actions=False,
            honor_recommended_actions=True,
            allow_exploratory_board_probe=allow_board_probe,
        )
        fallback.reason = "score-floor executive fallback"
        fallback.confidence = min(float(fallback.confidence or 0.05), 0.05)
        return fallback

    def _ordered_score_floor_actions(self, rescored: list) -> list[str]:
        ordered: list[str] = []
        for proposal in rescored:
            if not getattr(proposal, "legal", False):
                continue
            action_name = str(getattr(proposal, "action_name", "") or "")
            if not action_name or action_name in ordered:
                continue
            ordered.append(action_name)
        non_probe = [action_name for action_name in ordered if action_name != "ACTION6"]
        return non_probe or ordered

    def _should_allow_score_floor_board_probe(
        self,
        context: ExecutiveContext,
        candidate_actions: list[str],
    ) -> bool:
        if "ACTION6" not in context.observation.legal_actions:
            return False
        if active_cell_count(context.observation.image_input) <= 0:
            return False
        if self._board_probe_action_is_stale(context, "ACTION6"):
            return False
        recent_actions = [
            action_name
            for action_name in context.memory.recent_actions()[-6:]
            if action_name and action_name != "RESET"
        ]
        non_probe_attempts = [
            action_name for action_name in recent_actions if action_name != "ACTION6"
        ]
        if len(set(non_probe_attempts)) < min(3, max(len(candidate_actions) - 1, 1)):
            return False
        return True

    def _board_probe_action_is_stale(
        self,
        context: ExecutiveContext,
        action_name: str | None,
    ) -> bool:
        if not action_name:
            return False
        frame_signature = str(
            context.observation.derived_features.get("frame_signature", "") or ""
        ) or None
        counts = context.memory.context_outcome_counts(
            action_name,
            state_name=context.observation.game_state,
            frame_signature=frame_signature,
        )
        if counts["progress"] > 0:
            return False
        return counts["failed"] > 0 or counts["stalled"] >= 2

    def _should_reset_exhausted_score_floor(
        self,
        context: ExecutiveContext,
        candidate_actions: list[str],
    ) -> bool:
        if "RESET" not in context.observation.legal_actions:
            return False
        if not candidate_actions:
            return False
        if context.memory.stagnation_steps < max(4, len(candidate_actions)):
            return False
        if not context.memory.is_basin_exhausted(context.observation):
            return False
        recent_actions = context.memory.recent_actions()
        window = recent_actions[-max(len(candidate_actions) * 2, 8) :]
        if len(window) < len(candidate_actions):
            return False
        return all(action_name in window for action_name in candidate_actions)

    def _score_floor_memory_escape(
        self,
        context: ExecutiveContext,
    ) -> ActionDecision | None:
        llm_limit = context.config.llm_max_calls_per_episode
        if context.memory.llm_calls < (llm_limit + 1):
            return None
        if not hasattr(self.llm, "memory_escape_proposals"):
            return None
        preferred_actions = self._ordered_score_floor_actions(
            list(context.score_floor_proposals or [])
        )
        proposals = self.llm.memory_escape_proposals(
            context,
            "score-floor post-emergency memory escape",
            preferred_actions=preferred_actions or None,
        )
        if not proposals:
            return None
        if context.memory.is_replan_frontier_exhausted(
            context.observation,
            action_names=[proposal.action_name for proposal in proposals],
        ):
            return None
        proposal = proposals[0]
        metadata = dict(proposal.metadata)
        if context.score_floor_debug:
            metadata["score_floor_debug"] = dict(context.score_floor_debug)
        return ActionDecision(
            action_name=proposal.action_name,
            source="fallback",
            confidence=min(float(getattr(proposal, "confidence", 0.05) or 0.05), 0.05),
            reason="score-floor memory escape",
            subgoal=proposal.subgoal,
            metadata=metadata,
            used_fallback=True,
        )

    def _probe_dead_end_escape(
        self,
        context: ExecutiveContext,
        rescored: list,
    ) -> ActionDecision | None:
        if not context.probe_dead_end_replan:
            return None
        legal = [
            proposal
            for proposal in rescored
            if getattr(proposal, "legal", False)
            and getattr(proposal, "action_name", "") != "RESET"
        ]
        if not legal:
            return None
        top_legal = legal[0]
        if top_legal.source == "llm" and top_legal.action_name != "ACTION6":
            return None
        candidate_actions = self._probe_dead_end_escape_actions(context)
        if not candidate_actions:
            return None
        if hasattr(self.llm, "memory_escape_proposals"):
            proposals = self.llm.memory_escape_proposals(
                context,
                "probe-dead-end memory escape",
                preferred_actions=candidate_actions,
            )
            for proposal in proposals:
                if proposal.action_name not in candidate_actions:
                    continue
                metadata = self._finalize_decision_metadata(
                    context,
                    proposal.action_name,
                    dict(proposal.metadata),
                )
                return ActionDecision(
                    action_name=proposal.action_name,
                    source="fallback",
                    confidence=min(
                        float(getattr(proposal, "confidence", 0.05) or 0.05),
                        0.05,
                    ),
                    reason="probe-dead-end memory escape",
                    subgoal=proposal.subgoal,
                    metadata=metadata,
                    used_fallback=True,
                )
        fallback = self._memory_aware_fallback(
            context,
            candidate_actions=candidate_actions,
            exploratory=True,
            respect_suppressed_actions=False,
        )
        fallback.reason = "probe-dead-end executive escape"
        fallback.used_fallback = True
        return fallback

    def _probe_dead_end_followup_escape(
        self,
        context: ExecutiveContext,
        rescored: list,
    ) -> ActionDecision | None:
        if not context.memory.recent_probe_dead_end_escape(context.observation):
            return None
        legal = [
            proposal
            for proposal in rescored
            if getattr(proposal, "legal", False)
            and getattr(proposal, "action_name", "") != "RESET"
        ]
        if not legal:
            return None
        top_legal = legal[0]
        if top_legal.source == "llm" and top_legal.action_name != "ACTION6":
            return None
        candidate_actions = self._probe_dead_end_escape_actions(context)
        if not candidate_actions:
            return None
        fallback = self._memory_aware_fallback(
            context,
            candidate_actions=candidate_actions,
            exploratory=True,
            respect_suppressed_actions=False,
        )
        fallback.reason = "probe-dead-end followup escape"
        fallback.used_fallback = True
        return fallback

    def _probe_dead_end_escape_actions(
        self,
        context: ExecutiveContext,
    ) -> list[str]:
        candidates = [
            action_name
            for action_name in context.observation.legal_actions
            if action_name not in {"RESET", "ACTION6"}
        ]
        if not candidates:
            return []
        frame_signature = str(
            context.observation.derived_features.get("frame_signature", "") or ""
        ) or None
        recent_actions = set(context.memory.recent_actions()[-3:])
        return sorted(
            candidates,
            key=lambda action_name: (
                action_name in recent_actions,
                context.memory.context_outcome_counts(
                    action_name,
                    state_name=context.observation.game_state,
                    frame_signature=frame_signature,
                )["total"],
                action_name,
            ),
        )

    def _refresh_llm_memory_bundle(
        self,
        context: ExecutiveContext,
        symbolic_proposals: list,
        reactive_proposals: list,
    ) -> None:
        subgoal = self._select_contextual_subgoal(
            context,
            symbolic_proposals,
            reactive_proposals,
        )
        if not subgoal or not hasattr(context.hyperon, "contextual_memory_bundle"):
            return
        bundle = context.hyperon.contextual_memory_bundle(
            context.observation,
            subgoal,
            list(context.observation.legal_actions),
        )
        context.contextual_memory_bundles[subgoal] = bundle
        context.llm_memory_bundle = bundle
        context.llm_subgoal_hint = subgoal

    def _select_contextual_subgoal(
        self,
        context: ExecutiveContext,
        symbolic_proposals: list,
        reactive_proposals: list,
    ) -> str | None:
        ranked = sorted(
            [
                proposal
                for proposal in [*symbolic_proposals, *reactive_proposals]
                if getattr(proposal, "subgoal", None)
            ],
            key=lambda proposal: (
                self._contextual_board_probe_priority(context, proposal),
                float(getattr(proposal, "confidence", 0.0)),
            ),
            reverse=True,
        )
        if not ranked:
            return None
        return str(ranked[0].subgoal)

    def _contextual_board_probe_priority(
        self,
        context: ExecutiveContext,
        proposal,
    ) -> int:
        image_input = context.observation.image_input
        if active_cell_count(image_input) <= 0:
            return 0
        if getattr(proposal, "action_name", "") != "ACTION6":
            return 0
        metadata = getattr(proposal, "metadata", {})
        if click_target_bias(image_input, metadata) < 0:
            return 0
        if canonical_click_target_bonus(image_input, metadata) > 0:
            return 2
        return 1

    def _finalize_decision_metadata(
        self,
        context: ExecutiveContext,
        action_name: str,
        metadata: dict[str, object],
    ) -> dict[str, object]:
        if action_name != "ACTION6":
            return metadata
        image_input = context.observation.image_input
        if active_cell_count(image_input) <= 0:
            return metadata
        if canonical_click_target_bonus(image_input, metadata) > 0:
            return metadata
        return self._board_probe_metadata_for_context(context)

    def _finalize_decision_subgoal(
        self,
        context: ExecutiveContext,
        action_name: str,
        subgoal: str | None,
    ) -> str | None:
        if action_name != "ACTION6":
            return subgoal
        if active_cell_count(context.observation.image_input) <= 0:
            return subgoal
        normalized = str(subgoal or "")
        if normalized and normalized not in {
            "maintain-control-loop",
            "reuse-known-good-action",
        }:
            return subgoal
        return "inspect-active-board"

    def _memory_aware_fallback(
        self,
        context: ExecutiveContext,
        *,
        candidate_actions: list[str] | None = None,
        exploratory: bool = False,
        respect_suppressed_actions: bool = True,
        honor_recommended_actions: bool = False,
        allow_exploratory_board_probe: bool = False,
    ) -> ActionDecision:
        bundle = context.memory_bundle or MemoryBundle()
        legal_actions = list(candidate_actions or context.observation.legal_actions)
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
        suppressed = (
            set(bundle.suppressed_actions or []) if respect_suppressed_actions else set()
        )
        non_reset_legal = [
            action_name
            for action_name in legal_actions
            if action_name != "RESET" and action_name not in suppressed
        ]
        if exploratory:
            exhausted_replan_actions = context.memory.replan_frontier_exhausted_actions(
                context.observation,
                action_names=non_reset_legal,
            )
            if exhausted_replan_actions and len(
                set(non_reset_legal) - exhausted_replan_actions
            ) > 0:
                non_reset_legal = [
                    action_name
                    for action_name in non_reset_legal
                    if action_name not in exhausted_replan_actions
                ]
            alternative_actions = [
                action_name for action_name in non_reset_legal if action_name != "ACTION6"
            ]
            if alternative_actions and not allow_exploratory_board_probe:
                non_reset_legal = alternative_actions
            attempted_fallback_actions = context.memory.fallback_attempted_actions(
                context.observation,
                action_names=non_reset_legal,
            )
            if allow_exploratory_board_probe:
                attempted_fallback_actions.discard("ACTION6")
            if attempted_fallback_actions and len(set(non_reset_legal) - attempted_fallback_actions) > 0:
                non_reset_legal = [
                    action_name
                    for action_name in non_reset_legal
                    if action_name not in attempted_fallback_actions
                ]
        if non_reset_legal:
            board_probe_action = preferred_board_probe_action(
                non_reset_legal,
                context.observation.image_input,
                suppressed_actions=suppressed,
            )
            recommended_rank = {
                action_name: index
                for index, action_name in enumerate(bundle.recommended_actions or [])
                if action_name in non_reset_legal
            }
            action_name = max(
                non_reset_legal,
                key=lambda candidate: self._fallback_action_score(
                    context,
                    candidate,
                    recommended_rank=recommended_rank.get(candidate),
                    board_probe_action=board_probe_action,
                    exploratory=exploratory,
                    honor_recommended_actions=honor_recommended_actions,
                ),
            )
            metadata = {}
            if hasattr(context.adapter, "default_metadata_for_action"):
                metadata = dict(
                    context.adapter.default_metadata_for_action(action_name) or {}
                )
            if action_name == "ACTION6" and context.observation.image_input is not None:
                metadata = self._board_probe_metadata_for_context(context)
            return ActionDecision(
                action_name=action_name,
                source="fallback",
                confidence=0.08,
                reason=(
                    "memory-aware executive fallback"
                    if action_name in recommended_rank
                    else "memory-aware legal fallback"
                ),
                subgoal=(
                    "inspect-active-board"
                    if action_name == "ACTION6" and context.observation.image_input is not None
                    else None
                ),
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

    def _board_probe_metadata_for_context(
        self,
        context: ExecutiveContext,
    ) -> dict[str, int]:
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

    def _fallback_action_score(
        self,
        context: ExecutiveContext,
        action_name: str,
        *,
        recommended_rank: int | None,
        board_probe_action: str | None,
        exploratory: bool,
        honor_recommended_actions: bool,
    ) -> float:
        recent_actions = context.memory.recent_actions()
        score = 0.0
        if recommended_rank is not None and honor_recommended_actions:
            if exploratory:
                score += max(0.06, 0.18 - (recommended_rank * 0.05))
            else:
                score += max(0.20, 0.45 - (recommended_rank * 0.10))
        if board_probe_action and action_name == board_probe_action:
            score += 0.12 if exploratory else 0.18
        if action_name == context.memory.most_recent_action():
            score -= 0.08
        score += self._fallback_recency_bonus(recent_actions, action_name, exploratory=exploratory)
        score -= context.memory.repeated_failure_penalty(action_name)
        score -= context.memory.loop_penalty(action_name)
        score -= context.memory.context_stall_penalty(
            action_name,
            state_name=context.observation.game_state,
            frame_signature=str(
                context.observation.derived_features.get("frame_signature", "") or ""
            )
            or None,
        )
        return score

    def _fallback_recency_bonus(
        self,
        recent_actions: list[str],
        action_name: str,
        *,
        exploratory: bool,
    ) -> float:
        if not recent_actions:
            return 0.0
        tail = recent_actions[-8:]
        for distance, recent_action in enumerate(reversed(tail), start=1):
            if recent_action != action_name:
                continue
            if exploratory:
                return -max(0.0, 0.28 - ((distance - 1) * 0.05))
            return -max(0.0, 0.10 - ((distance - 1) * 0.02))
        return 0.18 if exploratory else 0.04


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
