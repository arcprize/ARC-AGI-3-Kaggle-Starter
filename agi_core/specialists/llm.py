from __future__ import annotations

import json
from typing import Any

from agi_core.image_input import (
    active_cell_count,
    board_probe_click_target,
    canonical_click_target_bonus,
    click_target_bias,
    default_click_target,
    preferred_board_probe_action,
)
from agi_core.types import ActionProposal, ExecutiveContext, MemoryBundle


class LLMSpecialist:
    source = "llm"

    def propose(self, context: ExecutiveContext) -> list[ActionProposal]:
        backend = context.llm_backend
        if backend is None or not getattr(backend, "available", False):
            return []
        llm_limit = context.config.llm_max_calls_per_episode
        force_llm_replan = bool(getattr(context, "force_llm_replan", False))
        if not context.memory.can_call_llm(llm_limit):
            if not force_llm_replan or context.memory.llm_calls >= (llm_limit + 1):
                return []
        if context.observation.game_state in context.config.reset_state_names:
            return []
        bundle = (
            context.llm_memory_bundle
            or context.memory_bundle
            or context.hyperon.current_memory_bundle()
        )
        context.memory.note_llm_call()
        payload = backend.generate_json(
            prompt=self._build_prompt(context),
            max_tokens=context.config.llm_max_tokens,
            temperature=context.config.llm_temperature,
        )
        if not isinstance(payload, dict):
            return self._memory_fallback_proposals(
                context,
                bundle,
                self._failure_reason_with_backend(context, "non-dict payload"),
            )
        legal_actions = set(context.observation.legal_actions)
        subgoal = payload.get("subgoal")
        llm_subgoal_hint = str(context.llm_subgoal_hint or "")
        base_reason = str(payload.get("reason", "llm proposal"))
        base_confidence = self._safe_float(payload.get("confidence", 0.50), 0.50)
        proposals: list[ActionProposal] = []
        candidates = payload.get("action_candidates", [])
        if not isinstance(candidates, list):
            return self._memory_fallback_proposals(
                context,
                bundle,
                self._failure_reason_with_backend(context, "invalid action_candidates"),
            )
        for raw_candidate in candidates[:3]:
            candidate = self._normalize_candidate(raw_candidate)
            action_name = candidate["action_name"]
            if action_name not in legal_actions:
                continue
            if getattr(context, "probe_dead_end_replan", False) and action_name == "ACTION6":
                continue
            metadata = dict(candidate.get("metadata", {}))
            if action_name == "ACTION6" and context.observation.image_input is not None:
                if (
                    not metadata
                    or click_target_bias(context.observation.image_input, metadata) < 0
                    or (
                        click_target_bias(context.observation.image_input, metadata) > 0
                        and canonical_click_target_bonus(
                            context.observation.image_input,
                            metadata,
                        )
                        <= 0
                    )
                ):
                    metadata = self._board_probe_metadata(context)
            candidate_subgoal = subgoal
            if (
                not candidate_subgoal
                and action_name == "ACTION6"
                and llm_subgoal_hint
            ):
                candidate_subgoal = llm_subgoal_hint
            proposals.append(
                ActionProposal(
                    action_name=action_name,
                    source=self.source,
                    confidence=self._safe_float(
                        candidate.get("confidence", base_confidence),
                        base_confidence,
                    ),
                    reason=str(candidate.get("reason", base_reason)),
                    subgoal=candidate_subgoal,
                    metadata=metadata,
                    expected_outcome=str(candidate.get("expected_outcome", "progress")),
                )
            )
        if proposals:
            proposals = self._drop_memory_blocked_llm_proposals(
                context,
                bundle,
                proposals,
            )
            if not proposals:
                return self._memory_fallback_proposals(
                    context,
                    bundle,
                    self._failure_reason_with_backend(
                        context,
                        "memory-blocked llm candidates",
                    ),
                )
            proposals = self._rerank_close_llm_proposals(
                context,
                bundle,
                proposals,
            )
            if self._should_replace_emergency_candidates_with_memory_escape(
                context,
                bundle,
                proposals,
            ):
                replacement = self._memory_fallback_proposals(
                    context,
                    bundle,
                    self._failure_reason_with_backend(
                        context,
                        "stale emergency llm candidates",
                    ),
                )
                if replacement:
                    return replacement
            return proposals
        return self._memory_fallback_proposals(
            context,
            bundle,
            self._failure_reason_with_backend(context, "no legal llm candidates"),
        )

    def memory_escape_proposals(
        self,
        context: ExecutiveContext,
        failure_reason: str,
        *,
        preferred_actions: list[str] | None = None,
    ) -> list[ActionProposal]:
        bundle = (
            context.llm_memory_bundle
            or context.memory_bundle
            or context.hyperon.current_memory_bundle()
        )
        return self._memory_fallback_proposals(
            context,
            bundle,
            failure_reason,
            preferred_actions=preferred_actions,
        )

    def _rerank_close_llm_proposals(
        self,
        context: ExecutiveContext,
        bundle: MemoryBundle,
        proposals: list[ActionProposal],
    ) -> list[ActionProposal]:
        if len(proposals) < 2:
            return proposals
        recommended_actions = [
            str(action_name or "")
            for action_name in list(bundle.recommended_actions or [])
            if str(action_name or "")
        ]
        suppressed_actions = {
            str(action_name or "")
            for action_name in list(bundle.suppressed_actions or [])
            if str(action_name or "")
        }
        working_memory = dict(bundle.working_memory or {})
        contextual_failure_actions = {
            str(proposal.action_name or "")
            for proposal in proposals
            if str(proposal.action_name or "")
            and self._has_contextual_failure_block(
                context,
                bundle,
                proposal,
            )
        }
        trained_prior_bucket = {
            str(action_name or ""): int(count or 0)
            for action_name, count in dict(bundle.trained_prior_bucket or {}).items()
            if str(action_name or "") and int(count or 0) > 0
        }
        prior_peak = max(trained_prior_bucket.values(), default=0)
        sorted_by_confidence = sorted(
            proposals,
            key=lambda proposal: float(proposal.confidence or 0.0),
            reverse=True,
        )
        leading_gap = (
            float(sorted_by_confidence[0].confidence or 0.0)
            - float(sorted_by_confidence[1].confidence or 0.0)
        )
        allowed_gap = 0.08
        if self._has_clear_trained_prior_leader(
            proposals,
            trained_prior_bucket=trained_prior_bucket,
            recommended_actions=recommended_actions,
        ):
            allowed_gap = 0.12
        elif self._has_clear_failure_demoted_leader(
            proposals,
            failure_actions=contextual_failure_actions,
            working_memory=working_memory,
        ):
            allowed_gap = 0.11
        elif self._has_clear_suppressed_raw_leader(
            proposals,
            suppressed_actions=suppressed_actions,
        ):
            allowed_gap = 0.12
        if leading_gap > allowed_gap:
            return proposals
        if (
            not recommended_actions
            and prior_peak <= 0
            and not contextual_failure_actions
            and not suppressed_actions
        ):
            return proposals
        ranked: list[tuple[tuple[float, float, int, str], int, ActionProposal]] = []
        for index, proposal in enumerate(proposals):
            action_name = str(proposal.action_name or "")
            memory_bonus = 0.0
            reason_tags: list[str] = []
            if action_name in recommended_actions:
                memory_bonus += max(
                    0.02,
                    0.06 - (recommended_actions.index(action_name) * 0.015),
                )
                reason_tags.append("memory-recommended")
            if action_name in suppressed_actions:
                memory_bonus -= 0.12
                reason_tags.append("suppression-demotion")
            if prior_peak > 0 and action_name in trained_prior_bucket:
                memory_bonus += 0.08 * (
                    float(trained_prior_bucket.get(action_name, 0)) / float(prior_peak)
                )
                reason_tags.append("trained-prior")
            if action_name in contextual_failure_actions:
                memory_bonus -= 0.10
                reason_tags.append("failure-demotion")
            if memory_bonus != 0.0:
                confidence_delta = 0.0
                if memory_bonus > 0.0:
                    confidence_delta = min(0.06, memory_bonus * 0.5)
                else:
                    confidence_delta = max(-0.05, memory_bonus * 0.35)
                proposal.confidence = min(
                    0.98,
                    max(0.01, float(proposal.confidence or 0.0) + confidence_delta),
                )
                if reason_tags:
                    proposal.reason = (
                        f"{proposal.reason}; memory-rerank={'+'.join(reason_tags)}"
                    )
            ranked.append(
                (
                    (
                        float(proposal.confidence or 0.0) + memory_bonus,
                        memory_bonus,
                        -index,
                        action_name,
                    ),
                    index,
                    proposal,
                )
            )
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [proposal for _, _, proposal in ranked]

    def _drop_memory_blocked_llm_proposals(
        self,
        context: ExecutiveContext,
        bundle: MemoryBundle,
        proposals: list[ActionProposal],
    ) -> list[ActionProposal]:
        suppressed_actions = {
            str(action_name or "")
            for action_name in list(bundle.suppressed_actions or [])
            if str(action_name or "")
        }
        failure_actions = self._working_memory_failure_actions(
            dict(bundle.working_memory or {})
        )
        blocked_flags: list[bool] = []
        blocked_candidates: list[ActionProposal] = []
        clean_candidates: list[ActionProposal] = []
        for proposal in proposals:
            action_name = str(proposal.action_name or "")
            blocked = action_name in suppressed_actions
            if not blocked and action_name in failure_actions:
                blocked = self._has_contextual_failure_block(
                    context,
                    bundle,
                    proposal,
                )
            blocked_flags.append(blocked)
            if blocked:
                blocked_candidates.append(proposal)
            else:
                clean_candidates.append(proposal)
        if blocked_flags and all(blocked_flags):
            return []
        if blocked_candidates and clean_candidates:
            top_blocked = max(
                float(proposal.confidence or 0.0)
                for proposal in blocked_candidates
            )
            top_clean = max(
                float(proposal.confidence or 0.0)
                for proposal in clean_candidates
            )
            if (top_blocked - top_clean) > 0.12:
                return clean_candidates
        return proposals

    def _has_contextual_failure_block(
        self,
        context: ExecutiveContext,
        bundle: MemoryBundle,
        proposal: ActionProposal,
    ) -> bool:
        action_name = str(proposal.action_name or "")
        if not action_name:
            return False
        working_memory = dict(bundle.working_memory or {})
        current_frame_signature = str(
            context.observation.derived_features.get("frame_signature", "") or ""
        )
        current_subgoal = str(
            proposal.subgoal
            or context.llm_subgoal_hint
            or working_memory.get("active_subgoal")
            or ""
        )
        has_detailed_action_row = False
        for row in list(working_memory.get("recent_failures", []) or []) + list(
            working_memory.get("open_failure_modes", []) or []
        ):
            if str(row.get("action_name", "") or "") != action_name:
                continue
            row_frame_signature = str(
                row.get("frame_signature", row.get("prior_frame_signature", "")) or ""
            )
            row_subgoal = str(
                row.get("subgoal", row.get("decision_subgoal", "")) or ""
            )
            if not row_frame_signature and not row_subgoal:
                continue
            has_detailed_action_row = True
            frame_matches = True
            if row_frame_signature:
                frame_matches = (
                    bool(current_frame_signature)
                    and row_frame_signature == current_frame_signature
                )
            subgoal_matches = True
            if row_subgoal:
                subgoal_matches = (
                    bool(current_subgoal)
                    and row_subgoal == current_subgoal
                )
            if frame_matches and subgoal_matches:
                return True
        if has_detailed_action_row:
            # A concrete mismatched failure context is stronger evidence than a
            # generic streak count; do not block on streaks alone in that case.
            return False
        failure_streaks = {
            str(row.get("action_name", "") or ""): int(row.get("count", 0) or 0)
            for row in list(working_memory.get("failure_streaks", []) or [])
            if str(row.get("action_name", "") or "")
        }
        if int(failure_streaks.get(action_name, 0) or 0) >= 2:
            return True
        return False

    def _has_clear_trained_prior_leader(
        self,
        proposals: list[ActionProposal],
        *,
        trained_prior_bucket: dict[str, int],
        recommended_actions: list[str],
    ) -> bool:
        if len(proposals) < 2 or not trained_prior_bucket:
            return False
        proposal_actions = {
            str(proposal.action_name or "")
            for proposal in proposals
            if str(proposal.action_name or "")
        }
        ranked = sorted(
            (
                (int(count or 0), str(action_name or ""))
                for action_name, count in trained_prior_bucket.items()
                if str(action_name or "") in proposal_actions and int(count or 0) > 0
            ),
            key=lambda item: (-item[0], item[1]),
        )
        if len(ranked) < 2:
            return False
        top_count, top_action = ranked[0]
        second_count = ranked[1][0]
        if top_count < 4 or top_count < (second_count * 2):
            return False
        if recommended_actions and top_action not in recommended_actions[:2]:
            return False
        return True

    def _has_clear_failure_demoted_leader(
        self,
        proposals: list[ActionProposal],
        *,
        failure_actions: set[str],
        working_memory: dict[str, Any],
    ) -> bool:
        if len(proposals) < 2 or not failure_actions:
            return False
        ranked = sorted(
            proposals,
            key=lambda proposal: float(proposal.confidence or 0.0),
            reverse=True,
        )
        top_action = str(ranked[0].action_name or "")
        if top_action not in failure_actions:
            return False
        if any(
            str(proposal.action_name or "") not in failure_actions
            for proposal in ranked[1:]
        ):
            return True
        failure_streaks = {
            str(row.get("action_name", "") or ""): int(row.get("count", 0) or 0)
            for row in list(working_memory.get("failure_streaks", []) or [])
            if str(row.get("action_name", "") or "")
        }
        return int(failure_streaks.get(top_action, 0) or 0) >= 3

    def _has_clear_suppressed_raw_leader(
        self,
        proposals: list[ActionProposal],
        *,
        suppressed_actions: set[str],
    ) -> bool:
        if len(proposals) < 2 or not suppressed_actions:
            return False
        ranked = sorted(
            proposals,
            key=lambda proposal: float(proposal.confidence or 0.0),
            reverse=True,
        )
        top_action = str(ranked[0].action_name or "")
        if top_action not in suppressed_actions:
            return False
        return any(
            str(proposal.action_name or "") not in suppressed_actions
            for proposal in ranked[1:]
        )

    def _build_prompt(self, context: ExecutiveContext) -> str:
        observation = context.observation
        memory = context.memory.summarize_for_hyperon()
        bundle = (
            context.llm_memory_bundle
            or context.memory_bundle
            or context.hyperon.current_memory_bundle()
        )
        basin_status = bundle.basin_status if bundle is not None else {}
        basin_exhausted = bool(basin_status.get("is_exhausted", False)) or context.memory.is_basin_exhausted(
            observation
        )
        image_input = observation.image_input
        board_payload = image_input.board if image_input is not None else []
        board_height = image_input.height if image_input is not None else 0
        board_width = image_input.width if image_input is not None else 0
        board_active_cells = active_cell_count(image_input)
        board_default_probe = (
            default_click_target(image_input) if image_input is not None else {}
        )
        board_source_attr = image_input.source_attr if image_input is not None else None
        board_frame_signature = image_input.frame_signature if image_input is not None else None
        llm_subgoal_hint = str(context.llm_subgoal_hint or "")
        prompt_state = {
            "game_id": observation.game_id,
            "step_index": observation.step_index,
            "game_state": observation.game_state,
            "levels_completed": observation.levels_completed,
            "legal_actions": list(observation.legal_actions),
            "board": {
                "height": board_height,
                "width": board_width,
                "active_cells": board_active_cells,
                "default_probe": board_default_probe,
                "source_attr": board_source_attr,
                "frame_signature": board_frame_signature,
                "matrix": board_payload,
            },
            "recent_actions": list(memory["recent_actions"][-8:]),
            "failure_streaks": self._compact_failure_streaks(memory["failure_streaks"]),
            "memory": self._compact_memory_bundle(bundle, basin_exhausted=basin_exhausted),
            "current_subgoal_hint": llm_subgoal_hint,
        }
        prompt_json = json.dumps(prompt_state, separators=(",", ":"), sort_keys=True)
        return (
            "You are a local action proposal model.\n"
            "Return JSON only. No markdown, no commentary.\n"
            'Use exactly this schema: {"subgoal":"...","confidence":0.0,"reason":"...",'
            '"action_candidates":[{"action":"ACTION1","confidence":0.0,"reason":"...","x":0,"y":0}]}.\n'
            "Include 1 to 3 legal action_candidates only.\n"
            f"state_json={prompt_json}\n"
            "Prefer one to three legal actions only.\n"
            "When board_active_cells>0 and ACTION6 is legal, use board_default_probe unless a better target is obvious.\n"
            "If current_subgoal_hint=inspect-active-board, prioritize ACTION6 with board_default_probe.\n"
            "If memory_basin_exhausted=true, do not repeat suppressed or recently stalled actions.\n"
            "If memory_trained_prior_focus or memory_trained_prior_bucket has a clear unsuppressed leader, prefer it unless recent failures strongly contradict it.\n"
            "When exhausted, prefer memory_recommended_actions or any unsuppressed legal action.\n"
        )

    def _compact_failure_streaks(self, failure_streaks: dict[str, Any]) -> dict[str, int]:
        ranked = sorted(
            (
                (str(action_name), int(count or 0))
                for action_name, count in dict(failure_streaks or {}).items()
                if int(count or 0) > 0
            ),
            key=lambda item: (-item[1], item[0]),
        )
        return {action_name: count for action_name, count in ranked[:6]}

    def _compact_memory_bundle(
        self,
        bundle: MemoryBundle | None,
        *,
        basin_exhausted: bool,
    ) -> dict[str, Any]:
        if bundle is None:
            return {
                "recommended_actions": [],
                "trained_prior_bucket": {},
                "suppressed_actions": [],
                "basin_exhausted": basin_exhausted,
                "basin_status": {},
                "working_memory": {},
                "episodic_recalls": [],
                "semantic_memories": [],
                "procedural_memories": [],
                "retrieval_summary": {},
                "episode_summaries": [],
                "recent_records": [],
                "pattern_action_stats": {},
                "sequence_action_stats": {},
                "board_cluster_action_stats": {},
                "board_context_action_stats": {},
                "basin_action_stats": {},
                "frame_action_stats": {},
                "state_action_stats": {},
                "subgoal_action_stats": {},
            }
        compact_trained_prior_bucket = self._compact_trained_prior_bucket(
            bundle.trained_prior_bucket
        )
        trained_prior_focus = [
            {"action_name": action_name, "count": count}
            for action_name, count in compact_trained_prior_bucket.items()
        ]
        compact_working_memory = self._compact_working_memory(bundle.working_memory)
        if not list(compact_working_memory.get("trained_prior_focus", []) or []):
            compact_working_memory["trained_prior_focus"] = list(trained_prior_focus)
        compact_retrieval_summary = self._compact_retrieval_summary(
            bundle.retrieval_summary
        )
        if not list(compact_retrieval_summary.get("trained_prior_focus", []) or []):
            compact_retrieval_summary["trained_prior_focus"] = list(trained_prior_focus)
        return {
            "recommended_actions": list(bundle.recommended_actions[:3]),
            "trained_prior_bucket": compact_trained_prior_bucket,
            "suppressed_actions": list(bundle.suppressed_actions[:6]),
            "basin_exhausted": basin_exhausted,
            "basin_status": self._compact_basin_status(bundle.basin_status),
            "working_memory": compact_working_memory,
            "episodic_recalls": self._compact_memory_rows(bundle.episodic_recalls),
            "semantic_memories": self._compact_memory_rows(bundle.semantic_memories),
            "procedural_memories": self._compact_memory_rows(bundle.procedural_memories),
            "retrieval_summary": compact_retrieval_summary,
            "episode_summaries": self._compact_episode_summaries(bundle.episode_summaries),
            "recent_records": self._compact_recent_records(bundle.recent_records),
            "pattern_action_stats": self._compact_action_stats(bundle.pattern_action_stats),
            "sequence_action_stats": self._compact_action_stats(bundle.sequence_action_stats),
            "board_cluster_action_stats": self._compact_action_stats(bundle.board_cluster_action_stats),
            "board_context_action_stats": self._compact_action_stats(bundle.board_context_action_stats),
            "basin_action_stats": self._compact_action_stats(bundle.basin_action_stats),
            "frame_action_stats": self._compact_action_stats(bundle.frame_action_stats),
            "state_action_stats": self._compact_action_stats(bundle.state_action_stats),
            "subgoal_action_stats": self._compact_action_stats(bundle.subgoal_action_stats),
        }

    def _compact_trained_prior_bucket(
        self,
        bucket: dict[str, Any],
    ) -> dict[str, int]:
        ranked: list[tuple[str, int]] = []
        for action_name, count in dict(bucket or {}).items():
            name = str(action_name or "")
            value = int(count or 0)
            if not name or value <= 0:
                continue
            ranked.append((name, value))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return {action_name: count for action_name, count in ranked[:4]}

    def _compact_episode_summaries(
        self,
        summaries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for summary in list(summaries or [])[:4]:
            normalized = dict(summary or {})
            compact.append(
                {
                    "prior_game_state": str(normalized.get("prior_game_state", "") or ""),
                    "prior_frame_signature": str(
                        normalized.get("prior_frame_signature", "") or ""
                    ),
                    "decision_subgoal": str(normalized.get("decision_subgoal", "") or ""),
                    "first_step_index": int(normalized.get("first_step_index", 0) or 0),
                    "last_step_index": int(normalized.get("last_step_index", 0) or 0),
                    "total_events": int(normalized.get("total_events", 0) or 0),
                    "recent_action_sequence": list(
                        normalized.get("recent_action_sequence", []) or []
                    )[:6],
                    "recent_outcome_sequence": list(
                        normalized.get("recent_outcome_sequence", []) or []
                    )[:6],
                    "decision_sources": {
                        str(key): int(value or 0)
                        for key, value in dict(
                            normalized.get("decision_sources", {}) or {}
                        ).items()
                    },
                    "max_stagnation_after": int(
                        normalized.get("max_stagnation_after", 0) or 0
                    ),
                    "board_context": self._compact_board_context(
                        normalized.get("board_context_summary", {})
                    ),
                    "action_outcomes": self._compact_action_stats(
                        normalized.get("action_outcomes", {})
                    ),
                }
            )
        return compact

    def _compact_working_memory(self, working_memory: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(working_memory or {})
        return {
            "active_state": str(normalized.get("active_state", "") or ""),
            "active_frame_signature": str(normalized.get("active_frame_signature", "") or ""),
            "active_subgoal": str(normalized.get("active_subgoal", "") or ""),
            "board_context_summary": self._compact_board_context(
                normalized.get("board_context_summary", {})
            ),
            "recent_action_window": list(normalized.get("recent_action_window", []) or [])[:8],
            "recommended_actions": list(normalized.get("recommended_actions", []) or [])[:4],
            "trained_prior_focus": list(normalized.get("trained_prior_focus", []) or [])[:4],
            "suppressed_actions": list(normalized.get("suppressed_actions", []) or [])[:6],
            "open_failure_modes": list(normalized.get("open_failure_modes", []) or [])[:4],
            "recent_failures": list(normalized.get("recent_failures", []) or [])[:4],
            "failure_streaks": list(normalized.get("failure_streaks", []) or [])[:6],
            "semantic_focus_actions": list(normalized.get("semantic_focus_actions", []) or [])[:3],
            "procedural_focus_actions": list(normalized.get("procedural_focus_actions", []) or [])[:3],
            "basin_exhausted": bool(normalized.get("basin_exhausted", False)),
            "basin_visits": int(normalized.get("basin_visits", 0) or 0),
            "stagnation_steps": int(normalized.get("stagnation_steps", 0) or 0),
            "llm_calls": int(normalized.get("llm_calls", 0) or 0),
            "pending_score_floor_reopen": self._compact_runtime_recovery_marker(
                normalized.get("pending_score_floor_reopen", {})
            ),
            "recent_score_floor_fallback": self._compact_runtime_recovery_marker(
                normalized.get("recent_score_floor_fallback", {})
            ),
            "recent_probe_dead_end_escape": self._compact_runtime_recovery_marker(
                normalized.get("recent_probe_dead_end_escape", {})
            ),
        }

    def _compact_runtime_recovery_marker(self, marker: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(marker or {})
        return {
            "state_name": str(normalized.get("state_name", "") or ""),
            "step_index": int(normalized.get("step_index", 0) or 0),
            "observation_tick": int(normalized.get("observation_tick", 0) or 0),
            "candidate_actions": list(normalized.get("candidate_actions", []) or [])[:4],
            "frame_signatures": list(normalized.get("frame_signatures", []) or [])[:4],
        }

    def _compact_memory_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for row in list(rows or [])[:4]:
            normalized = dict(row or {})
            compact.append(
                {
                    "memory_type": str(
                        normalized.get("memory_type", normalized.get("recall_type", "")) or ""
                    ),
                    "action_name": str(normalized.get("action_name", "") or ""),
                    "state": str(normalized.get("state", normalized.get("prior_game_state", "")) or ""),
                    "subgoal": str(normalized.get("subgoal", normalized.get("decision_subgoal", "")) or ""),
                    "previous_action_name": str(normalized.get("previous_action_name", "") or ""),
                    "lesson": str(normalized.get("lesson", "") or ""),
                    "strategy_tag": str(normalized.get("strategy_tag", "") or ""),
                    "outcome": str(normalized.get("outcome", "") or ""),
                    "score": round(float(normalized.get("score", 0.0) or 0.0), 4),
                    "salience": round(float(normalized.get("salience", 0.0) or 0.0), 4),
                    "confidence": round(float(normalized.get("confidence", 0.0) or 0.0), 4),
                    "retrieval_provenance": str(normalized.get("retrieval_provenance", "") or ""),
                }
            )
        return compact

    def _compact_retrieval_summary(self, summary: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(summary or {})
        return {
            "retrieval_version": str(normalized.get("retrieval_version", "") or ""),
            "state": str(normalized.get("state", "") or ""),
            "subgoal": str(normalized.get("subgoal", "") or ""),
            "active_subgoal": str(normalized.get("active_subgoal", "") or ""),
            "recent_record_count": int(normalized.get("recent_record_count", 0) or 0),
            "episode_summary_count": int(normalized.get("episode_summary_count", 0) or 0),
            "semantic_memory_count": int(normalized.get("semantic_memory_count", 0) or 0),
            "procedural_memory_count": int(normalized.get("procedural_memory_count", 0) or 0),
            "top_recall_sources": list(normalized.get("top_recall_sources", []) or [])[:4],
            "suggested_actions": list(normalized.get("suggested_actions", []) or [])[:4],
            "trained_prior_focus": list(normalized.get("trained_prior_focus", []) or [])[:4],
            "suppressed_actions": list(normalized.get("suppressed_actions", []) or [])[:6],
            "runtime_has_working_memory": bool(
                normalized.get("runtime_has_working_memory", False)
            ),
        }

    def _compact_basin_status(self, basin_status: dict[str, Any]) -> dict[str, Any]:
        status = dict(basin_status or {})
        return {
            "is_exhausted": bool(status.get("is_exhausted", False)),
            "known_legal_actions": list(status.get("known_legal_actions", [])[:6]),
            "exhausted_actions": list(status.get("exhausted_actions", [])[:6]),
        }

    def _compact_recent_records(self, recent_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for record in list(recent_records or [])[:4]:
            normalized = dict(record or {})
            compact.append(
                {
                    "action_name": str(normalized.get("action_name", "") or ""),
                    "outcome": str(normalized.get("outcome", "") or ""),
                    "prior_game_state": str(normalized.get("prior_game_state", "") or ""),
                    "decision_subgoal": str(normalized.get("decision_subgoal", "") or ""),
                    "stagnation_after": int(normalized.get("stagnation_after", 0) or 0),
                    "board_context": self._compact_board_context(
                        normalized.get("board_context_summary", {})
                    ),
                }
            )
        return compact

    def _compact_board_context(self, board_context: dict[str, Any]) -> dict[str, Any]:
        context = dict(board_context or {})
        return {
            "board_height": int(context.get("board_height", 0) or 0),
            "board_width": int(context.get("board_width", 0) or 0),
            "board_active_cells": int(context.get("board_active_cells", 0) or 0),
            "board_source_attr": str(context.get("board_source_attr", "") or ""),
            "board_default_probe": dict(context.get("board_default_probe") or {}),
        }

    def _compact_action_stats(
        self,
        stats: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        ranked = sorted(
            (
                (
                    str(action_name),
                    dict(values or {}),
                )
                for action_name, values in dict(stats or {}).items()
            ),
            key=lambda item: (
                -int(item[1].get("progress", 0) or 0),
                int(item[1].get("failed", 0) or 0),
                int(item[1].get("stalled", 0) or 0),
                item[0],
            ),
        )
        compact: dict[str, dict[str, Any]] = {}
        for action_name, values in ranked[:4]:
            compact[action_name] = {
                "progress": int(values.get("progress", 0) or 0),
                "failed": int(values.get("failed", 0) or 0),
                "stalled": int(values.get("stalled", 0) or 0),
                "total": int(values.get("total", 0) or 0),
            }
            last_outcome = str(values.get("last_outcome", "") or "")
            if last_outcome:
                compact[action_name]["last_outcome"] = last_outcome
        return compact

    def _should_replace_emergency_candidates_with_memory_escape(
        self,
        context: ExecutiveContext,
        bundle: MemoryBundle | None,
        proposals: list[ActionProposal],
    ) -> bool:
        if not getattr(context, "force_llm_replan", False):
            return False
        if bundle is None:
            return False
        basin_exhausted = bool(bundle.basin_status.get("is_exhausted", False)) or (
            context.memory.is_basin_exhausted(context.observation)
        )
        if not basin_exhausted:
            return False
        frame_signature = str(
            context.observation.derived_features.get("frame_signature", "") or ""
        ) or None
        stale_candidates = 0
        for proposal in proposals:
            counts = context.memory.context_outcome_counts(
                proposal.action_name,
                state_name=context.observation.game_state,
                frame_signature=frame_signature,
            )
            if counts["total"] <= 0 or counts["progress"] > 0:
                continue
            stale_candidates += 1
        return stale_candidates == len(proposals)

    def _memory_fallback_proposals(
        self,
        context: ExecutiveContext,
        bundle: MemoryBundle | None,
        failure_reason: str,
        *,
        preferred_actions: list[str] | None = None,
    ) -> list[ActionProposal]:
        if bundle is None:
            return []
        preferred_order: list[str] = []
        if preferred_actions:
            for action_name in preferred_actions:
                normalized = str(action_name or "")
                if (
                    normalized
                    and normalized != "RESET"
                    and normalized in context.observation.legal_actions
                    and normalized not in preferred_order
                ):
                    preferred_order.append(normalized)
        legal_actions = preferred_order or [
            action for action in context.observation.legal_actions if action != "RESET"
        ]
        if not legal_actions:
            return []
        basin_exhausted = bool(bundle.basin_status.get("is_exhausted", False)) or (
            context.memory.is_basin_exhausted(context.observation)
        )
        working_memory = dict(bundle.working_memory or {})
        focus_actions = self._working_memory_focus_actions(
            working_memory,
            legal_actions,
        )
        failure_actions = self._working_memory_failure_actions(working_memory)
        has_memory_guidance = bool(
            bundle.suppressed_actions
            or bundle.recommended_actions
            or bundle.trained_prior_bucket
            or focus_actions
            or failure_actions
        )
        if not basin_exhausted and not has_memory_guidance:
            return []
        base_suppressed = set(bundle.suppressed_actions)
        suppressed = set(base_suppressed)
        basin_break_excluded = self._basin_break_exclusions(
            context,
            legal_actions,
            basin_exhausted=basin_exhausted,
        )
        if basin_break_excluded and len(set(legal_actions) - basin_break_excluded) > 0:
            suppressed.update(basin_break_excluded)
        temporary_excluded = self._temporary_escape_exclusions(
            context,
            legal_actions,
            basin_exhausted=basin_exhausted,
        )
        if temporary_excluded and len(set(legal_actions) - temporary_excluded) > 0:
            suppressed.update(temporary_excluded)
        preferred_relaxed_actions: set[str] = set()
        if preferred_order:
            preferred_relaxed_actions = {
                action_name
                for action_name in preferred_order
                if action_name in base_suppressed
                and action_name not in basin_break_excluded
                and action_name not in temporary_excluded
            }
            if preferred_relaxed_actions and all(
                action_name in suppressed for action_name in preferred_order
            ):
                suppressed.difference_update(preferred_relaxed_actions)
        active_legal_actions = [
            action_name for action_name in legal_actions if action_name not in suppressed
        ]
        if basin_exhausted and context.memory.is_replan_frontier_exhausted(
            context.observation,
            action_names=active_legal_actions,
        ):
            return []
        recent_action = context.memory.most_recent_action()
        llm_subgoal_hint = str(context.llm_subgoal_hint or "")
        frontier_actions = self._score_floor_escape_frontier_actions(
            context,
            active_legal_actions or legal_actions,
            suppressed=suppressed,
            basin_exhausted=basin_exhausted,
        )
        candidate_actions = frontier_actions or (active_legal_actions or legal_actions)
        board_probe_action = preferred_board_probe_action(
            candidate_actions,
            context.observation.image_input,
            suppressed_actions=suppressed,
        )
        if basin_exhausted and self._board_probe_action_is_stale(context, board_probe_action):
            board_probe_action = None
        focus_actions = self._working_memory_focus_actions(working_memory, candidate_actions)
        ranked_actions: list[str] = []
        if preferred_order and basin_exhausted:
            ranked_actions.extend(
                action_name
                for action_name in preferred_order
                if action_name in candidate_actions
                and (
                    action_name not in suppressed
                    or action_name in preferred_relaxed_actions
                )
                and action_name != recent_action
                and action_name not in ranked_actions
            )
        ranked_actions.extend(
            action
            for action in focus_actions
            if action in candidate_actions
            and action not in suppressed
            and action != recent_action
            and action not in failure_actions
            and action not in ranked_actions
        )
        ranked_actions.extend(
            action
            for action in bundle.recommended_actions
            if action in candidate_actions
            and action not in suppressed
            and action not in ranked_actions
        )
        ranked_actions.extend(
            action
            for action in self._trained_prior_order(
                bundle,
                candidate_actions,
            )
            if action not in suppressed
            and action != recent_action
            and action not in failure_actions
            and action not in ranked_actions
        )
        if (
            board_probe_action is not None
            and board_probe_action != recent_action
            and board_probe_action not in ranked_actions
        ):
            ranked_actions.append(board_probe_action)
        ranked_actions.extend(
            action
            for action in candidate_actions
            if action not in suppressed
            and action not in failure_actions
            and action != recent_action
            and action not in ranked_actions
        )
        ranked_actions.extend(
            action
            for action in candidate_actions
            if action not in suppressed
            and action != recent_action
            and action not in ranked_actions
            and action in failure_actions
        )
        ranked_actions.extend(
            action
            for action in candidate_actions
            if action not in suppressed and action not in ranked_actions
        )
        if not ranked_actions and not basin_exhausted:
            return []
        if basin_exhausted and not ranked_actions:
            broader_board_probe_action = preferred_board_probe_action(
                legal_actions,
                context.observation.image_input,
            )
            if self._board_probe_action_is_stale(context, broader_board_probe_action):
                broader_board_probe_action = None
            ranked_actions.extend(
                self._rank_broader_escape_actions(
                    bundle=bundle,
                    legal_actions=legal_actions,
                    recent_action=recent_action,
                    board_probe_action=broader_board_probe_action,
                )
            )
            ranked_actions.extend(
                action for action in legal_actions if action not in ranked_actions
            )
        proposals: list[ActionProposal] = []
        for index, action_name in enumerate(ranked_actions[:3]):
            confidence = 0.72 if basin_exhausted else 0.58
            confidence = max(0.30, confidence - (index * 0.05))
            metadata = {}
            if hasattr(context.adapter, "default_metadata_for_action"):
                metadata = dict(context.adapter.default_metadata_for_action(action_name) or {})
            if action_name == "ACTION6" and context.observation.image_input is not None:
                metadata = self._board_probe_metadata(context)
            subgoal = "replan" if basin_exhausted else None
            if not subgoal and action_name == board_probe_action and llm_subgoal_hint:
                subgoal = llm_subgoal_hint
            proposals.append(
                ActionProposal(
                    action_name=action_name,
                    source=self.source,
                    confidence=confidence,
                    reason=f"memory-guided llm fallback ({failure_reason})",
                    subgoal=subgoal,
                    metadata=metadata,
                    expected_outcome="progress",
                )
            )
        return proposals

    def _working_memory_focus_actions(
        self,
        working_memory: dict[str, Any],
        candidate_actions: list[str],
    ) -> list[str]:
        ranked: list[str] = []
        for action_name in list(working_memory.get("procedural_focus_actions", []) or []):
            normalized = str(action_name or "")
            if (
                normalized
                and normalized in candidate_actions
                and normalized not in ranked
            ):
                ranked.append(normalized)
        for action_name in list(working_memory.get("semantic_focus_actions", []) or []):
            normalized = str(action_name or "")
            if (
                normalized
                and normalized in candidate_actions
                and normalized not in ranked
            ):
                ranked.append(normalized)
        return ranked

    def _working_memory_failure_actions(
        self,
        working_memory: dict[str, Any],
    ) -> set[str]:
        failure_actions: set[str] = set()
        for row in list(working_memory.get("recent_failures", []) or []):
            action_name = str(row.get("action_name", "") or "")
            if action_name:
                failure_actions.add(action_name)
        for row in list(working_memory.get("open_failure_modes", []) or []):
            action_name = str(row.get("action_name", "") or "")
            if action_name:
                failure_actions.add(action_name)
        for row in list(working_memory.get("failure_streaks", []) or []):
            action_name = str(row.get("action_name", "") or "")
            if not action_name:
                continue
            if int(row.get("count", 0) or 0) > 0:
                failure_actions.add(action_name)
        return failure_actions

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

    def _temporary_escape_exclusions(
        self,
        context: ExecutiveContext,
        legal_actions: list[str],
        *,
        basin_exhausted: bool,
    ) -> set[str]:
        if not basin_exhausted:
            return set()
        state_name = str(context.observation.game_state or "")
        frame_signature = str(
            context.observation.derived_features.get("frame_signature", "") or ""
        ) or None
        if not state_name:
            return set()
        excluded: set[str] = set()
        recent_replan_records = [
            record
            for record in reversed(context.memory.records[-6:])
            if record.action_name in legal_actions
            and record.prior_game_state == state_name
            and record.decision_source == "fallback"
            and record.decision_subgoal == "replan"
            and record.outcome in {"stalled", "failed"}
        ]
        if len(recent_replan_records) >= 4:
            actions = [record.action_name for record in recent_replan_records[:4]]
            if (
                actions[0] == actions[2]
                and actions[1] == actions[3]
                and actions[0] != actions[1]
            ):
                excluded.update({actions[0], actions[1]})
        frame_cycle = context.memory.recent_replan_frame_cycle(context.observation)
        if frame_cycle is not None:
            excluded.update(
                action_name
                for action_name in list(frame_cycle.get("actions") or [])
                if action_name in legal_actions
            )
        for record in reversed(context.memory.records[-8:]):
            if record.action_name not in legal_actions:
                continue
            if record.prior_game_state != state_name:
                continue
            if frame_signature and record.prior_frame_signature != frame_signature:
                continue
            if record.decision_source != "fallback":
                continue
            if record.decision_subgoal != "replan":
                continue
            if record.outcome not in {"stalled", "failed"}:
                continue
            excluded.add(record.action_name)
        return excluded

    def _basin_break_exclusions(
        self,
        context: ExecutiveContext,
        legal_actions: list[str],
        *,
        basin_exhausted: bool,
    ) -> set[str]:
        if not basin_exhausted:
            return set()
        state_name = str(context.observation.game_state or "")
        frame_signature = str(
            context.observation.derived_features.get("frame_signature", "") or ""
        ) or None
        if not state_name or not frame_signature:
            return set()
        excluded: set[str] = set()
        unseen_actions = 0
        for action_name in legal_actions:
            counts = context.memory.context_outcome_counts(
                action_name,
                state_name=state_name,
                frame_signature=frame_signature,
            )
            if counts["total"] <= 0:
                unseen_actions += 1
                continue
            if counts["progress"] > 0:
                continue
            if counts["failed"] > 0 or counts["stalled"] > 0:
                excluded.add(action_name)
        if unseen_actions <= 0:
            return set()
        return excluded

    def _score_floor_escape_frontier_actions(
        self,
        context: ExecutiveContext,
        legal_actions: list[str],
        *,
        suppressed: set[str],
        basin_exhausted: bool,
    ) -> list[str]:
        if not basin_exhausted:
            return []
        state_name = str(context.observation.game_state or "")
        frame_signature = str(
            context.observation.derived_features.get("frame_signature", "") or ""
        ) or None
        if not state_name or not frame_signature:
            return []
        attempted = context.memory.replan_frontier_attempted_actions(context.observation)
        frontier = [
            action_name
            for action_name in legal_actions
            if action_name not in suppressed and action_name not in attempted
        ]
        return frontier

    def _failure_reason_with_backend(
        self,
        context: ExecutiveContext,
        base_reason: str,
    ) -> str:
        backend = context.llm_backend
        backend_error = str(getattr(backend, "last_error", "") or "").strip()
        if not backend_error:
            return base_reason
        return f"{base_reason}; backend={backend_error}"

    def _rank_broader_escape_actions(
        self,
        *,
        bundle: MemoryBundle,
        legal_actions: list[str],
        recent_action: str | None,
        board_probe_action: str | None,
    ) -> list[str]:
        ranked: list[tuple[tuple[float, int, str], str]] = []
        prior_peak = max((int(value or 0) for value in bundle.trained_prior_bucket.values()), default=0)
        for action_name in legal_actions:
            game = bundle.action_stats.get(action_name, {})
            basin = bundle.basin_action_stats.get(action_name, {})
            state = bundle.state_action_stats.get(action_name, {})
            frame = bundle.frame_action_stats.get(action_name, {})
            subgoal = bundle.subgoal_action_stats.get(action_name, {})
            prior_count = int(bundle.trained_prior_bucket.get(action_name, 0) or 0)
            score = 0.0
            score += 5.0 * int(basin.get("progress", 0) or 0)
            score += 4.0 * int(state.get("progress", 0) or 0)
            score += 3.0 * int(subgoal.get("progress", 0) or 0)
            score += 2.0 * int(game.get("progress", 0) or 0)
            score += 1.0 * int(frame.get("progress", 0) or 0)
            score -= 2.0 * int(basin.get("failed", 0) or 0)
            score -= 1.25 * int(basin.get("stalled", 0) or 0)
            score -= 1.5 * int(frame.get("failed", 0) or 0)
            score -= 1.0 * int(frame.get("stalled", 0) or 0)
            score -= 1.0 * int(state.get("failed", 0) or 0)
            score -= 0.5 * int(state.get("stalled", 0) or 0)
            if prior_peak > 0:
                score += 1.5 * (prior_count / prior_peak)
            recent_penalty = 1 if action_name == recent_action else 0
            ranked.append(((score, -recent_penalty, action_name), action_name))
        ranked.sort(key=lambda item: item[0], reverse=True)
        ordered = [action_name for _, action_name in ranked]
        if board_probe_action in ordered:
            ordered.remove(board_probe_action)
            ordered.insert(0, board_probe_action)
        return ordered

    def _trained_prior_order(
        self,
        bundle: MemoryBundle,
        legal_actions: list[str],
    ) -> list[str]:
        ranked: list[tuple[int, str]] = []
        legal = {
            str(action_name or "")
            for action_name in list(legal_actions or [])
            if str(action_name or "")
        }
        for action_name, count in dict(bundle.trained_prior_bucket or {}).items():
            normalized = str(action_name or "")
            value = int(count or 0)
            if normalized not in legal or value <= 0:
                continue
            ranked.append((value, normalized))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [action_name for _, action_name in ranked]

    def _normalize_candidate(self, raw_candidate: Any) -> dict[str, Any]:
        if isinstance(raw_candidate, str):
            return {
                "action_name": raw_candidate.upper(),
                "confidence": 0.50,
                "reason": "llm string candidate",
                "metadata": {},
            }
        if not isinstance(raw_candidate, dict):
            return {
                "action_name": "",
                "confidence": 0.0,
                "reason": "invalid llm candidate",
                "metadata": {},
            }
        action_name = str(
            raw_candidate.get("action")
            or raw_candidate.get("name")
            or raw_candidate.get("action_name")
            or ""
        ).upper()
        metadata = {}
        if "x" in raw_candidate and "y" in raw_candidate:
            metadata = {"x": raw_candidate["x"], "y": raw_candidate["y"]}
        return {
            "action_name": action_name,
            "confidence": self._safe_float(raw_candidate.get("confidence", 0.50), 0.50),
            "reason": str(raw_candidate.get("reason", "llm structured candidate")),
            "metadata": metadata,
            "expected_outcome": raw_candidate.get("expected_outcome", "progress"),
        }

    def _safe_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
