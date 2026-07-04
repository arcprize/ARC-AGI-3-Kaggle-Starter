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
                "suppressed_actions": [],
                "basin_exhausted": basin_exhausted,
                "basin_status": {},
                "recent_records": [],
                "board_context_action_stats": {},
                "basin_action_stats": {},
                "frame_action_stats": {},
                "state_action_stats": {},
                "subgoal_action_stats": {},
            }
        return {
            "recommended_actions": list(bundle.recommended_actions[:3]),
            "suppressed_actions": list(bundle.suppressed_actions[:6]),
            "basin_exhausted": basin_exhausted,
            "basin_status": self._compact_basin_status(bundle.basin_status),
            "recent_records": self._compact_recent_records(bundle.recent_records),
            "board_context_action_stats": self._compact_action_stats(bundle.board_context_action_stats),
            "basin_action_stats": self._compact_action_stats(bundle.basin_action_stats),
            "frame_action_stats": self._compact_action_stats(bundle.frame_action_stats),
            "state_action_stats": self._compact_action_stats(bundle.state_action_stats),
            "subgoal_action_stats": self._compact_action_stats(bundle.subgoal_action_stats),
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
        has_memory_guidance = bool(bundle.suppressed_actions)
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
            for action in bundle.recommended_actions
            if action in candidate_actions
            and action not in suppressed
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
            and action != recent_action
            and action not in ranked_actions
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
