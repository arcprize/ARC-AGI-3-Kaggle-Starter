from __future__ import annotations

from typing import Any

from agi_core.types import ActionProposal, ExecutiveContext, MemoryBundle


class LLMSpecialist:
    source = "llm"

    def propose(self, context: ExecutiveContext) -> list[ActionProposal]:
        backend = context.llm_backend
        if backend is None or not getattr(backend, "available", False):
            return []
        if not context.memory.can_call_llm(context.config.llm_max_calls_per_episode):
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
            return self._memory_fallback_proposals(context, bundle, "non-dict payload")
        legal_actions = set(context.observation.legal_actions)
        subgoal = payload.get("subgoal")
        base_reason = str(payload.get("reason", "llm proposal"))
        base_confidence = self._safe_float(payload.get("confidence", 0.50), 0.50)
        proposals: list[ActionProposal] = []
        candidates = payload.get("action_candidates", [])
        if not isinstance(candidates, list):
            return self._memory_fallback_proposals(context, bundle, "invalid action_candidates")
        for raw_candidate in candidates[:3]:
            candidate = self._normalize_candidate(raw_candidate)
            action_name = candidate["action_name"]
            if action_name not in legal_actions:
                continue
            proposals.append(
                ActionProposal(
                    action_name=action_name,
                    source=self.source,
                    confidence=self._safe_float(
                        candidate.get("confidence", base_confidence),
                        base_confidence,
                    ),
                    reason=str(candidate.get("reason", base_reason)),
                    subgoal=subgoal,
                    metadata=dict(candidate.get("metadata", {})),
                    expected_outcome=str(candidate.get("expected_outcome", "progress")),
                )
            )
        if proposals:
            return proposals
        return self._memory_fallback_proposals(context, bundle, "no legal llm candidates")

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
        empty_dict: dict[str, object] = {}
        return (
            "You are a local action proposal model.\n"
            "Return JSON only with keys: subgoal, confidence, reason, action_candidates.\n"
            "Each action candidate must be an object with: action, confidence, reason, x, y.\n"
            f"game_id={observation.game_id}\n"
            f"step_index={observation.step_index}\n"
            f"game_state={observation.game_state}\n"
            f"levels_completed={observation.levels_completed}\n"
            f"legal_actions={observation.legal_actions}\n"
            f"recent_actions={memory['recent_actions']}\n"
            f"failure_streaks={memory['failure_streaks']}\n"
            f"memory_recommended_actions={bundle.recommended_actions if bundle is not None else []}\n"
            f"memory_suppressed_actions={bundle.suppressed_actions if bundle is not None else []}\n"
            f"memory_basin_exhausted={basin_exhausted}\n"
            f"memory_basin_status={basin_status}\n"
            f"memory_recent_records={bundle.recent_records if bundle is not None else []}\n"
            f"memory_basin_action_stats={bundle.basin_action_stats if bundle is not None else empty_dict}\n"
            f"memory_frame_action_stats={bundle.frame_action_stats if bundle is not None else empty_dict}\n"
            f"memory_state_action_stats={bundle.state_action_stats if bundle is not None else empty_dict}\n"
            f"memory_subgoal_action_stats={bundle.subgoal_action_stats if bundle is not None else empty_dict}\n"
            "Prefer one to three legal actions only.\n"
            "If memory_basin_exhausted=true, do not repeat suppressed or recently stalled actions.\n"
            "When exhausted, prefer memory_recommended_actions or any unsuppressed legal action.\n"
        )

    def _memory_fallback_proposals(
        self,
        context: ExecutiveContext,
        bundle: MemoryBundle | None,
        failure_reason: str,
    ) -> list[ActionProposal]:
        if bundle is None:
            return []
        legal_actions = [action for action in context.observation.legal_actions if action != "RESET"]
        if not legal_actions:
            return []
        basin_exhausted = bool(bundle.basin_status.get("is_exhausted", False)) or (
            context.memory.is_basin_exhausted(context.observation)
        )
        has_memory_guidance = bool(bundle.suppressed_actions)
        if not basin_exhausted and not has_memory_guidance:
            return []
        suppressed = set(bundle.suppressed_actions)
        recent_action = context.memory.most_recent_action()
        ranked_actions: list[str] = []
        ranked_actions.extend(
            action
            for action in bundle.recommended_actions
            if action in legal_actions
            and action not in suppressed
            and action not in ranked_actions
        )
        ranked_actions.extend(
            action
            for action in legal_actions
            if action not in suppressed
            and action != recent_action
            and action not in ranked_actions
        )
        ranked_actions.extend(
            action
            for action in legal_actions
            if action not in suppressed and action not in ranked_actions
        )
        if not ranked_actions and not basin_exhausted:
            return []
        if basin_exhausted and not ranked_actions:
            ranked_actions.extend(
                self._rank_broader_escape_actions(
                    bundle=bundle,
                    legal_actions=legal_actions,
                    recent_action=recent_action,
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
            proposals.append(
                ActionProposal(
                    action_name=action_name,
                    source=self.source,
                    confidence=confidence,
                    reason=f"memory-guided llm fallback ({failure_reason})",
                    subgoal="replan" if basin_exhausted else None,
                    metadata=metadata,
                    expected_outcome="progress",
                )
            )
        return proposals

    def _rank_broader_escape_actions(
        self,
        *,
        bundle: MemoryBundle,
        legal_actions: list[str],
        recent_action: str | None,
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
