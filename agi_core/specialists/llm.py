from __future__ import annotations

from typing import Any

from agi_core.types import ActionProposal, ExecutiveContext


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
        context.memory.note_llm_call()
        payload = backend.generate_json(
            prompt=self._build_prompt(context),
            max_tokens=context.config.llm_max_tokens,
            temperature=context.config.llm_temperature,
        )
        if not isinstance(payload, dict):
            return []
        legal_actions = set(context.observation.legal_actions)
        subgoal = payload.get("subgoal")
        base_reason = str(payload.get("reason", "llm proposal"))
        base_confidence = float(payload.get("confidence", 0.50))
        proposals: list[ActionProposal] = []
        candidates = payload.get("action_candidates", [])
        if not isinstance(candidates, list):
            return []
        for raw_candidate in candidates[:3]:
            candidate = self._normalize_candidate(raw_candidate)
            action_name = candidate["action_name"]
            if action_name not in legal_actions:
                continue
            proposals.append(
                ActionProposal(
                    action_name=action_name,
                    source=self.source,
                    confidence=float(candidate.get("confidence", base_confidence)),
                    reason=str(candidate.get("reason", base_reason)),
                    subgoal=subgoal,
                    metadata=dict(candidate.get("metadata", {})),
                    expected_outcome=str(candidate.get("expected_outcome", "progress")),
                )
            )
        return proposals

    def _build_prompt(self, context: ExecutiveContext) -> str:
        observation = context.observation
        memory = context.memory.summarize_for_hyperon()
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
            "Prefer one to three legal actions only.\n"
        )

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
            "confidence": float(raw_candidate.get("confidence", 0.50)),
            "reason": str(raw_candidate.get("reason", "llm structured candidate")),
            "metadata": metadata,
            "expected_outcome": raw_candidate.get("expected_outcome", "progress"),
        }
