from __future__ import annotations

from agi_core.image_input import active_cell_count, board_probe_click_target, default_click_target
from agi_core.types import ActionProposal, ExecutiveContext


class SymbolicSpecialist:
    source = "symbolic"

    def propose(self, context: ExecutiveContext) -> list[ActionProposal]:
        observation = context.observation
        proposals: list[ActionProposal] = []
        board_summary = None
        if observation.image_input is not None:
            board_summary = (
                f"board {observation.image_input.width}x{observation.image_input.height}"
            )
        subgoal_rows = context.hyperon.run_rules("goal-hints")
        subgoal = None
        if subgoal_rows:
            subgoal = subgoal_rows[0].get("subgoal")
        suppressed_actions: set[str] = set()
        if context.memory_bundle is not None:
            suppressed_actions = set(context.memory_bundle.suppressed_actions or [])
        for row in context.hyperon.run_rules("reset-needed"):
            proposals.append(
                ActionProposal(
                    action_name=row["action_name"],
                    source=self.source,
                    confidence=float(row.get("confidence", 0.95)),
                    reason=str(row.get("reason", "symbolic reset")),
                    subgoal=subgoal or "re-enter-level",
                )
            )
        prior_query = "action-priors"
        if subgoal:
            prior_query = f"action-priors:{subgoal}"
        for row in context.hyperon.run_rules(prior_query):
            action_name = str(row.get("action_name", "")).upper()
            if action_name not in observation.legal_actions:
                continue
            metadata = {}
            if hasattr(context.adapter, "default_metadata_for_action"):
                metadata = dict(context.adapter.default_metadata_for_action(action_name) or {})
            if action_name == "ACTION6" and observation.image_input is not None:
                metadata = self._board_probe_metadata(context)
            proposals.append(
                ActionProposal(
                    action_name=action_name,
                    source=self.source,
                    confidence=float(row.get("confidence", 0.70)),
                    reason=(
                        f"{row.get('reason', 'symbolic prior')} [{board_summary}]"
                        if board_summary
                        else str(row.get("reason", "symbolic prior"))
                    ),
                    subgoal=subgoal or "advance-level",
                    metadata=metadata,
                    expected_outcome="progress",
                )
            )
        if (
            str(subgoal or "") == "inspect-active-board"
            and observation.image_input is not None
            and "ACTION6" in observation.legal_actions
            and "ACTION6" not in suppressed_actions
            and active_cell_count(observation.image_input) > 0
        ):
            board_context = board_summary or "board"
            probe_confidence = 0.78
            existing_index = next(
                (
                    index
                    for index, proposal in enumerate(proposals)
                    if proposal.action_name == "ACTION6"
                ),
                None,
            )
            if existing_index is not None:
                existing = proposals.pop(existing_index)
                existing.confidence = max(existing.confidence, probe_confidence)
                existing.reason = f"symbolic board inspection target [{board_context}]"
                existing.subgoal = "inspect-active-board"
                existing.metadata = self._board_probe_metadata(context)
                existing.expected_outcome = "probe-active-board"
                proposals.insert(0, existing)
            else:
                proposals.insert(
                    0,
                    ActionProposal(
                        action_name="ACTION6",
                        source=self.source,
                        confidence=probe_confidence,
                        reason=f"symbolic board inspection target [{board_context}]",
                        subgoal="inspect-active-board",
                        metadata=self._board_probe_metadata(context),
                        expected_outcome="probe-active-board",
                    ),
                )
        if proposals:
            return proposals
        if observation.image_input is None or "ACTION6" not in observation.legal_actions:
            return proposals
        if "ACTION6" in suppressed_actions:
            return proposals
        active_cells = active_cell_count(observation.image_input)
        if active_cells <= 0:
            return proposals
        confidence = min(0.55, 0.34 + (min(active_cells, 8) * 0.02))
        board_context = board_summary or "board"
        proposals.append(
            ActionProposal(
                action_name="ACTION6",
                source=self.source,
                confidence=confidence,
                reason=f"symbolic board-target fallback [{board_context}; active_cells={active_cells}]",
                subgoal=subgoal or "inspect-active-board",
                metadata=self._board_probe_metadata(context),
                expected_outcome="probe-active-board",
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
