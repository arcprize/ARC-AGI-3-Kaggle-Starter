from __future__ import annotations

from agi_core.types import ActionProposal, ExecutiveContext


class SymbolicSpecialist:
    source = "symbolic"

    def propose(self, context: ExecutiveContext) -> list[ActionProposal]:
        observation = context.observation
        proposals: list[ActionProposal] = []
        subgoal_rows = context.hyperon.run_rules("goal-hints")
        subgoal = None
        if subgoal_rows:
            subgoal = subgoal_rows[0].get("subgoal")
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
            proposals.append(
                ActionProposal(
                    action_name=action_name,
                    source=self.source,
                    confidence=float(row.get("confidence", 0.70)),
                    reason=str(row.get("reason", "symbolic prior")),
                    subgoal=subgoal or "advance-level",
                    metadata=metadata,
                    expected_outcome="progress",
                )
            )
        return proposals
