from __future__ import annotations

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
        if context.memory.stagnation_steps >= 2:
            most_recent = context.memory.most_recent_action()
            for action_name in observation.legal_actions:
                if action_name in {"RESET", most_recent}:
                    continue
                proposals.append(
                    ActionProposal(
                        action_name=action_name,
                        source=self.source,
                        confidence=0.55,
                        reason="reactive stall breaker",
                        subgoal="break-stall",
                    )
                )
                break
        fallback = context.adapter.fallback_action_name(observation)
        proposals.append(
            ActionProposal(
                action_name=fallback,
                source=self.source,
                confidence=0.40,
                reason="deterministic reactive baseline",
                subgoal="maintain-control-loop",
                metadata=context.adapter.fallback_decision(observation).metadata,
            )
        )
        return proposals
