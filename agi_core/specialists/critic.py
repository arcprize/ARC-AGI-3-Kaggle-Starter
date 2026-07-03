from __future__ import annotations

from agi_core.types import ActionProposal, ExecutiveContext


class ProposalCritic:
    def rescore(
        self,
        context: ExecutiveContext,
        proposals: list[ActionProposal],
    ) -> list[ActionProposal]:
        legal_actions = set(context.observation.legal_actions)
        rescored: list[ActionProposal] = []
        for proposal in proposals:
            score = float(proposal.confidence)
            legal = proposal.action_name in legal_actions
            if proposal.action_name == "RESET" and context.observation.game_state in context.config.reset_state_names:
                legal = True
            proposal.legal = legal
            if not legal:
                proposal.score = -1.0
                rescored.append(proposal)
                continue
            score += context.config.source_bias.get(proposal.source, 0.0)
            score -= context.memory.repeated_failure_penalty(proposal.action_name)
            score -= context.memory.loop_penalty(proposal.action_name)
            score += context.memory.success_bonus(proposal.action_name)
            if (
                proposal.action_name == "RESET"
                and context.observation.game_state not in context.config.reset_state_names
            ):
                score -= 0.40
            if (
                context.memory.stagnation_steps >= 2
                and context.memory.most_recent_action() is not None
                and proposal.action_name != context.memory.most_recent_action()
            ):
                score += 0.05
            proposal.score = round(max(-1.0, min(score, 1.50)), 4)
            rescored.append(proposal)
        rescored.sort(
            key=lambda proposal: (
                proposal.score,
                proposal.confidence,
                context.config.source_bias.get(proposal.source, 0.0),
            ),
            reverse=True,
        )
        return rescored
