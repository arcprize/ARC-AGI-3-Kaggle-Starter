from __future__ import annotations

from agi_core.types import ActionProposal, ExecutiveContext


class ProposalCritic:
    def rescore(
        self,
        context: ExecutiveContext,
        proposals: list[ActionProposal],
    ) -> list[ActionProposal]:
        legal_actions = set(context.observation.legal_actions)
        frame_signature = str(
            context.observation.derived_features.get("frame_signature", "") or ""
        )
        bundle_cache: dict[str, object] = {}
        rescored: list[ActionProposal] = []
        for proposal in proposals:
            proposal_bundle = self._bundle_for_proposal(context, proposal, bundle_cache)
            basin_status = dict(proposal_bundle.basin_status or {})
            recommended_actions = list(proposal_bundle.recommended_actions or [])
            suppressed_actions = set(proposal_bundle.suppressed_actions or [])
            basin_action_stats = dict(proposal_bundle.basin_action_stats or {})
            state_action_stats = dict(proposal_bundle.state_action_stats or {})
            subgoal_action_stats = dict(proposal_bundle.subgoal_action_stats or {})
            basin_exhausted = bool(basin_status.get("is_exhausted")) or context.memory.is_basin_exhausted(
                context.observation
            )
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
            score -= context.memory.context_stall_penalty(
                proposal.action_name,
                state_name=context.observation.game_state,
                frame_signature=frame_signature,
            )
            score += context.memory.success_bonus(proposal.action_name)
            score += self._recommended_action_bonus(
                proposal.action_name,
                recommended_actions,
            )
            score -= self._suppressed_action_penalty(
                proposal.action_name,
                suppressed_actions,
            )
            score += self._bundle_context_bias(
                basin_action_stats.get(proposal.action_name, {}),
                success_scale=0.05,
                failed_scale=0.025,
                stalled_scale=0.015,
                positive_cap=0.20,
                negative_cap=0.12,
            )
            score += self._bundle_context_bias(
                state_action_stats.get(proposal.action_name, {}),
                success_scale=0.03,
                failed_scale=0.02,
                stalled_scale=0.012,
                positive_cap=0.14,
                negative_cap=0.10,
            )
            score += self._bundle_context_bias(
                subgoal_action_stats.get(proposal.action_name, {}),
                success_scale=0.035,
                failed_scale=0.02,
                stalled_scale=0.01,
                positive_cap=0.16,
                negative_cap=0.10,
            )
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
            if basin_exhausted and proposal.action_name != "RESET":
                if proposal.source == "symbolic":
                    score -= 0.45
                elif proposal.source == "llm":
                    score += 0.20
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

    def _bundle_for_proposal(
        self,
        context: ExecutiveContext,
        proposal: ActionProposal,
        cache: dict[str, object],
    ):
        subgoal = str(proposal.subgoal or "")
        if not subgoal:
            return context.memory_bundle or context.hyperon.current_memory_bundle()
        if subgoal in context.contextual_memory_bundles:
            return context.contextual_memory_bundles[subgoal]
        if subgoal in cache:
            return cache[subgoal]
        if hasattr(context.hyperon, "contextual_memory_bundle"):
            bundle = context.hyperon.contextual_memory_bundle(
                context.observation,
                subgoal,
                list(context.observation.legal_actions),
            )
            context.contextual_memory_bundles[subgoal] = bundle
            cache[subgoal] = bundle
            return bundle
        return context.memory_bundle or context.hyperon.current_memory_bundle()

    def _recommended_action_bonus(
        self,
        action_name: str,
        recommended_actions: list[str],
    ) -> float:
        if action_name not in recommended_actions:
            return 0.0
        index = recommended_actions.index(action_name)
        return max(0.03, 0.12 - (index * 0.03))

    def _suppressed_action_penalty(
        self,
        action_name: str,
        suppressed_actions: set[str],
    ) -> float:
        if action_name not in suppressed_actions:
            return 0.0
        return 0.18

    def _bundle_context_bias(
        self,
        stats: dict[str, object],
        *,
        success_scale: float,
        failed_scale: float,
        stalled_scale: float,
        positive_cap: float,
        negative_cap: float,
    ) -> float:
        if not stats:
            return 0.0
        progress = int(stats.get("progress", 0) or 0)
        failed = int(stats.get("failed", 0) or 0)
        stalled = int(stats.get("stalled", 0) or 0)
        positive = min(positive_cap, progress * success_scale)
        negative = min(negative_cap, (failed * failed_scale) + (stalled * stalled_scale))
        return positive - negative
