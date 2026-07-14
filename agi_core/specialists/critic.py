from __future__ import annotations

from agi_core.image_input import (
    active_cell_count,
    canonical_click_target_bonus,
    click_target_bias,
)
from agi_core.specialists.proposal_evidence import (
    build_evidence_view,
    bundle_context_bias,
    recommended_action_bonus,
    suppressed_action_penalty,
    trained_prior_bonus,
)
from agi_core.types import ActionProposal, ExecutiveContext


class ProposalCritic:
    def rescore(
        self,
        context: ExecutiveContext,
        proposals: list[ActionProposal],
    ) -> list[ActionProposal]:
        observation = context.observation
        memory = context.memory
        image_input = observation.image_input
        legal_actions = set(observation.legal_actions)
        game_state = observation.game_state
        source_bias = context.config.source_bias
        stagnation_steps = memory.stagnation_steps
        most_recent_action = memory.most_recent_action()
        active_cells = self._active_cell_count_cached(context)
        recent_probe_exit = self._recent_board_probe_exit(context, active_cells)
        frame_signature = ""
        if image_input is not None:
            frame_signature = str(image_input.frame_signature or "")
        if not frame_signature:
            frame_signature = str(
                observation.derived_features.get("frame_signature", "") or ""
            )
        bundle_cache: dict[str, object] = {}
        bundle_view_cache: dict[int, dict[str, object]] = {}
        breakdown_cache = context.multi_agent_frame_cache.setdefault(
            "critic_score_breakdowns",
            {},
        )
        rescored: list[ActionProposal] = []
        for proposal in proposals:
            proposal_bundle = self._bundle_for_proposal(context, proposal, bundle_cache)
            bundle_view = self._normalized_bundle_view(proposal_bundle, bundle_view_cache)
            basin_status = bundle_view["basin_status"]
            recommended_ranks = bundle_view["recommended_ranks"]
            trained_prior_bucket = bundle_view["trained_prior_bucket"]
            trained_prior_max = bundle_view["trained_prior_max"]
            suppressed_actions = bundle_view["suppressed_actions"]
            basin_action_stats = bundle_view["basin_action_stats"]
            state_action_stats = bundle_view["state_action_stats"]
            subgoal_action_stats = bundle_view["subgoal_action_stats"]
            semantic_memory_bonus = bundle_view["semantic_memory_bonus"]
            procedural_memory_bonus = bundle_view["procedural_memory_bonus"]
            episodic_recall_bonus = bundle_view["episodic_recall_bonus"]
            working_memory_focus_bonus = bundle_view["working_memory_focus_bonus"]
            working_memory_failure_penalty = bundle_view["working_memory_failure_penalty"]
            basin_exhausted = bool(basin_status.get("is_exhausted")) or memory.is_basin_exhausted(
                observation
            )
            score = float(proposal.confidence)
            legal = proposal.action_name in legal_actions
            if proposal.action_name == "RESET" and game_state in context.config.reset_state_names:
                legal = True
            proposal.legal = legal
            if not legal:
                proposal.score = -1.0
                rescored.append(proposal)
                continue
            breakdown: dict[str, float | str] = {
                "source_family": self._source_family(proposal.source),
                "base_confidence": round(float(proposal.confidence), 4),
            }
            source_bonus = source_bias.get(proposal.source, 0.0)
            score += source_bonus
            breakdown["source_bias"] = round(source_bonus, 4)
            repeated_failure_penalty = memory.repeated_failure_penalty(proposal.action_name)
            score -= repeated_failure_penalty
            breakdown["repeated_failure_penalty"] = round(-repeated_failure_penalty, 4)
            loop_penalty = memory.loop_penalty(proposal.action_name)
            score -= loop_penalty
            breakdown["loop_penalty"] = round(-loop_penalty, 4)
            context_penalty = memory.context_stall_penalty(
                proposal.action_name,
                state_name=game_state,
                frame_signature=frame_signature,
            )
            score -= context_penalty
            breakdown["context_stall_penalty"] = round(-context_penalty, 4)
            success_bonus = memory.success_bonus(proposal.action_name)
            score += success_bonus
            breakdown["success_bonus"] = round(success_bonus, 4)
            recommendation_bonus = recommended_action_bonus(
                proposal.action_name,
                recommended_ranks,
            )
            score += recommendation_bonus
            breakdown["recommended_bonus"] = round(recommendation_bonus, 4)
            prior_bonus = trained_prior_bonus(
                proposal.action_name,
                trained_prior_bucket,
                trained_prior_max,
            )
            score += prior_bonus
            breakdown["trained_prior_bonus"] = round(prior_bonus, 4)
            suppression_penalty = suppressed_action_penalty(
                proposal.action_name,
                suppressed_actions,
            )
            score -= suppression_penalty
            breakdown["suppressed_penalty"] = round(-suppression_penalty, 4)
            basin_bias = bundle_context_bias(
                basin_action_stats.get(proposal.action_name, {}),
                success_scale=0.05,
                failed_scale=0.025,
                stalled_scale=0.015,
                positive_cap=0.20,
                negative_cap=0.12,
            )
            score += basin_bias
            breakdown["basin_bias"] = round(basin_bias, 4)
            state_bias = bundle_context_bias(
                state_action_stats.get(proposal.action_name, {}),
                success_scale=0.03,
                failed_scale=0.02,
                stalled_scale=0.012,
                positive_cap=0.14,
                negative_cap=0.10,
            )
            score += state_bias
            breakdown["state_bias"] = round(state_bias, 4)
            subgoal_bias = bundle_context_bias(
                subgoal_action_stats.get(proposal.action_name, {}),
                success_scale=0.035,
                failed_scale=0.02,
                stalled_scale=0.01,
                positive_cap=0.16,
                negative_cap=0.10,
            )
            score += subgoal_bias
            breakdown["subgoal_bias"] = round(subgoal_bias, 4)
            semantic_bonus = float(semantic_memory_bonus.get(proposal.action_name, 0.0))
            procedural_bonus = float(procedural_memory_bonus.get(proposal.action_name, 0.0))
            episodic_bonus = float(episodic_recall_bonus.get(proposal.action_name, 0.0))
            wm_focus_bonus = float(working_memory_focus_bonus.get(proposal.action_name, 0.0))
            wm_failure_penalty = float(working_memory_failure_penalty.get(proposal.action_name, 0.0))
            score += semantic_bonus + procedural_bonus + episodic_bonus + wm_focus_bonus
            score -= wm_failure_penalty
            breakdown["semantic_bonus"] = round(semantic_bonus, 4)
            breakdown["procedural_bonus"] = round(procedural_bonus, 4)
            breakdown["episodic_bonus"] = round(episodic_bonus, 4)
            breakdown["working_memory_focus_bonus"] = round(wm_focus_bonus, 4)
            breakdown["working_memory_failure_penalty"] = round(-wm_failure_penalty, 4)
            if (
                proposal.action_name == "RESET"
                and game_state not in context.config.reset_state_names
            ):
                score -= 0.40
                breakdown["reset_penalty"] = -0.4
            if (
                stagnation_steps >= 2
                and most_recent_action is not None
                and proposal.action_name != most_recent_action
            ):
                score += 0.05
                breakdown["stagnation_rotation_bonus"] = 0.05
            if basin_exhausted and proposal.action_name != "RESET":
                if proposal.source == "symbolic":
                    score -= 0.45
                    breakdown["basin_exhausted_adjustment"] = -0.45
                elif proposal.source == "llm":
                    score += 0.20
                    breakdown["basin_exhausted_adjustment"] = 0.20
            if (
                proposal.action_name != "RESET"
                and stagnation_steps >= 4
                and (context.replan_mode or context.force_llm_replan)
            ):
                if proposal.source == "symbolic":
                    score -= 0.18
                    breakdown["replan_adjustment"] = -0.18
                elif proposal.source == "llm":
                    score += 0.12
                    breakdown["replan_adjustment"] = 0.12
            if proposal.action_name != "RESET" and context.probe_dead_end_replan:
                if proposal.source == "symbolic":
                    score -= 0.30
                    breakdown["probe_dead_end_adjustment"] = -0.30
                elif proposal.source == "llm":
                    score += 0.22
                    breakdown["probe_dead_end_adjustment"] = 0.22
            if proposal.action_name == "ACTION6":
                click_bias = click_target_bias(image_input, proposal.metadata)
                score += click_bias
                breakdown["click_bias"] = round(click_bias, 4)
                canonical_bonus = canonical_click_target_bonus(image_input, proposal.metadata)
                score += canonical_bonus
                breakdown["canonical_click_bonus"] = round(canonical_bonus, 4)
                if active_cells > 0 and click_bias > 0:
                    score += 0.08
                    breakdown["active_board_probe_bonus"] = 0.08
                    if str(proposal.subgoal or "") == "inspect-active-board":
                        score += 0.04
                        breakdown["inspection_probe_bonus"] = 0.04
                if (
                    str(proposal.subgoal or "") == "inspect-active-board"
                    and active_cells > 0
                    and most_recent_action is None
                    and int(observation.step_index) <= 0
                ):
                    score += 0.05
                    breakdown["opening_probe_bonus"] = 0.05
                if (
                    active_cells > 0
                    and stagnation_steps <= 1
                    and most_recent_action == "ACTION6"
                ):
                    score += 0.18
                    breakdown["probe_continuity_bonus"] = 0.18
                score -= recent_probe_exit
                if recent_probe_exit:
                    breakdown["probe_exit_penalty"] = round(-recent_probe_exit, 4)
            shape_bias = self._shape_target_bias(context, proposal)
            score += shape_bias
            if shape_bias:
                breakdown["shape_target_bias"] = round(shape_bias, 4)
            # Perception confidence weighting
            if game_state not in context.config.reset_state_names:
                pc = float(getattr(observation, "perception_confidence", 1.0))
                if pc < 0.4:
                    score -= 0.12
                    breakdown["perception_adjustment"] = -0.12
                elif pc > 0.9:
                    score += 0.06
                    breakdown["perception_adjustment"] = 0.06
                # Dynamics-specific: solver proposals weighed by dynamics confidence
                if proposal.source in ("solver:mirror", "solver:beam", "solver:navigation"):
                    dyn_conf = observation.derived_features.get("dynamics_confidence", {})
                    source_dyn = proposal.source.split(":")[-1]
                    dc = float(dyn_conf.get(source_dyn, 0.5))
                    dyn_adjustment = (dc - 0.5) * 0.20
                    score += dyn_adjustment
                    breakdown["dynamics_adjustment"] = round(dyn_adjustment, 4)
            proposal.score = round(max(-1.0, min(score, 1.50)), 4)
            breakdown_cache[id(proposal)] = breakdown
            rescored.append(proposal)
        rescored.sort(
            key=lambda proposal: (
                proposal.score,
                self._proposal_tiebreak_priority(context, proposal),
                proposal.confidence,
                source_bias.get(proposal.source, 0.0),
            ),
            reverse=True,
        )
        return rescored

    def _normalized_bundle_view(
        self,
        proposal_bundle,
        cache: dict[int, dict[str, object]],
    ) -> dict[str, object]:
        key = id(proposal_bundle)
        cached = cache.get(key)
        if cached is not None:
            return cached
        evidence = build_evidence_view(proposal_bundle)
        working_memory = dict(proposal_bundle.working_memory or {})
        working_memory["_semantic_focus_ranks"] = evidence.semantic_focus_ranks
        working_memory["_procedural_focus_ranks"] = evidence.procedural_focus_ranks
        working_memory["_recent_failure_actions"] = evidence.recent_failure_actions
        working_memory["_failure_streak_counts"] = evidence.failure_streak_counts
        cached = {
            "basin_status": evidence.basin_status,
            "working_memory": working_memory,
            "recommended_ranks": evidence.recommended_ranks,
            "trained_prior_bucket": evidence.trained_prior_bucket,
            "trained_prior_max": evidence.trained_prior_max,
            "suppressed_actions": evidence.suppressed_actions,
            "basin_action_stats": evidence.basin_action_stats,
            "state_action_stats": evidence.state_action_stats,
            "subgoal_action_stats": evidence.subgoal_action_stats,
            "semantic_memory_bonus": evidence.semantic_memory_bonus,
            "procedural_memory_bonus": evidence.procedural_memory_bonus,
            "episodic_recall_bonus": evidence.episodic_recall_bonus,
            "semantic_focus_ranks": evidence.semantic_focus_ranks,
            "procedural_focus_ranks": evidence.procedural_focus_ranks,
            "recent_failure_actions": evidence.recent_failure_actions,
            "open_failure_actions": set(),
            "failure_streak_counts": evidence.failure_streak_counts,
            "working_memory_focus_bonus": evidence.working_memory_focus_bonus,
            "working_memory_failure_penalty": evidence.working_memory_failure_penalty,
        }
        cache[key] = cached
        return cached

    def _source_family(self, source: str) -> str:
        if ":" in str(source or ""):
            prefix, _, suffix = str(source).partition(":")
            return f"{prefix}:{suffix}"
        return str(source or "")

    def _active_cell_count_cached(self, context: ExecutiveContext) -> int:
        cache = context.multi_agent_frame_cache
        active_cells = cache.get("critic_active_cell_count")
        if active_cells is None:
            active_cells = active_cell_count(context.observation.image_input)
            cache["critic_active_cell_count"] = active_cells
        return int(active_cells)

    def _recent_board_probe_exit(self, context: ExecutiveContext, active_cells: int) -> float:
        cache = context.multi_agent_frame_cache
        cached = cache.get("critic_recent_board_probe_exit_penalty")
        if cached is not None:
            return float(cached)
        if active_cells <= 0 or context.memory.stagnation_steps < 2:
            penalty = 0.0
        else:
            recent_actions = [
                action_name
                for action_name in context.memory.recent_actions()[-2:]
                if action_name
            ]
            penalty = 0.28 if recent_actions[-2:] == ["ACTION6", "ACTION6"] else 0.0
        cache["critic_recent_board_probe_exit_penalty"] = penalty
        return penalty

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
        recommended_ranks: dict[str, int],
    ) -> float:
        index = recommended_ranks.get(action_name)
        if index is None:
            return 0.0
        return max(0.03, 0.12 - (index * 0.03))

    def _suppressed_action_penalty(
        self,
        action_name: str,
        suppressed_actions: set[str],
    ) -> float:
        if action_name not in suppressed_actions:
            return 0.0
        return 0.18

    def _trained_prior_bonus(
        self,
        action_name: str,
        trained_prior_bucket: dict[str, int],
        max_count: int,
    ) -> float:
        if action_name not in trained_prior_bucket or not trained_prior_bucket:
            return 0.0
        if max_count <= 0:
            return 0.0
        normalized = float(trained_prior_bucket.get(action_name, 0)) / float(max_count)
        return min(0.10, 0.10 * normalized)

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

    def _memory_row_bonus_map(
        self,
        rows: list[dict[str, object]],
        *,
        scale: float,
    ) -> dict[str, float]:
        best_by_action: dict[str, float] = {}
        for row in rows:
            action_name = str(row.get("action_name", "") or "")
            if not action_name:
                continue
            confidence = float(row.get("confidence", 0.0) or 0.0)
            score = float(row.get("score", 0.0) or 0.0)
            best_by_action[action_name] = max(
                best_by_action.get(action_name, 0.0),
                min(scale, (confidence * 0.08) + (score * 0.01)),
            )
        return best_by_action

    def _episodic_recall_bonus_map(
        self,
        recalls: list[dict[str, object]],
    ) -> dict[str, float]:
        bonus_by_action: dict[str, float] = {}
        for row in recalls:
            action_name = str(row.get("action_name", "") or "")
            if not action_name:
                continue
            outcome = str(row.get("outcome", "") or "")
            current = bonus_by_action.get(action_name, 0.0)
            if outcome == "progress":
                bonus_by_action[action_name] = max(current, 0.08)
            elif outcome in {"failed", "stalled"}:
                bonus_by_action[action_name] = min(current, -0.04)
        return bonus_by_action

    def _working_memory_focus_bonus_map(
        self,
        semantic_focus_ranks: dict[str, int],
        procedural_focus_ranks: dict[str, int],
    ) -> dict[str, float]:
        bonus_by_action: dict[str, float] = {}
        for action_name, semantic_index in semantic_focus_ranks.items():
            bonus_by_action[action_name] = bonus_by_action.get(action_name, 0.0) + max(
                0.01,
                0.04 - (semantic_index * 0.01),
            )
        for action_name, procedural_index in procedural_focus_ranks.items():
            bonus_by_action[action_name] = bonus_by_action.get(action_name, 0.0) + max(
                0.01,
                0.05 - (procedural_index * 0.015),
            )
        return {
            action_name: min(0.08, bonus)
            for action_name, bonus in bonus_by_action.items()
        }

    def _working_memory_failure_penalty_map(
        self,
        recent_failure_actions: set[str],
        open_failure_actions: set[str],
        failure_streak_counts: dict[str, int],
    ) -> dict[str, float]:
        penalty_by_action: dict[str, float] = {}
        for action_name in recent_failure_actions:
            if action_name:
                penalty_by_action[action_name] = penalty_by_action.get(action_name, 0.0) + 0.06
        for action_name in open_failure_actions:
            if action_name:
                penalty_by_action[action_name] = penalty_by_action.get(action_name, 0.0) + 0.05
        for action_name, count in failure_streak_counts.items():
            if action_name:
                penalty_by_action[action_name] = penalty_by_action.get(action_name, 0.0) + min(
                    0.08,
                    int(count or 0) * 0.02,
                )
        return {
            action_name: min(0.16, penalty)
            for action_name, penalty in penalty_by_action.items()
        }

    def _board_probe_bonus(
        self,
        context: ExecutiveContext,
        metadata: dict[str, object] | None,
    ) -> float:
        if active_cell_count(context.observation.image_input) <= 0:
            return 0.0
        if click_target_bias(context.observation.image_input, metadata) <= 0:
            return 0.0
        return 0.08

    def _board_probe_subgoal_bonus(
        self,
        subgoal: str | None,
        context: ExecutiveContext,
        metadata: dict[str, object] | None,
    ) -> float:
        if str(subgoal or "") != "inspect-active-board":
            return 0.0
        if active_cell_count(context.observation.image_input) <= 0:
            return 0.0
        if click_target_bias(context.observation.image_input, metadata) <= 0:
            return 0.0
        return 0.04

    def _opening_board_probe_bonus(
        self,
        subgoal: str | None,
        context: ExecutiveContext,
    ) -> float:
        if str(subgoal or "") != "inspect-active-board":
            return 0.0
        if active_cell_count(context.observation.image_input) <= 0:
            return 0.0
        if context.memory.most_recent_action() is not None:
            return 0.0
        if int(context.observation.step_index) > 0:
            return 0.0
        return 0.05

    def _proposal_tiebreak_priority(
        self,
        context: ExecutiveContext,
        proposal: ActionProposal,
    ) -> int:
        if proposal.metadata.get("target_anchor"):
            if proposal.action_name == "ACTION5" and proposal.metadata.get("ready_to_switch"):
                return 4
            return 3
        if (
            proposal.action_name == "ACTION6"
            and active_cell_count(context.observation.image_input) > 0
        ):
            if str(proposal.subgoal or "") == "inspect-active-board":
                return 2
            return 1
        return 0

    def _recent_board_probe_continuity_bonus(
        self,
        context: ExecutiveContext,
    ) -> float:
        if active_cell_count(context.observation.image_input) <= 0:
            return 0.0
        if context.memory.stagnation_steps > 1:
            return 0.0
        if context.memory.most_recent_action() != "ACTION6":
            return 0.0
        return 0.18

    def _repeated_board_probe_exit_penalty(
        self,
        context: ExecutiveContext,
    ) -> float:
        if active_cell_count(context.observation.image_input) <= 0:
            return 0.0
        if context.memory.stagnation_steps < 2:
            return 0.0
        recent_actions = [
            action_name
            for action_name in context.memory.recent_actions()[-2:]
            if action_name
        ]
        if len(recent_actions) < 2:
            return 0.0
        if recent_actions[-2:] != ["ACTION6", "ACTION6"]:
            return 0.0
        return 0.28

    def _shape_target_bias(
        self,
        context: ExecutiveContext,
        proposal: ActionProposal,
    ) -> float:
        cache = getattr(context, "multi_agent_frame_cache", {}) or {}
        if not cache.get("shape_agent_mode"):
            return 0.0
        metadata = getattr(proposal, "metadata", {}) or {}
        target = metadata.get("target_anchor")
        if not isinstance(target, list) or len(target) != 2:
            return 0.0
        goal_dir = cache.get("goal_direction")
        ready_to_switch = bool(cache.get("ready_to_switch"))
        if proposal.action_name == "ACTION5":
            return 0.24 if ready_to_switch else -0.10
        if proposal.action_name == "ACTION6":
            return -0.18
        if proposal.action_name == goal_dir:
            return 0.22
        if proposal.action_name in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}:
            return 0.04
        return 0.0
