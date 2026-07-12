from __future__ import annotations

from agi_core.config import RuntimeConfig
from agi_core.image_input import (
    active_cell_count,
    board_probe_click_target,
    canonical_click_target_bonus,
    click_target_bias,
    default_click_target,
    preferred_board_probe_action,
)
from agi_core.memory import EpisodicMemory
from agi_core.runtime_sync import ensure_hyperon_context_synced
from agi_core.types import (
    ActionDecision,
    EpisodeRecord,
    ExecutiveContext,
    GoalState,
    MemoryBundle,
    Observation,
    ObservedTransition,
)
from agi_core.orch_or.parallel import ThreadPoolManager
from concurrent.futures import wait
import logging

logger = logging.getLogger(__name__)


class Executive:
    def __init__(
        self,
        config: RuntimeConfig,
        reactive,
        symbolic,
        llm,
        critic,
        attention=None,
        working_memory=None,
        goal_stack=None,
        episodic_memory=None,
        meta_cognition=None,
        chunk_library=None,
    ) -> None:
        self.config = config
        self.reactive = reactive
        self.symbolic = symbolic
        self.llm = llm
        self.critic = critic
        self.attention = attention
        self.working_memory = working_memory
        self.goal_stack = goal_stack
        self.episodic_memory = episodic_memory
        self.meta_cognition = meta_cognition
        self.chunk_library = chunk_library

    def decide(self, context: ExecutiveContext) -> ActionDecision:
        # Update attention visitation BEFORE hyperon/orchestrator computes salience
        if self.attention is not None:
            player_pos = getattr(context.observation, 'player_position', None)
            if player_pos is not None:
                self.attention.mark_visited(player_pos)
        context.memory_bundle = ensure_hyperon_context_synced(context)
        context.llm_memory_bundle = context.memory_bundle
        context.llm_subgoal_hint = None
        context.contextual_memory_bundles = {}

        # ── Cognition integration ──────────────────────────────────
        wm = self.working_memory
        mc = self.meta_cognition
        context.goal_stack = self.goal_stack
        context.working_memory = self.working_memory
        context.chunk_library = self.chunk_library
        context.attention = self.attention
        context.episodic_memory = self.episodic_memory
        if wm is not None:
            wm.tick()
        if mc is not None:
            stagnation_steps = int(getattr(context.memory, '_stagnation_steps', 0))
            recent_raw = wm.recent_actions(5) if wm is not None else []
            outcomes = []
            consec = getattr(mc, '_consecutive_failures', 0)
            for a in recent_raw:
                if consec > 0:
                    outcomes.append("failed")
                else:
                    outcomes.append("progress" if a not in ("RESET", "ACTION6") else "stalled")
            wm_load = wm.load if wm is not None else 0.0
            plan = getattr(context.memory, '_active_plan', None) or getattr(context, '_active_plan', None)
            has_plan = plan is not None
            confidence = mc.estimate_confidence(
                plan_steps=list(getattr(plan, 'steps', []) if plan is not None else []),
                goal_stack_depth=len(getattr(self.goal_stack, 'active_path', []) if self.goal_stack else []),
                working_memory_load=wm_load,
                stagnation_steps=stagnation_steps,
                recent_outcomes=outcomes,
                consecutive_failures=getattr(mc, '_consecutive_failures', 0),
            )
            goal_progress = getattr(self.goal_stack, 'progress_ratio', 0.0) if self.goal_stack else 0.0
            strategy = mc.select_strategy(
                confidence=confidence,
                stagnation_steps=stagnation_steps,
                has_plan=has_plan,
                working_memory_load=wm_load,
                goal_progress=goal_progress,
            )
            context.cognition_strategy = strategy.value
        pending_reopen = context.memory.pending_score_floor_reopen(context.observation)
        if pending_reopen is not None and "RESET" in context.observation.legal_actions:
            context.score_floor_debug = dict(pending_reopen.get("score_floor_debug") or {})
            context.memory.clear_pending_score_floor_reopen()
            decision = ActionDecision(
                action_name="RESET",
                source="fallback",
                confidence=0.05,
                reason="score-floor pending frontier reopen reset",
                metadata={"score_floor_debug": dict(context.score_floor_debug)},
                used_fallback=True,
            )
            context.memory.note_decision(decision)
            self._finalize_cognition(context, decision)
            return decision
        context.probe_dead_end_replan = (
            self._should_force_probe_dead_end_llm_replan(context)
            if self.llm is not None else False
        )
        context.force_llm_replan = (
            self._should_force_emergency_llm_replan(context)
            if self.llm is not None else False
        )
        replan_mode = self._should_enter_replan_mode(context)
        context.replan_mode = replan_mode

        # ── Gate specialists by cognition strategy ────────────────
        mc = self.meta_cognition
        cur_strategy = None
        if mc is not None and hasattr(mc, '_mode_history') and mc._mode_history:
            cur_strategy = mc._mode_history[-1]
        use_reactive = not replan_mode and (mc is None or mc.should_use_specialist(cur_strategy))
        use_symbolic = mc is None or mc.should_use_specialist(cur_strategy)

        reactive_proposals = []
        symbolic_proposals = []
        specialist_tasks = {}
        if use_reactive:
            reactive_task = ThreadPoolManager.submit_unique(
                f"executive:{id(self)}:reactive",
                self.reactive.propose,
                context,
            )
        else:
            reactive_task = None
        if use_symbolic:
            symbolic_task = ThreadPoolManager.submit_unique(
                f"executive:{id(self)}:symbolic",
                self.symbolic.propose,
                context,
            )
        else:
            symbolic_task = None
        if reactive_task is not None:
            specialist_tasks["reactive"] = reactive_task
        if symbolic_task is not None:
            specialist_tasks["symbolic"] = symbolic_task
        done = set()
        if specialist_tasks:
            done, _ = wait(set(specialist_tasks.values()), timeout=2.0)
        if reactive_task is not None and reactive_task in done:
            try:
                reactive_proposals = reactive_task.result()
            except Exception:
                logger.debug("reactive specialist failed", exc_info=True)
                reactive_proposals = []
        if symbolic_task is not None and symbolic_task in done:
            try:
                symbolic_proposals = symbolic_task.result()
            except Exception:
                logger.debug("symbolic specialist failed", exc_info=True)
                symbolic_proposals = []
        proposals = list(reactive_proposals) + list(symbolic_proposals)
        score_floor_fallback_cooldown = context.memory.recent_score_floor_fallback(
            context.observation
        )
        best_symbolic_confidence = max(
            (proposal.confidence for proposal in symbolic_proposals),
            default=0.0,
        )
        skip_early_probe_llm = self._should_skip_early_probe_llm(
            context,
            symbolic_proposals,
        )
        if self.llm is not None:
            self._refresh_llm_memory_bundle(context, symbolic_proposals, reactive_proposals)
            if context.force_llm_replan and not context.llm_subgoal_hint:
                context.llm_subgoal_hint = "replan"
            if (
                (
                    replan_mode
                    or context.force_llm_replan
                    or best_symbolic_confidence < context.config.symbolic_confidence_gate
                )
                and not score_floor_fallback_cooldown
                and not skip_early_probe_llm
            ):
                proposals.extend(self.llm.propose(context))
        context.hyperon.update_proposals(proposals)
        rescored = self.critic.rescore(context, proposals)
        if self._all_legal_proposals_are_score_floor(context, rescored):
            fallback = self._score_floor_fallback(context, rescored)
            fallback.proposals = rescored
            context.memory.note_decision(fallback)
            self._finalize_cognition(context, fallback)
            return fallback
        probe_dead_end_escape = self._probe_dead_end_escape(context, rescored)
        if probe_dead_end_escape is not None:
            probe_dead_end_escape.proposals = rescored
            context.memory.note_decision(probe_dead_end_escape)
            self._finalize_cognition(context, probe_dead_end_escape)
            return probe_dead_end_escape
        probe_dead_end_followup_escape = self._probe_dead_end_followup_escape(
            context,
            rescored,
        )
        if probe_dead_end_followup_escape is not None:
            probe_dead_end_followup_escape.proposals = rescored
            context.memory.note_decision(probe_dead_end_followup_escape)
            self._finalize_cognition(context, probe_dead_end_followup_escape)
            return probe_dead_end_followup_escape
        for proposal in rescored:
            if proposal.legal:
                metadata = self._finalize_decision_metadata(
                    context,
                    proposal.action_name,
                    dict(proposal.metadata),
                )
                subgoal = self._finalize_decision_subgoal(
                    context,
                    proposal.action_name,
                    proposal.subgoal,
                )
                decision = ActionDecision(
                    action_name=proposal.action_name,
                    source=proposal.source,
                    confidence=proposal.confidence,
                    reason=proposal.reason,
                    subgoal=subgoal,
                    metadata=metadata,
                    proposals=rescored,
                )
                context.memory.note_decision(decision)
                self._finalize_cognition(context, decision)
                return decision
        fallback = self._memory_aware_fallback(context)
        fallback.proposals = rescored
        context.memory.note_decision(fallback)
        self._finalize_cognition(context, fallback)
        return fallback

    def _finalize_cognition(self, context, decision: ActionDecision) -> None:
        """Update cognition modules after a decision is made.
        
        Guarded by ``_last_cog_frame`` — only fires once per decision."""
        step = context.observation.step_index
        if getattr(self, '_last_cog_frame', -1) == step:
            return
        self._last_cog_frame = step
        wm = self.working_memory
        mc = self.meta_cognition
        gs = self.goal_stack
        em = self.episodic_memory
        cl = self.chunk_library
        action_name = decision.action_name
        if wm is not None:
            from agi_core.cognition.working_memory import WMItem
            wm.push(WMItem(
                key=f"action_{context.observation.step_index}",
                content=action_name,
                category="outcome",
            ))
            wm.record_action(action_name)
        if self.attention is not None:
            player_pos = getattr(context.observation, 'player_position', None)
            if player_pos is not None:
                self.attention.mark_visited(player_pos)
        if mc is not None:
            mc.note_visit(action_name)
        self._pending_cognition_decision = {
            "game_id": str(context.observation.game_id or ""),
            "action_name": action_name,
            "source": str(decision.source or ""),
            "subgoal": decision.subgoal,
        }
        if gs is not None and decision.subgoal:
            sub = decision.subgoal
            found = gs.find(sub)
            if found is None:
                gs.push(sub)
            gs.mark_active()
        # Forgetting is handled automatically within EpisodicMemory.record()
        # every 10 episodes — no need for a separate frame-level trigger.
        # Periodic episodic consolidation + mid-game recording (every 50 frames)
        self._episodic_consolidation_counter = getattr(self, '_episodic_consolidation_counter', 0) + 1
        if em is not None and self._episodic_consolidation_counter >= 50:
            # Record a mid-game episode snapshot for consolidation
            recent = wm.recent_actions(15) if wm is not None else []
            observed_outcome = str(
                getattr(self, "_last_observed_outcome", "unknown") or "unknown"
            )
            if len(recent) >= 5 and observed_outcome != "unknown":
                dyn = getattr(context.observation, 'derived_features', {}).get('primary_dynamics', 'unknown')
                gc = len(getattr(context.observation, 'goal_positions', []) or [])
                board_hash = hash(str(getattr(context.observation, 'image_input', None)))
                player_start = getattr(context.observation, 'player_position', (0, 0)) or (0, 0)
                episode_outcome = (
                    "progress" if observed_outcome == "advanced" else observed_outcome
                )
                em.record(dyn, recent, episode_outcome, gc, player_start, board_hash)
            em.consolidate()
            self._episodic_consolidation_counter = 0
        # Frame-level chunk extraction from recent action window
        if cl is not None and wm is not None:
            recent = wm.recent_actions(10)
            if len(recent) >= 5:
                dynamics = getattr(context.observation, 'derived_features', {}).get('primary_dynamics', 'unknown')
                gc = len(getattr(context.observation, 'goal_positions', []) or [])
                cl.extract_from_window(recent, dynamics, goal_count=gc)
        # Stale chunk decay every 100 frames
        self._decay_counter = getattr(self, '_decay_counter', 0) + 1
        if cl is not None and self._decay_counter >= 100:
            cl.decay_stale()
            self._decay_counter = 0
        # MetaCognition visit decay every 150 frames
        self._mc_decay_counter = getattr(self, '_mc_decay_counter', 0) + 1
        if mc is not None and hasattr(mc, 'decay_visits') and self._mc_decay_counter >= 150:
            mc.decay_visits()
            self._mc_decay_counter = 0
        # Log cognition state every 50 frames for debugging
        if getattr(self, '_cog_log_counter', 0) % 50 == 0:
            attrs = []
            if wm is not None: attrs.append(f'wm_load={wm.load:.2f}')
            if mc is not None: attrs.append(f'consec={mc._consecutive_failures}')
            strat = getattr(context, 'cognition_strategy', '')
            if strat: attrs.append(f'strat={strat}')
            if gs is not None: attrs.append(f'gs_depth={gs.depth}')
            if cl is not None: attrs.append(f'chunks={cl.count}')
            if em is not None: attrs.append(f'traces={len(em._traces)}')
            logger.debug('Cognition: %s', ' | '.join(a for a in attrs if a))
        self._cog_log_counter = getattr(self, '_cog_log_counter', 0) + 1

    def note_transition(self, transition: ObservedTransition) -> None:
        """Apply outcome-sensitive cognition only after the next observation."""
        outcome = str(transition.local_outcome or "unknown")
        self._last_observed_outcome = outcome
        pending = getattr(self, "_pending_cognition_decision", None)
        if pending is None:
            return
        if transition.boundary in {"game", "restart"} and not transition.action_name:
            self._pending_cognition_decision = None
            return
        if transition.action_name and transition.action_name != pending.get("action_name"):
            if transition.boundary is not None:
                self._pending_cognition_decision = None
            return
        if (
            transition.decision_source
            and transition.decision_source != pending.get("source")
        ):
            return

        if outcome in {"progress", "advanced"}:
            cognition_outcome = "progress"
        elif outcome in {"failed", "stalled"}:
            cognition_outcome = outcome
        else:
            if transition.boundary is not None:
                self._pending_cognition_decision = None
            return

        if self.meta_cognition is not None:
            self.meta_cognition.note_outcome(cognition_outcome)
        subgoal = pending.get("subgoal")
        if (
            self.goal_stack is not None
            and subgoal
            and cognition_outcome in {"failed", "stalled"}
        ):
            found = self.goal_stack.find(str(subgoal))
            if (
                found is not None
                and found.status.name in ("PENDING", "ACTIVE")
                and self.goal_stack.depth > 1
                and not self.goal_stack.check_progress(str(subgoal))
            ):
                self.goal_stack.backtrack()
        self._pending_cognition_decision = None

    def reset_episode(self) -> None:
        """Clear episode-local cognition without discarding learned traces."""
        if self.working_memory is not None:
            self.working_memory.clear()
        if self.meta_cognition is not None:
            self.meta_cognition.reset()
        if self.goal_stack is not None:
            self.goal_stack.reset()
        if self.attention is not None and hasattr(self.attention, "reset_visitation"):
            self.attention.reset_visitation()
        self._pending_cognition_decision = None
        self._last_observed_outcome = "unknown"
        self._last_cog_frame = -1
        self._cog_log_counter = 0

    def _all_legal_proposals_are_score_floor(
        self,
        context: ExecutiveContext,
        proposals: list,
    ) -> bool:
        legal = [
            proposal
            for proposal in proposals
            if getattr(proposal, "legal", False)
        ]
        if not legal:
            return False
        if all(float(getattr(proposal, "score", 0.0)) <= -0.95 for proposal in legal):
            return True
        if self._should_force_probe_recovery_score_floor(context, legal):
            return True
        if context.memory.stagnation_steps < 6:
            return False
        recent_actions = [
            action_name
            for action_name in context.memory.recent_actions()[-6:]
            if action_name and action_name != "RESET"
        ]
        diverse_stall_cycle = (
            len(recent_actions) >= 6
            and len(set(recent_actions)) >= 4
        )
        basin_status = {}
        if context.memory_bundle is not None:
            basin_status = dict(context.memory_bundle.basin_status or {})
        basin_exhausted = bool(basin_status.get("is_exhausted")) or context.memory.is_basin_exhausted(
            context.observation
        )
        if not basin_exhausted and not diverse_stall_cycle:
            return False
        if any(str(getattr(proposal, "source", "") or "") == "llm" for proposal in legal):
            return False
        return all(float(getattr(proposal, "score", 0.0)) <= 0.0 for proposal in legal)

    def _should_force_probe_recovery_score_floor(
        self,
        context: ExecutiveContext,
        legal_proposals: list,
    ) -> bool:
        if active_cell_count(context.observation.image_input) <= 0:
            return False
        if context.memory.stagnation_steps < 5:
            return False
        if any(str(getattr(proposal, "source", "") or "") == "llm" for proposal in legal_proposals):
            return False
        if not all(float(getattr(proposal, "score", 0.0)) <= 0.0 for proposal in legal_proposals):
            return False
        recent_actions = [
            action_name
            for action_name in context.memory.recent_actions()[-5:]
            if action_name and action_name != "RESET"
        ]
        if len(recent_actions) < 5:
            return False
        if recent_actions[:2] != ["ACTION6", "ACTION6"]:
            return False
        return len(set(recent_actions[2:])) >= 3

    def _should_skip_early_probe_llm(
        self,
        context: ExecutiveContext,
        symbolic_proposals: list,
    ) -> bool:
        del symbolic_proposals
        if context.force_llm_replan or context.replan_mode:
            return False
        if context.memory.stagnation_steps > 1:
            return False
        if context.memory.most_recent_action() != "ACTION6":
            return False
        if active_cell_count(context.observation.image_input) <= 0:
            return False
        return "ACTION6" in context.observation.legal_actions

    def _should_enter_replan_mode(self, context: ExecutiveContext) -> bool:
        basin_status = {}
        working_memory = {}
        if context.memory_bundle is not None:
            basin_status = dict(context.memory_bundle.basin_status or {})
            working_memory = dict(context.memory_bundle.working_memory or {})
        basin_exhausted = bool(basin_status.get("is_exhausted")) or context.memory.is_basin_exhausted(
            context.observation
        )
        failure_pressure = self._working_memory_failure_pressure(working_memory)
        recent_actions = [
            action_name
            for action_name in context.memory.recent_actions()[-6:]
            if action_name and action_name != "RESET"
        ]
        diverse_stall_cycle = (
            context.memory.stagnation_steps >= 6
            and len(recent_actions) >= 6
            and len(set(recent_actions)) >= 4
        )
        return (
            (
                context.memory.stagnation_steps
                >= context.config.executive_replan_stagnation_threshold
            )
            or (
                basin_exhausted
                and context.memory.stagnation_steps >= 4
            )
            or (
                failure_pressure["open_mode_count"] >= 3
                and context.memory.stagnation_steps >= 3
            )
            or (
                failure_pressure["max_failure_streak"] >= 3
                and failure_pressure["recent_failure_count"] >= 2
                and context.memory.stagnation_steps >= 2
            )
            or diverse_stall_cycle
        ) and (
            context.observation.game_state not in context.config.reset_state_names
            and context.observation.game_state != context.config.win_state_name
        )

    def _working_memory_failure_pressure(self, working_memory: dict) -> dict[str, int]:
        open_failure_modes = list(working_memory.get("open_failure_modes", []) or [])
        recent_failures = list(working_memory.get("recent_failures", []) or [])
        failure_streaks = list(working_memory.get("failure_streaks", []) or [])
        max_failure_streak = 0
        for streak in failure_streaks:
            if not isinstance(streak, dict):
                continue
            try:
                count = int(streak.get("count", 0) or 0)
            except (TypeError, ValueError):
                count = 0
            if count > max_failure_streak:
                max_failure_streak = count
        return {
            "open_mode_count": len(open_failure_modes),
            "recent_failure_count": len(recent_failures),
            "max_failure_streak": max_failure_streak,
        }

    def _working_memory_failure_signal_for_action(
        self,
        working_memory: dict,
        action_name: str,
    ) -> dict[str, int]:
        target = str(action_name or "")
        recent_failure_count = 0
        for row in list(working_memory.get("recent_failures", []) or []):
            if not isinstance(row, dict):
                continue
            if str(row.get("action_name", "") or "") == target:
                recent_failure_count += 1
        failure_streak = 0
        for row in list(working_memory.get("failure_streaks", []) or []):
            if not isinstance(row, dict):
                continue
            if str(row.get("action_name", "") or "") != target:
                continue
            try:
                failure_streak = max(failure_streak, int(row.get("count", 0) or 0))
            except (TypeError, ValueError):
                continue
        return {
            "recent_failure_count": recent_failure_count,
            "failure_streak": failure_streak,
        }

    def _working_memory_focus_rank(
        self,
        working_memory: dict,
        action_name: str,
    ) -> int | None:
        target = str(action_name or "")
        rank = 0
        for raw_name in list(working_memory.get("procedural_focus_actions", []) or []):
            if str(raw_name or "") == target:
                return rank
            rank += 1
        for raw_name in list(working_memory.get("semantic_focus_actions", []) or []):
            if str(raw_name or "") == target:
                return rank
            rank += 1
        return None

    def _should_force_emergency_llm_replan(self, context: ExecutiveContext) -> bool:
        if context.observation.game_state in context.config.reset_state_names:
            return False
        if context.observation.game_state == context.config.win_state_name:
            return False
        if "RESET" in context.observation.legal_actions:
            return False
        if context.probe_dead_end_replan:
            return True
        if context.memory.can_call_llm(context.config.llm_max_calls_per_episode):
            return False
        basin_status = {}
        working_memory = {}
        if context.memory_bundle is not None:
            basin_status = dict(context.memory_bundle.basin_status or {})
            working_memory = dict(context.memory_bundle.working_memory or {})
        basin_exhausted = bool(basin_status.get("is_exhausted")) or context.memory.is_basin_exhausted(
            context.observation
        )
        if not basin_exhausted:
            return False
        failure_pressure = self._working_memory_failure_pressure(working_memory)
        recent_actions = [
            action_name
            for action_name in context.memory.recent_actions()[-8:]
            if action_name and action_name != "RESET"
        ]
        if (
            failure_pressure["max_failure_streak"] >= 3
            and failure_pressure["recent_failure_count"] >= 2
            and context.memory.stagnation_steps >= 4
            and len(recent_actions) >= 4
        ):
            return len(set(recent_actions)) >= 2
        if context.memory.stagnation_steps < 6 or len(recent_actions) < 6:
            return False
        return len(set(recent_actions)) >= 3

    def _should_force_probe_dead_end_llm_replan(
        self,
        context: ExecutiveContext,
    ) -> bool:
        if active_cell_count(context.observation.image_input) <= 0:
            return False
        if context.memory.stagnation_steps < 2:
            return False
        working_memory = {}
        if context.memory_bundle is not None:
            working_memory = dict(context.memory_bundle.working_memory or {})
        recent_actions = [
            action_name
            for action_name in context.memory.recent_actions()[-2:]
            if action_name
        ]
        if recent_actions == ["ACTION6", "ACTION6"]:
            return "ACTION6" in context.observation.legal_actions
        if context.memory.most_recent_action() != "ACTION6":
            return False
        action_failure = self._working_memory_failure_signal_for_action(
            working_memory,
            "ACTION6",
        )
        return (
            "ACTION6" in context.observation.legal_actions
            and action_failure["failure_streak"] >= 2
            and action_failure["recent_failure_count"] >= 1
        )

    def _score_floor_fallback(self, context: ExecutiveContext, rescored: list) -> ActionDecision:
        candidate_actions = self._ordered_score_floor_actions(rescored)
        frame_cache = getattr(context, "multi_agent_frame_cache", {}) or {}
        if (
            frame_cache.get("shape_agent_mode")
            and not frame_cache.get("ready_to_switch")
        ):
            non_switch_candidates = [
                action_name for action_name in candidate_actions if action_name != "ACTION5"
            ]
            if non_switch_candidates:
                candidate_actions = non_switch_candidates
        allow_board_probe = self._should_allow_score_floor_board_probe(
            context,
            candidate_actions,
        )
        if allow_board_probe and "ACTION6" in context.observation.legal_actions and "ACTION6" not in candidate_actions:
            candidate_actions.append("ACTION6")
        context.score_floor_proposals = list(rescored)
        reopen_summary = context.memory.replan_frontier_reopen_summary(
            context.observation,
            action_names=candidate_actions,
        )
        active_reopen_candidates = [
            action_name
            for action_name, count in dict(reopen_summary.get("action_counts") or {}).items()
            if int(count or 0) > 0
        ]
        active_reopen_summary: dict[str, object] = {}
        if 1 < len(active_reopen_candidates) < len(candidate_actions):
            active_reopen_summary = context.memory.replan_frontier_reopen_summary(
                context.observation,
                action_names=active_reopen_candidates,
            )
        context.score_floor_debug = dict(reopen_summary)
        if active_reopen_summary:
            context.score_floor_debug["active_reopen"] = dict(active_reopen_summary)
        reopen_detected = bool(reopen_summary.get("is_reopened")) or bool(
            active_reopen_summary.get("is_reopened")
        )
        if reopen_detected and "RESET" not in context.observation.legal_actions:
            context.score_floor_debug["pending_reset"] = True
            context.memory.mark_pending_score_floor_reopen(
                context.observation,
                candidate_actions=active_reopen_candidates or candidate_actions,
                debug=context.score_floor_debug,
            )
        if (
            "RESET" in context.observation.legal_actions
            and reopen_detected
        ):
            context.memory.clear_pending_score_floor_reopen()
            return ActionDecision(
                action_name="RESET",
                source="fallback",
                confidence=0.05,
                reason="score-floor frontier reopen reset",
                metadata={"score_floor_debug": dict(context.score_floor_debug)},
                used_fallback=True,
            )
        if (
            context.memory.is_replan_frontier_exhausted(
                context.observation,
                action_names=candidate_actions,
            )
            and "RESET" in context.observation.legal_actions
        ):
            return ActionDecision(
                action_name="RESET",
                source="fallback",
                confidence=0.05,
                reason="score-floor frontier exhausted reset",
                metadata={"score_floor_debug": dict(context.score_floor_debug)},
                used_fallback=True,
            )
        if self._should_reset_exhausted_score_floor(context, candidate_actions):
            return ActionDecision(
                action_name="RESET",
                source="fallback",
                confidence=0.05,
                reason="score-floor basin reset",
                metadata={"score_floor_debug": dict(context.score_floor_debug)},
                used_fallback=True,
            )
        memory_escape = self._score_floor_memory_escape(context)
        if memory_escape is not None:
            return memory_escape
        fallback = self._memory_aware_fallback(
            context,
            candidate_actions=candidate_actions,
            exploratory=True,
            respect_suppressed_actions=False,
            honor_recommended_actions=True,
            allow_exploratory_board_probe=allow_board_probe,
        )
        fallback.reason = "score-floor executive fallback"
        fallback.confidence = min(float(fallback.confidence or 0.05), 0.05)
        return fallback

    def _ordered_score_floor_actions(self, rescored: list) -> list[str]:
        ordered: list[str] = []
        for proposal in rescored:
            if not getattr(proposal, "legal", False):
                continue
            action_name = str(getattr(proposal, "action_name", "") or "")
            if not action_name or action_name in ordered:
                continue
            ordered.append(action_name)
        non_probe = [action_name for action_name in ordered if action_name != "ACTION6"]
        return non_probe or ordered

    def _should_allow_score_floor_board_probe(
        self,
        context: ExecutiveContext,
        candidate_actions: list[str],
    ) -> bool:
        frame_cache = getattr(context, "multi_agent_frame_cache", {}) or {}
        if (
            frame_cache.get("shape_agent_mode")
            or frame_cache.get("mirror_puzzle_active")
            or frame_cache.get("active_shape_target")
        ):
            return False
        if "ACTION6" not in context.observation.legal_actions:
            return False
        if active_cell_count(context.observation.image_input) <= 0:
            return False
        if self._board_probe_action_is_stale(context, "ACTION6"):
            return False
        recent_actions = [
            action_name
            for action_name in context.memory.recent_actions()[-6:]
            if action_name and action_name != "RESET"
        ]
        non_probe_attempts = [
            action_name for action_name in recent_actions if action_name != "ACTION6"
        ]
        if len(set(non_probe_attempts)) < min(3, max(len(candidate_actions) - 1, 1)):
            return False
        return True

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

    def _should_reset_exhausted_score_floor(
        self,
        context: ExecutiveContext,
        candidate_actions: list[str],
    ) -> bool:
        if "RESET" not in context.observation.legal_actions:
            return False
        if not candidate_actions:
            return False
        if context.memory.stagnation_steps < max(4, len(candidate_actions)):
            return False
        if not context.memory.is_basin_exhausted(context.observation):
            return False
        recent_actions = context.memory.recent_actions()
        window = recent_actions[-max(len(candidate_actions) * 2, 8) :]
        if len(window) < len(candidate_actions):
            return False
        return all(action_name in window for action_name in candidate_actions)

    def _score_floor_memory_escape(
        self,
        context: ExecutiveContext,
    ) -> ActionDecision | None:
        llm_limit = context.config.llm_max_calls_per_episode
        if self.llm is None or context.memory.llm_calls < (llm_limit + 1):
            return None
        if not hasattr(self.llm, "memory_escape_proposals"):
            return None
        preferred_actions = self._score_floor_escape_preferred_actions(context)
        proposals = self.llm.memory_escape_proposals(
            context,
            "score-floor post-emergency memory escape",
            preferred_actions=preferred_actions or None,
        )
        if not proposals:
            return None
        if context.memory.is_replan_frontier_exhausted(
            context.observation,
            action_names=[proposal.action_name for proposal in proposals],
        ):
            return None
        proposal = proposals[0]
        metadata = dict(proposal.metadata)
        if context.score_floor_debug:
            metadata["score_floor_debug"] = dict(context.score_floor_debug)
        return ActionDecision(
            action_name=proposal.action_name,
            source="fallback",
            confidence=min(float(getattr(proposal, "confidence", 0.05) or 0.05), 0.05),
            reason="score-floor memory escape",
            subgoal=proposal.subgoal,
            metadata=metadata,
            used_fallback=True,
        )

    def _score_floor_escape_preferred_actions(
        self,
        context: ExecutiveContext,
    ) -> list[str]:
        ordered_actions = self._ordered_score_floor_actions(
            list(context.score_floor_proposals or [])
        )
        if not ordered_actions:
            return []
        working_memory = {}
        recommended_actions: list[str] = []
        trained_prior_bucket: dict[str, int] = {}
        if context.memory_bundle is not None:
            working_memory = dict(context.memory_bundle.working_memory or {})
            recommended_actions = [
                str(action_name or "")
                for action_name in list(context.memory_bundle.recommended_actions or [])
                if str(action_name or "") in ordered_actions
            ]
            trained_prior_bucket = {
                str(action_name): int(count)
                for action_name, count in dict(
                    context.memory_bundle.trained_prior_bucket or {}
                ).items()
                if str(action_name) in ordered_actions and int(count or 0) > 0
            }
        focus_actions = []
        for action_name in list(working_memory.get("procedural_focus_actions", []) or []):
            name = str(action_name or "")
            if name and name in ordered_actions and name not in focus_actions:
                focus_actions.append(name)
        for action_name in list(working_memory.get("semantic_focus_actions", []) or []):
            name = str(action_name or "")
            if name and name in ordered_actions and name not in focus_actions:
                focus_actions.append(name)
        prior_peak = max(trained_prior_bucket.values(), default=0)
        return sorted(
            ordered_actions,
            key=lambda action_name: (
                action_name not in focus_actions,
                recommended_actions.index(action_name)
                if action_name in recommended_actions
                else len(recommended_actions) + 1,
                -(
                    float(trained_prior_bucket.get(action_name, 0)) / float(prior_peak)
                    if prior_peak > 0
                    else 0.0
                ),
                self._working_memory_failure_signal_for_action(
                    working_memory,
                    action_name,
                )["failure_streak"],
                self._working_memory_failure_signal_for_action(
                    working_memory,
                    action_name,
                )["recent_failure_count"],
                ordered_actions.index(action_name),
            ),
        )

    def _probe_dead_end_escape(
        self,
        context: ExecutiveContext,
        rescored: list,
    ) -> ActionDecision | None:
        if not context.probe_dead_end_replan:
            return None
        legal = [
            proposal
            for proposal in rescored
            if getattr(proposal, "legal", False)
            and getattr(proposal, "action_name", "") != "RESET"
        ]
        if not legal:
            return None
        top_legal = legal[0]
        if top_legal.source == "llm" and top_legal.action_name != "ACTION6":
            return None
        candidate_actions = self._probe_dead_end_escape_actions(context)
        if not candidate_actions:
            return None
        if self.llm is not None and hasattr(self.llm, "memory_escape_proposals"):
            proposals = self.llm.memory_escape_proposals(
                context,
                "probe-dead-end memory escape",
                preferred_actions=candidate_actions,
            )
            for proposal in proposals:
                if proposal.action_name not in candidate_actions:
                    continue
                metadata = self._finalize_decision_metadata(
                    context,
                    proposal.action_name,
                    dict(proposal.metadata),
                )
                return ActionDecision(
                    action_name=proposal.action_name,
                    source="fallback",
                    confidence=min(
                        float(getattr(proposal, "confidence", 0.05) or 0.05),
                        0.05,
                    ),
                    reason="probe-dead-end memory escape",
                    subgoal=proposal.subgoal,
                    metadata=metadata,
                    used_fallback=True,
                )
        fallback = self._memory_aware_fallback(
            context,
            candidate_actions=candidate_actions,
            exploratory=True,
            respect_suppressed_actions=False,
        )
        fallback.reason = "probe-dead-end executive escape"
        fallback.used_fallback = True
        return fallback

    def _probe_dead_end_followup_escape(
        self,
        context: ExecutiveContext,
        rescored: list,
    ) -> ActionDecision | None:
        if not context.memory.recent_probe_dead_end_escape(context.observation):
            return None
        legal = [
            proposal
            for proposal in rescored
            if getattr(proposal, "legal", False)
            and getattr(proposal, "action_name", "") != "RESET"
        ]
        if not legal:
            return None
        top_legal = legal[0]
        if top_legal.source == "llm" and top_legal.action_name != "ACTION6":
            return None
        candidate_actions = self._probe_dead_end_escape_actions(context)
        if not candidate_actions:
            return None
        fallback = self._memory_aware_fallback(
            context,
            candidate_actions=candidate_actions,
            exploratory=True,
            respect_suppressed_actions=False,
        )
        fallback.reason = "probe-dead-end followup escape"
        fallback.used_fallback = True
        return fallback

    def _probe_dead_end_escape_actions(
        self,
        context: ExecutiveContext,
    ) -> list[str]:
        candidates = [
            action_name
            for action_name in context.observation.legal_actions
            if action_name not in {"RESET", "ACTION6"}
        ]
        if not candidates:
            return []
        working_memory = {}
        if context.memory_bundle is not None:
            working_memory = dict(context.memory_bundle.working_memory or {})
        focus_actions = []
        for action_name in list(working_memory.get("procedural_focus_actions", []) or []):
            name = str(action_name or "")
            if name and name not in focus_actions:
                focus_actions.append(name)
        for action_name in list(working_memory.get("semantic_focus_actions", []) or []):
            name = str(action_name or "")
            if name and name not in focus_actions:
                focus_actions.append(name)
        frame_signature = str(
            context.observation.derived_features.get("frame_signature", "") or ""
        ) or None
        recent_actions = set(context.memory.recent_actions()[-3:])
        return sorted(
            candidates,
            key=lambda action_name: (
                action_name not in focus_actions,
                self._working_memory_failure_signal_for_action(
                    working_memory,
                    action_name,
                )["failure_streak"],
                self._working_memory_failure_signal_for_action(
                    working_memory,
                    action_name,
                )["recent_failure_count"],
                action_name in recent_actions,
                context.memory.context_outcome_counts(
                    action_name,
                    state_name=context.observation.game_state,
                    frame_signature=frame_signature,
                )["total"],
                action_name,
            ),
        )

    def _refresh_llm_memory_bundle(
        self,
        context: ExecutiveContext,
        symbolic_proposals: list,
        reactive_proposals: list,
    ) -> None:
        subgoal = self._select_contextual_subgoal(
            context,
            symbolic_proposals,
            reactive_proposals,
        )
        if not subgoal or not hasattr(context.hyperon, "contextual_memory_bundle"):
            return
        bundle = context.contextual_memory_bundles.get(subgoal)
        if bundle is None:
            bundle = context.hyperon.contextual_memory_bundle(
                context.observation,
                subgoal,
                list(context.observation.legal_actions),
            )
        context.contextual_memory_bundles[subgoal] = bundle
        context.llm_memory_bundle = bundle
        context.llm_subgoal_hint = subgoal

    def _select_contextual_subgoal(
        self,
        context: ExecutiveContext,
        symbolic_proposals: list,
        reactive_proposals: list,
    ) -> str | None:
        ranked = sorted(
            [
                proposal
                for proposal in [*symbolic_proposals, *reactive_proposals]
                if getattr(proposal, "subgoal", None)
            ],
            key=lambda proposal: (
                self._contextual_board_probe_priority(context, proposal),
                self._contextual_memory_priority(context, proposal),
                float(getattr(proposal, "confidence", 0.0)),
            ),
            reverse=True,
        )
        if not ranked:
            return None
        return str(ranked[0].subgoal)

    def _contextual_memory_priority(
        self,
        context: ExecutiveContext,
        proposal,
    ) -> float:
        global_bundle = context.memory_bundle or MemoryBundle()
        bundle = self._contextual_bundle_for_proposal(context, proposal)
        working_memory = dict(global_bundle.working_memory or {})
        action_name = str(getattr(proposal, "action_name", "") or "")
        score = 0.0
        focus_rank = self._working_memory_focus_rank(working_memory, action_name)
        if focus_rank is not None:
            score += max(0.10, 0.24 - (focus_rank * 0.06))
        recommended_actions = [
            str(name or "")
            for name in list(
                (bundle.recommended_actions or [])
                or (global_bundle.recommended_actions or [])
            )
            if str(name or "")
        ]
        if action_name in recommended_actions:
            score += max(0.08, 0.18 - (recommended_actions.index(action_name) * 0.05))
        trained_prior_bucket = {
            str(name): int(count)
            for name, count in dict(bundle.trained_prior_bucket or {}).items()
            if str(name) and int(count or 0) > 0
        }
        if action_name in trained_prior_bucket:
            prior_peak = max(trained_prior_bucket.values(), default=0)
            if prior_peak > 0:
                score += 0.12 * (
                    float(trained_prior_bucket.get(action_name, 0)) / float(prior_peak)
                )
                score += min(
                    0.06,
                    float(trained_prior_bucket.get(action_name, 0)) * 0.005,
                )
        action_failure = self._working_memory_failure_signal_for_action(
            working_memory,
            action_name,
        )
        score -= min(
            0.22,
            (action_failure["failure_streak"] * 0.05)
            + (action_failure["recent_failure_count"] * 0.03),
        )
        return score

    def _contextual_bundle_for_proposal(
        self,
        context: ExecutiveContext,
        proposal,
    ) -> MemoryBundle:
        subgoal = str(getattr(proposal, "subgoal", "") or "")
        if not subgoal:
            return context.memory_bundle or MemoryBundle()
        bundle = context.contextual_memory_bundles.get(subgoal)
        if bundle is not None:
            return bundle
        if hasattr(context.hyperon, "contextual_memory_bundle"):
            bundle = context.hyperon.contextual_memory_bundle(
                context.observation,
                subgoal,
                list(context.observation.legal_actions),
            )
            context.contextual_memory_bundles[subgoal] = bundle
            return bundle
        return context.memory_bundle or MemoryBundle()

    def _contextual_board_probe_priority(
        self,
        context: ExecutiveContext,
        proposal,
    ) -> int:
        image_input = context.observation.image_input
        if active_cell_count(image_input) <= 0:
            return 0
        if getattr(proposal, "action_name", "") != "ACTION6":
            return 0
        metadata = getattr(proposal, "metadata", {})
        if click_target_bias(image_input, metadata) < 0:
            return 0
        if canonical_click_target_bonus(image_input, metadata) > 0:
            return 2
        return 1

    def _finalize_decision_metadata(
        self,
        context: ExecutiveContext,
        action_name: str,
        metadata: dict[str, object],
    ) -> dict[str, object]:
        if action_name != "ACTION6":
            return metadata
        image_input = context.observation.image_input
        if active_cell_count(image_input) <= 0:
            return metadata
        if canonical_click_target_bonus(image_input, metadata) > 0:
            return metadata
        return self._board_probe_metadata_for_context(context)

    def _finalize_decision_subgoal(
        self,
        context: ExecutiveContext,
        action_name: str,
        subgoal: str | None,
    ) -> str | None:
        if action_name != "ACTION6":
            return subgoal
        if active_cell_count(context.observation.image_input) <= 0:
            return subgoal
        normalized = str(subgoal or "")
        if normalized and normalized not in {
            "maintain-control-loop",
            "reuse-known-good-action",
        }:
            return subgoal
        return "inspect-active-board"

    def _memory_aware_fallback(
        self,
        context: ExecutiveContext,
        *,
        candidate_actions: list[str] | None = None,
        exploratory: bool = False,
        respect_suppressed_actions: bool = True,
        honor_recommended_actions: bool = False,
        allow_exploratory_board_probe: bool = False,
    ) -> ActionDecision:
        bundle = context.memory_bundle or MemoryBundle()
        working_memory = dict(bundle.working_memory or {})
        trained_prior_bucket = {
            str(action_name): int(count)
            for action_name, count in dict(bundle.trained_prior_bucket or {}).items()
            if str(action_name) and int(count or 0) > 0
        }
        legal_actions = list(candidate_actions or context.observation.legal_actions)
        legal_actions_known = bool(
            context.observation.derived_features.get("legal_actions_known", True)
        )
        if not legal_actions:
            return ActionDecision(
                action_name="RESET",
                source="fallback",
                confidence=0.01,
                reason=(
                    "fail-closed empty legality reset"
                    if legal_actions_known
                    else "fail-closed unknown legality reset"
                ),
                used_fallback=True,
            )
        suppressed = (
            set(bundle.suppressed_actions or []) if respect_suppressed_actions else set()
        )
        non_reset_legal = [
            action_name
            for action_name in legal_actions
            if action_name != "RESET" and action_name not in suppressed
        ]
        if exploratory:
            exhausted_replan_actions = context.memory.replan_frontier_exhausted_actions(
                context.observation,
                action_names=non_reset_legal,
            )
            if exhausted_replan_actions and len(
                set(non_reset_legal) - exhausted_replan_actions
            ) > 0:
                non_reset_legal = [
                    action_name
                    for action_name in non_reset_legal
                    if action_name not in exhausted_replan_actions
                ]
            alternative_actions = [
                action_name for action_name in non_reset_legal if action_name != "ACTION6"
            ]
            if alternative_actions and not allow_exploratory_board_probe:
                non_reset_legal = alternative_actions
            attempted_fallback_actions = context.memory.fallback_attempted_actions(
                context.observation,
                action_names=non_reset_legal,
            )
            if allow_exploratory_board_probe:
                attempted_fallback_actions.discard("ACTION6")
            if attempted_fallback_actions and len(set(non_reset_legal) - attempted_fallback_actions) > 0:
                non_reset_legal = [
                    action_name
                    for action_name in non_reset_legal
                    if action_name not in attempted_fallback_actions
                ]
        if non_reset_legal:
            board_probe_action = preferred_board_probe_action(
                non_reset_legal,
                context.observation.image_input,
                suppressed_actions=suppressed,
            )
            recommended_rank = {
                action_name: index
                for index, action_name in enumerate(bundle.recommended_actions or [])
                if action_name in non_reset_legal
            }
            action_name = max(
                non_reset_legal,
                key=lambda candidate: self._fallback_action_score(
                    context,
                    candidate,
                    working_memory=working_memory,
                    trained_prior_bucket=trained_prior_bucket,
                    recommended_rank=recommended_rank.get(candidate),
                    board_probe_action=board_probe_action,
                    exploratory=exploratory,
                    honor_recommended_actions=honor_recommended_actions,
                ),
            )
            metadata = {}
            if hasattr(context.adapter, "default_metadata_for_action"):
                metadata = dict(
                    context.adapter.default_metadata_for_action(action_name) or {}
                )
            if action_name == "ACTION6" and context.observation.image_input is not None:
                metadata = self._board_probe_metadata_for_context(context)
            return ActionDecision(
                action_name=action_name,
                source="fallback",
                confidence=0.08,
                reason=(
                    "memory-aware executive fallback"
                    if action_name in recommended_rank
                    else "memory-aware legal fallback"
                ),
                subgoal=(
                    "inspect-active-board"
                    if action_name == "ACTION6" and context.observation.image_input is not None
                    else None
                ),
                metadata=metadata,
                used_fallback=True,
            )
        if not non_reset_legal and "RESET" in legal_actions:
            return ActionDecision(
                action_name="RESET",
                source="fallback",
                confidence=0.08,
                reason="memory-aware exhausted fallback reset",
                used_fallback=True,
            )
        return context.adapter.fallback_decision(context.observation)

    def _board_probe_metadata_for_context(
        self,
        context: ExecutiveContext,
    ) -> dict[str, int]:
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

    def _fallback_action_score(
        self,
        context: ExecutiveContext,
        action_name: str,
        *,
        working_memory: dict,
        trained_prior_bucket: dict[str, int],
        recommended_rank: int | None,
        board_probe_action: str | None,
        exploratory: bool,
        honor_recommended_actions: bool,
    ) -> float:
        recent_actions = context.memory.recent_actions()
        score = 0.0
        focus_rank = self._working_memory_focus_rank(working_memory, action_name)
        if focus_rank is not None:
            score += max(0.05, 0.16 - (focus_rank * 0.04))
        trained_prior_active = action_name in trained_prior_bucket
        if recommended_rank is not None and (
            honor_recommended_actions or trained_prior_active
        ):
            if exploratory:
                score += max(0.06, 0.18 - (recommended_rank * 0.05))
            else:
                score += max(0.20, 0.45 - (recommended_rank * 0.10))
        if trained_prior_active:
            max_count = max(trained_prior_bucket.values(), default=0)
            if max_count > 0:
                normalized_count = float(trained_prior_bucket.get(action_name, 0)) / float(
                    max_count
                )
                score += 0.12 * normalized_count
        if board_probe_action and action_name == board_probe_action:
            score += 0.12 if exploratory else 0.18
        if action_name == context.memory.most_recent_action():
            score -= 0.08
        score += self._fallback_recency_bonus(recent_actions, action_name, exploratory=exploratory)
        action_failure = self._working_memory_failure_signal_for_action(
            working_memory,
            action_name,
        )
        score -= min(
            0.20,
            (action_failure["failure_streak"] * 0.04)
            + (action_failure["recent_failure_count"] * 0.03),
        )
        score -= context.memory.repeated_failure_penalty(action_name)
        score -= context.memory.loop_penalty(action_name)
        score -= context.memory.context_stall_penalty(
            action_name,
            state_name=context.observation.game_state,
            frame_signature=str(
                context.observation.derived_features.get("frame_signature", "") or ""
            )
            or None,
        )
        return score

    def _fallback_recency_bonus(
        self,
        recent_actions: list[str],
        action_name: str,
        *,
        exploratory: bool,
    ) -> float:
        if not recent_actions:
            return 0.0
        tail = recent_actions[-8:]
        for distance, recent_action in enumerate(reversed(tail), start=1):
            if recent_action != action_name:
                continue
            if exploratory:
                return -max(0.0, 0.28 - ((distance - 1) * 0.05))
            return -max(0.0, 0.10 - ((distance - 1) * 0.02))
        return 0.18 if exploratory else 0.04


class ExecutiveRuntime:
    def __init__(
        self,
        adapter,
        executive: Executive,
        memory: EpisodicMemory,
        hyperon,
        llm_backend,
        config: RuntimeConfig,
    ) -> None:
        self.adapter = adapter
        self.executive = executive
        self.memory = memory
        self.hyperon = hyperon
        self.llm_backend = llm_backend
        self.config = config
        self._bound_game_id = ""
        self._last_observation: Observation | None = None

    def bind_game(self, game_id: str) -> None:
        game_key = str(game_id or "")
        changed_game = game_key != self._bound_game_id
        if changed_game:
            self._bound_game_id = game_key
            self._last_observation = None
            if hasattr(self.executive, "reset_agents"):
                self.executive.reset_agents()
            elif hasattr(self.executive, "reset_episode"):
                self.executive.reset_episode()
        self.adapter.bind_game(game_id)
        self.memory.bind_game(game_id)
        if hasattr(self.hyperon, "bind_game"):
            self.hyperon.bind_game(game_id)

    def decide(self, frames, latest_frame) -> ActionDecision:
        observation = self._observe_and_persist(frames, latest_frame)
        context = ExecutiveContext(
            observation=observation,
            goals=self.default_goals(),
            memory=self.memory,
            adapter=self.adapter,
            hyperon=self.hyperon,
            llm_backend=self.llm_backend,
            config=self.config,
            episode_id=observation.game_id,
        )
        return self.executive.decide(context)

    def note_terminal_observation(self, frames, latest_frame) -> None:
        self._observe_and_persist(frames, latest_frame)

    def _observe_and_persist(self, frames, latest_frame):
        previous = self._last_observation
        observation = self.adapter.observe(frames, latest_frame)
        self.memory.note_observation(observation)
        records = self.memory.drain_new_records()
        if hasattr(self.hyperon, "record_transitions"):
            self.hyperon.record_transitions(records)
        else:
            for record in records:
                self.hyperon.record_transition(record)
        transitions = [
            self._classify_observed_transition(previous, observation, record)
            for record in records
        ]
        boundary = self._observation_boundary(previous, observation)
        if not transitions and boundary is not None:
            transitions.append(
                ObservedTransition(
                    game_id=observation.game_id,
                    local_outcome="unknown",
                    boundary=boundary,
                )
            )
        if hasattr(self.executive, "note_transition"):
            for transition in transitions:
                self.executive.note_transition(transition)
        self._last_observation = observation
        return observation

    def _classify_observed_transition(
        self,
        previous: Observation | None,
        current: Observation,
        record: EpisodeRecord,
    ) -> ObservedTransition:
        boundary = self._observation_boundary(previous, current)
        previous_position = self._observation_player_position(previous)
        current_position = self._observation_player_position(current)
        player_moved = bool(
            previous_position is not None
            and current_position is not None
            and previous_position != current_position
        )
        previous_signature = self._observation_frame_signature(previous)
        current_signature = self._observation_frame_signature(current)
        frame_changed = bool(
            previous_signature
            and current_signature
            and previous_signature != current_signature
        )

        if boundary in {"win", "level"} or int(record.progress_delta or 0) > 0:
            outcome = "progress"
        elif boundary == "game_over" or str(record.outcome or "") == "failed":
            outcome = "failed"
        elif boundary in {"restart", "game"}:
            outcome = "unknown"
        elif player_moved:
            outcome = "advanced"
        elif previous_signature and current_signature and not frame_changed:
            outcome = "stalled"
        elif frame_changed:
            outcome = "unknown"
        else:
            outcome = "stalled" if str(record.outcome or "") == "stalled" else "unknown"

        return ObservedTransition(
            game_id=current.game_id,
            action_name=str(record.action_name or ""),
            decision_source=str(record.decision_source or ""),
            decision_subgoal=record.decision_subgoal,
            local_outcome=outcome,
            player_moved=player_moved,
            frame_changed=frame_changed,
            boundary=boundary,
            record=record,
        )

    def _observation_boundary(
        self,
        previous: Observation | None,
        current: Observation,
    ) -> str | None:
        if previous is None:
            return None
        if str(previous.game_id or "") != str(current.game_id or ""):
            return "game"
        state = str(current.game_state or "").upper()
        if state == str(self.config.win_state_name or "").upper():
            return "win"
        if int(current.levels_completed or 0) > int(previous.levels_completed or 0):
            return "level"
        reset_states = {
            str(name or "").upper() for name in self.config.reset_state_names
        }
        if state in reset_states:
            return "game_over"
        previous_state = str(previous.game_state or "").upper()
        if previous_state in reset_states and state not in reset_states:
            return "restart"
        if (
            bool(current.derived_features.get("step_index_reliable", True))
            and bool(previous.derived_features.get("step_index_reliable", True))
            and int(current.step_index or 0) < int(previous.step_index or 0)
        ):
            return "restart"
        return None

    @staticmethod
    def _observation_player_position(
        observation: Observation | None,
    ) -> tuple[int, int] | None:
        if observation is None:
            return None
        value = observation.derived_features.get("player_position")
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return int(value[0]), int(value[1])
        return None

    @staticmethod
    def _observation_frame_signature(observation: Observation | None) -> str:
        if observation is None:
            return ""
        return str(observation.derived_features.get("frame_signature", "") or "")

    def default_goals(self) -> list[GoalState]:
        return [
            GoalState(
                name="finish-episode",
                target_state=self.config.win_state_name,
                priority=1.0,
            ),
            GoalState(name="avoid-reset-loop", priority=0.8),
            GoalState(name="prefer-forward-progress", priority=0.9),
        ]
