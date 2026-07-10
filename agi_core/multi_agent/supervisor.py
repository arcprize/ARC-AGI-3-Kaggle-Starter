from __future__ import annotations

import logging
import time
from typing import Any

from agi_core.multi_agent.agents import (
    ExplorationAgent,
    InteractionAgent,
    MovementAgent,
    PuzzleAgent,
    prepare_frame_cache,
)
from agi_core.multi_agent.consensus import ConsensusModule
from agi_core.multi_agent.strategy_agent import StrategyAgent
from agi_core.perception.shape_recognition import ensure_shape_recognition
from agi_core.cuda import is_enabled as _cuda_is_enabled, init_cuda
from agi_core.orch_or.analyzer import build_game_model
from agi_core.orch_or.orchestrator import ObjectiveReduction
from agi_core.orch_or.parallel import ThreadPoolManager
from agi_core.types import ActionDecision, AgentProposal, ExecutiveContext

logger = logging.getLogger(__name__)


def _gameplay_memory_db_path(config: Any) -> Any:
    helper = getattr(config, "gameplay_memory_db_path", None)
    if callable(helper):
        return helper()
    return getattr(config, "memory_db_path", None)


class AgentSupervisor(ObjectiveReduction):
    """Multi-agent supervisor that replaces ObjectiveReduction when enabled.
    
    Manages 4 parallel strategy agents, runs them each frame, collects
    their proposals via the ConsensusModule, and selects the best action.
    Falls back to ObjectiveReduction's plan system when agents cannot
    reach consensus.
    """

    def __init__(
        self,
        executive: Any,
        config: Any,
        agents: list[StrategyAgent] | None = None,
        consensus: ConsensusModule | None = None,
    ) -> None:
        super().__init__(executive=executive, config=config)
        self._agents = agents or [
            MovementAgent(),
            ExplorationAgent(),
            InteractionAgent(),
            PuzzleAgent(),
        ]
        self._consensus = consensus or ConsensusModule()
        self._agent_map: dict[str, StrategyAgent] = {
            a.agent_id: a for a in self._agents
        }
        self._workers = getattr(config, 'multi_agent_workers', 4)
        self._agent_timeout = max(0.5, float(getattr(config, 'planner_parallel_timeout', None) or 2.0))
        self._agent_health: dict[str, dict[str, float]] = {}
        self._agent_metrics: dict[str, dict[str, float]] = {}
        # Persistent solver selector (preserves state across frames)
        self._solver_selector = None
        self._last_solver_action: str = ""
        self._last_solver_player_pos: tuple[int, int] | None = None
        self._solver_stall_count: int = 0
        self._solver_cooldown: int = 0  # frames to skip solver after stagnation
        self._solver_recent_states: list[tuple[str, tuple[int, int] | None]] = []
        self._sticky_shape_target: dict[str, Any] | None = None
        # Initialise CUDA if enabled
        if getattr(config, 'cuda_enabled', False):
            init_cuda(getattr(config, 'cuda_device', 0))

    # ── public API ────────────────────────────────────────────────────

    def decide(self, context: ExecutiveContext) -> ActionDecision:
        """Run multi-agent consensus for the current frame.
        
        Falls back to ObjectiveReduction.decide() when:
        - No agents produce proposals
        - Consensus confidence is below threshold
        - An agent-free fallback is needed (score-floor reopen etc.)
        """
        # Set cognition strategy early (before any early return paths)
        context.cognition_strategy = "multi-agent"

        # Log health if any stale tasks remain from previous frame
        stale_count = ThreadPoolManager.active_worker_count()
        if stale_count > self._workers:
            logger.debug("AgentSupervisor: %d active workers from previous frame", stale_count)

        # Let the base Executive handle early exits first
        context.hyperon.sync(context.observation, context.memory, context.goals)
        context.memory_bundle = context.hyperon.current_memory_bundle()

        # Check score-floor reopen before running agents
        pending_reopen = context.memory.pending_score_floor_reopen(context.observation)
        if pending_reopen is not None and "RESET" in context.observation.legal_actions:
            decision = self._fallback_executive(context)
            self._finalize_cognition(context, decision)
            return decision

        self._stabilize_shape_frame_cache(context, prepare_frame_cache(context))

        # Try solver system first (respect cooldown after stagnation)
        solver_decision = None
        if self._solver_cooldown <= 0:
            solver_decision = self._try_solver_direct(context)
        else:
            self._solver_cooldown -= 1
        if solver_decision is not None:
            return solver_decision

        frame_cache = prepare_frame_cache(context)
        if frame_cache.get("shape_agent_mode") and frame_cache.get("active_shape_target"):
            agent_first = self._decide_from_agents(context)
            if agent_first is not None:
                return agent_first

        # Check circuit breakers — skip agents that have failed repeatedly
        active_agents = [a for a in self._agents if not self._is_circuit_open(a.agent_id)]
        if len(active_agents) < len(self._agents):
            logger.debug("AgentSupervisor: %d agents circuit-broken", len(self._agents) - len(active_agents))
        if not active_agents:
            return super().decide(context)

        prepare_frame_cache(context)

        # Run all agents in parallel with sync fallback for timed-out agents
        agent_proposals = self._run_agents(active_agents, context)
        participating_agents = set(agent_proposals)

        # Periodic pool health check (every 100 frames)
        self._health_counter = getattr(self, '_health_counter', 0) + 1
        if self._health_counter >= 100:
            active = ThreadPoolManager.active_worker_count()
            if active > self._workers * 2:
                logger.warning("AgentSupervisor: %d active pool workers (expected ≤%d)", active, self._workers)
            # Log agent health summary
            for aid, h in self._agent_health.items():
                if h.get("consecutive_failures", 0) >= 2:
                    logger.debug("AgentSupervisor: %s health: %s", aid, h)
            self._health_counter = 0

        if not agent_proposals:
            logger.debug("AgentSupervisor: no agent proposals, falling back")
            return super().decide(context)

        # Consensus: collect, weight, diversify
        combined = self._consensus.collect(agent_proposals, context)
        if not combined:
            return super().decide(context)

        # AgentProposal already satisfies the shared ActionProposal contract used
        # by Hyperon and the critic, so reuse the same objects here.
        context.hyperon.update_proposals(combined)
        rescored = self._executive.critic.rescore(context, combined)

        # Score-floor and probe-escape checks from the base executive
        if self._executive._all_legal_proposals_are_score_floor(context, rescored):
            return self._executive._score_floor_fallback(context, rescored)
        probe_escape = self._executive._probe_dead_end_escape(context, rescored)
        if probe_escape is not None:
            return probe_escape

        # Pick the best rescored legal proposal directly; AgentProposal retains
        # agent_id so we no longer need a second lookup map.
        for rp in rescored:
            if getattr(rp, "legal", True):
                agent_id = getattr(rp, "agent_id", "")
                metadata = self._executive._finalize_decision_metadata(
                    context,
                    rp.action_name,
                    dict(rp.metadata),
                )
                subgoal = self._executive._finalize_decision_subgoal(
                    context,
                    rp.action_name,
                    rp.subgoal,
                )
                decision = ActionDecision(
                    action_name=rp.action_name,
                    source=rp.source,
                    confidence=rp.confidence,
                    reason=rp.reason,
                    subgoal=subgoal,
                    metadata=metadata,
                    proposals=rescored,
                )
                context.memory.note_decision(decision)
                self._finalize_cognition(
                    context,
                    decision,
                    winning_agent_id=agent_id,
                    participating_agent_ids=participating_agents,
                )
                return decision

        return super().decide(context)

    def _decide_from_agents(self, context: ExecutiveContext) -> ActionDecision | None:
        active_agents = [a for a in self._agents if not self._is_circuit_open(a.agent_id)]
        if len(active_agents) < len(self._agents):
            logger.debug("AgentSupervisor: %d agents circuit-broken", len(self._agents) - len(active_agents))
        if not active_agents:
            return None

        prepare_frame_cache(context)
        agent_proposals = self._run_agents(active_agents, context)
        participating_agents = set(agent_proposals)

        self._health_counter = getattr(self, '_health_counter', 0) + 1
        if self._health_counter >= 100:
            active = ThreadPoolManager.active_worker_count()
            if active > self._workers * 2:
                logger.warning("AgentSupervisor: %d active pool workers (expected <=%d)", active, self._workers)
            for aid, h in self._agent_health.items():
                if h.get("consecutive_failures", 0) >= 2:
                    logger.debug("AgentSupervisor: %s health: %s", aid, h)
            self._health_counter = 0

        if not agent_proposals:
            logger.debug("AgentSupervisor: no agent proposals, falling back")
            return None

        combined = self._consensus.collect(agent_proposals, context)
        if not combined:
            return None

        context.hyperon.update_proposals(combined)
        rescored = self._executive.critic.rescore(context, combined)

        if self._executive._all_legal_proposals_are_score_floor(context, rescored):
            return self._executive._score_floor_fallback(context, rescored)
        probe_escape = self._executive._probe_dead_end_escape(context, rescored)
        if probe_escape is not None:
            return probe_escape

        for rp in rescored:
            if getattr(rp, "legal", True):
                agent_id = getattr(rp, "agent_id", "")
                metadata = self._executive._finalize_decision_metadata(
                    context,
                    rp.action_name,
                    dict(rp.metadata),
                )
                subgoal = self._executive._finalize_decision_subgoal(
                    context,
                    rp.action_name,
                    rp.subgoal,
                )
                decision = ActionDecision(
                    action_name=rp.action_name,
                    source=rp.source,
                    confidence=rp.confidence,
                    reason=rp.reason,
                    subgoal=subgoal,
                    metadata=metadata,
                    proposals=rescored,
                )
                context.memory.note_decision(decision)
                self._finalize_cognition(
                    context,
                    decision,
                    winning_agent_id=agent_id,
                    participating_agent_ids=participating_agents,
                )
                return decision
        return None

    def _try_solver_direct(self, context: ExecutiveContext) -> ActionDecision | None:
        """Try the solver system and return its decision directly if it activates.
        
        This mirrors ObjectiveReduction._try_solver_system() — when a solver
        activates, its decision is used immediately without agent consensus.
        
        Tries all registered solvers in priority order (mirror first, then the
        SolverSelector's choice, then all others).
        """
        try:
            board_data = _get_board_data(context.observation)
            if board_data is None:
                return None
            game_model = build_game_model(board_data)
            if game_model is None:
                return None
            ensure_shape_recognition(
                getattr(context.observation, "image_input", None),
                getattr(context.observation, "derived_features", {}),
            )

            # Priority 1: MirrorSolver if mirror cells are present (reuse instance for state)
            if _board_has_mirror(board_data):
                from agi_core.knowledge.solvers.mirror_solver import MirrorSolver
                if not hasattr(self, '_mirror_solver') or self._mirror_solver is None:
                    self._mirror_solver = MirrorSolver(
                        memory_db_path=_gameplay_memory_db_path(context.config)
                    )
                solver = self._mirror_solver
                sctx = _build_solver_context(solver, game_model, context, self)
                if solver.should_activate(sctx):
                    decision = solver.decide(sctx)
                    if decision is not None:
                        self._seed_sticky_shape_target(context, decision)
                        if self._update_solver_stall(game_model.player_pos, decision.action_name):
                            logger.debug("AgentSupervisor: solver %s stalled (%d), cooling down for 10 frames",
                                        solver.dynamics, self._solver_stall_count)
                            self._solver_stall_count = 0
                            self._solver_cooldown = 10
                            return None  # fall through to agents
                        logger.debug("AgentSupervisor: solver mirror -> %s", decision.action_name)
                        context.memory.note_decision(decision)
                        self._finalize_cognition(context, decision)
                        return decision

            # Priority 2: SolverSelector (from hyperon / dynamics)
            from agi_core.knowledge.solver_selector import SolverSelector
            if self._solver_selector is None:
                self._solver_selector = SolverSelector(
                    context.hyperon,
                    memory_db_path=_gameplay_memory_db_path(context.config),
                )
            solver = self._solver_selector.select_solver(game_model, context)
            if solver is not None:
                sctx = _build_solver_context(solver, game_model, context, self)
                if solver.should_activate(sctx):
                    decision = solver.decide(sctx)
                    if decision is not None:
                        self._seed_sticky_shape_target(context, decision)
                        logger.debug("AgentSupervisor: solver %s -> %s", solver.dynamics, decision.action_name)
                        context.memory.note_decision(decision)
                        self._finalize_cognition(context, decision)
                        return decision
        except Exception:
            logger.debug("AgentSupervisor: solver system failed", exc_info=True)
        # No solver activated — reset solver stall counter
        self._solver_stall_count = 0
        return None

    def _run_agents(
        self,
        agents: list[StrategyAgent],
        context: ExecutiveContext,
    ) -> dict[str, list[AgentProposal]]:
        """Execute all agents in parallel, then sync-fallback for timed-out ones.
        
        Returns a dict mapping agent_id → list[AgentProposal].
        """
        proposals: dict[str, list[AgentProposal]] = {}
        tasks = [(lambda a=a: a.produce(context), a.agent_id) for a in agents]
        t0 = time.monotonic()
        results = ThreadPoolManager.run_all(tasks, timeout=self._agent_timeout)
        elapsed = time.monotonic() - t0
        self._record_agent_metric("supervisor", "exec_time_ms", elapsed * 1000)
        if elapsed > 0.5:
            logger.debug("AgentSupervisor: agents took %.2fs", elapsed)
        for result, agent_id in results:
            if result:
                proposals[agent_id] = result
                self._record_success(agent_id, result)
            else:
                self._record_failure(agent_id)
        # Sync fallback for timed-out agents
        missing = [a for a in agents if a.agent_id not in proposals]
        for agent in missing:
            try:
                result = agent.produce(context)
                if result:
                    proposals[agent.agent_id] = result
                    self._record_success(agent.agent_id, result)
                else:
                    self._record_failure(agent.agent_id)
            except Exception:
                logger.debug("AgentSupervisor: agent %s sync fallback failed", agent.agent_id)
                self._record_failure(agent.agent_id)
        return proposals

    def _record_success(self, agent_id: str, result: list[AgentProposal]) -> None:
        """Record a successful agent execution."""
        self._reset_circuit_breaker(agent_id)
        self._record_agent_metric(agent_id, "success", 1.0)
        self._record_agent_metric(agent_id, "proposal_count", len(result))

    def _record_failure(self, agent_id: str) -> None:
        """Record a failed agent execution."""
        self._record_agent_failure(agent_id)
        self._record_agent_metric(agent_id, "success", 0.0)
        self._record_agent_metric(agent_id, "proposal_count", 0.0)

    def _is_circuit_open(self, agent_id: str) -> bool:
        """Check if an agent should be skipped due to repeated failures.
        
        Opens after 3 consecutive failures.
        Auto-closes after a successful agent execution resets it.
        """
        health = self._agent_health.get(agent_id)
        if health is None:
            return False
        return health.get("consecutive_failures", 0) >= 3

    def _record_agent_metric(self, agent_id: str, key: str, value: float) -> None:
        """Record a numeric metric for an agent (decayed moving average)."""
        m = self._agent_metrics.setdefault(agent_id, {})
        old = m.get(key, 0.0)
        m[key] = old * 0.9 + value * 0.1  # exponential moving average

    def _record_agent_failure(self, agent_id: str) -> None:
        """Record an agent failure for circuit breaker tracking."""
        health = self._agent_health.setdefault(agent_id, {"consecutive_failures": 0})
        health["consecutive_failures"] = health.get("consecutive_failures", 0) + 1

    def _reset_circuit_breaker(self, agent_id: str) -> None:
        """Reset circuit breaker on successful agent execution."""
        self._agent_health.pop(agent_id, None)

    def reset_agents(self) -> None:
        """Reset all agents, consensus stats, circuit breakers, and metrics for a new game."""
        for agent in self._agents:
            agent.reset()
        self._consensus._agent_stats.clear()
        self._agent_health.clear()
        self._agent_metrics.clear()
        self._mirror_solver = None
        self._solver_selector = None
        self._solver_stall_count = 0
        self._solver_cooldown = 0
        self._last_solver_action = ""
        self._last_solver_player_pos = None
        self._solver_recent_states = []
        self._sticky_shape_target = None

    # ── internal ──────────────────────────────────────────────────────

    def _finalize_cognition(
        self,
        context: ExecutiveContext,
        decision: ActionDecision,
        winning_agent_id: str = "",
        participating_agent_ids: set[str] | None = None,
    ) -> None:
        """Run cognition finalization for all agents.
        
        The winning agent gets the actual outcome; all other agents that
        submitted proposals get a failed outcome (their proposal wasn't chosen).
        """
        success = "fallback" not in decision.source
        participants = set(participating_agent_ids or ())
        for agent_id in participants:
            agent = self._agent_map.get(agent_id)
            if agent is None:
                continue
            if agent_id == winning_agent_id:
                outcome = "progress" if success else "stalled"
                agent.apply_cognition(decision.action_name, outcome, decision.subgoal)
                agent.note_outcome(success)
                self._consensus.note_outcome(agent_id, success, None)
            else:
                agent.note_outcome(False)
                self._consensus.note_outcome(agent_id, False, None)
        self._executive._finalize_cognition(context, decision)

    def _update_solver_stall(
        self,
        player_pos: tuple[int, int] | None,
        action_name: str,
    ) -> bool:
        """Return True when a solver is repeating without moving."""
        if (
            action_name == self._last_solver_action
            and player_pos == self._last_solver_player_pos
        ):
            self._solver_stall_count += 1
        else:
            self._solver_stall_count = 0
        self._last_solver_action = action_name
        self._last_solver_player_pos = player_pos
        self._solver_recent_states.append((action_name, player_pos))
        if len(self._solver_recent_states) > 6:
            self._solver_recent_states = self._solver_recent_states[-6:]
        self._solver_cooldown = 0
        if self._is_solver_oscillating():
            self._solver_stall_count = max(self._solver_stall_count, 10)
            return True
        return self._solver_stall_count >= 10

    def _is_solver_oscillating(self) -> bool:
        recent = self._solver_recent_states
        if len(recent) < 6:
            return False
        actions = [action for action, _ in recent[-6:]]
        positions = [pos for _, pos in recent[-6:]]
        if any(pos is None for pos in positions):
            return False
        first = actions[0]
        second = actions[1]
        if first == second:
            return False
        if actions != [first, second, first, second, first, second]:
            return False
        unique_positions = {pos for pos in positions if pos is not None}
        return len(unique_positions) <= 2

    def _stabilize_shape_frame_cache(
        self,
        context: ExecutiveContext,
        frame_cache: dict[str, Any],
    ) -> dict[str, Any]:
        levels_completed = int(getattr(context.observation, "levels_completed", 0) or 0)
        mirror_active = bool(frame_cache.get("mirror_puzzle_active"))
        active_target = frame_cache.get("active_shape_target")
        active_stage = int(frame_cache.get("active_shape_stage", 0) or 0)
        player_position = (
            getattr(context.observation, "derived_features", {}) or {}
        ).get("player_position")
        if mirror_active and active_target:
            sticky = self._sticky_shape_target
            if (
                sticky is not None
                and sticky.get("levels_completed") == levels_completed
                and sticky.get("orientation") == frame_cache.get("mirror_orientation")
                and int(sticky.get("stage", 0) or 0) == active_stage
                and isinstance(sticky.get("target"), tuple)
            ):
                sticky_target = sticky["target"]
                target_shift = (
                    abs(int(active_target[0]) - int(sticky_target[0]))
                    + abs(int(active_target[1]) - int(sticky_target[1]))
                )
                sticky_distance = None
                if isinstance(player_position, tuple) and len(player_position) == 2:
                    sticky_distance = (
                        abs(int(sticky_target[0]) - int(player_position[0]))
                        + abs(int(sticky_target[1]) - int(player_position[1]))
                    )
                if target_shift >= 6 and (sticky_distance is None or sticky_distance > 2):
                    frame_cache["active_shape_target"] = sticky_target
                    active_target = sticky_target
            self._sticky_shape_target = {
                "levels_completed": levels_completed,
                "target": active_target,
                "stage": active_stage,
                "orientation": frame_cache.get("mirror_orientation"),
            }
            return frame_cache
        sticky = self._sticky_shape_target
        if (
            mirror_active
            and not active_target
            and sticky is not None
            and sticky.get("levels_completed") == levels_completed
            and sticky.get("orientation") == frame_cache.get("mirror_orientation")
        ):
            frame_cache["active_shape_target"] = sticky["target"]
            frame_cache["active_shape_stage"] = sticky["stage"]
            frame_cache["shape_agent_mode"] = levels_completed >= 2
            return frame_cache
        if not mirror_active:
            self._sticky_shape_target = None
        return frame_cache

    def _seed_sticky_shape_target(
        self,
        context: ExecutiveContext,
        decision: ActionDecision,
    ) -> None:
        metadata = getattr(decision, "metadata", {}) or {}
        target = metadata.get("target_anchor")
        if not isinstance(target, list) or len(target) != 2:
            return
        frame_cache = getattr(context, "multi_agent_frame_cache", {}) or {}
        shape_recognition = frame_cache.get("shape_recognition")
        if not isinstance(shape_recognition, dict):
            shape_recognition = ensure_shape_recognition(
                getattr(context.observation, "image_input", None),
                getattr(context.observation, "derived_features", {}),
            )
        mirror_puzzle = shape_recognition.get("mirror_puzzle", {}) if isinstance(shape_recognition, dict) else {}
        if not isinstance(mirror_puzzle, dict) or not mirror_puzzle.get("active"):
            return
        self._sticky_shape_target = {
            "levels_completed": int(getattr(context.observation, "levels_completed", 0) or 0),
            "target": (int(target[0]), int(target[1])),
            "stage": int(metadata.get("shape_stage", 0) or 0),
            "orientation": mirror_puzzle.get("orientation"),
        }


# ── Module-level helpers for solver integration ────────────────────────


def _board_has_mirror(board: list[list[int]]) -> bool:
    """Quick check for mirror cells (color 10) on the board."""
    return any(val == 10 for row in board for val in row)


def _get_board_data(observation: Any) -> list[list[int]] | None:
    """Extract the 64×64 int8 board from an observation."""
    image_input = getattr(observation, 'image_input', None)
    if image_input is None:
        return None
    return getattr(image_input, 'board', None)


def _build_solver_context(solver: Any, game_model: Any, context: ExecutiveContext,
                          supervisor: AgentSupervisor) -> Any:
    """Build a SolverContext for the given solver."""
    shape_recognition = ensure_shape_recognition(
        getattr(context.observation, "image_input", None),
        getattr(context.observation, "derived_features", {}),
    )
    mirror_puzzle = (
        shape_recognition.get("mirror_puzzle", {})
        if isinstance(shape_recognition, dict)
        else {}
    )
    pctx: dict[str, Any] = {
        "non_beam_controllers": getattr(supervisor, '_non_beam_controllers', set()),
        "failed_directions": getattr(supervisor, '_failed_directions', set()),
        "stable_goal": getattr(supervisor, '_stable_goal', None),
        "switch_attempted": getattr(supervisor, '_switch_count', 0),
        "levels_completed": getattr(context.observation, "levels_completed", 0),
        "action_priors": {},
        "memory_db_path": _gameplay_memory_db_path(context.config),
        "beam_cells": [],
        "mirror_axis": game_model.features.mirror_axis if hasattr(game_model, 'features') else -1,
        "goal_history": list(getattr(supervisor, '_goal_history', [])),
        "abstract_memories": list(context.memory_bundle.abstract_memories) if context.memory_bundle and context.memory_bundle.abstract_memories else [],
        "shape_recognition": shape_recognition,
        "shape_stage_targets": list(mirror_puzzle.get("stage_targets", [])) if isinstance(mirror_puzzle, dict) else [],
    }
    if supervisor._solver_selector is not None:
        return supervisor._solver_selector.build_solver_context(game_model, pctx, context)
    from agi_core.knowledge.solver_selector import SolverSelector
    sel = SolverSelector(
        context.hyperon,
        memory_db_path=_gameplay_memory_db_path(context.config),
    )
    return sel.build_solver_context(game_model, pctx, context)
