from __future__ import annotations

import logging
import time
from concurrent.futures import Future
from typing import Any

from agi_core.multi_agent.agents import (
    ExplorationAgent,
    InteractionAgent,
    MovementAgent,
    PuzzleAgent,
    build_frame_snapshot,
    prepare_frame_cache,
)
from agi_core.multi_agent.consensus import ConsensusModule
from agi_core.multi_agent.runtime import (
    AgentOutcomeUpdate,
    ProcessMultiAgentExecutor,
    build_multi_agent_executor,
)
from agi_core.multi_agent.strategy_agent import StrategyAgent
from agi_core.perception.shape_recognition import ensure_shape_recognition
from agi_core.runtime_sync import ensure_hyperon_context_synced
from agi_core.cuda import is_enabled as _cuda_is_enabled, init_cuda
from agi_core.orch_or.analyzer import build_game_model
from agi_core.orch_or.orchestrator import ObjectiveReduction
from agi_core.orch_or.parallel import ThreadPoolManager
from agi_core.types import (
    ActionDecision,
    AgentProposal,
    ExecutiveContext,
    ObservedTransition,
)

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
        try:
            self._workers = int(getattr(config, 'multi_agent_workers', 4) or 4)
        except Exception:
            self._workers = 4
        self._agent_timeout = max(0.5, float(getattr(config, 'planner_parallel_timeout', None) or 2.0))
        self._executor = build_multi_agent_executor(config, self._agents)
        self._process_executor = isinstance(self._executor, ProcessMultiAgentExecutor)
        self._agent_health: dict[str, dict[str, float]] = {}
        self._agent_metrics: dict[str, dict[str, float]] = {}
        self._agent_futures: dict[str, tuple[Future, int]] = {}
        self._circuit_cooldown_seconds = max(1.0, self._agent_timeout * 2.0)
        self._pending_agent_credit: dict[str, Any] | None = None
        self._solver_candidate_feedback: tuple[Any, Any, Any, ActionDecision] | None = None
        self._pending_solver_feedback: tuple[Any, Any, Any, ActionDecision] | None = None
        self._solver_attempt_steps = 0
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
        stale_count = self._executor.active_worker_count()
        if self._workers > 0 and stale_count >= self._workers:
            logger.warning(
                "AgentSupervisor: pool saturated by %d queued/running tasks (budget=%d)",
                stale_count,
                self._workers,
            )

        # Let the base Executive handle early exits first
        context.memory_bundle = ensure_hyperon_context_synced(context)

        # Check score-floor reopen before running agents
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

        self._stabilize_shape_frame_cache(context, prepare_frame_cache(context))

        # Fan out the strategy agents before checking the direct solver.  The
        # solver remains authoritative, but it must not bypass the multi-agent
        # pool on every solver-active frame.
        active_agents = [a for a in self._agents if not self._is_circuit_open(a.agent_id)]
        if len(active_agents) < len(self._agents):
            logger.debug("AgentSupervisor: %d agents circuit-broken", len(self._agents) - len(active_agents))
        agent_proposals = self._run_agents(active_agents, context)
        participating_agents = set(agent_proposals)

        # Try solver system first (respect cooldown after stagnation)
        solver_decision = None
        self._solver_candidate_feedback = None
        if self._solver_cooldown <= 0:
            solver_decision = self._try_solver_direct(context)
        else:
            self._solver_cooldown -= 1
        if solver_decision is not None:
            solver_proposal = self._solver_decision_as_proposal(solver_decision)
            if self._proposal_is_legal(context, solver_proposal):
                if str(solver_decision.source).startswith("solver:"):
                    self._seed_sticky_shape_target(context, solver_decision)
                context.memory.note_decision(solver_decision)
                self._finalize_cognition(context, solver_decision)
                return solver_decision
            agent_proposals = dict(agent_proposals)
            agent_proposals.setdefault("solver", []).append(solver_proposal)
            participating_agents.add("solver")

        selected = self._decide_from_agents(
            context,
            agent_proposals=agent_proposals,
            participating_agent_ids=participating_agents,
        )
        if selected is not None:
            return selected

        logger.debug("AgentSupervisor: no legal consensus decision, falling back")
        return super().decide(context)

    def _decide_from_agents(
        self,
        context: ExecutiveContext,
        *,
        agent_proposals: dict[str, list[AgentProposal]] | None = None,
        participating_agent_ids: set[str] | None = None,
    ) -> ActionDecision | None:
        if agent_proposals is None:
            active_agents = [a for a in self._agents if not self._is_circuit_open(a.agent_id)]
            if len(active_agents) < len(self._agents):
                logger.debug("AgentSupervisor: %d agents circuit-broken", len(self._agents) - len(active_agents))
            if not active_agents:
                return None

            prepare_frame_cache(context)
            agent_proposals = self._run_agents(active_agents, context)
        participating_agents = (
            set(participating_agent_ids)
            if participating_agent_ids is not None
            else set(agent_proposals).intersection(self._agent_map)
        )

        self._health_counter = getattr(self, '_health_counter', 0) + 1
        if self._health_counter >= 100:
            active = self._executor.active_worker_count()
            if self._workers > 0 and active >= self._workers:
                logger.warning("AgentSupervisor: %d active pool tasks (budget=%d)", active, self._workers)
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
            return self._finalize_fallback_decision(
                context,
                self._executive._score_floor_fallback(context, rescored),
                rescored,
            )
        probe_escape = self._executive._probe_dead_end_escape(context, rescored)
        if probe_escape is not None:
            return self._finalize_fallback_decision(context, probe_escape, rescored)

        for rp in rescored:
            if self._proposal_is_legal(context, rp):
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
                if str(decision.source).startswith("solver:"):
                    self._seed_sticky_shape_target(context, decision)
                context.memory.note_decision(decision)
                self._finalize_cognition(
                    context,
                    decision,
                    winning_agent_id=agent_id,
                    participating_agent_ids=participating_agents,
                )
                return decision
        return None

    @staticmethod
    def _solver_decision_as_proposal(decision: ActionDecision) -> AgentProposal:
        """Expose a solver decision to the same consensus/critic path as agents."""
        return AgentProposal(
            action_name=decision.action_name,
            source=decision.source,
            confidence=decision.confidence,
            reason=decision.reason,
            agent_id="solver",
            agent_confidence=1.0,
            subgoal=decision.subgoal,
            metadata=dict(decision.metadata),
        )

    @staticmethod
    def _proposal_is_legal(
        context: ExecutiveContext,
        proposal: AgentProposal,
    ) -> bool:
        """Fail closed even if a critic or solver incorrectly marks a proposal legal."""
        action_name = str(proposal.action_name or "")
        legal_actions = {
            str(name or "") for name in context.observation.legal_actions
        }
        if action_name in legal_actions:
            return bool(getattr(proposal, "legal", True))
        return bool(
            action_name == "RESET"
            and str(context.observation.game_state or "")
            in set(context.config.reset_state_names)
        )

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
            # Sync hyperon episode facts for solver selection
            try:
                ep = context.hyperon.facts.setdefault("episode", {})
                ep["board_type"] = game_model.features.board_type
                ep["primary_dynamics"] = game_model.features.primary_dynamics
                ep["dynamics"] = list(game_model.features.dynamics)
            except Exception:
                pass
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
                        if self._update_solver_stall(game_model.player_pos, decision.action_name):
                            logger.debug("AgentSupervisor: solver %s stalled (%d), cooling down for 10 frames",
                                        solver.dynamics, self._solver_stall_count)
                            self._solver_stall_count = 0
                            self._solver_cooldown = 10
                            self._close_solver_attempt(context, solver, succeeded=False)
                            return None  # fall through to agents
                        logger.debug("AgentSupervisor: solver mirror -> %s", decision.action_name)
                        self._solver_candidate_feedback = (
                            solver,
                            sctx,
                            game_model,
                            decision,
                        )
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
                        if self._update_solver_stall(game_model.player_pos, decision.action_name):
                            logger.debug(
                                "AgentSupervisor: solver %s stalled (%d), cooling down for 10 frames",
                                solver.dynamics,
                                self._solver_stall_count,
                            )
                            self._solver_stall_count = 0
                            self._solver_cooldown = 10
                            self._close_solver_attempt(context, solver, succeeded=False)
                            return None
                        logger.debug("AgentSupervisor: solver %s -> %s", solver.dynamics, decision.action_name)
                        self._solver_candidate_feedback = (
                            solver,
                            sctx,
                            game_model,
                            decision,
                        )
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
        """Run one current-frame call per idle agent and discard late results."""
        proposals: dict[str, list[AgentProposal]] = {}
        completed_agents: set[str] = set()
        step_index = int(getattr(context.observation, "step_index", 0) or 0)
        snapshot = build_frame_snapshot(context)

        # Reap work from older frames. Its execution health is useful, but its
        # proposals are stale and must never be applied to the current frame.
        for agent in agents:
            pending = self._agent_futures.get(agent.agent_id)
            if pending is None:
                continue
            future, submitted_step = pending
            if not future.done():
                continue
            self._agent_futures.pop(agent.agent_id, None)
            try:
                late_result = future.result()
            except Exception:
                logger.debug(
                    "AgentSupervisor: late agent %s failed",
                    agent.agent_id,
                    exc_info=True,
                )
                self._record_failure(agent.agent_id)
            else:
                if late_result is None:
                    self._record_failure(agent.agent_id)
                else:
                    self._record_success(agent.agent_id, late_result)
                logger.debug(
                    "AgentSupervisor: discarded stale %s result from step %d",
                    agent.agent_id,
                    submitted_step,
                )

        current: dict[str, Future] = {}
        for agent in agents:
            if agent.agent_id in self._agent_futures:
                continue
            future = self._executor.submit(
                agent.agent_id,
                agent,
                snapshot,
            )
            if future is None:
                continue
            self._agent_futures[agent.agent_id] = (future, step_index)
            current[agent.agent_id] = future

        t0 = time.monotonic()
        done: set[Future] = set()
        if current:
            deadline = time.monotonic() + self._agent_timeout
            while True:
                done = {
                    future
                    for future in current.values()
                    if future.done()
                }
                if len(done) >= len(current):
                    break
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.001)
        elapsed = time.monotonic() - t0
        self._record_agent_metric("supervisor", "exec_time_ms", elapsed * 1000)

        for agent in agents:
            agent_id = agent.agent_id
            future = current.get(agent_id)
            if future is None or future not in done:
                continue
            completed_agents.add(agent_id)
            pending = self._agent_futures.get(agent_id)
            if pending is not None and pending[0] is future:
                self._agent_futures.pop(agent_id, None)
            try:
                result = future.result()
            except Exception:
                logger.debug(
                    "AgentSupervisor: agent %s failed",
                    agent_id,
                    exc_info=True,
                )
                self._record_failure(agent_id)
                continue
            if result is None:
                self._record_failure(agent_id)
                continue
            if result:
                proposals[agent_id] = result
            self._record_success(agent_id, result)
        logger.debug(
            "AgentSupervisor: parallel agent batch finished %d/%d in %.2fms (workers=%d)",
            len(completed_agents),
            len(agents),
            elapsed * 1000,
            self._workers,
        )
        # Running calls stay registered and are not resubmitted on later frames.
        # A timeout is scheduling pressure, not proof that an agent is broken.
        running = [
            agent_id
            for agent_id, future in current.items()
            if future not in done
        ]
        if running:
            logger.debug(
                "AgentSupervisor: %d agents still running after %.2fms: %s",
                len(running),
                elapsed * 1000,
                ", ".join(running),
            )
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
        if health.get("consecutive_failures", 0) < 3:
            return False
        open_until = float(health.get("open_until", 0.0) or 0.0)
        if time.monotonic() < open_until:
            return True
        # Half-open: allow one deduplicated probe. Success clears the breaker;
        # another failure reopens it for a fresh cooldown.
        health["half_open"] = 1.0
        return False

    def _record_agent_metric(self, agent_id: str, key: str, value: float) -> None:
        """Record a numeric metric for an agent (decayed moving average)."""
        m = self._agent_metrics.setdefault(agent_id, {})
        old = m.get(key, 0.0)
        m[key] = old * 0.9 + value * 0.1  # exponential moving average

    def _record_agent_failure(self, agent_id: str) -> None:
        """Record an agent failure for circuit breaker tracking."""
        health = self._agent_health.setdefault(agent_id, {"consecutive_failures": 0})
        health["consecutive_failures"] = health.get("consecutive_failures", 0) + 1
        if health["consecutive_failures"] >= 3:
            health["open_until"] = time.monotonic() + self._circuit_cooldown_seconds
            health["half_open"] = 0.0

    def _reset_circuit_breaker(self, agent_id: str) -> None:
        """Reset circuit breaker on successful agent execution."""
        self._agent_health.pop(agent_id, None)

    def reset_agents(self) -> None:
        """Reset all agents, consensus stats, circuit breakers, and metrics for a new game."""
        self._executor.reset()
        if not self._process_executor:
            for agent in self._agents:
                agent.reset()
        self._consensus._agent_stats.clear()
        self._agent_health.clear()
        self._agent_metrics.clear()
        self._agent_futures.clear()
        self._mirror_solver = None
        self._solver_selector = None
        self._solver_stall_count = 0
        self._solver_cooldown = 0
        self._last_solver_action = ""
        self._last_solver_player_pos = None
        self._solver_recent_states = []
        self._sticky_shape_target = None
        self._pending_agent_credit = None
        self._solver_candidate_feedback = None
        self._pending_solver_feedback = None
        self._solver_attempt_steps = 0
        super().reset_episode()

    # ── internal ──────────────────────────────────────────────────────

    def _finalize_cognition(
        self,
        context: ExecutiveContext,
        decision: ActionDecision,
        winning_agent_id: str = "",
        participating_agent_ids: set[str] | None = None,
    ) -> None:
        """Record attribution now; apply outcome feedback on the next frame."""
        if winning_agent_id and winning_agent_id in self._agent_map:
            self._pending_agent_credit = {
                "game_id": str(context.observation.game_id or ""),
                "agent_id": winning_agent_id,
                "action_name": decision.action_name,
                "source": decision.source,
                "subgoal": decision.subgoal,
            }
        else:
            self._pending_agent_credit = None
        if (
            str(decision.source or "").startswith("solver:")
            and self._solver_candidate_feedback is not None
        ):
            solver, solver_context, game_model, solver_decision = (
                self._solver_candidate_feedback
            )
            if (
                self._solver_selector is not None
                and self._solver_selector.active_solver is solver
            ):
                self._solver_selector.record_outcome(
                    context,
                    game_model,
                    solver_decision,
                    dict(getattr(solver_context, "plan_context", {}) or {}),
                )
                self._pending_solver_feedback = None
            else:
                self._pending_solver_feedback = self._solver_candidate_feedback
        else:
            self._pending_solver_feedback = None
        self._solver_candidate_feedback = None
        self._executive._finalize_cognition(context, decision)

    def note_transition(self, transition: ObservedTransition) -> None:
        """Credit only the selected contributor after observing the result."""
        pending = self._pending_agent_credit
        if pending is not None:
            matches = (
                (not transition.action_name or transition.action_name == pending["action_name"])
                and (
                    not transition.decision_source
                    or transition.decision_source == pending["source"]
                )
            )
            outcome = str(transition.local_outcome or "unknown")
            if matches and outcome in {"progress", "advanced", "failed", "stalled"}:
                agent_id = str(pending["agent_id"])
                success = outcome in {"progress", "advanced"}
                agent = self._agent_map.get(agent_id)
                if agent is not None and not self._process_executor:
                    agent.apply_cognition(
                        str(pending["action_name"]),
                        "progress" if success else outcome,
                        pending.get("subgoal"),
                    )
                    agent.note_outcome(success)
                if agent is not None and self._process_executor:
                    self._executor.apply_outcome(
                        agent_id,
                        AgentOutcomeUpdate(
                            action_name=str(pending["action_name"]),
                            outcome="progress" if success else outcome,
                            subgoal=pending.get("subgoal"),
                            success=success,
                        ),
                    )
                if agent is not None:
                    self._consensus.note_outcome(agent_id, success, None)
                self._pending_agent_credit = None
            elif transition.boundary is not None:
                self._pending_agent_credit = None

        if self._solver_selector is not None:
            self._solver_selector.note_transition(transition)
        self._note_solver_transition(transition)
        if hasattr(self._executive, "note_transition"):
            self._executive.note_transition(transition)
        if transition.boundary == "level":
            self._executor.reset()
            if not self._process_executor:
                for agent in self._agents:
                    agent.reset()
            self.reset_level_state()
            self._mirror_solver = None
            self._solver_stall_count = 0
            self._solver_cooldown = 0
            self._sticky_shape_target = None
        elif transition.boundary in {"win", "game_over", "restart", "game"}:
            self.reset_agents()

    def _note_solver_transition(self, transition: ObservedTransition) -> None:
        pending = self._pending_solver_feedback
        if pending is None:
            return
        solver, solver_context, game_model, decision = pending
        matches = (
            (not transition.action_name or transition.action_name == decision.action_name)
            and (
                not transition.decision_source
                or transition.decision_source == decision.source
            )
        )
        outcome = str(transition.local_outcome or "unknown")
        if matches and outcome != "unknown":
            try:
                solver.learn(solver_context, outcome)
            except Exception:
                logger.debug("AgentSupervisor: solver learning failed", exc_info=True)
            self._solver_attempt_steps += 1
            try:
                from agi_core.knowledge.puzzle_memory import (
                    PuzzleBoardState,
                    initialize_puzzle_schema,
                    store_board_state,
                )

                db_path = getattr(solver, "_memory_db_path", None)
                if db_path is not None:
                    initialize_puzzle_schema(db_path)
                    player = game_model.player_pos
                    store_board_state(
                        db_path,
                        PuzzleBoardState(
                            game_id=transition.game_id,
                            step=(
                                int(transition.record.step_index)
                                if transition.record is not None
                                else 0
                            ),
                            frame_signature=(
                                str(transition.record.prior_frame_signature or "")
                                if transition.record is not None
                                else ""
                            ),
                            player_x=player[0] if player else -1,
                            player_y=player[1] if player else -1,
                            goal_positions=list(game_model.goal_positions),
                            action_taken=decision.action_name,
                            outcome=outcome,
                            progress_delta=(
                                int(transition.record.progress_delta or 0)
                                if transition.record is not None
                                else 0
                            ),
                        ),
                    )
            except Exception:
                logger.debug("AgentSupervisor: solver board-state persistence failed", exc_info=True)

        if transition.boundary in {"win", "level"}:
            self._persist_solver_attempt(
                transition.game_id,
                solver,
                succeeded=True,
            )
        elif transition.boundary in {"game_over", "restart", "game"}:
            self._persist_solver_attempt(
                transition.game_id,
                solver,
                succeeded=False,
            )
        if matches or transition.boundary is not None:
            self._pending_solver_feedback = None

    def _close_solver_attempt(
        self,
        context: ExecutiveContext,
        solver: Any,
        *,
        succeeded: bool,
    ) -> None:
        if (
            self._solver_selector is not None
            and self._solver_selector.active_solver is solver
        ):
            self._solver_selector.on_boundary(
                "level" if succeeded else "stalled",
                game_id=str(context.observation.game_id or ""),
            )
            self._pending_solver_feedback = None
            return
        self._persist_solver_attempt(
            str(context.observation.game_id or ""),
            solver,
            succeeded=succeeded,
        )
        self._pending_solver_feedback = None

    def _persist_solver_attempt(
        self,
        game_id: str,
        solver: Any,
        *,
        succeeded: bool,
    ) -> None:
        db_path = getattr(solver, "_memory_db_path", None)
        if db_path is None:
            self._solver_attempt_steps = 0
            return
        try:
            from agi_core.knowledge.puzzle_memory import (
                initialize_puzzle_schema,
                update_solver_outcome,
            )

            initialize_puzzle_schema(db_path)
            update_solver_outcome(
                db_path,
                game_id=game_id,
                dynamics=str(getattr(solver, "dynamics", "unknown") or "unknown"),
                solver_name=str(getattr(solver, "dynamics", "unknown") or "unknown"),
                succeeded=succeeded,
                steps_taken=max(1, self._solver_attempt_steps),
            )
        except Exception:
            logger.debug("AgentSupervisor: solver outcome persistence failed", exc_info=True)
        finally:
            self._solver_attempt_steps = 0

    def _finalize_fallback_decision(
        self,
        context: ExecutiveContext,
        decision: ActionDecision,
        rescored: list[AgentProposal],
    ) -> ActionDecision:
        decision.proposals = rescored
        context.memory.note_decision(decision)
        self._finalize_cognition(context, decision)
        return decision

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
