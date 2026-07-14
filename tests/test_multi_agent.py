from __future__ import annotations

import unittest
import time
from unittest.mock import MagicMock, patch

from agi_core.config import RuntimeConfig
from agi_core.multi_agent import (
    AgentSupervisor,
    ConsensusModule,
    ExplorationAgent,
    InteractionAgent,
    MovementAgent,
    PuzzleAgent,
)
from agi_core.multi_agent.agents import StrategyAgent, _ordered_mirror_actions, _target_metadata, prepare_frame_cache
from agi_core.multi_agent.consensus import _consensus_context_signals
from agi_core.memory import EpisodicMemory
from agi_core.orch_or.parallel import ProcessPoolManager, ThreadPoolManager
from agi_core.perception.shape_recognition import recognize_shape_recognition
from agi_core.runtime_sync import ensure_hyperon_context_synced
from agi_core.types import (
    ActionDecision,
    AgentProposal,
    ExecutiveContext,
    Observation,
    ObservedTransition,
)


class FastProcessTestAgent(StrategyAgent):
    def __init__(self) -> None:
        super().__init__(agent_id="fast")

    def produce(self, context) -> list[AgentProposal]:
        return [
            self._make_proposal(
                action_name="ACTION1",
                confidence=0.9,
                reason="fast",
                subgoal="navigate",
            )
        ]


class SlowProcessTestAgent(StrategyAgent):
    def __init__(self) -> None:
        super().__init__(agent_id="slow")

    def produce(self, context) -> list[AgentProposal]:
        time.sleep(0.3)
        return [
            self._make_proposal(
                action_name="ACTION2",
                confidence=0.6,
                reason="slow",
                subgoal="navigate",
            )
        ]


class OutcomeAwareProcessAgent(StrategyAgent):
    def __init__(self) -> None:
        super().__init__(agent_id="stateful")

    def produce(self, context) -> list[AgentProposal]:
        action_name = "ACTION2" if self._success_count else "ACTION1"
        return [
            self._make_proposal(
                action_name=action_name,
                confidence=0.75,
                reason=f"state:{self._success_count}",
                subgoal="navigate",
            )
        ]


class FailingProcessAgent(StrategyAgent):
    def __init__(self) -> None:
        super().__init__(agent_id="failing")

    def produce(self, context) -> list[AgentProposal]:
        raise RuntimeError("boom")


class TestMultiAgent(unittest.TestCase):
    def tearDown(self) -> None:
        ThreadPoolManager._shutdown()
        ThreadPoolManager.set_max_workers(0)
        ProcessPoolManager._shutdown()
        ProcessPoolManager.set_max_workers(0)
        return super().tearDown()

    def _require_process_executor(self, supervisor: AgentSupervisor) -> None:
        if not supervisor._process_executor:
            self.skipTest("process executor unavailable on this host")

    def test_movement_agent_produces_proposals(self):
        agent = MovementAgent()
        ctx = self._mock_context(legal=["ACTION1", "ACTION2", "ACTION3", "ACTION4"])
        proposals = agent.produce(ctx)
        self.assertGreater(len(proposals), 0)
        for p in proposals:
            self.assertEqual(p.agent_id, "movement")
            self.assertIn(p.action_name, ["ACTION1", "ACTION2", "ACTION3", "ACTION4"])
            self.assertGreater(p.confidence, 0.0)

    def test_exploration_agent_produces_switch_proposals(self):
        agent = ExplorationAgent()
        ctx = self._mock_context(legal=["ACTION1", "ACTION5"])
        proposals = agent.produce(ctx)
        self.assertGreater(len(proposals), 0)
        self.assertTrue(any(p.action_name == "ACTION5" for p in proposals))

    def test_interaction_agent_produces_click_proposals(self):
        agent = InteractionAgent()
        ctx = self._mock_context(legal=["ACTION6"])
        proposals = agent.produce(ctx)
        self.assertGreater(len(proposals), 0)
        self.assertEqual(proposals[0].action_name, "ACTION6")

    def test_puzzle_agent_produces_proposals(self):
        agent = PuzzleAgent()
        ctx = self._mock_context(legal=["ACTION4", "ACTION5", "ACTION6"])
        proposals = agent.produce(ctx)
        self.assertGreater(len(proposals), 0)
        # Should prefer movement actions for puzzle
        self.assertTrue(any(p.action_name in ("ACTION4", "ACTION5") for p in proposals))

    def test_consensus_collects_and_weights(self):
        consensus = ConsensusModule()
        consensus._agent_stats["movement"] = {"successes": 5.0, "total": 10.0}
        agent = MovementAgent()
        ctx = self._mock_context(legal=["ACTION1", "ACTION2", "ACTION3", "ACTION4"])
        proposals = {"movement": agent.produce(ctx)}
        result = consensus.collect(proposals, ctx)
        self.assertGreater(len(result), 0)
        for p in result:
            self.assertIsInstance(p, AgentProposal)

    def test_consensus_diversity_bonus(self):
        consensus = ConsensusModule()
        consensus._agent_stats["a1"] = {"successes": 5.0, "total": 10.0}
        consensus._agent_stats["a2"] = {"successes": 5.0, "total": 10.0}
        def _make(an, aid):
            return AgentProposal(action_name=an, source=aid, confidence=0.5, reason="r",
                                 agent_id=aid, agent_confidence=0.5, agent_goal_depth=1)
        proposals = {
            "a1": [_make("ACTION1", "a1")],
            "a2": [_make("ACTION1", "a2")],
        }
        ctx = self._mock_context()
        result = consensus.collect(proposals, ctx)
        for p in result:
            self.assertLessEqual(p.confidence, 0.45)  # reduced by redundancy penalty

    def test_consensus_unique_action_boosted(self):
        consensus = ConsensusModule()
        # Pre-seed agent stats so weights are higher
        consensus._agent_stats["a1"] = {"successes": 5.0, "total": 10.0}
        consensus._agent_stats["a2"] = {"successes": 5.0, "total": 10.0}
        def _make(an, aid):
            return AgentProposal(action_name=an, source=aid, confidence=0.5, reason="r",
                                 agent_id=aid, agent_confidence=0.5, agent_goal_depth=1)
        proposals = {
            "a1": [_make("ACTION1", "a1")],
            "a2": [_make("ACTION2", "a2")],
        }
        ctx = self._mock_context()
        result = consensus.collect(proposals, ctx)
        for p in result:
            self.assertGreater(p.confidence, 0.25)

    def test_agent_success_rate_tracking(self):
        agent = MovementAgent()
        self.assertEqual(agent.success_rate, 0.5)
        agent.note_outcome(True)
        self.assertGreater(agent.success_rate, 0.5)
        agent.note_outcome(False)
        self.assertLess(agent.success_rate, 1.0)

    def test_agent_reset(self):
        agent = MovementAgent()
        agent.note_outcome(True)
        agent.note_outcome(True)
        agent.reset()
        self.assertEqual(agent.success_rate, 0.5)
        self.assertEqual(len(agent.wm._items), 0)

    def test_supervisor_creates_agents(self):
        config = MagicMock(spec=RuntimeConfig)
        config.multi_agent_executor = "process"
        config.cuda_enabled = False
        executive = MagicMock()
        sup = AgentSupervisor(executive=executive, config=config)
        self.assertEqual(len(sup._agents), 4)
        self.assertIn("movement", sup._agent_map)
        self.assertIn("exploration", sup._agent_map)
        self.assertIn("interaction", sup._agent_map)
        self.assertIn("puzzle", sup._agent_map)
        self._require_process_executor(sup)
        self.assertTrue(sup._process_executor)

    def test_supervisor_process_mode_does_not_start_workers_during_construction(self):
        config = MagicMock(spec=RuntimeConfig)
        config.multi_agent_executor = "process"
        config.cuda_enabled = False
        with patch(
            "agi_core.multi_agent.runtime._ProcessAgentWorker.ensure_started"
        ) as ensure_started:
            AgentSupervisor(executive=MagicMock(), config=config)
        ensure_started.assert_not_called()

    def test_supervisor_defers_cuda_init_until_first_decide(self):
        config = MagicMock(spec=RuntimeConfig)
        config.multi_agent_workers = 2
        config.planner_parallel_timeout = None
        config.multi_agent_executor = "thread"
        config.cuda_enabled = True
        config.cuda_device = 0
        executive = MagicMock()
        with patch("agi_core.multi_agent.supervisor.init_cuda") as init_cuda:
            supervisor = AgentSupervisor(executive=executive, config=config)
            init_cuda.assert_not_called()
            supervisor._try_solver_direct = MagicMock(return_value=None)
            supervisor._run_agents = MagicMock(return_value={})
            supervisor._consensus.collect = MagicMock(return_value=[])
            context = self._mock_context(legal=["ACTION1"])
            context.memory.pending_score_floor_reopen.return_value = None
            context.hyperon.current_memory_bundle.return_value = None
            executive.decide.return_value = ActionDecision(
                action_name="ACTION1",
                source="fallback",
                confidence=0.1,
                reason="fallback",
            )

            supervisor.decide(context)

        init_cuda.assert_called_once_with(0)

    def test_supervisor_preserves_bootstrapped_thread_pool_worker_budget(self):
        config = MagicMock(spec=RuntimeConfig)
        config.multi_agent_workers = 6
        config.multi_agent_executor = "thread"
        config.cuda_enabled = False
        executive = MagicMock()
        with patch("agi_core.multi_agent.supervisor.ThreadPoolManager.set_max_workers") as set_workers:
            supervisor = AgentSupervisor(executive=executive, config=config)

        self.assertEqual(supervisor._workers, 6)
        set_workers.assert_not_called()

    def test_run_agents_keeps_parallel_partial_results_without_serial_rerun(self):
        import threading
        import time

        config = MagicMock(spec=RuntimeConfig)
        config.multi_agent_workers = 2
        config.planner_parallel_timeout = 0.01
        config.multi_agent_executor = "thread"
        config.cuda_enabled = False
        executive = MagicMock()
        fast_agent = MagicMock()
        fast_agent.agent_id = "fast"
        fast_proposal = AgentProposal(
            action_name="ACTION1",
            source="agent-fast",
            confidence=0.9,
            reason="fast",
            agent_id="fast",
            subgoal="navigate",
        )
        slow_agent = MagicMock()
        slow_agent.agent_id = "slow"
        started = threading.Barrier(2)
        release = threading.Event()
        fast_calls: list[float] = []
        slow_calls: list[float] = []

        def fast_produce(context):
            fast_calls.append(time.monotonic())
            started.wait(timeout=0.2)
            return [fast_proposal]

        def slow_produce(context):
            slow_calls.append(time.monotonic())
            started.wait(timeout=0.2)
            release.wait(timeout=0.2)
            return []

        fast_agent.produce.side_effect = fast_produce
        slow_agent.produce.side_effect = slow_produce
        sup = AgentSupervisor(
            executive=executive,
            config=config,
            agents=[fast_agent, slow_agent],
        )
        sup._agent_timeout = 0.02
        ctx = self._mock_context(legal=["ACTION1"])

        t0 = time.monotonic()
        try:
            proposals = sup._run_agents([fast_agent, slow_agent], ctx)
        finally:
            release.set()
            pending = sup._agent_futures.get("slow")
            if pending is not None:
                pending[0].result(timeout=1.0)
        elapsed = time.monotonic() - t0

        self.assertIn("fast", proposals)
        self.assertNotIn("slow", proposals)
        self.assertEqual(len(fast_calls), 1)
        self.assertEqual(len(slow_calls), 1)
        self.assertLess(abs(fast_calls[0] - slow_calls[0]), 0.05)
        self.assertLess(elapsed, 0.15)

    def test_run_agents_empty_result_does_not_trip_circuit_breaker(self):
        config = MagicMock(spec=RuntimeConfig)
        config.multi_agent_workers = 2
        config.planner_parallel_timeout = None
        config.multi_agent_executor = "thread"
        config.cuda_enabled = False
        agent = MagicMock()
        agent.agent_id = "idle"
        agent.produce.return_value = []
        sup = AgentSupervisor(
            executive=MagicMock(),
            config=config,
            agents=[agent],
        )
        sup._agent_health["idle"] = {"consecutive_failures": 2}

        proposals = sup._run_agents([agent], self._mock_context())

        self.assertEqual(proposals, {})
        self.assertNotIn("idle", sup._agent_health)

    def test_run_agents_does_not_duplicate_in_flight_agent(self):
        import threading

        config = MagicMock(spec=RuntimeConfig)
        config.multi_agent_workers = 2
        config.planner_parallel_timeout = None
        config.multi_agent_executor = "thread"
        config.cuda_enabled = False
        release = threading.Event()
        agent = MagicMock()
        agent.agent_id = "slow"

        def slow_produce(context):
            release.wait(timeout=0.5)
            return []

        agent.produce.side_effect = slow_produce
        supervisor = AgentSupervisor(
            executive=MagicMock(),
            config=config,
            agents=[agent],
        )
        supervisor._agent_timeout = 0.01
        context = self._mock_context()

        try:
            self.assertEqual(supervisor._run_agents([agent], context), {})
            self.assertEqual(supervisor._run_agents([agent], context), {})
            self.assertEqual(agent.produce.call_count, 1)
        finally:
            release.set()
            pending = supervisor._agent_futures.get("slow")
            if pending is not None:
                pending[0].result(timeout=1.0)

    def test_process_executor_partial_results_without_serial_rerun(self):
        config = MagicMock(spec=RuntimeConfig)
        config.multi_agent_workers = 2
        config.planner_parallel_timeout = 0.2
        config.multi_agent_executor = "process"
        config.cuda_enabled = False
        supervisor = AgentSupervisor(
            executive=MagicMock(),
            config=config,
            agents=[FastProcessTestAgent(), SlowProcessTestAgent()],
        )
        self._require_process_executor(supervisor)
        supervisor._agent_timeout = 0.2

        t0 = time.monotonic()
        proposals = supervisor._run_agents(
            supervisor._agents,
            self._mock_context(legal=["ACTION1", "ACTION2"]),
        )
        elapsed = time.monotonic() - t0

        self.assertIn("fast", proposals)
        self.assertNotIn("slow", proposals)
        self.assertLess(elapsed, 0.3)

    def test_process_executor_does_not_duplicate_in_flight_agent(self):
        config = MagicMock(spec=RuntimeConfig)
        config.multi_agent_workers = 1
        config.planner_parallel_timeout = 0.05
        config.multi_agent_executor = "process"
        config.cuda_enabled = False
        supervisor = AgentSupervisor(
            executive=MagicMock(),
            config=config,
            agents=[SlowProcessTestAgent()],
        )
        self._require_process_executor(supervisor)
        supervisor._agent_timeout = 0.05
        context = self._mock_context(legal=["ACTION2"])

        self.assertEqual(supervisor._run_agents(supervisor._agents, context), {})
        self.assertEqual(supervisor._executor.active_worker_count(), 1)
        self.assertEqual(supervisor._run_agents(supervisor._agents, context), {})
        self.assertEqual(supervisor._executor.active_worker_count(), 1)

    def test_process_executor_reaps_stale_completion_without_reusing_proposal(self):
        config = MagicMock(spec=RuntimeConfig)
        config.multi_agent_workers = 1
        config.planner_parallel_timeout = 0.05
        config.multi_agent_executor = "process"
        config.cuda_enabled = False
        supervisor = AgentSupervisor(
            executive=MagicMock(),
            config=config,
            agents=[SlowProcessTestAgent()],
        )
        self._require_process_executor(supervisor)
        supervisor._agent_timeout = 0.05
        context = self._mock_context(legal=["ACTION2"])

        self.assertEqual(supervisor._run_agents(supervisor._agents, context), {})
        time.sleep(0.45)
        self.assertEqual(supervisor._run_agents(supervisor._agents, context), {})
        self.assertGreater(
            supervisor._agent_metrics["slow"]["proposal_count"],
            0.0,
        )

    def test_process_executor_outcome_feedback_updates_worker_owned_cognition(self):
        config = MagicMock(spec=RuntimeConfig)
        config.multi_agent_workers = 1
        config.planner_parallel_timeout = 0.1
        config.multi_agent_executor = "process"
        config.cuda_enabled = False
        supervisor = AgentSupervisor(
            executive=MagicMock(),
            config=config,
            agents=[OutcomeAwareProcessAgent()],
        )
        self._require_process_executor(supervisor)
        context = self._mock_context(legal=["ACTION1", "ACTION2"])

        initial = supervisor._run_agents(supervisor._agents, context)
        self.assertEqual(initial["stateful"][0].action_name, "ACTION1")
        supervisor._pending_agent_credit = {
            "game_id": "ls20",
            "agent_id": "stateful",
            "action_name": "ACTION1",
            "source": "agent-stateful",
            "subgoal": "navigate",
        }
        supervisor.note_transition(
            ObservedTransition(
                game_id="ls20",
                action_name="ACTION1",
                decision_source="agent-stateful",
                local_outcome="progress",
            )
        )
        updated = supervisor._run_agents(supervisor._agents, context)
        self.assertEqual(updated["stateful"][0].action_name, "ACTION2")

    def test_process_executor_level_reset_clears_worker_cognition(self):
        config = MagicMock(spec=RuntimeConfig)
        config.multi_agent_workers = 1
        config.planner_parallel_timeout = 0.1
        config.multi_agent_executor = "process"
        config.cuda_enabled = False
        supervisor = AgentSupervisor(
            executive=MagicMock(),
            config=config,
            agents=[OutcomeAwareProcessAgent()],
        )
        self._require_process_executor(supervisor)
        context = self._mock_context(legal=["ACTION1", "ACTION2"])
        supervisor._pending_agent_credit = {
            "game_id": "ls20",
            "agent_id": "stateful",
            "action_name": "ACTION1",
            "source": "agent-stateful",
            "subgoal": "navigate",
        }
        supervisor.note_transition(
            ObservedTransition(
                game_id="ls20",
                action_name="ACTION1",
                decision_source="agent-stateful",
                local_outcome="progress",
            )
        )
        progressed = supervisor._run_agents(supervisor._agents, context)
        self.assertEqual(progressed["stateful"][0].action_name, "ACTION2")
        supervisor.note_transition(ObservedTransition(game_id="ls20", boundary="level"))
        reset = supervisor._run_agents(supervisor._agents, context)
        self.assertEqual(reset["stateful"][0].action_name, "ACTION1")

    def test_process_executor_failures_degrade_without_crashing(self):
        config = MagicMock(spec=RuntimeConfig)
        config.multi_agent_workers = 1
        config.planner_parallel_timeout = 0.1
        config.multi_agent_executor = "process"
        config.cuda_enabled = False
        supervisor = AgentSupervisor(
            executive=MagicMock(),
            config=config,
            agents=[FailingProcessAgent()],
        )
        self._require_process_executor(supervisor)

        proposals = supervisor._run_agents(
            supervisor._agents,
            self._mock_context(legal=["ACTION1"]),
        )

        self.assertEqual(proposals, {})
        self.assertGreaterEqual(
            supervisor._agent_health["failing"]["consecutive_failures"],
            1,
        )

    def test_process_and_thread_modes_produce_equivalent_proposals(self):
        thread_config = MagicMock(spec=RuntimeConfig)
        thread_config.multi_agent_workers = 2
        thread_config.planner_parallel_timeout = 0.1
        thread_config.multi_agent_executor = "thread"
        thread_config.cuda_enabled = False
        process_config = MagicMock(spec=RuntimeConfig)
        process_config.multi_agent_workers = 2
        process_config.planner_parallel_timeout = 0.1
        process_config.multi_agent_executor = "process"
        process_config.cuda_enabled = False
        context = self._mock_context(legal=["ACTION1", "ACTION2"])

        thread_sup = AgentSupervisor(
            executive=MagicMock(),
            config=thread_config,
            agents=[FastProcessTestAgent(), OutcomeAwareProcessAgent()],
        )
        process_sup = AgentSupervisor(
            executive=MagicMock(),
            config=process_config,
            agents=[FastProcessTestAgent(), OutcomeAwareProcessAgent()],
        )
        self._require_process_executor(process_sup)
        thread_props = thread_sup._run_agents(thread_sup._agents, context)
        process_props = process_sup._run_agents(process_sup._agents, context)

        self.assertEqual(
            {agent_id: [p.action_name for p in proposals] for agent_id, proposals in thread_props.items()},
            {agent_id: [p.action_name for p in proposals] for agent_id, proposals in process_props.items()},
        )

    def test_circuit_breaker_allows_half_open_recovery_probe(self):
        config = MagicMock(spec=RuntimeConfig)
        config.multi_agent_workers = 2
        config.planner_parallel_timeout = None
        config.cuda_enabled = False
        supervisor = AgentSupervisor(executive=MagicMock(), config=config)

        for _ in range(3):
            supervisor._record_failure("movement")
        self.assertTrue(supervisor._is_circuit_open("movement"))

        supervisor._agent_health["movement"]["open_until"] = time.monotonic() - 1.0
        self.assertFalse(supervisor._is_circuit_open("movement"))
        self.assertEqual(supervisor._agent_health["movement"]["half_open"], 1.0)

        supervisor._record_success("movement", [])
        self.assertFalse(supervisor._is_circuit_open("movement"))

    def test_game_over_transition_resets_episode_state_after_feedback(self):
        config = MagicMock(spec=RuntimeConfig)
        config.multi_agent_workers = 2
        config.planner_parallel_timeout = None
        config.cuda_enabled = False
        executive = MagicMock()
        supervisor = AgentSupervisor(executive=executive, config=config)
        supervisor._agent_health["movement"] = {"consecutive_failures": 3}
        supervisor._solver_cooldown = 7
        supervisor._sticky_shape_target = {"target": [3, 4]}

        supervisor.note_transition(
            ObservedTransition(
                game_id="ls20",
                action_name="ACTION1",
                decision_source="agent-movement",
                local_outcome="failed",
                boundary="game_over",
            )
        )

        self.assertEqual(supervisor._agent_health, {})
        self.assertEqual(supervisor._solver_cooldown, 0)
        self.assertIsNone(supervisor._sticky_shape_target)
        executive.note_transition.assert_called_once()
        executive.reset_episode.assert_called_once()

    def test_solver_and_agents_share_consensus_and_critic_path(self):
        import threading

        config = MagicMock(spec=RuntimeConfig)
        config.multi_agent_workers = 2
        config.planner_parallel_timeout = None
        config.multi_agent_executor = "thread"
        config.cuda_enabled = False
        executive = MagicMock()
        overlap_confirmed = threading.Event()
        active_lock = threading.Lock()
        active_count = 0

        def make_agent(agent_id: str, action_name: str):
            agent = MagicMock()
            agent.agent_id = agent_id
            proposal = AgentProposal(
                action_name=action_name,
                source=f"agent-{agent_id}",
                confidence=0.7,
                reason="parallel-probe",
                agent_id=agent_id,
            )

            def produce(context):
                nonlocal active_count
                with active_lock:
                    active_count += 1
                    if active_count == 2:
                        overlap_confirmed.set()
                overlap_confirmed.wait(timeout=0.2)
                with active_lock:
                    active_count -= 1
                return [proposal]

            agent.produce.side_effect = produce
            return agent

        agents = [
            make_agent("movement", "ACTION1"),
            make_agent("exploration", "ACTION2"),
        ]
        solver_decision = ActionDecision(
            action_name="ACTION3",
            source="solver:mirror",
            confidence=0.8,
            reason="solver remains authoritative",
        )
        sup = AgentSupervisor(
            executive=executive,
            config=config,
            agents=agents,
        )
        sup._try_solver_direct = MagicMock(return_value=solver_decision)
        sup._finalize_cognition = MagicMock()
        executive.critic.rescore.side_effect = lambda context, proposals: proposals
        executive._all_legal_proposals_are_score_floor.return_value = False
        executive._probe_dead_end_escape.return_value = None
        executive._finalize_decision_metadata.side_effect = (
            lambda context, action_name, metadata: metadata
        )
        executive._finalize_decision_subgoal.side_effect = (
            lambda context, action_name, subgoal: subgoal
        )
        ctx = self._mock_context(legal=["ACTION1", "ACTION2", "ACTION3"])
        ctx.goals = []
        ctx.memory.pending_score_floor_reopen.return_value = None
        ctx.hyperon.current_memory_bundle.return_value = None

        decision = sup.decide(ctx)

        self.assertEqual(decision.action_name, "ACTION3")
        self.assertEqual(decision.source, "solver:mirror")
        self.assertTrue(overlap_confirmed.is_set())
        for agent in agents:
            agent.produce.assert_called_once()
        sup._try_solver_direct.assert_called_once_with(ctx)
        executive.critic.rescore.assert_not_called()
        ctx.hyperon.update_proposals.assert_not_called()
        self.assertEqual(sup._agent_health, {})

    def test_illegal_solver_proposal_cannot_bypass_legal_agent(self):
        config = MagicMock(spec=RuntimeConfig)
        config.multi_agent_workers = 2
        config.planner_parallel_timeout = None
        config.multi_agent_executor = "thread"
        config.cuda_enabled = False
        config.reset_state_names = ("NOT_PLAYED", "GAME_OVER")
        executive = MagicMock()
        agent = MagicMock()
        agent.agent_id = "movement"
        agent.produce.return_value = [
            AgentProposal(
                action_name="ACTION1",
                source="agent-movement",
                confidence=0.7,
                reason="legal move",
                agent_id="movement",
            )
        ]
        supervisor = AgentSupervisor(
            executive=executive,
            config=config,
            agents=[agent],
        )
        supervisor._try_solver_direct = MagicMock(
            return_value=ActionDecision(
                action_name="ACTION3",
                source="solver:probe",
                confidence=0.99,
                reason="illegal probe",
            )
        )
        supervisor._finalize_cognition = MagicMock()
        executive.critic.rescore.side_effect = lambda context, proposals: proposals
        executive._all_legal_proposals_are_score_floor.return_value = False
        executive._probe_dead_end_escape.return_value = None
        executive._finalize_decision_metadata.side_effect = (
            lambda context, action_name, metadata: metadata
        )
        executive._finalize_decision_subgoal.side_effect = (
            lambda context, action_name, subgoal: subgoal
        )
        context = self._mock_context(legal=["ACTION1"])
        context.memory.pending_score_floor_reopen.return_value = None

        decision = supervisor.decide(context)

        self.assertEqual(decision.action_name, "ACTION1")
        self.assertEqual(decision.source, "agent-movement")
        solver_proposal = next(
            proposal
            for proposal in decision.proposals
            if proposal.source == "solver:probe"
        )
        self.assertNotIn(
            solver_proposal.action_name,
            context.observation.legal_actions,
        )

    def test_finalize_cognition_defers_credit_until_observed_transition(self):
        config = MagicMock(spec=RuntimeConfig)
        config.multi_agent_executor = "thread"
        config.cuda_enabled = False
        executive = MagicMock()
        sup = AgentSupervisor(executive=executive, config=config)
        decision = ActionDecision(
            action_name="ACTION1",
            source="agent-movement",
            confidence=0.8,
            reason="test",
            subgoal="navigate",
        )
        ctx = self._mock_context()

        sup._finalize_cognition(
            ctx,
            decision,
            winning_agent_id="movement",
            participating_agent_ids={"movement", "interaction"},
        )

        self.assertEqual(sup._agent_map["movement"]._total_count, 0)
        self.assertEqual(sup._agent_map["interaction"]._total_count, 0)
        sup.note_transition(
            ObservedTransition(
                game_id=ctx.observation.game_id,
                action_name="ACTION1",
                decision_source="agent-movement",
                decision_subgoal="navigate",
                local_outcome="advanced",
                player_moved=True,
            )
        )

        self.assertEqual(sup._agent_map["movement"]._total_count, 1)
        self.assertEqual(sup._agent_map["movement"]._success_count, 1)
        self.assertEqual(sup._agent_map["interaction"]._total_count, 0)
        self.assertEqual(sup._agent_map["interaction"]._success_count, 0)
        self.assertEqual(sup._agent_map["exploration"]._total_count, 0)
        self.assertEqual(sup._agent_map["puzzle"]._total_count, 0)
        executive._finalize_cognition.assert_called_once_with(ctx, decision)
        executive.note_transition.assert_called_once()

    def test_finalize_cognition_solver_does_not_penalize_agents(self):
        config = MagicMock(spec=RuntimeConfig)
        executive = MagicMock()
        sup = AgentSupervisor(executive=executive, config=config)
        decision = ActionDecision(
            action_name="ACTION4",
            source="solver:mirror",
            confidence=0.9,
            reason="solver",
            subgoal="solve-puzzle",
        )
        ctx = self._mock_context()

        sup._finalize_cognition(ctx, decision)

        for agent in sup._agent_map.values():
            self.assertEqual(agent._total_count, 0)
            self.assertEqual(agent._success_count, 0)
        executive._finalize_cognition.assert_called_once_with(ctx, decision)

    def test_solver_stall_ignores_repeated_action_while_player_moves(self):
        config = MagicMock(spec=RuntimeConfig)
        executive = MagicMock()
        sup = AgentSupervisor(executive=executive, config=config)

        self.assertFalse(sup._update_solver_stall((10, 10), "ACTION1"))
        for x in range(11, 23):
            self.assertFalse(sup._update_solver_stall((x, 10), "ACTION1"))
        self.assertEqual(sup._solver_stall_count, 0)

    def test_solver_stall_triggers_for_repeated_action_without_movement(self):
        config = MagicMock(spec=RuntimeConfig)
        executive = MagicMock()
        sup = AgentSupervisor(executive=executive, config=config)

        self.assertFalse(sup._update_solver_stall((10, 10), "ACTION1"))
        tripped = False
        for _ in range(10):
            tripped = sup._update_solver_stall((10, 10), "ACTION1")
        self.assertTrue(tripped)
        self.assertGreaterEqual(sup._solver_stall_count, 10)

    def test_solver_stall_triggers_for_two_action_oscillation(self):
        config = MagicMock(spec=RuntimeConfig)
        executive = MagicMock()
        sup = AgentSupervisor(executive=executive, config=config)

        sequence = [
            ((10, 10), "ACTION1"),
            ((10, 13), "ACTION2"),
            ((10, 10), "ACTION1"),
            ((10, 13), "ACTION2"),
            ((10, 10), "ACTION1"),
            ((10, 13), "ACTION2"),
        ]
        tripped = False
        for pos, action in sequence:
            tripped = sup._update_solver_stall(pos, action)
        self.assertTrue(tripped)
        self.assertGreaterEqual(sup._solver_stall_count, 10)

    def test_generic_solver_stall_enters_cooldown(self):
        config = MagicMock(spec=RuntimeConfig)
        config.multi_agent_workers = 2
        config.planner_parallel_timeout = None
        config.cuda_enabled = False
        supervisor = AgentSupervisor(executive=MagicMock(), config=config)
        context = self._mock_context(legal=["ACTION1", "ACTION2"])
        game_model = MagicMock()
        game_model.player_pos = (4, 5)
        game_model.features.board_type = "puzzle"
        game_model.features.primary_dynamics = "beam"
        game_model.features.dynamics = ["beam"]
        solver = MagicMock()
        solver.dynamics = "beam"
        solver.should_activate.return_value = True
        solver.decide.return_value = ActionDecision(
            action_name="ACTION1",
            source="solver:beam",
            confidence=0.8,
            reason="repeat",
        )
        supervisor._solver_selector = MagicMock()
        supervisor._solver_selector.select_solver.return_value = solver
        supervisor._update_solver_stall = MagicMock(return_value=True)

        with (
            patch("agi_core.multi_agent.supervisor._get_board_data", return_value=[[0]]),
            patch("agi_core.multi_agent.supervisor.build_game_model", return_value=game_model),
            patch("agi_core.multi_agent.supervisor._board_has_mirror", return_value=False),
            patch("agi_core.multi_agent.supervisor._build_solver_context", return_value=MagicMock()),
            patch("agi_core.multi_agent.supervisor.ensure_shape_recognition"),
        ):
            decision = supervisor._try_solver_direct(context)

        self.assertIsNone(decision)
        self.assertEqual(supervisor._solver_cooldown, 10)
        supervisor._update_solver_stall.assert_called_once_with((4, 5), "ACTION1")

    def test_strategy_agent_make_proposal(self):
        agent = MovementAgent()
        p = agent._make_proposal(
            action_name="ACTION1",
            confidence=0.8,
            reason="test",
            subgoal="navigate",
        )
        self.assertIsInstance(p, AgentProposal)
        self.assertEqual(p.agent_id, "movement")
        self.assertEqual(p.action_name, "ACTION1")
        self.assertEqual(p.subgoal, "navigate")

    def test_supervisor_decide_reuses_agent_proposals_for_rescore(self):
        config = MagicMock(spec=RuntimeConfig)
        config.multi_agent_workers = 4
        config.planner_parallel_timeout = None
        config.cuda_enabled = False
        executive = MagicMock()
        agent = MagicMock()
        agent.agent_id = "movement"
        proposal = AgentProposal(
            action_name="ACTION1",
            source="agent-movement",
            confidence=0.7,
            reason="move",
            agent_id="movement",
            subgoal="navigate",
        )
        agent.produce.return_value = [proposal]
        executive.critic.rescore.return_value = [proposal]
        executive._all_legal_proposals_are_score_floor.return_value = False
        executive._probe_dead_end_escape.return_value = None
        executive._finalize_decision_metadata.side_effect = lambda context, action_name, metadata: metadata
        executive._finalize_decision_subgoal.side_effect = lambda context, action_name, subgoal: subgoal
        sup = AgentSupervisor(executive=executive, config=config, agents=[agent])
        sup._try_solver_direct = MagicMock(return_value=None)
        sup._finalize_cognition = MagicMock()
        ctx = self._mock_context(legal=["ACTION1"])
        ctx.hyperon.current_memory_bundle.return_value = None
        ctx.memory.pending_score_floor_reopen.return_value = None

        decision = sup.decide(ctx)

        ctx.hyperon.update_proposals.assert_called_once()
        reused = ctx.hyperon.update_proposals.call_args.args[0]
        self.assertIs(reused[0], proposal)
        executive.critic.rescore.assert_called_once()
        rescored_arg = executive.critic.rescore.call_args.args[1]
        self.assertIs(rescored_arg[0], proposal)
        self.assertEqual(decision.action_name, "ACTION1")
        sup._finalize_cognition.assert_called_once()

    def test_supervisor_score_floor_fallback_notes_and_finalizes_decision(self):
        config = MagicMock(spec=RuntimeConfig)
        config.multi_agent_workers = 4
        config.planner_parallel_timeout = None
        config.cuda_enabled = False
        executive = MagicMock()
        agent = MagicMock()
        agent.agent_id = "movement"
        proposal = AgentProposal(
            action_name="ACTION1",
            source="agent-movement",
            confidence=0.7,
            reason="move",
            agent_id="movement",
            subgoal="navigate",
        )
        fallback = ActionDecision(
            action_name="ACTION2",
            source="fallback",
            confidence=0.05,
            reason="score-floor",
        )
        agent.produce.return_value = [proposal]
        executive.critic.rescore.return_value = [proposal]
        executive._all_legal_proposals_are_score_floor.return_value = True
        executive._score_floor_fallback.return_value = fallback
        sup = AgentSupervisor(executive=executive, config=config, agents=[agent])
        sup._try_solver_direct = MagicMock(return_value=None)
        sup._finalize_cognition = MagicMock()
        ctx = self._mock_context(legal=["ACTION1", "ACTION2"])
        ctx.memory.pending_score_floor_reopen.return_value = None

        decision = sup.decide(ctx)

        self.assertIs(decision, fallback)
        self.assertIs(decision.proposals, executive.critic.rescore.return_value)
        ctx.memory.note_decision.assert_called_once_with(fallback)
        sup._finalize_cognition.assert_called_once_with(ctx, fallback)

    def test_supervisor_pending_reopen_reset_does_not_delegate_to_base_executive(self):
        config = MagicMock(spec=RuntimeConfig)
        config.multi_agent_workers = 4
        config.planner_parallel_timeout = None
        config.cuda_enabled = False
        executive = MagicMock()
        sup = AgentSupervisor(executive=executive, config=config)
        sup._finalize_cognition = MagicMock()
        ctx = self._mock_context(legal=["RESET"])
        ctx.goals = []
        ctx.hyperon.current_memory_bundle.return_value = None
        ctx.memory.pending_score_floor_reopen.return_value = {
            "score_floor_debug": {"pending_reset": True}
        }

        decision = sup.decide(ctx)

        self.assertEqual(decision.action_name, "RESET")
        self.assertEqual(decision.metadata["score_floor_debug"]["pending_reset"], True)
        ctx.memory.clear_pending_score_floor_reopen.assert_called_once()
        ctx.memory.note_decision.assert_called_once_with(decision)
        sup._finalize_cognition.assert_called_once_with(ctx, decision)
        executive.decide.assert_not_called()

    def test_runtime_sync_resyncs_when_memory_changes_on_same_observation(self):
        ctx = self._mock_context()
        ctx.memory = EpisodicMemory()
        ctx.goals = []
        ctx.observation.game_id = "ls20"
        ctx.hyperon.current_memory_bundle.side_effect = ["bundle-a", "bundle-b"]
        ctx.memory_bundle = None

        first = ensure_hyperon_context_synced(ctx)
        ctx.memory.note_observation(ctx.observation)
        ctx.memory.note_decision(
            ActionDecision(
                action_name="ACTION1",
                source="symbolic",
                confidence=0.6,
                reason="same-frame-update",
            )
        )
        second = ensure_hyperon_context_synced(ctx)

        self.assertEqual(first, "bundle-a")
        self.assertEqual(second, "bundle-b")
        self.assertEqual(ctx.hyperon.sync.call_count, 2)

    def test_runtime_sync_resyncs_when_goals_change_on_same_observation(self):
        ctx = self._mock_context()
        goal_a = MagicMock()
        goal_a.name = "goal-a"
        goal_a.target_state = "WIN"
        goal_a.priority = 1
        goal_a.metadata = {}
        goal_b = MagicMock()
        goal_b.name = "goal-b"
        goal_b.target_state = "LOSE"
        goal_b.priority = 2
        goal_b.metadata = {"mode": "fallback"}
        ctx.goals = [goal_a]
        ctx.hyperon.current_memory_bundle.side_effect = ["bundle-a", "bundle-b"]
        ctx.memory_bundle = None

        first = ensure_hyperon_context_synced(ctx)
        ctx.goals = [goal_b]
        second = ensure_hyperon_context_synced(ctx)

        self.assertEqual(first, "bundle-a")
        self.assertEqual(second, "bundle-b")
        self.assertEqual(ctx.hyperon.sync.call_count, 2)

    def test_prepare_frame_cache_precomputes_shared_features(self):
        ctx = self._mock_context(
            legal=["ACTION2", "ACTION4", "ACTION6"],
            derived_features={
                "player_position": (2, 2),
                "goal_positions": [(5, 2), (6, 2)],
            },
            board=[
                [9, 9, 9],
                [9, 4, 9],
                [10, 9, 9],
            ],
        )

        cache = prepare_frame_cache(ctx)

        self.assertEqual(cache["legal_set"], {"ACTION2", "ACTION4", "ACTION6"})
        self.assertTrue(cache["has_beam"])
        self.assertTrue(cache["has_mirror"])
        self.assertEqual(cache["goal_direction"], "ACTION4")
        self.assertIn("shape_recognition", cache)
        self.assertIs(cache, prepare_frame_cache(ctx))

    def test_prepare_frame_cache_fills_agent_fields_after_hyperon_sync(self):
        ctx = self._mock_context(legal=["ACTION1", "ACTION5"])
        sync_token = ("ar25", 3, "PLAYING")
        ctx.multi_agent_frame_cache["__hyperon_sync_token__"] = sync_token

        cache = prepare_frame_cache(ctx)

        self.assertEqual(cache["__hyperon_sync_token__"], sync_token)
        self.assertEqual(cache["legal_actions"], ("ACTION1", "ACTION5"))
        self.assertEqual(cache["legal_set"], {"ACTION1", "ACTION5"})

    def test_prepare_frame_cache_respects_explicit_false_beam_hint(self):
        ctx = self._mock_context(
            legal=["ACTION2", "ACTION4", "ACTION6"],
            derived_features={
                "player_position": (2, 2),
                "goal_positions": [(5, 2)],
                "has_beam": False,
            },
            board=[
                [9, 9, 9, 9, 9, 9, 9, 9],
                [9, 9, 9, 10, 9, 9, 9, 9],
                [9, 9, 9, 10, 9, 9, 9, 9],
                [9, 9, 9, 10, 9, 9, 9, 9],
                [9, 5, 5, 10, 9, 4, 4, 9],
                [9, 5, 5, 10, 9, 4, 4, 9],
                [9, 9, 9, 10, 9, 9, 9, 9],
                [9, 9, 9, 10, 9, 9, 9, 9],
            ],
        )

        cache = prepare_frame_cache(ctx)

        self.assertFalse(cache["has_beam"])

    def test_prepare_frame_cache_keeps_beam_uncertain_when_analyzer_failed(self):
        ctx = self._mock_context(
            legal=["ACTION2", "ACTION4", "ACTION6"],
            derived_features={
                "player_position": (2, 2),
                "goal_positions": [(5, 2)],
                "has_beam": False,
                "analyzer_perception_failed": True,
            },
            board=[
                [9, 9, 9],
                [9, 4, 9],
                [9, 9, 9],
            ],
        )

        cache = prepare_frame_cache(ctx)

        self.assertFalse(cache["has_beam"])
        self.assertTrue(cache["beam_visible"])
        self.assertTrue(cache["beam_evidence_uncertain"])

    def test_exploration_agent_marks_uncertain_beam_switch_reason(self):
        agent = ExplorationAgent()
        ctx = self._mock_context(
            legal=["ACTION1", "ACTION5"],
            derived_features={
                "analyzer_perception_failed": True,
                "has_beam": False,
            },
            board=[
                [9, 4, 9],
                [9, 9, 9],
                [9, 9, 9],
            ],
        )

        proposals = agent.produce(ctx)

        switch = next(p for p in proposals if p.action_name == "ACTION5")
        self.assertEqual(switch.reason, "exploration:switch+beam-uncertain")

    def test_puzzle_agent_uses_uncertain_beam_as_puzzle_signal(self):
        agent = PuzzleAgent()
        ctx = self._mock_context(
            legal=["ACTION5", "ACTION4"],
            derived_features={
                "analyzer_perception_failed": True,
                "has_beam": False,
            },
            board=[
                [9, 4, 9],
                [9, 9, 9],
                [9, 9, 9],
            ],
        )

        proposals = agent.produce(ctx)

        self.assertTrue(proposals)
        self.assertTrue(all("+beam-uncertain" in p.reason for p in proposals))

    def test_prepare_frame_cache_exposes_shape_target(self):
        ctx = self._mock_context(
            legal=["ACTION2", "ACTION4", "ACTION5"],
            derived_features={
                "shape_recognition": {
                    "version": 1,
                    "active_anchor": [3, 3],
                    "mirror": {"present": True},
                    "goal_clusters": [{"center": [7, 3]}],
                    "mirror_puzzle": {
                        "active": True,
                        "active_stage": 1,
                        "stage_targets": [{"target_anchor": [7, 3]}],
                    },
                },
            },
        )

        cache = prepare_frame_cache(ctx)

        self.assertTrue(cache["mirror_puzzle_active"])
        self.assertEqual(cache["active_shape_target"], (7, 3))
        self.assertFalse(cache["shape_agent_mode"])

    def test_prepare_frame_cache_normalizes_single_stage_vertical_target(self):
        ctx = self._mock_context(
            legal=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"],
            derived_features={
                "player_position": (8, 42),
                "goal_positions": [(52, 46), (58, 52), (55, 49)],
                "shape_recognition": {
                    "version": 2,
                    "active_anchor": [8, 42],
                    "mirror": {"present": True},
                    "goal_clusters": [{"center": [55, 49]}],
                    "mirror_puzzle": {
                        "active": True,
                        "orientation": "vertical",
                        "axis_pixel": 31,
                        "active_stage": 1,
                        "stage_targets": [{"target_anchor": [10, 46]}],
                    },
                },
            },
        )

        cache = prepare_frame_cache(ctx)

        self.assertEqual(cache["active_shape_target"], (4, 46))

    def test_prepare_frame_cache_enables_shape_agent_mode_after_level_two(self):
        ctx = self._mock_context(
            legal=["ACTION2", "ACTION4", "ACTION5"],
            derived_features={
                "shape_recognition": {
                    "version": 1,
                    "active_anchor": [3, 3],
                    "mirror": {"present": True},
                    "goal_clusters": [{"center": [7, 3]}],
                    "mirror_puzzle": {
                        "active": True,
                        "active_stage": 1,
                        "stage_targets": [{"target_anchor": [7, 3]}],
                    },
                },
            },
        )
        ctx.observation.levels_completed = 2

        cache = prepare_frame_cache(ctx)

        self.assertTrue(cache["shape_agent_mode"])

    def test_puzzle_agent_uses_shape_target_metadata(self):
        agent = PuzzleAgent()
        ctx = self._mock_context(
            legal=["ACTION2", "ACTION4", "ACTION5", "ACTION6"],
            derived_features={
                "player_position": (2, 2),
                "shape_recognition": {
                    "version": 1,
                    "active_anchor": [2, 2],
                    "mirror": {"present": True},
                    "goal_clusters": [{"center": [8, 2]}],
                    "mirror_puzzle": {
                        "active": True,
                        "active_stage": 1,
                        "stage_targets": [{"target_anchor": [8, 2]}],
                    },
                },
            },
        )
        ctx.observation.levels_completed = 2

        proposals = agent.produce(ctx)

        self.assertTrue(any("target_anchor" in proposal.metadata for proposal in proposals))
        self.assertFalse(any(proposal.action_name == "ACTION6" for proposal in proposals))

    def test_ordered_mirror_actions_prioritize_both_target_axes(self):
        ctx = self._mock_context(
            legal=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"],
            derived_features={
                "player_position": (2, 6),
                "shape_recognition": {
                    "version": 1,
                    "active_anchor": [2, 6],
                    "mirror": {"present": True},
                    "goal_clusters": [{"center": [8, 2]}],
                    "mirror_puzzle": {
                        "active": True,
                        "active_stage": 1,
                        "stage_targets": [{"target_anchor": [8, 2]}],
                    },
                },
            },
        )
        ctx.observation.levels_completed = 2

        ordered = _ordered_mirror_actions(ctx)

        self.assertEqual(ordered[:2], ["ACTION4", "ACTION1"])
        self.assertNotIn("ACTION5", ordered)
        self.assertEqual(ordered, ["ACTION4", "ACTION1"])

    def test_prepare_frame_cache_requires_exactish_anchor_before_switch(self):
        ctx = self._mock_context(
            legal=["ACTION2", "ACTION4", "ACTION5"],
            derived_features={
                "player_position": (5, 5),
                "shape_recognition": {
                    "version": 1,
                    "active_anchor": [5, 5],
                    "mirror": {"present": True},
                    "goal_clusters": [{"center": [7, 5]}],
                    "mirror_puzzle": {
                        "active": True,
                        "active_stage": 1,
                        "stage_targets": [{"target_anchor": [7, 5]}],
                    },
                },
            },
        )
        ctx.observation.levels_completed = 2

        cache = prepare_frame_cache(ctx)

        self.assertEqual(cache["active_shape_target_distance"], 2)
        self.assertFalse(cache["ready_to_switch"])

    def test_prepare_frame_cache_preserves_origin_coordinates(self):
        ctx = self._mock_context(
            legal=["ACTION1", "ACTION3", "ACTION5"],
            derived_features={
                "player_position": (0, 0),
                "shape_recognition": {
                    "version": 2,
                    "frame_signature": "sig-origin",
                    "player_hint": [0, 0],
                    "active_anchor": [0, 0],
                    "goal_clusters": [{"center": [0, 0]}],
                    "mirror": {"present": True},
                    "mirror_puzzle": {
                        "active": True,
                        "active_stage": 1,
                        "stage_targets": [{"target_anchor": [0, 0]}],
                    },
                },
            },
        )
        ctx.observation.levels_completed = 2

        cache = prepare_frame_cache(ctx)
        target_metadata = _target_metadata(ctx)

        self.assertEqual(cache["active_shape_target"], (0, 0))
        self.assertEqual(cache["active_shape_target_distance"], 0)
        self.assertTrue(cache["ready_to_switch"])
        self.assertEqual(target_metadata["target_anchor"], [0, 0])
        self.assertEqual(target_metadata["shape_stage"], 1)

    def test_supervisor_reuses_sticky_shape_target_when_stage_targets_disappear(self):
        config = MagicMock(spec=RuntimeConfig)
        executive = MagicMock()
        sup = AgentSupervisor(executive=executive, config=config)

        ctx1 = self._mock_context(
            legal=["ACTION2", "ACTION4", "ACTION5"],
            derived_features={
                "shape_recognition": {
                    "version": 1,
                    "active_anchor": [3, 3],
                    "mirror": {"present": True},
                    "goal_clusters": [{"center": [7, 3]}],
                    "mirror_puzzle": {
                        "active": True,
                        "active_stage": 1,
                        "orientation": "vertical",
                        "stage_targets": [{"target_anchor": [7, 3]}],
                    },
                },
            },
        )
        ctx1.observation.levels_completed = 2
        cache1 = sup._stabilize_shape_frame_cache(ctx1, prepare_frame_cache(ctx1))
        self.assertEqual(cache1["active_shape_target"], (7, 3))

        ctx2 = self._mock_context(
            legal=["ACTION2", "ACTION4", "ACTION5"],
            derived_features={
                "shape_recognition": {
                    "version": 1,
                    "active_anchor": [3, 3],
                    "mirror": {"present": True},
                    "goal_clusters": [{"center": [7, 3]}],
                    "mirror_puzzle": {
                        "active": True,
                        "active_stage": 1,
                        "orientation": "vertical",
                        "stage_targets": [],
                    },
                },
            },
        )
        ctx2.observation.levels_completed = 2
        cache2 = sup._stabilize_shape_frame_cache(ctx2, prepare_frame_cache(ctx2))

        self.assertEqual(cache2["active_shape_target"], (7, 3))
        self.assertTrue(cache2["shape_agent_mode"])

    def test_supervisor_keeps_sticky_shape_target_when_stage_anchor_jumps_mid_stage(self):
        config = MagicMock(spec=RuntimeConfig)
        executive = MagicMock()
        sup = AgentSupervisor(executive=executive, config=config)

        ctx1 = self._mock_context(
            legal=["ACTION2", "ACTION4", "ACTION5"],
            derived_features={
                "player_position": (20, 20),
                "shape_recognition": {
                    "version": 1,
                    "active_anchor": [20, 20],
                    "mirror": {"present": True},
                    "goal_clusters": [{"center": [46, 43]}],
                    "mirror_puzzle": {
                        "active": True,
                        "active_stage": 1,
                        "orientation": "vertical",
                        "stage_targets": [{"target_anchor": [46, 43]}],
                    },
                },
            },
        )
        ctx1.observation.levels_completed = 2
        cache1 = sup._stabilize_shape_frame_cache(ctx1, prepare_frame_cache(ctx1))
        self.assertEqual(cache1["active_shape_target"], (46, 43))

        ctx2 = self._mock_context(
            legal=["ACTION2", "ACTION4", "ACTION5"],
            derived_features={
                "player_position": (22, 22),
                "shape_recognition": {
                    "version": 1,
                    "active_anchor": [22, 22],
                    "mirror": {"present": True},
                    "goal_clusters": [{"center": [36, 48]}],
                    "mirror_puzzle": {
                        "active": True,
                        "active_stage": 1,
                        "orientation": "vertical",
                        "stage_targets": [{"target_anchor": [36, 48]}],
                    },
                },
            },
        )
        ctx2.observation.levels_completed = 2

        cache2 = sup._stabilize_shape_frame_cache(ctx2, prepare_frame_cache(ctx2))

        self.assertEqual(cache2["active_shape_target"], (46, 43))
        self.assertEqual(cache2["shape_stage_targets"][0]["target_anchor"], [46, 43])

    def test_supervisor_keeps_sticky_stage_targets_for_same_stage_attempt(self):
        config = MagicMock(spec=RuntimeConfig)
        executive = MagicMock()
        sup = AgentSupervisor(executive=executive, config=config)

        ctx1 = self._mock_context(
            legal=["ACTION2", "ACTION4", "ACTION5"],
            derived_features={
                "player_position": (20, 20),
                "shape_recognition": {
                    "version": 1,
                    "active_anchor": [20, 20],
                    "mirror": {"present": True},
                    "goal_clusters": [{"center": [46, 43]}],
                    "mirror_puzzle": {
                        "active": True,
                        "active_stage": 1,
                        "orientation": "horizontal",
                        "stage_targets": [
                            {"target_anchor": [46, 43]},
                            {"target_anchor": [13, 43]},
                        ],
                    },
                },
            },
        )
        ctx1.observation.levels_completed = 3
        sup._stabilize_shape_frame_cache(ctx1, prepare_frame_cache(ctx1))

        ctx2 = self._mock_context(
            legal=["ACTION2", "ACTION4", "ACTION5"],
            derived_features={
                "player_position": (22, 22),
                "shape_recognition": {
                    "version": 1,
                    "active_anchor": [22, 22],
                    "mirror": {"present": True},
                    "goal_clusters": [{"center": [36, 48]}],
                    "mirror_puzzle": {
                        "active": True,
                        "active_stage": 1,
                        "orientation": "horizontal",
                        "stage_targets": [
                            {"target_anchor": [36, 48]},
                            {"target_anchor": [31, 43]},
                        ],
                    },
                },
            },
        )
        ctx2.observation.levels_completed = 3

        cache2 = sup._stabilize_shape_frame_cache(ctx2, prepare_frame_cache(ctx2))

        self.assertEqual(cache2["active_shape_target"], (46, 43))
        self.assertEqual(cache2["shape_stage_targets"][0]["target_anchor"], [46, 43])
        self.assertEqual(cache2["shape_stage_targets"][1]["target_anchor"], [13, 43])

    def test_supervisor_seeds_sticky_shape_target_from_solver_metadata(self):
        config = MagicMock(spec=RuntimeConfig)
        executive = MagicMock()
        sup = AgentSupervisor(executive=executive, config=config)
        ctx = self._mock_context(
            legal=["ACTION2", "ACTION4", "ACTION5"],
            derived_features={
                "shape_recognition": {
                    "version": 1,
                    "mirror": {"present": True},
                    "mirror_puzzle": {
                        "active": True,
                        "orientation": "vertical",
                        "active_stage": 1,
                        "stage_targets": [],
                    },
                },
            },
        )
        ctx.observation.levels_completed = 2
        decision = ActionDecision(
            action_name="ACTION2",
            source="solver:mirror",
            confidence=0.75,
            reason="mirror_placement_down",
            metadata={"target_anchor": [46, 43], "shape_stage": 1},
        )

        sup._seed_sticky_shape_target(ctx, decision)

        self.assertEqual(
            sup._sticky_shape_target,
            {
                "levels_completed": 2,
                "target": (46, 43),
                "stage": 1,
                "orientation": "vertical",
                "stage_targets": [{"target_anchor": [46, 43]}],
            },
        )

    def test_shape_recognition_is_backward_safe_when_absent(self):
        artifact = recognize_shape_recognition(None)
        self.assertFalse(artifact["mirror_puzzle"]["active"])

    def test_consensus_context_signals_are_cached_on_context(self):
        ctx = self._mock_context(
            derived_features={
                "last_action": "action6",
                "primary_dynamics": "Beam",
            }
        )

        first = _consensus_context_signals(ctx)
        self.assertEqual(first, ("ACTION6", "beam"))

        ctx.observation.derived_features["last_action"] = "ACTION1"
        ctx.observation.derived_features["primary_dynamics"] = "navigation"

        second = _consensus_context_signals(ctx)
        self.assertEqual(second, ("ACTION6", "beam"))

    def _mock_context(
        self,
        legal: list[str] | None = None,
        derived_features: dict | None = None,
        board: list[list[int]] | None = None,
    ) -> ExecutiveContext:
        obs = MagicMock(spec=Observation)
        obs.legal_actions = legal or ["ACTION1"]
        obs.step_index = 0
        obs.game_state = "PLAYING"
        obs.derived_features = derived_features or {}
        obs.image_input = MagicMock(board=board) if board is not None else None
        obs.player_position = None
        ctx = MagicMock(spec=ExecutiveContext)
        ctx.observation = obs
        ctx.memory = MagicMock()
        ctx.hyperon = MagicMock()
        ctx.multi_agent_frame_cache = {}
        return ctx
