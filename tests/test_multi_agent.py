from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from agi_core.config import RuntimeConfig
from agi_core.multi_agent import (
    AgentSupervisor,
    ConsensusModule,
    ExplorationAgent,
    InteractionAgent,
    MovementAgent,
    PuzzleAgent,
)
from agi_core.multi_agent.agents import StrategyAgent, _ordered_mirror_actions, prepare_frame_cache
from agi_core.multi_agent.consensus import _consensus_context_signals
from agi_core.perception.shape_recognition import recognize_shape_recognition
from agi_core.types import ActionDecision, AgentProposal, ExecutiveContext, Observation


class TestMultiAgent(unittest.TestCase):

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
        executive = MagicMock()
        sup = AgentSupervisor(executive=executive, config=config)
        self.assertEqual(len(sup._agents), 4)
        self.assertIn("movement", sup._agent_map)
        self.assertIn("exploration", sup._agent_map)
        self.assertIn("interaction", sup._agent_map)
        self.assertIn("puzzle", sup._agent_map)

    def test_finalize_cognition_only_penalizes_participating_agents(self):
        config = MagicMock(spec=RuntimeConfig)
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

        self.assertEqual(sup._agent_map["movement"]._total_count, 1)
        self.assertEqual(sup._agent_map["movement"]._success_count, 1)
        self.assertEqual(sup._agent_map["interaction"]._total_count, 1)
        self.assertEqual(sup._agent_map["interaction"]._success_count, 0)
        self.assertEqual(sup._agent_map["exploration"]._total_count, 0)
        self.assertEqual(sup._agent_map["puzzle"]._total_count, 0)
        executive._finalize_cognition.assert_called_once_with(ctx, decision)

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
