from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from agi_core.storage import (
    import_legacy_learning_json,
    initialize_memory_schema,
    initialize_schema,
    load_memory_bundle,
    record_transition,
    upsert_abstract_memory_rows,
    upsert_training_counts,
)
from agi_core.knowledge.hyperon_bridge import HyperonBridge
from agi_core.image_input import BoardImageInput, compute_board_signature
from agi_core.memory import EpisodicMemory
from agi_core.types import (
    ActionDecision,
    ActionProposal,
    EpisodeRecord,
    GoalState,
    MemoryBundle,
    Observation,
)


class FakeGroundedAtom:
    def __init__(self, value) -> None:
        self.value = value

    def get_object(self):
        return self.value


class FakeSymbolAtom:
    def __init__(self, name: str) -> None:
        self._name = name

    def get_name(self) -> str:
        return self._name


class FakeVariableAtom(FakeSymbolAtom):
    pass


class FakeExpressionAtom:
    def __init__(self, *children) -> None:
        self._children = list(children)

    def get_children(self):
        return list(self._children)


class FakeEnvironment:
    @staticmethod
    def custom_env(**kwargs):
        return kwargs


class FakeRuntime:
    def __init__(self) -> None:
        self.programs: list[str] = []
        self.reset_calls = 0

    def run(self, program: str):
        self.programs.append(program)
        query = program.strip()
        if not query.startswith("!("):
            return []
        if query == "!(reset-required GAME_OVER)":
            return [[FakeGroundedAtom(True)]]
        if query == "!(goal-hint ACTIVE)":
            return [[FakeSymbolAtom("advance-level")]]
        if query == "!(stagnation-hint True)":
            return [[FakeSymbolAtom("break-stall")]]
        if query == "!(symbolic-prior ACTION1)":
            return [[FakeGroundedAtom(0.72)]]
        if query == "!(symbolic-prior ACTION2)":
            return [[FakeGroundedAtom(0.68)]]
        return []

    def reset(self) -> None:
        self.reset_calls += 1


class FakeHyperonModule:
    SymbolAtom = FakeSymbolAtom
    VariableAtom = FakeVariableAtom
    ExpressionAtom = FakeExpressionAtom
    GroundedAtom = FakeGroundedAtom
    Environment = FakeEnvironment


def make_observation(**overrides) -> Observation:
    payload = {
        "game_id": "ls20",
        "step_index": 1,
        "game_state": "ACTIVE",
        "levels_completed": 0,
        "last_action": "ACTION1",
        "legal_actions": ["ACTION1", "ACTION2"],
    }
    payload.update(overrides)
    return Observation(**payload)


class HyperonBridgeTests(unittest.TestCase):
    def _persist_drained_records(self, bridge: HyperonBridge, memory: EpisodicMemory) -> None:
        for record in memory.drain_new_records():
            bridge.record_transition(record)

    def test_disabled_bridge_uses_shadow_rules(self) -> None:
        bridge = HyperonBridge(enabled=False)
        memory = EpisodicMemory()
        observation = make_observation()

        bridge.sync(
            observation,
            memory,
            [GoalState(name="finish-episode", target_state="WIN")],
        )

        self.assertFalse(bridge.available)
        self.assertEqual(
            bridge.run_rules("goal-hints"),
            [{"subgoal": "advance-level", "confidence": 0.70}],
        )

    def test_shadow_goal_hints_prefers_board_inspection_when_active_cells_exist(self) -> None:
        bridge = HyperonBridge(enabled=False)
        memory = EpisodicMemory()
        board = [[0, 0, 0], [0, 4, 4], [0, 4, 4]]
        signature = compute_board_signature(board)
        observation = make_observation(
            legal_actions=["ACTION1", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=signature,
                source_attr="frame",
            ),
            derived_features={
                "frame_signature": signature,
                "board_active_cells": 4,
                "board_default_probe": {"x": 2, "y": 2},
                "board_source_attr": "frame",
            },
        )

        bridge.sync(
            observation,
            memory,
            [GoalState(name="finish-episode", target_state="WIN")],
        )

        self.assertEqual(
            bridge.run_rules("goal-hints"),
            [{"subgoal": "inspect-active-board", "confidence": 0.74}],
        )

    def test_shadow_goal_hints_reuse_working_memory_break_stall_subgoal(self) -> None:
        class BreakStallMemory(EpisodicMemory):
            def working_memory_snapshot(self):
                return {
                    "active_subgoal": "break-stall",
                    "recent_failures": [{"action_name": "ACTION1", "outcome": "stalled"}],
                }

        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-memory-break-stall"},
        )

        bridge.sync(
            observation,
            BreakStallMemory(),
            [GoalState(name="finish-episode", target_state="WIN")],
        )

        self.assertEqual(
            bridge.run_rules("goal-hints"),
            [{"subgoal": "break-stall", "confidence": 0.73}],
        )

    def test_game_goal_hints_puzzle_board_type(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            step_index=1, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"],
            derived_features={"frame_signature": "sig-gg-puzzle"},
        )
        bridge.sync(observation, EpisodicMemory(), [])
        bridge.facts.setdefault("episode", {})["board_type"] = "puzzle"
        hints = bridge.run_rules("game-goal-hints")
        self.assertTrue(len(hints) >= 1)
        self.assertEqual(hints[0].get("subgoal"), "explore-board")
        self.assertGreaterEqual(hints[0].get("confidence", 0), 0.7)

    def test_game_goal_hints_platformer_board_type(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            step_index=1, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"],
            derived_features={"frame_signature": "sig-gg-platformer"},
        )
        bridge.sync(observation, EpisodicMemory(), [])
        bridge.facts.setdefault("episode", {})["board_type"] = "platformer"
        hints = bridge.run_rules("game-goal-hints")
        self.assertTrue(len(hints) >= 1)
        self.assertEqual(hints[0].get("subgoal"), "navigate-to-goal")

    def test_game_stagnation_puzzle_board_type_lower_threshold(self) -> None:
        bridge = HyperonBridge(enabled=False)
        memory = EpisodicMemory()
        observation = make_observation(
            step_index=5, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-gs-puzzle"},
        )
        bridge.sync(observation, memory, [])
        bridge.facts.setdefault("episode", {})["board_type"] = "puzzle"
        bridge.facts["memory"]["stagnation_steps"] = 2
        result = bridge.run_rules("game-stagnation")
        self.assertTrue(len(result) >= 1)
        self.assertEqual(result[0].get("flag"), "stagnation")

    def test_game_stagnation_unknown_board_type_default_threshold(self) -> None:
        bridge = HyperonBridge(enabled=False)
        memory = EpisodicMemory()
        observation = make_observation(
            step_index=5, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-gs-unknown"},
        )
        bridge.sync(observation, memory, [])
        bridge.facts.setdefault("episode", {})["board_type"] = "unknown"
        bridge.facts["memory"]["stagnation_steps"] = 2
        result = bridge.run_rules("game-stagnation")
        # unknown threshold is 3, so 2 steps should NOT trigger stagnation
        self.assertEqual(len(result), 0)

    def test_game_reset_needed_ar25_game_over(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            step_index=1, game_state="GAME_OVER", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION5", "RESET"],
            derived_features={"frame_signature": "sig-grn-ar25"},
        )
        observation.game_id = "ar25"
        bridge.sync(observation, EpisodicMemory(), [])
        result = bridge.run_rules("game-reset-needed")
        self.assertTrue(len(result) >= 1)
        self.assertEqual(result[0].get("action_name"), "RESET")

    def test_game_reset_needed_ar25_active_no_reset(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            step_index=1, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION5", "RESET"],
            derived_features={"frame_signature": "sig-grn-ar25-active"},
        )
        observation.game_id = "ar25"
        bridge.sync(observation, EpisodicMemory(), [])
        result = bridge.run_rules("game-reset-needed")
        self.assertEqual(len(result), 0)

    def test_beam_strategy_ar25_level_1(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            step_index=1, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            derived_features={"frame_signature": "sig-bs-ar25-l1"},
        )
        observation.game_id = "ar25"
        bridge.sync(observation, EpisodicMemory(), [])
        # levels_completed=0 means level 1.
        bridge.facts.setdefault("episode", {})["levels_completed"] = 0
        result = bridge.run_rules("beam-strategy")
        self.assertTrue(len(result) >= 1)
        self.assertEqual(result[0].get("direction"), "down")

    def test_beam_strategy_ar25_level_2(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            step_index=1, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            derived_features={"frame_signature": "sig-bs-ar25-l2"},
        )
        observation.game_id = "ar25"
        bridge.sync(observation, EpisodicMemory(), [])
        bridge.facts.setdefault("episode", {})["levels_completed"] = 1
        result = bridge.run_rules("beam-strategy")
        self.assertTrue(len(result) >= 1)
        self.assertEqual(result[0].get("direction"), "right")

    def test_board_explore_bias_puzzle_vertical(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            step_index=1, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            derived_features={"frame_signature": "sig-beb-puzzle"},
        )
        bridge.sync(observation, EpisodicMemory(), [])
        bridge.facts.setdefault("episode", {})["board_type"] = "puzzle"
        result = bridge.run_rules("board-explore-bias")
        self.assertEqual(result[0].get("axis"), "vertical")

    def test_board_explore_bias_platformer_horizontal(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            step_index=1, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            derived_features={"frame_signature": "sig-beb-platformer"},
        )
        bridge.sync(observation, EpisodicMemory(), [])
        bridge.facts.setdefault("episode", {})["board_type"] = "platformer"
        result = bridge.run_rules("board-explore-bias")
        self.assertEqual(result[0].get("axis"), "horizontal")

    def test_board_interact_range_puzzle_wider(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            step_index=1, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"],
            derived_features={"frame_signature": "sig-bir-puzzle"},
        )
        bridge.sync(observation, EpisodicMemory(), [])
        bridge.facts.setdefault("episode", {})["board_type"] = "puzzle"
        result = bridge.run_rules("board-interact-range")
        self.assertEqual(result[0].get("range"), 4)

    def test_board_move_bias_puzzle_down(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            step_index=1, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            derived_features={"frame_signature": "sig-bmb-puzzle"},
        )
        bridge.sync(observation, EpisodicMemory(), [])
        bridge.facts.setdefault("episode", {})["board_type"] = "puzzle"
        result = bridge.run_rules("board-move-bias")
        self.assertEqual(result[0].get("direction"), "down")

    def test_board_move_bias_platformer_right(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            step_index=1, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            derived_features={"frame_signature": "sig-bmb-platformer"},
        )
        bridge.sync(observation, EpisodicMemory(), [])
        bridge.facts.setdefault("episode", {})["board_type"] = "platformer"
        result = bridge.run_rules("board-move-bias")
        self.assertEqual(result[0].get("direction"), "right")

    def test_board_beam_priority_puzzle_high(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            step_index=1, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            derived_features={"frame_signature": "sig-bbp-puzzle"},
        )
        bridge.sync(observation, EpisodicMemory(), [])
        bridge.facts.setdefault("episode", {})["board_type"] = "puzzle"
        result = bridge.run_rules("board-beam-priority")
        self.assertEqual(result[0].get("priority"), "high")

    def test_board_beam_priority_platformer_low(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            step_index=1, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            derived_features={"frame_signature": "sig-bbp-platformer"},
        )
        bridge.sync(observation, EpisodicMemory(), [])
        bridge.facts.setdefault("episode", {})["board_type"] = "platformer"
        result = bridge.run_rules("board-beam-priority")
        self.assertEqual(result[0].get("priority"), "low")

    def test_board_switch_urgency_puzzle_high(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            step_index=1, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-bsu-puzzle"},
        )
        bridge.sync(observation, EpisodicMemory(), [])
        bridge.facts.setdefault("episode", {})["board_type"] = "puzzle"
        result = bridge.run_rules("board-switch-urgency")
        self.assertEqual(result[0].get("urgency"), "high")

    def test_board_switch_urgency_platformer_low(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            step_index=1, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-bsu-platformer"},
        )
        bridge.sync(observation, EpisodicMemory(), [])
        bridge.facts.setdefault("episode", {})["board_type"] = "platformer"
        result = bridge.run_rules("board-switch-urgency")
        self.assertEqual(result[0].get("urgency"), "low")

    def test_board_path_preference_puzzle_direct(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            step_index=1, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-bpp-puzzle"},
        )
        bridge.sync(observation, EpisodicMemory(), [])
        bridge.facts.setdefault("episode", {})["board_type"] = "puzzle"
        result = bridge.run_rules("board-path-preference")
        self.assertEqual(result[0].get("preference"), "direct")

    def test_board_path_preference_platformer_safe(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            step_index=1, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-bpp-platformer"},
        )
        bridge.sync(observation, EpisodicMemory(), [])
        bridge.facts.setdefault("episode", {})["board_type"] = "platformer"
        result = bridge.run_rules("board-path-preference")
        self.assertEqual(result[0].get("preference"), "safe")

    def test_board_retry_tolerance_puzzle_low(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            step_index=1, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-brt-puzzle"},
        )
        bridge.sync(observation, EpisodicMemory(), [])
        bridge.facts.setdefault("episode", {})["board_type"] = "puzzle"
        result = bridge.run_rules("board-retry-tolerance")
        self.assertEqual(result[0].get("tolerance"), 2)

    def test_board_retry_tolerance_platformer_high(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            step_index=1, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-brt-platformer"},
        )
        bridge.sync(observation, EpisodicMemory(), [])
        bridge.facts.setdefault("episode", {})["board_type"] = "platformer"
        result = bridge.run_rules("board-retry-tolerance")
        self.assertEqual(result[0].get("tolerance"), 5)

    def test_board_goal_proximity_puzzle_tight(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            step_index=1, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-bgp-puzzle"},
        )
        bridge.sync(observation, EpisodicMemory(), [])
        bridge.facts.setdefault("episode", {})["board_type"] = "puzzle"
        result = bridge.run_rules("board-goal-proximity")
        self.assertEqual(result[0].get("proximity"), 1)

    def test_board_goal_proximity_platformer_loose(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            step_index=1, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-bgp-platformer"},
        )
        bridge.sync(observation, EpisodicMemory(), [])
        bridge.facts.setdefault("episode", {})["board_type"] = "platformer"
        result = bridge.run_rules("board-goal-proximity")
        self.assertEqual(result[0].get("proximity"), 3)
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            step_index=1, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            derived_features={"frame_signature": "sig-bbp-platformer"},
        )
        bridge.sync(observation, EpisodicMemory(), [])
        bridge.facts.setdefault("episode", {})["board_type"] = "platformer"
        result = bridge.run_rules("board-beam-priority")
        self.assertEqual(result[0].get("priority"), "low")

    def test_board_entity_count_puzzle_multiple(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(step_index=1, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION5"], derived_features={"frame_signature": "sig-bec-puzzle"})
        bridge.sync(observation, EpisodicMemory(), [])
        bridge.facts.setdefault("episode", {})["board_type"] = "puzzle"
        result = bridge.run_rules("board-entity-count")
        self.assertEqual(result[0].get("count"), "multiple")

    def test_board_entity_count_platformer_single(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(step_index=1, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION5"], derived_features={"frame_signature": "sig-bec-platformer"})
        bridge.sync(observation, EpisodicMemory(), [])
        bridge.facts.setdefault("episode", {})["board_type"] = "platformer"
        result = bridge.run_rules("board-entity-count")
        self.assertEqual(result[0].get("count"), "single")

    def test_board_gravity_enabled_platformer_true(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(step_index=1, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"], derived_features={"frame_signature": "sig-bge-platformer"})
        bridge.sync(observation, EpisodicMemory(), [])
        bridge.facts.setdefault("episode", {})["board_type"] = "platformer"
        result = bridge.run_rules("board-gravity-enabled")
        self.assertEqual(result[0].get("gravity"), "true")

    def test_board_gravity_enabled_puzzle_false(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(step_index=1, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"], derived_features={"frame_signature": "sig-bge-puzzle"})
        bridge.sync(observation, EpisodicMemory(), [])
        bridge.facts.setdefault("episode", {})["board_type"] = "puzzle"
        result = bridge.run_rules("board-gravity-enabled")
        self.assertEqual(result[0].get("gravity"), "false")

    def test_board_horizontal_scroll_platformer_true(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(step_index=1, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"], derived_features={"frame_signature": "sig-bhs-platformer"})
        bridge.sync(observation, EpisodicMemory(), [])
        bridge.facts.setdefault("episode", {})["board_type"] = "platformer"
        result = bridge.run_rules("board-horizontal-scroll")
        self.assertEqual(result[0].get("scroll"), "true")

    def test_board_horizontal_scroll_puzzle_false(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(step_index=1, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"], derived_features={"frame_signature": "sig-bhs-puzzle"})
        bridge.sync(observation, EpisodicMemory(), [])
        bridge.facts.setdefault("episode", {})["board_type"] = "puzzle"
        result = bridge.run_rules("board-horizontal-scroll")
        self.assertEqual(result[0].get("scroll"), "false")

    def test_board_exit_type_puzzle_action(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(step_index=1, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION5"], derived_features={"frame_signature": "sig-bet-puzzle"})
        bridge.sync(observation, EpisodicMemory(), [])
        bridge.facts.setdefault("episode", {})["board_type"] = "puzzle"
        result = bridge.run_rules("board-exit-type")
        self.assertEqual(result[0].get("exit"), "action")

    def test_board_exit_type_platformer_position(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(step_index=1, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION5"], derived_features={"frame_signature": "sig-bet-platformer"})
        bridge.sync(observation, EpisodicMemory(), [])
        bridge.facts.setdefault("episode", {})["board_type"] = "platformer"
        result = bridge.run_rules("board-exit-type")
        self.assertEqual(result[0].get("exit"), "position")

    def test_sync_records_live_board_summary_without_persisting_board(self) -> None:
        bridge = HyperonBridge(enabled=False)
        memory = EpisodicMemory()
        board = [[7, 8], [9, 0]]
        signature = compute_board_signature(board)
        observation = make_observation(
            image_input=BoardImageInput(
                board=board,
                height=2,
                width=2,
                frame_signature=signature,
                source_attr="grid",
            ),
            derived_features={"frame_signature": signature},
        )

        bridge.sync(
            observation,
            memory,
            [GoalState(name="finish-episode", target_state="WIN")],
        )

        episode = bridge.facts["episode"]
        self.assertTrue(episode["has_image_input"])
        self.assertEqual(episode["board_height"], 2)
        self.assertEqual(episode["board_width"], 2)
        self.assertEqual(episode["board_active_cells"], 3)
        self.assertEqual(episode["board_default_probe"], {"x": 0, "y": 0})
        self.assertEqual(episode["board_source_attr"], "grid")
        self.assertNotIn("board", episode)
        self.assertIn("working_memory", bridge.facts["memory"])

    def test_sync_uses_derived_board_runtime_summary_fields(self) -> None:
        bridge = HyperonBridge(enabled=False)
        memory = EpisodicMemory()
        board = [[0, 0, 0], [0, 7, 7], [0, 7, 7]]
        signature = compute_board_signature(board)
        observation = make_observation(
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=signature,
                source_attr="frame",
            ),
            derived_features={
                "frame_signature": signature,
                "board_active_cells": 4,
                "board_default_probe": {"x": 2, "y": 2},
                "board_source_attr": "frame",
            },
        )

        bridge.sync(
            observation,
            memory,
            [GoalState(name="finish-episode", target_state="WIN")],
        )

        episode = bridge.facts["episode"]
        self.assertEqual(episode["board_active_cells"], 4)
        self.assertEqual(episode["board_default_probe"], {"x": 2, "y": 2})
        self.assertEqual(episode["board_source_attr"], "frame")

    def test_probe_failure_falls_back_without_breaking_rules(self) -> None:
        bridge = HyperonBridge()
        memory = EpisodicMemory()
        observation = make_observation(game_state="GAME_OVER", legal_actions=["RESET"])

        with patch.object(bridge, "_probe_runtime", return_value=(False, "denied")):
            bridge.sync(
                observation,
                memory,
                [GoalState(name="finish-episode", target_state="WIN")],
            )

        self.assertFalse(bridge.available)
        self.assertEqual(bridge.last_error, "denied")
        self.assertEqual(
            bridge.run_rules("reset-needed"),
            [
                {
                    "action_name": "RESET",
                    "confidence": 0.99,
                    "reason": "symbolic reset guard",
                }
            ],
        )

    def test_runtime_queries_use_hyperon_results_when_available(self) -> None:
        bridge = HyperonBridge()
        bridge.backend = FakeHyperonModule
        fake_runtime = FakeRuntime()
        memory = EpisodicMemory()
        observation = make_observation()

        with patch.object(bridge, "_probe_runtime", return_value=(True, None)):
            with patch.object(bridge, "_create_runtime", return_value=fake_runtime):
                bridge.sync(
                    observation,
                    memory,
                    [GoalState(name="finish-episode", target_state="WIN")],
                )

        self.assertTrue(bridge.available)
        self.assertEqual(
            bridge.run_rules("goal-hints"),
            [{"subgoal": "advance-level", "confidence": 0.72}],
        )
        self.assertEqual(
            bridge.run_rules("action-priors"),
            [
                {
                    "action_name": "ACTION1",
                    "confidence": 0.72,
                    "reason": "hyperon symbolic prior",
                },
                {
                    "action_name": "ACTION2",
                    "confidence": 0.68,
                    "reason": "hyperon symbolic prior",
                },
            ],
        )
        self.assertIsNotNone(bridge.current_memory_bundle().working_memory)

    def test_runtime_goal_hints_fall_back_to_board_inspection_when_runtime_has_no_hint(self) -> None:
        bridge = HyperonBridge()
        bridge.backend = FakeHyperonModule
        memory = EpisodicMemory()
        board = [[0, 0, 0], [0, 4, 4], [0, 4, 4]]
        signature = compute_board_signature(board)
        observation = make_observation(
            legal_actions=["ACTION1", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=signature,
                source_attr="frame",
            ),
            derived_features={
                "frame_signature": signature,
                "board_active_cells": 4,
                "board_default_probe": {"x": 2, "y": 2},
                "board_source_attr": "frame",
            },
        )

        class NoGoalHintRuntime(FakeRuntime):
            def run(self, program: str):
                query = program.strip()
                if query == "!(goal-hint ACTIVE)":
                    return []
                return super().run(program)

        with patch.object(bridge, "_probe_runtime", return_value=(True, None)):
            with patch.object(bridge, "_create_runtime", return_value=NoGoalHintRuntime()):
                bridge.sync(
                    observation,
                    memory,
                    [GoalState(name="finish-episode", target_state="WIN")],
                )

        self.assertEqual(
            bridge.run_rules("goal-hints"),
            [{"subgoal": "inspect-active-board", "confidence": 0.74}],
        )

    def test_runtime_goal_hints_upgrade_generic_hint_to_board_inspection(self) -> None:
        bridge = HyperonBridge()
        bridge.backend = FakeHyperonModule
        memory = EpisodicMemory()
        board = [[0, 0, 0], [0, 4, 4], [0, 4, 4]]
        signature = compute_board_signature(board)
        observation = make_observation(
            legal_actions=["ACTION1", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=signature,
                source_attr="frame",
            ),
            derived_features={
                "frame_signature": signature,
                "board_active_cells": 4,
                "board_default_probe": {"x": 2, "y": 2},
                "board_source_attr": "frame",
            },
        )

        with patch.object(bridge, "_probe_runtime", return_value=(True, None)):
            with patch.object(bridge, "_create_runtime", return_value=FakeRuntime()):
                bridge.sync(
                    observation,
                    memory,
                    [GoalState(name="finish-episode", target_state="WIN")],
                )

        self.assertEqual(
            bridge.run_rules("goal-hints"),
            [{"subgoal": "inspect-active-board", "confidence": 0.74}],
        )

    def test_runtime_goal_hints_reuse_working_memory_break_stall_subgoal(self) -> None:
        class BreakStallMemory(EpisodicMemory):
            def working_memory_snapshot(self):
                return {
                    "active_subgoal": "break-stall",
                    "recent_failures": [{"action_name": "ACTION1", "outcome": "stalled"}],
                }

        bridge = HyperonBridge()
        bridge.backend = FakeHyperonModule
        observation = make_observation(
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-runtime-memory-break-stall"},
        )

        with patch.object(bridge, "_probe_runtime", return_value=(True, None)):
            with patch.object(bridge, "_create_runtime", return_value=FakeRuntime()):
                bridge.sync(
                    observation,
                    BreakStallMemory(),
                    [GoalState(name="finish-episode", target_state="WIN")],
                )

        self.assertEqual(
            bridge.run_rules("goal-hints"),
            [{"subgoal": "break-stall", "confidence": 0.73}],
        )

    def test_runtime_queries_fall_back_to_shadow_for_custom_terminal_states(self) -> None:
        bridge = HyperonBridge(
            skip_probe=True,
            reset_state_names=("READY", "LOSE"),
            win_state_name="DONE",
        )
        bridge.backend = FakeHyperonModule
        fake_runtime = FakeRuntime()

        with patch.object(bridge, "_create_runtime", return_value=fake_runtime):
            bridge.sync(
                make_observation(
                    game_state="DONE",
                    legal_actions=["ACTION1", "ACTION2"],
                    derived_features={"frame_signature": "sig-done"},
                ),
                EpisodicMemory(),
                [GoalState(name="finish-episode", target_state="DONE")],
            )

        self.assertTrue(bridge.available)
        self.assertEqual(
            bridge.run_rules("goal-hints"),
            [{"subgoal": "hold-win", "confidence": 0.95}],
        )

        bridge.sync(
            make_observation(
                step_index=2,
                game_state="LOSE",
                legal_actions=["RESET"],
                derived_features={"frame_signature": "sig-lose"},
            ),
            EpisodicMemory(),
            [GoalState(name="finish-episode", target_state="DONE")],
        )

        self.assertEqual(
            bridge.run_rules("reset-needed"),
            [
                {
                    "action_name": "RESET",
                    "confidence": 0.99,
                    "reason": "symbolic reset guard",
                }
            ],
        )

        bridge.update_proposals(
            [
                ActionProposal(
                    action_name="ACTION1",
                    source="symbolic",
                    confidence=0.72,
                    reason="prior",
                    expected_outcome="progress",
                )
            ]
        )
        self.assertTrue(
            any("(proposal ACTION1 symbolic 0.72 progress)" in program for program in fake_runtime.programs)
        )

    def test_skip_probe_enables_direct_runtime_initialization(self) -> None:
        bridge = HyperonBridge(skip_probe=True)
        bridge.backend = FakeHyperonModule
        fake_runtime = FakeRuntime()
        memory = EpisodicMemory()
        observation = make_observation()

        with patch.object(bridge, "_probe_runtime", side_effect=AssertionError("probe should be skipped")):
            with patch.object(bridge, "_create_runtime", return_value=fake_runtime):
                bridge.sync(
                    observation,
                    memory,
                    [GoalState(name="finish-episode", target_state="WIN")],
                )

        self.assertTrue(bridge.available)
        self.assertIsNone(bridge.last_error)

    def test_sync_reuses_runtime_when_reset_capability_is_available(self) -> None:
        bridge = HyperonBridge(skip_probe=True)
        bridge.backend = FakeHyperonModule
        fake_runtime = FakeRuntime()
        memory = EpisodicMemory()
        create_calls: list[int] = []

        def create_runtime():
            create_calls.append(1)
            return fake_runtime

        with patch.object(bridge, "_create_runtime", side_effect=create_runtime):
            bridge.sync(
                make_observation(),
                memory,
                [GoalState(name="finish-episode", target_state="WIN")],
            )
            bridge.sync(
                make_observation(step_index=2, last_action="ACTION2"),
                memory,
                [GoalState(name="finish-episode", target_state="WIN")],
            )

        self.assertEqual(len(create_calls), 1)
        self.assertGreaterEqual(fake_runtime.reset_calls, 1)

    def test_update_proposals_forces_next_identical_sync_to_clear_runtime_facts(self) -> None:
        bridge = HyperonBridge(skip_probe=True)
        bridge.backend = FakeHyperonModule
        fake_runtime = FakeRuntime()
        memory = EpisodicMemory()

        with patch.object(bridge, "_create_runtime", return_value=fake_runtime):
            observation = make_observation()
            bridge.sync(
                observation,
                memory,
                [GoalState(name="finish-episode", target_state="WIN")],
            )
            bridge.update_proposals(
                [
                    ActionProposal(
                        action_name="ACTION1",
                        source="symbolic",
                        confidence=0.72,
                        reason="prior",
                        expected_outcome="progress",
                    )
                ]
            )
            reset_calls_before = fake_runtime.reset_calls
            bridge.sync(
                observation,
                memory,
                [GoalState(name="finish-episode", target_state="WIN")],
            )

        self.assertGreater(fake_runtime.reset_calls, reset_calls_before)

    def test_persistent_learning_updates_action_priors_in_sqlite_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            learning_path = Path(tmpdir) / "arc_learned.json"
            memory_db_path = Path(tmpdir) / "memory.sqlite3"
            bridge = HyperonBridge(
                enabled=False,
                learning_path=learning_path,
                memory_db_path=memory_db_path,
                trained_priors_db_path=memory_db_path,
            )
            memory = EpisodicMemory()
            memory.bind_game("ls20")

            before = make_observation(step_index=0, last_action=None)
            after = make_observation(
                step_index=1,
                levels_completed=1,
                last_action="ACTION1",
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name="ACTION1",
                    source="symbolic",
                    confidence=0.7,
                    reason="test",
                )
            )
            memory.note_observation(after)
            self._persist_drained_records(bridge, memory)

            bridge.sync(
                after,
                memory,
                [GoalState(name="finish-episode", target_state="WIN")],
            )

            priors = bridge.run_rules("action-priors")
            self.assertEqual(priors[0]["action_name"], "ACTION1")
            self.assertGreater(priors[0]["confidence"], priors[1]["confidence"])
            bundle = load_memory_bundle(
                memory_db_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature="",
                subgoal=None,
                legal_actions=["ACTION1", "ACTION2"],
            )
            self.assertEqual(bundle["action_stats"]["ACTION1"]["progress"], 1)

            second_bridge = HyperonBridge(
                enabled=False,
                learning_path=learning_path,
                memory_db_path=memory_db_path,
                trained_priors_db_path=memory_db_path,
            )
            second_bridge.sync(
                make_observation(step_index=2, last_action="ACTION1"),
                EpisodicMemory(),
                [GoalState(name="finish-episode", target_state="WIN")],
            )
            self.assertEqual(
                second_bridge.current_memory_bundle().action_stats["ACTION1"]["progress"],
                1,
            )

    def test_state_and_subgoal_learning_bias_current_symbolic_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            learning_path = Path(tmpdir) / "arc_learned.json"
            memory_db_path = Path(tmpdir) / "memory.sqlite3"
            bridge = HyperonBridge(
                enabled=False,
                learning_path=learning_path,
                memory_db_path=memory_db_path,
                trained_priors_db_path=memory_db_path,
            )
            memory = EpisodicMemory()
            memory.bind_game("ls20")

            before = make_observation(
                step_index=0,
                game_state="ACTIVE",
                last_action=None,
            )
            after = make_observation(
                step_index=1,
                game_state="ACTIVE",
                levels_completed=1,
                last_action="ACTION2",
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name="ACTION2",
                    source="symbolic",
                    confidence=0.7,
                    reason="test",
                    subgoal="break-stall",
                )
            )
            memory.note_observation(after)
            self._persist_drained_records(bridge, memory)

            bridge.sync(
                after,
                memory,
                [GoalState(name="finish-episode", target_state="WIN")],
            )

            priors = bridge.run_rules("action-priors:break-stall")
            self.assertEqual(priors[0]["action_name"], "ACTION2")
            bundle = load_memory_bundle(
                memory_db_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature="",
                subgoal="break-stall",
                legal_actions=["ACTION1", "ACTION2"],
            )
            self.assertEqual(bundle["state_action_stats"]["ACTION2"]["progress"], 1)
            self.assertEqual(bundle["subgoal_action_stats"]["ACTION2"]["progress"], 1)

    def test_frame_signature_learning_biases_matching_symbolic_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            learning_path = Path(tmpdir) / "arc_learned.json"
            memory_db_path = Path(tmpdir) / "memory.sqlite3"
            bridge = HyperonBridge(
                enabled=False,
                learning_path=learning_path,
                memory_db_path=memory_db_path,
                trained_priors_db_path=memory_db_path,
            )
            memory = EpisodicMemory()
            memory.bind_game("ls20")

            before = make_observation(
                step_index=0,
                game_state="ACTIVE",
                last_action=None,
                derived_features={"frame_signature": "sig-live"},
            )
            after = make_observation(
                step_index=1,
                game_state="ACTIVE",
                levels_completed=1,
                last_action="ACTION2",
                derived_features={"frame_signature": "sig-after"},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name="ACTION2",
                    source="symbolic",
                    confidence=0.7,
                    reason="test",
                )
            )
            memory.note_observation(after)
            self._persist_drained_records(bridge, memory)

            bridge.sync(
                make_observation(
                    step_index=2,
                    game_state="ACTIVE",
                    last_action="ACTION2",
                    derived_features={"frame_signature": "sig-live"},
                ),
                memory,
                [GoalState(name="finish-episode", target_state="WIN")],
            )

            priors = bridge.run_rules("action-priors")
            self.assertEqual(priors[0]["action_name"], "ACTION2")
            bundle = load_memory_bundle(
                memory_db_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature="sig-live",
                subgoal=None,
                legal_actions=["ACTION1", "ACTION2"],
            )
            self.assertEqual(bundle["frame_action_stats"]["ACTION2"]["progress"], 1)

    def test_state_scoped_basin_action_stats_prevent_cross_state_frame_contamination(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            learning_path = Path(tmpdir) / "arc_learned.json"
            memory_db_path = Path(tmpdir) / "memory.sqlite3"
            bridge = HyperonBridge(
                enabled=False,
                learning_path=learning_path,
                memory_db_path=memory_db_path,
                trained_priors_db_path=memory_db_path,
            )
            memory = EpisodicMemory()
            memory.bind_game("ls20")

            stalled_before = make_observation(
                step_index=0,
                game_state="NOT_FINISHED",
                last_action=None,
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-shared"},
            )
            stalled_after = make_observation(
                step_index=1,
                game_state="NOT_FINISHED",
                last_action="ACTION1",
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-shared"},
            )
            memory.note_observation(stalled_before)
            memory.note_decision(
                ActionDecision(
                    action_name="ACTION1",
                    source="symbolic",
                    confidence=0.7,
                    reason="stalled-basin",
                )
            )
            memory.note_observation(stalled_after)
            self._persist_drained_records(bridge, memory)

            progress_before = make_observation(
                step_index=2,
                game_state="ACTIVE",
                last_action=None,
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-shared"},
            )
            progress_after = make_observation(
                step_index=3,
                game_state="ACTIVE",
                levels_completed=1,
                last_action="ACTION1",
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-next"},
            )
            memory.note_observation(progress_before)
            memory.note_decision(
                ActionDecision(
                    action_name="ACTION1",
                    source="symbolic",
                    confidence=0.7,
                    reason="progress-other-state",
                )
            )
            memory.note_observation(progress_after)
            self._persist_drained_records(bridge, memory)

            observation = make_observation(
                step_index=4,
                game_state="NOT_FINISHED",
                last_action="ACTION1",
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-shared"},
            )
            bridge.sync(
                observation,
                memory,
                [GoalState(name="finish-episode", target_state="WIN")],
            )

            bundle = bridge.current_memory_bundle()
            self.assertEqual(bundle.basin_action_stats["ACTION1"]["stalled"], 1)
            self.assertEqual(bundle.frame_action_stats["ACTION1"]["progress"], 1)
            priors = bridge.run_rules("action-priors")
            self.assertEqual(priors[0]["action_name"], "ACTION2")

    def test_symbolic_priors_prefer_state_and_subgoal_over_coarse_frame_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            learning_path = Path(tmpdir) / "arc_learned.json"
            memory_db_path = Path(tmpdir) / "memory.sqlite3"
            bridge = HyperonBridge(
                enabled=False,
                learning_path=learning_path,
                memory_db_path=memory_db_path,
                trained_priors_db_path=memory_db_path,
            )
            memory = EpisodicMemory()
            memory.bind_game("ls20")

            coarse_frame_before = make_observation(
                step_index=0,
                game_state="ACTIVE",
                last_action=None,
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-shared"},
            )
            coarse_frame_after = make_observation(
                step_index=1,
                game_state="ACTIVE",
                levels_completed=1,
                last_action="ACTION1",
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-next"},
            )
            memory.note_observation(coarse_frame_before)
            memory.note_decision(
                ActionDecision(
                    action_name="ACTION1",
                    source="symbolic",
                    confidence=0.7,
                    reason="coarse-frame-progress",
                    subgoal="advance-level",
                )
            )
            memory.note_observation(coarse_frame_after)
            self._persist_drained_records(bridge, memory)

            focused_before = make_observation(
                step_index=2,
                game_state="NOT_FINISHED",
                last_action=None,
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-other"},
            )
            focused_after = make_observation(
                step_index=3,
                game_state="NOT_FINISHED",
                levels_completed=1,
                last_action="ACTION2",
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-other-next"},
            )
            memory.note_observation(focused_before)
            memory.note_decision(
                ActionDecision(
                    action_name="ACTION2",
                    source="symbolic",
                    confidence=0.7,
                    reason="state-subgoal-progress",
                    subgoal="break-stall",
                )
            )
            memory.note_observation(focused_after)
            self._persist_drained_records(bridge, memory)

            observation = make_observation(
                step_index=4,
                game_state="NOT_FINISHED",
                last_action="ACTION2",
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-shared"},
            )
            bridge.sync(
                observation,
                memory,
                [GoalState(name="finish-episode", target_state="WIN")],
            )

            priors = bridge.run_rules("action-priors:break-stall")
            self.assertEqual(priors[0]["action_name"], "ACTION2")

    def test_hyperon_learned_state_facts_use_state_scope_not_basin_scope(self) -> None:
        bridge = HyperonBridge(enabled=False)
        bridge.facts["episode"] = {
            "game_state": "NOT_FINISHED",
            "frame_signature": "sig-target",
        }
        learned = bridge._memory_bundle_as_learned(
            {
                "recent_records": [],
                "action_stats": {},
                "basin_action_stats": {
                    "ACTION1": {"progress": 0, "failed": 0, "stalled": 1, "total": 1}
                },
                "state_action_stats": {
                    "ACTION2": {"progress": 1, "failed": 0, "stalled": 0, "total": 1}
                },
                "frame_action_stats": {
                    "ACTION1": {"progress": 0, "failed": 0, "stalled": 1, "total": 1}
                },
                "subgoal_action_stats": {},
            }
        )

        self.assertEqual(
            learned["states"]["NOT_FINISHED"]["actions"],
            {"ACTION2": {"progress": 1, "failed": 0, "stalled": 0, "total": 1}},
        )
        self.assertNotEqual(
            learned["states"]["NOT_FINISHED"]["actions"],
            {"ACTION1": {"progress": 0, "failed": 0, "stalled": 1, "total": 1}},
        )

    def test_repeated_stalled_frame_signature_is_downranked(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            learning_path = Path(tmpdir) / "arc_learned.json"
            learned_payload = {
                "games": {
                    "ls20": {
                        "actions": {},
                        "states": {},
                        "subgoals": {},
                        "frame_signatures": {
                            "sig-loop": {
                                "actions": {
                                    "ACTION1": {
                                        "progress": 0,
                                        "failed": 0,
                                        "stalled": 3,
                                        "total": 3,
                                        "last_outcome": "stalled",
                                    }
                                }
                            }
                        },
                        "records_seen": 3,
                    }
                }
            }
            learning_path.write_text(json.dumps(learned_payload), encoding="utf-8")
            memory_db_path = Path(tmpdir) / "memory.sqlite3"
            bridge = HyperonBridge(
                enabled=False,
                learning_path=learning_path,
                memory_db_path=memory_db_path,
                trained_priors_db_path=memory_db_path,
            )
            memory = EpisodicMemory()
            before = make_observation(
                step_index=0,
                game_state="NOT_FINISHED",
                last_action=None,
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-loop"},
            )
            after = make_observation(
                step_index=1,
                game_state="NOT_FINISHED",
                last_action="ACTION1",
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-loop"},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name="ACTION1",
                    source="symbolic",
                    confidence=0.7,
                    reason="stalled-loop",
                )
            )
            memory.note_observation(after)
            self._persist_drained_records(bridge, memory)

            observation = make_observation(
                step_index=2,
                game_state="NOT_FINISHED",
                last_action="ACTION1",
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-loop"},
            )
            bridge.sync(
                observation,
                memory,
                [GoalState(name="finish-episode", target_state="WIN")],
            )

            priors = bridge.run_rules("action-priors")
            self.assertEqual(priors[0]["action_name"], "ACTION2")

    def test_repeated_stalled_state_without_frame_signature_is_downranked(self) -> None:
        bridge = HyperonBridge(enabled=False)
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        for index in range(3):
            before = make_observation(
                step_index=index,
                game_state="NOT_FINISHED",
                last_action=None,
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={},
            )
            after = make_observation(
                step_index=index + 1,
                game_state="NOT_FINISHED",
                last_action="ACTION1",
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name="ACTION1",
                    source="symbolic",
                    confidence=0.7,
                    reason="state-only-stalled-loop",
                )
            )
            memory.note_observation(after)

        observation = make_observation(
            step_index=4,
            game_state="NOT_FINISHED",
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={},
        )
        bridge.sync(
            observation,
            memory,
            [GoalState(name="finish-episode", target_state="WIN")],
        )

        priors = bridge.run_rules("action-priors")
        self.assertEqual(priors[0]["action_name"], "ACTION2")

    def test_stagnation_novelty_prefers_untried_action_in_current_frame_context(self) -> None:
        bridge = HyperonBridge(enabled=False)
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        for index in range(4):
            before = make_observation(
                step_index=index,
                game_state="NOT_FINISHED",
                last_action=None,
                legal_actions=["ACTION1", "ACTION2", "ACTION3"],
                derived_features={"frame_signature": "sig-novel"},
            )
            after = make_observation(
                step_index=index + 1,
                game_state="NOT_FINISHED",
                last_action="ACTION3",
                legal_actions=["ACTION1", "ACTION2", "ACTION3"],
                derived_features={"frame_signature": "sig-novel"},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name="ACTION3",
                    source="symbolic",
                    confidence=0.6,
                    reason="repeat-stall",
                    subgoal="break-stall",
                )
            )
            memory.note_observation(after)

        observation = make_observation(
            step_index=5,
            game_state="NOT_FINISHED",
            last_action="ACTION3",
            legal_actions=["ACTION1", "ACTION2", "ACTION3"],
            derived_features={"frame_signature": "sig-novel"},
        )
        bridge.sync(
            observation,
            memory,
            [GoalState(name="finish-episode", target_state="WIN")],
        )

        priors = bridge.run_rules("action-priors:break-stall")
        self.assertEqual(priors[0]["action_name"], "ACTION1")
        by_name = {row["action_name"]: row for row in priors}
        self.assertGreater(by_name["ACTION1"]["confidence"], by_name["ACTION3"]["confidence"])

    def test_board_probe_prior_prefers_action6_when_active_cells_exist(self) -> None:
        bridge = HyperonBridge(enabled=False)
        memory = EpisodicMemory()
        board = [[0, 0, 0], [0, 5, 5], [0, 5, 5]]
        signature = compute_board_signature(board)
        observation = make_observation(
            step_index=2,
            game_state="NOT_FINISHED",
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=signature,
                source_attr="frame",
            ),
            derived_features={
                "frame_signature": signature,
                "board_active_cells": 4,
                "board_default_probe": {"x": 2, "y": 2},
                "board_source_attr": "frame",
            },
        )

        bridge.sync(
            observation,
            memory,
            [GoalState(name="finish-episode", target_state="WIN")],
        )

        priors = bridge.run_rules("action-priors")
        self.assertEqual(priors[0]["action_name"], "ACTION6")
        by_name = {row["action_name"]: row for row in priors}
        self.assertGreater(by_name["ACTION6"]["confidence"], by_name["ACTION1"]["confidence"])

    def test_board_probe_prior_is_stronger_for_inspect_active_board_subgoal(self) -> None:
        bridge = HyperonBridge(enabled=False)
        memory = EpisodicMemory()
        board = [[0, 0, 0], [0, 5, 5], [0, 5, 5]]
        signature = compute_board_signature(board)
        observation = make_observation(
            step_index=2,
            game_state="NOT_FINISHED",
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION6"],
            image_input=BoardImageInput(
                board=board,
                height=3,
                width=3,
                frame_signature=signature,
                source_attr="frame",
            ),
            derived_features={
                "frame_signature": signature,
                "board_active_cells": 4,
                "board_default_probe": {"x": 2, "y": 2},
                "board_source_attr": "frame",
            },
        )

        bridge.sync(
            observation,
            memory,
            [GoalState(name="finish-episode", target_state="WIN")],
        )

        generic_priors = bridge.run_rules("action-priors")
        inspect_priors = bridge.run_rules("action-priors:inspect-active-board")

        generic_by_name = {row["action_name"]: row for row in generic_priors}
        inspect_by_name = {row["action_name"]: row for row in inspect_priors}
        self.assertEqual(inspect_priors[0]["action_name"], "ACTION6")
        self.assertGreater(
            inspect_by_name["ACTION6"]["confidence"],
            generic_by_name["ACTION6"]["confidence"],
        )

    def test_generic_action_priors_ignore_active_subgoal_specific_trained_priors(self) -> None:
        class ActiveSubgoalMemory(EpisodicMemory):
            def working_memory_snapshot(self):
                return {"active_subgoal": "inspect-active-board"}

        with tempfile.TemporaryDirectory() as tmpdir:
            memory_db_path = Path(tmpdir) / "invalid-memory.sqlite3"
            memory_db_path.write_text("not-a-sqlite-db", encoding="utf-8")
            trained_priors_db_path = Path(tmpdir) / "offline-priors.sqlite3"
            initialize_schema(trained_priors_db_path)
            upsert_training_counts(
                trained_priors_db_path,
                [
                    {
                        "scope": "state",
                        "game_id": "ls20",
                        "state": "ACTIVE",
                        "action_name": "ACTION1",
                        "count": 8,
                    },
                    {
                        "scope": "subgoal",
                        "game_id": "ls20",
                        "subgoal": "inspect-active-board",
                        "action_name": "ACTION6",
                        "count": 11,
                    },
                ],
            )
            bridge = HyperonBridge(
                enabled=False,
                memory_db_path=memory_db_path,
                trained_priors_db_path=trained_priors_db_path,
            )
            memory = ActiveSubgoalMemory()
            memory.bind_game("ls20")
            observation = make_observation(
                step_index=2,
                game_state="ACTIVE",
                last_action="ACTION1",
                legal_actions=["ACTION1", "ACTION6"],
                derived_features={"frame_signature": "sig-neutral-priors"},
            )

            bridge.sync(
                observation,
                memory,
                [GoalState(name="finish-episode", target_state="WIN")],
            )

            generic_priors = bridge.run_rules("action-priors")
            inspect_priors = bridge.run_rules("action-priors:inspect-active-board")

            self.assertEqual(generic_priors[0]["action_name"], "ACTION1")
            self.assertEqual(inspect_priors[0]["action_name"], "ACTION6")

    def test_episode_basin_exhaustion_crushes_symbolic_confidence_after_full_cycle(self) -> None:
        bridge = HyperonBridge(enabled=False)
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        sequence = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
        for index, action_name in enumerate(sequence):
            before = make_observation(
                step_index=index,
                game_state="NOT_FINISHED",
                last_action=None,
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
                derived_features={"frame_signature": "sig-basin"},
            )
            after = make_observation(
                step_index=index + 1,
                game_state="NOT_FINISHED",
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
                derived_features={"frame_signature": "sig-basin"},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="symbolic",
                    confidence=0.6,
                    reason="basin-cycle",
                    subgoal="break-stall",
                )
            )
            memory.note_observation(after)

        observation = make_observation(
            step_index=5,
            game_state="NOT_FINISHED",
            last_action="ACTION4",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
            derived_features={"frame_signature": "sig-basin"},
        )
        bridge.sync(
            observation,
            memory,
            [GoalState(name="finish-episode", target_state="WIN")],
        )

        priors = bridge.run_rules("action-priors:break-stall")
        self.assertTrue(all(float(row["confidence"]) <= 0.30 for row in priors))

    def test_action_priors_demote_runtime_working_memory_failure_action(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            step_index=4,
            game_state="ACTIVE",
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-runtime-prior-failure"},
        )
        bridge.sync(
            observation,
            EpisodicMemory(),
            [GoalState(name="finish-episode", target_state="WIN")],
        )
        bridge._current_memory_bundle = MemoryBundle(
            working_memory={
                "open_failure_modes": [
                    {"action_name": "ACTION1", "subgoal": "break-stall", "outcome": "stalled"}
                ],
                "recent_failures": [
                    {"action_name": "ACTION1", "subgoal": "break-stall", "outcome": "stalled"}
                ],
                "failure_streaks": [
                    {"action_name": "ACTION1", "count": 3}
                ],
            }
        )

        priors = bridge.run_rules("action-priors:break-stall")

        self.assertEqual(priors[0]["action_name"], "ACTION2")

    def test_action_priors_prefer_runtime_working_memory_focus_action(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            step_index=4,
            game_state="ACTIVE",
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-runtime-prior-focus"},
        )
        bridge.sync(
            observation,
            EpisodicMemory(),
            [GoalState(name="finish-episode", target_state="WIN")],
        )
        bridge._current_memory_bundle = MemoryBundle(
            working_memory={
                "semantic_focus_actions": ["ACTION2"],
                "procedural_focus_actions": ["ACTION2"],
            }
        )

        priors = bridge.run_rules("action-priors:break-stall")

        self.assertEqual(priors[0]["action_name"], "ACTION2")

    def test_game_action_priors_ar25_overrides_defaults(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            step_index=1, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"],
            derived_features={"frame_signature": "sig-ar25"},
        )
        observation.game_id = "ar25"
        bridge.sync(observation, EpisodicMemory(), [])
        # Manually set board_type to unknown so game priors dominate.
        bridge.facts.setdefault("episode", {})["board_type"] = "unknown"
        priors = bridge.run_rules("game-action-priors")
        prior_map = {p["action_name"]: p["confidence"] for p in priors}
        self.assertAlmostEqual(prior_map.get("ACTION2", 0), 0.78, places=2)
        self.assertAlmostEqual(prior_map.get("ACTION4", 0), 0.72, places=2)
        self.assertAlmostEqual(prior_map.get("ACTION5", 0), 0.45, places=2)
        self.assertAlmostEqual(prior_map.get("ACTION6", 0), 0.30, places=2)
        # Verify reasons mention game-prior.
        action2_row = next(p for p in priors if p["action_name"] == "ACTION2")
        self.assertIn("game-prior:ar25", action2_row.get("reason", ""))

    def test_game_action_priors_board_type_puzzle(self) -> None:
        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            step_index=1, game_state="ACTIVE", last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"],
            derived_features={"frame_signature": "sig-puzzle"},
        )
        bridge.sync(observation, EpisodicMemory(), [])
        # Manually set board_type to puzzle.
        bridge.facts.setdefault("episode", {})["board_type"] = "puzzle"
        priors = bridge.run_rules("game-action-priors")
        prior_map = {p["action_name"]: p["confidence"] for p in priors}
        self.assertAlmostEqual(prior_map.get("ACTION5", 0), 0.80, places=2)
        self.assertAlmostEqual(prior_map.get("ACTION6", 0), 0.70, places=2)
        self.assertAlmostEqual(prior_map.get("ACTION1", 0), 0.30, places=2)
        action5_row = next(p for p in priors if p["action_name"] == "ACTION5")
        self.assertIn("board-prior:puzzle", action5_row.get("reason", ""))

    def test_sync_promotes_runtime_working_memory_focus_into_recommended_actions(self) -> None:
        class RuntimeFocusMemory(EpisodicMemory):
            def working_memory_snapshot(self):
                return {
                    "procedural_focus_actions": ["ACTION2"],
                    "semantic_focus_actions": ["ACTION2"],
                }

        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            step_index=4,
            game_state="ACTIVE",
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-runtime-recommended-focus"},
        )
        bridge.sync(
            observation,
            RuntimeFocusMemory(),
            [GoalState(name="finish-episode", target_state="WIN")],
        )

        bundle = bridge.facts["memory_bundle"]

        self.assertTrue(bundle["recommended_actions"])
        self.assertEqual(bundle["recommended_actions"][0], "ACTION2")
        self.assertEqual(bundle["working_memory"]["recommended_actions"][0], "ACTION2")
        self.assertEqual(bundle["retrieval_summary"]["suggested_actions"][0], "ACTION2")

    def test_sync_surfaces_pending_decision_runtime_guidance(self) -> None:
        bridge = HyperonBridge(enabled=False)
        memory = EpisodicMemory()
        observation = make_observation(
            step_index=4,
            game_state="ACTIVE",
            last_action=None,
            legal_actions=["ACTION1", "ACTION2", "ACTION3"],
            derived_features={"frame_signature": "sig-runtime-pending-guidance"},
        )
        memory.note_observation(observation)
        memory.note_decision(
            ActionDecision(
                action_name="ACTION3",
                source="symbolic",
                confidence=0.52,
                reason="live working-memory seed",
                subgoal="inspect-active-board",
            )
        )

        bridge.sync(
            observation,
            memory,
            [GoalState(name="finish-episode", target_state="WIN")],
        )

        bundle = bridge.facts["memory_bundle"]

        self.assertEqual(bundle["working_memory"]["active_subgoal"], "inspect-active-board")
        self.assertEqual(bundle["working_memory"]["recent_action_window"], ["ACTION3"])
        self.assertEqual(bundle["working_memory"]["recommended_actions"][0], "ACTION3")
        self.assertEqual(bundle["recommended_actions"][0], "ACTION3")
        self.assertEqual(bridge.facts["memory"]["working_memory"]["active_subgoal"], "inspect-active-board")

    def test_sync_promotes_episodic_runtime_focus_into_recommended_actions(self) -> None:
        bridge = HyperonBridge(enabled=False)
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        before = make_observation(
            step_index=1,
            game_state="ACTIVE",
            last_action=None,
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-runtime-focus-before"},
        )
        after = make_observation(
            step_index=2,
            game_state="ACTIVE",
            last_action="ACTION2",
            legal_actions=["ACTION1", "ACTION2"],
            levels_completed=1,
            derived_features={"frame_signature": "sig-runtime-focus-after"},
        )
        memory.note_observation(before)
        memory.note_decision(
            ActionDecision(
                action_name="ACTION2",
                source="symbolic",
                confidence=0.5,
                reason="progress seed",
                subgoal="break-stall",
            )
        )
        memory.note_observation(after)

        bridge.sync(
            after,
            memory,
            [GoalState(name="finish-episode", target_state="WIN")],
        )

        bundle = bridge.facts["memory_bundle"]

        self.assertEqual(bundle["working_memory"]["active_state"], "ACTIVE")
        self.assertEqual(
            bundle["working_memory"]["active_frame_signature"],
            "sig-runtime-focus-after",
        )
        self.assertEqual(bundle["working_memory"]["legal_actions"], ["ACTION1", "ACTION2"])
        self.assertEqual(
            bundle["working_memory"]["board_context_summary"]["board_active_cells"],
            0,
        )
        self.assertEqual(bundle["working_memory"]["recommended_actions"][0], "ACTION2")
        self.assertEqual(bundle["working_memory"]["recent_action_window"], ["ACTION2"])
        self.assertEqual(bundle["working_memory"]["semantic_focus_actions"][0], "ACTION2")
        self.assertEqual(bundle["working_memory"]["procedural_focus_actions"][0], "ACTION2")
        self.assertEqual(bundle["recommended_actions"][0], "ACTION2")

    def test_sync_demotes_runtime_working_memory_failure_in_recommended_actions(self) -> None:
        class RuntimeFailureMemory(EpisodicMemory):
            def working_memory_snapshot(self):
                return {
                    "recent_failures": [{"action_name": "ACTION1", "outcome": "stalled"}],
                    "failure_streaks": [{"action_name": "ACTION1", "count": 3}],
                }

        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            step_index=4,
            game_state="ACTIVE",
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-runtime-recommended-failure"},
        )
        bridge.sync(
            observation,
            RuntimeFailureMemory(),
            [GoalState(name="finish-episode", target_state="WIN")],
        )

        bundle = bridge.facts["memory_bundle"]

        self.assertTrue(bundle["recommended_actions"])
        self.assertEqual(bundle["recommended_actions"][0], "ACTION2")
        self.assertGreater(
            bundle["recommended_actions"].index("ACTION1"),
            bundle["recommended_actions"].index("ACTION2"),
        )

    def test_sync_preserves_runtime_recent_failures_with_canonical_keys(self) -> None:
        class RuntimeFailureMemory(EpisodicMemory):
            def working_memory_snapshot(self):
                return {
                    "recent_failures": [
                        {
                            "action_name": "ACTION1",
                            "subgoal": "break-stall",
                            "outcome": "stalled",
                            "stagnation_after": 2,
                            "frame_signature": "sig-runtime-failure-canonical",
                            "prior_game_state": "ACTIVE",
                        }
                    ]
                }

        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            step_index=4,
            game_state="ACTIVE",
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-runtime-failure-canonical-live"},
        )

        bridge.sync(
            observation,
            RuntimeFailureMemory(),
            [GoalState(name="finish-episode", target_state="WIN")],
        )

        bundle = bridge.facts["memory_bundle"]

        self.assertEqual(bundle["working_memory"]["recent_failures"][0]["action_name"], "ACTION1")
        self.assertEqual(bundle["working_memory"]["recent_failures"][0]["subgoal"], "break-stall")
        self.assertEqual(
            bundle["working_memory"]["recent_failures"][0]["frame_signature"],
            "sig-runtime-failure-canonical",
        )

    def test_sync_preserves_episodic_runtime_open_failure_modes(self) -> None:
        bridge = HyperonBridge(enabled=False)
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        before = make_observation(
            step_index=1,
            game_state="ACTIVE",
            last_action=None,
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-runtime-open-failure-before"},
        )
        after = make_observation(
            step_index=2,
            game_state="ACTIVE",
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-runtime-open-failure-after"},
        )
        memory.note_observation(before)
        memory.note_decision(
            ActionDecision(
                action_name="ACTION1",
                source="symbolic",
                confidence=0.5,
                reason="runtime open failure seed",
                subgoal="break-stall",
            )
        )
        memory.note_observation(after)

        bridge.sync(
            after,
            memory,
            [GoalState(name="finish-episode", target_state="WIN")],
        )

        bundle = bridge.facts["memory_bundle"]

        self.assertEqual(bundle["working_memory"]["open_failure_modes"][0]["action_name"], "ACTION1")
        self.assertEqual(bundle["working_memory"]["open_failure_modes"][0]["subgoal"], "break-stall")

    def test_sync_preserves_runtime_open_failure_modes_without_recent_failures(self) -> None:
        class RuntimeOpenFailureModesMemory(EpisodicMemory):
            def working_memory_snapshot(self):
                return {
                    "open_failure_modes": [
                        {
                            "action_name": "ACTION1",
                            "subgoal": "break-stall",
                            "outcome": "stalled",
                            "stagnation_after": 2,
                            "frame_signature": "sig-runtime-open-only",
                        }
                    ]
                }

        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            step_index=4,
            game_state="ACTIVE",
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-runtime-open-only-live"},
        )

        bridge.sync(
            observation,
            RuntimeOpenFailureModesMemory(),
            [GoalState(name="finish-episode", target_state="WIN")],
        )

        bundle = bridge.facts["memory_bundle"]

        self.assertEqual(bundle["working_memory"]["open_failure_modes"][0]["action_name"], "ACTION1")
        self.assertEqual(bundle["working_memory"]["open_failure_modes"][0]["subgoal"], "break-stall")
        self.assertEqual(
            bundle["working_memory"]["open_failure_modes"][0]["frame_signature"],
            "sig-runtime-open-only",
        )

    def test_sync_preserves_runtime_active_subgoal_in_retrieval_summary(self) -> None:
        bridge = HyperonBridge(enabled=False)
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        before = make_observation(
            step_index=1,
            game_state="ACTIVE",
            last_action=None,
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-runtime-active-subgoal-before"},
        )
        after = make_observation(
            step_index=2,
            game_state="ACTIVE",
            last_action="ACTION2",
            legal_actions=["ACTION1", "ACTION2"],
            levels_completed=1,
            derived_features={"frame_signature": "sig-runtime-active-subgoal-after"},
        )
        memory.note_observation(before)
        memory.note_decision(
            ActionDecision(
                action_name="ACTION2",
                source="symbolic",
                confidence=0.5,
                reason="runtime active subgoal seed",
                subgoal="break-stall",
            )
        )
        memory.note_observation(after)

        bridge.sync(
            after,
            memory,
            [GoalState(name="finish-episode", target_state="WIN")],
        )

        bundle = bridge.facts["memory_bundle"]

        self.assertEqual(bundle["working_memory"]["active_subgoal"], "break-stall")
        self.assertEqual(bundle["retrieval_summary"]["active_subgoal"], "break-stall")

    def test_sync_mirrors_runtime_working_memory_suppression_and_basin_flags(self) -> None:
        class RuntimeGuidanceMemory(EpisodicMemory):
            def working_memory_snapshot(self):
                return {
                    "semantic_focus_actions": ["ACTION2"],
                    "procedural_focus_actions": ["ACTION2"],
                }

        bridge = HyperonBridge(enabled=False)
        observation = make_observation(
            step_index=4,
            game_state="ACTIVE",
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-runtime-working-mirror"},
        )
        bridge.sync(
            observation,
            RuntimeGuidanceMemory(),
            [GoalState(name="finish-episode", target_state="WIN")],
        )

        bundle = bridge.facts["memory_bundle"]
        working_memory = bundle["working_memory"]

        self.assertEqual(working_memory["recommended_actions"][0], "ACTION2")
        self.assertEqual(working_memory["suppressed_actions"], [])
        self.assertFalse(working_memory["basin_exhausted"])
        self.assertEqual(working_memory["basin_visits"], 0)

    def test_sync_preserves_runtime_basin_exhaustion_from_episodic_memory(self) -> None:
        bridge = HyperonBridge(enabled=False)
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        for index, action_name in enumerate(("ACTION1", "ACTION2")):
            before = make_observation(
                step_index=index,
                game_state="NOT_FINISHED",
                last_action=None,
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-runtime-basin"},
            )
            after = make_observation(
                step_index=index + 1,
                game_state="NOT_FINISHED",
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-runtime-basin"},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="symbolic",
                    confidence=0.5,
                    reason="runtime basin seed",
                )
            )
            memory.note_observation(after)

        bridge.sync(
            after,
            memory,
            [GoalState(name="finish-episode", target_state="WIN")],
        )

        bundle = bridge.facts["memory_bundle"]

        self.assertTrue(bundle["working_memory"]["basin_exhausted"])
        self.assertEqual(bundle["working_memory"]["basin_visits"], 2)
        self.assertEqual(bundle["working_memory"]["suppressed_actions"], ["ACTION1", "ACTION2"])

    def test_offline_trained_priors_bias_matching_frame_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            learning_path = Path(tmpdir) / "arc_learned.json"
            trained_priors_db_path = Path(tmpdir) / "arc_demo_action_priors.sqlite3"
            initialize_schema(trained_priors_db_path)
            upsert_training_counts(
                trained_priors_db_path,
                [
                    {
                        "scope": "game",
                        "game_id": "ls20",
                        "action_name": "ACTION2",
                        "count": 2,
                    },
                    {
                        "scope": "state",
                        "game_id": "ls20",
                        "state": "ACTIVE",
                        "action_name": "ACTION2",
                        "count": 3,
                    },
                    {
                        "scope": "frame",
                        "game_id": "ls20",
                        "frame_signature": "sig-123",
                        "action_name": "ACTION1",
                        "count": 9,
                    },
                    {
                        "scope": "frame",
                        "game_id": "ls20",
                        "frame_signature": "sig-123",
                        "action_name": "ACTION2",
                        "count": 1,
                    },
                ],
            )
            bridge = HyperonBridge(
                enabled=False,
                learning_path=learning_path,
                trained_priors_db_path=trained_priors_db_path,
            )
            memory = EpisodicMemory()
            observation = make_observation()
            observation.derived_features["frame_signature"] = "sig-123"

            bridge.sync(
                observation,
                memory,
                [GoalState(name="finish-episode", target_state="WIN")],
            )

            priors = bridge.run_rules("action-priors")
            self.assertEqual(priors[0]["action_name"], "ACTION1")

    def test_missing_trained_priors_file_is_a_no_op(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trained_priors_db_path = Path(tmpdir) / "missing.sqlite3"
            bridge = HyperonBridge(enabled=False, trained_priors_db_path=trained_priors_db_path)
            bridge.sync(
                make_observation(),
                EpisodicMemory(),
                [GoalState(name="finish-episode", target_state="WIN")],
            )

            self.assertEqual(bridge.facts.get("trained_prior_bucket"), {})

    def test_invalid_trained_priors_db_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trained_priors_db_path = Path(tmpdir) / "invalid.sqlite3"
            trained_priors_db_path.write_text("not-a-sqlite-db", encoding="utf-8")
            bridge = HyperonBridge(enabled=False, trained_priors_db_path=trained_priors_db_path)

            bridge.sync(
                make_observation(),
                EpisodicMemory(),
                [GoalState(name="finish-episode", target_state="WIN")],
            )

            self.assertEqual(bridge.facts.get("trained_prior_bucket"), {})

    def test_invalid_memory_db_falls_back_to_in_memory_records_and_keeps_offline_priors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_db_path = Path(tmpdir) / "invalid-memory.sqlite3"
            memory_db_path.write_text("not-a-sqlite-db", encoding="utf-8")
            trained_priors_db_path = Path(tmpdir) / "offline-priors.sqlite3"
            initialize_schema(trained_priors_db_path)
            upsert_training_counts(
                trained_priors_db_path,
                [
                    {
                        "scope": "game",
                        "game_id": "ls20",
                        "action_name": "ACTION2",
                        "count": 5,
                    }
                ],
            )
            bridge = HyperonBridge(
                enabled=False,
                memory_db_path=memory_db_path,
                trained_priors_db_path=trained_priors_db_path,
            )
            memory = EpisodicMemory()
            memory.bind_game("ls20")
            before = make_observation(
                step_index=0,
                game_state="ACTIVE",
                last_action=None,
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-a"},
            )
            after = make_observation(
                step_index=1,
                game_state="ACTIVE",
                levels_completed=1,
                last_action="ACTION1",
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-b"},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name="ACTION1",
                    source="symbolic",
                    confidence=0.7,
                    reason="fallback-memory-progress",
                )
            )
            memory.note_observation(after)

            bridge.sync(
                make_observation(
                    step_index=2,
                    game_state="ACTIVE",
                    last_action="ACTION1",
                    legal_actions=["ACTION1", "ACTION2"],
                    derived_features={"frame_signature": "sig-a"},
                ),
                memory,
                [GoalState(name="finish-episode", target_state="WIN")],
            )

            bundle = bridge.facts["memory_bundle"]
            self.assertFalse(bridge._memory_store_available)
            self.assertEqual(bundle["action_stats"]["ACTION1"]["progress"], 1)
            self.assertEqual(bundle["state_action_stats"]["ACTION1"]["progress"], 1)
            self.assertEqual(bundle["frame_action_stats"]["ACTION1"]["progress"], 1)
            self.assertEqual(bundle["trained_prior_bucket"], {"ACTION2": 5})

    def test_fallback_memory_bundle_prefers_subgoal_trained_prior_when_frame_is_missing(self) -> None:
        class ActiveSubgoalMemory(EpisodicMemory):
            def working_memory_snapshot(self):
                return {"active_subgoal": "inspect-active-board"}

        with tempfile.TemporaryDirectory() as tmpdir:
            memory_db_path = Path(tmpdir) / "invalid-memory.sqlite3"
            memory_db_path.write_text("not-a-sqlite-db", encoding="utf-8")
            trained_priors_db_path = Path(tmpdir) / "offline-priors.sqlite3"
            initialize_schema(trained_priors_db_path)
            upsert_training_counts(
                trained_priors_db_path,
                [
                    {
                        "scope": "state",
                        "game_id": "ls20",
                        "state": "ACTIVE",
                        "action_name": "ACTION1",
                        "count": 8,
                    },
                    {
                        "scope": "subgoal",
                        "game_id": "ls20",
                        "subgoal": "inspect-active-board",
                        "action_name": "ACTION6",
                        "count": 11,
                    },
                ],
            )
            bridge = HyperonBridge(
                enabled=False,
                memory_db_path=memory_db_path,
                trained_priors_db_path=trained_priors_db_path,
            )
            memory = ActiveSubgoalMemory()
            memory.bind_game("ls20")
            observation = make_observation(
                step_index=1,
                game_state="ACTIVE",
                last_action="ACTION1",
                legal_actions=["ACTION1", "ACTION6"],
                derived_features={"frame_signature": "sig-missing"},
            )

            bridge.sync(
                observation,
                memory,
                [GoalState(name="inspect-active-board")],
            )

            bundle = bridge.facts["memory_bundle"]
            self.assertFalse(bridge._memory_store_available)
            self.assertEqual(bundle["trained_prior_bucket"], {"ACTION6": 11, "ACTION1": 8})
            self.assertEqual(bundle["recommended_actions"][0], "ACTION6")
            self.assertEqual(
                bundle["working_memory"]["trained_prior_focus"],
                [
                    {"action_name": "ACTION6", "count": 11},
                    {"action_name": "ACTION1", "count": 8},
                ],
            )
            self.assertEqual(
                bundle["retrieval_summary"]["trained_prior_focus"],
                [
                    {"action_name": "ACTION6", "count": 11},
                    {"action_name": "ACTION1", "count": 8},
                ],
            )

    def test_record_transition_failure_disables_writes_but_preserves_memory_reads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_db_path = Path(tmpdir) / "memory.sqlite3"
            bridge = HyperonBridge(
                enabled=False,
                memory_db_path=memory_db_path,
            )
            persisted_memory = EpisodicMemory()
            persisted_memory.bind_game("ls20")
            persisted_before = make_observation(
                step_index=0,
                game_state="ACTIVE",
                last_action=None,
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-loop"},
            )
            persisted_after = make_observation(
                step_index=1,
                game_state="ACTIVE",
                levels_completed=1,
                last_action="ACTION2",
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-loop"},
            )
            persisted_memory.note_observation(persisted_before)
            persisted_memory.note_decision(
                ActionDecision(
                    action_name="ACTION2",
                    source="symbolic",
                    confidence=0.7,
                    reason="persisted-progress",
                )
            )
            persisted_memory.note_observation(persisted_after)
            self._persist_drained_records(bridge, persisted_memory)

            memory = EpisodicMemory()
            memory.bind_game("ls20")
            before = make_observation(
                step_index=2,
                game_state="ACTIVE",
                last_action=None,
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-loop"},
            )
            after = make_observation(
                step_index=3,
                game_state="ACTIVE",
                last_action="ACTION1",
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-loop"},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name="ACTION1",
                    source="symbolic",
                    confidence=0.6,
                    reason="transition-write-failure",
                )
            )
            memory.note_observation(after)

            with patch(
                "agi_core.knowledge.hyperon_bridge.persist_transition",
                side_effect=sqlite3.DatabaseError("write failed"),
            ):
                self._persist_drained_records(bridge, memory)

            bridge.sync(
                make_observation(
                    step_index=2,
                    game_state="ACTIVE",
                    last_action="ACTION1",
                    legal_actions=["ACTION1", "ACTION2"],
                    derived_features={"frame_signature": "sig-loop"},
                ),
                memory,
                [GoalState(name="finish-episode", target_state="WIN")],
            )

            bundle = bridge.facts["memory_bundle"]
            self.assertTrue(bridge._memory_store_available)
            self.assertFalse(bridge._memory_write_available)
            self.assertEqual(bundle["action_stats"]["ACTION1"]["stalled"], 1)
            self.assertEqual(bundle["action_stats"]["ACTION2"]["progress"], 1)
            self.assertEqual(bundle["state_action_stats"]["ACTION2"]["progress"], 1)
            self.assertEqual(bundle["recent_records"][0]["action_name"], "ACTION1")
            self.assertIn("ACTION2", bundle["recommended_actions"])

    def test_record_transition_retries_and_flushes_backlog_after_transient_write_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_db_path = Path(tmpdir) / "memory.sqlite3"
            bridge = HyperonBridge(enabled=False, memory_db_path=memory_db_path)
            first_record = EpisodeRecord(
                game_id="ls20",
                step_index=1,
                action_name="ACTION1",
                outcome="stalled",
                progress_delta=0,
                game_state="ACTIVE",
                prior_game_state="ACTIVE",
                levels_completed=0,
                prior_frame_signature="sig-loop",
                next_game_state="ACTIVE",
                next_frame_signature="sig-loop",
                prior_legal_actions=["ACTION1", "ACTION2"],
                stagnation_before=0,
                stagnation_after=1,
                decision_subgoal=None,
                decision_source="symbolic",
            )
            second_record = EpisodeRecord(
                game_id="ls20",
                step_index=2,
                action_name="ACTION2",
                outcome="progress",
                progress_delta=1,
                game_state="ACTIVE",
                prior_game_state="ACTIVE",
                levels_completed=1,
                prior_frame_signature="sig-loop",
                next_game_state="ACTIVE",
                next_frame_signature="sig-next",
                prior_legal_actions=["ACTION1", "ACTION2"],
                stagnation_before=1,
                stagnation_after=0,
                decision_subgoal=None,
                decision_source="symbolic",
            )

            with patch(
                "agi_core.knowledge.hyperon_bridge.persist_transition",
                side_effect=sqlite3.DatabaseError("write failed"),
            ):
                bridge.record_transition(first_record)

            self.assertFalse(bridge._memory_write_available)
            self.assertEqual(len(bridge._unpersisted_records), 1)

            bridge.record_transition(second_record)

            self.assertTrue(bridge._memory_write_available)
            self.assertEqual(bridge._unpersisted_records, {})
            bundle = load_memory_bundle(
                memory_db_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature="sig-loop",
                subgoal=None,
                legal_actions=["ACTION1", "ACTION2"],
            )
            self.assertEqual(bundle["action_stats"]["ACTION1"]["stalled"], 1)
            self.assertEqual(bundle["action_stats"]["ACTION2"]["progress"], 1)

    def test_record_transition_partial_flush_failure_does_not_replay_persisted_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_db_path = Path(tmpdir) / "memory.sqlite3"
            bridge = HyperonBridge(enabled=False, memory_db_path=memory_db_path)
            first_record = EpisodeRecord(
                game_id="ls20",
                step_index=1,
                action_name="ACTION1",
                outcome="stalled",
                progress_delta=0,
                game_state="ACTIVE",
                prior_game_state="ACTIVE",
                levels_completed=0,
                prior_frame_signature="sig-loop",
                next_game_state="ACTIVE",
                next_frame_signature="sig-loop",
                prior_legal_actions=["ACTION1", "ACTION2"],
                stagnation_before=0,
                stagnation_after=1,
                decision_subgoal=None,
                decision_source="symbolic",
            )
            second_record = EpisodeRecord(
                game_id="ls20",
                step_index=2,
                action_name="ACTION2",
                outcome="progress",
                progress_delta=1,
                game_state="ACTIVE",
                prior_game_state="ACTIVE",
                levels_completed=1,
                prior_frame_signature="sig-loop",
                next_game_state="ACTIVE",
                next_frame_signature="sig-next",
                prior_legal_actions=["ACTION1", "ACTION2"],
                stagnation_before=1,
                stagnation_after=0,
                decision_subgoal=None,
                decision_source="symbolic",
            )

            bridge._unpersisted_records = {
                bridge._record_key(first_record): first_record,
                bridge._record_key(second_record): second_record,
            }

            persist_calls = {"count": 0}
            real_persist_transition = record_transition

            def flaky_persist(db_path, record):
                persist_calls["count"] += 1
                if persist_calls["count"] == 2:
                    raise sqlite3.DatabaseError("write failed after first commit")
                return real_persist_transition(db_path, record)

            with patch(
                "agi_core.knowledge.hyperon_bridge.persist_transition",
                side_effect=flaky_persist,
            ):
                with self.assertRaises(sqlite3.DatabaseError):
                    bridge._flush_unpersisted_records()

            self.assertEqual(
                list(bridge._unpersisted_records.values()),
                [second_record],
            )

            bridge.record_transition(second_record)

            self.assertEqual(bridge._unpersisted_records, {})
            bundle = load_memory_bundle(
                memory_db_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature="sig-loop",
                subgoal=None,
                legal_actions=["ACTION1", "ACTION2"],
            )
            self.assertEqual(bundle["action_stats"]["ACTION1"]["stalled"], 1)
            self.assertEqual(bundle["action_stats"]["ACTION2"]["progress"], 1)

    def test_sync_reads_archive_memory_when_hot_memory_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            hot_memory_db_path = Path(tmpdir) / "hot-memory.sqlite3"
            archive_memory_db_path = Path(tmpdir) / "archive-memory.sqlite3"
            initialize_memory_schema(hot_memory_db_path)
            initialize_memory_schema(archive_memory_db_path)
            record = EpisodeRecord(
                game_id="ls20",
                step_index=1,
                action_name="ACTION2",
                outcome="progress",
                progress_delta=1,
                game_state="ACTIVE",
                prior_game_state="ACTIVE",
                levels_completed=1,
                prior_frame_signature="sig-archive",
                next_game_state="ACTIVE",
                next_frame_signature="sig-archive-next",
                prior_legal_actions=["ACTION1", "ACTION2"],
                stagnation_before=0,
                stagnation_after=0,
                decision_subgoal="break-stall",
                decision_source="symbolic",
            )
            bridge = HyperonBridge(
                enabled=False,
                memory_db_path=hot_memory_db_path,
                archive_memory_db_path=archive_memory_db_path,
            )
            record_transition(archive_memory_db_path, record)

            bridge.sync(
                make_observation(
                    step_index=2,
                    game_state="ACTIVE",
                    last_action="ACTION1",
                    legal_actions=["ACTION1", "ACTION2"],
                    derived_features={"frame_signature": "sig-live"},
                ),
                EpisodicMemory(),
                [GoalState(name="finish-episode", target_state="WIN")],
            )

            bundle = bridge.current_memory_bundle()
            self.assertEqual(bundle.action_stats["ACTION2"]["progress"], 1)

    def test_load_memory_bundle_retry_reapplies_pending_record_overlay(self) -> None:
        bridge = HyperonBridge(enabled=False, memory_db_path=Path("memory.sqlite3"))
        bridge._memory_store_available = True
        observation = make_observation(
            step_index=2,
            game_state="ACTIVE",
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-live"},
        )
        pending_record = EpisodeRecord(
            game_id="ls20",
            step_index=1,
            action_name="ACTION2",
            outcome="progress",
            progress_delta=1,
            game_state="ACTIVE",
            prior_game_state="ACTIVE",
            levels_completed=1,
            prior_frame_signature="sig-live",
            next_game_state="ACTIVE",
            next_frame_signature="sig-next",
            prior_legal_actions=["ACTION1", "ACTION2"],
            stagnation_before=0,
            stagnation_after=0,
            decision_subgoal="advance-level",
            decision_source="symbolic",
        )
        persisted_bundle = {
            "working_memory": {},
            "episodic_recalls": [],
            "semantic_memories": [],
            "procedural_memories": [],
            "retrieval_summary": {},
            "board_context_summary": {},
            "board_context_action_stats": {},
            "pattern_action_stats": {},
            "sequence_action_stats": {},
            "board_cluster_action_stats": {},
            "episode_summaries": [],
            "recent_records": [],
            "basin_status": {},
            "basin_action_stats": {},
            "action_stats": {},
            "state_action_stats": {},
            "frame_action_stats": {},
            "subgoal_action_stats": {},
            "trained_prior_bucket": {},
            "recommended_actions": [],
            "suppressed_actions": [],
        }
        overlay_bundle = dict(persisted_bundle)
        overlay_bundle["recent_records"] = [{"action_name": "ACTION2"}]
        overlay_bundle["action_stats"] = {
            "ACTION2": {
                "progress": 1,
                "failed": 0,
                "stalled": 0,
                "total": 1,
                "last_outcome": "progress",
            }
        }

        with patch(
            "agi_core.knowledge.hyperon_bridge.load_memory_bundle_from_stores",
            return_value=persisted_bundle,
        ):
            with patch.object(
                bridge,
                "_pending_records_for_game",
                return_value=[pending_record],
            ):
                with patch.object(
                    bridge,
                    "_overlay_records_onto_bundle",
                    side_effect=[RuntimeError("overlay failed once"), overlay_bundle],
                ) as overlay_mock:
                    bundle = bridge._load_memory_bundle(
                        observation=observation,
                        subgoal="advance-level",
                        legal_actions=["ACTION1", "ACTION2"],
                        memory=EpisodicMemory(),
                    )

        self.assertEqual(overlay_mock.call_count, 2)
        self.assertEqual(bundle["recent_records"][0]["action_name"], "ACTION2")
        self.assertEqual(bundle["action_stats"]["ACTION2"]["progress"], 1)

    def test_load_memory_bundle_retry_does_not_touch_retrieved_twice(self) -> None:
        bridge = HyperonBridge(enabled=False, memory_db_path=Path("memory.sqlite3"))
        bridge._memory_store_available = True
        observation = make_observation(
            step_index=2,
            game_state="ACTIVE",
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-live"},
        )
        pending_record = EpisodeRecord(
            game_id="ls20",
            step_index=1,
            action_name="ACTION2",
            outcome="progress",
            progress_delta=1,
            game_state="ACTIVE",
            prior_game_state="ACTIVE",
            levels_completed=1,
            prior_frame_signature="sig-live",
            next_game_state="ACTIVE",
            next_frame_signature="sig-next",
            prior_legal_actions=["ACTION1", "ACTION2"],
            stagnation_before=0,
            stagnation_after=0,
            decision_subgoal="advance-level",
            decision_source="symbolic",
        )
        persisted_bundle = {
            "working_memory": {},
            "episodic_recalls": [],
            "semantic_memories": [],
            "procedural_memories": [],
            "retrieval_summary": {},
            "board_context_summary": {},
            "board_context_action_stats": {},
            "pattern_action_stats": {},
            "sequence_action_stats": {},
            "board_cluster_action_stats": {},
            "episode_summaries": [],
            "recent_records": [],
            "basin_status": {},
            "basin_action_stats": {},
            "action_stats": {},
            "state_action_stats": {},
            "frame_action_stats": {},
            "subgoal_action_stats": {},
            "trained_prior_bucket": {},
            "recommended_actions": [],
            "suppressed_actions": [],
        }

        with patch(
            "agi_core.knowledge.hyperon_bridge.load_memory_bundle_from_stores",
            return_value=persisted_bundle,
        ) as load_bundle_mock:
            with patch.object(
                bridge,
                "_pending_records_for_game",
                return_value=[pending_record],
            ):
                with patch.object(
                    bridge,
                    "_overlay_records_onto_bundle",
                    side_effect=[RuntimeError("overlay failed once"), persisted_bundle],
                ):
                    bridge._load_memory_bundle(
                        observation=observation,
                        subgoal="advance-level",
                        legal_actions=["ACTION1", "ACTION2"],
                        memory=EpisodicMemory(),
                    )

        self.assertEqual(load_bundle_mock.call_count, 2)
        self.assertTrue(load_bundle_mock.call_args_list[0].kwargs["touch_retrieved"])
        self.assertFalse(load_bundle_mock.call_args_list[1].kwargs["touch_retrieved"])

    def test_contextual_memory_bundle_does_not_touch_access_counters(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_db_path = Path(tmpdir) / "memory.sqlite3"
            initialize_memory_schema(memory_db_path)
            record_transition(
                memory_db_path,
                EpisodeRecord(
                    game_id="ls20",
                    step_index=3,
                    action_name="ACTION2",
                    outcome="progress",
                    progress_delta=1,
                    game_state="ACTIVE",
                    prior_game_state="ACTIVE",
                    levels_completed=1,
                    prior_frame_signature="sig-contextual",
                    next_game_state="ACTIVE",
                    next_frame_signature="sig-contextual-next",
                    prior_legal_actions=["ACTION1", "ACTION2"],
                    stagnation_before=0,
                    stagnation_after=0,
                    decision_subgoal="advance-level",
                    decision_source="symbolic",
                ),
            )
            bridge = HyperonBridge(enabled=False, memory_db_path=memory_db_path)
            observation = make_observation(
                step_index=27,
                game_state="ACTIVE",
                last_action="ACTION1",
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-contextual"},
            )

            bundle = bridge.contextual_memory_bundle(
                observation,
                "advance-level",
                ["ACTION1", "ACTION2"],
            )

            self.assertTrue(bundle.semantic_memories)
            self.assertTrue(bundle.procedural_memories)
            with closing(sqlite3.connect(memory_db_path)) as connection:
                semantic_counts = connection.execute(
                    "SELECT access_count, last_access_step FROM live_semantic_memory"
                ).fetchall()
                procedural_counts = connection.execute(
                    "SELECT access_count, last_access_step FROM live_procedural_memory"
                ).fetchall()
            self.assertTrue(all(int(row[0] or 0) == 0 for row in semantic_counts))
            self.assertTrue(all(int(row[1] or 0) == 0 for row in semantic_counts))
            self.assertTrue(all(int(row[0] or 0) == 0 for row in procedural_counts))
            self.assertTrue(all(int(row[1] or 0) == 0 for row in procedural_counts))

    def test_contextual_memory_bundle_inherits_runtime_failure_context(self) -> None:
        bridge = HyperonBridge(enabled=False)
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        before = make_observation(
            step_index=1,
            game_state="ACTIVE",
            last_action=None,
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-runtime-contextual-before"},
        )
        after = make_observation(
            step_index=2,
            game_state="ACTIVE",
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-runtime-contextual-after"},
        )
        memory.note_observation(before)
        memory.note_decision(
            ActionDecision(
                action_name="ACTION1",
                source="symbolic",
                confidence=0.4,
                reason="runtime failure seed",
                subgoal="break-stall",
            )
        )
        memory.note_observation(after)
        bridge.sync(after, memory, [GoalState(name="finish-episode", target_state="WIN")])

        bundle = bridge.contextual_memory_bundle(
            after,
            "break-stall",
            ["ACTION1", "ACTION2"],
        )

        self.assertEqual(bundle.working_memory["recent_action_window"], ["ACTION1"])
        self.assertEqual(
            bundle.working_memory["open_failure_modes"][0]["action_name"],
            "ACTION1",
        )
        self.assertEqual(
            bundle.working_memory["recent_failures"][0]["action_name"],
            "ACTION1",
        )
        self.assertEqual(
            bundle.working_memory["failure_streaks"][0]["action_name"],
            "ACTION1",
        )
        self.assertTrue(bundle.retrieval_summary["runtime_has_working_memory"])

    def test_sync_clears_unpersisted_records_when_same_game_restarts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_db_path = Path(tmpdir) / "memory.sqlite3"
            bridge = HyperonBridge(enabled=False, memory_db_path=memory_db_path)
            bridge.sync(
                make_observation(
                    step_index=4,
                    game_state="ACTIVE",
                    last_action="ACTION2",
                    legal_actions=["ACTION1", "ACTION2"],
                    derived_features={
                        "frame_signature": "sig-loop",
                        "step_index_reliable": True,
                    },
                ),
                EpisodicMemory(),
                [GoalState(name="finish-episode", target_state="WIN")],
            )

            stale_record = EpisodeRecord(
                game_id="ls20",
                step_index=5,
                action_name="ACTION1",
                outcome="stalled",
                progress_delta=0,
                game_state="ACTIVE",
                prior_game_state="ACTIVE",
                levels_completed=0,
                prior_frame_signature="sig-loop",
                next_game_state="ACTIVE",
                next_frame_signature="sig-loop",
                prior_legal_actions=["ACTION1", "ACTION2"],
                stagnation_before=0,
                stagnation_after=1,
                decision_subgoal=None,
                decision_source="symbolic",
            )
            with patch(
                "agi_core.knowledge.hyperon_bridge.persist_transition",
                side_effect=sqlite3.DatabaseError("write failed"),
            ):
                bridge.record_transition(stale_record)

            self.assertEqual(len(bridge._unpersisted_records), 1)

            bridge.sync(
                make_observation(
                    step_index=1,
                    game_state="ACTIVE",
                    last_action=None,
                    legal_actions=["ACTION1", "ACTION2"],
                    derived_features={
                        "frame_signature": "sig-loop",
                        "step_index_reliable": True,
                    },
                ),
                EpisodicMemory(),
                [GoalState(name="finish-episode", target_state="WIN")],
            )

            self.assertEqual(bridge._unpersisted_records, {})
            self.assertEqual(bridge.current_memory_bundle().action_stats, {})

    def test_sync_clears_unpersisted_records_when_leaving_reset_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_db_path = Path(tmpdir) / "memory.sqlite3"
            bridge = HyperonBridge(enabled=False, memory_db_path=memory_db_path)
            bridge.sync(
                make_observation(
                    step_index=8,
                    game_state="GAME_OVER",
                    last_action="RESET",
                    legal_actions=["RESET"],
                    derived_features={
                        "frame_signature": "sig-reset",
                        "step_index_reliable": False,
                    },
                ),
                EpisodicMemory(),
                [GoalState(name="finish-episode", target_state="WIN")],
            )

            stale_record = EpisodeRecord(
                game_id="ls20",
                step_index=9,
                action_name="ACTION1",
                outcome="stalled",
                progress_delta=0,
                game_state="GAME_OVER",
                prior_game_state="GAME_OVER",
                levels_completed=0,
                prior_frame_signature="sig-reset",
                next_game_state="GAME_OVER",
                next_frame_signature="sig-reset",
                prior_legal_actions=["RESET"],
                stagnation_before=0,
                stagnation_after=1,
                decision_subgoal=None,
                decision_source="symbolic",
            )
            with patch(
                "agi_core.knowledge.hyperon_bridge.persist_transition",
                side_effect=sqlite3.DatabaseError("write failed"),
            ):
                bridge.record_transition(stale_record)

            self.assertEqual(len(bridge._unpersisted_records), 1)

            bridge.sync(
                make_observation(
                    step_index=1,
                    game_state="ACTIVE",
                    last_action=None,
                    legal_actions=["ACTION1", "ACTION2"],
                    derived_features={
                        "frame_signature": "sig-live",
                        "step_index_reliable": False,
                    },
                ),
                EpisodicMemory(),
                [GoalState(name="finish-episode", target_state="WIN")],
            )

            self.assertEqual(bridge._unpersisted_records, {})
            self.assertEqual(bridge.current_memory_bundle().action_stats, {})

    def test_sync_keeps_unpersisted_records_when_step_index_drop_is_unreliable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_db_path = Path(tmpdir) / "memory.sqlite3"
            bridge = HyperonBridge(enabled=False, memory_db_path=memory_db_path)
            bridge.sync(
                make_observation(
                    step_index=4,
                    game_state="ACTIVE",
                    last_action="ACTION2",
                    legal_actions=["ACTION1", "ACTION2"],
                    derived_features={
                        "frame_signature": "sig-loop",
                        "step_index_reliable": False,
                    },
                ),
                EpisodicMemory(),
                [GoalState(name="finish-episode", target_state="WIN")],
            )

            stale_record = EpisodeRecord(
                game_id="ls20",
                step_index=5,
                action_name="ACTION1",
                outcome="stalled",
                progress_delta=0,
                game_state="ACTIVE",
                prior_game_state="ACTIVE",
                levels_completed=0,
                prior_frame_signature="sig-loop",
                next_game_state="ACTIVE",
                next_frame_signature="sig-loop",
                prior_legal_actions=["ACTION1", "ACTION2"],
                stagnation_before=0,
                stagnation_after=1,
                decision_subgoal=None,
                decision_source="symbolic",
            )
            with patch(
                "agi_core.knowledge.hyperon_bridge.persist_transition",
                side_effect=sqlite3.DatabaseError("write failed"),
            ):
                bridge.record_transition(stale_record)

            bridge.sync(
                make_observation(
                    step_index=1,
                    game_state="ACTIVE",
                    last_action=None,
                    legal_actions=["ACTION1", "ACTION2"],
                    derived_features={
                        "frame_signature": "sig-loop",
                        "step_index_reliable": False,
                    },
                ),
                EpisodicMemory(),
                [GoalState(name="finish-episode", target_state="WIN")],
            )

            self.assertEqual(len(bridge._unpersisted_records), 1)
            self.assertEqual(bridge.current_memory_bundle().action_stats["ACTION1"]["stalled"], 1)

    def test_custom_terminal_state_names_drive_shadow_rules_and_restart_boundary(self) -> None:
        bridge = HyperonBridge(
            enabled=False,
            reset_state_names=("READY", "LOSE"),
            win_state_name="DONE",
        )

        bridge.sync(
            make_observation(
                game_state="DONE",
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-done"},
            ),
            EpisodicMemory(),
            [GoalState(name="finish-episode", target_state="DONE")],
        )
        self.assertEqual(
            bridge.run_rules("goal-hints"),
            [{"subgoal": "hold-win", "confidence": 0.95}],
        )

        bridge.sync(
            make_observation(
                step_index=2,
                game_state="LOSE",
                legal_actions=["RESET"],
                derived_features={"frame_signature": "sig-lose"},
            ),
            EpisodicMemory(),
            [GoalState(name="finish-episode", target_state="DONE")],
        )
        self.assertEqual(
            bridge.run_rules("reset-needed"),
            [
                {
                    "action_name": "RESET",
                    "confidence": 0.99,
                    "reason": "symbolic reset guard",
                }
            ],
        )

    def test_fallback_memory_bundle_reuses_learned_knowledge_when_memory_db_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            learning_path = Path(tmpdir) / "arc_learned.json"
            learning_path.write_text(
                json.dumps(
                    {
                        "games": {
                            "ls20": {
                                "actions": {
                                    "ACTION2": {
                                        "progress": 3,
                                        "failed": 0,
                                        "stalled": 0,
                                        "total": 3,
                                        "last_outcome": "progress",
                                    }
                                },
                                "states": {
                                    "ACTIVE": {
                                        "actions": {
                                            "ACTION2": {
                                                "progress": 2,
                                                "failed": 0,
                                                "stalled": 0,
                                                "total": 2,
                                                "last_outcome": "progress",
                                            }
                                        }
                                    }
                                },
                                "subgoals": {
                                    "break-stall": {
                                        "actions": {
                                            "ACTION2": {
                                                "progress": 1,
                                                "failed": 0,
                                                "stalled": 0,
                                                "total": 1,
                                                "last_outcome": "progress",
                                            }
                                        }
                                    }
                                },
                                "frame_signatures": {
                                    "sig-learned": {
                                        "actions": {
                                            "ACTION2": {
                                                "progress": 1,
                                                "failed": 0,
                                                "stalled": 0,
                                                "total": 1,
                                                "last_outcome": "progress",
                                            }
                                        }
                                    }
                                },
                                "records_seen": 3,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            memory_db_path = Path(tmpdir) / "invalid-memory.sqlite3"
            memory_db_path.write_text("not-a-sqlite-db", encoding="utf-8")
            bridge = HyperonBridge(
                enabled=False,
                learning_path=learning_path,
                memory_db_path=memory_db_path,
            )

            observation = make_observation(
                game_state="ACTIVE",
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-learned"},
            )
            bridge.sync(
                observation,
                EpisodicMemory(),
                [GoalState(name="finish-episode", target_state="WIN")],
            )

            bundle = bridge.facts["memory_bundle"]
            self.assertFalse(bridge._memory_store_available)
            self.assertEqual(bundle["action_stats"]["ACTION2"]["progress"], 3)
            self.assertEqual(bundle["state_action_stats"]["ACTION2"]["progress"], 2)
            self.assertEqual(bundle["frame_action_stats"]["ACTION2"]["progress"], 1)
            self.assertEqual(bundle["subgoal_action_stats"], {})
            self.assertIn("ACTION2", bundle["recommended_actions"])

    def test_fallback_memory_bundle_uses_short_key_learned_knowledge_for_full_runtime_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            learning_path = Path(tmpdir) / "arc_learned.json"
            learning_path.write_text(
                json.dumps(
                    {
                        "games": {
                            "ls20": {
                                "actions": {
                                    "ACTION2": {
                                        "progress": 3,
                                        "failed": 0,
                                        "stalled": 0,
                                        "total": 3,
                                        "last_outcome": "progress",
                                    }
                                },
                                "states": {
                                    "ACTIVE": {
                                        "actions": {
                                            "ACTION2": {
                                                "progress": 2,
                                                "failed": 0,
                                                "stalled": 0,
                                                "total": 2,
                                                "last_outcome": "progress",
                                            }
                                        }
                                    }
                                },
                                "frame_signatures": {
                                    "sig-learned": {
                                        "actions": {
                                            "ACTION2": {
                                                "progress": 1,
                                                "failed": 0,
                                                "stalled": 0,
                                                "total": 1,
                                                "last_outcome": "progress",
                                            }
                                        }
                                    }
                                },
                                "records_seen": 3,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            memory_db_path = Path(tmpdir) / "invalid-memory.sqlite3"
            memory_db_path.write_text("not-a-sqlite-db", encoding="utf-8")
            bridge = HyperonBridge(
                enabled=False,
                learning_path=learning_path,
                memory_db_path=memory_db_path,
            )

            observation = make_observation(
                game_id="ls20-9607627b",
                game_state="ACTIVE",
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-learned"},
            )
            bridge.sync(
                observation,
                EpisodicMemory(),
                [GoalState(name="finish-episode", target_state="WIN")],
            )

            bundle = bridge.facts["memory_bundle"]
            self.assertFalse(bridge._memory_store_available)
            self.assertEqual(bundle["action_stats"]["ACTION2"]["progress"], 3)
            self.assertEqual(bundle["state_action_stats"]["ACTION2"]["progress"], 2)
            self.assertEqual(bundle["frame_action_stats"]["ACTION2"]["progress"], 1)
            self.assertIn("ACTION2", bundle["recommended_actions"])

    def test_memory_bundle_as_learned_preserves_real_subgoal_name(self) -> None:
        bridge = HyperonBridge(enabled=False)
        bridge.facts["episode"] = {
            "game_state": "ACTIVE",
            "frame_signature": "sig-subgoal",
        }

        learned = bridge._memory_bundle_as_learned(
            {
                "action_stats": {},
                "state_action_stats": {},
                "frame_action_stats": {},
                "subgoal_action_stats": {
                    "ACTION2": {"progress": 1, "failed": 0, "stalled": 0, "total": 1}
                },
                "recent_records": [],
            },
            subgoal_name="break-stall",
        )

        self.assertEqual(
            learned["subgoals"]["break-stall"]["actions"]["ACTION2"]["progress"],
            1,
        )
        self.assertNotIn("current", learned["subgoals"])

    def test_memory_bundle_as_learned_infers_subgoal_name_from_recent_records(self) -> None:
        bridge = HyperonBridge(enabled=False)
        bridge.facts["episode"] = {
            "game_state": "ACTIVE",
            "frame_signature": "sig-subgoal",
        }

        learned = bridge._memory_bundle_as_learned(
            {
                "action_stats": {},
                "state_action_stats": {},
                "frame_action_stats": {},
                "subgoal_action_stats": {
                    "ACTION2": {"progress": 1, "failed": 0, "stalled": 0, "total": 1}
                },
                "working_memory": {},
                "retrieval_summary": {},
                "recent_records": [
                    {"decision_subgoal": "break-stall"},
                    {"decision_subgoal": "break-stall"},
                ],
            },
        )

        self.assertEqual(
            learned["subgoals"]["break-stall"]["actions"]["ACTION2"]["progress"],
            1,
        )

    def test_legacy_ingest_path_is_initialized_and_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            learning_path = Path(tmpdir) / "arc_learned.json"
            learning_path.write_text(json.dumps({"games": {}}), encoding="utf-8")
            bridge = HyperonBridge(enabled=False, learning_path=learning_path)

            class LegacyRecord:
                game_id = "ls20"
                step_index = 1
                action_name = "ACTION1"
                outcome = "progress"
                progress_delta = 1
                game_state = "ACTIVE"
                prior_game_state = "ACTIVE"
                decision_subgoal = "advance-level"
                prior_frame_signature = "sig-1"

            bridge._ingest_episode_records([LegacyRecord()])

            self.assertEqual(
                bridge.learned_knowledge["games"]["ls20"]["actions"]["ACTION1"]["progress"],
                1,
            )

    def test_sync_surfaces_abstract_focus_actions_from_memory_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_db_path = Path(tmpdir) / "memory.sqlite3"
            initialize_memory_schema(memory_db_path)
            upsert_abstract_memory_rows(
                memory_db_path,
                [
                    {
                        "subgoal": "advance-level",
                        "state": "ACTIVE",
                        "board_height": 2,
                        "board_width": 2,
                        "board_active_cells": 4,
                        "board_source_attr": "frame",
                        "motif_tags": [
                            "axis_horizontal",
                            "axis_vertical",
                            "mirror_alignment_candidate",
                            "piece_translation_required",
                            "selection_on_piece",
                        ],
                        "action_name": "ACTION2",
                        "lead_in_actions": ["ACTION6"],
                        "lesson_text": "Transferred mirror alignment move",
                        "salience": 2.0,
                        "confidence": 0.8,
                        "progress": 3,
                        "failed": 0,
                        "stalled": 0,
                        "total": 3,
                        "last_outcome": "progress",
                        "last_step_index": 8,
                        "source_games": ["ar25", "bp35"],
                    }
                ],
            )
            bridge = HyperonBridge(
                enabled=False,
                memory_db_path=memory_db_path,
                archive_memory_db_path=None,
                trained_priors_db_path=memory_db_path,
            )
            board = [[1, 1], [1, 1]]
            signature = compute_board_signature(board)

            bridge.sync(
                make_observation(
                    game_id="zz99",
                    game_state="ACTIVE",
                    legal_actions=["ACTION1", "ACTION2"],
                    image_input=BoardImageInput(
                        board=board,
                        height=2,
                        width=2,
                        frame_signature=signature,
                        source_attr="frame",
                    ),
                    derived_features={
                        "frame_signature": signature,
                        "board_active_cells": 4,
                        "board_default_probe": {"x": 1, "y": 1},
                        "board_source_attr": "frame",
                    },
                ),
                EpisodicMemory(),
                [GoalState(name="finish-episode", target_state="WIN")],
            )

            bundle = bridge.current_memory_bundle()
            self.assertTrue(bundle.abstract_memories)
            self.assertEqual(
                bundle.working_memory["abstract_focus_actions"][0]["action_name"],
                "ACTION2",
            )
            self.assertIn(
                "abstraction-transfer",
                bridge.facts["memory_bundle"]["retrieval_summary"]["top_recall_sources"],
            )


if __name__ == "__main__":
    unittest.main()
