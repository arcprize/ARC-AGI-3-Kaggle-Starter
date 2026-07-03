from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agi_core.storage import initialize_schema, upsert_training_counts
from agi_core.knowledge.hyperon_bridge import HyperonBridge
from agi_core.memory import EpisodicMemory
from agi_core.types import ActionDecision, ActionProposal, GoalState, Observation


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

    def test_persistent_learning_updates_action_priors_and_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            learning_path = Path(tmpdir) / "arc_learned.json"
            bridge = HyperonBridge(enabled=False, learning_path=learning_path)
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

            bridge.sync(
                after,
                memory,
                [GoalState(name="finish-episode", target_state="WIN")],
            )

            priors = bridge.run_rules("action-priors")
            self.assertEqual(priors[0]["action_name"], "ACTION1")
            self.assertGreater(priors[0]["confidence"], priors[1]["confidence"])
            self.assertTrue(learning_path.exists())

            learned = json.loads(learning_path.read_text(encoding="utf-8"))
            self.assertEqual(
                learned["games"]["ls20"]["actions"]["ACTION1"]["progress"],
                1,
            )

            second_bridge = HyperonBridge(enabled=False, learning_path=learning_path)
            second_bridge.sync(
                make_observation(step_index=2, last_action="ACTION1"),
                EpisodicMemory(),
                [GoalState(name="finish-episode", target_state="WIN")],
            )
            self.assertEqual(
                second_bridge.learned_knowledge["games"]["ls20"]["actions"]["ACTION1"]["progress"],
                1,
            )

    def test_state_and_subgoal_learning_bias_current_symbolic_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            learning_path = Path(tmpdir) / "arc_learned.json"
            bridge = HyperonBridge(enabled=False, learning_path=learning_path)
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

            bridge.sync(
                after,
                memory,
                [GoalState(name="finish-episode", target_state="WIN")],
            )

            priors = bridge.run_rules("action-priors:break-stall")
            self.assertEqual(priors[0]["action_name"], "ACTION2")

            learned = json.loads(learning_path.read_text(encoding="utf-8"))
            self.assertEqual(
                learned["games"]["ls20"]["states"]["ACTIVE"]["actions"]["ACTION2"]["progress"],
                1,
            )
            self.assertEqual(
                learned["games"]["ls20"]["subgoals"]["break-stall"]["actions"]["ACTION2"]["progress"],
                1,
            )

    def test_frame_signature_learning_biases_matching_symbolic_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            learning_path = Path(tmpdir) / "arc_learned.json"
            bridge = HyperonBridge(enabled=False, learning_path=learning_path)
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

            learned = json.loads(learning_path.read_text(encoding="utf-8"))
            self.assertEqual(
                learned["games"]["ls20"]["frame_signatures"]["sig-live"]["actions"]["ACTION2"][
                    "progress"
                ],
                1,
            )

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


if __name__ == "__main__":
    unittest.main()
