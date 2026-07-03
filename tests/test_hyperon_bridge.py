from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agi_core.storage import (
    import_legacy_learning_json,
    initialize_memory_schema,
    initialize_schema,
    load_memory_bundle,
    upsert_training_counts,
)
from agi_core.knowledge.hyperon_bridge import HyperonBridge
from agi_core.memory import EpisodicMemory
from agi_core.types import (
    ActionDecision,
    ActionProposal,
    EpisodeRecord,
    GoalState,
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


if __name__ == "__main__":
    unittest.main()
