from __future__ import annotations

import unittest

from agi_core.image_input import BoardImageInput
from agi_core.memory import EpisodicMemory
from agi_core.types import ActionDecision, Observation


class EpisodicMemoryTests(unittest.TestCase):
    def test_working_memory_snapshot_exposes_pending_decision_guidance(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        observation = Observation(
            game_id="ls20",
            step_index=0,
            game_state="ACTIVE",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION1", "ACTION2", "ACTION3"],
            derived_features={"frame_signature": "sig-live"},
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

        snapshot = memory.working_memory_snapshot()

        self.assertEqual(snapshot["active_state"], "ACTIVE")
        self.assertEqual(snapshot["active_frame_signature"], "sig-live")
        self.assertEqual(snapshot["active_subgoal"], "inspect-active-board")
        self.assertEqual(snapshot["legal_actions"], ["ACTION1", "ACTION2", "ACTION3"])
        self.assertEqual(snapshot["recent_action_window"], ["ACTION3"])
        self.assertEqual(snapshot["recommended_actions"], ["ACTION3"])
        self.assertEqual(snapshot["semantic_focus_actions"], [])
        self.assertEqual(snapshot["procedural_focus_actions"], [])

    def test_working_memory_snapshot_exposes_runtime_focus_actions(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        before = Observation(
            game_id="ls20",
            step_index=0,
            game_state="ACTIVE",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION1", "ACTION2", "ACTION3"],
            derived_features={"frame_signature": "sig-focus-before"},
        )
        progress_after = Observation(
            game_id="ls20",
            step_index=1,
            game_state="ACTIVE",
            levels_completed=1,
            last_action="ACTION2",
            legal_actions=["ACTION1", "ACTION2", "ACTION3"],
            derived_features={"frame_signature": "sig-focus-after"},
        )
        memory.note_observation(before)
        memory.note_decision(
            ActionDecision(
                action_name="ACTION2",
                source="symbolic",
                confidence=0.5,
                reason="focus seed",
                subgoal="break-stall",
            )
        )
        memory.note_observation(progress_after)

        snapshot = memory.working_memory_snapshot()

        self.assertEqual(snapshot["active_state"], "ACTIVE")
        self.assertEqual(snapshot["active_frame_signature"], "sig-focus-after")
        self.assertEqual(snapshot["legal_actions"], ["ACTION1", "ACTION2", "ACTION3"])
        self.assertEqual(snapshot["board_context_summary"]["board_active_cells"], 0)
        self.assertEqual(snapshot["recent_action_window"], ["ACTION2"])
        self.assertEqual(snapshot["recommended_actions"], ["ACTION2"])
        self.assertEqual(snapshot["suppressed_actions"], [])
        self.assertFalse(snapshot["basin_exhausted"])
        self.assertEqual(snapshot["basin_visits"], 0)
        self.assertEqual(snapshot["semantic_focus_actions"], ["ACTION2"])
        self.assertEqual(snapshot["procedural_focus_actions"], ["ACTION2"])

    def test_working_memory_snapshot_exposes_runtime_open_failure_modes(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        before = Observation(
            game_id="ls20",
            step_index=0,
            game_state="ACTIVE",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-failure-before"},
        )
        after = Observation(
            game_id="ls20",
            step_index=1,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-failure-after"},
        )
        memory.note_observation(before)
        memory.note_decision(
            ActionDecision(
                action_name="ACTION1",
                source="symbolic",
                confidence=0.5,
                reason="failure seed",
                subgoal="break-stall",
            )
        )
        memory.note_observation(after)

        snapshot = memory.working_memory_snapshot()

        self.assertEqual(snapshot["open_failure_modes"][0]["action_name"], "ACTION1")
        self.assertEqual(snapshot["open_failure_modes"][0]["subgoal"], "break-stall")
        self.assertEqual(snapshot["open_failure_modes"][0]["frame_signature"], "sig-failure-before")

    def test_working_memory_snapshot_marks_runtime_basin_exhaustion(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        for index, action_name in enumerate(("ACTION1", "ACTION2")):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-basin"},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-basin"},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="symbolic",
                    confidence=0.5,
                    reason="basin seed",
                )
            )
            memory.note_observation(after)

        snapshot = memory.working_memory_snapshot()

        self.assertTrue(snapshot["basin_exhausted"])
        self.assertEqual(snapshot["basin_visits"], 2)
        self.assertEqual(snapshot["suppressed_actions"], ["ACTION1", "ACTION2"])

    def test_finalized_transition_records_extended_fields(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        before = Observation(
            game_id="ls20",
            step_index=0,
            game_state="ACTIVE",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION1", "ACTION2"],
            image_input=BoardImageInput(
                board=[[1, 0], [0, 1]],
                height=2,
                width=2,
                frame_signature="sig-before",
                source_attr="frame",
            ),
            derived_features={"frame_signature": "sig-before"},
        )
        after = Observation(
            game_id="ls20",
            step_index=1,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-after"},
        )
        memory.note_observation(before)
        memory.note_decision(
            ActionDecision(
                action_name="ACTION1",
                source="symbolic",
                confidence=0.5,
                reason="seed",
                subgoal="break-stall",
            )
        )
        memory.note_observation(after)

        [record] = memory.drain_new_records()
        self.assertEqual(record.next_game_state, "ACTIVE")
        self.assertEqual(record.next_frame_signature, "sig-after")
        self.assertEqual(record.prior_legal_actions, ["ACTION1", "ACTION2"])
        self.assertEqual(record.stagnation_before, 0)
        self.assertEqual(record.stagnation_after, 1)
        self.assertEqual(record.prior_board_height, 2)
        self.assertEqual(record.prior_board_width, 2)
        self.assertEqual(record.prior_board_active_cells, 2)
        self.assertEqual(record.prior_board_source_attr, "frame")
        self.assertEqual(record.prior_board_default_probe, {"x": 1, "y": 1})

    def test_is_basin_exhausted_only_when_all_actions_stall_without_progress(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        for index, action_name in enumerate(("ACTION1", "ACTION2")):
            before = Observation(
                game_id="ls20",
                step_index=index,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=None,
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-loop"},
            )
            after = Observation(
                game_id="ls20",
                step_index=index + 1,
                game_state="NOT_FINISHED",
                levels_completed=0,
                last_action=action_name,
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"frame_signature": "sig-loop"},
            )
            memory.note_observation(before)
            memory.note_decision(
                ActionDecision(
                    action_name=action_name,
                    source="symbolic",
                    confidence=0.5,
                    reason="stall",
                )
            )
            memory.note_observation(after)
            memory.drain_new_records()

        current = Observation(
            game_id="ls20",
            step_index=3,
            game_state="NOT_FINISHED",
            levels_completed=0,
            last_action="ACTION2",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-loop"},
        )
        self.assertTrue(memory.is_basin_exhausted(current))

    def test_custom_terminal_state_names_drive_outcome_classification(self) -> None:
        memory = EpisodicMemory(
            win_state_name="DONE",
            reset_state_names=("READY", "LOSE"),
        )
        memory.bind_game("ls20")
        before = Observation(
            game_id="ls20",
            step_index=3,
            game_state="ACTIVE",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION1"],
        )
        win_after = Observation(
            game_id="ls20",
            step_index=4,
            game_state="DONE",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1"],
        )
        lose_after = Observation(
            game_id="ls20",
            step_index=5,
            game_state="LOSE",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["RESET"],
        )

        memory.note_observation(before)
        memory.note_decision(
            ActionDecision(
                action_name="ACTION1",
                source="symbolic",
                confidence=0.5,
                reason="custom-win",
            )
        )
        memory.note_observation(win_after)
        [win_record] = memory.drain_new_records()
        self.assertEqual(win_record.outcome, "progress")

        memory.note_decision(
            ActionDecision(
                action_name="ACTION1",
                source="symbolic",
                confidence=0.5,
                reason="custom-reset-loss",
            )
        )
        memory.note_observation(lose_after)
        [lose_record] = memory.drain_new_records()
        self.assertEqual(lose_record.outcome, "failed")

    def test_same_game_restart_boundary_clears_stale_episode_state(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        before = Observation(
            game_id="ls20",
            step_index=4,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-before", "step_index_reliable": True},
        )
        after = Observation(
            game_id="ls20",
            step_index=5,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-after", "step_index_reliable": True},
        )
        restarted = Observation(
            game_id="ls20",
            step_index=1,
            game_state="ACTIVE",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-restart", "step_index_reliable": True},
        )

        memory.note_observation(before)
        memory.note_decision(
            ActionDecision(
                action_name="ACTION1",
                source="fallback",
                confidence=0.1,
                reason="score-floor restart stale",
            )
        )
        memory.note_llm_call()
        memory.mark_pending_score_floor_reopen(
            before,
            candidate_actions=["ACTION1"],
            debug={"frame_signatures": ["sig-before"]},
        )
        memory.note_observation(after)
        self.assertEqual(memory.stagnation_steps, 1)
        self.assertEqual(memory.llm_calls, 1)
        self.assertEqual(memory.recent_actions(), ["ACTION1"])
        self.assertIsNotNone(memory.pending_score_floor_reopen(after))

        memory.note_observation(restarted)

        self.assertEqual(memory.stagnation_steps, 0)
        self.assertEqual(memory.llm_calls, 0)
        self.assertEqual(memory.recent_actions(), [])
        self.assertEqual(memory.drain_new_records(), [])
        self.assertIsNone(memory.pending_score_floor_reopen(restarted))

    def test_reset_exit_boundary_clears_stale_episode_state(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        reset_observation = Observation(
            game_id="ls20",
            step_index=9,
            game_state="GAME_OVER",
            levels_completed=0,
            last_action="RESET",
            legal_actions=["RESET"],
            derived_features={"step_index_reliable": True},
        )
        resumed = Observation(
            game_id="ls20",
            step_index=10,
            game_state="ACTIVE",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION1"],
            derived_features={"frame_signature": "sig-resumed", "step_index_reliable": True},
        )

        memory.note_observation(reset_observation)
        memory.note_decision(
            ActionDecision(
                action_name="RESET",
                source="fallback",
                confidence=0.1,
                reason="restart",
            )
        )
        memory.note_llm_call()

        memory.note_observation(resumed)

        self.assertEqual(memory.stagnation_steps, 0)
        self.assertEqual(memory.llm_calls, 0)
        self.assertEqual(memory.recent_actions(), [])
        self.assertEqual(memory.drain_new_records(), [])

    def test_pending_score_floor_reopen_uses_observation_tick_when_step_index_is_unreliable(self) -> None:
        memory = EpisodicMemory()
        observation = Observation(
            game_id="ls20",
            step_index=0,
            game_state="ACTIVE",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-a", "step_index_reliable": False},
        )
        later = Observation(
            game_id="ls20",
            step_index=10,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-b", "step_index_reliable": False},
        )
        much_later = Observation(
            game_id="ls20",
            step_index=100,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"frame_signature": "sig-c", "step_index_reliable": False},
        )

        memory.note_observation(observation)
        memory.mark_pending_score_floor_reopen(
            observation,
            candidate_actions=["ACTION1"],
            debug={"frame_signatures": ["sig-a"]},
        )
        memory.note_observation(later)
        self.assertIsNotNone(memory.pending_score_floor_reopen(later))

        for index in range(3):
            memory.note_observation(
                Observation(
                    game_id="ls20",
                    step_index=101 + index,
                    game_state="ACTIVE",
                    levels_completed=0,
                    last_action="ACTION1",
                    legal_actions=["ACTION1", "ACTION2"],
                    derived_features={
                        "frame_signature": f"sig-extra-{index}",
                        "step_index_reliable": False,
                    },
                )
            )

        memory.note_observation(much_later)
        self.assertIsNone(memory.pending_score_floor_reopen(much_later))

    def test_recent_score_floor_fallback_uses_observation_tick_when_step_index_is_unreliable(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        before = Observation(
            game_id="ls20",
            step_index=0,
            game_state="ACTIVE",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"step_index_reliable": False},
        )
        after = Observation(
            game_id="ls20",
            step_index=10,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"step_index_reliable": False},
        )
        expired = Observation(
            game_id="ls20",
            step_index=100,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"step_index_reliable": False},
        )

        memory.note_observation(before)
        memory.note_decision(
            ActionDecision(
                action_name="ACTION1",
                source="fallback",
                confidence=0.05,
                reason="score-floor executive fallback",
            )
        )
        memory.note_observation(after)
        self.assertTrue(memory.recent_score_floor_fallback(after))

        memory.note_observation(
            Observation(
                game_id="ls20",
                step_index=101,
                game_state="ACTIVE",
                levels_completed=0,
                last_action="ACTION1",
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"step_index_reliable": False},
            )
        )
        memory.note_observation(expired)
        self.assertFalse(memory.recent_score_floor_fallback(expired))

    def test_recent_probe_dead_end_escape_uses_observation_tick_when_step_index_is_unreliable(self) -> None:
        memory = EpisodicMemory()
        memory.bind_game("ls20")
        before = Observation(
            game_id="ls20",
            step_index=0,
            game_state="ACTIVE",
            levels_completed=0,
            last_action=None,
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"step_index_reliable": False},
        )
        after = Observation(
            game_id="ls20",
            step_index=10,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"step_index_reliable": False},
        )
        expired = Observation(
            game_id="ls20",
            step_index=100,
            game_state="ACTIVE",
            levels_completed=0,
            last_action="ACTION1",
            legal_actions=["ACTION1", "ACTION2"],
            derived_features={"step_index_reliable": False},
        )

        memory.note_observation(before)
        memory.note_decision(
            ActionDecision(
                action_name="ACTION1",
                source="fallback",
                confidence=0.05,
                reason="probe-dead-end memory escape",
            )
        )
        memory.note_observation(after)
        self.assertTrue(memory.recent_probe_dead_end_escape(after))

        memory.note_observation(
            Observation(
                game_id="ls20",
                step_index=101,
                game_state="ACTIVE",
                levels_completed=0,
                last_action="ACTION1",
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"step_index_reliable": False},
            )
        )
        memory.note_observation(
            Observation(
                game_id="ls20",
                step_index=102,
                game_state="ACTIVE",
                levels_completed=0,
                last_action="ACTION1",
                legal_actions=["ACTION1", "ACTION2"],
                derived_features={"step_index_reliable": False},
            )
        )
        memory.note_observation(expired)
        self.assertFalse(memory.recent_probe_dead_end_escape(expired))


if __name__ == "__main__":
    unittest.main()
