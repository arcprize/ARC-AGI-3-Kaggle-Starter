from __future__ import annotations

import unittest

from agi_core.image_input import BoardImageInput
from agi_core.memory import EpisodicMemory
from agi_core.types import ActionDecision, Observation


class EpisodicMemoryTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
