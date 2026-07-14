from __future__ import annotations

import unittest
from dataclasses import dataclass
from enum import Enum
from unittest.mock import patch

from agi_core.adapters.arc import ArcAdapter
from agi_core.image_input import compute_board_signature
from agi_core.types import ActionDecision


class StubGameState(Enum):
    NOT_PLAYED = "not_played"
    GAME_OVER = "game_over"
    ACTIVE = "active"
    WIN = "win"


class StubGameAction(Enum):
    RESET = "reset"
    ACTION1 = "action1"
    ACTION2 = "action2"
    ACTION3 = "action3"
    ACTION4 = "action4"
    ACTION5 = "action5"
    ACTION6 = "action6"

    def is_complex(self) -> bool:
        return self is type(self).ACTION6

    def set_data(self, payload) -> None:
        self.payload = payload


@dataclass
class StubFrame:
    state: StubGameState
    levels_completed: int = 0
    last_action: StubGameAction | None = None
    available_actions: list[StubGameAction] | None = None
    step_index: int | None = None
    frame: object | None = None
    frames: object | None = None
    board: object | None = None
    grid: object | None = None


class ArcAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = ArcAdapter(
            game_action_cls=StubGameAction,
            default_game_id="ls20",
            deterministic_action="ACTION2",
        )

    def test_observe_normalizes_frame_state_and_actions(self) -> None:
        board = [[1, 2], [3, 8]]
        frame = StubFrame(
            state=StubGameState.ACTIVE,
            levels_completed=3,
            last_action=StubGameAction.ACTION1,
            available_actions=[StubGameAction.ACTION2, StubGameAction.ACTION6],
            step_index=7,
            frame=[board],
        )

        observation = self.adapter.observe([frame], frame)

        self.assertEqual(observation.game_id, "ls20")
        self.assertEqual(observation.game_state, "ACTIVE")
        self.assertEqual(observation.step_index, 7)
        self.assertEqual(observation.levels_completed, 3)
        self.assertEqual(observation.last_action, "ACTION1")
        self.assertEqual(observation.legal_actions, ["ACTION2", "ACTION6"])
        self.assertTrue(observation.derived_features["step_index_reliable"])
        self.assertIsNotNone(observation.image_input)
        assert observation.image_input is not None
        self.assertEqual(observation.image_input.board, board)
        self.assertEqual(observation.image_input.source_attr, "frame")
        self.assertEqual(observation.derived_features["board_height"], 2)
        self.assertEqual(observation.derived_features["board_width"], 2)
        self.assertEqual(observation.derived_features["board_active_cells"], 4)
        self.assertEqual(observation.derived_features["board_default_probe"], {"x": 1, "y": 1})
        self.assertEqual(observation.derived_features["board_source_attr"], "frame")
        self.assertTrue(observation.derived_features["has_image_input"])
        self.assertEqual(
            observation.derived_features["frame_signature"],
            compute_board_signature(board),
        )

    def test_observe_prefers_first_frame_layer_over_late_overlay_on_late_levels(self) -> None:
        visible_board = [
            [0, 10, 0],
            [5, 5, 5],
            [0, 11, 0],
        ]
        overlay_board = [
            [10, 10, 10],
            [0, 0, 0],
            [11, 11, 11],
        ]
        frame = StubFrame(
            state=StubGameState.ACTIVE,
            levels_completed=3,
            frame=[visible_board, overlay_board],
        )

        observation = self.adapter.observe([frame], frame)

        assert observation.image_input is not None
        self.assertEqual(observation.image_input.board, visible_board)
        self.assertEqual(observation.image_input.source_attr, "frame[0]")
        self.assertEqual(
            observation.derived_features["frame_signature"],
            compute_board_signature(visible_board),
        )

    def test_observe_normalizes_integer_action_ids(self) -> None:
        frame = StubFrame(
            state=StubGameState.ACTIVE,
            available_actions=[1, 4, 6],
        )

        observation = self.adapter.observe([frame], frame)

        self.assertEqual(observation.legal_actions, ["ACTION1", "ACTION4", "ACTION6"])

    def test_observe_filters_unsupported_explicit_actions(self) -> None:
        frame = StubFrame(
            state=StubGameState.ACTIVE,
            available_actions=["ACTION7", StubGameAction.ACTION2, "bogus"],
        )

        observation = self.adapter.observe([frame], frame)

        self.assertEqual(observation.legal_actions, ["ACTION2"])
        self.assertTrue(observation.derived_features["legal_actions_known"])

    def test_observe_fails_closed_when_explicit_actions_are_all_unsupported(self) -> None:
        frame = StubFrame(
            state=StubGameState.ACTIVE,
            available_actions=["ACTION7", "ACTION9"],
        )

        observation = self.adapter.observe([frame], frame)

        self.assertEqual(observation.legal_actions, [])
        self.assertTrue(observation.derived_features["legal_actions_known"])

    def test_observe_reports_explicit_empty_actions_as_known_empty_legality(self) -> None:
        frame = StubFrame(
            state=StubGameState.ACTIVE,
            available_actions=[],
        )

        observation = self.adapter.observe([frame], frame)

        self.assertEqual(observation.legal_actions, [])
        self.assertTrue(observation.derived_features["legal_actions_known"])

    def test_reset_state_reports_reset_as_only_action(self) -> None:
        frame = StubFrame(state=StubGameState.GAME_OVER)
        observation = self.adapter.observe([frame], frame)

        self.assertEqual(observation.legal_actions, ["RESET"])
        fallback = self.adapter.fallback_decision(observation)
        self.assertEqual(fallback.action_name, "RESET")

    def test_reset_state_with_explicit_empty_actions_still_reports_reset(self) -> None:
        frame = StubFrame(
            state=StubGameState.GAME_OVER,
            available_actions=[],
        )

        observation = self.adapter.observe([frame], frame)

        self.assertEqual(observation.legal_actions, ["RESET"])
        self.assertTrue(observation.derived_features["legal_actions_known"])

    def test_reset_state_with_only_unsupported_explicit_actions_still_reports_reset(self) -> None:
        frame = StubFrame(
            state=StubGameState.GAME_OVER,
            available_actions=["ACTION7", "ACTION9"],
        )

        observation = self.adapter.observe([frame], frame)

        self.assertEqual(observation.legal_actions, ["RESET"])
        self.assertTrue(observation.derived_features["legal_actions_known"])

    def test_reset_state_with_supported_non_reset_actions_still_reports_reset(self) -> None:
        frame = StubFrame(
            state=StubGameState.GAME_OVER,
            available_actions=[StubGameAction.ACTION1, StubGameAction.ACTION2],
        )

        observation = self.adapter.observe([frame], frame)

        self.assertEqual(observation.legal_actions, ["RESET"])
        self.assertTrue(observation.derived_features["legal_actions_known"])

    def test_active_frame_without_explicit_actions_fails_closed(self) -> None:
        frame = StubFrame(state=StubGameState.ACTIVE)

        observation = self.adapter.observe([frame], frame)

        self.assertEqual(observation.legal_actions, [])
        self.assertFalse(observation.derived_features["legal_actions_known"])
        self.assertEqual(self.adapter.fallback_action_name(observation), "ACTION2")
        self.assertEqual(observation.step_index, 0)
        self.assertFalse(observation.derived_features["step_index_reliable"])

    def test_fallback_decision_marks_board_probe_subgoal_when_action6_is_default(self) -> None:
        adapter = ArcAdapter(
            game_action_cls=StubGameAction,
            default_game_id="ls20",
            deterministic_action="ACTION6",
        )
        board = [[0, 0, 0], [0, 5, 5], [0, 5, 5]]
        frame = StubFrame(
            state=StubGameState.ACTIVE,
            frame=[board],
        )

        observation = adapter.observe([frame], frame)
        fallback = adapter.fallback_decision(observation)

        self.assertEqual(fallback.action_name, "ACTION6")
        self.assertEqual(fallback.subgoal, "inspect-active-board")
        self.assertEqual(fallback.metadata, {"x": 2, "y": 2})

    def test_custom_reset_state_names_drive_legality_and_fallback(self) -> None:
        adapter = ArcAdapter(
            game_action_cls=StubGameAction,
            default_game_id="ls20",
            deterministic_action="ACTION2",
            reset_state_names=("READY", "LOSE"),
        )
        frame = StubFrame(state="lose")

        observation = adapter.observe([frame], frame)

        self.assertEqual(observation.game_state, "LOSE")
        self.assertEqual(observation.legal_actions, ["RESET"])
        self.assertEqual(adapter.fallback_action_name(observation), "RESET")

    def test_observe_extracts_board_from_frames_attr(self) -> None:
        board = [[5, 6, 7], [8, 9, 0]]
        frame = StubFrame(
            state=StubGameState.ACTIVE,
            available_actions=[StubGameAction.ACTION1],
            frames=[board],
        )

        observation = self.adapter.observe([frame], frame)

        assert observation.image_input is not None
        self.assertEqual(observation.image_input.board, board)
        self.assertEqual(observation.image_input.source_attr, "frames")
        self.assertEqual(observation.image_input.width, 3)
        self.assertEqual(observation.image_input.height, 2)

    def test_observe_extracts_board_from_board_attr(self) -> None:
        board = [[1], [2], [3]]
        frame = StubFrame(
            state=StubGameState.ACTIVE,
            available_actions=[StubGameAction.ACTION1],
            board=board,
        )

        observation = self.adapter.observe([frame], frame)

        assert observation.image_input is not None
        self.assertEqual(observation.image_input.board, board)
        self.assertEqual(observation.image_input.source_attr, "board")

    def test_observe_extracts_board_from_grid_attr(self) -> None:
        board = [[4, 4], [4, 4]]
        frame = StubFrame(
            state=StubGameState.ACTIVE,
            available_actions=[StubGameAction.ACTION1],
            grid=board,
        )

        observation = self.adapter.observe([frame], frame)

        assert observation.image_input is not None
        self.assertEqual(observation.image_input.board, board)
        self.assertEqual(observation.image_input.source_attr, "grid")

    def test_observe_uses_analyzer_beam_detection_not_raw_color4_presence(self) -> None:
        board = [[9] * 8 for _ in range(8)]
        for y in range(8):
            board[y][3] = 10
        for y in (4, 5):
            for x in (1, 2):
                board[y][x] = 5
        for y in (4, 5):
            board[y][5] = 4
            board[y][6] = 4
        frame = StubFrame(
            state=StubGameState.ACTIVE,
            available_actions=[StubGameAction.ACTION1],
            frame=[board],
        )

        observation = self.adapter.observe([frame], frame)

        self.assertFalse(observation.derived_features["has_beam"])

    def test_observe_surfaces_analyzer_perception_failure(self) -> None:
        board = [[1, 2], [3, 8]]
        frame = StubFrame(
            state=StubGameState.ACTIVE,
            available_actions=[StubGameAction.ACTION1],
            frame=[board],
        )

        with patch("agi_core.adapters.arc.build_game_model", side_effect=RuntimeError("boom")):
            observation = self.adapter.observe([frame], frame)

        self.assertTrue(observation.derived_features["analyzer_perception_failed"])
        self.assertEqual(
            observation.derived_features["analyzer_perception_error"],
            "RuntimeError: boom",
        )
        self.assertIsNone(observation.derived_features["player_position"])
        self.assertEqual(observation.derived_features["goal_positions"], [])
        self.assertFalse(observation.derived_features["has_beam"])
        self.assertFalse(observation.derived_features["beam_visible"])
        self.assertFalse(observation.derived_features["beam_evidence_uncertain"])

    def test_observe_surfaces_uncertain_beam_evidence_on_analyzer_failure(self) -> None:
        board = [[9, 4], [3, 4]]
        frame = StubFrame(
            state=StubGameState.ACTIVE,
            available_actions=[StubGameAction.ACTION1],
            frame=[board],
        )

        with patch("agi_core.adapters.arc.build_game_model", side_effect=RuntimeError("boom")):
            observation = self.adapter.observe([frame], frame)

        self.assertTrue(observation.derived_features["analyzer_perception_failed"])
        self.assertFalse(observation.derived_features["has_beam"])
        self.assertTrue(observation.derived_features["beam_visible"])
        self.assertTrue(observation.derived_features["beam_evidence_uncertain"])

    def test_observe_leaves_image_input_empty_when_board_missing(self) -> None:
        frame = StubFrame(
            state=StubGameState.ACTIVE,
            available_actions=[StubGameAction.ACTION1],
        )

        observation = self.adapter.observe([frame], frame)

        self.assertIsNone(observation.image_input)
        self.assertEqual(observation.derived_features["board_height"], 0)
        self.assertEqual(observation.derived_features["board_width"], 0)

    def test_observe_reuses_last_image_input_on_late_level_missing_board(self) -> None:
        board = [[5, 6, 7], [8, 9, 0]]
        valid_frame = StubFrame(
            state=StubGameState.ACTIVE,
            levels_completed=3,
            available_actions=[StubGameAction.ACTION1],
            frame=[board],
        )
        missing_board_frame = StubFrame(
            state=StubGameState.ACTIVE,
            available_actions=[StubGameAction.ACTION1],
        )

        first = self.adapter.observe([valid_frame], valid_frame)
        second = self.adapter.observe([valid_frame, missing_board_frame], missing_board_frame)

        assert first.image_input is not None
        assert second.image_input is not None
        self.assertEqual(second.levels_completed, 3)
        self.assertEqual(second.image_input.board, board)
        self.assertEqual(second.image_input.source_attr, "frame")
        self.assertEqual(
            second.derived_features["frame_signature"],
            compute_board_signature(board),
        )
        self.assertTrue(second.derived_features["has_image_input"])

    def test_observe_reuses_last_image_input_on_late_level_empty_board_payload(self) -> None:
        board = [[5, 6, 7], [8, 9, 0]]
        valid_frame = StubFrame(
            state=StubGameState.ACTIVE,
            levels_completed=3,
            available_actions=[StubGameAction.ACTION1],
            frame=[board],
        )
        degraded_frame = StubFrame(
            state=StubGameState.ACTIVE,
            available_actions=[StubGameAction.ACTION1],
            frame=[[]],
        )

        first = self.adapter.observe([valid_frame], valid_frame)
        second = self.adapter.observe([valid_frame, degraded_frame], degraded_frame)

        assert first.image_input is not None
        assert second.image_input is not None
        self.assertEqual(second.levels_completed, 3)
        self.assertEqual(second.image_input.board, board)
        self.assertEqual(second.image_input.source_attr, "frame")
        self.assertTrue(second.derived_features["has_image_input"])

    def test_observe_does_not_reuse_missing_board_before_late_levels(self) -> None:
        board = [[5, 6, 7], [8, 9, 0]]
        valid_frame = StubFrame(
            state=StubGameState.ACTIVE,
            levels_completed=2,
            available_actions=[StubGameAction.ACTION1],
            frame=[board],
        )
        missing_board_frame = StubFrame(
            state=StubGameState.ACTIVE,
            levels_completed=2,
            available_actions=[StubGameAction.ACTION1],
        )

        first = self.adapter.observe([valid_frame], valid_frame)
        second = self.adapter.observe([valid_frame, missing_board_frame], missing_board_frame)

        assert first.image_input is not None
        self.assertIsNone(second.image_input)
        self.assertEqual(second.levels_completed, 2)
        self.assertEqual(second.derived_features["board_active_cells"], 0)
        self.assertEqual(second.derived_features["board_default_probe"], {})
        self.assertIsNone(second.derived_features["board_source_attr"])
        self.assertFalse(second.derived_features["has_image_input"])

    def test_default_action6_metadata_uses_board_center_when_available(self) -> None:
        board = [[1, 2, 3, 4], [5, 6, 7, 8]]
        frame = StubFrame(
            state=StubGameState.ACTIVE,
            available_actions=[StubGameAction.ACTION6],
            frame=[board],
        )

        observation = self.adapter.observe([frame], frame)

        self.assertEqual(observation.legal_actions, ["ACTION6"])
        self.assertEqual(
            self.adapter.default_metadata_for_action("ACTION6"),
            {"x": 2, "y": 1},
        )

    def test_default_action6_metadata_targets_centroid_of_active_cells(self) -> None:
        board = [
            [0, 0, 0, 0],
            [0, 9, 9, 0],
            [0, 9, 9, 0],
            [0, 0, 0, 0],
        ]
        frame = StubFrame(
            state=StubGameState.ACTIVE,
            available_actions=[StubGameAction.ACTION6],
            frame=[board],
        )

        self.adapter.observe([frame], frame)

        self.assertEqual(
            self.adapter.default_metadata_for_action("ACTION6"),
            {"x": 2, "y": 2},
        )

    def test_default_action6_metadata_prefers_transition_target_on_dense_boards(self) -> None:
        board = [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
        frame = StubFrame(
            state=StubGameState.ACTIVE,
            available_actions=[StubGameAction.ACTION6],
            frame=[board],
        )

        observation = self.adapter.observe([frame], frame)

        self.assertEqual(observation.derived_features["board_default_probe"], {"x": 2, "y": 1})
        self.assertEqual(
            self.adapter.default_metadata_for_action("ACTION6"),
            {"x": 2, "y": 1},
        )

    def test_complex_action_maps_coordinates(self) -> None:
        decision = ActionDecision(
            action_name="ACTION6",
            source="llm",
            confidence=0.5,
            reason="click center",
            metadata={"x": 9, "y": 13},
        )

        action = self.adapter.to_game_action(decision)

        self.assertIs(action, StubGameAction.ACTION6)
        self.assertEqual(getattr(action, "payload", None), {"x": 9, "y": 13})

    def test_complex_action_invalid_metadata_falls_back_to_board_probe_default(self) -> None:
        board = [[1, 2, 3], [4, 5, 6]]
        frame = StubFrame(
            state=StubGameState.ACTIVE,
            available_actions=[StubGameAction.ACTION6],
            frame=[board],
        )
        self.adapter.observe([frame], frame)
        decision = ActionDecision(
            action_name="ACTION6",
            source="llm",
            confidence=0.5,
            reason="bad payload",
            metadata={"x": "bad", "y": object()},
        )

        action = self.adapter.to_game_action(decision)

        self.assertIs(action, StubGameAction.ACTION6)
        self.assertEqual(getattr(action, "payload", None), {"x": 1, "y": 1})

    def test_complex_action_invalid_metadata_falls_back_to_safe_center_without_board(self) -> None:
        decision = ActionDecision(
            action_name="ACTION6",
            source="llm",
            confidence=0.5,
            reason="bad payload",
            metadata={"x": None, "y": "oops"},
        )

        action = self.adapter.to_game_action(decision)

        self.assertIs(action, StubGameAction.ACTION6)
        self.assertEqual(getattr(action, "payload", None), {"x": 32, "y": 32})

    def test_unsupported_decision_action_falls_back_to_reset(self) -> None:
        decision = ActionDecision(
            action_name="ACTION7",
            source="symbolic",
            confidence=0.5,
            reason="unsupported action",
        )

        action = self.adapter.to_game_action(decision)

        self.assertIs(action, StubGameAction.RESET)
