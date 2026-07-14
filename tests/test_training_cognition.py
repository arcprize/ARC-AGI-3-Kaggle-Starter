from __future__ import annotations

import unittest
from pathlib import Path

from agi_core.training.demo_zip import (
    _CognitionContext,
    _create_cognition_context,
    _detect_bg_color,
    _infer_player_position,
    _infer_goal_positions,
    _update_cognition,
    TrainingStep,
)


class TestTrainingCognition(unittest.TestCase):
    """Smoke tests for the post-hoc cognition training integration."""

    def test_create_cognition_context(self):
        ctx = _create_cognition_context()
        self.assertIsNotNone(ctx.working_memory)
        self.assertIsNotNone(ctx.meta_cognition)
        self.assertIsNotNone(ctx.goal_stack)
        self.assertIsNotNone(ctx.chunk_library)
        self.assertIsNotNone(ctx.consolidated_memory)
        self.assertIsNotNone(ctx.attention)

    def test_detect_bg_color(self):
        board = [[9, 9, 9], [9, 0, 9], [9, 9, 9]]
        self.assertEqual(_detect_bg_color(board), 9)
        board2 = [[5, 5], [5, 5]]
        self.assertEqual(_detect_bg_color(board2), 5)

    def test_infer_player_position(self):
        from agi_core.image_input import BoardImageInput
        board = [[9, 9, 9, 9], [9, 1, 0, 9], [9, 0, 0, 9], [9, 9, 9, 9]]
        img = BoardImageInput(board=board, height=4, width=4, frame_signature="s", source_attr="test")
        pos = _infer_player_position(img)
        self.assertIsNotNone(pos)
        self.assertEqual(pos, (1, 1))

    def test_infer_goal_positions(self):
        from agi_core.image_input import BoardImageInput
        board = [[9, 9, 9], [9, 11, 9], [9, 9, 9]]
        img = BoardImageInput(board=board, height=3, width=3, frame_signature="s", source_attr="test")
        goals = _infer_goal_positions(img)
        self.assertEqual(len(goals), 1)
        self.assertEqual(goals[0], (1, 1))

    def test_update_cognition_returns_metrics(self):
        ctx = _create_cognition_context()
        step = TrainingStep(
            game_id="test", recording_guid="r1", step_index=0,
            timestamp="", frame=None, available_actions=["ACTION1"],
            action_id=1, action_name="ACTION1", action_data={},
            levels_completed=0, state="playing", win_levels=5,
            full_reset=False, frame_signature="sig1",
        )
        metrics = _update_cognition(ctx, step, None, [step], 0, 0)
        self.assertIn("cognition_strategy", metrics)
        self.assertIn("cognition_confidence", metrics)
        self.assertIn("wm_load", metrics)
        self.assertIn("goal_depth", metrics)
        self.assertIn("goal_progress", metrics)
        self.assertIn("mean_salience", metrics)
        self.assertIn("max_salience", metrics)
        self.assertIn("salient_cell_count", metrics)

    def test_detect_bg_color_empty(self):
        self.assertEqual(_detect_bg_color([]), 9)

    def test_cognition_context_game_id(self):
        ctx = _create_cognition_context()
        ctx.game_id = "test_game"
        self.assertEqual(ctx.game_id, "test_game")
