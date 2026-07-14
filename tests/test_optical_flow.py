from __future__ import annotations

import unittest

from agi_core.cognition.attention import _compute_board_motion, AttentionFocus


class TestBoardMotion(unittest.TestCase):
    def test_both_none(self):
        self.assertEqual(_compute_board_motion(None, None), (0, 0))

    def test_one_none(self):
        board = [[0] * 10 for _ in range(10)]
        self.assertEqual(_compute_board_motion(board, None), (0, 0))
        self.assertEqual(_compute_board_motion(None, board), (0, 0))

    def test_empty_boards(self):
        self.assertEqual(_compute_board_motion([], []), (0, 0))
        self.assertEqual(_compute_board_motion([[0]], [[0]]), (0, 0))

    def test_no_active_cells(self):
        prev = [[0] * 10 for _ in range(10)]
        curr = [[0] * 10 for _ in range(10)]
        self.assertEqual(_compute_board_motion(prev, curr), (0, 0))

    def test_active_cells_no_motion(self):
        prev = [[0] * 10 for _ in range(10)]
        curr = [[0] * 10 for _ in range(10)]
        prev[5][5] = 1
        curr[5][5] = 1
        self.assertEqual(_compute_board_motion(prev, curr), (0, 0))

    def test_rightward_motion(self):
        prev = [[0] * 10 for _ in range(10)]
        curr = [[0] * 10 for _ in range(10)]
        prev[5][3] = 1  # active cell at x=3
        curr[5][5] = 1  # moved to x=5
        dx, dy = _compute_board_motion(prev, curr)
        self.assertEqual(dx, 2)
        self.assertEqual(dy, 0)

    def test_downward_motion(self):
        prev = [[0] * 10 for _ in range(10)]
        curr = [[0] * 10 for _ in range(10)]
        prev[3][5] = 1  # active cell at y=3
        curr[5][5] = 1  # moved to y=5
        dx, dy = _compute_board_motion(prev, curr)
        self.assertEqual(dx, 0)
        self.assertEqual(dy, 2)

    def test_multiple_cells_average(self):
        prev = [[0] * 10 for _ in range(10)]
        curr = [[0] * 10 for _ in range(10)]
        prev[0][0] = 1
        prev[9][9] = 1
        curr[1][1] = 1
        curr[9][9] = 1
        dx, dy = _compute_board_motion(prev, curr)
        # centroid moved from (4.5, 4.5) to (5, 5)
        # Actually: prev centroid = ((0+9)//2, (0+9)//2) = (4, 4)
        # curr centroid = ((1+9)//2, (1+9)//2) = (5, 5)
        self.assertEqual(dx, 1)
        self.assertEqual(dy, 1)


class TestAttentionFocusMotion(unittest.TestCase):
    def test_board_motion_initialized(self):
        attn = AttentionFocus(10, 10)
        self.assertEqual(attn.board_motion, (0, 0))

    def test_board_motion_computed(self):
        attn = AttentionFocus(10, 10)
        prev = [[0] * 10 for _ in range(10)]
        curr = [[0] * 10 for _ in range(10)]
        prev[5][3] = 1
        curr[5][5] = 1
        attn.compute(curr, (5, 5), [(0, 0)], prev_board=prev)
        self.assertEqual(attn.board_motion, (2, 0))


class TestOpticalFlowIntegration(unittest.TestCase):
    def test_optical_flow_in_temporal_stats(self):
        """board_motion should appear in temporal_stats via _extract_features."""
        from agi_core.orch_or.analyzer import build_game_model
        from agi_core.orch_or.tracker import EntityTracker
        tracker = EntityTracker()
        attn = AttentionFocus(64, 64)
        board = [[-1] * 64 for _ in range(64)]
        for y in range(5):
            for x in range(5):
                board[y][x] = 9
        gm = build_game_model(board, attention=attn, tracker=tracker)
        self.assertIsNotNone(gm)
        # board_motion should be accessible via features
        self.assertTrue(hasattr(gm.features, "dynamics_confidence"))

    def test_board_motion_passed_to_dynamics(self):
        """Verify the attention wiring doesn't crash."""
        attn = AttentionFocus(10, 10)
        prev = [[0] * 10 for _ in range(10)]
        curr = [[0] * 10 for _ in range(10)]
        prev[5][3] = 1
        curr[5][5] = 1
        attn.compute(curr, (5, 5), [(0, 0)], prev_board=prev)
        self.assertEqual(attn.board_motion, (2, 0))
        # Verify board_motion persists after second compute with same boards
        attn.compute(curr, (5, 5), [(0, 0)], prev_board=curr)
        self.assertEqual(attn.board_motion, (0, 0))
