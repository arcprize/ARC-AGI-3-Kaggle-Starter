from __future__ import annotations

import unittest
from unittest.mock import patch

from agi_core.image_input import compute_board_signature
from agi_core.orch_or.game_model import EntityTrack
from agi_core.orch_or.tracker import EntityTracker


class TestComputeBoardSignature(unittest.TestCase):
    def test_signature_is_string(self):
        board = [[0] * 10 for _ in range(10)]
        sig = compute_board_signature(board)
        self.assertIsInstance(sig, str)
        self.assertGreater(len(sig), 0)

    def test_identical_boards_same_signature(self):
        board1 = [[5] * 64 for _ in range(64)]
        board2 = [[5] * 64 for _ in range(64)]
        self.assertEqual(compute_board_signature(board1), compute_board_signature(board2))

    def test_different_boards_different_signature(self):
        board1 = [[5] * 64 for _ in range(64)]
        board2 = [[9] * 64 for _ in range(64)]
        self.assertNotEqual(compute_board_signature(board1), compute_board_signature(board2))

    def test_none_board(self):
        self.assertIsNone(compute_board_signature(None))

    def test_empty_board(self):
        self.assertIsNone(compute_board_signature([]))


class TestPerceptionCacheIntegration(unittest.TestCase):
    """Verify perception caching works end-to-end."""

    def test_cached_model_identical_components(self):
        """Calling build_game_model twice with the same board produces the
        same components when cached."""
        from agi_core.orch_or.analyzer import build_game_model
        board = [[-1] * 64 for _ in range(64)]
        for y in range(5):
            for x in range(5):
                board[y][x] = 9
        gm1 = build_game_model(board)
        gm2 = build_game_model(board)
        self.assertEqual(len(gm1.components), len(gm2.components))
        for c1, c2 in zip(gm1.components, gm2.components):
            self.assertEqual(c1.bbox, c2.bbox)
            self.assertEqual(c1.dominant_color, c2.dominant_color)

    def test_cached_model_not_stale_across_frames(self):
        """Entity tracker still ages tracks even when board is cached."""
        from agi_core.orch_or.analyzer import build_game_model
        tracker = EntityTracker()
        board = [[-1] * 64 for _ in range(64)]
        for y in range(5):
            for x in range(5):
                board[y][x] = 9
        gm1 = build_game_model(board, tracker=tracker)
        # Second frame — same board, tracker should still update
        gm2 = build_game_model(board, tracker=tracker)
        self.assertEqual(len(gm2.entity_tracks), len(gm1.entity_tracks))
        # Tracks should have aged
        for t in gm2.entity_tracks:
            self.assertGreater(t.age, 0)

    def test_board_signature_changes_on_frame_change(self):
        """Verify compute_board_signature detects actual changes."""
        board1 = [[-1] * 64 for _ in range(64)]
        board2 = [[-1] * 64 for _ in range(64)]
        board2[32][32] = 5  # one cell changed
        sig1 = compute_board_signature(board1)
        sig2 = compute_board_signature(board2)
        self.assertNotEqual(sig1, sig2)

    def test_empty_board_signature_is_none(self):
        """Empty boards should not crash signature computation."""
        sig = compute_board_signature([])
        self.assertIsNone(sig)
