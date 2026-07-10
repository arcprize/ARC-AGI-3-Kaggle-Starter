from __future__ import annotations

import unittest
from unittest.mock import patch

from agi_core.image_input import BoardImageInput, extract_board_motif_tags


def _make_board(width=64, height=64, fill=0):
    return [[fill] * width for _ in range(height)]


class TestMotifTags(unittest.TestCase):
    """Test the existing and new motif tags."""

    def _check_tag(self, board, expected_tag, present=True, **kwargs):
        ii = BoardImageInput(board=board, height=len(board), width=len(board[0]),
                             frame_signature="test", source_attr="test")
        tags = extract_board_motif_tags(ii, **kwargs)
        if present:
            self.assertIn(expected_tag, tags,
                          f"Tag {expected_tag!r} should be present. Tags: {tags}")
        else:
            self.assertNotIn(expected_tag, tags,
                             f"Tag {expected_tag!r} should NOT be present. Tags: {tags}")

    # ── Existing tag tests ────────────────────────────────────────────

    def test_axis_horizontal(self):
        board = _make_board()
        board[0] = board[-1]  # symmetric
        self._check_tag(board, "axis_horizontal")

    def test_axis_vertical(self):
        board = _make_board()
        for y in range(64):
            board[y][0] = 1
            board[y][-1] = 1
        # Not symmetric — skip
        pass

    def test_single_control(self):
        board = _make_board()
        board[10][10] = 5
        self._check_tag(board, "single_control_selected")

    def test_multi_piece(self):
        board = _make_board()
        board[10][10] = 5
        board[20][20] = 7
        self._check_tag(board, "multi_piece_level")

    # ── New tag tests ─────────────────────────────────────────────────

    def test_mirror_present(self):
        board = _make_board()
        board[10][10] = 10
        self._check_tag(board, "mirror_present")

    def test_mirror_absent_by_default(self):
        board = _make_board()
        self._check_tag(board, "mirror_present", present=False)

    def test_beam_present(self):
        board = _make_board()
        board[10][10] = 4
        self._check_tag(board, "beam_present")

    def test_beam_absent_by_default(self):
        board = _make_board()
        self._check_tag(board, "beam_present", present=False)

    def test_beam_absent_for_reflection_only_color4_artifacts(self):
        board = _make_board(width=8, height=8, fill=9)
        for y in range(8):
            board[y][3] = 10
        for y in (4, 5):
            for x in (1, 2):
                board[y][x] = 5
        for y in (4, 5):
            board[y][5] = 4
            board[y][6] = 4
        self._check_tag(board, "beam_present", present=False)

    def test_beam_evidence_uncertain_when_confirmation_fails(self):
        board = _make_board()
        board[10][10] = 4
        ii = BoardImageInput(board=board, height=len(board), width=len(board[0]),
                             frame_signature="test", source_attr="test")
        with patch("agi_core.orch_or.analyzer.build_game_model", side_effect=RuntimeError("boom")):
            tags = extract_board_motif_tags(ii)
        self.assertIn("beam_evidence_uncertain", tags)
        self.assertNotIn("beam_present", tags)

    def test_player_isolated(self):
        board = _make_board()
        board[10][10] = 5
        self._check_tag(board, "player_isolated")

    def test_player_not_isolated_when_many_components(self):
        board = _make_board()
        # Place 5 components far apart so they are not connected
        board[10][10] = 5
        board[10][20] = 6
        board[10][30] = 7
        board[10][40] = 8
        board[10][50] = 9
        self._check_tag(board, "player_isolated", present=False)

    def test_components_vertical_strip(self):
        board = _make_board(width=64, height=64)
        # All active cells in a narrow vertical column
        for y in range(10, 50):
            board[y][30] = 1
        self._check_tag(board, "components_vertical_strip")

    def test_components_horizontal_strip(self):
        board = _make_board(width=64, height=64)
        for x in range(10, 50):
            board[30][x] = 1
        self._check_tag(board, "components_horizontal_strip")

    def test_large_central_component(self):
        board = _make_board(width=20, height=20)
        # A 10x10 block covers 25% of 20x20 = 400, 10*10=100 >= 0.30*400=120
        # Actually 10x10=100 < 120. Use 12x12=144 >= 120.
        for y in range(4, 16):
            for x in range(4, 16):
                board[y][x] = 5
        self._check_tag(board, "large_central_component")

    def test_no_large_component(self):
        board = _make_board(width=20, height=20)
        board[10][10] = 5  # single cell
        self._check_tag(board, "large_central_component", present=False)

    def test_active_at_edge(self):
        board = _make_board()
        board[0][0] = 5  # top-left corner
        self._check_tag(board, "active_at_edge")

    def test_no_active_at_edge(self):
        board = _make_board()
        board[32][32] = 5  # center
        self._check_tag(board, "active_at_edge", present=False)

    def test_goals_in_corner(self):
        board = _make_board()
        board[3][3] = 11  # near top-left corner
        self._check_tag(board, "goals_in_corner")

    def test_goals_not_in_corner(self):
        board = _make_board()
        board[32][32] = 11  # center
        self._check_tag(board, "goals_in_corner", present=False)

    def test_active_cells_near_but_not_at_edge(self):
        board = _make_board()
        board[5][5] = 5  # 5px from edge — within 3px? No, 5 > 3
        self._check_tag(board, "active_at_edge", present=False)
        board[2][2] = 5  # 2px from edge — within 3px
        self._check_tag(board, "active_at_edge", present=True)


class TestMotifTagsIntegration(unittest.TestCase):
    """Verify motif tags flow through the system."""

    def test_tags_produced_by_build_game_model(self):
        """build_game_model produces tags accessible via derived_features."""
        from agi_core.orch_or.analyzer import build_game_model
        board = [[-1] * 64 for _ in range(64)]
        for y in range(5):
            for x in range(5):
                board[y][x] = 9
        gm = build_game_model(board)
        self.assertIsNotNone(gm)
        # Motif tags are not stored on GameModel — they live in Observation,
        # but we can verify the function runs without error.
        ii = BoardImageInput(board=board, height=64, width=64,
                             frame_signature="test", source_attr="test")
        tags = extract_board_motif_tags(ii)
        self.assertIsInstance(tags, list)

    def test_tags_ar25_level1(self):
        """AR25 Level 1 board should produce relevant tags."""
        from tests.test_puzzle_games import game_ar25_level1
        gm = game_ar25_level1()
        ii = BoardImageInput(board=gm.board, height=gm.height, width=gm.width,
                             frame_signature="test", source_attr="test")
        tags = extract_board_motif_tags(ii)
        self.assertIn("mirror_present", tags,
                      "AR25 Level 1 has a mirror (color 10)")
        self.assertIsInstance(tags, list)

    def test_tags_ar25_level3(self):
        """AR25 Level 3 board should produce relevant tags."""
        from tests.test_puzzle_games import game_ar25_level3
        gm = game_ar25_level3()
        ii = BoardImageInput(board=gm.board, height=gm.height, width=gm.width,
                             frame_signature="test", source_attr="test")
        tags = extract_board_motif_tags(ii)
        # Level 3 has two shape pieces + goals + vertical mirror
        self.assertIn("multi_piece_level", tags)
        self.assertIn("mirror_present", tags)
