from __future__ import annotations

import unittest

from agi_core.orch_or.game_model import Component, EntityTrack
from agi_core.orch_or.analyzer import _detect_dynamics, _compute_temporal_stats


class TestTemporalStats(unittest.TestCase):
    def test_empty_tracks_returns_defaults(self):
        stats = _compute_temporal_stats([], None)
        self.assertFalse(stats["any_moved"])
        self.assertEqual(stats["moving_count"], 0)
        self.assertEqual(stats["player_velocity"], (0, 0))
        self.assertEqual(stats["player_stationary_streak"], 0)

    def test_no_player_returns_defaults(self):
        track = EntityTrack(entity_id=1, color=5)
        track.moved_this_frame = True
        track.velocity = (1, 0)
        stats = _compute_temporal_stats([track], None)
        self.assertTrue(stats["any_moved"])
        self.assertEqual(stats["moving_count"], 1)
        # No player → no player velocity
        self.assertEqual(stats["player_velocity"], (0, 0))

    def test_player_track_matched(self):
        track = EntityTrack(entity_id=1, color=5)
        track.positions = [(10, 20)]
        track.velocity = (2, 3)
        track.stationary_streak = 5
        track.moved_this_frame = True
        stats = _compute_temporal_stats([track], (10, 20))
        self.assertEqual(stats["player_velocity"], (2, 3))
        self.assertEqual(stats["player_stationary_streak"], 5)
        self.assertTrue(stats["downward_motion"])

    def test_horizontal_motion_detected(self):
        track = EntityTrack(entity_id=1, color=5)
        track.positions = [(10, 20)]
        track.velocity = (3, 0)
        track.moved_this_frame = True
        stats = _compute_temporal_stats([track], (10, 20))
        self.assertTrue(stats["horizontal_motion"])
        self.assertFalse(stats["downward_motion"])

    def test_stationary_no_motion(self):
        track = EntityTrack(entity_id=1, color=5)
        track.positions = [(10, 20)]
        track.velocity = (0, 0)
        track.moved_this_frame = False
        stats = _compute_temporal_stats([track], (10, 20))
        self.assertFalse(stats["any_moved"])
        self.assertEqual(stats["moving_count"], 0)


class TestTemporalDynamicsBoosters(unittest.TestCase):
    """Verify _detect_dynamics uses temporal_stats to add/remove dynamics."""

    def _make_board(self, height=20, width=20, fill=0):
        return [[fill] * width for _ in range(height)]

    def _make_comp(self, x, y, color, size=1):
        return Component(
            cells=[(x + dx, y) for dx in range(size)],
            bbox=(x, y, x + size - 1, y),
            size=size, dominant_color=color,
            touches_border=(x == 0 or y == 0),
        )

    def test_gravity_temporal_booster(self):
        """Downward motion adds gravity even without bottom-third position."""
        board = self._make_board()
        comps = [self._make_comp(5, 5, 5)]
        tstats = {"downward_motion": True}
        result, primary = _detect_dynamics(
            board, comps, (5, 5), [(10, 10)], bg_color=0,
            has_beam=False, has_click_interaction=False,
            movement_axes={"x", "y"}, board_width=20, board_height=20,
            temporal_stats=tstats,
        )
        self.assertIn("gravity", result,
                      "downward_motion should add gravity")

    def test_no_gravity_without_temporal(self):
        """Without temporal stats, gravity requires bottom-third position."""
        board = self._make_board()
        comps = [self._make_comp(5, 5, 5)]
        result, primary = _detect_dynamics(
            board, comps, (5, 5), [(10, 10)], bg_color=0,
            has_beam=False, has_click_interaction=False,
            movement_axes={"x", "y"}, board_width=20, board_height=20,
        )
        self.assertNotIn("gravity", result,
                         "player at y=5 should not trigger gravity")

    def test_sliding_temporal_booster(self):
        """Horizontal-only motion adds sliding even without long corridors."""
        board = self._make_board(height=10, width=10)
        comps = [self._make_comp(3, 3, 5)]
        tstats = {"horizontal_motion": True}
        result, primary = _detect_dynamics(
            board, comps, (3, 3), [(8, 3)], bg_color=0,
            has_beam=False, has_click_interaction=False,
            movement_axes={"x", "y"}, board_width=10, board_height=10,
            temporal_stats=tstats,
        )
        self.assertIn("sliding", result,
                      "horizontal_motion should add sliding")

    def test_multi_agent_filtered_by_movement(self):
        """Two candidate entities but only one moved → no multi-agent."""
        board = self._make_board()
        # Two non-player components with size >= 4
        comps = [
            self._make_comp(5, 5, 5, size=4),    # player at (5,5) — excluded from candidates
            self._make_comp(10, 10, 6, size=4),  # non-player entity A
            self._make_comp(15, 15, 7, size=4),  # non-player entity B
        ]
        tstats = {"moving_count": 1}
        result, primary = _detect_dynamics(
            board, comps, (5, 5), [(3, 3)], bg_color=0,
            has_beam=False, has_click_interaction=False,
            movement_axes={"x", "y"}, board_width=20, board_height=20,
            temporal_stats=tstats,
        )
        self.assertNotIn("multi-agent", result,
                         "only 1 moving entity should not be multi-agent")

    def test_multi_agent_strengthened_by_movement(self):
        """Two moving entities → multi-agent detected."""
        board = self._make_board()
        comps = [
            self._make_comp(5, 5, 5, size=4),
            self._make_comp(10, 10, 6, size=4),
            self._make_comp(15, 15, 7, size=4),
        ]
        tstats = {"moving_count": 2}
        result, primary = _detect_dynamics(
            board, comps, (5, 5), [(3, 3)], bg_color=0,
            has_beam=False, has_click_interaction=False,
            movement_axes={"x", "y"}, board_width=20, board_height=20,
            temporal_stats=tstats,
        )
        self.assertIn("multi-agent", result,
                      "2 moving entities should detect multi-agent")

    def test_puzzle_stationary_booster(self):
        """Stationary player near horizontal mirror adds puzzle via temporal stats."""
        board = self._make_board(height=30, width=30)
        # Create a horizontal mirror strip (wider than tall) — color 10
        mirror_comp = Component(
            cells=[(x, 15) for x in range(5, 20)],
            bbox=(5, 15, 19, 15),
            size=15, dominant_color=10,
            touches_border=False,
        )
        player_comp = self._make_comp(12, 5, 0, size=5)
        comps = [player_comp, mirror_comp]
        tstats = {"player_stationary_streak": 15}
        result, primary = _detect_dynamics(
            board, comps, (12, 5), [(12, 3)], bg_color=0,
            has_beam=False, has_click_interaction=False,
            movement_axes={"x", "y"}, board_width=30, board_height=30,
            temporal_stats=tstats,
        )
        # Horizontal mirror with goals + stationary player → puzzle
        self.assertIn("puzzle", result,
                      "horizontal mirror + stationary player should trigger puzzle")


class TestTemporalIntegration(unittest.TestCase):
    """Verify temporal_stats flow through build_game_model."""

    def test_build_game_model_with_tracker(self):
        """build_game_model with tracker produces temporal_stats internally."""
        from agi_core.orch_or.analyzer import build_game_model
        from agi_core.orch_or.tracker import EntityTracker

        tracker = EntityTracker()
        board = [[-1] * 64 for _ in range(64)]
        for y in range(5):
            for x in range(5):
                board[y][x] = 9

        gm = build_game_model(board, tracker=tracker)
        self.assertIsNotNone(gm)
        # Dynamics should be computed without error
        self.assertIn(gm.features.primary_dynamics,
                      ["navigation", "static", "puzzle", "gravity",
                       "sliding", "multi-agent", "hybrid", "unknown"])

    def test_dynamics_detection_does_not_crash_with_temporal(self):
        """Ensure _detect_dynamics handles temporal_stats gracefully."""
        board = [[0] * 10 for _ in range(10)]
        comps = [
            Component(cells=[(5, 5)], bbox=(5, 5, 5, 5), size=1,
                      dominant_color=5, touches_border=False),
        ]
        result, primary = _detect_dynamics(
            board, comps, (5, 5), [(8, 8)], bg_color=0,
            has_beam=False, has_click_interaction=False,
            movement_axes={"x", "y"}, board_width=10, board_height=10,
            temporal_stats={},
        )
        self.assertIsInstance(result, list)
        self.assertIsInstance(primary, str)
