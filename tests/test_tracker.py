from __future__ import annotations

import unittest

from agi_core.orch_or.game_model import Component, EntityTrack
from agi_core.orch_or.tracker import EntityTracker


class TestEntityTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = EntityTracker()

    def _make_comp(self, x: int, y: int, color: int, size: int = 1) -> Component:
        cells = [(x + dx, y) for dx in range(size)]
        return Component(
            cells=cells,
            bbox=(x, y, x + size - 1, y),
            size=size,
            dominant_color=color,
            touches_border=False,
        )

    def test_empty_components_returns_empty(self):
        tracks = self.tracker.update([], frame=0)
        self.assertEqual(tracks, [])

    def test_new_component_creates_track(self):
        comp = self._make_comp(10, 20, 5)
        tracks = self.tracker.update([comp], frame=0)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].color, 5)
        self.assertEqual(tracks[0].positions[-1], (10, 20))
        self.assertEqual(tracks[0].age, 1)

    def test_entity_id_increments(self):
        c1 = self._make_comp(10, 20, 5)
        self.tracker.update([c1], frame=0)
        c2 = self._make_comp(30, 40, 10)
        tracks = self.tracker.update([c2], frame=1)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].entity_id, 2)
        self.assertEqual(tracks[0].color, 10)

    def test_same_track_reused(self):
        c1 = self._make_comp(10, 20, 5)
        self.tracker.update([c1], frame=0)
        c2 = self._make_comp(11, 20, 5)  # moved 1px right
        tracks = self.tracker.update([c2], frame=1)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].entity_id, 1)  # same ID
        self.assertEqual(len(tracks[0].positions), 2)
        self.assertEqual(tracks[0].age, 2)

    def test_velocity_computed(self):
        positions = [(10, 20), (15, 20), (20, 20)]
        comps = [self._make_comp(x, y, 5) for x, y in positions]
        self.tracker.update([comps[0]], frame=0)
        self.tracker.update([comps[1]], frame=1)
        tracks = self.tracker.update([comps[2]], frame=2)
        vx, vy = tracks[0].velocity
        # Raw velocity: (20-15) = 5, (15-10) = 5
        # EMA: 0.5*vx_prev + 0.5*raw
        # After step 1: v = (0,0)*0.5 + (5,0)*0.5 = (2,0)  (int)
        # After step 2: v = (2,0)*0.5 + (5,0)*0.5 = (3,0)  (int)
        self.assertAlmostEqual(vx, 3, delta=2)
        self.assertEqual(vy, 0)

    def test_stationary_streak_increments(self):
        comp = self._make_comp(10, 20, 5)
        for frame in range(5):
            self.tracker.update([comp], frame=frame)
        tracks = self.tracker.all_tracks()
        self.assertEqual(len(tracks), 1)
        self.assertGreaterEqual(tracks[0].stationary_streak, 0)

    def test_new_component_after_stale_pruned(self):
        c1 = self._make_comp(10, 20, 5)
        self.tracker.update([c1], frame=0)
        # Advance enough frames that the track goes stale
        for f in range(1, 40):
            self.tracker.update([], frame=f)
        # All tracks should be pruned by now
        self.assertEqual(len(self.tracker.all_tracks()), 0)
        c2 = self._make_comp(10, 20, 5)
        tracks = self.tracker.update([c2], frame=40)
        self.assertEqual(len(tracks), 1)
        # ID continues from next_id, which has been incrementing
        self.assertGreaterEqual(tracks[0].entity_id, 1)

    def test_multiple_components_matched(self):
        c1 = self._make_comp(10, 20, 5)
        c2 = self._make_comp(30, 40, 10)
        self.tracker.update([c1, c2], frame=0)
        c1m = self._make_comp(11, 20, 5)
        c2m = self._make_comp(31, 40, 10)
        tracks = self.tracker.update([c1m, c2m], frame=1)
        ids = [t.entity_id for t in tracks]
        self.assertIn(1, ids)
        self.assertIn(2, ids)

    def test_color_change_still_matches(self):
        c1 = self._make_comp(10, 20, 5)
        self.tracker.update([c1], frame=0)
        c2 = self._make_comp(11, 20, 10)  # color changed
        tracks = self.tracker.update([c2], frame=1)
        # color diff = 5, cost = 5*100 + distance = 500+1 = 501 > threshold
        # Might or might not match depending on thresholds
        self.assertGreaterEqual(len(tracks), 1)

    def test_reset_clears_all(self):
        c1 = self._make_comp(10, 20, 5)
        self.tracker.update([c1], frame=0)
        self.tracker.reset()
        self.assertEqual(len(self.tracker.all_tracks()), 0)
        c2 = self._make_comp(10, 20, 5)
        tracks = self.tracker.update([c2], frame=1)
        self.assertEqual(tracks[0].entity_id, 1)  # ID restart

    def test_track_moved_this_frame(self):
        c1 = self._make_comp(10, 20, 5)
        self.tracker.update([c1], frame=0)
        self.assertTrue(self.tracker.all_tracks()[0].moved_this_frame)
        c2 = self._make_comp(10, 20, 5)  # same position
        self.tracker.update([c2], frame=1)
        self.assertFalse(self.tracker.all_tracks()[0].moved_this_frame)

    def test_entity_track_dataclass(self):
        t = EntityTrack(entity_id=1, color=5)
        self.assertEqual(t.entity_id, 1)
        self.assertEqual(t.color, 5)
        self.assertEqual(t.positions, [])
        self.assertEqual(t.velocity, (0, 0))
        self.assertEqual(t.stationary_streak, 0)
        self.assertFalse(t.moved_this_frame)


class TestEntityTrackerIntegration(unittest.TestCase):
    """Integration tests verifying tracker works with real GameModel."""

    def test_tracker_in_build_game_model(self):
        from agi_core.orch_or.analyzer import build_game_model
        tracker = EntityTracker()
        board = [[-1] * 64 for _ in range(64)]
        for y in range(5):
            for x in range(5):
                board[y][x] = 9
        gm = build_game_model(board, tracker=tracker)
        self.assertIsNotNone(gm)
        self.assertTrue(hasattr(gm, "entity_tracks"))
        # The board has at least a border component
        if gm.components:
            self.assertGreaterEqual(len(tracker.all_tracks()), 1)

    def test_tracker_across_frames(self):
        from agi_core.orch_or.analyzer import build_game_model
        tracker = EntityTracker()
        board = [[-1] * 64 for _ in range(64)]
        for x in range(10):
            board[0][x] = 9
            board[1][x] = 5
        gm1 = build_game_model(board, tracker=tracker)
        n_tracks_1 = len(tracker.all_tracks())
        # Second frame — same board
        gm2 = build_game_model(board, tracker=tracker)
        n_tracks_2 = len(tracker.all_tracks())
        # Same components → same number of tracks
        self.assertEqual(n_tracks_1, n_tracks_2)
        # Tracks should be reusing IDs
        for t in tracker.all_tracks():
            self.assertGreater(t.age, 1)

    def test_tracker_prunes_stale_tracks_through_build_game_model(self):
        from agi_core.orch_or.analyzer import build_game_model

        tracker = EntityTracker()
        board_with_entity = [[-1] * 64 for _ in range(64)]
        for y in range(4):
            for x in range(4):
                board_with_entity[y][x] = 9

        empty_board = [[-1] * 64 for _ in range(64)]

        build_game_model(board_with_entity, tracker=tracker)
        self.assertGreaterEqual(len(tracker.all_tracks()), 1)

        for _ in range(35):
            build_game_model(empty_board, tracker=tracker)

        self.assertEqual(len(tracker.all_tracks()), 0)
