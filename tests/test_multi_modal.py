from __future__ import annotations

import unittest

from agi_core.cognition.multi_modal import ActionEffectLog, ActionEffect, InferredMechanics
from agi_core.orch_or.game_model import EntityTrack


class TestActionEffectLog(unittest.TestCase):
    def setUp(self):
        self.log = ActionEffectLog(max_len=8)

    def test_empty_log_returns_default_mechanics(self):
        m = self.log.infer()
        self.assertAlmostEqual(m.gravity_evidence, 0.0)
        self.assertAlmostEqual(m.sliding_evidence, 0.0)
        self.assertAlmostEqual(m.entity_switch_evidence, 0.0)
        self.assertFalse(m.has_entity_switch)
        self.assertEqual(m.primary_flow, "")

    def test_single_effect_returns_default(self):
        self.log.record("ACTION1", (0, 0), (0, 1), board_changed=True)
        m = self.log.infer()
        self.assertEqual(m.gravity_evidence, 0.0)

    def test_entity_switch_detected(self):
        self.log.record("ACTION5", (5, 5), (5, 5),
                        player_color_before=5, player_color_after=0,
                        board_changed=True)
        m = self.log.infer()
        self.assertTrue(m.has_entity_switch)
        self.assertGreater(m.entity_switch_evidence, 0.5)

    def test_no_entity_switch_without_color_change(self):
        self.log.record("ACTION5", (5, 5), (6, 5),
                        player_color_before=5, player_color_after=5,
                        board_changed=True)
        m = self.log.infer()
        self.assertFalse(m.has_entity_switch)
        self.assertEqual(m.entity_switch_evidence, 0.0)

    def test_gravity_detected_from_entity_deltas(self):
        def _make_track(eid, pos):
            t = EntityTrack(entity_id=eid, color=5)
            t.positions = [pos]
            return t
        # Simulate entity moving down across frames
        for step in range(3):
            before = _make_track(1, (5, step))
            after = _make_track(1, (5, step + 1))
            self.log.record("ACTION1", (5, 5), (5, 5),
                            entity_tracks_before=[before],
                            entity_tracks_after=[after],
                            board_changed=True)
        m = self.log.infer()
        self.assertGreater(m.gravity_evidence, 0.3)

    def test_sliding_detected_from_entity_deltas(self):
        def _make_track(eid, pos):
            t = EntityTrack(entity_id=eid, color=5)
            t.positions = [pos]
            return t
        for step in range(3):
            before = _make_track(1, (step, 5))
            after = _make_track(1, (step + 1, 5))
            self.log.record("ACTION1", (5, 5), (5, 5),
                            entity_tracks_before=[before],
                            entity_tracks_after=[after],
                            board_changed=True)
        m = self.log.infer()
        self.assertGreater(m.sliding_evidence, 0.3)

    def test_primary_flow_detected(self):
        track = EntityTrack(entity_id=1, color=5)
        track.positions = [(0, 0)]
        after_track = EntityTrack(entity_id=1, color=5)
        after_track.positions = [(3, 0)]  # moved right by 3
        for _ in range(3):
            self.log.record("ACTION4", (0, 0), (3, 0),
                            entity_tracks_before=[track],
                            entity_tracks_after=[after_track],
                            board_changed=True)
        m = self.log.infer()
        self.assertEqual(m.primary_flow, "right")

    def test_reset_clears(self):
        self.log.record("ACTION5", (5, 5), (5, 5),
                        player_color_before=5, player_color_after=0,
                        board_changed=True)
        self.log.reset()
        m = self.log.infer()
        self.assertFalse(m.has_entity_switch)

    def test_max_len_enforced(self):
        for i in range(20):
            self.log.record(f"ACTION{i % 6 + 1}", None, None)
        self.assertLessEqual(len(self.log._history), 8)


class TestInferredMechanics(unittest.TestCase):
    def test_as_dict(self):
        m = InferredMechanics(
            gravity_evidence=0.8,
            sliding_evidence=0.3,
            entity_switch_evidence=0.0,
            has_entity_switch=True,
            primary_flow="down",
        )
        d = m.as_dict()
        self.assertEqual(d["gravity_evidence"], 0.8)
        self.assertEqual(d["sliding_evidence"], 0.3)
        self.assertEqual(d["has_entity_switch"], True)
        self.assertEqual(d["primary_flow"], "down")

    def test_default_as_dict(self):
        m = InferredMechanics()
        d = m.as_dict()
        self.assertEqual(d["gravity_evidence"], 0.0)
        self.assertEqual(d["entity_switch_evidence"], 0.0)
        self.assertFalse(d["has_entity_switch"])


class TestMultiModalIntegration(unittest.TestCase):
    """Verify multi-modal perception integrates with the full pipeline."""

    def test_log_creatable_in_orchestrator(self):
        """Verify the import works from orchestrator's context."""
        from agi_core.cognition.multi_modal import ActionEffectLog, InferredMechanics
        log = ActionEffectLog()
        self.assertIsNotNone(log)
        m = log.infer()
        self.assertIsInstance(m, InferredMechanics)

    def test_inferred_mechanics_in_plan_context(self):
        """inferred_mechanics dict is available when building a game model
        via the orchestrator path."""
        from agi_core.orch_or.analyzer import build_game_model
        board = [[-1] * 64 for _ in range(64)]
        for y in range(5):
            for x in range(5):
                board[y][x] = 9
        gm = build_game_model(board, inferred_mechanics={})
        self.assertIsNotNone(gm)
        self.assertIsNotNone(gm.features)
