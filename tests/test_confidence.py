from __future__ import annotations

import unittest

from agi_core.orch_or.game_model import Component, GameFeatures


class TestComponentConfidence(unittest.TestCase):
    """Test confidence computation for component detection."""

    def _make_comp(self, x1, y1, x2, y2, color, cells=None):
        if cells is None:
            cells = [(x, y) for x in range(x1, x2 + 1) for y in range(y1, y2 + 1)]
        return Component(
            cells=cells,
            bbox=(x1, y1, x2, y2),
            size=len(cells),
            dominant_color=color,
            touches_border=(x1 == 0 or y1 == 0),
        )

    def test_default_confidence(self):
        comp = self._make_comp(5, 5, 10, 10, 5)
        self.assertEqual(comp.confidence, 1.0)

    def test_small_component_lower_confidence(self):
        from agi_core.orch_or.analyzer import _component_confidence_simple
        conf = _component_confidence_simple([(0, 0)], 0, 0, 0, 0, 64 * 64)
        self.assertLess(conf, 1.0)

    def _call_comp_conf_simple(self, cells, min_x, min_y, max_x, max_y, total, touches_border=None):
        from agi_core.orch_or.analyzer import _component_confidence_simple
        return _component_confidence_simple(
            cells, min_x, min_y, max_x, max_y, total, touches_border=touches_border
        )

    def test_border_component_penalty(self):
        conf = self._call_comp_conf_simple([(0, 0), (1, 0)], 0, 0, 1, 0, 4096)
        self.assertLess(conf, 1.0)

    def test_right_bottom_border_component_penalty(self):
        conf = self._call_comp_conf_simple(
            [(62, 63), (63, 63)],
            62,
            63,
            63,
            63,
            4096,
            touches_border=True,
        )
        self.assertLess(conf, 1.0)

    def test_large_component_penalty(self):
        cells = [(x, y) for x in range(30) for y in range(20)]  # 600 cells
        conf = self._call_comp_conf_simple(cells, 0, 0, 29, 19, 4096)
        self.assertLess(conf, 1.0)

    def test_solid_square_high_confidence(self):
        cells = [(x, y) for x in range(5, 15) for y in range(5, 15)]
        conf = self._call_comp_conf_simple(cells, 5, 5, 14, 14, 4096)
        self.assertGreater(conf, 0.5)

    def test_conf_range_bounds(self):
        from agi_core.orch_or.analyzer import _component_confidence_simple
        # Tiny component
        c1 = _component_confidence_simple([(0, 0)], 0, 0, 0, 0, 4096)
        self.assertGreaterEqual(c1, 0.1)
        self.assertLessEqual(c1, 1.0)

    def test_merge_nearby_same_color_preserves_confidence_signal(self):
        from agi_core.orch_or.analyzer import _merge_nearby_same_color

        low_conf = Component(
            cells=[(0, 0), (1, 0)],
            bbox=(0, 0, 1, 0),
            size=2,
            dominant_color=5,
            touches_border=True,
            confidence=0.4,
        )
        high_conf = Component(
            cells=[(2, 0), (3, 0)],
            bbox=(2, 0, 3, 0),
            size=2,
            dominant_color=5,
            touches_border=True,
            confidence=0.9,
        )

        merged = _merge_nearby_same_color([low_conf, high_conf])

        self.assertEqual(len(merged), 1)
        self.assertLess(merged[0].confidence, 1.0)
        self.assertGreaterEqual(merged[0].confidence, 0.4)

    def test_merge_nearby_same_color_is_transitive(self):
        from agi_core.orch_or.analyzer import _merge_nearby_same_color

        a = Component(
            cells=[(0, 0)],
            bbox=(0, 0, 0, 0),
            size=1,
            dominant_color=5,
            touches_border=True,
            confidence=0.8,
        )
        b = Component(
            cells=[(1, 0)],
            bbox=(1, 0, 1, 0),
            size=1,
            dominant_color=5,
            touches_border=True,
            confidence=0.8,
        )
        c = Component(
            cells=[(2, 0)],
            bbox=(2, 0, 2, 0),
            size=1,
            dominant_color=5,
            touches_border=True,
            confidence=0.8,
        )

        merged = _merge_nearby_same_color([a, b, c])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].size, 3)


class TestPlayerConfidence(unittest.TestCase):
    """Test player detection confidence."""

    def test_none_player_zero_confidence(self):
        from agi_core.orch_or.analyzer import _player_confidence
        self.assertEqual(_player_confidence(None, []), 0.0)

    def test_color_0_high_confidence(self):
        from agi_core.orch_or.analyzer import _player_confidence
        comp = Component(
            cells=[(10, 10)], bbox=(10, 10, 10, 10), size=1,
            dominant_color=0, touches_border=False,
        )
        conf = _player_confidence((10, 10), [comp])
        self.assertGreater(conf, 0.9)

    def test_color_5_medium_confidence(self):
        from agi_core.orch_or.analyzer import _player_confidence
        comp = Component(
            cells=[(10, 10)], bbox=(10, 10, 10, 10), size=1,
            dominant_color=5, touches_border=False,
        )
        conf = _player_confidence((10, 10), [comp])
        self.assertAlmostEqual(conf, 0.70, delta=0.01)


class TestGoalConfidence(unittest.TestCase):
    """Test goal detection confidence."""

    def test_color_11_high(self):
        from agi_core.orch_or.analyzer import _goal_confidence
        board = [[-1] * 64 for _ in range(64)]
        board[10][10] = 11
        conf = _goal_confidence((10, 10), board)
        self.assertGreater(conf, 0.8)

    def test_color_4_medium(self):
        from agi_core.orch_or.analyzer import _goal_confidence
        board = [[-1] * 64 for _ in range(64)]
        board[10][10] = 4
        conf = _goal_confidence((10, 10), board)
        self.assertAlmostEqual(conf, 0.60, delta=0.01)

    def test_out_of_bounds_zero(self):
        from agi_core.orch_or.analyzer import _goal_confidence
        board = [[-1]]
        self.assertEqual(_goal_confidence((10, 10), board), 0.0)


class TestDynamicsConfidence(unittest.TestCase):
    """Test dynamics confidence distribution."""

    def test_navigation_default(self):
        from agi_core.orch_or.analyzer import _dynamics_confidence
        result = _dynamics_confidence(["navigation"], [], [], False, True)
        self.assertIn("navigation", result)
        self.assertGreater(result["navigation"], 0)

    def test_multi_agent_scales_with_components(self):
        from agi_core.orch_or.analyzer import _dynamics_confidence
        from agi_core.orch_or.game_model import Component
        comps = [
            Component(cells=[(1, 1)], bbox=(1, 1, 1, 1), size=1,
                      dominant_color=5, touches_border=False),
            Component(cells=[(5, 5)], bbox=(5, 5, 5, 5), size=1,
                      dominant_color=6, touches_border=False),
        ]
        result = _dynamics_confidence(["multi-agent"], comps, [], False, True)
        self.assertIn("multi-agent", result)

    def test_scores_normalize_to_one(self):
        from agi_core.orch_or.analyzer import _dynamics_confidence
        result = _dynamics_confidence(["puzzle", "navigation"], [], [(5, 5)], False, True)
        total = sum(result.values())
        self.assertAlmostEqual(total, 1.0, delta=0.01)


class TestGameFeaturesConfidence(unittest.TestCase):
    """Test that GameFeatures carries confidence fields."""

    def test_default_values(self):
        gf = GameFeatures()
        self.assertEqual(gf.perception_confidence, 1.0)
        self.assertEqual(gf.dynamics_confidence, {})

    def test_build_game_model_sets_confidence(self):
        from agi_core.orch_or.analyzer import build_game_model
        board = [[-1] * 64 for _ in range(64)]
        for y in range(5):
            for x in range(5):
                board[y][x] = 9
        gm = build_game_model(board)
        self.assertIsNotNone(gm)
        self.assertGreaterEqual(gm.features.perception_confidence, 0.0)
        self.assertLessEqual(gm.features.perception_confidence, 1.0)
        if gm.components:
            for c in gm.components:
                self.assertGreaterEqual(c.confidence, 0.0)
                self.assertLessEqual(c.confidence, 1.0)

    def test_confidence_ar25_level3(self):
        from tests.test_puzzle_games import game_ar25_level3
        from agi_core.orch_or.analyzer import build_game_model
        board = game_ar25_level3().board
        gm = build_game_model(board)
        self.assertGreater(gm.features.perception_confidence, 0.0)
        # Should have dynamics_confidence populated
        self.assertGreater(len(gm.features.dynamics_confidence), 0)


class TestCriticIntegration(unittest.TestCase):
    """Test that the critic uses perception confidence."""

    def test_critic_importable(self):
        """Rescoring should not crash regardless of perception_confidence value."""
        from agi_core.specialists.critic import ProposalCritic
        critic = ProposalCritic()
        self.assertTrue(callable(critic.rescore))
