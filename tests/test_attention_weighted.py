from __future__ import annotations

import unittest

from agi_core.cognition.attention import SalienceMap
from agi_core.orch_or.game_model import Component
from agi_core.orch_or.analyzer import _apply_salience_weights


class TestApplySalienceWeights(unittest.TestCase):
    def setUp(self):
        self._make_salience = lambda data: SalienceMap(64, 64)

    def _high_salience_map(self):
        sm = SalienceMap(64, 64)
        for y in range(64):
            for x in range(64):
                sm.data[y][x] = 0.8  # uniformly high
        return sm

    def _low_salience_map(self):
        sm = SalienceMap(64, 64)
        for y in range(64):
            for x in range(64):
                sm.data[y][x] = 0.01  # uniformly low
        return sm

    def _make_comp(self, x, y, color=5, size=9):
        cells = [(x + dx, y + dy) for dx in range(3) for dy in range(3)]
        return Component(
            cells=cells,
            bbox=(x, y, x + 2, y + 2),
            size=size,
            dominant_color=color,
            touches_border=(x == 0 or y == 0),
            confidence=1.0,
        )

    def test_none_attention_does_nothing(self):
        comp = self._make_comp(10, 10)
        _apply_salience_weights([comp], None)
        self.assertEqual(comp.confidence, 1.0)

    def test_empty_components_does_nothing(self):
        _apply_salience_weights([], object())

    def test_high_salience_no_change(self):
        comp = self._make_comp(10, 10)
        sm = self._high_salience_map()
        _apply_salience_weights([comp], sm)
        self.assertEqual(comp.confidence, 1.0)

    def test_low_salience_reduces_confidence(self):
        comp = self._make_comp(10, 10)
        sm = self._low_salience_map()
        _apply_salience_weights([comp], sm)
        self.assertLess(comp.confidence, 0.5)

    def test_salience_below_005_gives_02x(self):
        comp = self._make_comp(10, 10)
        sm = SalienceMap(64, 64)
        sm.data[10][10] = 0.04  # center at (11, 11)
        # Component bbox (10,10,12,12), center (11,11)
        # sm.data[11][11] is still 0.0 (not set)
        # Actually, wait — the component is at (x,y) to (x+2, y+2)
        # Center is (11, 11). sm.data[11][11] = 0.0 (default)
        _apply_salience_weights([comp], sm)
        # salience at center is 0.0 < 0.05 → 0.2x multiplier
        self.assertAlmostEqual(comp.confidence, 0.2, delta=0.01)

    def test_salience_between_005_and_015_gives_05x(self):
        comp = self._make_comp(10, 10)
        sm = SalienceMap(64, 64)
        sm.data[11][11] = 0.10  # center at (11,11)
        _apply_salience_weights([comp], sm)
        self.assertAlmostEqual(comp.confidence, 0.5, delta=0.01)

    def test_salience_between_015_and_030_gives_08x(self):
        comp = self._make_comp(10, 10)
        sm = SalienceMap(64, 64)
        sm.data[11][11] = 0.20  # center at (11,11)
        _apply_salience_weights([comp], sm)
        self.assertAlmostEqual(comp.confidence, 0.8, delta=0.01)

    def test_multiple_components_independent(self):
        sm = self._high_salience_map()
        c1 = self._make_comp(10, 10)
        c2 = self._make_comp(30, 30)
        sm.data[11][11] = 0.01  # c1 center = low
        _apply_salience_weights([c1, c2], sm)
        self.assertLess(c1.confidence, 0.5)  # c1 affected
        self.assertEqual(c2.confidence, 1.0)  # c2 unchanged

    def test_attention_focus_object(self):
        """Verify the function works with an AttentionFocus object."""
        from agi_core.cognition.attention import AttentionFocus
        attn = AttentionFocus(64, 64)
        comp = self._make_comp(10, 10)
        _apply_salience_weights([comp], attn)
        # Default salience is 0.0 → should reduce confidence
        self.assertLess(comp.confidence, 1.0)


class TestAttentionWeightedIntegration(unittest.TestCase):
    """Verify salience weighting flows through build_game_model."""

    def test_build_game_model_with_attention(self):
        from agi_core.cognition.attention import AttentionFocus
        from agi_core.orch_or.analyzer import build_game_model
        board = [[-1] * 64 for _ in range(64)]
        for y in range(5):
            for x in range(5):
                board[y][x] = 9
        attn = AttentionFocus(64, 64)
        gm = build_game_model(board, attention=attn)
        self.assertIsNotNone(gm)
        # Components should exist regardless of salience weighting
        self.assertGreater(len(gm.components), 0)
        for c in gm.components:
            self.assertGreaterEqual(c.confidence, 0.0)
            self.assertLessEqual(c.confidence, 1.0)
