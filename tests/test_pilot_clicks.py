"""Regression checks for click semantics inside the online pilot."""

from collections import Counter
from types import SimpleNamespace
import unittest

import numpy as np

from arc3x.pilot import Pilot


def entry(action, grid, level=1):
    return SimpleNamespace(
        action=action,
        frame=SimpleNamespace(
            grid=tuple(tuple(int(v) for v in row) for row in grid),
            level=level,
        ),
    )


class PilotClickTests(unittest.TestCase):
    def test_hud_only_clicks_are_inert_not_live_cells(self):
        """A counter changing on every click must not sustain blind probing."""
        history = [entry("", np.zeros((8, 8), dtype=np.int16))]
        for i in range(12):
            grid = np.zeros((8, 8), dtype=np.int16)
            grid[0, 0] = i + 1  # a ticking HUD, unrelated to the click position
            history.append(entry(f"MOUSE(row={i % 8}, col={(i * 3) % 8})", grid))

        pilot = Pilot()
        pilot.observe(history)

        self.assertEqual(pilot.click_model.verdict()[0], "inert")
        self.assertEqual(pilot.live_clicks, set())
        self.assertIsNone(pilot._click(history[-1].frame and np.asarray(history[-1].frame.grid), [6], ["MOUSE"]))

    def test_coordinate_free_click_transfers_cautiously_to_next_level(self):
        """A settled step rule gets one confirming click on a fresh level."""
        pilot = Pilot()
        model = pilot.click_model
        model.n = 4
        model.support = Counter({"widget": 4})
        model.follow = [(0, 0, 4.0, 4.0), (1, 1, 4.0, 4.0), (2, 2, 4.0, 4.0), (3, 3, 4.0, 4.0)]
        pilot.level = 2
        frame = np.zeros((8, 10), dtype=np.int16)

        plan = pilot._click(frame, [6], ["MOUSE"])

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.phase, "click-step")
        self.assertEqual(len(plan.presses), 1)
        self.assertEqual((plan.presses[0].row, plan.presses[0].col), (4, 5))

    def test_learned_responsive_colour_precedes_generic_rarity(self):
        """A colour with real click evidence is tried before a novel pixel."""
        pilot = Pilot()
        pilot.click_model.live = Counter({7: 4})
        frame = np.zeros((8, 8), dtype=np.int16)
        frame[1:3, 1:3] = 7
        frame[6, 6] = 2

        plan = pilot._click(frame, [6], ["MOUSE"])

        self.assertIsNotNone(plan)
        assert plan is not None
        first = plan.presses[0]
        self.assertEqual(int(frame[first.row, first.col]), 7)

    def test_sidecar_only_replays_a_high_confidence_coordinate_free_click(self):
        """The baseline-protecting mode never starts ordinary click exploration."""
        pilot = Pilot()
        frame = np.zeros((8, 10), dtype=np.int16)

        self.assertIsNone(pilot.assist(frame, ["MOUSE"], 1, max_actions=4))

        pilot.click_model.n = 4
        pilot.click_model.support = Counter({"widget": 4})
        pilot.click_model.follow = [
            (0, 0, 4.0, 4.0),
            (1, 1, 4.0, 4.0),
            (2, 2, 4.0, 4.0),
            (3, 3, 4.0, 4.0),
        ]
        plan = pilot.assist(frame, ["MOUSE"], 1, max_actions=4)

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.phase, "sidecar-step")
        self.assertEqual(len(plan.presses), 1)
        self.assertEqual(pilot.sidecar_actions, 1)


if __name__ == "__main__":
    unittest.main()
