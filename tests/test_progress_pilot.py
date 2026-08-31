"""Regression checks for incremental progress evidence in the pilot."""

import unittest
from types import SimpleNamespace

import numpy as np

from arc3x.pilot import Pilot


def frame_entry(action, grid, level=1):
    return SimpleNamespace(
        action=action,
        frame=SimpleNamespace(
            grid=tuple(tuple(int(v) for v in row) for row in grid),
            level=level,
        ),
    )


class PilotProgressTests(unittest.TestCase):
    def test_consumed_colour_becomes_target_without_level_win(self):
        history = []
        for step in range(36):
            grid = np.zeros((20, 20), dtype=np.int16)
            remaining = 4 if step < 20 else 3 if step < 25 else 2 if step < 30 else 1
            for y, x in [(1, 1), (1, 2), (2, 1), (2, 2)][:remaining]:
                grid[y, x] = 2
            # Add ordinary moving scenery so CellSense publishes a playfield
            # larger than the four-pixel fixture object; otherwise Progress
            # correctly classifies the tiny synthetic board as a flood.
            for x in range(10):
                grid[10, x] = 1 if (step + x) % 2 else 0
            history.append(frame_entry("" if step == 0 else "UP", grid))

        pilot = Pilot()
        pilot.observe(history)

        self.assertIn(2, pilot.progress.consumed)

    def test_level_cut_does_not_compare_counts_across_boards(self):
        first = np.zeros((8, 8), dtype=np.int16)
        first[:4, :4] = 2
        second = np.zeros((8, 8), dtype=np.int16)
        second[:1, :4] = 2
        third = np.zeros((8, 8), dtype=np.int16)
        third[:4, :4] = 2
        history = [
            frame_entry("", first, 1),
            frame_entry("UP", second, 1),
            frame_entry("UP", third, 2),
        ]

        pilot = Pilot()
        pilot.observe(history)

        self.assertEqual(pilot.progress.fell[2], 0)


if __name__ == "__main__":
    unittest.main()
