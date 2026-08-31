"""Regression checks for the human-mind history boundary."""

import unittest
from types import SimpleNamespace

import numpy as np

from arc3x.mindgraft import Mind, transitions


def entry(action, grid, level=1):
    return SimpleNamespace(
        action=action,
        frame=SimpleNamespace(grid=tuple(tuple(int(v) for v in row) for row in grid), level=level),
    )


class MindHistoryTests(unittest.TestCase):
    def test_level_transition_is_not_motion_training(self):
        before = np.zeros((8, 8), dtype=np.int16)
        before[2, 2] = 9
        after = np.zeros((8, 8), dtype=np.int16)
        after[6, 6] = 9
        history = [entry("", before), entry("UP", after, level=2)]

        trs = transitions(history)
        self.assertEqual(len(trs), 1)
        self.assertTrue(trs[0].level_up)

        mind = Mind()
        self.assertEqual(mind.absorb(history), 1)
        self.assertEqual(mind.folded, 0)
        self.assertEqual(mind.mech.votes, {})

    def test_same_level_transitions_are_incremental(self):
        first = np.zeros((8, 8), dtype=np.int16)
        first[2, 2] = 9
        second = first.copy()
        second[1, 2] = 9
        history = [entry("", first), entry("UP", second)]

        mind = Mind()
        self.assertEqual(mind.absorb(history), 1)
        self.assertEqual(mind.absorb(history), 0)
        self.assertEqual(mind.folded, 1)


if __name__ == "__main__":
    unittest.main()
