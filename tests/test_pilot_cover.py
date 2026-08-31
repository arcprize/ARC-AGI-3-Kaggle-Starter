"""Regression checks for general level-0 map coverage."""

from collections import Counter
import unittest

import numpy as np

from arc3x.pilot import Pilot


def grounded_pilot() -> tuple[Pilot, np.ndarray]:
    frame = np.zeros((8, 8), dtype=np.int16)
    frame[1, 1] = 9
    pilot = Pilot()
    mech = pilot.mind.mech
    mech.background = 0
    mech.avatar = 9
    mech.body = {9}
    mech.deltas = {1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0)}
    mech.passable = Counter({0: 1})
    mech.pos = (1, 1)
    return pilot, frame


class PilotCoverageTests(unittest.TestCase):
    def test_cover_walks_to_unvisited_reachable_cell(self):
        pilot, frame = grounded_pilot()
        pilot.visited.add((1, 1))

        plan = pilot._cover(frame, [1, 2, 3, 4], ["UP", "DOWN", "LEFT", "RIGHT"])

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.phase, "cover")
        self.assertGreater(len(plan.presses), 0)
        # The remembered route includes more than the start, preventing the next
        # laboratory turn from choosing the exact same corridor again.
        self.assertGreater(len(pilot.visited), 1)

    def test_cover_is_not_used_on_an_unconceded_later_level(self):
        pilot, frame = grounded_pilot()
        pilot.level = 2

        self.assertIsNone(pilot._cover(frame, [1, 2, 3, 4], ["UP", "DOWN", "LEFT", "RIGHT"]))

    def test_level_change_forgets_board_coordinates_but_keeps_mechanics(self):
        pilot, _frame = grounded_pilot()
        pilot.visited.update({(1, 1), (2, 1)})
        before = dict(pilot.mind.mech.moves)

        pilot._roll(2, 0)

        self.assertEqual(pilot.visited, set())
        self.assertEqual(pilot.mind.mech.moves, before)

    def test_contextual_use_is_tried_once_at_an_adjacent_unknown_object(self):
        pilot, frame = grounded_pilot()
        frame[1, 2] = 7
        pilot.mind.mech.deltas = {4: (0, 1)}
        pilot.mind.mech.shifts[4] = 2

        plan = pilot._use_frontier(frame, [4, 5], ["RIGHT", "SPACE"])

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.phase, "use-frontier")
        self.assertEqual(plan.presses[0].aid, 5)
        self.assertIn((7, 5), pilot.tried_use)
        self.assertIsNone(pilot._use_frontier(frame, [4, 5], ["RIGHT", "SPACE"]))

    def test_stalled_goal_route_yields_to_other_level_phases(self):
        """A route that changed no informative state must not replay forever."""
        pilot, frame = grounded_pilot()
        pilot.goal_hint = {2}
        frame[1, 4] = 2
        pilot.stalls = 1

        self.assertIsNone(pilot._execute(frame, [1, 2, 3, 4], ["UP", "DOWN", "LEFT", "RIGHT"]))


if __name__ == "__main__":
    unittest.main()
