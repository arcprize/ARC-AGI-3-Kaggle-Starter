"""Tests for the observe -> imagine -> execute bridge in ``Pilot``."""

from collections import Counter
from types import SimpleNamespace
import unittest

import numpy as np

from arc3x.pilot import Pilot


def entry(action, grid, level=1):
    return SimpleNamespace(
        action=action,
        frame=SimpleNamespace(
            grid=tuple(tuple(int(v) for v in row) for row in grid), level=level
        ),
    )


def imagined_fixture() -> tuple[Pilot, np.ndarray]:
    """A two-step pickup that is safe to solve entirely in the dream."""
    frame = np.zeros((8, 8), dtype=np.int16)
    frame[1, 1] = 9
    frame[1, 3] = 2
    pilot = Pilot()
    mech = pilot.mind.mech
    mech.background = 0
    mech.avatar = 9
    mech.body = {9}
    mech.deltas = {4: (0, 1)}
    mech.passable = Counter({0: 1})
    mech.pos = (1, 1)
    # The target disappeared under a previous real movement observation, and
    # eight later held-out moves confirmed the movement copy.  The fixture
    # deliberately does not use a named game, a fixed route, or a goal colour.
    pilot.dream.collect = Counter({2: 1})
    pilot.dream.hits_move = 8
    return pilot, frame


def imagined_click_fixture() -> tuple[Pilot, np.ndarray]:
    """A learned paint rule that makes a ratcheting on-board objective."""
    frame = np.zeros((8, 8), dtype=np.int16)
    frame[2, 2] = 3  # a colour with prior click-response evidence
    pilot = Pilot()
    pilot.mind.mech.background = 0
    # The outer pilot uses this to shortlist an on-board candidate.
    pilot.progress.rose = Counter({2: 2})
    # The dream uses the same evidence as a scalar before/after objective.
    pilot.dream.prog.rose = Counter({2: 2})
    pilot.dream.prog.peak = Counter({2: 1})
    click = pilot.click_model
    click.n = 4
    click.support = Counter({"paint": 4})
    click.fills = Counter({2: 4})
    click.fill_cells = {(0, 0), (0, 1), (1, 0), (1, 1)}
    click.live = Counter({3: 4})
    return pilot, frame


class MentalPilotTests(unittest.TestCase):
    def test_dream_shares_the_observed_mechanics_model(self):
        pilot = Pilot()
        self.assertIs(pilot.dream.m, pilot.mind.mech)

        before = np.zeros((6, 6), dtype=np.int16)
        after = before.copy()
        after[0, 0] = 1
        pilot.observe([entry("", before), entry("UP", after)])

        self.assertEqual(pilot._dream_seen, 2)
        # The first prediction must abstain: it was scored before the new
        # transition taught the mechanics learner anything.
        self.assertEqual(pilot.dream.abstains, 1)

    def test_conservative_sidecar_can_skip_mental_observation(self):
        """v20's sidecar remains a control rather than a hidden mental variant."""
        pilot = Pilot()
        before = np.zeros((6, 6), dtype=np.int16)
        after = before.copy()
        after[0, 0] = 1

        pilot.observe([entry("", before), entry("UP", after)], observe_dream=False)

        self.assertEqual(pilot._dream_seen, 0)
        self.assertEqual(pilot.dream.abstains, 0)
        self.assertEqual(pilot.mind.seen, 1)

    def test_confident_dream_returns_a_complete_objective_improving_plan(self):
        pilot, frame = imagined_fixture()

        plan = pilot._imagined_plan(frame, [4], ["RIGHT"])

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.phase, "imagine")
        self.assertEqual([press.aid for press in plan.presses], [4, 4])
        self.assertIn("4096->0", plan.why)

    def test_unverified_dream_cannot_take_an_action(self):
        pilot, frame = imagined_fixture()
        pilot.dream.hits_move = 0

        self.assertIsNone(pilot._imagined_plan(frame, [4], ["RIGHT"]))
        self.assertIsNone(
            pilot.assist(frame, ["RIGHT"], 1, max_actions=4, allow_imagination=True)
        )

    def test_mental_sidecar_executes_only_one_step_then_replans(self):
        pilot, frame = imagined_fixture()

        plan = pilot.assist(frame, ["RIGHT"], 1, max_actions=4, allow_imagination=True)

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.phase, "sidecar-imagine")
        self.assertEqual(len(plan.presses), 1)
        self.assertEqual(plan.presses[0].aid, 4)
        self.assertEqual(pilot.sidecar_actions, 1)

    def test_held_out_click_predictions_must_be_accurate_before_planning(self):
        pilot, frame = imagined_click_fixture()
        click = pilot.click_model
        predicted = click.predict(frame, 2, 2)
        self.assertIsNotNone(predicted)
        assert predicted is not None
        for _ in range(4):
            self.assertEqual(click.grade(frame, predicted, predicted), "hit")

        self.assertTrue(click.predictive)
        plan = pilot._imagined_plan(frame, [6], ["MOUSE"])

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.phase, "imagine-click")
        self.assertEqual((plan.presses[0].row, plan.presses[0].col), (2, 2))

    def test_unverified_click_rule_cannot_take_an_mental_action(self):
        pilot, frame = imagined_click_fixture()

        self.assertIsNone(pilot._imagined_plan(frame, [6], ["MOUSE"]))


if __name__ == "__main__":
    unittest.main()
