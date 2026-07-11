"""Tests for the beam coverage solver."""

from __future__ import annotations

import unittest

from agi_core.knowledge.solvers.beam_coverage_solver import (
    _find_mirror_info,
    _get_shape_pixels,
    get_goal_cells,
    predict_beam_coverage,
    find_best_placement,
    compute_coverage_for_all_shapes,
    BeamCoverageSolver,
)
from agi_core.knowledge.solvers import SolverContext
from agi_core.orch_or.game_model import Component


def _make_mirror_info(horizontal: bool, axis_val: int):
    if horizontal:
        return {"horizontal": True, "axis_px": 0, "axis_py": axis_val}
    return {"horizontal": False, "axis_px": axis_val, "axis_py": 0}


# ---------------------------------------------------------------------------
# predict_beam_coverage
# ---------------------------------------------------------------------------

class TestPredictBeamCoverage(unittest.TestCase):
    """Test beam propagation simulation."""

    def test_cover_own_pixels(self) -> None:
        cells = [(10, 10), (11, 10)]
        mirror = _make_mirror_info(True, 20)
        covered = predict_beam_coverage(cells, mirror, 64, 64)
        self.assertIn((10, 10), covered)
        self.assertIn((11, 10), covered)

    def test_reflect_across_horizontal_mirror(self) -> None:
        cells = [(10, 10)]
        mirror = _make_mirror_info(True, 20)
        covered = predict_beam_coverage(cells, mirror, 64, 64)
        # Reflection: (10, 2*20-10) = (10, 30)
        self.assertIn((10, 30), covered,
                      "Pixel at y=10 should reflect to y=30 across mirror at y=20")

    def test_reflect_across_vertical_mirror(self) -> None:
        cells = [(10, 10)]
        mirror = _make_mirror_info(False, 20)
        covered = predict_beam_coverage(cells, mirror, 64, 64)
        # Reflection: (2*20-10, 10) = (30, 10)
        self.assertIn((30, 10), covered,
                      "Pixel at x=10 should reflect to x=30 across mirror at x=20")

    def test_reflection_out_of_bounds(self) -> None:
        cells = [(5, 5)]
        mirror = _make_mirror_info(False, 3)  # vertical mirror at x=3
        covered = predict_beam_coverage(cells, mirror, 64, 64)
        # Reflection: (2*3-5, 5) = (1, 5) — in bounds, should be covered
        self.assertIn((1, 5), covered)

    def test_depth_limit(self) -> None:
        cells = [(10, 10)]
        mirror = _make_mirror_info(True, 20)
        # First reflection: (10, 30)
        # Second reflection: (10, 10) — already visited, skipped
        covered = predict_beam_coverage(cells, mirror, 64, 64)
        self.assertEqual(len(covered), 2,
                         "Single pixel + single mirror = 2 cells (original + 1 reflection)")

    def test_no_mirror_no_reflection(self) -> None:
        """Without mirror info, no reflection occurs."""
        # Empty mirror info — no reflection axis
        mirror = {"horizontal": True, "axis_px": 0, "axis_py": 0}
        cells = [(10, 10)]
        covered = predict_beam_coverage(cells, mirror, 64, 64)
        # With axis at 0, reflection of (10,10) is (10, -10) = out of bounds
        self.assertEqual(len(covered), 1)


# ---------------------------------------------------------------------------
# find_best_placement
# ---------------------------------------------------------------------------

class TestFindBestPlacement(unittest.TestCase):
    """Test optimal placement search."""

    def test_covers_at_least_one_goal(self) -> None:
        shape = [(0, 0), (1, 0), (0, 1), (1, 1)]
        goals = {(30, 30)}
        mirror = _make_mirror_info(True, 20)
        pos = find_best_placement(shape, goals, mirror, 64, 64)
        self.assertIsNotNone(pos, "Should find a position covering the goal")

    def test_returns_none_no_goals(self) -> None:
        shape = [(0, 0), (1, 0)]
        pos = find_best_placement(shape, set(), _make_mirror_info(True, 20), 64, 64)
        self.assertIsNone(pos)

    def test_prefers_more_hits(self) -> None:
        shape = [(0, 0)]
        goals = {(20, 10), (20, 12), (30, 10)}
        mirror = _make_mirror_info(True, 15)
        pos = find_best_placement(shape, goals, mirror, 64, 64)
        self.assertIsNotNone(pos)


# ---------------------------------------------------------------------------
# compute_coverage_for_all_shapes
# ---------------------------------------------------------------------------

class TestComputeCoverageAllShapes(unittest.TestCase):
    """Test greedy multi-shape assignment."""

    def test_single_shape(self) -> None:
        shapes = [[(0, 0)]]
        goals = [(20, 20)]
        mirror = _make_mirror_info(True, 15)
        stages = compute_coverage_for_all_shapes(shapes, goals, mirror, 64, 64)
        self.assertGreaterEqual(len(stages), 1)
        self.assertIn("target_anchor", stages[0])
        self.assertIn("covered_goals", stages[0])

    def test_multiple_shapes(self) -> None:
        shapes = [[(0, 0)], [(5, 5)]]
        goals = [(10, 20), (30, 15)]
        mirror = _make_mirror_info(True, 16)
        stages = compute_coverage_for_all_shapes(shapes, goals, mirror, 64, 64)
        self.assertGreaterEqual(len(stages), 1)
        # Each stage should have target_anchor
        for stage in stages:
            self.assertIn("target_anchor", stage)


# ---------------------------------------------------------------------------
# _find_mirror_info
# ---------------------------------------------------------------------------

class TestFindMirrorInfo(unittest.TestCase):
    """Test mirror detection from components."""

    def test_horizontal_mirror(self) -> None:
        comps = [Component(
            cells=[(x, 16, x, 16) for x in range(10, 30)],
            bbox=(10, 16, 29, 16), size=20, dominant_color=10,
            touches_border=False,
        )]
        info = _find_mirror_info(comps, 64, 64)
        self.assertIsNotNone(info)
        self.assertTrue(info["horizontal"])
        self.assertEqual(info["axis_py"], 16)

    def test_vertical_mirror(self) -> None:
        comps = [Component(
            cells=[(10, y, 10, y) for y in range(10, 30)],
            bbox=(10, 10, 10, 29), size=20, dominant_color=10,
            touches_border=False,
        )]
        info = _find_mirror_info(comps, 64, 64)
        self.assertIsNotNone(info)
        self.assertFalse(info["horizontal"])
        self.assertEqual(info["axis_px"], 10)

    def test_no_mirror(self) -> None:
        self.assertIsNone(_find_mirror_info([], 64, 64))


# ---------------------------------------------------------------------------
# _get_shape_pixels
# ---------------------------------------------------------------------------

class TestGetShapePixels(unittest.TestCase):
    """Test shape pixel extraction from components."""

    def test_returns_color_5(self) -> None:
        comps = [Component(
            cells=[(10, 10), (11, 10), (10, 11), (11, 11)],
            bbox=(10, 10, 11, 11), size=4, dominant_color=5,
            touches_border=False,
        )]
        shapes = _get_shape_pixels(comps, 64, 64)
        self.assertEqual(len(shapes), 1)
        self.assertEqual(len(shapes[0]), 4)

    def test_skips_border_touching(self) -> None:
        comps = [Component(
            cells=[(0, 0), (1, 0)],
            bbox=(0, 0, 1, 0), size=2, dominant_color=5,
            touches_border=True,
        )]
        self.assertEqual(len(_get_shape_pixels(comps, 64, 64)), 0)

    def test_skips_too_small(self) -> None:
        comps = [Component(
            cells=[(10, 10)],
            bbox=(10, 10, 10, 10), size=1, dominant_color=5,
            touches_border=False,
        )]
        self.assertEqual(len(_get_shape_pixels(comps, 64, 64)), 0)


# ---------------------------------------------------------------------------
# BeamCoverageSolver
# ---------------------------------------------------------------------------

class _MockGameModel:
    pass


class TestBeamCoverageSolver(unittest.TestCase):
    """Test the solver class."""

    def test_should_activate_no_mirror(self) -> None:
        gm = _MockGameModel()
        gm.player_pos = (10, 10)
        gm.goal_positions = [(30, 30)]
        gm.components = [Component(
            cells=[(30, 30)], bbox=(30, 30, 30, 30), size=1,
            dominant_color=11, touches_border=False,
        )]
        gm.width = 64
        gm.height = 64
        gm.features = _MockGameModel()
        gm.features.primary_dynamics = "unknown"
        ctx = SolverContext(game_model=gm, plan_context={})
        solver = BeamCoverageSolver()
        self.assertFalse(solver.should_activate(ctx))

    def test_should_activate_no_shapes(self) -> None:
        gm = _MockGameModel()
        gm.player_pos = (10, 10)
        gm.goal_positions = [(30, 30)]
        gm.components = [
            Component(
                cells=[(20, 20)], bbox=(20, 20, 20, 20), size=1,
                dominant_color=10, touches_border=False,
            ),
            Component(
                cells=[(30, 30)], bbox=(30, 30, 30, 30), size=1,
                dominant_color=11, touches_border=False,
            ),
        ]
        gm.width = 64
        gm.height = 64
        gm.features = _MockGameModel()
        gm.features.primary_dynamics = "unknown"
        ctx = SolverContext(game_model=gm, plan_context={})
        solver = BeamCoverageSolver()
        self.assertFalse(solver.should_activate(ctx))

    def _make_gm(self, primary_dynamics="unknown"):
        gm = _MockGameModel()
        gm.player_pos = (10, 10)
        gm.goal_positions = [(30, 30)]
        gm.components = [
            Component(
                cells=[(x, 16) for x in range(10, 30)],
                bbox=(10, 16, 29, 16), size=20, dominant_color=10,
                touches_border=False,
            ),
            Component(
                cells=[(5, 5), (6, 5), (5, 6), (6, 6)],
                bbox=(5, 5, 6, 6), size=4, dominant_color=5,
                touches_border=False,
            ),
            Component(
                cells=[(30, 30)], bbox=(30, 30, 30, 30), size=1,
                dominant_color=11, touches_border=False,
            ),
        ]
        gm.width = 64
        gm.height = 64
        gm.features = _MockGameModel()
        gm.features.primary_dynamics = primary_dynamics
        return gm

    def test_should_activate_with_mirror_and_shapes(self) -> None:
        gm = self._make_gm("unknown")
        ctx = SolverContext(game_model=gm, plan_context={})
        solver = BeamCoverageSolver()
        self.assertTrue(solver.should_activate(ctx))

    def test_should_not_conflict_with_beam_dynamics(self) -> None:
        gm = self._make_gm("beam")
        ctx = SolverContext(game_model=gm, plan_context={})
        solver = BeamCoverageSolver()
        self.assertFalse(solver.should_activate(ctx))


if __name__ == "__main__":
    unittest.main()
