"""Tests for the beam coverage solver."""

from __future__ import annotations

import unittest

from agi_core.knowledge.solvers.beam_coverage_solver import (
    _find_mirror_info,
    _get_shape_pixels,
    get_goal_cells,
    predict_beam_coverage,
    predict_beam_cells,
    covered_goal_cells,
    _base_forbidden_cells,
    _forbidden_cells_for_mirror_row,
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
        self.assertIn((10, 10), covered)  # grid (3,3) center pixel
        # Both cells map to same grid cell (3,3), so only one pixel returned

    def test_reflect_across_horizontal_mirror(self) -> None:
        cells = [(10, 10)]
        mirror = _make_mirror_info(True, 20)
        covered = predict_beam_coverage(cells, mirror, 64, 64)
        # Grid: cell at (3,3), mirror at grid y=6 (20//3), reflection at grid (3,9)
        # Pixel center of (3,9) = (10, 28)
        self.assertIn((10, 28), covered,
                      "Pixel at (10,10) reflects to grid (3,9)=pixel (10,28)")

    def test_reflect_across_vertical_mirror(self) -> None:
        cells = [(10, 10)]
        mirror = _make_mirror_info(False, 20)
        covered = predict_beam_coverage(cells, mirror, 64, 64)
        # Grid: cell at (3,3), mirror at grid x=6 (20//3), reflection at grid (9,3)
        # Pixel center of (9,3) = (28, 10)
        self.assertIn((28, 10), covered,
                      "Pixel at (10,10) reflects to grid (9,3)=pixel (28,10)")

    def test_reflection_out_of_bounds(self) -> None:
        cells = [(5, 5)]
        mirror = _make_mirror_info(False, 3)  # vertical mirror at pixel x=3
        covered = predict_beam_coverage(cells, mirror, 64, 64)
        # Grid: cell at (1,1), mirror at grid x=1 (3//3=1), reflection at grid (1,1)
        # Same cell — reflection is self (within same grid cell)
        self.assertIn((4, 4), covered, "Pixel (5,5) and its reflection (same grid cell)")

    def test_depth_limit(self) -> None:
        cells = [(10, 10)]
        mirror = _make_mirror_info(True, 20)
        # Grid reflection: (3,3) → (3,9) → (3,3) already visited
        covered = predict_beam_coverage(cells, mirror, 64, 64)
        # 2 grid cells x 9 pixels each = 18 pixels
        self.assertEqual(len(covered), 18,
                         "2 grid cells (original + reflection) x 9 pixels each = 18")

    def test_no_mirror_no_reflection(self) -> None:
        """Without mirror info, no reflection occurs."""
        mirror = {"horizontal": True, "axis_px": 0, "axis_py": 0}
        cells = [(10, 10)]
        covered = predict_beam_coverage(cells, mirror, 64, 64)
        # 1 grid cell = 9 pixels (reflection at grid (3,-3) out of bounds)
        self.assertEqual(len(covered), 9)

    def test_logical_cells_match_sprite_reflection_coordinates(self) -> None:
        cells = [(12, 12)]  # logical cell (4, 4)
        mirror = _make_mirror_info(True, 30)  # logical mirror row 10
        self.assertEqual(
            predict_beam_cells(cells, mirror, 64, 64),
            {(4, 4), (4, 16)},
        )

    def test_goal_evaluator_deduplicates_rendered_goal_pixels(self) -> None:
        beam = {(4, 14)}
        goals = [(x, y) for x in range(12, 15) for y in range(42, 45)]
        self.assertEqual(covered_goal_cells(beam, goals), {(4, 14)})

    def test_level3_engine_fixture_accepts_source_and_reflection_union(self) -> None:
        l_shape = [(11 + dx, 14 + dy) for dx, dy in
                   [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3)]]
        dumbbell = [(3 + dx, 14 + dy) for dx, dy in
                    [(0, 0), (1, 0), (2, 0), (3, 0), (1, 1), (2, 1)]]
        goals = [
            (14, 17), (14, 1), (11, 4), (11, 14), (11, 17), (11, 1),
            (3, 14), (4, 14), (5, 14), (6, 14), (4, 15), (5, 15),
            (3, 4), (4, 4), (5, 4), (6, 4), (5, 3), (4, 3),
            (11, 15), (11, 16), (12, 17), (13, 17), (11, 3), (11, 2),
            (12, 1), (13, 1),
        ]
        mirror = _make_mirror_info(True, 27)
        beam = predict_beam_cells(
            [(x * 3, y * 3) for x, y in l_shape], mirror, 63, 63, gs=3,
        ) | predict_beam_cells(
            [(x * 3, y * 3) for x, y in dumbbell], mirror, 63, 63, gs=3,
        )
        self.assertEqual(
            covered_goal_cells(beam, [(x * 3, y * 3) for x, y in goals]),
            {(x, y) for x, y in goals},
        )

    def test_beam_placement_allows_overlap_and_beam_crossing(self) -> None:
        self.assertEqual(_base_forbidden_cells(object()), set())
        self.assertEqual(
            _forbidden_cells_for_mirror_row({(1, 1)}, 63, 27),
            {(1, 1)},
        )


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

    def test_prefers_complete_shape_recognition_goal_set(self) -> None:
        gm = self._make_gm("unknown")
        gm.goal_positions = [(30, 30)]
        full_goals = [[x * 3, y * 3] for x, y in ((1, 1), (2, 2), (3, 3), (4, 4))]
        ctx = SolverContext(
            game_model=gm,
            plan_context={
                "shape_recognition": {
                    "mirror_puzzle": {"goal_cells": full_goals},
                },
            },
        )
        solver = BeamCoverageSolver()
        self.assertEqual(solver._goal_cells(ctx, None), [tuple(goal) for goal in full_goals])
        self.assertTrue(solver.should_activate(ctx))
        self.assertEqual(solver._planned_goal_cells, [tuple(goal) for goal in full_goals])

    def test_goal_set_survives_component_dropout(self) -> None:
        gm = self._make_gm("unknown")
        full_goals = [[3, 3], [6, 6], [9, 9]]
        ctx = SolverContext(
            game_model=gm,
            plan_context={
                "shape_recognition": {"mirror_puzzle": {"goal_cells": full_goals}},
            },
        )
        solver = BeamCoverageSolver()
        expected = [tuple(goal) for goal in full_goals]
        self.assertEqual(solver._goal_cells(ctx, None), expected)
        ctx.plan_context["shape_recognition"] = {"mirror_puzzle": {"goal_cells": []}}
        gm.components = []
        self.assertEqual(solver._goal_cells(ctx, None), expected)

    def test_active_attempt_survives_mirror_and_shape_dropout(self) -> None:
        gm = self._make_gm("beam")
        gm.components = []
        gm.goal_positions = []
        gm.features.grid_scale = 3
        gm.features.player_color = 5
        ctx = SolverContext(
            game_model=gm,
            plan_context={
                "levels_completed": 0,
                "shape_recognition": {"mirror_puzzle": {"goal_cells": [[3, 3], [6, 6]]}},
            },
        )
        solver = BeamCoverageSolver()
        solver._last_levels_completed = 0
        solver._planned_shape_count = 1
        solver._active_stage_index = 0
        solver._cached_mirror_info = _make_mirror_info(True, 27)
        solver._mirror_target_gy = 9
        solver._cached_targets = [{"target_anchor": [37, 49], "covered_goals": []}]
        solver._base_targets = [dict(solver._cached_targets[0])]
        solver._planned_goal_cells = [(3, 3), (6, 6)]

        self.assertTrue(solver.has_active_attempt)
        self.assertTrue(solver.should_activate(ctx))
        decision = solver.decide(ctx)
        self.assertIsNotNone(decision)
        self.assertNotEqual(decision.reason, "coverage_position_mirror_up")
        self.assertEqual(solver._mirror_target_gy, 9)
        self.assertEqual(solver._cached_targets[0]["target_anchor"], [37, 49])

    def test_should_not_conflict_with_beam_dynamics(self) -> None:
        gm = self._make_gm("beam")
        ctx = SolverContext(game_model=gm, plan_context={})
        solver = BeamCoverageSolver()
        self.assertFalse(solver.should_activate(ctx))


if __name__ == "__main__":
    unittest.main()
