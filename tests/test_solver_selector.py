from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from agi_core.knowledge.solver_selector import SolverSelector
from agi_core.knowledge.solvers import DynamicsSolver, register_solver
from agi_core.knowledge.solvers.mirror_solver import MirrorSolver
from agi_core.orch_or.game_model import GameFeatures, GameModel
from agi_core.storage.memory_cache import drop_cache
from agi_core.types import ActionDecision, ObservedTransition


class _DummyNavigationSolver(DynamicsSolver):
    dynamics = "dummy-navigation"

    def __init__(self, memory_db_path=None) -> None:
        super().__init__(memory_db_path)
        self.reset_calls = 0

    def should_activate(self, ctx):
        return True

    def decide(self, ctx):
        return None

    def reset_level(self) -> None:
        self.reset_calls += 1


class _DummyMirrorSolver(DynamicsSolver):
    dynamics = "dummy-mirror"

    def __init__(self, memory_db_path=None) -> None:
        super().__init__(memory_db_path)
        self.reset_calls = 0

    def should_activate(self, ctx):
        return True

    def decide(self, ctx):
        return None

    def reset_level(self) -> None:
        self.reset_calls += 1


register_solver("dummy-navigation", _DummyNavigationSolver)
register_solver("dummy-mirror", _DummyMirrorSolver)


class SolverSelectorFeedbackTests(unittest.TestCase):
    def _workspace_tempdir(self) -> Path:
        root = Path.cwd() / "tmp" / "test_solver_selector"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _make_game_model(self, dynamics: str = "navigation") -> GameModel:
        return GameModel(
            board=[[0]],
            height=1,
            width=1,
            player_pos=(0, 0),
            goal_positions=[(0, 0)],
            obstacles=[],
            components=[],
            walkable={(0, 0)},
            features=GameFeatures(primary_dynamics=dynamics),
        )

    def test_solver_learning_waits_for_observation_and_aggregates_level_success(self) -> None:
        tmpdir = self._workspace_tempdir()
        db_path = tmpdir / "solver-learning.sqlite3"
        try:
            if db_path.exists():
                db_path.unlink()
            hyperon = SimpleNamespace(facts={"episode": {}, "memory": {}})
            selector = SolverSelector(hyperon, memory_db_path=db_path)
            solver = MagicMock()
            solver.dynamics = "beam"
            selector._active_solver = solver
            selector._active_name = "beam"
            game_model = GameModel(
                board=[[0]],
                height=1,
                width=1,
                player_pos=(0, 0),
                goal_positions=[(0, 0)],
                obstacles=[],
                components=[],
                walkable={(0, 0)},
                features=GameFeatures(primary_dynamics="beam"),
            )
            context = SimpleNamespace(
                observation=SimpleNamespace(game_id="vc33", step_index=1),
                hyperon=hyperon,
            )
            first = ActionDecision(
                action_name="ACTION1",
                source="solver:beam",
                confidence=0.8,
                reason="move",
            )

            selector.record_outcome(context, game_model, first, {})
            solver.learn.assert_not_called()
            selector.note_transition(
                ObservedTransition(
                    game_id="vc33",
                    action_name="ACTION1",
                    decision_source="solver:beam",
                    local_outcome="advanced",
                    player_moved=True,
                )
            )
            solver.learn.assert_called_once()

            context.observation.step_index = 2
            second = ActionDecision(
                action_name="ACTION2",
                source="solver:beam",
                confidence=0.8,
                reason="finish",
            )
            selector.record_outcome(context, game_model, second, {})
            selector.note_transition(
                ObservedTransition(
                    game_id="vc33",
                    action_name="ACTION2",
                    decision_source="solver:beam",
                    local_outcome="progress",
                    boundary="level",
                )
            )

            self.assertEqual(solver.learn.call_count, 2)
            with sqlite3.connect(db_path) as connection:
                row = connection.execute(
                    """SELECT attempt_count, success_count, avg_steps_to_complete
                       FROM solver_outcomes
                       WHERE game_id = ? AND dynamics = ? AND solver_name = ?""",
                    ["vc33", "beam", "beam"],
                ).fetchone()
            connection.close()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], 1)
            self.assertEqual(row[1], 1)
            self.assertEqual(row[2], 2.0)
            drop_cache(db_path)
        finally:
            if db_path.exists():
                db_path.unlink()

    def test_select_solver_switches_when_active_solver_has_no_attempt(self) -> None:
        hyperon = SimpleNamespace(
            facts={"episode": {}, "memory": {}},
            run_rules=lambda query: [{"solver": "dummy-mirror"}] if query == "solver-select" else [],
        )
        selector = SolverSelector(hyperon)
        old_solver = _DummyNavigationSolver()
        selector._active_solver = old_solver
        selector._active_name = "dummy-navigation"
        context = SimpleNamespace(observation=SimpleNamespace(last_action="ACTION1"))

        selected = selector.select_solver(self._make_game_model("puzzle"), context)

        self.assertIsInstance(selected, _DummyMirrorSolver)
        self.assertEqual(old_solver.reset_calls, 1)
        self.assertEqual(selector._active_name, "dummy-mirror")

    def test_select_solver_retains_active_attempt_across_recommendation_change(self) -> None:
        hyperon = SimpleNamespace(
            facts={"episode": {}, "memory": {}},
            run_rules=lambda query: [{"solver": "dummy-mirror"}] if query == "solver-select" else [],
        )
        selector = SolverSelector(hyperon)
        old_solver = _DummyNavigationSolver()
        old_solver.has_active_attempt = True
        selector._active_solver = old_solver
        selector._active_name = "dummy-navigation"
        context = SimpleNamespace(observation=SimpleNamespace(last_action="ACTION1"))

        selected = selector.select_solver(self._make_game_model("puzzle"), context)

        self.assertIs(selected, old_solver)
        self.assertEqual(old_solver.reset_calls, 0)
        self.assertEqual(selector._active_name, "dummy-navigation")
        self.assertEqual(selector.last_retention_reason, "active-attempt")

    def test_select_solver_switches_after_action5_when_recommendation_changes(self) -> None:
        hyperon = SimpleNamespace(
            facts={"episode": {}, "memory": {}},
            run_rules=lambda query: [{"solver": "dummy-mirror"}] if query == "solver-select" else [],
        )
        selector = SolverSelector(hyperon)
        old_solver = _DummyNavigationSolver()
        selector._active_solver = old_solver
        selector._active_name = "dummy-navigation"
        context = SimpleNamespace(observation=SimpleNamespace(last_action="ACTION5"))

        selected = selector.select_solver(self._make_game_model("puzzle"), context)

        self.assertIsInstance(selected, _DummyMirrorSolver)
        self.assertEqual(old_solver.reset_calls, 1)
        self.assertEqual(selector._active_name, "dummy-mirror")

    def test_select_solver_retains_beam_coverage_across_generic_beam_recommendation(self) -> None:
        hyperon = SimpleNamespace(
            facts={"episode": {}, "memory": {}},
            run_rules=lambda query: [{"solver": "beam"}] if query == "solver-select" else [],
        )
        selector = SolverSelector(hyperon)
        old_solver = MagicMock()
        old_solver.dynamics = "beam-coverage"
        selector._active_solver = old_solver
        selector._active_name = "beam-coverage"
        context = SimpleNamespace(observation=SimpleNamespace(last_action="ACTION1"))

        selected = selector.select_solver(self._make_game_model("beam"), context)

        self.assertIs(selected, old_solver)
        old_solver.reset_level.assert_not_called()
        self.assertEqual(selector._active_name, "beam-coverage")
        self.assertEqual(selector.last_retention_reason, "beam-coverage-ownership")

    def test_select_solver_does_not_retain_generic_beam_coverage_after_action5(self) -> None:
        hyperon = SimpleNamespace(
            facts={"episode": {}, "memory": {}},
            run_rules=lambda query: [{"solver": "beam"}] if query == "solver-select" else [],
        )
        selector = SolverSelector(hyperon)
        old_solver = MagicMock()
        old_solver.dynamics = "beam-coverage"
        selector._active_solver = old_solver
        selector._active_name = "beam-coverage"
        context = SimpleNamespace(observation=SimpleNamespace(last_action="ACTION5"))

        selected = selector.select_solver(self._make_game_model("beam"), context)

        self.assertIsNot(selected, old_solver)
        old_solver.reset_level.assert_called_once()

    def test_select_solver_retains_mirror_with_active_stage_attempt(self) -> None:
        hyperon = SimpleNamespace(
            facts={"episode": {}, "memory": {}},
            run_rules=lambda query: [{"solver": "gravity"}] if query == "solver-select" else [],
        )
        selector = SolverSelector(hyperon)
        old_solver = MirrorSolver()
        old_solver._switch_count = 1
        selector._active_solver = old_solver
        selector._active_name = "mirror"
        context = SimpleNamespace(observation=SimpleNamespace(last_action="ACTION5"))

        selected = selector.select_solver(self._make_game_model("gravity"), context)

        self.assertIs(selected, old_solver)
        self.assertEqual(selector._active_name, "mirror")


if __name__ == "__main__":
    unittest.main()
