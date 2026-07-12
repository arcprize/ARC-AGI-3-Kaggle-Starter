from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from agi_core.knowledge.solver_selector import SolverSelector
from agi_core.orch_or.game_model import GameFeatures, GameModel
from agi_core.storage.memory_cache import drop_cache
from agi_core.types import ActionDecision, ObservedTransition


class SolverSelectorFeedbackTests(unittest.TestCase):
    def test_solver_learning_waits_for_observation_and_aggregates_level_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "solver.sqlite3"
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


if __name__ == "__main__":
    unittest.main()
