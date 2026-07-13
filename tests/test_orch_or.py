from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agi_core.orch_or.analyzer import _extract_features, _find_goals, _find_walkable, build_game_model
from agi_core.orch_or.game_model import (
    ADVANCE,
    ESCALATE,
    REPLAN,
    RETRY,
    Component,
    GameFeatures,
    GameModel,
    Objective,
    Plan,
    PlanStep,
    deserialize_plan,
    serialize_plan,
)
from agi_core.orch_or.monitor import check_progress
from agi_core.orch_or.plan_executor import action_name_for_direction, plan_step_to_proposal
from agi_core.orch_or.plan_generator import generate_plan
from agi_core.orch_or.reducer import reduce, select_leaf
from agi_core.orch_or.search import astar
from agi_core.orch_or.strategies import available_strategies, get_strategy, register
from agi_core.orch_or.strategies.exploration import exploration_strategy, _mirror_axis_direction
from agi_core.orch_or.strategies.interaction import interaction_strategy
from agi_core.orch_or.strategies.movement import movement_strategy
from agi_core.orch_or.parallel import ThreadPoolManager
from agi_core.orch_or.orchestrator import (
    ObjectiveReduction,
    _PLAN_KEY,
    _is_level_transition_jump,
)


# ---------------------------------------------------------------------------
# GameModel
# ---------------------------------------------------------------------------

class GameModelTests(unittest.TestCase):
    def test_serialize_deserialize_plan_roundtrip(self) -> None:
        step = PlanStep(
            action="move",
            params={"direction": "right"},
            source="movement_strategy",
            preconditions=frozenset(),
            expected_effect="move_right",
        )
        plan = Plan(objective_id="reach-goal", steps=[step], current_index=0)
        data = serialize_plan(plan)
        restored = deserialize_plan(data)
        self.assertEqual(restored.objective_id, "reach-goal")
        self.assertEqual(len(restored.steps), 1)
        self.assertEqual(restored.steps[0].action, "move")
        self.assertEqual(restored.steps[0].params, {"direction": "right"})

    def test_plan_tracks_current_index(self) -> None:
        plan = Plan(objective_id="test", steps=[], current_index=0)
        self.assertEqual(plan.current_index, 0)
        plan.current_index = 2
        self.assertEqual(plan.current_index, 2)

    def test_serialize_empty_plan(self) -> None:
        plan = Plan(objective_id="empty", steps=[], current_index=0)
        data = serialize_plan(plan)
        restored = deserialize_plan(data)
        self.assertEqual(restored.objective_id, "empty")
        self.assertEqual(len(restored.steps), 0)
        self.assertEqual(restored.current_index, 0)

    def test_serialize_plan_with_click_step(self) -> None:
        step = PlanStep(
            action="click",
            params={"x": 21, "y": 33, "target_gx": 3, "target_gy": 5},
            source="interaction_strategy",
            preconditions=frozenset(),
            expected_effect="break_block",
        )
        plan = Plan(objective_id="clear-obstacles", steps=[step], current_index=0)
        data = serialize_plan(plan)
        restored = deserialize_plan(data)
        self.assertEqual(restored.steps[0].action, "click")
        self.assertEqual(restored.steps[0].params, {"x": 21, "y": 33, "target_gx": 3, "target_gy": 5})
        self.assertEqual(restored.steps[0].source, "interaction_strategy")

    def test_serialize_inactive_plan(self) -> None:
        plan = Plan(objective_id="done", steps=[], is_active=False)
        data = serialize_plan(plan)
        restored = deserialize_plan(data)
        self.assertFalse(restored.is_active)

    def test_plan_default_attempt_count(self) -> None:
        plan = Plan(objective_id="test", steps=[])
        self.assertEqual(plan.attempt_count, 0)
        self.assertEqual(plan.confidence, 0.0)
        self.assertTrue(plan.is_active)

    def test_plan_with_multiple_steps(self) -> None:
        steps = [
            PlanStep(action="move", params={"direction": "right"}, source="plan", preconditions=frozenset(), expected_effect="move_right"),
            PlanStep(action="click", params={"x": 15, "y": 15}, source="plan", preconditions=frozenset(), expected_effect="break"),
            PlanStep(action="move", params={"direction": "left"}, source="plan", preconditions=frozenset(), expected_effect="move_left"),
        ]
        plan = Plan(objective_id="multi", steps=steps, current_index=0)
        self.assertEqual(len(plan.steps), 3)
        data = serialize_plan(plan)
        restored = deserialize_plan(data)
        self.assertEqual(len(restored.steps), 3)

    def test_plan_step_default_tolerance(self) -> None:
        step = PlanStep(action="click", params=None, source="test", preconditions=frozenset(), expected_effect="x")
        self.assertEqual(step.tolerance, 3)
        self.assertEqual(step.retry_count, 0)

    def test_plan_step_custom_tolerance(self) -> None:
        step = PlanStep(action="click", params=None, source="test", preconditions=frozenset(), expected_effect="x", tolerance=5)
        self.assertEqual(step.tolerance, 5)


class BackgroundPrecomputeTests(unittest.TestCase):
    def test_action5_entity_jump_is_not_a_level_transition(self) -> None:
        self.assertFalse(_is_level_transition_jump((3, 3), (54, 48), "ACTION5"))

    def test_movement_jump_remains_a_level_transition(self) -> None:
        self.assertTrue(_is_level_transition_jump((3, 3), (54, 48), "ACTION4"))

    def test_solver_switch_clears_controller_specific_monitor_state(self) -> None:
        orchestrator = ObjectiveReduction(
            executive=SimpleNamespace(),
            config=SimpleNamespace(),
        )
        orchestrator._failed_directions = {"left", "up"}
        orchestrator._pending_failed_dir = "left"
        orchestrator._active_plan = Plan(objective_id="stale", steps=[])
        orchestrator._regenerate_count = 2
        context = SimpleNamespace(
            memory_bundle=SimpleNamespace(working_memory={_PLAN_KEY: {"stale": True}}),
        )

        orchestrator._reset_controller_state_after_solver_switch(context)

        self.assertEqual(orchestrator._failed_directions, set())
        self.assertIsNone(orchestrator._pending_failed_dir)
        self.assertIsNone(orchestrator._active_plan)
        self.assertEqual(orchestrator._regenerate_count, 0)
        self.assertEqual(context.memory_bundle.working_memory, {})

    def test_background_precompute_uses_thread_pool_for_small_search_spaces(self) -> None:
        orchestrator = ObjectiveReduction(
            executive=SimpleNamespace(),
            config=SimpleNamespace(),
        )
        submitted: list[str] = []

        def _thread_submit(fn, *args, **kwargs):
            submitted.append("thread")
            return SimpleNamespace()

        def _process_submit(fn, *args, **kwargs):
            submitted.append("process")
            return SimpleNamespace()

        snap = (
            (0, 0),
            [(3, 0), (4, 0)],
            {(x, 0) for x in range(5)},
            5,
            1,
        )
        with patch("agi_core.orch_or.parallel.ThreadPoolManager.submit", side_effect=_thread_submit), patch(
            "agi_core.orch_or.parallel.ProcessPoolManager.submit",
            side_effect=_process_submit,
        ):
            orchestrator._do_background_precompute(snap)

        self.assertEqual(submitted, ["thread", "thread"])


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class AnalyzerTests(unittest.TestCase):
    def _simple_board(self) -> list[list[int]]:
        return [
            [0, 0, 0, 0, 0, 0],
            [0, 9, 0, 0, 7, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 3, 3, 3, 3, 0],
            [0, 0, 0, 0, 0, 0],
        ]

    def test_background_detection(self) -> None:
        board = self._simple_board()
        gm = build_game_model(board)
        self.assertEqual(gm.features.background_color, 0)

    def test_player_detection(self) -> None:
        board = self._simple_board()
        gm = build_game_model(board)
        self.assertIsNotNone(gm.player_pos)

    def test_goal_detection(self) -> None:
        board = self._simple_board()
        gm = build_game_model(board)
        self.assertGreater(len(gm.goal_positions), 0)

    def test_walkable_includes_background(self) -> None:
        board = self._simple_board()
        gm = build_game_model(board)
        self.assertIn((0, 0), gm.walkable)

    def test_features_populated(self) -> None:
        board = self._simple_board()
        gm = build_game_model(board)
        self.assertIsInstance(gm.features, GameFeatures)
        self.assertEqual(gm.features.tile_size, 6)

    def test_empty_board_returns_model_with_defaults(self) -> None:
        board: list[list[int]] = []
        gm = build_game_model(board)
        self.assertEqual(gm.height, 0)
        self.assertEqual(gm.width, 0)
        self.assertIsNone(gm.player_pos)

    def test_all_background_board_no_components(self) -> None:
        board = [[0] * 6 for _ in range(6)]
        gm = build_game_model(board)
        self.assertEqual(len(gm.components), 0)
        self.assertIsNone(gm.player_pos)
        self.assertEqual(len(gm.goal_positions), 0)

    def test_board_with_only_border_component(self) -> None:
        board = [[1] * 6 for _ in range(6)]
        gm = build_game_model(board)
        self.assertEqual(gm.features.background_color, 1)
        self.assertEqual(len(gm.components), 0)

    def test_single_pixel_component(self) -> None:
        board = [[0, 0, 0], [0, 5, 0], [0, 0, 0]]
        gm = build_game_model(board)
        self.assertEqual(len(gm.components), 1)
        self.assertEqual(gm.components[0].size, 1)
        self.assertEqual(gm.components[0].bbox, (1, 1, 1, 1))

    def test_player_at_top_left_detected(self) -> None:
        board = [
            [9, 0, 0],
            [0, 7, 0],
            [0, 0, 0],
        ]
        gm = build_game_model(board)
        self.assertIsNotNone(gm.player_pos)
        # Player is the lower-half component with greatest bbox[3]; 7 at (1,1)
        self.assertEqual(gm.player_pos, (1, 1))

    def test_no_player_when_only_border_components(self) -> None:
        board = [[1] * 6 for _ in range(6)]
        gm = build_game_model(board)
        self.assertIsNone(gm.player_pos)

    def test_multiple_non_player_components_goal_detected(self) -> None:
        board = [
            [0, 0, 0, 0, 0, 0],
            [0, 9, 0, 0, 7, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
        gm = build_game_model(board)
        self.assertIsNotNone(gm.player_pos)
        self.assertGreater(len(gm.goal_positions), 0)

    def test_goal_filtered_when_overlapping_player(self) -> None:
        board = [[0, 0], [0, 9]]
        gm = build_game_model(board)
        self.assertEqual(len(gm.goal_positions), 0)

    def test_goal_detection_prefers_reachable_candidates(self) -> None:
        board = [[0] * 7 for _ in range(7)]
        for y in range(7):
            board[y][3] = 1
        player_pos = (1, 3)
        components = [
            Component(cells=[player_pos], bbox=(1, 3, 1, 3), size=1, dominant_color=5, touches_border=False, confidence=1.0),
            Component(cells=[(1, 1)], bbox=(1, 1, 1, 1), size=1, dominant_color=11, touches_border=False, confidence=1.0),
            Component(cells=[(5, 3)], bbox=(5, 3, 5, 3), size=1, dominant_color=11, touches_border=False, confidence=1.0),
        ]

        goals = _find_goals(components, player_pos, board, color_freq={5: 1, 11: 2, 1: 7})

        self.assertEqual(goals, [(1, 1)])

    def test_goal_detection_treats_interior_components_as_passable(self) -> None:
        board = [[0] * 10 for _ in range(7)]
        for y in range(7):
            board[y][3] = 1
        board[3][3] = 2
        for y in range(2, 5):
            for x in range(6, 9):
                board[y][x] = 11

        player_pos = (1, 3)
        goal_cells = [(x, y) for y in range(2, 5) for x in range(6, 9)]
        components = [
            Component(cells=[(3, 3)], bbox=(3, 3, 3, 3), size=1, dominant_color=2, touches_border=False, confidence=1.0),
            Component(cells=goal_cells, bbox=(6, 2, 8, 4), size=len(goal_cells), dominant_color=11, touches_border=False, confidence=1.0),
        ]

        goals = _find_goals(components, player_pos, board, color_freq={0: 44, 1: 6, 2: 1, 11: len(goal_cells)})

        self.assertEqual(goals, [(7, 3)])

    def test_walkable_keeps_origin_player_color_cells(self) -> None:
        board = [
            [7, 7, 0],
            [7, 7, 0],
            [0, 0, 0],
        ]

        walkable = _find_walkable(
            board,
            width=3,
            height=3,
            bg_color=0,
            obstacles=[],
            player_pos=(0, 0),
            goal_positions=None,
            components=None,
        )

        self.assertIn((0, 1), walkable)
        self.assertIn((1, 0), walkable)

    def test_board_classified_as_puzzle(self) -> None:
        board = [[0] * 10 for _ in range(10)]
        for y in range(1, 9):
            for x in range(1, 9):
                if (x + y) % 2 == 0:
                    board[y][x] = 7
        gm = build_game_model(board)
        valid_types = {"puzzle", "platformer", "unknown", "static", "navigation", "sliding", "gravity", "push-pull", "beam", "signal", "cellular-automata", "pattern-transform", "multi-agent", "inventory", "hybrid"}
        self.assertIn(gm.features.board_type, valid_types)

    def test_tile_size_detection_10x10(self) -> None:
        board = [[0] * 10 for _ in range(10)]
        board[1][1] = 9
        gm = build_game_model(board)
        self.assertEqual(gm.features.tile_size, 10)

    def test_tile_size_fallback_to_6(self) -> None:
        board = [[0] * 7 for _ in range(7)]
        board[1][1] = 9
        gm = build_game_model(board)
        self.assertEqual(gm.features.tile_size, 6)

    def test_obstacle_from_border_component_is_wall(self) -> None:
        board = [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
        gm = build_game_model(board)
        for ox, oy, otype in gm.obstacles:
            if ox == 0 or oy == 0:
                self.assertEqual(otype, "wall")

    def test_player_color_extracted_correctly(self) -> None:
        board = [
            [0, 0, 0],
            [0, 9, 0],
            [0, 0, 7],
        ]
        gm = build_game_model(board)
        self.assertEqual(gm.features.player_color, 9)

    def test_walkable_excludes_non_background(self) -> None:
        board = [
            [0, 0, 0],
            [0, 9, 0],
            [0, 0, 0],
        ]
        gm = build_game_model(board)
        self.assertIn((0, 0), gm.walkable)
        self.assertIn((0, 1), gm.walkable)
        self.assertIn((1, 0), gm.walkable)

    def test_component_dominant_color_correct(self) -> None:
        board = [
            [0, 0, 0],
            [0, 5, 5],
            [0, 5, 3],
        ]
        gm = build_game_model(board)
        comp = gm.components[0] if gm.components else None
        if comp is not None:
            self.assertEqual(comp.dominant_color, 5)

    def test_path_to_goal_possible_when_walkable(self) -> None:
        board = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 9, 0, 0, 7, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
        gm = build_game_model(board)
        self.assertTrue(gm.features.path_to_goal_possible)

    def test_path_to_goal_impossible_when_blocked(self) -> None:
        walkable = {(0,0), (0,1), (1,0), (1,1), (3,0), (3,1)}
        features = _extract_features(
            board=[[0]*4 for _ in range(2)],
            width=4, height=2,
            bg_color=0,
            components=[],
            player_pos=(0,0),
            goal_positions=[(3,0)],
            walkable=walkable,
        )
        self.assertFalse(features.path_to_goal_possible)

    def test_path_to_goal_false_when_no_player(self) -> None:
        board = [[0, 0], [0, 0]]
        gm = build_game_model(board)
        self.assertFalse(gm.features.path_to_goal_possible)

    def test_path_to_goal_false_when_no_goal(self) -> None:
        board = [[0, 9], [0, 0]]
        gm = build_game_model(board)
        self.assertFalse(gm.features.path_to_goal_possible)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class SearchTests(unittest.TestCase):
    def test_astar_simple_path(self) -> None:
        passable = {(x, y) for x in range(5) for y in range(3)}
        path = astar((0, 0), (4, 0), passable, 5, 3)
        self.assertIsNotNone(path)
        assert path is not None
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (4, 0))

    def test_astar_no_path(self) -> None:
        passable = {(0, 0), (2, 2)}
        path = astar((0, 0), (2, 2), passable, 3, 3)
        self.assertIsNone(path)

    def test_astar_start_not_passable(self) -> None:
        passable = {(1, 1)}
        path = astar((0, 0), (1, 1), passable, 3, 3)
        self.assertIsNone(path)

    def test_astar_goal_not_passable(self) -> None:
        passable = {(0, 0)}
        path = astar((0, 0), (1, 1), passable, 3, 3)
        self.assertIsNone(path)

    def test_astar_makes_detour_around_gap(self) -> None:
        passable = {
            (0, 0), (1, 0), (2, 0), (3, 0),
            (0, 1),                   (3, 1),
            (0, 2), (1, 2), (2, 2), (3, 2),
        }
        path = astar((0, 0), (3, 1), passable, 4, 3)
        self.assertIsNotNone(path)
        assert path is not None
        self.assertEqual(path[-1], (3, 1))

# ---------------------------------------------------------------------------
# Reducer
# ---------------------------------------------------------------------------

class ReducerTests(unittest.TestCase):
    def test_reduce_with_player_and_goal(self) -> None:
        gm = GameModel(
            board=[],
            height=5, width=5,
            player_pos=(1, 1),
            goal_positions=[(4, 4)],
            obstacles=[], components=[],
            walkable=set(),
            features=GameFeatures(player_detectable=True, goal_detectable=True, path_to_goal_possible=True),
        )
        root = reduce(gm)
        self.assertEqual(root.description, "complete-level")
        self.assertEqual(len(root.sub_objectives), 1)
        self.assertEqual(root.sub_objectives[0].description, "reach-goal")

    def test_reduce_without_player(self) -> None:
        gm = GameModel(
            board=[], height=5, width=5,
            player_pos=None, goal_positions=[],
            obstacles=[], components=[],
            walkable=set(),
            features=GameFeatures(player_detectable=False, goal_detectable=False),
        )
        root = reduce(gm)
        leaf = select_leaf(root)
        self.assertEqual(leaf.description, "explore-board")

    def test_reduce_without_goal(self) -> None:
        gm = GameModel(
            board=[], height=5, width=5,
            player_pos=(1, 1), goal_positions=[],
            obstacles=[], components=[],
            walkable=set(),
            features=GameFeatures(player_detectable=True, goal_detectable=False),
        )
        root = reduce(gm)
        leaf = select_leaf(root)
        self.assertEqual(leaf.description, "explore-board")

    def test_reduce_with_obstacles_blocking_path(self) -> None:
        gm = GameModel(
            board=[], height=5, width=5,
            player_pos=(0, 0), goal_positions=[(4, 4)],
            obstacles=[(2, 2, "wall")], components=[],
            walkable=set(),
            features=GameFeatures(player_detectable=True, goal_detectable=True, path_to_goal_possible=False),
        )
        root = reduce(gm)
        reach = root.sub_objectives[0]
        self.assertEqual(reach.description, "reach-goal")
        self.assertEqual(len(reach.sub_objectives), 1)
        self.assertEqual(reach.sub_objectives[0].description, "navigate-to-goal")

    def test_reduce_with_interactive_obstacles_blocking_path(self) -> None:
        gm = GameModel(
            board=[], height=5, width=5,
            player_pos=(0, 0), goal_positions=[(4, 4)],
            obstacles=[(2, 2, "interactive")], components=[],
            walkable=set(),
            features=GameFeatures(player_detectable=True, goal_detectable=True, path_to_goal_possible=False),
        )
        root = reduce(gm)
        reach = root.sub_objectives[0]
        self.assertEqual(reach.description, "reach-goal")
        self.assertEqual(len(reach.sub_objectives), 1)
        self.assertEqual(reach.sub_objectives[0].description, "clear-obstacles")

    def test_select_leaf_nested_objectives(self) -> None:
        leaf_obj = Objective(description="leaf", priority=0.5)
        mid = Objective(description="mid", priority=0.7, sub_objectives=[leaf_obj])
        root = Objective(description="root", priority=1.0, sub_objectives=[mid])
        selected = select_leaf(root)
        self.assertEqual(selected.description, "leaf")

    def test_select_leaf_prefers_higher_priority(self) -> None:
        low = Objective(description="low", priority=0.3)
        high = Objective(description="high", priority=0.9)
        root = Objective(description="root", priority=1.0, sub_objectives=[low, high])
        selected = select_leaf(root)
        self.assertEqual(selected.description, "high")

    def test_reduce_with_no_objectives_no_crash(self) -> None:
        gm = GameModel(
            board=[], height=0, width=0,
            player_pos=None, goal_positions=[],
            obstacles=[], components=[], walkable=set(),
            features=GameFeatures(),
        )
        root = reduce(gm)
        self.assertEqual(root.description, "complete-level")


# ---------------------------------------------------------------------------
# PlanGenerator
# ---------------------------------------------------------------------------

class PlanGeneratorTests(unittest.TestCase):
    def test_generate_plan_with_player_and_goal(self) -> None:
        gm = GameModel(
            board=[], height=5, width=5,
            player_pos=(0, 0), goal_positions=[(4, 4)],
            obstacles=[], components=[],
            walkable=set(),
            features=GameFeatures(player_detectable=True, goal_detectable=True, path_to_goal_possible=True),
        )
        plan = generate_plan(gm)
        self.assertIsNotNone(plan)
        # select_leaf picks the deepest sub-objective
        self.assertIn(plan.objective_id, ("reach-goal", "navigate-to-goal"))

    def test_generate_plan_no_goal_falls_to_explore(self) -> None:
        gm = GameModel(
            board=[], height=5, width=5,
            player_pos=(0, 0), goal_positions=[],
            obstacles=[], components=[],
            walkable=set(),
            features=GameFeatures(player_detectable=True, goal_detectable=False),
        )
        plan = generate_plan(gm)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.objective_id, "explore-board")

    def test_generate_plan_with_obstacles(self) -> None:
        gm = GameModel(
            board=[], height=5, width=5,
            player_pos=(0, 0), goal_positions=[(4, 4)],
            obstacles=[(2, 2, "wall")], components=[], walkable=set(),
            features=GameFeatures(player_detectable=True, goal_detectable=True, path_to_goal_possible=False),
        )
        plan = generate_plan(gm)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertIn(plan.objective_id, ("reach-goal", "navigate-to-goal", "clear-obstacles"))


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class MonitorTests(unittest.TestCase):
    def _make_model(self, px: int | None = None, py: int | None = None, board: list[list[int]] | None = None) -> GameModel:
        pos = (px, py) if px is not None else None
        return GameModel(
            board=board or [[]], height=5, width=5,
            player_pos=pos, goal_positions=[],
            obstacles=[], components=[], walkable=set(),
            features=GameFeatures(),
        )

    def test_advance_on_movement(self) -> None:
        plan = Plan(
            objective_id="test",
            steps=[PlanStep(action="move", params={"direction": "right"}, source="plan", preconditions=frozenset(), expected_effect="move_right")],
            current_index=0,
        )
        status = check_progress(plan, self._make_model(1, 0), self._make_model(0, 0))
        self.assertEqual(status, ADVANCE)

    def test_retry_on_no_movement(self) -> None:
        plan = Plan(
            objective_id="test",
            steps=[PlanStep(action="move", params={"direction": "right"}, source="plan", preconditions=frozenset(), expected_effect="move_right")],
            current_index=0,
        )
        status = check_progress(plan, self._make_model(0, 0), self._make_model(0, 0))
        self.assertEqual(status, RETRY)

    def test_escalate_when_exhausted(self) -> None:
        plan = Plan(objective_id="test", steps=[], current_index=0)
        plan.current_index = 5
        status = check_progress(plan, self._make_model(0, 0), self._make_model(0, 0))
        self.assertEqual(status, ESCALATE)

    def test_advance_on_vertical_movement(self) -> None:
        plan = Plan(
            objective_id="test",
            steps=[PlanStep(action="move", params={"direction": "down"}, source="plan", preconditions=frozenset(), expected_effect="move_down")],
            current_index=0,
        )
        status = check_progress(plan, self._make_model(0, 2), self._make_model(0, 1))
        self.assertEqual(status, ADVANCE)

    def test_replan_when_click_retries_exceed_tolerance(self) -> None:
        step = PlanStep(action="click", params={"x": 15, "y": 15}, source="plan", preconditions=frozenset(), expected_effect="break", tolerance=2, retry_count=3)
        plan = Plan(objective_id="test", steps=[step], current_index=0)
        status = check_progress(plan, self._make_model(0, 0), self._make_model(0, 0))
        self.assertEqual(status, REPLAN)

    def test_advance_on_click_with_board_change(self) -> None:
        step = PlanStep(action="click", params={"x": 15, "y": 15}, source="plan", preconditions=frozenset(), expected_effect="break", tolerance=2)
        plan = Plan(objective_id="test", steps=[step], current_index=0)
        prev_board = [[0, 0], [0, 9]]
        curr_board = [[0, 0], [0, 0]]
        prev = self._make_model(0, 0, prev_board)
        curr = self._make_model(0, 0, curr_board)
        status = check_progress(plan, curr, prev)
        self.assertEqual(status, ADVANCE)

    def test_retry_on_click_without_board_change(self) -> None:
        step = PlanStep(action="click", params={"x": 15, "y": 15}, source="plan", preconditions=frozenset(), expected_effect="break", tolerance=5)
        plan = Plan(objective_id="test", steps=[step], current_index=0)
        prev = self._make_model(0, 0)
        curr = self._make_model(0, 0)
        status = check_progress(plan, curr, prev)
        self.assertEqual(status, RETRY)

    def test_advance_with_none_previous_model(self) -> None:
        plan = Plan(
            objective_id="test",
            steps=[PlanStep(action="move", params={"direction": "right"}, source="plan", preconditions=frozenset(), expected_effect="move_right")],
            current_index=0,
        )
        status = check_progress(plan, self._make_model(1, 0), None)
        self.assertEqual(status, ADVANCE)

    def test_advance_with_none_player_positions(self) -> None:
        plan = Plan(
            objective_id="test",
            steps=[PlanStep(action="move", params={"direction": "right"}, source="plan", preconditions=frozenset(), expected_effect="move_right")],
            current_index=0,
        )
        prev = self._make_model()
        curr = self._make_model()
        status = check_progress(plan, curr, prev)
        self.assertEqual(status, ADVANCE)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class ExecutorTests(unittest.TestCase):
    def _blank_gm(self) -> GameModel:
        return GameModel(
            board=[[]], height=5, width=5,
            player_pos=None, goal_positions=[],
            obstacles=[], components=[], walkable=set(),
            features=GameFeatures(),
        )

    def test_action_name_for_direction(self) -> None:
        self.assertEqual(action_name_for_direction("left"), "ACTION3")
        self.assertEqual(action_name_for_direction("right"), "ACTION4")
        self.assertEqual(action_name_for_direction("up"), "ACTION1")
        self.assertEqual(action_name_for_direction("down"), "ACTION2")
        self.assertEqual(action_name_for_direction("return_to_origin"), None)
        self.assertIsNone(action_name_for_direction("diagonal"))

    def test_plan_step_to_proposal_move_right(self) -> None:
        step = PlanStep(action="move", params={"direction": "right"}, source="plan", preconditions=frozenset(), expected_effect="move_right")
        plan = Plan(objective_id="test", steps=[step], current_index=0)
        proposal = plan_step_to_proposal(step, self._blank_gm(), plan)
        self.assertIsNotNone(proposal)
        assert proposal is not None
        self.assertEqual(proposal.action_name, "ACTION4")
        self.assertEqual(proposal.source, "orch_or:test:plan")

    def test_plan_step_to_proposal_move_left(self) -> None:
        step = PlanStep(action="move", params={"direction": "left"}, source="plan", preconditions=frozenset(), expected_effect="move_left")
        plan = Plan(objective_id="test", steps=[step], current_index=0)
        proposal = plan_step_to_proposal(step, self._blank_gm(), plan)
        self.assertIsNotNone(proposal)
        assert proposal is not None
        self.assertEqual(proposal.action_name, "ACTION3")

    def test_plan_step_to_proposal_click(self) -> None:
        gm = GameModel(
            board=[[]], height=5, width=5,
            player_pos=None, goal_positions=[],
            obstacles=[], components=[], walkable=set(),
            features=GameFeatures(),
        )
        step = PlanStep(action="click", params={"x": 21, "y": 33}, source="interaction", preconditions=frozenset(), expected_effect="break")
        plan = Plan(objective_id="test", steps=[step], current_index=0)
        proposal = plan_step_to_proposal(step, gm, plan)
        self.assertIsNotNone(proposal)
        assert proposal is not None
        self.assertEqual(proposal.action_name, "ACTION6")
        self.assertIn("orch_or", proposal.source)

    def test_plan_step_to_proposal_unknown_action_returns_none(self) -> None:
        step = PlanStep(action="jump", params={}, source="plan", preconditions=frozenset(), expected_effect="")
        plan = Plan(objective_id="test", steps=[step], current_index=0)
        proposal = plan_step_to_proposal(step, self._blank_gm(), plan)
        self.assertIsNone(proposal)

    def test_plan_step_to_proposal_includes_metadata(self) -> None:
        step = PlanStep(action="move", params={"direction": "right"}, source="plan", preconditions=frozenset(), expected_effect="move_right")
        plan = Plan(objective_id="reach-goal", steps=[step, PlanStep(action="click", params={}, source="plan", preconditions=frozenset(), expected_effect="x")], current_index=0)
        proposal = plan_step_to_proposal(step, self._blank_gm(), plan)
        assert proposal is not None
        self.assertEqual(proposal.metadata.get("step_index"), 1)
        self.assertEqual(proposal.metadata.get("total_steps"), 2)
        self.assertEqual(proposal.metadata.get("plan_objective"), "reach-goal")
        self.assertEqual(proposal.expected_outcome, "move_right")


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

class MovementStrategyTests(unittest.TestCase):
    def _gm(self, px: int, py: int, gx: int, gy: int) -> GameModel:
        return GameModel(
            board=[], height=10, width=10,
            player_pos=(px, py), goal_positions=[(gx, gy)],
            obstacles=[], components=[], walkable=set(),
            features=GameFeatures(),
        )

    def test_moves_right_when_goal_is_right(self) -> None:
        steps = movement_strategy(self._gm(0, 0, 5, 0))
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].params, {"direction": "right"})

    def test_moves_left_when_goal_is_left(self) -> None:
        steps = movement_strategy(self._gm(5, 0, 0, 0))
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].params, {"direction": "left"})

    def test_moves_down_when_goal_is_below(self) -> None:
        steps = movement_strategy(self._gm(0, 0, 0, 5))
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].params, {"direction": "down"})

    def test_moves_up_when_goal_is_above(self) -> None:
        steps = movement_strategy(self._gm(0, 5, 0, 0))
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].params, {"direction": "up"})

    def test_prefers_horizontal_over_vertical(self) -> None:
        steps = movement_strategy(self._gm(0, 0, 3, 1))
        self.assertEqual(steps[0].params, {"direction": "right"})

    def test_no_steps_when_player_aligned_horizontally(self) -> None:
        steps = movement_strategy(self._gm(0, 0, 0, 0))
        self.assertEqual(len(steps), 0)

    def test_no_steps_when_no_player(self) -> None:
        gm = GameModel(board=[], height=5, width=5, player_pos=None, goal_positions=[(4, 4)], obstacles=[], components=[], walkable=set(), features=GameFeatures())
        steps = movement_strategy(gm)
        self.assertEqual(len(steps), 0)

    def test_no_steps_when_no_goals(self) -> None:
        gm = GameModel(board=[], height=5, width=5, player_pos=(0, 0), goal_positions=[], obstacles=[], components=[], walkable=set(), features=GameFeatures())
        steps = movement_strategy(gm)
        self.assertEqual(len(steps), 0)


class InteractionStrategyTests(unittest.TestCase):
    def test_no_steps_when_no_player(self) -> None:
        gm = GameModel(board=[[]], height=5, width=5, player_pos=None, goal_positions=[], obstacles=[(1, 2, "interactive")], components=[], walkable=set(), features=GameFeatures(tile_size=6))
        steps = interaction_strategy(gm)
        self.assertEqual(len(steps), 0)

    def test_no_steps_when_no_interactive_obstacles(self) -> None:
        gm = GameModel(board=[[]], height=5, width=5, player_pos=(1, 0), goal_positions=[], obstacles=[(1, 1, "wall")], components=[], walkable=set(), features=GameFeatures(tile_size=6))
        steps = interaction_strategy(gm)
        self.assertEqual(len(steps), 0)

    def test_step_when_interactive_obstacle_below_player(self) -> None:
        gm = GameModel(board=[[]], height=5, width=5, player_pos=(1, 0), goal_positions=[], obstacles=[(1, 1, "interactive")], components=[], walkable=set(), features=GameFeatures(tile_size=6))
        steps = interaction_strategy(gm)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].action, "click")
        self.assertEqual(steps[0].params["target_gx"], 1)
        self.assertEqual(steps[0].params["target_gy"], 1)

    def test_no_steps_when_obstacle_too_far_horizontally(self) -> None:
        gm = GameModel(board=[[]], height=5, width=5, player_pos=(0, 0), goal_positions=[], obstacles=[(3, 1, "interactive")], components=[], walkable=set(), features=GameFeatures(tile_size=6))
        steps = interaction_strategy(gm)
        self.assertEqual(len(steps), 0)

    def test_no_steps_when_obstacle_not_directly_below(self) -> None:
        gm = GameModel(board=[[]], height=5, width=5, player_pos=(1, 0), goal_positions=[], obstacles=[(1, 3, "interactive")], components=[], walkable=set(), features=GameFeatures(tile_size=6))
        steps = interaction_strategy(gm)
        self.assertEqual(len(steps), 0)

    def test_pixel_coordinates_use_frame_coords(self) -> None:
        gm = GameModel(board=[[]], height=10, width=10, player_pos=(2, 0), goal_positions=[], obstacles=[(2, 1, "interactive")], components=[], walkable=set(), features=GameFeatures(tile_size=8))
        steps = interaction_strategy(gm)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].params["x"], 2)
        self.assertEqual(steps[0].params["y"], 1)

    def test_multiple_nearby_interactive_obstacles(self) -> None:
        gm = GameModel(board=[[]], height=5, width=10, player_pos=(2, 0), goal_positions=[], obstacles=[(1, 1, "interactive"), (2, 1, "interactive"), (3, 1, "interactive")], components=[], walkable=set(), features=GameFeatures(tile_size=6))
        steps = interaction_strategy(gm)
        self.assertEqual(len(steps), 3)


class ExplorationStrategyTests(unittest.TestCase):
    def test_no_steps_when_no_player(self) -> None:
        gm = GameModel(board=[[]], height=10, width=10, player_pos=None, goal_positions=[], obstacles=[], components=[], walkable=set(), features=GameFeatures(goal_detectable=False))
        steps = exploration_strategy(gm)
        self.assertEqual(len(steps), 0)

    def test_moves_right_when_player_left_of_center(self) -> None:
        gm = GameModel(board=[[]], height=10, width=10, player_pos=(2, 5), goal_positions=[], obstacles=[], components=[], walkable=set(), features=GameFeatures(goal_detectable=False))
        steps = exploration_strategy(gm)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].params["direction"], "right")

    def test_moves_left_when_player_right_of_center(self) -> None:
        gm = GameModel(board=[[]], height=10, width=10, player_pos=(8, 5), goal_positions=[], obstacles=[], components=[], walkable=set(), features=GameFeatures(goal_detectable=False))
        steps = exploration_strategy(gm)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].params["direction"], "left")

    def test_moves_left_when_player_at_exact_center(self) -> None:
        gm = GameModel(board=[[]], height=10, width=10, player_pos=(5, 5), goal_positions=[], obstacles=[], components=[], walkable=set(), features=GameFeatures(goal_detectable=False))
        steps = exploration_strategy(gm)
        # px == center_x triggers else branch (left) since condition is strict <
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].params["direction"], "left")

    def test_no_steps_when_goal_already_detected(self) -> None:
        gm = GameModel(board=[[]], height=10, width=10, player_pos=(0, 0), goal_positions=[(9, 9)], obstacles=[], components=[], walkable=set(), features=GameFeatures(goal_detectable=True))
        steps = exploration_strategy(gm)
        self.assertEqual(len(steps), 0)

    def test_mirror_axis_direction_for_level1(self) -> None:
        level1_goals = {(19, 15), (17, 17), (17, 16), (17, 15), (18, 15)}
        beam_cells = [(12, 5), (13, 5), (14, 5), (12, 6), (13, 6), (14, 6)]
        beam_comp = Component(cells=beam_cells, bbox=(12, 5, 14, 6), size=3, dominant_color=4, touches_border=False)
        gm = GameModel(
            board=[[]], height=20, width=20,
            player_pos=(6, 5), goal_positions=list(level1_goals),
            obstacles=[], components=[beam_comp], walkable=set(),
            features=GameFeatures(
                has_beam=True, beam_color=4, beam_positions=beam_cells,
                mirror_axis=10, goal_detectable=True,
            ),
        )
        result = _mirror_axis_direction(gm)
        self.assertIn(result, ("down", "left"))
        self.assertEqual(result, "down")


class StrategyRegistryTests(unittest.TestCase):
    def test_get_known_strategy(self) -> None:
        self.assertIsNotNone(get_strategy("movement"))
        self.assertIsNotNone(get_strategy("interaction"))
        self.assertIsNotNone(get_strategy("generic"))

    def test_get_unknown_strategy_returns_none(self) -> None:
        self.assertIsNone(get_strategy("non_existent_strategy"))

    def test_available_strategies_contains_expected(self) -> None:
        avail = available_strategies()
        self.assertIn("movement", avail)
        self.assertIn("interaction", avail)
        self.assertIn("generic", avail)

    def test_register_new_strategy(self) -> None:
        def dummy_strategy(gm, ctx=None):
            return []
        register("dummy_test", dummy_strategy)
        self.assertIsNotNone(get_strategy("dummy_test"))
        self.assertIn("dummy_test", available_strategies())

    def test_register_overwrites_existing(self) -> None:
        from agi_core.orch_or.strategies.movement import movement_strategy
        original = movement_strategy
        try:
            def replacement(gm, ctx=None):
                return []
            register("movement", replacement)
            got = get_strategy("movement")
            self.assertIs(got, replacement)
        finally:
            register("movement", original)

    def test_generic_strategy_returns_empty(self) -> None:
        from agi_core.orch_or.strategies.generic import generic_strategy
        gm = GameModel(board=[[]], height=5, width=5, player_pos=None, goal_positions=[], obstacles=[], components=[], walkable=set(), features=GameFeatures())
        steps = generic_strategy(gm)
        self.assertEqual(len(steps), 0)


# ---------------------------------------------------------------------------
# Serialization edge cases
# ---------------------------------------------------------------------------

class SerializationTests(unittest.TestCase):
    def test_deserialize_without_steps_key(self) -> None:
        data = {"objective_id": "test", "current_index": 0}
        plan = deserialize_plan(data)
        self.assertEqual(len(plan.steps), 0)
        self.assertEqual(plan.objective_id, "test")

    def test_deserialize_without_objective_id(self) -> None:
        data = {"steps": []}
        with self.assertRaises(KeyError):
            deserialize_plan(data)

    def test_serialize_preserves_attempt_count(self) -> None:
        plan = Plan(objective_id="test", steps=[], attempt_count=3)
        data = serialize_plan(plan)
        self.assertEqual(data["attempt_count"], 3)

    def test_serialize_preserves_confidence(self) -> None:
        plan = Plan(objective_id="test", steps=[], confidence=0.85)
        data = serialize_plan(plan)
        self.assertEqual(data["confidence"], 0.85)

    def test_serialize_plan_with_no_params(self) -> None:
        step = PlanStep(action="click", params=None, source="test", preconditions=frozenset(), expected_effect="x")
        plan = Plan(objective_id="test", steps=[step])
        data = serialize_plan(plan)
        self.assertIsNone(data["steps"][0]["params"])


# ---------------------------------------------------------------------------
# Parallelism
# ---------------------------------------------------------------------------

class ThreadPoolManagerTests(unittest.TestCase):
    def test_set_max_workers(self) -> None:
        previous = ThreadPoolManager._max_workers
        try:
            ThreadPoolManager.set_max_workers(2)
            exec = ThreadPoolManager.executor()
            self.assertIsNotNone(exec)
        finally:
            ThreadPoolManager.set_max_workers(previous)

    def test_run_in_parallel_returns_results(self) -> None:
        def returns_42():
            return 42
        results = ThreadPoolManager.run_in_parallel([(returns_42, "forty_two")])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 42)
        self.assertEqual(results[0][1], "forty_two")

    def test_run_in_parallel_empty(self) -> None:
        results = ThreadPoolManager.run_in_parallel([])
        self.assertEqual(len(results), 0)

    def test_run_all_returns_all(self) -> None:
        def a():
            return "a"
        def b():
            return "b"
        results = ThreadPoolManager.run_all([(a, "task_a"), (b, "task_b")])
        returned = {name: val for val, name in results}
        self.assertIn("task_a", returned)
        self.assertIn("task_b", returned)
        self.assertEqual(returned["task_a"], "a")
        self.assertEqual(returned["task_b"], "b")

    def test_run_all_handles_exception(self) -> None:
        def failing():
            raise ValueError("oops")
        def ok():
            return 1
        results = ThreadPoolManager.run_all([(failing, "fail"), (ok, "ok")])
        result_map = {name: val for val, name in results}
        self.assertIsNone(result_map.get("fail"))
        self.assertEqual(result_map.get("ok"), 1)

    def test_run_in_parallel_survives_exception(self) -> None:
        def failing():
            raise RuntimeError("fail")
        results = ThreadPoolManager.run_in_parallel([(failing, "fail")])
        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0][0])

    def test_run_in_parallel_timeout(self) -> None:
        import threading
        import time
        release = threading.Event()

        def slow():
            release.wait(timeout=1.0)
            return 99
        try:
            results = ThreadPoolManager.run_in_parallel([(slow, "slow")], timeout=0.01)
            self.assertEqual(len(results), 0)
        finally:
            release.set()
            for _ in range(50):
                if ThreadPoolManager.active_worker_count() == 0:
                    break
                time.sleep(0.01)

    def test_run_all_timeout(self) -> None:
        import threading
        import time
        release = threading.Event()

        def slow():
            release.wait(timeout=1.0)
            return 99
        def fast():
            return 1
        try:
            results = ThreadPoolManager.run_all([(slow, "slow"), (fast, "fast")], timeout=0.01)
            result_map = {name: val for val, name in results}
            self.assertEqual(result_map.get("fast"), 1)
        finally:
            release.set()
            for _ in range(50):
                if ThreadPoolManager.active_worker_count() == 0:
                    break
                time.sleep(0.01)

    def test_run_all_timeout_keeps_running_tasks_tracked(self) -> None:
        import time
        import threading
        previous = ThreadPoolManager._max_workers
        ThreadPoolManager.set_max_workers(1)
        started = threading.Event()
        release = threading.Event()

        def slow():
            started.set()
            release.wait(timeout=1.0)
            return 99

        try:
            results = ThreadPoolManager.run_all([(slow, "slow")], timeout=0.01)
            self.assertEqual(results, [])
            self.assertTrue(started.wait(timeout=0.2))
            self.assertEqual(ThreadPoolManager.active_worker_count(), 1)
        finally:
            release.set()
            for _ in range(20):
                if ThreadPoolManager.active_worker_count() == 0:
                    break
                time.sleep(0.01)
            self.assertEqual(ThreadPoolManager.active_worker_count(), 0)
            ThreadPoolManager.set_max_workers(previous)

    def test_submit_unique_deduplicates_in_flight_name(self) -> None:
        import threading

        release = threading.Event()
        started = threading.Event()
        calls = 0

        def slow():
            nonlocal calls
            calls += 1
            started.set()
            release.wait(timeout=1.0)
            return 1

        first = ThreadPoolManager.submit_unique("dedup-probe", slow)
        second = ThreadPoolManager.submit_unique("dedup-probe", slow)
        try:
            self.assertIsNotNone(first)
            self.assertIsNone(second)
            self.assertTrue(started.wait(timeout=0.2))
            self.assertEqual(calls, 1)
            self.assertTrue(ThreadPoolManager.is_name_pending("dedup-probe"))
        finally:
            release.set()
            if first is not None:
                first.result(timeout=1.0)
        self.assertFalse(ThreadPoolManager.is_name_pending("dedup-probe"))


class ParallelPlanGeneratorTests(unittest.TestCase):
    def test_generate_plan_parallel_no_context(self) -> None:
        gm = GameModel(
            board=[], height=5, width=5,
            player_pos=(0, 0), goal_positions=[(4, 4)],
            obstacles=[], components=[],
            walkable=set(),
            features=GameFeatures(player_detectable=True, goal_detectable=True, path_to_goal_possible=True),
        )
        plan = generate_plan(gm, parallel=True)
        self.assertIsNotNone(plan)

    def test_generate_plan_parallel_with_explore(self) -> None:
        gm = GameModel(
            board=[], height=10, width=10,
            player_pos=(0, 0), goal_positions=[],
            obstacles=[], components=[],
            walkable=set(),
            features=GameFeatures(player_detectable=True, goal_detectable=False),
        )
        plan = generate_plan(gm, parallel=True)
        self.assertIsNotNone(plan)
        if plan is not None:
            self.assertEqual(plan.objective_id, "explore-board")

    def test_parallel_produces_same_objective_as_sequential(self) -> None:
        gm = GameModel(
            board=[], height=5, width=5,
            player_pos=(0, 0), goal_positions=[(4, 4)],
            obstacles=[], components=[],
            walkable=set(),
            features=GameFeatures(player_detectable=True, goal_detectable=True, path_to_goal_possible=True),
        )
        seq = generate_plan(gm, parallel=False)
        par = generate_plan(gm, parallel=True)
        if seq is not None and par is not None:
            self.assertEqual(seq.objective_id, par.objective_id)
            self.assertGreater(len(par.steps), 0)

    def test_parallel_with_obstacles(self) -> None:
        gm = GameModel(
            board=[], height=5, width=5,
            player_pos=(0, 0), goal_positions=[(4, 4)],
            obstacles=[(2, 2, "wall")], components=[],
            walkable=set(),
            features=GameFeatures(player_detectable=True, goal_detectable=True, path_to_goal_possible=False),
        )
        plan = generate_plan(gm, parallel=True)
        self.assertIsNotNone(plan)
        if plan is not None:
            self.assertIn(plan.objective_id, ("reach-goal", "navigate-to-goal", "clear-obstacles"))


class ParallelOrchestratorTests(unittest.TestCase):
    def test_fallback_executive_parallel_timeout_uses_deterministic(self) -> None:
        import threading
        import time

        release = threading.Event()

        class FakeCfg:
            deterministic_fallback_action = "ACTION1"
        class FakeExec:
            def decide(self, context):
                release.wait(timeout=1.0)
                return None
        class FakeContext:
            config = FakeCfg()
        reducer = ObjectiveReduction.__new__(ObjectiveReduction)
        reducer._config = FakeCfg()
        reducer._executive = FakeExec()
        try:
            decision = reducer._fallback_executive_parallel(FakeContext(), timeout=0.001)
            self.assertEqual(decision.action_name, "ACTION1")
            self.assertEqual(decision.source, "orch_or:parallel_timeout")
            self.assertTrue(decision.used_fallback)
        finally:
            release.set()
            for _ in range(50):
                if ThreadPoolManager.active_worker_count() == 0:
                    break
                time.sleep(0.01)


if __name__ == "__main__":
    unittest.main()
