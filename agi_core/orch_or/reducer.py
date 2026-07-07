from __future__ import annotations

from agi_core.orch_or.game_model import GameModel, Objective


def _root_obj() -> Objective:
    return Objective(description="complete-level", priority=1.0)

def _reach_goal() -> Objective:
    return Objective(description="reach-goal", priority=0.95)

def _navigate() -> Objective:
    return Objective(description="navigate-to-goal", priority=0.90)

def _seek_goal() -> Objective:
    return Objective(description="seek-goal", priority=0.92)

def _clear_obstacles() -> Objective:
    return Objective(description="clear-obstacles", priority=0.85)

def _explore() -> Objective:
    return Objective(description="explore-board", priority=0.50)


def _goals_are_dynamic(game_model: GameModel, goal_history: list[list[tuple[int, int]]]) -> bool:
    if not goal_history or len(goal_history) < 2:
        return False
    latest = set(game_model.goal_positions)
    for prev in goal_history[-3:]:
        prev_set = set(prev)
        if prev_set and prev_set != latest:
            return True
    return False


def reduce(
    game_model: GameModel,
    root: Objective | None = None,
    goal_hints: list[dict] | None = None,
    goal_history: list[list[tuple[int, int]]] | None = None,
) -> Objective:
    if root is None:
        root = _root_obj()
    features = game_model.features
    if features.player_detectable and features.goal_detectable:
        # Apply hyperon goal hints to adjust sub-objective priorities.
        _hint_boost_clear: float = 0.0
        if goal_hints:
            for hint in goal_hints:
                hinted_subgoal = str(hint.get("subgoal", "") or "")
                hint_conf = float(hint.get("confidence", 0.0))
                if hinted_subgoal == "clear-obstacles" and hint_conf > 0.5:
                    _hint_boost_clear = 0.07

        dynamic = goal_history is not None and _goals_are_dynamic(game_model, goal_history)
        def nav_obj():
            return _seek_goal() if dynamic else _navigate()

        if features.has_beam:
            if features.path_to_goal_possible:
                reach = _reach_goal()
                reach.sub_objectives = [nav_obj()]
                root.sub_objectives = [reach]
                return root
            reach = _reach_goal()
            reach.sub_objectives = [_explore()]
            root.sub_objectives = [reach]
            return root
        if features.board_type == "puzzle":
            reach = _reach_goal()
            has_interactive = any(otype == "interactive" for _, _, otype in game_model.obstacles)
            if features.path_to_goal_possible:
                obj = nav_obj()
                if has_interactive:
                    clear_obj = _clear_obstacles()
                    obj.sub_objectives = [clear_obj]
                reach.sub_objectives = [obj]
            elif has_interactive:
                reach.sub_objectives = [_clear_obstacles(), _explore()]
            else:
                reach.sub_objectives = [_explore()]
            root.sub_objectives = [reach]
            return root
        if features.path_to_goal_possible:
            reach = _reach_goal()
            reach.sub_objectives = [nav_obj()]
            root.sub_objectives = [reach]
            return root
        has_interactive = any(otype == "interactive" for _, _, otype in game_model.obstacles)
        if has_interactive:
            reach = _reach_goal()
            clear_obj = _clear_obstacles()
            clear_obj.priority = min(clear_obj.priority + _hint_boost_clear, 0.95)
            reach.sub_objectives = [clear_obj]
            root.sub_objectives = [reach]
            return root
        reach = _reach_goal()
        reach.sub_objectives = [nav_obj()]
        root.sub_objectives = [reach]
        return root
    if features.player_detectable and not features.goal_detectable:
        root.sub_objectives = [_explore()]
        return root
    if not features.player_detectable:
        root.sub_objectives = [_explore()]
        return root
    root.sub_objectives = [_explore()]
    return root


def select_leaf(objective: Objective) -> Objective:
    if not objective.sub_objectives:
        return objective
    best = max(objective.sub_objectives, key=lambda o: o.priority)
    return select_leaf(best)
