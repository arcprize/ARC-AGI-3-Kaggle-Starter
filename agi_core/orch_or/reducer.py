from __future__ import annotations

from agi_core.orch_or.game_model import GameModel, Objective


def _root_obj() -> Objective:
    return Objective(description="complete-level", priority=1.0)

def _reach_goal() -> Objective:
    return Objective(description="reach-goal", priority=0.95)

def _navigate() -> Objective:
    return Objective(description="navigate-to-goal", priority=0.90)

def _clear_obstacles() -> Objective:
    return Objective(description="clear-obstacles", priority=0.85)

def _explore() -> Objective:
    return Objective(description="explore-board", priority=0.50)



def reduce(
    game_model: GameModel,
    root: Objective | None = None,
    goal_hints: list[dict] | None = None,
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

        if features.has_beam:
            # Beam exists - use explore-board for beam navigation
            # Exception: platformer with direct path can just navigate
            if features.path_to_goal_possible and features.board_type == "platformer":
                reach = _reach_goal()
                nav = _navigate()
                reach.sub_objectives = [nav]
                root.sub_objectives = [reach]
                return root
            reach = _reach_goal()
            reach.sub_objectives = [_explore()]
            root.sub_objectives = [reach]
            return root
        # Puzzle boards need exploration/puzzle strategy for switch/click
        if features.board_type == "puzzle":
            reach = _reach_goal()
            reach.sub_objectives = [_explore()]
            root.sub_objectives = [reach]
            return root
        if features.path_to_goal_possible:
            reach = _reach_goal()
            nav = _navigate()
            reach.sub_objectives = [nav]
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
        # No beam, no path, no interactive: fallback to navigate
        reach = _reach_goal()
        nav = _navigate()
        reach.sub_objectives = [nav]
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
