from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agi_core.orch_or.game_model import Plan
from agi_core.orch_or.parallel import ThreadPoolManager
from agi_core.orch_or.reducer import reduce, select_leaf
from agi_core.orch_or.strategies import available_strategies, get_strategy

if TYPE_CHECKING:
    from agi_core.orch_or.game_model import GameModel, Objective, PlanStep

_PARALLEL_TARGETS = frozenset({"reach-goal", "navigate-to-goal", "clear-obstacles"})

_STRATEGY_MAP: dict[str, str] = {
    "reach-goal": "movement",
    "navigate-to-goal": "movement",
    "clear-obstacles": "interaction",
    "explore-board": "exploration",
    "complete-level": "generic",
}

_PUZZLE_OBJECTIVES = frozenset({"reach-goal", "navigate-to-goal", "explore-board"})


def generate_plan(
    game_model: GameModel,
    context: dict[str, Any] | None = None,
    *,
    parallel: bool = False,
    timeout: float | None = None,
) -> Plan | None:
    goal_hints: list[dict] | None = None
    if context and "goal_hints" in context:
        goal_hints = context["goal_hints"]
    objective_tree = reduce(game_model, goal_hints=goal_hints)
    leaf = select_leaf(objective_tree)
    primary_name = _STRATEGY_MAP.get(leaf.description, "generic")
    if parallel and leaf.description in _PARALLEL_TARGETS:
        return _generate_plan_parallel(leaf, game_model, context, primary_name, timeout)
    strategy_fn = get_strategy(primary_name)
    if strategy_fn is None:
        strategy_fn = get_strategy("generic")
    if strategy_fn is None:
        return None
    puzzle_fn = get_strategy("puzzle")
    if game_model.features.board_type == "puzzle" and puzzle_fn is not None:
        # For puzzle boards, try the puzzle strategy first so we can switch to
        # the correct controllable entity before exploring.
        steps = puzzle_fn(game_model, context)
        if not steps:
            steps = strategy_fn(game_model, context)
    else:
        steps = strategy_fn(game_model, context)
        if not steps and leaf.description in _PUZZLE_OBJECTIVES:
            if puzzle_fn is not None:
                steps = puzzle_fn(game_model, context)
    if not steps:
        return None
    # Adjust plan confidence from hyperon action priors.
    _DIR_TO_ACTION = {"up": "ACTION1", "down": "ACTION2", "left": "ACTION3", "right": "ACTION4"}
    action_priors = (context or {}).get("action_priors", {})
    conf_boost = 0.0
    conf_penalty = 0.0
    for step in steps:
        action_name = None
        if step.action == "switch":
            action_name = "ACTION5"
        elif step.action == "click":
            action_name = "ACTION6"
        elif step.action == "move" and step.params and "direction" in step.params:
            action_name = _DIR_TO_ACTION.get(step.params["direction"])
        if action_name and action_name in action_priors:
            prior = action_priors[action_name]
            conf_boost = max(conf_boost, prior)
            if prior < 0.15:
                conf_penalty = max(conf_penalty, 0.15 - prior)
    confidence = min(0.5 + conf_boost * 0.3 - conf_penalty * 0.5, 0.8)
    return Plan(
        objective_id=leaf.description,
        steps=steps,
        created_at_step=context.get("game_step", 0) if context else 0,
        confidence=confidence,
    )


def _generate_plan_parallel(
    leaf: Objective,
    game_model: GameModel,
    context: dict[str, Any] | None,
    primary_name: str,
    timeout: float | None,
) -> Plan | None:
    all_strategies = available_strategies()
    targets = [primary_name]
    if primary_name == "movement":
        targets.append("interaction")
    elif primary_name == "interaction":
        targets.append("movement")
    targets = [s for s in all_strategies if s in targets or s in ("exploration", "puzzle", "generic")]
    targets = list(dict.fromkeys(targets))
    tasks = [
        (lambda fn=get_strategy(s): fn(game_model, context) if fn else [], s)
        for s in targets
    ]
    raw_results = ThreadPoolManager.run_all(tasks, timeout=timeout)
    all_steps: list[PlanStep] = []
    seen_actions: set[str] = set()
    for steps, name in raw_results:
        if not steps:
            continue
        for step in steps:
            action_key = f"{step.action}:{step.params}"
            if action_key not in seen_actions:
                seen_actions.add(action_key)
                all_steps.append(step)
    if not all_steps:
        return None
    switch_order = {"switch": 0, "click": 1, "move": 2, "wait": 3}
    all_steps.sort(key=lambda s: (switch_order.get(s.action, 99), s.params.get("direction", "") if s.params else ""))
    return Plan(
        objective_id=leaf.description,
        steps=all_steps,
        created_at_step=context.get("game_step", 0) if context else 0,
        confidence=0.6,
    )




