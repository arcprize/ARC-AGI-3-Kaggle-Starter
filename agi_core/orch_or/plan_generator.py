from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agi_core.orch_or.game_model import Plan, PlanStep
from agi_core.orch_or.parallel import ThreadPoolManager
from agi_core.orch_or.reducer import reduce, select_leaf
from agi_core.orch_or.strategies import available_strategies, get_strategy

if TYPE_CHECKING:
    from agi_core.orch_or.game_model import GameModel, Objective, PlanStep

_PARALLEL_TARGETS = frozenset({"reach-goal", "navigate-to-goal", "seek-goal", "clear-obstacles"})

_STRATEGY_MAP: dict[str, str] = {
    "reach-goal": "movement",
    "navigate-to-goal": "movement",
    "seek-goal": "goal_seeker",
    "clear-obstacles": "interaction",
    "explore-board": "exploration",
    "complete-level": "generic",
}


def _apply_priority_boost(root: Objective, subgoal: str) -> None:
    """Boost priority of a sub-objective that matches a hyperon goal-hint."""
    stack: list[Objective] = [root]
    while stack:
        obj = stack.pop()
        if obj.description == subgoal:
            obj.priority = min(obj.priority + 0.15, 1.0)
            return
        stack.extend(obj.sub_objectives)


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
    goal_history: list[list[tuple[int, int]]] | None = None
    if context and "goal_history" in context:
        goal_history = context["goal_history"]
    objective_tree = reduce(game_model, goal_hints=goal_hints, goal_history=goal_history)
    # Apply hyperon goal-hint priority overrides after building the tree.
    if goal_hints:
        for hint in goal_hints:
            hinted_subgoal = str(hint.get("subgoal", "") or "")
            hint_conf = float(hint.get("confidence", 0.0))
            if hint_conf >= 0.7:
                _apply_priority_boost(objective_tree, hinted_subgoal)
    leaf = select_leaf(objective_tree)
    primary_name = _STRATEGY_MAP.get(leaf.description, "generic")
    matched: list | None = None

    # Check chunk library for cached action sequences
    if context and context.get("chunk_library"):
        cl = context["chunk_library"]
        recent = context.get("recent_actions", [])
        matched = cl.match(
            dynamics=leaf.description,
            goal_count=context.get("goal_count", 0),
            action_window=recent[-8:] if recent else [],
        )
        if matched:
            best = matched[0]
            steps = [PlanStep(action=a, params={}, source="chunk",
                              preconditions=frozenset(), expected_effect=a)
                     for a in best.actions]
            if steps:
                return Plan(objective_id=leaf.description, steps=steps,
                            created_at_step=context.get("game_step", 0),
                            confidence=min(0.9, best.reliability * 0.8))

    # Fall back to episodic memory traces if no chunk matched
    if not matched and context and context.get("episodic_memory"):
        em = context["episodic_memory"]
        if hasattr(em, 'analogical_retrieve'):
            recent = context.get("recent_actions", [])
            traces = em.analogical_retrieve(
                dynamics=leaf.description,
                action_sequence=recent[-5:] if recent else [],
                top_k=1,
            )
            if traces:
                best = traces[0]
                typical = best.schema.get("typical_actions", [])
                if typical:
                    steps = [PlanStep(action=a, params={}, source="episodic",
                                      preconditions=frozenset(), expected_effect=a)
                             for a in typical]
                    if steps:
                        return Plan(objective_id=leaf.description, steps=steps,
                                    created_at_step=context.get("game_step", 0),
                                    confidence=min(0.7, best.strength * 0.6))

    if parallel and leaf.description in _PARALLEL_TARGETS:
        return _generate_plan_parallel(leaf, game_model, context, primary_name, timeout)
    strategy_fn = get_strategy(primary_name)
    if strategy_fn is None:
        strategy_fn = get_strategy("generic")
    if strategy_fn is None:
        return None
    steps = strategy_fn(game_model, context)
    if not steps:
        return None
    # Override step tolerance from board-retry-tolerance MeTTa rule.
    try:
        retry_rows = (context or {}).get("board_retry_tolerance", [])
        if retry_rows:
            tol = int(retry_rows[0].get("tolerance", 3))
            for s in steps:
                s.tolerance = tol
    except Exception:
        pass
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
    # Run ALL strategies in parallel (puzzle is handled by solver system).
    targets = [s for s in all_strategies if s != "puzzle"]
    tasks = [
        (lambda fn=get_strategy(s): fn(game_model, context) if fn else [], s)
        for s in targets
    ]
    raw_results = ThreadPoolManager.run_all(
        tasks,
        timeout=timeout or 1.0,
        deduplicate_names=True,
    )
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
    return Plan(
        objective_id=leaf.description,
        steps=all_steps,
        created_at_step=context.get("game_step", 0) if context else 0,
        confidence=0.6,
    )




