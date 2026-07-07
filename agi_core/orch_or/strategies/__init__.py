from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from agi_core.orch_or.strategies.exploration import exploration_strategy
from agi_core.orch_or.strategies.goal_seeker import goal_seeker_strategy
from agi_core.orch_or.strategies.generic import generic_strategy
from agi_core.orch_or.strategies.interaction import interaction_strategy
from agi_core.orch_or.strategies.movement import movement_strategy

if TYPE_CHECKING:
    from agi_core.orch_or.game_model import GameModel, PlanStep

StrategyFn = Callable[..., list["PlanStep"]]


_registry: dict[str, StrategyFn] = {
    "movement": movement_strategy,
    "interaction": interaction_strategy,
    "exploration": exploration_strategy,
    "goal_seeker": goal_seeker_strategy,
    "generic": generic_strategy,
}


def register(name: str, fn: StrategyFn) -> None:
    _registry[name] = fn


def get_strategy(name: str) -> StrategyFn | None:
    return _registry.get(name)


def available_strategies() -> list[str]:
    return list(_registry.keys())
