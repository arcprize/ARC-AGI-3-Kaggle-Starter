from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agi_core.types import ActionDecision

if TYPE_CHECKING:
    from agi_core.orch_or.game_model import GameModel
    from agi_core.types import ExecutiveContext


@dataclass
class SolverContext:
    game_model: GameModel
    plan_context: dict[str, Any]
    episode: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)


class DynamicsSolver(ABC):
    """Base class for all dynamics-specific solvers."""

    dynamics: str = "unknown"

    def __init__(self, memory_db_path: str | Path | None = None) -> None:
        self._memory_db_path = Path(memory_db_path) if memory_db_path else None

    @abstractmethod
    def should_activate(self, ctx: SolverContext) -> bool:
        """Return True if this solver can handle the current board."""
        ...

    @abstractmethod
    def decide(self, ctx: SolverContext) -> ActionDecision | None:
        """Produce an action decision, or None to fall through."""
        ...

    def learn(self, ctx: SolverContext, outcome: str) -> None:
        """Called after the solver's decision produces an outcome."""

    def reset_level(self) -> None:
        """Reset per-level state (called on level transitions)."""


# Solver registry: maps dynamics category string to solver class.
# Solvers are registered here and imported lazily to avoid circular imports.
_SOLVER_REGISTRY: dict[str, type[DynamicsSolver]] = {}


def register_solver(dynamics: str, solver_cls: type[DynamicsSolver]) -> None:
    _SOLVER_REGISTRY[dynamics] = solver_cls


def get_solver(dynamics: str) -> type[DynamicsSolver] | None:
    return _SOLVER_REGISTRY.get(dynamics)


def get_all_solvers() -> dict[str, type[DynamicsSolver]]:
    return dict(_SOLVER_REGISTRY)


# Import all solver modules to trigger registration via register_solver().
# Each solver's module-level register_solver call populates _SOLVER_REGISTRY.
from agi_core.knowledge.solvers import (
    noop_solver,
    navigation_solver,
    multi_agent_solver,
    sliding_solver,
    gravity_solver,
    push_pull_solver,
    signal_solver,
    observe_solver,
    inventory_solver,
    beam_solver,
    mirror_solver,
    hybrid_solver,
)
