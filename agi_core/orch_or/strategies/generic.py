from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agi_core.orch_or.game_model import GameModel, PlanStep


def generic_strategy(game_model: GameModel, context: dict | None = None) -> list[PlanStep]:
    return []
