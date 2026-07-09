from __future__ import annotations

import logging
from typing import Any

from agi_core.knowledge.solvers import DynamicsSolver, SolverContext, register_solver
from agi_core.types import ActionDecision

logger = logging.getLogger(__name__)


class ObserveSolver(DynamicsSolver):
    """Solver for cellular-automata and pattern-transform boards.

    Records board state snapshots, detects evolution patterns,
    and plans actions to exploit discovered rules.
    """

    dynamics = "observe"

    def __init__(self, memory_db_path=None):
        super().__init__(memory_db_path)
        self._board_history: list[list[list[int]]] = []
        self._probe_count: int = 0
        self._max_probes: int = 10
        self.reset_level()

    def reset_level(self):
        self._board_history.clear()
        self._probe_count = 0

    def should_activate(self, ctx: SolverContext) -> bool:
        gm = ctx.game_model
        return (gm.features.primary_dynamics
                in ("cellular-automata", "pattern-transform"))

    def decide(self, ctx: SolverContext) -> ActionDecision | None:
        gm = ctx.game_model
        self._board_history.append([row[:] for row in gm.board])

        if len(self._board_history) < 3:
            return None  # need more samples

        # Detect board evolution without player action
        no_input_change = self._detect_untriggered_change()
        if no_input_change:
            # Cellular automata — just wait and observe
            return None

        # Pattern transform — probe with ACTION6 to discover rules
        if self._probe_count < self._max_probes:
            self._probe_count += 1
            probe_x, probe_y = self._pick_probe_point(gm)
            return ActionDecision(
                action_name="ACTION6",
                source="solver:observe",
                confidence=0.45,
                reason="probe_pattern",
                metadata={"x": probe_x, "y": probe_y},
            )

        return None

    def _detect_untriggered_change(self) -> bool:
        if len(self._board_history) < 2:
            return False
        return self._board_history[-1] != self._board_history[-2]

    def _pick_probe_point(self, gm) -> tuple[int, int]:
        import random
        if gm.goal_positions:
            return random.choice(gm.goal_positions)
        if gm.player_pos:
            return (gm.player_pos[0] + random.randint(-3, 3),
                    gm.player_pos[1] + random.randint(-3, 3))
        return (32, 32)


register_solver("cellular-automata", ObserveSolver)
register_solver("pattern-transform", ObserveSolver)
