from __future__ import annotations

import logging
from typing import Any

from agi_core.knowledge.solvers import DynamicsSolver, SolverContext, register_solver
from agi_core.orch_or.game_model import DIR_TO_ACTION
from agi_core.types import ActionDecision

logger = logging.getLogger(__name__)


class SignalSolver(DynamicsSolver):
    """Solver for signal/circuit boards — switch-based logic puzzles.

    Identifies interactive cells (ACTION6 targets), maps switch-to-effect
    relationships, and plans activation sequences.
    """

    dynamics = "signal"

    def __init__(self, memory_db_path=None):
        super().__init__(memory_db_path)
        self._tried_interactions: set[tuple[int, int]] = set()
        self._last_board: list[list[int]] | None = None
        self._switch_effects: dict[tuple[int, int], list[tuple[int, int]]] = {}
        self.reset_level()

    def reset_level(self):
        self._tried_interactions.clear()
        self._last_board = None
        self._switch_effects.clear()

    def should_activate(self, ctx: SolverContext) -> bool:
        gm = ctx.game_model
        return gm.player_pos is not None and self._count_interactive(gm) >= 2

    def decide(self, ctx: SolverContext) -> ActionDecision | None:
        gm = ctx.game_model
        px, py = gm.player_pos

        # Detect board changes from last frame to map switch effects
        if self._last_board is not None:
            changed = self._board_diff(self._last_board, gm.board)
            if changed:
                for sx, sy in self._tried_interactions:
                    if (sx, sy) not in self._switch_effects:
                        nearby = [c for c in changed if abs(c[0]-sx)+abs(c[1]-sy) <= 3]
                        if nearby:
                            self._switch_effects[(sx, sy)] = nearby
        self._last_board = [row[:] for row in gm.board]

        # If we have mapped switches, try the one that opens paths toward goals
        if gm.features.goal_detectable and gm.goal_positions:
            target = min(gm.goal_positions, key=lambda g: abs(g[0]-px)+abs(g[1]-py))
            for switch, effects in self._switch_effects.items():
                if any(abs(e[0]-target[0])+abs(e[1]-target[1]) <= 6 for e in effects):
                    # This switch affects the goal area — press it
                    return ActionDecision(
                        action_name="ACTION6",
                        source="solver:signal",
                        confidence=0.60,
                        reason="activate_switch",
                        metadata={"x": switch[0], "y": switch[1]},
                    )

        # Try unexplored interactive cells
        for c in gm.components:
            if c.dominant_color == 0 or c.dominant_color == gm.features.player_color:
                continue
            for cx, cy in c.cells:
                if (cx, cy) not in self._tried_interactions:
                    self._tried_interactions.add((cx, cy))
                    return ActionDecision(
                        action_name="ACTION6",
                        source="solver:signal",
                        confidence=0.50,
                        reason="probe_switch",
                        metadata={"x": cx, "y": cy},
                    )
        return None

    def _count_interactive(self, gm) -> int:
        bg = gm.features.background_color
        return sum(1 for c in gm.components
                   if not c.touches_border
                   and c.dominant_color not in (bg, -1, gm.features.player_color)
                   and c.size <= 20)

    def _board_diff(self, old, new):
        diffs = []
        for y in range(min(len(old), len(new))):
            for x in range(min(len(old[y]), len(new[y]))):
                if old[y][x] != new[y][x]:
                    diffs.append((x, y))
        return diffs


register_solver("signal", SignalSolver)
