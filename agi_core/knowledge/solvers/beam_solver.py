from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agi_core.knowledge.beam_solver import solve_beam_puzzle
from agi_core.knowledge.puzzle_memory import (
    PuzzleBoardState,
    initialize_puzzle_schema,
    save_puzzle_solution,
    store_board_state,
)
from agi_core.knowledge.solvers import DynamicsSolver, SolverContext, register_solver
from agi_core.orch_or.game_model import DIR_TO_ACTION
from agi_core.orch_or.search import astar
from agi_core.types import ActionDecision

logger = logging.getLogger(__name__)


class BeamSolver(DynamicsSolver):
    """Solver for beam/light-propagation boards.

    Detects beam entities, computes mirror targets using the beam solver,
    navigates the mirror entity, and switches entities when positioned.
    Refactored from PuzzleSystem._handle_beam_controller().
    """

    dynamics = "beam"

    def __init__(self, memory_db_path: str | Path | None = None) -> None:
        super().__init__(memory_db_path)
        self._mirror_frames: int = 0
        self._beam_cache: dict[str, Any] | None = None
        self._beam_cache_frame: int = -1
        self._beam_target_x: int | None = None
        self._beam_target_y: int | None = None
        self._beam_target_alt_idx: int = 0
        self._switch_back_issued: bool = False
        self._beam_nav_oscillation: int = 0
        self._beam_last_dir: str = ""
        self._last_action: str = ""
        self._last_step: int = -1
        self._episode_steps: int = 0
        self._last_move_target: tuple[int, int] | None = None
        self._impassable_cells: set[tuple[int, int]] = set()
        self._expanded_walkable_cache: set[tuple[int, int]] | None = None
        self._cache_game_id: str = ""
        if self._memory_db_path:
            self._memory_db_path.parent.mkdir(parents=True, exist_ok=True)
            self.reset_level()
            initialize_puzzle_schema(self._memory_db_path)

    def reset_level(self) -> None:
        self._mirror_frames = 0
        self._beam_cache = None
        self._beam_cache_frame = -1
        self._beam_target_x = None
        self._beam_target_y = None
        self._beam_target_alt_idx = 0
        self._switch_back_issued = False
        self._beam_nav_oscillation = 0
        self._beam_last_dir = ""
        self._last_action = ""
        self._last_step = -1
        self._last_move_target = None
        self._impassable_cells.clear()
        self._expanded_walkable_cache = None

    def should_activate(self, ctx: SolverContext) -> bool:
        gm = ctx.game_model
        if not gm.features.has_beam:
            return False
        if gm.player_pos is None:
            return False
        # Check if player IS a beam controller (near beam cells)
        beam_color = gm.features.beam_color
        if beam_color < 0:
            return False
        px, py = gm.player_pos
        non_beam_set = ctx.plan_context.get("non_beam_controllers", set())
        if gm.features.player_color in non_beam_set:
            return False
        beam_cells = self._get_beam_cells(ctx, gm)
        if not beam_cells:
            return False
        near_beam = any(abs(bx - px) <= 3 and abs(by - py) <= 3 for bx, by in beam_cells)
        return near_beam or gm.features.mirror_axis >= 0

    def decide(self, ctx: SolverContext) -> ActionDecision | None:
        gm = ctx.game_model
        player = gm.player_pos
        goals = gm.goal_positions
        if player is None or not goals:
            return None

        px, py = player
        beam_cells = self._get_beam_cells(ctx, gm)
        if not beam_cells:
            return None

        return self._handle_beam_controller(ctx, gm, beam_cells, px, py)

    def _handle_beam_controller(
        self, ctx: SolverContext, gm, beam_cells, px, py
    ) -> ActionDecision | None:
        goals = gm.goal_positions
        features = gm.features
        plan_context = ctx.plan_context

        # Compute beam solver result (cached per frame)
        solver_result = self._compute_beam_solver(ctx, gm, beam_cells, px, py)
        if solver_result is None:
            return self._fallback(ctx)

        best_mx = solver_result.get("best_mirror_x")
        best_my = solver_result.get("best_mirror_y")
        if best_mx is not None:
            self._beam_target_x = best_mx
        if best_my is not None:
            self._beam_target_y = best_my
        if self._beam_target_x is None or self._beam_target_y is None:
            return self._fallback(ctx)

        target_x = self._beam_target_x
        target_y = self._beam_target_y

        # Snap to nearest walkable cell
        expanded = self._build_expanded_walkable(gm, ctx)
        if (target_x, target_y) not in expanded:
            best_dist = float("inf")
            for wx, wy in expanded:
                d = abs(wx - target_x) + abs(wy - target_y)
                if d < best_dist:
                    best_dist = d
                    target_x, target_y = wx, wy

        # Navigate toward target
        if abs(px - target_x) > 2 or abs(py - target_y) > 2:
            path = astar(
                (px, py), (target_x, target_y), expanded,
                gm.width, gm.height, max_steps=2000,
            )
            if path and len(path) > 1:
                nx, ny = path[1]
                d = "right" if nx > px else ("left" if nx < px else ("down" if ny > py else "up"))
                if self._beam_last_dir and d in ("up", "down") and self._beam_last_dir in ("up", "down") and d != self._beam_last_dir:
                    self._beam_nav_oscillation += 1
                else:
                    self._beam_nav_oscillation = 0
                self._beam_last_dir = d
                if self._beam_nav_oscillation >= 2:
                    self._beam_nav_oscillation = 0
                    return None
                action = DIR_TO_ACTION.get(d, "ACTION1")
                self._last_move_target = (nx, ny)
                store_board_state(self._memory_db_path, PuzzleBoardState(
                    game_id=ctx.episode.get("game_id", ""),
                    step=ctx.episode.get("step", 0),
                    player_x=nx, player_y=ny,
                    goal_positions=goals, beam_cells=beam_cells,
                    mirror_x=nx, goals_lit=features.goals_lit_by_beam,
                    action_taken=action, outcome="mirror_navigate",
                )) if self._memory_db_path else None
                return ActionDecision(
                    action_name=action,
                    source="solver:beam",
                    confidence=0.85,
                    reason=f"navigate_mirror_{d}",
                )

        self._mirror_frames += 1
        uncovered = solver_result.get("uncovered_goals", [])
        near_uncovered = any(abs(gx - px) + abs(gy - py) <= 8 for gx, gy in uncovered) if uncovered else True
        mirror_timeout = self._mirror_frames >= 15
        at_target = abs(px - target_x) <= 4 and abs(py - target_y) <= 4

        # Try alternative positions
        if at_target and self._mirror_frames >= 5 and not features.goals_lit_by_beam:
            alternatives = solver_result.get("alternative_positions", [])
            if alternatives and self._beam_target_alt_idx < len(alternatives):
                idx = self._beam_target_alt_idx
                self._beam_target_x, self._beam_target_y = alternatives[idx]
                self._beam_target_alt_idx += 1
                self._mirror_frames = 0
                target_x, target_y = alternatives[idx]

        if self._mirror_frames >= 25:
            self._mirror_frames = 0
            if not at_target:
                return self._fallback(ctx)
            return self._wait_near_target(px, py, target_x, target_y)

        switch_count = int(plan_context.get("switch_attempted", 0))
        if ((switch_count < 2
             and ((features.goals_lit_by_beam > 0 and near_uncovered) or mirror_timeout))
            and not self._switch_back_issued):
            self._switch_back_issued = True
            plan_context["switch_attempted"] = switch_count + 1
            store_board_state(self._memory_db_path, PuzzleBoardState(
                game_id=ctx.episode.get("game_id", ""),
                step=ctx.episode.get("step", 0),
                player_x=px, player_y=py,
                goal_positions=goals, beam_cells=beam_cells,
                mirror_x=px, goals_lit=features.goals_lit_by_beam,
                action_taken="ACTION5", outcome="switch_from_mirror",
            )) if self._memory_db_path else None
            return ActionDecision(
                action_name="ACTION5",
                source="solver:beam",
                confidence=0.85,
                reason="mirror_positioned_switch_back",
            )

        if at_target:
            return self._wait_near_target(px, py, target_x, target_y)
        return self._fallback(ctx)

    def _compute_beam_solver(self, ctx, gm, beam_cells, px, py):
        goals = gm.goal_positions
        current_step = ctx.episode.get("step", 0)
        if current_step != self._last_step:
            self._beam_cache = None
            self._last_step = current_step
        if self._beam_cache is not None:
            beam_cache_key = f"{gm.features.mirror_axis}_{px}_{py}"
            if beam_cache_key == self._beam_cache.get("key"):
                return self._beam_cache.get("result")
        result = solve_beam_puzzle(
            goal_positions=goals,
            beam_cells=beam_cells,
            mirror_axis=gm.features.mirror_axis,
            player_pos=(px, py),
            current_mirror_pos=(px, py),
            width=gm.width, height=gm.height,
            memory_db_path=self._memory_db_path,
            game_id=ctx.episode.get("game_id", ""),
        )
        self._beam_cache = {"key": f"{gm.features.mirror_axis}_{px}_{py}", "result": result}
        return result

    def _get_beam_cells(self, ctx, gm):
        if gm.features.has_beam and gm.features.beam_color >= 0:
            beam_comps = [c for c in gm.components
                          if c.dominant_color == gm.features.beam_color and not c.touches_border]
            if beam_comps:
                return beam_comps[0].cells[:20]
        return ctx.plan_context.get("beam_cells", [])

    def _build_expanded_walkable(self, gm, ctx):
        gid = ctx.episode.get("game_id", "")
        if self._expanded_walkable_cache is not None and self._cache_game_id == gid:
            w = set(self._expanded_walkable_cache)
        else:
            w = set(gm.walkable)
            for c in gm.components:
                if not c.touches_border:
                    for cx, cy in c.cells:
                        w.add((cx, cy))
            self._expanded_walkable_cache = set(w)
            self._cache_game_id = gid
        if gm.player_pos:
            w.add(gm.player_pos)
        for gx, gy in gm.goal_positions:
            w.add((gx, gy))
        w -= self._impassable_cells
        return w

    def _wait_near_target(self, px, py, target_x, target_y):
        if px < target_x:
            return ActionDecision(action_name="ACTION4", source="solver:beam", confidence=0.40, reason="wait_target_right")
        if px > target_x:
            return ActionDecision(action_name="ACTION3", source="solver:beam", confidence=0.40, reason="wait_target_left")
        if py < target_y:
            return ActionDecision(action_name="ACTION2", source="solver:beam", confidence=0.40, reason="wait_target_down")
        if py > target_y:
            return ActionDecision(action_name="ACTION1", source="solver:beam", confidence=0.40, reason="wait_target_up")
        return ActionDecision(action_name="ACTION2", source="solver:beam", confidence=0.40, reason="wait_target_hold")

    def _fallback(self, ctx):
        return None


register_solver("beam", BeamSolver)
