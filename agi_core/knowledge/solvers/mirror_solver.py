from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

from agi_core.knowledge.puzzle_memory import (
    initialize_puzzle_schema,
    load_puzzle_solution,
    query_puzzle_solution_fuzzy,
    query_similar_state,
    store_board_state,
)
from agi_core.knowledge.solvers import DynamicsSolver, SolverContext, register_solver
from agi_core.orch_or.analyzer import _component_center
from agi_core.orch_or.game_model import DIR_TO_ACTION
from agi_core.orch_or.search import astar
from agi_core.perception.shape_recognition import ensure_shape_recognition
from agi_core.types import ActionDecision

logger = logging.getLogger(__name__)


def _is_interior_shape_piece(comp, gm) -> bool:
    if comp.dominant_color not in (5, 0) or not (4 <= comp.size <= 100):
        return False
    x1, y1, x2, y2 = comp.bbox
    return 0 < x1 and x2 < gm.width - 1 and 0 < y1 and y2 < gm.height - 1


def _reflect_horizontal_cells(
    cells: set[tuple[int, int]] | list[tuple[int, int]],
    mirror_py: int,
) -> set[tuple[int, int]]:
    return {(x, 2 * mirror_py - y) for x, y in cells}


def _horizontal_player_side_cells(
    comp,
    player_y: int,
    mirror_py: int,
) -> set[tuple[int, int]]:
    player_sign = 1 if player_y >= mirror_py else -1
    cells = {
        (x, y)
        for x, y in comp.cells
        if ((y - mirror_py) >= 0) == (player_sign > 0)
    }
    return cells or set(comp.cells)


def _has_large_interior_goal_region(gm) -> bool:
    return any(
        comp.dominant_color == 11
        and comp.size > 100
        and any(cx < gm.width - 1 for cx, _ in comp.cells)
        for comp in gm.components
    )


def _shape_player_distance(comp, player: tuple[int, int]) -> int:
    px, py = player
    x1, y1, x2, y2 = comp.bbox
    dx = 0 if x1 <= px <= x2 else min(abs(px - x1), abs(px - x2))
    dy = 0 if y1 <= py <= y2 else min(abs(py - y1), abs(py - y2))
    return dx + dy


def _active_shape_component(shape_comps: list, player: tuple[int, int]):
    return min(
        shape_comps,
        key=lambda comp: (
            _shape_player_distance(comp, player),
            abs(_component_center(comp)[0] - player[0]) + abs(_component_center(comp)[1] - player[1]),
        ),
    )


def _target_distance(player: tuple[int, int], target: tuple[int, int]) -> int:
    return abs(int(target[0]) - int(player[0])) + abs(int(target[1]) - int(player[1]))


def _visible_or_cached_shape_count(
    shape_comps: list | None,
    cached_count: int,
) -> int:
    visible_count = len(shape_comps) if shape_comps else 0
    return visible_count if visible_count > 0 else int(cached_count or 0)


def _horizontal_small_goal_target(gm, mirror_py: int, switch_count: int, gs: int) -> tuple[int, int] | None:
    goal_components = [
        comp
        for comp in gm.components
        if comp.dominant_color == 11
        and any(cx < gm.width - 1 for cx, _ in comp.cells)
    ]
    if not goal_components:
        return None
    if switch_count == 1:
        # First shape: target the upper goal clusters (above mirror)
        upper = [
            comp
            for comp in goal_components
            if _component_center(comp)[1] < mirror_py
        ]
        if not upper:
            return None
        target_comp = min(upper, key=lambda comp: comp.bbox[0])
        target_px = max(0, min(gm.width - 1, target_comp.bbox[2] + gs))
        target_py = max(0, min(gm.height - 1, target_comp.bbox[1] - gs))
        return (target_px, target_py)
    if switch_count == 2:
        lower_goal_components = [
            comp
            for comp in goal_components
            if (_component_center(comp)[1] >= mirror_py)
        ]
        if not lower_goal_components:
            return None
        target_comp = min(lower_goal_components, key=lambda comp: comp.bbox[0])
        target_px = max(0, min(gm.width - 1, target_comp.bbox[2] - gs))
        target_py = max(0, min(gm.height - 1, target_comp.bbox[1]))
        return (target_px, target_py)
    return None


def _artifact_stage_target(
    artifact: dict[str, Any] | None,
    switch_count: int,
) -> tuple[tuple[int, int], int] | None:
    if not isinstance(artifact, dict):
        return None
    mirror_puzzle = artifact.get("mirror_puzzle", {})
    if not isinstance(mirror_puzzle, dict) or not mirror_puzzle.get("active"):
        return None
    stage_targets = mirror_puzzle.get("stage_targets", [])
    if not isinstance(stage_targets, list) or not stage_targets:
        return None
    stage_index = min(max(switch_count - 1, 0), len(stage_targets) - 1)
    target = stage_targets[stage_index].get("target_anchor")
    if not isinstance(target, list) or len(target) != 2:
        return None
    return (int(target[0]), int(target[1])), len(stage_targets)


def _plan_stage_target(
    plan_context: dict[str, Any] | None,
    switch_count: int,
) -> tuple[tuple[int, int], int] | None:
    if not isinstance(plan_context, dict):
        return None
    stage_targets = plan_context.get("shape_stage_targets")
    if isinstance(stage_targets, list) and stage_targets:
        stage_index = min(max(switch_count - 1, 0), len(stage_targets) - 1)
        target = dict(stage_targets[stage_index] or {}).get("target_anchor")
        if isinstance(target, list) and len(target) == 2:
            return (int(target[0]), int(target[1])), len(stage_targets)
    active_target = plan_context.get("active_shape_target")
    mirror_active = bool(plan_context.get("mirror_puzzle_active"))
    if (
        mirror_active
        and isinstance(active_target, (list, tuple))
        and len(active_target) == 2
    ):
        return (int(active_target[0]), int(active_target[1])), max(
            int(plan_context.get("active_shape_stage", 1) or 1),
            1,
        )
    return _artifact_stage_target(plan_context.get("shape_recognition"), switch_count)


def _plan_stage_target_metadata(
    plan_context: dict[str, Any] | None,
    switch_count: int,
) -> dict[str, Any] | None:
    if not isinstance(plan_context, dict):
        return None
    stage_targets = plan_context.get("shape_stage_targets")
    if isinstance(stage_targets, list) and stage_targets:
        stage_index = min(max(switch_count - 1, 0), len(stage_targets) - 1)
        stage_target = stage_targets[stage_index]
        return stage_target if isinstance(stage_target, dict) else None
    active_target = plan_context.get("active_shape_target")
    mirror_active = bool(plan_context.get("mirror_puzzle_active"))
    if (
        mirror_active
        and isinstance(active_target, (list, tuple))
        and len(active_target) == 2
    ):
        return {
            "switch_stage": max(int(plan_context.get("active_shape_stage", 1) or 1), 1),
            "target_anchor": [int(active_target[0]), int(active_target[1])],
        }
    artifact = plan_context.get("shape_recognition")
    if not isinstance(artifact, dict):
        return None
    mirror_puzzle = artifact.get("mirror_puzzle", {})
    if not isinstance(mirror_puzzle, dict):
        return None
    stage_targets = mirror_puzzle.get("stage_targets", [])
    if not isinstance(stage_targets, list) or not stage_targets:
        return None
    stage_index = min(max(switch_count - 1, 0), len(stage_targets) - 1)
    stage_target = stage_targets[stage_index]
    return stage_target if isinstance(stage_target, dict) else None


class MirrorSolver(DynamicsSolver):
    """Solver for mirror reflection puzzles (color-10 mirror, no beam).

    Handles vertical mirrors (reflect X, move LEFT/RIGHT) and horizontal
    mirrors (reflect Y, move UP/DOWN).  Supports multi-shape-piece levels
    (Level 3+) by cycling through entities via ACTION5 switches.
    """

    dynamics = "mirror"

    def __init__(self, memory_db_path: str | Path | None = None) -> None:
        super().__init__(memory_db_path)
        self._last_move_target: tuple[int, int] | None = None
        self._impassable_cells: set[tuple[int, int]] = set()
        self._cache_board_key: tuple[int, tuple[int, int] | None, int] | None = None
        self._expanded_walkable_cache: set[tuple[int, int]] | None = None
        self._cached_mirror_gx: int = -1
        self._cached_mirror_gy: int = -1
        self._cached_horizontal_target_key: tuple[int, int, int] | None = None
        self._cached_horizontal_target: tuple[int, int] | None = None
        self._cached_stage_target_key: tuple[str, int] | None = None
        self._cached_stage_target: tuple[int, int] | None = None
        self._last_levels_completed: int | None = None
        self._max_shape_count_seen: int = 0
        self._mirror_is_horizontal: bool = False
        self._positioned_steps: int = 0
        self._switch_count: int = 0  # 0=mirror, 1=first shape, 2=second shape, …
        self._last_shape_pos: tuple[int, int] | None = None
        self._shape_stuck_steps: int = 0
        self._level_id: str = ""
        if self._memory_db_path:
            self._memory_db_path.parent.mkdir(parents=True, exist_ok=True)
            self.reset_level()
            initialize_puzzle_schema(self._memory_db_path)

    @property
    def has_active_attempt(self) -> bool:
        return self._positioned_steps > 0 or self._switch_count > 0

    def reset_level(self, full: bool = True) -> None:
        self._last_move_target = None
        self._impassable_cells.clear()
        self._cache_board_key = None
        self._expanded_walkable_cache = None
        self._cached_mirror_gx = -1
        self._cached_mirror_gy = -1
        self._cached_horizontal_target_key = None
        self._cached_horizontal_target = None
        self._cached_stage_target_key = None
        self._cached_stage_target = None
        self._last_levels_completed = None
        self._max_shape_count_seen = 0
        self._mirror_is_horizontal = False
        self._positioned_steps = 0
        self._last_shape_pos = None
        self._shape_stuck_steps = 0
        # _switch_count is NOT reset here — it persists across entity
        # switches within the same level.  The orchestrator's own
        # _switch_count (tracked separately) is used for switch-back logic.

    @staticmethod
    def _clear_failed_directions(ctx) -> None:
        if ctx.plan_context is not None:
            ctx.plan_context["failed_directions"] = set()

    # ── Axis detection ────────────────────────────────────────────────

    @staticmethod
    def _detect_mirror_axis(mirror_comp) -> bool:
        mx1, my1, mx2, my2 = mirror_comp.bbox
        return (mx2 - mx1) > (my2 - my1)

    @staticmethod
    def _select_mirror_component(gm):
        mirrors = [c for c in gm.components if c.dominant_color == 10]
        if not mirrors:
            return None
        player = gm.player_pos
        if player is not None and len(mirrors) > 1:
            px, py = player
            mirrors.sort(
                key=lambda c: min(
                    abs(cx - px) + abs(cy - py) for cx, cy in c.cells
                )
            )
        return mirrors[0]

    # ── Activation ─────────────────────────────────────────────────────

    def should_activate(self, ctx: SolverContext) -> bool:
        gm = ctx.game_model
        if gm.player_pos is None or not gm.goal_positions:
            return False
        mirror_comp = self._select_mirror_component(gm)
        if mirror_comp is None:
            return False
        planned_stage_target = _plan_stage_target(ctx.plan_context, max(self._switch_count, 1))
        if planned_stage_target and (not gm.features.has_beam or self.has_active_attempt):
            return True
        if gm.features.has_beam and not self._detect_mirror_axis(mirror_comp):
            return False
        horiz = self._detect_mirror_axis(mirror_comp)
        if horiz:
            # Horizontal mirror boards without an active beam still need the
            # mirror solver to own post-switch shape navigation. Only keep the
            # old beam-specific exclusion when a live beam board is present.
            return not gm.features.has_beam
        px, py = gm.player_pos
        gs = max(int(getattr(gm.features, "grid_scale", 1) or 1), 1)
        player_gx = px // gs
        mirror_gx = (
            sum(cx for cx, _ in mirror_comp.cells) // max(len(mirror_comp.cells), 1)
        ) // gs
        if abs(player_gx - mirror_gx) <= 1:
            return True
        mirr_cells = list(mirror_comp.cells)
        mirror_px = sum(cx for cx, _ in mirr_cells) // len(mirr_cells)
        goal_xs = [gx for gx, _ in gm.goal_positions]
        if not goal_xs:
            return False
        return ((px < mirror_px < min(goal_xs)) or
                (px > mirror_px > max(goal_xs)) or
                (px == mirror_px))

    # ── Decision ───────────────────────────────────────────────────────

    def decide(self, ctx: SolverContext) -> ActionDecision | None:
        try:
            return self._decide_impl(ctx)
        except Exception as exc:
            import traceback
            logger.error("MirrorSolver FAILED at step %d: %s\n%s",
                         ctx.plan_context.get("game_step", -1) if ctx.plan_context else -1,
                         exc, traceback.format_exc())
            return None

    def _decide_impl(self, ctx: SolverContext) -> ActionDecision | None:
        gm = ctx.game_model
        ensure_shape_recognition(None, ctx.plan_context)
        goals = gm.goal_positions
        if gm.player_pos is None or not goals:
            return None

        self._sync_level_state(ctx.plan_context.get("levels_completed"))

        px, py = gm.player_pos
        gs = gm.features.grid_scale
        if gs <= 1:
            return None

        # Find mirror component & axis
        all_c10 = [c for c in gm.components if c.dominant_color == 10]
        if not all_c10:
            return None
        if len(all_c10) > 1:
            _pp = gm.player_pos
            if _pp is not None:
                _ppx, _ppy = _pp
                all_c10.sort(key=lambda c: min(abs(cx - _ppx) + abs(cy - _ppy) for cx, cy in c.cells))
        mirror_comp = all_c10[0]
        self._mirror_is_horizontal = self._detect_mirror_axis(mirror_comp)

        # Mirror centre
        mirr_xs = [cx for cx, _ in mirror_comp.cells]
        mirr_ys = [cy for _, cy in mirror_comp.cells]
        if not mirr_xs:
            return None
        mirror_px = sum(mirr_xs) // len(mirr_xs)
        mirror_py = sum(mirr_ys) // len(mirr_ys)
        mirror_gx = mirror_px // gs
        mirror_gy = mirror_py // gs

        player_gx = px // gs
        player_gy = py // gs

        # Shape pieces
        shape_comps = [c for c in gm.components if _is_interior_shape_piece(c, gm)]
        self._max_shape_count_seen = max(self._max_shape_count_seen, len(shape_comps))
        has_shape = bool(shape_comps)
        spx, spy = (px, py)
        planned_stage_target = _plan_stage_target(ctx.plan_context, self._switch_count)

        # Goals
        goal_cells = [
            c for comp in gm.components
            if comp.dominant_color == 11
            and any(cx < gm.width - 1 for cx, _ in comp.cells)
            for c in comp.cells
        ]
        if not goal_cells:
            return None
        min_gpx = min(c[0] for c in goal_cells)
        min_gpy = min(c[1] for c in goal_cells)
        max_gpx = max(c[0] for c in goal_cells)
        max_gpy = max(c[1] for c in goal_cells)
        min_ggx = min_gpx // gs
        max_ggx = max_gpx // gs
        min_ggy = min_gpy // gs
        max_ggy = max_gpy // gs

        # ── Player IS the mirror? ──────────────────────────────────────
        if self._mirror_is_horizontal:
            player_is_mirror = abs(player_gy - mirror_gy) <= 1
            if not player_is_mirror and mirr_ys:
                live_mirror_py = sum(mirr_ys) // len(mirr_ys)
                for c in gm.components:
                    if c.dominant_color == 0 and not c.touches_border and 1 <= c.size <= 30:
                        _, cy = _component_center(c)
                        if abs(cy - live_mirror_py) <= gs:
                            player_is_mirror = True
                            py = cy
                            player_gy = py // gs
                            mirror_gy = live_mirror_py // gs
                            break
        else:
            player_is_mirror = abs(player_gx - mirror_gx) <= 1
            if not player_is_mirror and mirr_xs:
                live_mirror_px = sum(mirr_xs) // len(mirr_xs)
                for c in gm.components:
                    if c.dominant_color == 0 and not c.touches_border and 1 <= c.size <= 30:
                        cx, _ = _component_center(c)
                        if abs(cx - live_mirror_px) <= gs:
                            player_is_mirror = True
                            px = cx
                            player_gx = px // gs
                            mirror_gx = live_mirror_px // gs
                            break

        # Cache mirror position
        if self._cached_mirror_gx < 0 or player_is_mirror:
            self._cached_mirror_gx = mirror_gx
            self._cached_mirror_gy = mirror_gy
        if not player_is_mirror:
            mirror_gx = self._cached_mirror_gx
            mirror_gy = self._cached_mirror_gy

        # ── Player IS mirror: position it or switch ────────────────────
        if player_is_mirror and has_shape:
            # For horizontal mirrors, skip mirror positioning — the mirror
            # is a passive reflector. Switch directly to the first shape
            # piece and navigate it using beam-coverage-aware targets.
            if self._switch_count == 0 and self._mirror_is_horizontal:
                self._switch_count = 1
                self._cached_horizontal_target_key = None
                self._cached_horizontal_target = None
                self._clear_failed_directions(ctx)
                ctx.plan_context["switch_attempted"] = 1
                return ActionDecision("ACTION5", "solver:mirror", 0.70,
                                      reason="mirror_switch_horiz")
            # After the first switch (_switch_count > 0) the player should
            # be a shape piece — ignore false mirror detection from stray
            # color-0 cells near the mirror Y.
            if self._switch_count == 0:
                return self._navigate_mirror(
                    gm, ctx, shape_comps, mirror_comp, mirror_gx, mirror_gy,
                    min_ggx, max_ggx, min_ggy, max_ggy, gs,
                )
            # Fall through to shape-piece navigation.
            return self._navigate_shape(
                gm, ctx, (spx, spy), mirror_gx, mirror_gy, mirror_py, gs,
                min_ggx, max_ggx, min_ggy, max_ggy,
                shape_comps,
            )

        # ── Player is a shape piece ────────────────────────────────────
        if not self._mirror_is_horizontal:
            if not ((player_gx < mirror_gx < min_ggx) or
                    (player_gx > mirror_gx > max_ggx)):
                if planned_stage_target is None:
                    return None

        return self._navigate_shape(
            gm, ctx, (spx, spy), mirror_gx, mirror_gy, mirror_py, gs,
            min_ggx, max_ggx, min_ggy, max_ggy,
            shape_comps,
        )

    # ── Mirror navigation (player IS the mirror) ──────────────────────

    def _navigate_mirror(
        self, gm, ctx, shape_comps, mirror_comp,
        mirror_gx: int, mirror_gy: int,
        min_ggx: int, max_ggx: int,
        min_ggy: int, max_ggy: int,
        gs: int,
    ) -> ActionDecision | None:
        # For horizontal mirrors, skip positioning and switch directly.
        if self._mirror_is_horizontal:
            if shape_comps and self._switch_count < len(shape_comps):
                self._switch_count += 1
                self._cached_horizontal_target_key = None
                self._cached_horizontal_target = None
                self._clear_failed_directions(ctx)
                ctx.plan_context["switch_attempted"] = self._switch_count
                return ActionDecision("ACTION5", "solver:mirror", 0.70,
                                      reason="mirror_switch_horiz")
            return ActionDecision("ACTION6", "solver:mirror", 0.60,
                                  reason="mirror_skip_horiz")

        # Vertical mirror: position it first.
        best_shape = max(shape_comps, key=lambda c: c.size)
        shape_right_gx = best_shape.bbox[2] // gs
        target_mgx = (min_ggx + shape_right_gx) // 2
        dx = target_mgx - mirror_gx
        mirror_width = mirror_comp.bbox[2] - mirror_comp.bbox[0] + 1
        alignment_tolerance = 1 if mirror_comp.touches_border and mirror_width >= gs * 3 else 0
        if dx < -alignment_tolerance:
            return ActionDecision("ACTION3", "solver:mirror", 0.70,
                                  reason="mirror_nav_left")
        elif dx > alignment_tolerance:
            return ActionDecision("ACTION4", "solver:mirror", 0.70,
                                  reason="mirror_nav_right")
        self._positioned_steps += 1
        if self._positioned_steps >= 3 and shape_comps and self._switch_count < len(shape_comps):
            self._switch_count += 1
            self._cached_horizontal_target_key = None
            self._cached_horizontal_target = None
            self._clear_failed_directions(ctx)
            return ActionDecision("ACTION5", "solver:mirror", 0.70,
                                  reason="mirror_switch_to_shape")
        return ActionDecision("ACTION6", "solver:mirror", 0.60,
                              reason="mirror_positioned_wait")

    # ── Shape-piece navigation (after switch) ─────────────────────────

    def _navigate_shape(
        self, gm, ctx,
        player: tuple[int, int],
        mirror_gx: int, mirror_gy: int, mirror_py: int, gs: int,
        min_ggx: int, max_ggx: int,
        min_ggy: int, max_ggy: int,
        shape_comps: list | None = None,
    ) -> ActionDecision | None:
        px, py = player

        if self._mirror_is_horizontal:
            artifact_target = _plan_stage_target(ctx.plan_context, self._switch_count)
            max_switches_override = 0
            if artifact_target is not None:
                (target_px, target_py), max_switches_override = artifact_target
            else:
                horizontal_target = None
                if _has_large_interior_goal_region(gm):
                    horizontal_target = self._cached_horizontal_target
                    target_key = (
                        self._switch_count,
                        mirror_py,
                        _visible_or_cached_shape_count(shape_comps, self._max_shape_count_seen),
                    )
                    if horizontal_target is None or self._cached_horizontal_target_key != target_key:
                        horizontal_target = self._best_horizontal_shape_target(
                            gm, player, mirror_py, shape_comps or []
                        )
                        self._cached_horizontal_target_key = target_key
                        self._cached_horizontal_target = horizontal_target
                else:
                    self._cached_horizontal_target_key = None
                    self._cached_horizontal_target = None
                if horizontal_target is not None:
                    target_px, target_py = horizontal_target
                else:
                    small_goal_target = _horizontal_small_goal_target(gm, mirror_py, self._switch_count, gs)
                    if small_goal_target is not None:
                        target_px, target_py = small_goal_target
                    else:
                        # Clamp target to the side of the mirror opposite the goal cluster
                        # so the shape's beam can reach goals via reflection.
                        if self._switch_count == 1:
                            target_gx = (min_ggx + max_ggx) // 2
                            target_gy = max(0, min(gm.height // gs - 1, mirror_gy - 4))
                        else:
                            target_gx = (min_ggx + max_ggx) // 2
                            target_gy = max(0, min(gm.height // gs - 1, mirror_gy + 2))
                        target_py = max(0, min(gm.height - 1, target_gy * gs + 1))
                        target_px = max(0, min(gm.width - 1, target_gx * gs + 1))
        else:
            max_switches_override = 0
            artifact_target = _plan_stage_target(ctx.plan_context, self._switch_count)
            stage_target_meta = _plan_stage_target_metadata(
                ctx.plan_context,
                self._switch_count,
            )
            trust_single_stage_target = bool(
                isinstance(stage_target_meta, dict)
                and (
                    stage_target_meta.get("target_bbox") is not None
                    or stage_target_meta.get("reflected_coverage") is not None
                    or int(stage_target_meta.get("fresh_hits", 0) or 0) > 0
                )
            )
            if artifact_target is not None and (
                artifact_target[1] > 1 or trust_single_stage_target
            ):
                (target_px, target_py), max_switches_override = artifact_target
            else:
                target_gx = 2 * mirror_gx - max_ggx
                target_gy = min_ggy
                target_px = max(0, min(gm.width - 1, target_gx * gs + 1))
                target_py = target_gy * gs + 1

        # Stuck detection: switch to next shape piece if stuck
        if self._last_shape_pos == (px, py):
            self._shape_stuck_steps += 1
        else:
            self._shape_stuck_steps = 0
        self._last_shape_pos = (px, py)

        # Use entity tracker's stationary streak if available
        entity_tracks = ctx.plan_context.get("entity_tracks", [])
        if entity_tracks and self._shape_stuck_steps < 5:
            for et in entity_tracks:
                et_center = et.get("center")
                if et_center and abs(et_center[0] - px) + abs(et_center[1] - py) <= 3:
                    if et.get("stationary") and et.get("age", 0) > 3:
                        self._shape_stuck_steps = max(self._shape_stuck_steps, int(et.get("age", 0)))
                    break

        max_switches = max_switches_override or _visible_or_cached_shape_count(
            shape_comps, self._max_shape_count_seen
        )
        cached_stage_target = (
            self._cached_stage_target
            if self._cached_stage_target_key == (("horizontal" if self._mirror_is_horizontal else "vertical"), int(self._switch_count))
            else None
        )
        if (
            self._mirror_is_horizontal
            and cached_stage_target is not None
            and self._switch_count < max_switches
        ):
            raw_target = (int(target_px), int(target_py))
            target_shift = _target_distance(cached_stage_target, raw_target)
            cached_distance = _target_distance(player, cached_stage_target)
            if target_shift >= 12 and cached_distance <= 10:
                self._switch_count += 1
                self._cached_horizontal_target_key = None
                self._cached_horizontal_target = None
                self._cached_stage_target_key = None
                self._cached_stage_target = None
                self._clear_failed_directions(ctx)
                ctx.plan_context["switch_attempted"] = self._switch_count
                return ActionDecision("ACTION5", "solver:mirror", 0.67,
                                      reason="mirror_switch_target_jump")
        if self._shape_stuck_steps >= 20 and self._switch_count < max_switches:
            self._switch_count += 1
            self._cached_horizontal_target_key = None
            self._cached_horizontal_target = None
            self._cached_stage_target_key = None
            self._cached_stage_target = None
            self._clear_failed_directions(ctx)
            ctx.plan_context["switch_attempted"] = self._switch_count
            return ActionDecision("ACTION5", "solver:mirror", 0.65,
                                  reason="mirror_switch_other_shape")

        arrival_tolerance = 1
        if self._mirror_is_horizontal:
            if self._switch_count < max_switches:
                arrival_tolerance = max(1, gs)
            else:
                # Final horizontal placements need an exact anchor match.
                arrival_tolerance = 0
        if abs(px - target_px) + abs(py - target_py) <= arrival_tolerance:
            max_switches = _visible_or_cached_shape_count(shape_comps, self._max_shape_count_seen)
            if self._switch_count < max_switches:
                self._switch_count += 1
                self._cached_horizontal_target_key = None
                self._cached_horizontal_target = None
                self._cached_stage_target_key = None
                self._cached_stage_target = None
                self._clear_failed_directions(ctx)
                ctx.plan_context["switch_attempted"] = self._switch_count
                return ActionDecision("ACTION5", "solver:mirror", 0.68,
                                      reason="mirror_switch_next_shape")
            return ActionDecision("ACTION6", "solver:mirror", 0.60,
                                  reason="mirror_placed_wait")

        target_px, target_py = self._stabilize_stage_target(
            orientation="horizontal" if self._mirror_is_horizontal else "vertical",
            stage=self._switch_count,
            player=player,
            target=(target_px, target_py),
        )

        expanded = self._build_expanded_walkable(gm, ctx)
        try:
            path = astar(player, (target_px, target_py), expanded,
                         gm.width, gm.height, max_steps=5000)
        except Exception:
            path = None
        if path and len(path) > 1:
            nx, ny = path[1]
            d = ("right" if nx > px else "left" if nx < px else
                 "down" if ny > py else "up")
            self._last_move_target = (nx, ny)
            return ActionDecision(
                DIR_TO_ACTION.get(d, "ACTION1"), "solver:mirror", 0.75,
                reason=f"mirror_placement_{d}",
                metadata={"target_anchor": [target_px, target_py], "shape_stage": self._switch_count},
            )

        # Heuristic fallback
        failed = ctx.plan_context.get("failed_directions", set())
        dx = target_px - px
        dy = target_py - py
        candidates = []
        if dx > 0: candidates.append("right")
        elif dx < 0: candidates.append("left")
        if dy > 0: candidates.append("down")
        elif dy < 0: candidates.append("up")
        vec = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
        for d in candidates:
            if d in failed:
                continue
            dvx, dvy = vec[d]
            nx, ny = px + dvx, py + dvy
            if not expanded or (nx, ny) in expanded:
                return ActionDecision(
                    DIR_TO_ACTION.get(d, "ACTION1"), "solver:mirror", 0.55,
                    reason=f"mirror_heuristic_{d}",
                    metadata={"target_anchor": [target_px, target_py], "shape_stage": self._switch_count},
                )
        for d, (dvx, dvy) in vec.items():
            nx, ny = px + dvx, py + dvy
            if 0 <= nx < gm.width and 0 <= ny < gm.height:
                return ActionDecision(
                    DIR_TO_ACTION.get(d, "ACTION1"), "solver:mirror", 0.40,
                    reason=f"mirror_forced_{d}",
                    metadata={"target_anchor": [target_px, target_py], "shape_stage": self._switch_count},
                )
        return None

    def _best_horizontal_shape_target(
        self,
        gm,
        player: tuple[int, int],
        mirror_py: int,
        shape_comps: list,
    ) -> tuple[int, int] | None:
        if not shape_comps:
            return None
        goal_cells = {
            cell
            for comp in gm.components
            if comp.dominant_color == 11
            and any(cx < gm.width - 1 for cx, _ in comp.cells)
            for cell in comp.cells
        }
        if not goal_cells:
            return None
        active_shape = _active_shape_component(shape_comps, player)
        other_coverage: set[tuple[int, int]] = set()
        for comp in shape_comps:
            if comp is active_shape:
                continue
            other_cells = _horizontal_player_side_cells(comp, player[1], mirror_py)
            other_coverage.update(_reflect_horizontal_cells(other_cells, mirror_py))
        remaining_goals = goal_cells - other_coverage
        active_cells = _horizontal_player_side_cells(active_shape, player[1], mirror_py)
        xs = [x for x, _ in active_cells]
        ys = [y for _, y in active_cells]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        rel_cells = {(x - x1, y - y1) for x, y in active_cells}
        shape_w = x2 - x1 + 1
        shape_h = y2 - y1 + 1
        current_cx = sum(x for x, _ in active_cells) // len(active_cells)
        current_cy = sum(y for _, y in active_cells) // len(active_cells)
        best: tuple[int, int, int, int, int] | None = None
        for tx in range(0, gm.width - shape_w + 1):
            for ty in range(0, gm.height - shape_h + 1):
                placed = {(tx + dx, ty + dy) for dx, dy in rel_cells}
                reflected = _reflect_horizontal_cells(placed, mirror_py)
                if any(x < 0 or x >= gm.width or y < 0 or y >= gm.height for x, y in reflected):
                    continue
                fresh_hits = len(reflected & remaining_goals)
                total_hits = len((reflected | other_coverage) & goal_cells)
                if fresh_hits <= 0 and total_hits <= 0:
                    continue
                center_x = sum(x for x, _ in placed) // len(placed)
                center_y = sum(y for _, y in placed) // len(placed)
                dist = abs(center_x - current_cx) + abs(center_y - current_cy)
                candidate = (fresh_hits, total_hits, -dist, center_x, center_y)
                if best is None or candidate > best:
                    best = candidate
        if best is None:
            return None
        return (best[3], best[4])

    def _stabilize_stage_target(
        self,
        *,
        orientation: str,
        stage: int,
        player: tuple[int, int],
        target: tuple[int, int],
    ) -> tuple[int, int]:
        cache_key = (orientation, int(stage))
        cached_target = self._cached_stage_target
        if self._cached_stage_target_key == cache_key and cached_target is not None:
            target_shift = abs(int(target[0]) - int(cached_target[0])) + abs(int(target[1]) - int(cached_target[1]))
            cached_distance = abs(int(cached_target[0]) - int(player[0])) + abs(int(cached_target[1]) - int(player[1]))
            if target_shift >= 6 and cached_distance > 2:
                return cached_target
        self._cached_stage_target_key = cache_key
        self._cached_stage_target = (int(target[0]), int(target[1]))
        return self._cached_stage_target

    # ── Walkable helper ───────────────────────────────────────────────

    def _sync_level_state(self, levels_completed: object) -> None:
        try:
            current_level = int(levels_completed)
        except (TypeError, ValueError):
            return
        if self._last_levels_completed is None:
            self._last_levels_completed = current_level
            return
        if current_level > self._last_levels_completed:
            self.reset_level()
            self._switch_count = 0
        self._last_levels_completed = current_level

    def _build_expanded_walkable(self, gm, ctx):
        board_key = (id(gm.board), gm.player_pos, len(gm.components))
        if self._expanded_walkable_cache is not None and self._cache_board_key == board_key:
            w = set(self._expanded_walkable_cache)
        else:
            w = set(gm.walkable)
            for c in gm.components:
                if not c.touches_border:
                    for cx, cy in c.cells:
                        w.add((cx, cy))
            self._expanded_walkable_cache = set(w)
            self._cache_board_key = board_key
        if gm.player_pos:
            w.add(gm.player_pos)
        for gx, gy in gm.goal_positions:
            w.add((gx, gy))
        w -= self._impassable_cells
        return w


register_solver("puzzle", MirrorSolver)
register_solver("mirror", MirrorSolver)
