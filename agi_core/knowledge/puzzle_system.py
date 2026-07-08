from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agi_core.knowledge.beam_solver import solve_beam_puzzle
from agi_core.orch_or.search import astar
from agi_core.knowledge.puzzle_memory import (
    PuzzleBoardState,
    initialize_puzzle_schema,
    load_puzzle_solution,
    query_puzzle_solution_fuzzy,
    query_similar_state,
    save_puzzle_solution,
    store_board_state,
    store_interaction_record,
    query_best_interaction,
)
from agi_core.orch_or.parallel import ThreadPoolManager
from agi_core.orch_or.game_model import PlanStep
from agi_core.types import ActionDecision

if TYPE_CHECKING:
    from agi_core.orch_or.game_model import GameModel
    from agi_core.types import ExecutiveContext

logger = logging.getLogger(__name__)


class PuzzleSystem:
    def __init__(self, memory_db_path: str | Path | None = None) -> None:
        self._memory_db_path = Path(memory_db_path) if memory_db_path else None
        self._action_history: dict[str, int] = {}
        self._mirror_frames: int = 0
        self._last_pos: tuple[int, int] | None = None
        self._stall_frames: int = 0
        self._expanded_walkable_cache: set[tuple[int, int]] | None = None
        self._cache_game_id: str = ""
        self._stall_direction_index: int = 0
        self._beam_cache: dict[str, Any] | None = None
        self._beam_cache_frame: int = -1
        self._last_action: str = ""
        self._last_step: int = -1
        self._beam_target_x: int | None = None
        self._beam_target_y: int | None = None
        self._beam_nav_oscillation: int = 0
        self._beam_last_dir: str = ""
        self._nav_oscillation: int = 0
        self._nav_last_dir: str = ""
        self._tried_interactions: set[tuple[int, int, str]] = set()
        self._phantom_frames: int = 0
        self._last_player_pos: tuple[int, int] | None = None
        self._last_was_phantom_action: bool = False
        self._impassable_cells: set[tuple[int, int]] = set()
        self._last_move_target: tuple[int, int] | None = None
        self._episode_steps: int = 0
        self._best_progress: int = 0
        self._beam_target_alt_idx: int = 0
        self._beam_puzzle_completed: bool = False
        self._last_cached_action_step: int = -1
        self._switch_back_issued: bool = False
        if self._memory_db_path:
            self._memory_db_path.parent.mkdir(parents=True, exist_ok=True)
            initialize_puzzle_schema(self._memory_db_path)

    def should_activate(self, game_model: GameModel) -> bool:
        features = game_model.features
        if not features.goal_detectable or not game_model.goal_positions:
            return False
        if features.has_beam:
            return True
        has_interactive = any(otype == "interactive" for _, _, otype in game_model.obstacles)
        if features.board_type in ("puzzle", "unknown") and (has_interactive or features.path_to_goal_possible):
            return True
        # Activate for any board with interactive obstacles or sprite entities
        if has_interactive and game_model.player_pos is not None:
            return True
        # Activate when sprite detection finds clickable entities
        if game_model.sprites and any(s.is_clickable and not s.is_player for s in game_model.sprites):
            return True
        # Activate for mirror reflection puzzles: color-10 mirror with goals on opposite side
        if game_model.player_pos is not None and any(c.dominant_color == 10 for c in game_model.components):
            px, py = game_model.player_pos
            mirror_comp = next((c for c in game_model.components if c.dominant_color == 10), None)
            if mirror_comp and mirror_comp.cells:
                mirror_px = sum(cx for cx, _ in mirror_comp.cells) // len(mirror_comp.cells)
                goal_xs = [gx for gx, _ in game_model.goal_positions]
                if goal_xs and ((px < mirror_px < min(goal_xs)) or (px > mirror_px > max(goal_xs))):
                    return True
        return False

    def decide(
        self,
        context: ExecutiveContext,
        game_model: GameModel,
        plan_context: dict[str, Any],
    ) -> ActionDecision:
        player = game_model.player_pos
        goals = game_model.goal_positions
        if player is None or not goals:
            return self._fallback(context)

        current_step = context.observation.step_index
        if current_step != self._last_step:
            self._beam_cache = None
            self._last_step = current_step

        px, py = player
        features = game_model.features
        non_beam_set = plan_context.get("non_beam_controllers", set())
        beam_cells = self._get_beam_cells(context, game_model)

        # Dynamic impassable detection: if player didn't move, mark target as impassable
        if self._last_player_pos is not None and self._last_move_target is not None:
            if player == self._last_player_pos:
                self._impassable_cells.add(self._last_move_target)
                if len(self._impassable_cells) > 200:
                    self._impassable_cells.clear()
            else:
                self._last_move_target = None
        self._last_player_pos = player

        goal_history = plan_context.get("goal_history", [])
        if len(goal_history) >= 2:
            latest = set(goals)
            prev = set(goal_history[-1])
            if latest != prev and latest and prev:
                self._expanded_walkable_cache = None

        if self._stall_frames >= 5:
            stall_goal_idx = self._stall_frames % max(len(goals), 1)
            goals = [goals[stall_goal_idx]] + [g for i, g in enumerate(goals) if i != stall_goal_idx]

        self._last_action = context.observation.last_action or self._last_action

        # Phantom player detection: movement actions that don't change position
        is_movement = self._last_action in ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
        if is_movement and self._last_player_pos is not None and player == self._last_player_pos:
            self._phantom_frames += 1
        elif player != self._last_player_pos:
            self._phantom_frames = 0
        self._last_player_pos = player

        # Phantom player recovery: try selecting visible entities via ACTION6 or ACTION5
        if self._phantom_frames >= 2:
            if game_model.sprites:
                clickable = [s for s in game_model.sprites if s.is_clickable and not s.is_player]
                clickable.sort(key=lambda s: abs(s.pixel_x - px) + abs(s.pixel_y - py))
                board = game_model.board
                for s in clickable:
                    tx, ty = s.pixel_x, s.pixel_y
                    found = False
                    for dy in range(s.grid_h * 3):
                        for dx in range(s.grid_w * 3):
                            cy = s.pixel_y + dy
                            cx = s.pixel_x + dx
                            if cy < len(board) and cx < len(board[cy]) and board[cy][cx] == s.color:
                                tx, ty = cx, cy
                                found = True
                                break
                        if found:
                            break
                    target = (tx, ty)
                    if (target[0], target[1], "interactive") not in self._tried_interactions:
                        self._tried_interactions.add((target[0], target[1], "interactive"))
                        self._store_state(context, game_model, [], "ACTION6", "click_obstacle")
                        return ActionDecision(
                            action_name="ACTION6",
                            source="puzzle_system:phantom",
                            confidence=0.55,
                            reason=f"phantom_select_{s.sprite_id}",
                            metadata={"x": target[0], "y": target[1]},
                        )
            # If no sprites to select or all tried, try ACTION5 (switch)
            switch_count = int(plan_context.get("switch_attempted", 0))
            if switch_count < 3:
                plan_context["switch_attempted"] = switch_count + 1
                self._store_state(context, game_model, [], "ACTION5", "phantom_switch")
                return ActionDecision(
                    action_name="ACTION5",
                    source="puzzle_system:phantom",
                    confidence=0.50,
                    reason="phantom_switch_entity",
                )

        # Reset step tracker on RESET or new episode
        if context.observation.last_action in ("RESET", None) or context.observation.step_index == 0:
            self._last_cached_action_step = -1
            self._switch_back_issued = False
            self._beam_puzzle_completed = False

        # If the board has a beam but current entity doesn't match beam color,
        # suggest switching to the beam-controlling entity (mirror).
        beam_color = features.beam_color
        switch_count = plan_context.get("switch_attempted", 0)
        has_non_beam_entities = (
            features.has_beam
            and features.player_color >= 0
            and features.player_color != beam_color
            and beam_color >= 0
            and switch_count < 3
            and not self._beam_puzzle_completed
            and any(c.dominant_color == beam_color for c in game_model.components if not c.touches_border)
        )
        if has_non_beam_entities:
            plan_context["switch_attempted"] = switch_count + 1
            self._store_state(context, game_model, beam_cells, "ACTION5", "switch_to_beam")
            return ActionDecision(
                action_name="ACTION5",
                source="puzzle_system:beam_switch",
                confidence=0.65,
                reason="switch_to_beam_controller",
            )

        is_beam_controller = (
            features.has_beam
            and features.player_color >= 0
            and features.player_color not in non_beam_set
            and (features.mirror_axis >= 0 or (beam_cells and any(abs(bx - px) <= 3 and abs(by - py) <= 3 for bx, by in beam_cells)))
        )

        if is_beam_controller and features.mirror_axis >= 0 and beam_cells:
            result = self._handle_beam_controller(
                context, game_model, plan_context, beam_cells, px, py,
            )
            if result is not None:
                return result
            # Beam controller yielded — skip cached solution, go to navigate_to_goal
            return self._navigate_to_goal(context, game_model, plan_context, goals, player)

        # Mirror reflection puzzle: player must position itself so its reflection
        # across a static mirror (color 10) covers the goals.
        mirror_result = self._try_mirror_placement(context, game_model, plan_context, goals, player)
        if mirror_result is not None:
            return mirror_result

        cached = self._query_memory(context, goals)
        if cached is not None:
            return self._action_from_memory(context, cached, game_model, plan_context)

        return self._navigate_to_goal(context, game_model, plan_context, goals, player)

    def _handle_beam_controller(
        self,
        context: ExecutiveContext,
        game_model: GameModel,
        plan_context: dict[str, Any],
        beam_cells: list[tuple[int, int]],
        px: int,
        py: int,
    ) -> ActionDecision:
        goals = game_model.goal_positions
        features = game_model.features
        is_vertical_mirror = features.mirror_axis >= 64
        mirror_pos = features.mirror_axis if not is_vertical_mirror else py
        hyperon_result = None
        try:
            hyperon_result = context.hyperon.run_rules("beam-solver")
        except Exception as exc:
            logger.debug("puzzle_system beam-solver query failed: %s", exc)
        if hyperon_result and len(hyperon_result) > 0 and hyperon_result[0].get("mirror_direction"):
            mirror_dir = hyperon_result[0]["mirror_direction"]
            action = _dir_to_action(mirror_dir)
            self._store_state(context, game_model, beam_cells, action, "mirror_move_hyperon")
            return ActionDecision(
                action_name=action,
                source="puzzle_system:hyperon_beam",
                confidence=0.85,
                reason=f"aim_mirror_{mirror_dir}",
            )
        beam_cache_key = (tuple(sorted(goals)), len(beam_cells), mirror_pos)
        if self._beam_cache is not None and self._beam_cache_frame == context.observation.step_index:
            cached_result = self._beam_cache.get(beam_cache_key)
            if cached_result is not None:
                solver_result = cached_result
            else:
                solver_result = solve_beam_puzzle(
                    goal_positions=goals, beam_cells=beam_cells,
                    mirror_axis=mirror_pos, player_pos=(px, py),
                    current_mirror_pos=(px, py),
                    width=game_model.width, height=game_model.height,
                    memory_db_path=self._memory_db_path,
                    game_id=context.observation.game_id,
                )
                self._beam_cache[beam_cache_key] = solver_result
        else:
            self._beam_cache = {}
            self._beam_cache_frame = context.observation.step_index
            solver_result = solve_beam_puzzle(
                goal_positions=goals, beam_cells=beam_cells,
                mirror_axis=mirror_pos, player_pos=(px, py),
                current_mirror_pos=(px, py),
                width=game_model.width, height=game_model.height,
                memory_db_path=self._memory_db_path,
                game_id=context.observation.game_id,
            )
            self._beam_cache[beam_cache_key] = solver_result
        best_mx = solver_result.get("best_mirror_x")
        best_my = solver_result.get("best_mirror_y")
        if best_mx is not None:
            self._beam_target_x = best_mx
        if best_my is not None:
            self._beam_target_y = best_my
        if self._beam_target_x is None or self._beam_target_y is None:
            return self._fallback(context, game_model)
        target_x = self._beam_target_x
        target_y = self._beam_target_y
        # Snap target to nearest walkable cell if it's not walkable
        expanded = self._build_expanded_walkable(game_model, context, plan_context)
        if (target_x, target_y) not in expanded:
            best_dist = float('inf')
            for wx, wy in expanded:
                d = abs(wx - target_x) + abs(wy - target_y)
                if d < best_dist:
                    best_dist = d
                    target_x, target_y = wx, wy
        if abs(px - target_x) > 2 or abs(py - target_y) > 2:
            path = astar((px, py), (target_x, target_y), expanded, game_model.width, game_model.height, max_steps=2000)
            if path and len(path) > 1:
                nx, ny = path[1]
                d = "right" if nx > px else ("left" if nx < px else ("down" if ny > py else "up"))
                # Detect oscillation: if alternating between up/down, yield to other strategies
                if self._beam_last_dir and d in ("up", "down") and self._beam_last_dir in ("up", "down") and d != self._beam_last_dir:
                    self._beam_nav_oscillation += 1
                else:
                    self._beam_nav_oscillation = 0
                self._beam_last_dir = d
                if self._beam_nav_oscillation >= 2:
                    self._beam_nav_oscillation = 0
                    return None
                action = _dir_to_action(d)
                self._store_state(context, game_model, beam_cells, action, "mirror_navigate")
                return ActionDecision(
                    action_name=action,
                    source="puzzle_system:beam_nav",
                    confidence=0.85,
                    reason=f"navigate_mirror_{d}",
                )

        self._mirror_frames += 1
        switch_count = int(plan_context.get("switch_attempted", 0))
        uncovered = solver_result.get("uncovered_goals", [])
        near_uncovered = any(abs(gx - px) + abs(gy - py) <= 8 for gx, gy in uncovered) if uncovered else True
        mirror_timeout = self._mirror_frames >= 15
        at_target = abs(px - target_x) <= 4 and abs(py - target_y) <= 4
        # Try alternative mirror positions if at target but no success
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
                return self._fallback(context, game_model)
            return self._wait_near_target(game_model, px, py, target_x, target_y)
        if ((switch_count < 2 and ((features.goals_lit_by_beam > 0 and near_uncovered) or mirror_timeout))) and not self._switch_back_issued:
            self._switch_back_issued = True
            plan_context["switch_attempted"] = switch_count + 1
            self._store_state(context, game_model, beam_cells, "ACTION5", "switch_from_mirror")
            return ActionDecision(
                action_name="ACTION5",
                source="puzzle_system:beam",
                confidence=0.85,
                reason="mirror_positioned_switch_back",
            )

        if at_target:
            return self._wait_near_target(game_model, px, py, target_x, target_y)
        return self._fallback(context)

    def _try_mirror_placement(
        self,
        context: ExecutiveContext,
        game_model: GameModel,
        plan_context: dict[str, Any],
        goals: list[tuple[int, int]],
        player: tuple[int, int],
    ) -> ActionDecision | None:
        """Compute where to position the player so its reflection across a static
        mirror (color 10) covers the goals, then navigate there via A*.
        Only applies when there is NO beam — beam puzzles use _handle_beam_controller."""
        if game_model.features.has_beam:
            return None
        px, py = player
        mirror_comp = next((c for c in game_model.components if c.dominant_color == 10), None)
        if mirror_comp is None:
            return None
        grid_scale = game_model.features.grid_scale
        if grid_scale <= 1:
            return None
        mirror_pixel_xs = [cx for cx, _ in mirror_comp.cells]
        if not mirror_pixel_xs:
            return None
        mirror_pixel_x = sum(mirror_pixel_xs) // len(mirror_pixel_xs)
        mirror_grid_x = mirror_pixel_x // grid_scale
        player_grid_x = px // grid_scale
        # Find goal component(s) with color 11 — the actual goal markers, not the
        # energy bar (which also uses color 11 at the rightmost column on 64-wide boards).
        # Energy bar cells are at column 63; goal markers are at interior columns.
        goal_comp = next((c for c in game_model.components
                          if c.dominant_color == 11
                          and any(cx < game_model.width - 1 for cx, _ in c.cells)), None)
        if goal_comp is None:
            return None
        # Filter out any remaining energy bar cells that might be merged in
        goal_cells = [c for c in goal_comp.cells
                      if c[0] < game_model.width - 1 and c[1] < game_model.height - 1]
        if not goal_cells:
            return None
            return None
        min_goal_px = min(c[0] for c in goal_cells)
        min_goal_py = min(c[1] for c in goal_cells)
        max_goal_px = max(c[0] for c in goal_cells)
        max_goal_py = max(c[1] for c in goal_cells)
        min_goal_gx = min_goal_px // grid_scale
        max_goal_gx = max_goal_px // grid_scale
        min_goal_gy = min_goal_py // grid_scale
        max_goal_gy = max_goal_py // grid_scale
        # Check player and goals are on opposite sides of the mirror
        if not ((player_grid_x < mirror_grid_x < min_goal_gx) or
                (player_grid_x > mirror_grid_x > max_goal_gx)):
            return None
        # Compute reflected goal bounding box in grid coords
        ref_min_gx = 2 * mirror_grid_x - max_goal_gx
        ref_max_gx = 2 * mirror_grid_x - min_goal_gx
        ref_min_gy = min_goal_gy
        ref_max_gy = max_goal_gy
        # Get shape piece dimensions from the component whose bbox encloses the
        # player position (the player may be at the transparent center of the sprite).
        player_comp = next((c for c in game_model.components
                           if c.bbox[0] <= px <= c.bbox[2] and c.bbox[1] <= py <= c.bbox[3]
                           and c.dominant_color in (0, 5)), None)
        if player_comp is None:
            return None
        bbox = player_comp.bbox
        shape_grid_w = (bbox[2] - bbox[0] + 1) // grid_scale
        shape_grid_h = (bbox[3] - bbox[1] + 1) // grid_scale
        # Compute anchor so shape's reflected bounding box aligns with goal bounding box.
        # Reflection flips left-right: shape piece's LEFTMOST pixel (at gx) reflects to
        # RIGHTMOST reflected position (2*mirror - gx), and its RIGHTMOST pixel (gx+w-1)
        # reflects to LEFTMOST reflected position (2*mirror - (gx+w-1)).
        # We set gx so the RIGHTMOST reflected pixel equals the GOAL's RIGHTMOST grid x.
        target_grid_x = 2 * mirror_grid_x - max_goal_gx
        target_grid_y = min_goal_gy
        max_grid_x = (game_model.width - 1) // grid_scale
        max_grid_y = (game_model.height - 1) // grid_scale
        target_grid_x = max(0, min(max_grid_x - shape_grid_w + 1, target_grid_x))
        target_grid_y = max(0, min(max_grid_y - shape_grid_h + 1, target_grid_y))
        target_px = target_grid_x * grid_scale + 1
        target_py = target_grid_y * grid_scale + 1
        if abs(px - target_px) + abs(py - target_py) <= 1:
            return ActionDecision(
                action_name="ACTION6",
                source="puzzle_system:mirror_placed",
                confidence=0.60,
                reason="mirror_placed_wait",
                metadata={"x": px, "y": py},
            )
        expanded = self._build_expanded_walkable(game_model, context, plan_context)
        path = astar(player, (target_px, target_py), expanded,
                     game_model.width, game_model.height, max_steps=5000)
        if path and len(path) > 1:
            nx, ny = path[1]
            d = "right" if nx > px else ("left" if nx < px else ("down" if ny > py else "up"))
            self._last_move_target = (nx, ny)
            self._store_state(context, game_model, [], _dir_to_action(d), "mirror_placement")
            return ActionDecision(
                action_name=_dir_to_action(d),
                source="puzzle_system:mirror_placement",
                confidence=0.75,
                reason=f"mirror_placement_{d}",
            )
        # Heuristic fallback: move toward target if A* fails
        failed = plan_context.get("failed_directions", set()) if plan_context else set()
        dx = target_px - px
        dy = target_py - py
        candidates = []
        if dx > 0:
            candidates.append("right")
        elif dx < 0:
            candidates.append("left")
        if dy > 0:
            candidates.append("down")
        elif dy < 0:
            candidates.append("up")
        for d in candidates:
            if d in failed:
                continue
            dvx, dvy = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}[d]
            nx, ny = px + dvx, py + dvy
            if not expanded or (nx, ny) in expanded:
                self._last_move_target = (nx, ny)
                self._store_state(context, game_model, [], _dir_to_action(d), "mirror_heuristic")
                return ActionDecision(
                    action_name=_dir_to_action(d),
                    source="puzzle_system:mirror_heuristic",
                    confidence=0.55,
                    reason=f"mirror_heuristic_{d}",
                )
        return None

    def _count_goals_covered_by_reflection(
        self,
        game_model: GameModel,
        px: int,
        py: int,
        mirror_pixel_x: int,
    ) -> int:
        covered = 0
        player_comp = next((c for c in game_model.components
                           if c.bbox[0] <= px <= c.bbox[2] and c.bbox[1] <= py <= c.bbox[3]
                           and c.dominant_color in (0, 5)), None)
        if player_comp is None:
            return 0
        # Get player sprite pixels (non-transparent)
        player_cells = [(cx, cy) for cx, cy in player_comp.cells
                        if not (cx == px and cy == py)]
        # Reflect each pixel across the mirror
        reflected = {(2 * mirror_pixel_x - cx, cy) for cx, cy in player_cells}
        # Check overlap with goal cells
        goal_comp = next((c for c in game_model.components if c.dominant_color == 11), None)
        if goal_comp is None:
            return 0
        goal_cells = set(goal_comp.cells)
        for r in reflected:
            if r in goal_cells:
                covered += 1
        return covered

    def _find_better_reflection_position(
        self,
        game_model: GameModel,
        px: int,
        py: int,
        mirror_pixel_x: int,
        target_grid_x: int,
        target_grid_y: int,
        shape_grid_w: int,
        shape_grid_h: int,
    ) -> tuple[int, int] | None:
        grid_scale = game_model.features.grid_scale
        best_coverage = self._count_goals_covered_by_reflection(game_model, px, py, mirror_pixel_x)
        best_pos = None
        # Scan ±2 grid cells in x and y
        for dgx in range(-2, 3):
            for dgy in range(-2, 3):
                if dgx == 0 and dgy == 0:
                    continue
                gx = target_grid_x + dgx
                gy = target_grid_y + dgy
                max_gx = (game_model.width - 1) // grid_scale
                max_gy = (game_model.height - 1) // grid_scale
                if gx < 0 or gx > max_gx - shape_grid_w + 1:
                    continue
                if gy < 0 or gy > max_gy - shape_grid_h + 1:
                    continue
                test_px = gx * grid_scale + 1
                test_py = gy * grid_scale + 1
                coverage = self._count_goals_covered_by_reflection(game_model, test_px, test_py, mirror_pixel_x)
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_pos = (test_px, test_py)
        return best_pos

    def _wait_near_target(
        self,
        game_model: GameModel,
        px: int,
        py: int,
        target_x: int,
        target_y: int,
    ) -> ActionDecision:
        if px < target_x:
            return ActionDecision(action_name="ACTION4", source="puzzle_system:wait", confidence=0.40, reason="wait_target_right")
        if px > target_x:
            return ActionDecision(action_name="ACTION3", source="puzzle_system:wait", confidence=0.40, reason="wait_target_left")
        if py < target_y:
            return ActionDecision(action_name="ACTION2", source="puzzle_system:wait", confidence=0.40, reason="wait_target_down")
        if py > target_y:
            return ActionDecision(action_name="ACTION1", source="puzzle_system:wait", confidence=0.40, reason="wait_target_up")
        return ActionDecision(action_name="ACTION2", source="puzzle_system:wait", confidence=0.40, reason="wait_target_hold")

    def _build_expanded_walkable(
        self, game_model: GameModel, context: ExecutiveContext, plan_context: dict[str, Any],
    ) -> set[tuple[int, int]]:
        gid = context.observation.game_id
        if self._expanded_walkable_cache is not None and self._cache_game_id == gid:
            w = set(self._expanded_walkable_cache)
        else:
            w = set(game_model.walkable)
            for c in game_model.components:
                if not c.touches_border:
                    for cx, cy in c.cells:
                        w.add((cx, cy))
            self._expanded_walkable_cache = set(w)
            self._cache_game_id = gid
        if game_model.player_pos:
            w.add(game_model.player_pos)
        for gx, gy in game_model.goal_positions:
            w.add((gx, gy))
        w -= self._impassable_cells
        return w

    def _astar_parallel(
        self,
        starts: list[tuple[int, int]],
        goals: list[tuple[int, int]],
        passable: set[tuple[int, int]],
        width: int,
        height: int,
        max_steps: int = 2000,
        timeout: float = 0.5,
    ) -> list[list[tuple[int, int]] | None]:
        """Run multiple A* paths in parallel and return results within timeout."""
        if len(starts) <= 1:
            result = []
            for s, g in zip(starts, goals):
                result.append(astar(s, g, passable, width, height, max_steps))
            return result
        tasks = []
        for s, g in zip(starts, goals):
            tasks.append((
                lambda s=s, g=g, p=passable, w=width, h=height, ms=max_steps: astar(s, g, p, w, h, ms),
                f"astar_{s}_{g}",
            ))
        raw = ThreadPoolManager.run_all(tasks, timeout=timeout)
        results = []
        for path, name in raw:
            results.append(path)
        return results

    def _detect_impassable(self, prev_pos: tuple[int, int] | None, curr_pos: tuple[int, int] | None, move_target: tuple[int, int] | None) -> None:
        """Track cells where the game engine blocked movement.
        
        When the player sends a movement action but doesn't move, the target cell
        is likely blocked by the game engine's collision model. We track these cells
        and exclude them from future pathfinding.
        """
        if prev_pos is not None and curr_pos is not None and move_target is not None:
            if prev_pos == curr_pos and move_target != curr_pos:
                self._impassable_cells.add(move_target)
                if len(self._impassable_cells) > 200:
                    self._impassable_cells.clear()

    def _find_blocking_obstacles(
        self,
        player: tuple[int, int],
        goal: tuple[int, int],
        expanded: set[tuple[int, int]],
        obstacles: list[tuple[int, int, str]],
        width: int,
        height: int,
        max_bfs: int = 5000,
    ) -> list[tuple[int, int, str]]:
        if not obstacles:
            return []
        goal_x, goal_y = goal
        start = player
        if start == goal:
            return []
        visited: set[tuple[int, int]] = {start}
        queue = [start]
        idx = 0
        while idx < len(queue) and len(visited) < max_bfs:
            cx, cy = queue[idx]
            idx += 1
            if (cx, cy) == goal:
                return []
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                    if (nx, ny) in expanded:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        if goal in visited:
            return []
        blockers: list[tuple[int, int, str]] = []
        for ox, oy, otype in obstacles:
            if otype != "interactive":
                continue
            for cx, cy in visited:
                if abs(cx - ox) + abs(cy - oy) <= 2:
                    blockers.append((ox, oy, otype))
                    break
        logger.debug("find_blocking: visited=%d obstacles=%d interactive=%d blockers=%d",
                     len(visited), len(obstacles),
                     sum(1 for _,_,t in obstacles if t == "interactive"),
                     len(blockers))
        return blockers

    def _query_abstract_interaction(
        self,
        plan_context: dict[str, Any],
        player: tuple[int, int],
        blockers: list[tuple[int, int, str]],
        game_model: GameModel,
    ) -> tuple[int, int] | None:
        abstract_list = plan_context.get("abstract_memories", [])
        if not abstract_list or not blockers:
            return None
        px, py = player
        blocker_set = {(ox, oy) for ox, oy, _ in blockers}
        best_score = -1.0
        best_pos = None
        for mem in abstract_list:
            action = mem.get("action_name", "")
            if action != "ACTION6":
                continue
            motif_overlap = mem.get("motif_overlap", 0) or 0
            progress = mem.get("progress", 0) or 0
            failed = mem.get("failed", 0) or 0
            stalled = mem.get("stalled", 0) or 0
            total = mem.get("total", 0) or 0
            confidence = mem.get("confidence", 0.5) or 0.5
            success_bias = (progress * 1.0 - failed * 0.5 - stalled * 0.25) / max(1, total)
            source_games = mem.get("source_games", [])
            cross_game = len(source_games) if isinstance(source_games, list) else 0
            motif_tags = mem.get("motif_tags", [])
            is_interaction_memory = "interaction_step" in motif_tags if isinstance(motif_tags, list) else False
            interaction_bonus = 0.15 if is_interaction_memory else 0.0
            score = (motif_overlap * 0.25) + (success_bias * 0.35) + (confidence * 0.25) + (min(3, cross_game) * 0.05) + interaction_bonus
            # Parse lesson_text for stored click coordinates to check alignment
            lesson = mem.get("lesson", "")
            trained_coords = None
            if "at (" in lesson and ")" in lesson:
                try:
                    coord_part = lesson.split("at (")[-1].split(")")[0]
                    tx_str, ty_str = coord_part.split(",")
                    trained_coords = (int(tx_str.strip()), int(ty_str.strip()))
                except (ValueError, IndexError):
                    pass
            for ox, oy in blocker_set:
                dist = abs(ox - px) + abs(oy - py)
                alignment_bonus = 0.0
                if trained_coords is not None:
                    trained_dist = abs(ox - trained_coords[0]) + abs(oy - trained_coords[1])
                    if trained_dist <= 2:
                        alignment_bonus = 0.20
                click_score = score + alignment_bonus - dist * 0.02
                if click_score > best_score:
                    best_score = click_score
                    best_pos = (ox, oy)
        return best_pos

    def _try_edge_passage(
        self,
        player: tuple[int, int],
        goal: tuple[int, int],
        expanded: set[tuple[int, int]],
        game_model: GameModel,
    ) -> ActionDecision | None:
        px, py = player
        gx, gy = goal
        board = game_model.board
        h, w = game_model.height, game_model.width
        # BFS from player to find visited cells
        visited: set[tuple[int, int]] = {player}
        queue = [player]
        idx = 0
        while idx < len(queue) and len(visited) < 2000:
            cx, cy = queue[idx]
            idx += 1
            if (cx, cy) == goal:
                return None
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited and (nx, ny) in expanded:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        if goal in visited:
            return None
        # Find barrier cells: non-expanded cells adjacent to visited cells
        for vx, vy in visited:
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = vx + dx, vy + dy
                if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in expanded:
                    # Check if this barrier cell is at or near the bottom edge
                    # If barrier spans to within 2px of bottom, try padding row
                    if ny >= h - 3:
                        bottom_y = h - 1
                        bottom_expanded = set(expanded)
                        for bx in range(w):
                            cv = board[bottom_y][bx]
                            if cv != -1 and cv >= 0:
                                bottom_expanded.add((bx, bottom_y))
                        alt_path = astar(player, (gx, gy), bottom_expanded, w, h, max_steps=2000)
                        if alt_path and len(alt_path) > 1:
                            n = alt_path[1]
                            d = "right" if n[0] > px else ("left" if n[0] < px else ("down" if n[1] > py else "up"))
                            return ActionDecision(
                                action_name=_dir_to_action(d),
                                source="puzzle_system:edge",
                                confidence=0.60,
                                reason=f"edge_passage_{d}",
                            )
                    # Also check right edge
                    if nx >= w - 3:
                        right_x = w - 1
                        right_expanded = set(expanded)
                        for by in range(h):
                            cv = board[by][right_x]
                            if cv != -1 and cv >= 0:
                                right_expanded.add((right_x, by))
                        alt_path = astar(player, (gx, gy), right_expanded, w, h, max_steps=2000)
                        if alt_path and len(alt_path) > 1:
                            n = alt_path[1]
                            d = "right" if n[0] > px else ("left" if n[0] < px else ("down" if n[1] > py else "up"))
                            return ActionDecision(
                                action_name=_dir_to_action(d),
                                source="puzzle_system:edge",
                                confidence=0.60,
                                reason=f"edge_passage_{d}",
                            )
        return None

    def _try_expanded_path(
        self,
        player: tuple[int, int],
        goal: tuple[int, int],
        expanded: set[tuple[int, int]],
        game_model: GameModel,
        blockers: list[tuple[int, int, str]],
    ) -> tuple[list[tuple[int, int]] | None, tuple[int, int, int] | None]:
        if not blockers:
            return (None, None)
        expanded_after = set(expanded)
        for b in blockers:
            bx, by, _ = b
            # Add blocker cell and adjacent cells from any component
            expanded_after.add((bx, by))
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = bx + dx, by + dy
                if 0 <= nx < game_model.width and 0 <= ny < game_model.height:
                    expanded_after.add((nx, ny))
            # Also add the full component containing the blocker
            for c in game_model.components:
                if any(cell[0] == bx and cell[1] == by for cell in c.cells):
                    for cell in c.cells:
                        expanded_after.add(cell)
        path = astar(player, goal, expanded_after, game_model.width, game_model.height, max_steps=2000)
        if path and len(path) > 1:
            px, py = player
            best = min(blockers, key=lambda b: abs(b[0] - px) + abs(b[1] - py))
            return (path, (best[0], best[1], 6))
        return (None, None)

    def _local_search_step(
        self,
        player: tuple[int, int],
        goal: tuple[int, int],
        expanded: set[tuple[int, int]],
        game_model: GameModel,
    ) -> ActionDecision | None:
        px, py = player
        gx, gy = goal
        best_dist = abs(gx - px) + abs(gy - py)
        best_dir = None
        goff = {"up": (0,-1), "down": (0,1), "left": (-1,0), "right": (1,0)}
        for gd, (dx, dy) in goff.items():
            nx, ny = px + dx, py + dy
            if 0 <= nx < game_model.width and 0 <= ny < game_model.height and (nx, ny) in expanded:
                d2 = abs(gx - nx) + abs(gy - ny)
                if d2 < best_dist:
                    best_dist = d2
                    best_dir = gd
        if best_dir is not None:
            action = _dir_to_action(best_dir)
            nx2, ny2 = px + goff[best_dir][0], py + goff[best_dir][1]
            self._last_move_target = (nx2, ny2)
            return ActionDecision(
                action_name=action,
                source="puzzle_system:local_search",
                confidence=0.55,
                reason=f"local_search_{best_dir}",
            )
        return None

    def _navigate_to_goal(
        self,
        context: ExecutiveContext,
        game_model: GameModel,
        plan_context: dict[str, Any],
        goals: list[tuple[int, int]],
        player: tuple[int, int],
    ) -> ActionDecision:
        px, py = player
        if self._last_pos == (px, py):
            self._stall_frames += 1
        else:
            self._stall_frames = 0
            self._tried_interactions.clear()
        self._last_pos = (px, py)
        goal_history = plan_context.get("goal_history", [])
        goal_stability: dict[tuple[int, int], float] = {}
        for i, frame_goals in enumerate(goal_history):
            recency = 1.0 + 0.3 * (i / max(len(goal_history), 1))
            for g in frame_goals:
                for cg in goals:
                    if abs(g[0] - cg[0]) + abs(g[1] - cg[1]) <= 3:
                        goal_stability[cg] = goal_stability.get(cg, 0) + recency
        for g in goals:
            goal_stability.setdefault(g, 0)
        self._episode_steps += 1
        beam_cells = self._get_beam_cells(context, game_model)
        def goal_score(g):
            dist = abs(g[0] - px) + abs(g[1] - py)
            beam_bonus = 5.0 if game_model.features.has_beam and g in set(beam_cells) else 0.0
            return goal_stability.get(g, 0) * 10 - dist + beam_bonus
        nearest = max(goals, key=goal_score)
        gx, gy = nearest
        expanded = self._build_expanded_walkable(game_model, context, plan_context)
        astar_path = astar(player, (gx, gy), expanded, game_model.width, game_model.height, max_steps=2000)
        if astar_path and len(astar_path) > 1:
            nx, ny = astar_path[1]
            d = "right" if nx > px else ("left" if nx < px else ("down" if ny > py else "up"))
            # Detect oscillation: if alternating between up/down, try edge passage
            if self._nav_last_dir and d in ("up", "down") and self._nav_last_dir in ("up", "down") and d != self._nav_last_dir:
                self._nav_oscillation += 1
            else:
                self._nav_oscillation = 0
            self._nav_last_dir = d
            if self._nav_oscillation >= 2:
                self._nav_oscillation = 0
                # Clear impassable cells on oscillation to reset blocked paths
                self._impassable_cells.clear()
                # Run edge passage, local search, and blocker detection in parallel
                from agi_core.orch_or.parallel import ThreadPoolManager
                try:
                    pool = ThreadPoolManager.executor()
                    bfs_future = pool.submit(self._try_edge_passage, player, (gx, gy), expanded, game_model)
                    local_future = pool.submit(self._local_search_step, player, gx, gy, expanded, game_model)
                    # Process results as they complete
                    edge_result = bfs_future.result(timeout=0.5)
                    if edge_result is not None:
                        return edge_result
                    local_result = local_future.result(timeout=0.3)
                    if local_result is not None:
                        return local_result
                except Exception:
                    # Fallback to sequential
                    edge_result = self._try_edge_passage(player, (gx, gy), expanded, game_model)
                    if edge_result is not None:
                        return edge_result
                    local_result = self._local_search_step(player, gx, gy, expanded, game_model)
                    if local_result is not None:
                        return local_result
                plan_context["switch_attempted"] = plan_context.get("switch_attempted", 0) + 1
            action = _dir_to_action(d)
            self._last_move_target = (nx, ny)
            self._store_state(context, game_model, [], action, "navigate")
            return ActionDecision(
                action_name=action,
                source="puzzle_system:navigate",
                confidence=0.75,
                reason=f"move_{d}",
            )
        # Phase E: Try edge passage around barriers that span the board
        edge_decision = self._try_edge_passage(player, (gx, gy), expanded, game_model)
        if edge_decision is not None:
            return edge_decision
        # Phase S: Sprite-based entity targeting — click on non-player selectable entities
        if game_model.sprites:
            clickable = [s for s in game_model.sprites if s.is_clickable and not s.is_player]
            clickable.sort(key=lambda s: abs(s.pixel_x - px) + abs(s.pixel_y - py))
            board = game_model.board
            for s in clickable:
                # Find a non-transparent pixel within the sprite's area
                tx, ty = s.pixel_x, s.pixel_y
                found = False
                for dy in range(s.grid_h * 3):
                    for dx in range(s.grid_w * 3):
                        cy = s.pixel_y + dy
                        cx = s.pixel_x + dx
                        if cy < len(board) and cx < len(board[cy]) and board[cy][cx] == s.color:
                            tx, ty = cx, cy
                            found = True
                            break
                    if found:
                        break
                target = (tx, ty)
                if (target[0], target[1], "interactive") not in self._tried_interactions:
                    self._tried_interactions.add((target[0], target[1], "interactive"))
                    self._store_state(context, game_model, [], "ACTION6", "click_obstacle")
                    return ActionDecision(
                        action_name="ACTION6",
                        source="puzzle_system:sprite",
                        confidence=0.65,
                        reason=f"select_entity_at_{s.sprite_id}",
                        metadata={"x": target[0], "y": target[1]},
                    )
        # Phase 1: A* failed — find blocking interactive obstacles
        blockers = self._find_blocking_obstacles(
            player, (gx, gy), expanded,
            game_model.obstacles,
            game_model.width, game_model.height,
        )
        if blockers:
            # Phase 4: Query cross-game abstract interaction memory
            abs_target = self._query_abstract_interaction(plan_context, player, blockers, game_model)
            if abs_target is not None:
                ax, ay = abs_target
                # Skip if this position was already tried
                if (ax, ay, "interactive") not in self._tried_interactions:
                    self._tried_interactions.add((ax, ay, "interactive"))
                    self._store_state(context, game_model, [], "ACTION6", "click_obstacle")
                    return ActionDecision(
                        action_name="ACTION6",
                        source="puzzle_system:abstract",
                        confidence=0.60,
                        reason="abstract_interaction_target",
                        metadata={"x": ax, "y": ay},
                    )
            # Filter out already-tried interactions
            fresh = [(ox, oy, ot) for ox, oy, ot in blockers
                     if (ox, oy, ot) not in self._tried_interactions]
            if fresh:
                nearest = min(fresh, key=lambda b: abs(b[0] - px) + abs(b[1] - py))
                nx, ny, _ = nearest
                self._tried_interactions.add(nearest)
                self._store_state(context, game_model, [], "ACTION6", "click_obstacle")
                return ActionDecision(
                    action_name="ACTION6",
                    source="puzzle_system:obstacle",
                    confidence=0.50,
                    reason="interact_blocking_obstacle",
                    metadata={"x": nx, "y": ny},
                )
        # Phase 2: Query interaction memory for known-effective click targets
        mem_target = query_best_interaction(
            self._memory_db_path,
            context.observation.game_id,
            px, py,
            game_model.obstacles,
        )
        if mem_target is not None:
            mx, my = mem_target
            if abs(mx - px) + abs(my - py) <= 3:
                self._store_state(context, game_model, [], "ACTION6", "memory_interaction")
                return ActionDecision(
                    action_name="ACTION6",
                    source="puzzle_system:interact_memory",
                    confidence=0.60,
                    reason="learned_interaction_target",
                    metadata={"x": mx, "y": my},
                )
        if self._stall_frames >= 3:
            all_dirs = ["up", "down", "left", "right"]
            self._stall_direction_index = (self._stall_direction_index + 1) % 4
            fallback_dir = all_dirs[self._stall_direction_index]
            action = _dir_to_action(fallback_dir)
            self._store_state(context, game_model, [], action, "stall_escape")
            return ActionDecision(
                action_name=action,
                source="puzzle_system:stall_escape",
                confidence=0.50,
                reason=f"stall_escape_{fallback_dir}",
            )
        for ox, oy, otype in sorted(game_model.obstacles, key=lambda o: abs(o[0] - px) + abs(o[1] - py)):
            if otype != "interactive":
                continue
            comp = next((c for c in game_model.components if c.bbox[0] <= ox <= c.bbox[2] and c.bbox[1] <= oy <= c.bbox[3]), None)
            if comp is not None:
                click_cell = min(comp.cells, key=lambda cell: abs(cell[0] - px) + abs(cell[1] - py))
                cx2, cy2 = click_cell
            else:
                cx2, cy2 = ox, oy
            if abs(cx2 - px) + abs(cy2 - py) <= 2:
                self._store_state(context, game_model, [], "ACTION6", "click_obstacle")
                return ActionDecision(
                    action_name="ACTION6",
                    source="puzzle_system:interact",
                    confidence=0.70,
                    reason="clear_obstacle",
                    metadata={"x": cx2, "y": cy2},
                )
        return self._fallback(context)

    def _query_memory(
        self,
        context: ExecutiveContext,
        goals: list[tuple[int, int]],
    ) -> dict[str, Any] | None:
        cached = load_puzzle_solution(
            self._memory_db_path,
            context.observation.game_id,
            goals,
        )
        if cached is not None:
            return cached
        fuzzy = query_puzzle_solution_fuzzy(
            self._memory_db_path,
            context.observation.game_id,
            goals,
        )
        if fuzzy is not None:
            return fuzzy
        similar = query_similar_state(
            self._memory_db_path,
            context.observation.game_id,
            goals,
        )
        if similar is not None:
            return similar
        # Dynamic cross-game transfer: try all games in memory
        try:
            from pathlib import Path
            import sqlite3
            from contextlib import closing
            db = Path(self._memory_db_path) if self._memory_db_path else None
            if db and db.exists():
                with closing(sqlite3.connect(db)) as conn:
                    game_ids = [r[0] for r in conn.execute(
                        "SELECT DISTINCT game_id FROM puzzle_board_states WHERE outcome='progress' ORDER BY step DESC LIMIT 20"
                    ).fetchall() if r[0] != context.observation.game_id]
                for gid in game_ids[:5]:
                    cross = query_similar_state(self._memory_db_path, gid, goals)
                    if cross is not None:
                        return cross
        except Exception:
            pass
        return None

    def _action_from_memory(
        self,
        context: ExecutiveContext,
        cached: dict[str, Any],
        game_model: GameModel,
        plan_context: dict[str, Any] | None = None,
    ) -> ActionDecision:
        action = cached.get("action", "")
        if action:
            return ActionDecision(
                action_name=action,
                source=f"puzzle_system:{cached.get('source', 'memory')}",
                confidence=float(cached.get("confidence", 0.7)),
                reason=cached.get("outcome", "memory_recall"),
            )
        failed = (plan_context or {}).get("failed_directions", set())
        mirror_x = cached.get("mirror_target_x")
        if mirror_x is not None and mirror_x >= 0 and game_model.player_pos is not None:
            px, _ = game_model.player_pos
            if abs(px - mirror_x) <= 3:
                # Instead of returning ACTION5 (which loops), fall through to
                # navigate_to_goal or fallback to avoid infinite switch loops.
                return self._fallback(context)
            dir = "left" if px > mirror_x else "right"
            if dir in failed:
                dir = "right" if dir == "left" else "left"
            if dir not in failed:
                return ActionDecision(
                    action_name=_dir_to_action(dir),
                    source="puzzle_system:cached",
                    confidence=0.85,
                    reason=f"cached_mirror_{dir}",
                )
        return self._fallback(context)

    def _fallback(self, context: ExecutiveContext, game_model: GameModel | None = None) -> ActionDecision:
        if game_model is not None and game_model.player_pos is not None:
            px, py = game_model.player_pos
            for ox, oy, otype in game_model.obstacles:
                if otype == "interactive":
                    self._store_state(context, game_model, [], "ACTION6", "fallback_click")
                    return ActionDecision(
                        action_name="ACTION6",
                        source="puzzle_system:fallback",
                        confidence=0.30,
                        reason="puzzle_system_interact_fallback",
                        metadata={"x": ox, "y": oy},
                    )
        # Try action priors from plan_context before deterministic fallback
        action_priors = (self._action_history or {})
        if action_priors:
            best_action = max(action_priors, key=lambda a: action_priors[a])
            if best_action in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"):
                return ActionDecision(
                    action_name=best_action,
                    source="puzzle_system:fallback_prior",
                    confidence=0.20,
                    reason=f"fallback_prior_{best_action}",
                )
        return ActionDecision(
            action_name=context.config.deterministic_fallback_action,
            source="puzzle_system:fallback",
            confidence=0.05,
            reason="puzzle_system_no_action",
            used_fallback=True,
        )

    def _store_state(
        self,
        context: ExecutiveContext,
        game_model: GameModel,
        beam_cells: list[tuple[int, int]],
        action_taken: str,
        outcome: str,
    ) -> None:
        if action_taken:
            if outcome in ("navigate", "mirror_move", "mirror_move_hyperon", "click_obstacle", "approach_obstacle"):
                self._action_history[action_taken] = self._action_history.get(action_taken, 0) + 1
            if self._last_pos is not None and game_model.player_pos is not None:
                if game_model.player_pos != self._last_pos:
                    for k in list(self._action_history.keys()):
                        self._action_history[k] = max(0, self._action_history[k] - 1)
        if self._memory_db_path is None:
            return
        player = game_model.player_pos
        state = PuzzleBoardState(
            game_id=context.observation.game_id,
            step=context.observation.step_index,
            player_x=player[0] if player else -1,
            player_y=player[1] if player else -1,
            goal_positions=game_model.goal_positions,
            beam_cells=beam_cells,
            mirror_x=player[0] if player else -1,
            goals_lit=game_model.features.goals_lit_by_beam,
            action_taken=action_taken,
            outcome=outcome,
        )
        store_board_state(self._memory_db_path, state)

    def _get_beam_cells(self, context: ExecutiveContext, game_model: GameModel) -> list[tuple[int, int]]:
        beam_color = game_model.features.beam_color
        if beam_color < 0:
            return game_model.features.beam_positions
        all_cells: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for c in game_model.components:
            if c.dominant_color == beam_color:
                for cell in c.cells:
                    if cell not in seen:
                        seen.add(cell)
                        all_cells.append(cell)
        if all_cells:
            return all_cells
        beam_cells = self._scan_board_for_beam(context, beam_color)
        if beam_cells:
            return beam_cells
        return game_model.features.beam_positions

    def _scan_board_for_beam(self, context: ExecutiveContext, beam_color: int) -> list[tuple[int, int]]:
        try:
            from agi_core.orch_or.orchestrator import _get_board_data
            board = _get_board_data(context.observation)
            if not board:
                return []
            h, w = len(board), len(board[0])
            cells = [(x, y) for y in range(h) for x in range(w) if board[y][x] == beam_color]
            return cells
        except Exception:
            return []

    @staticmethod
    def _is_beam_controller_pos(
        player: tuple[int, int] | None,
        features: Any,
    ) -> bool:
        if player is None or not features.has_beam or not features.beam_positions:
            return False
        px, py = player
        pc = features.player_color
        if pc <= 0 or pc == features.background_color or pc == features.beam_color:
            return False
        return any(abs(bx - px) <= 3 and abs(by - py) <= 3
                   for bx, by in features.beam_positions)


def _dir_to_action(direction: str) -> str:
    return {"up": "ACTION1", "down": "ACTION2", "left": "ACTION3", "right": "ACTION4"}.get(direction, "ACTION1")
