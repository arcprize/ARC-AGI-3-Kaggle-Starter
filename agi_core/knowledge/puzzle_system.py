from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agi_core.knowledge.beam_solver import solve_beam_puzzle
from agi_core.knowledge.puzzle_memory import (
    PuzzleBoardState,
    initialize_puzzle_schema,
    load_puzzle_solution,
    query_similar_state,
    save_puzzle_solution,
    store_board_state,
)
from agi_core.orch_or.game_model import PlanStep
from agi_core.types import ActionDecision, ActionProposal

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
        if features.board_type == "puzzle" and (has_interactive or features.path_to_goal_possible):
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
        beam_cells = self._get_beam_cells(game_model)

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

        is_beam_controller = (
            features.has_beam
            and features.player_color >= 0
            and features.player_color not in non_beam_set
            and self._is_beam_controller_pos(player, features)
        )

        if is_beam_controller and features.mirror_axis >= 0 and beam_cells:
            return self._handle_beam_controller(
                context, game_model, plan_context, beam_cells, px, py,
            )

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
        solver_result = solve_beam_puzzle(
            goal_positions=goals,
            beam_cells=beam_cells,
            mirror_axis=mirror_pos,
            player_pos=(px, py),
            current_mirror_pos=(px, py),
            width=game_model.width,
            height=game_model.height,
            memory_db_path=self._memory_db_path,
            game_id=context.observation.game_id,
        )
        mirror_dir = solver_result.get("mirror_direction")
        if mirror_dir:
            action = _dir_to_action(mirror_dir)
            self._store_state(context, game_model, beam_cells, action, "mirror_move")
            return ActionDecision(
                action_name=action,
                source="puzzle_system:beam",
                confidence=0.85,
                reason=f"aim_mirror_{mirror_dir}",
            )

        self._mirror_frames += 1
        switch_count = int(plan_context.get("switch_attempted", 0))
        uncovered = solver_result.get("uncovered_goals", [])
        near_uncovered = any(abs(gx - px) + abs(gy - py) <= 8 for gx, gy in uncovered) if uncovered else True
        mirror_timeout = self._mirror_frames >= 15
        if ((switch_count < 2 and ((features.goals_lit_by_beam > 0 and near_uncovered) or mirror_timeout))):
            plan_context["switch_attempted"] = switch_count + 1
            self._store_state(context, game_model, beam_cells, "ACTION5", "switch_from_mirror")
            save_puzzle_solution(
                self._memory_db_path,
                context.observation.game_id,
                goals,
                mirror_target_x=px,
                switch_sequence=["ACTION5"],
                confidence=0.7,
            )
            return ActionDecision(
                action_name="ACTION5",
                source="puzzle_system:beam",
                confidence=0.85,
                reason="mirror_positioned_switch_back",
            )

        return self._fallback(context)

    def _build_expanded_walkable(
        self, game_model: GameModel, context: ExecutiveContext, plan_context: dict[str, Any],
    ) -> set[tuple[int, int]]:
        gid = context.observation.game_id
        if self._expanded_walkable_cache is not None and self._cache_game_id == gid:
            return self._expanded_walkable_cache
        w = set(game_model.walkable)
        for c in game_model.components:
            if not c.touches_border:
                for cx, cy in c.cells:
                    w.add((cx, cy))
        self._expanded_walkable_cache = w
        self._cache_game_id = gid
        return w

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
        self._last_pos = (px, py)
        goal_history = plan_context.get("goal_history", [])
        goal_stability: dict[tuple[int, int], int] = {}
        for frame_goals in goal_history:
            for g in frame_goals:
                for cg in goals:
                    if abs(g[0] - cg[0]) + abs(g[1] - cg[1]) <= 3:
                        goal_stability[cg] = goal_stability.get(cg, 0) + 1
        for g in goals:
            goal_stability.setdefault(g, 0)
        def goal_score(g):
            dist = abs(g[0] - px) + abs(g[1] - py)
            return goal_stability.get(g, 0) * 10 - dist
        nearest = max(goals, key=goal_score)
        gx, gy = nearest
        dx, dy = gx - px, gy - py
        _DIR_TO_ACTION = {"up": "ACTION1", "down": "ACTION2", "left": "ACTION3", "right": "ACTION4"}
        action_priors = plan_context.get("action_priors", {})
        candidates = []
        if dx > 0: candidates.append("right")
        elif dx < 0: candidates.append("left")
        if dy > 0: candidates.append("down")
        elif dy < 0: candidates.append("up")
        if abs(dx) == abs(dy) and len(candidates) >= 2:
            pri_h = action_priors.get(_DIR_TO_ACTION.get(candidates[0], ""), 0.0)
            pri_v = action_priors.get(_DIR_TO_ACTION.get(candidates[1], ""), 0.0)
            if pri_v > pri_h:
                candidates = candidates[::-1]
        for d_name, d_prior in action_priors.items():
            act_dir = {"ACTION1": "up", "ACTION2": "down", "ACTION3": "left", "ACTION4": "right"}.get(d_name)
            if act_dir and act_dir in candidates and d_prior > 0.8:
                candidates.remove(act_dir)
                candidates.insert(0, act_dir)
        failed = plan_context.get("failed_directions", set())
        candidates = [d for d in candidates if self._action_history.get(d, 0) < 3]
        expanded_walkable = self._build_expanded_walkable(game_model, context, plan_context)
        obs_walkable = set(expanded_walkable)
        for c in game_model.components:
            if not c.touches_border:
                for cx, cy in c.cells:
                    obs_walkable.add((cx, cy))
        for d in candidates:
            if d in failed: continue
            off = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
            dvx, dvy = off.get(d, (0, 0))
            nx, ny = px + dvx, py + dvy
            if (nx, ny) in expanded_walkable:
                action = _dir_to_action(d)
                self._store_state(context, game_model, [], action, "navigate")
                return ActionDecision(
                    action_name=action,
                    source="puzzle_system:navigate",
                    confidence=0.75,
                    reason=f"move_{d}",
                )
        if self._stall_frames >= 3:
            all_dirs = ["up", "down", "left", "right"]
            self._stall_direction_index = (self._stall_direction_index + 1) % 4
            fallback_dir = all_dirs[self._stall_direction_index]
            if fallback_dir not in failed:
                self._store_state(context, game_model, [], _dir_to_action(fallback_dir), "stall_escape")
                return ActionDecision(
                    action_name=_dir_to_action(fallback_dir),
                    source="puzzle_system:stall_escape",
                    confidence=0.50,
                    reason=f"stall_escape_{fallback_dir}",
                )
        if action_priors:
            best_action = max(action_priors, key=action_priors.get)
            best_dir = {"ACTION1": "up", "ACTION2": "down", "ACTION3": "left", "ACTION4": "right"}.get(best_action)
            if best_dir and best_dir not in failed:
                self._store_state(context, game_model, [], best_action, "prior_fallback")
                return ActionDecision(
                    action_name=best_action,
                    source="puzzle_system:prior",
                    confidence=0.55,
                    reason=f"prior_fallback_{best_dir}",
                )
        from agi_core.orch_or.search import astar
        astar_path = astar(player, (gx, gy), obs_walkable, game_model.width, game_model.height)
        if astar_path and len(astar_path) > 1:
            nx, ny = astar_path[1]
            fx, fy = astar_path[0]
            d = "right" if nx > fx else ("left" if nx < fx else ("down" if ny > fy else "up"))
            if d not in failed:
                self._store_state(context, game_model, [], _dir_to_action(d), "astar_navigate")
                return ActionDecision(
                    action_name=_dir_to_action(d),
                    source="puzzle_system:astar",
                    confidence=0.70,
                    reason=f"astar_{d}",
                )
        for ox, oy, otype in sorted(game_model.obstacles, key=lambda o: abs(o[0] - px) + abs(o[1] - py)):
            if otype != "interactive":
                continue
            if abs(ox - px) + abs(oy - py) <= 2:
                self._store_state(context, game_model, [], "ACTION6", "click_obstacle")
                return ActionDecision(
                    action_name="ACTION6",
                    source="puzzle_system:interact",
                    confidence=0.70,
                    reason="clear_obstacle",
                    metadata={"x": ox, "y": oy},
                )
            adj = [(ox + dx, oy + dy) for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)) if (ox + dx, oy + dy) in expanded_walkable]
            if adj:
                adj_goal = min(adj, key=lambda t: abs(t[0] - px) + abs(t[1] - py))
                from agi_core.orch_or.search import astar
                path = astar(player, adj_goal, expanded_walkable, game_model.width, game_model.height)
                if path and len(path) > 1:
                    step_idx = 0
                    steps = []
                    while step_idx < len(path) - 1:
                        next_idx = min(step_idx + 3, len(path) - 1)
                        prev = path[step_idx]
                        curr = path[next_idx]
                        fx, fy = prev
                        tx, ty = curr
                        d = "right" if tx > fx else ("left" if tx < fx else ("down" if ty > fy else "up"))
                        steps.append(PlanStep(
                            action="move", params={"direction": d},
                            source="puzzle_system", preconditions=frozenset(),
                            expected_effect=f"move_{d}",
                        ))
                        step_idx = next_idx
                    if steps:
                        self._store_state(context, game_model, [], "ACTION6", "approach_obstacle")
                        return ActionDecision(
                            action_name=steps[0].params["direction"],
                            source="puzzle_system:interact",
                            confidence=0.70,
                            reason="approach_obstacle",
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
        similar = query_similar_state(
            self._memory_db_path,
            context.observation.game_id,
            goals,
        )
        if similar is not None:
            return similar
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
                return ActionDecision(
                    action_name="ACTION5",
                    source="puzzle_system:cached",
                    confidence=0.85,
                    reason="cached_solution_switch_back",
                )
            dir = "left" if px > mirror_x else "right"
            if dir in failed:
                dir = "right" if dir == "left" else "left"
            if dir in failed:
                return self._fallback(context)
            return ActionDecision(
                action_name=_dir_to_action(dir),
                source="puzzle_system:cached",
                confidence=0.85,
                reason=f"cached_mirror_{dir}",
            )
        return self._fallback(context)

    def _fallback(self, context: ExecutiveContext) -> ActionDecision:
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

    @staticmethod
    def _get_beam_cells(game_model: GameModel) -> list[tuple[int, int]]:
        beam_color = game_model.features.beam_color
        if beam_color < 0:
            return game_model.features.beam_positions
        for c in game_model.components:
            if c.dominant_color == beam_color and not c.touches_border:
                return c.cells
        return game_model.features.beam_positions

    @staticmethod
    def _is_beam_controller_pos(
        player: tuple[int, int] | None,
        features: Any,
    ) -> bool:
        if player is None or not features.has_beam or not features.beam_positions:
            return False
        px, py = player
        pc = features.player_color
        if pc <= 0 or pc == features.background_color or pc == features.beam_color or pc == 5:
            return False
        return any(abs(bx - px) <= 3 and abs(by - py) <= 3
                   for bx, by in features.beam_positions)


def _dir_to_action(direction: str) -> str:
    return {"up": "ACTION1", "down": "ACTION2", "left": "ACTION3", "right": "ACTION4"}.get(direction, "ACTION1")
