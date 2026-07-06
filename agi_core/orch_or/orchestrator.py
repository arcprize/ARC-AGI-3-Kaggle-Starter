from __future__ import annotations

import logging
from concurrent.futures import TimeoutError
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

from agi_core.orch_or.analyzer import build_game_model
from agi_core.orch_or.game_model import (
    Plan,
    PlanStep,
    deserialize_plan,
    serialize_plan,
)
from agi_core.orch_or.monitor import check_progress
from agi_core.orch_or.parallel import ThreadPoolManager
from agi_core.orch_or.plan_executor import (
    plan_step_to_proposal,
)
from agi_core.orch_or.plan_generator import generate_plan
from agi_core.types import ActionDecision, ActionProposal

if TYPE_CHECKING:
    from agi_core.orch_or.game_model import GameModel, MonitorStatus
    from agi_core.types import ExecutiveContext


_PLAN_KEY = "orch_or_plan"
_SWITCH_KEY = "orch_or_switch_attempted"
_SWITCH_POS_KEY = "orch_or_player_pos_before_switch"
_BEAM_COVERAGE_KEY = "orch_or_beam_coverage"
_BEAM_DIR_KEY = "orch_or_beam_best_direction"
_BEAM_PROX_KEY = "orch_or_beam_proximity"
_BEAM_COLOR_KEY = "orch_or_beam_controller_color"
_LEVEL_JUMP_THRESHOLD = 35


class ObjectiveReduction:
    _MAX_REGENERATE = 3

    def __init__(self, executive, config) -> None:
        self._executive = executive
        self._config = config
        self._active_plan: Plan | None = None
        self._previous_model: GameModel | None = None
        self._regenerate_count: int = 0
        self._skip_monitoring: bool = False
        self._switch_attempted: bool = False
        self._switch_pos: tuple[int, int] | None = None
        self._non_beam_controllers: set[int] = set()
        self._failed_directions: set[str] = set()
        self._pending_failed_dir: str | None = None
        self._stable_goal: tuple[int, int] | None = None
        self._hyperon_priors: dict[str, float] = {}

    def decide(self, context: ExecutiveContext) -> ActionDecision:
        context.hyperon.sync(context.observation, context.memory, context.goals)
        context.memory_bundle = context.hyperon.current_memory_bundle()
        self._load_plan(context)
        board_data = _get_board_data(context.observation)
        if not board_data:
            logger.debug("no board data, falling back to executive")
            return self._fallback_executive(context)
        game_model = build_game_model(board_data)
        # Feed board_type into hyperon facts so game-action-priors can use it.
        try:
            if hasattr(context.hyperon, "facts"):
                context.hyperon.facts.setdefault("episode", {})["board_type"] = game_model.features.board_type
        except Exception:
            pass
        logger.debug(
            "game_model: player=%s goals=%s features=pd:%s gd:%s ptg:%s bt:%s",
            game_model.player_pos, game_model.goal_positions,
            game_model.features.player_detectable,
            game_model.features.goal_detectable,
            game_model.features.path_to_goal_possible,
            game_model.features.board_type,
        )

        # Clear per-level state when a new level starts (player jumps far)
        if self._previous_model is not None and self._previous_model.player_pos is not None and game_model.player_pos is not None:
            prev_p = self._previous_model.player_pos
            curr_p = game_model.player_pos
            if abs(curr_p[0] - prev_p[0]) + abs(curr_p[1] - prev_p[1]) > _LEVEL_JUMP_THRESHOLD:
                self._non_beam_controllers.clear()
                self._switch_attempted = False
                self._failed_directions.clear()
                self._stable_goal = None
        elif self._previous_model is None:
            self._non_beam_controllers.clear()
            self._switch_attempted = False
            self._failed_directions.clear()
            self._stable_goal = None
        # Track a stable goal from platformer boards — these persist even when
        # puzzle boards change goals dynamically.
        # For non-beam-controlling entities (after a switch), the stable goal
        # is locked once set so dynamic puzzle goals don't override it.
        is_non_controller = game_model.features.player_color in self._non_beam_controllers
        if game_model.features.board_type == "platformer" and game_model.features.goal_detectable and game_model.goal_positions and game_model.player_pos is not None:
            if not is_non_controller or self._stable_goal is None:
                px, py = game_model.player_pos
                nearest = min(game_model.goal_positions, key=lambda g: abs(g[0] - px) + abs(g[1] - py))
                self._stable_goal = nearest
        elif self._stable_goal is None and game_model.goal_positions:
            self._stable_goal = game_model.goal_positions[0]

        # --- Hyperon integration: query rules for decision support ---
        hyperon_priors: dict[str, float] = {}
        subgoal_from_hints: str | None = None
        # Query beam strategy hints (game/level-specific direction bias).
        beam_strategy: list[dict[str, Any]] = []
        try:
            beam_strategy = context.hyperon.run_rules("beam-strategy") or []
        except Exception:
            pass
        try:
            # Check for reset-needed
            for row in context.hyperon.run_rules("reset-needed"):
                if row.get("action_name"):
                    logger.debug("hyperon reset-needed: %s", row)
                    return ActionDecision(
                        action_name=row["action_name"],
                        source="orch_or:hyperon",
                        confidence=float(row.get("confidence", 0.95)),
                        reason=str(row.get("reason", "hyperon reset")),
                        subgoal="re-enter-level",
                        proposals=[ActionProposal(
                            action_name=row["action_name"],
                            source="orch_or:hyperon",
                            confidence=float(row.get("confidence", 0.95)),
                            reason=str(row.get("reason", "hyperon reset")),
                            subgoal="re-enter-level",
                        )],
                    )
            # Query game-aware goal hints (factors in board_type).
            goal_hints = context.hyperon.run_rules("game-goal-hints")
            if goal_hints:
                logger.debug("hyperon game-goal-hints: %s", goal_hints)
                subgoal_from_hints = goal_hints[0].get("subgoal")
            # Query action priors for current subgoal
            prior_query = "game-action-priors"
            if subgoal_from_hints:
                prior_query = f"game-action-priors:{subgoal_from_hints}"
            for row in context.hyperon.run_rules(prior_query):
                action_name = str(row.get("action_name", "")).upper()
                if not action_name:
                    continue
                hyperon_priors[action_name] = float(row.get("confidence", 0.5))
            if hyperon_priors:
                logger.debug("hyperon game-action-priors: %s", hyperon_priors)
        except Exception:
            logger.debug("hyperon rules query failed (disabled?)", exc_info=True)
        self._hyperon_priors = hyperon_priors

        logger.debug("orchestrator state: switch_attempted=%s non_beam=%s stable_goal=%s", self._switch_attempted, self._non_beam_controllers, self._stable_goal)
        plan_context: dict[str, Any] = {
            "game_step": context.observation.step_index,
            "switch_attempted": self._switch_attempted,
            "non_beam_controllers": self._non_beam_controllers.copy(),
            "failed_directions": self._failed_directions.copy(),
            "stable_goal": self._stable_goal,
            "action_priors": hyperon_priors,
            "beam_strategy": beam_strategy,
        }
        if subgoal_from_hints:
            plan_context["goal_hints"] = goal_hints
        _update_beam_context(game_model, self._previous_model, self._get_working_memory(context), plan_context)

        if not self._skip_monitoring:
            self._check_previous_action(game_model, context)
            if self._active_plan is not None:
                self._persist_plan(context)
        self._skip_monitoring = False

        # --- Hyperon integration: contextual memory bundle ---
        try:
            if context.memory_bundle is not None and context.observation is not None:
                legal = list(context.observation.legal_actions) if hasattr(context.observation, "legal_actions") else []
                ctx_bundle = context.hyperon.contextual_memory_bundle(
                    context.observation, subgoal_from_hints, legal,
                )
                if ctx_bundle is not None:
                    context.memory_bundle = ctx_bundle
        except Exception:
            logger.debug("hyperon contextual bundle failed", exc_info=True)

        # Write structured working memory fields that hyperon can consume.
        wm = self._get_working_memory(context)
        wm["active_subgoal"] = subgoal_from_hints or "unknown"
        wm["stable_goal"] = self._stable_goal
        wm["switch_attempted"] = self._switch_attempted
        wm["failed_directions"] = list(self._failed_directions)
        if hyperon_priors:
            wm["recommended_actions"] = sorted(hyperon_priors, key=hyperon_priors.get, reverse=True)
        plan_context["action_priors"] = hyperon_priors

        if self._active_plan is None or not self._active_plan.is_active:
            parallel = bool(getattr(self._config, "planner_parallel", False))
            p_timeout = getattr(self._config, "planner_parallel_timeout", None)
            self._active_plan = generate_plan(
                game_model,
                context=plan_context,
                parallel=parallel,
                timeout=p_timeout,
            )
            if self._active_plan is not None:
                logger.debug("plan generated: id=%s steps=%d", self._active_plan.objective_id, len(self._active_plan.steps))
                if any(s.action == "switch" for s in self._active_plan.steps):
                    self._switch_attempted = True
                self._persist_plan(context)
                self._regenerate_count = 0
            else:
                logger.debug("generate_plan returned None")
        if self._active_plan is None or not self._active_plan.steps:
            logger.debug("no active plan (plan=%s steps=%s), falling back", self._active_plan is not None, len(self._active_plan.steps) if self._active_plan else "N/A")
            if self._stable_goal is not None and game_model.player_pos is not None and game_model.features.player_color in self._non_beam_controllers:
                px, py = game_model.player_pos
                gx, gy = self._stable_goal
                dx, dy = gx - px, gy - py
                if abs(dx) >= abs(dy):
                    direction = "right" if dx > 0 else ("left" if dx < 0 else "down")
                else:
                    direction = "down" if dy > 0 else ("up" if dy < 0 else "right")
                step = PlanStep(
                    action="move", params={"direction": direction},
                    source="orch_or:stable_goal", preconditions=frozenset(),
                    expected_effect=f"move_{direction}",
                )
                self._active_plan = Plan(
                    objective_id="reach-goal", steps=[step],
                    created_at_step=context.observation.step_index,
                    confidence=0.5,
                )
                self._persist_plan(context)
                self._skip_monitoring = True
                return self.decide(context)
            try:
                context.hyperon.update_proposals([])
            except Exception:
                pass
            return self._fallback_executive(context)
        while True:
            self._active_plan.attempt_count += 1
            current_step_index = self._active_plan.current_index
            if current_step_index >= len(self._active_plan.steps):
                return self._handle_plan_exhausted(context, game_model)
            step = self._active_plan.steps[current_step_index]
            proposal = plan_step_to_proposal(step, game_model, self._active_plan)
            if proposal is not None:
                # Boost proposal confidence from hyperon action priors.
                hyperon_action_prior = hyperon_priors.get(proposal.action_name, 0.0)
                if hyperon_action_prior > 0.0:
                    proposal.confidence = min(proposal.confidence + hyperon_action_prior * 0.25, 0.95)
                logger.debug("proposal resolved: action=%s conf=%.3f", proposal.action_name, proposal.confidence)
                break
            logger.debug("step %d returned None proposal, skipping", current_step_index)
            self._active_plan.current_index += 1
            self._persist_plan(context)
        decision = ActionDecision(
            action_name=proposal.action_name,
            source=proposal.source,
            confidence=proposal.confidence,
            reason=proposal.reason,
            subgoal=proposal.subgoal,
            metadata=proposal.metadata,
            proposals=[proposal],
        )
        context.memory.note_decision(decision)
        self._persist_plan(context)
        # Feed the proposals back to hyperon so the MeTTa runtime can learn
        # about the actions ORCH-OR considered.
        try:
            context.hyperon.update_proposals(decision.proposals)
        except Exception:
            pass
        return decision

    def _check_previous_action(
        self, game_model: GameModel, context: ExecutiveContext,
    ) -> None:
        if self._active_plan is not None and self._active_plan.steps:
            if self._active_plan.current_index < len(self._active_plan.steps):
                status = check_progress(
                    self._active_plan, game_model, self._previous_model,
                )
                self._handle_status(status, context, game_model)
        self._previous_model = game_model

    def _handle_plan_exhausted(
        self, context: ExecutiveContext, game_model,
    ) -> ActionDecision:
        if self._regenerate_count >= self._MAX_REGENERATE:
            self._regenerate_count = 0
            wm = self._get_working_memory(context)
            wm.pop(_SWITCH_KEY, None)
            wm.pop(_SWITCH_POS_KEY, None)
            # Query hyperon for game-aware action priors to guide the fallback direction.
            hyperon_priors: dict[str, float] = {}
            try:
                for row in context.hyperon.run_rules("game-action-priors"):
                    action_name = str(row.get("action_name", "")).upper()
                    if action_name:
                        hyperon_priors[action_name] = float(row.get("confidence", 0.5))
            except Exception:
                pass
            # Instead of falling back to the reactive executive (which may give
            # a conflicting direction), emit a simple move toward the stable
            # goal for non-beam-controllers.
            if self._stable_goal is not None and game_model.player_pos is not None and game_model.features.player_color in self._non_beam_controllers:
                px, py = game_model.player_pos
                gx, gy = self._stable_goal
                dx, dy = gx - px, gy - py
                direction = "right" if dx > 0 else ("left" if dx < 0 else ("down" if dy > 0 else "up"))
                # Use hyperon prior to pick axis when both need movement.
                _DIR_TO_ACTION = {"up": "ACTION1", "down": "ACTION2", "left": "ACTION3", "right": "ACTION4"}
                if dx != 0 and dy != 0:
                    horiz = "right" if dx > 0 else "left"
                    vert = "down" if dy > 0 else "up"
                    pri_h = hyperon_priors.get(_DIR_TO_ACTION.get(horiz, ""), 0.0)
                    pri_v = hyperon_priors.get(_DIR_TO_ACTION.get(vert, ""), 0.0)
                    direction = horiz if pri_h >= pri_v else vert
                step = PlanStep(
                    action="move", params={"direction": direction},
                    source="orch_or:stable_goal", preconditions=frozenset(),
                    expected_effect=f"move_{direction}",
                )
                self._active_plan = Plan(
                    objective_id="reach-goal", steps=[step],
                    created_at_step=context.observation.step_index,
                    confidence=0.5,
                )
                self._persist_plan(context)
                self._skip_monitoring = True
                return self.decide(context)
            return self._fallback_executive(context)
        self._regenerate_count += 1
        # Query hyperon action priors to include in plan context.
        hyperon_priors = {}
        try:
            for row in context.hyperon.run_rules("game-action-priors"):
                action_name = str(row.get("action_name", "")).upper()
                if action_name:
                    hyperon_priors[action_name] = float(row.get("confidence", 0.5))
        except Exception:
            pass
        plan_context: dict[str, Any] = {
            "game_step": context.observation.step_index,
            "switch_attempted": self._switch_attempted,
            "non_beam_controllers": self._non_beam_controllers.copy(),
            "failed_directions": self._failed_directions.copy(),
            "stable_goal": self._stable_goal,
            "action_priors": hyperon_priors,
        }
        try:
            gh = context.hyperon.run_rules("game-goal-hints")
            if gh:
                plan_context["goal_hints"] = gh
        except Exception:
            pass
        _update_beam_context(game_model, self._previous_model, self._get_working_memory(context), plan_context)
        parallel = bool(getattr(self._config, "planner_parallel", False))
        p_timeout = getattr(self._config, "planner_parallel_timeout", None)
        result_plan = generate_plan(game_model, context=plan_context, parallel=parallel, timeout=p_timeout)
        if result_plan is not None:
            if any(s.action == "switch" for s in result_plan.steps):
                self._switch_attempted = True
            self._active_plan = result_plan
            self._persist_plan(context)
            self._skip_monitoring = True
            return self.decide(context)
        self._regenerate_count = 0
        return self._fallback_executive(context)

    def _load_plan(self, context: ExecutiveContext) -> None:
        plan_data = self._get_working_memory(context).get(_PLAN_KEY)
        if plan_data is not None:
            self._active_plan = deserialize_plan(plan_data)
        # Don't clear self._active_plan if WM is empty — keep the in-memory plan
        # so retry_count accumulates across frames

    def _persist_plan(self, context: ExecutiveContext) -> None:
        wm = self._get_working_memory(context)
        if self._active_plan is not None:
            wm[_PLAN_KEY] = serialize_plan(self._active_plan)
        elif _PLAN_KEY in wm:
            del wm[_PLAN_KEY]

    def _get_working_memory(self, context: ExecutiveContext) -> dict[str, Any]:
        if context.memory_bundle is not None:
            wm = context.memory_bundle.working_memory
            if wm is None:
                wm = {}
                context.memory_bundle.working_memory = wm
            return wm
        return {}

    def _handle_status(self, status: MonitorStatus, context: ExecutiveContext, game_model: GameModel | None = None) -> None:
        if status == "advance" and self._active_plan is not None:
            self._active_plan.current_index += 1
            step = self._active_plan.steps[self._active_plan.current_index - 1] if self._active_plan.current_index > 0 else None
            if step is not None:
                if step.action == "switch":
                    self._switch_attempted = True
                    # The entity we switched FROM controlled the beam;
                    # the entity we switched TO does not.
                    if game_model is not None and game_model.features.player_color >= 0:
                        self._non_beam_controllers.add(game_model.features.player_color)
                else:
                    self._switch_pos = None
        elif status == "retry" and self._active_plan is not None:
            step = self._active_plan.steps[self._active_plan.current_index]
            step.retry_count += 1
            if step.params and "direction" in step.params:
                self._pending_failed_dir = step.params["direction"]
            # Early abort: if hyperon gives low confidence to this action,
            # accelerate retry_count so we give up faster.
            _DIR_TO_ACTION = {"up": "ACTION1", "down": "ACTION2", "left": "ACTION3", "right": "ACTION4"}
            action_name = _DIR_TO_ACTION.get((step.params or {}).get("direction", ""))
            if action_name:
                hyperon_conf = self._hyperon_priors.get(action_name, 0.5)
                if hyperon_conf < 0.2:
                    step.retry_count = max(step.retry_count, step.tolerance - 1)
            # Also check hyperon's game-stagnation signal.
            try:
                for row in context.hyperon.run_rules("game-stagnation"):
                    if row.get("flag") == "stagnation":
                        step.retry_count = max(step.retry_count, step.tolerance - 1)
                        break
            except Exception:
                pass
        elif status == "switch_entity" and self._active_plan is not None:
            # Query hyperon to see if it recommends switching.
            hyperon_recommends_switch = False
            try:
                for row in context.hyperon.run_rules("game-action-priors"):
                    if str(row.get("action_name", "")).upper() == "ACTION5":
                        confidence = float(row.get("confidence", 0.0))
                        if confidence > 0.3:
                            hyperon_recommends_switch = True
                        break
            except Exception:
                pass
            if self._switch_attempted and not hyperon_recommends_switch:
                # Already switched once and still stuck, and hyperon doesn't
                # strongly recommend another switch — record failed dir and
                # fall back.
                if self._pending_failed_dir is not None:
                    self._failed_directions.add(self._pending_failed_dir)
                    self._pending_failed_dir = None
                self._active_plan = None
                self._regenerate_count = 0
            else:
                self._switch_attempted = True
                self._inject_switch_before_current(self._active_plan)
        elif status in ("replan", "escalate"):
            if self._pending_failed_dir is not None:
                self._failed_directions.add(self._pending_failed_dir)
                self._pending_failed_dir = None
            self._active_plan = None
            self._regenerate_count = 0

    @staticmethod
    def _inject_switch_before_current(plan: Plan) -> None:
        switch_step = PlanStep(
            action="switch",
            params={},
            source="monitor_stuck",
            preconditions=frozenset(),
            expected_effect="switch_entity",
        )
        cur = plan.current_index
        plan.steps.insert(cur, switch_step)

    def _fallback_executive(self, context: ExecutiveContext) -> ActionDecision:
        logger.debug("falling back to base executive")
        parallel = bool(getattr(self._config, "planner_parallel", False))
        decision: ActionDecision
        if parallel:
            timeout = float(getattr(self._config, "planner_parallel_timeout", 2.0) or 2.0)
            decision = self._fallback_executive_parallel(context, timeout)
        else:
            decision = self._executive.decide(context)
        # Feed the executive's proposals back to hyperon.
        try:
            context.hyperon.update_proposals(decision.proposals)
        except Exception:
            pass
        return decision

    def _fallback_executive_parallel(self, context: ExecutiveContext, timeout: float = 2.0) -> ActionDecision:
        try:
            future = ThreadPoolManager.executor().submit(self._executive.decide, context)
            return future.result(timeout=timeout)
        except TimeoutError:
            return ActionDecision(
                action_name=context.config.deterministic_fallback_action,
                source="orch_or:parallel_timeout",
                confidence=0.05,
                reason="executive fallback timed out, using deterministic action",
                used_fallback=True,
            )


def _update_beam_context(
    current: GameModel,
    previous: GameModel | None,
    wm: dict[str, Any],
    ctx: dict[str, Any],
) -> None:
    if not current.features.has_beam or previous is None or not previous.features.has_beam:
        wm.pop(_BEAM_COVERAGE_KEY, None)
        wm.pop(_BEAM_DIR_KEY, None)
        wm.pop(_BEAM_PROX_KEY, None)
        wm.pop(_BEAM_COLOR_KEY, None)
        return
    prev_goals_lit = previous.features.goals_lit_by_beam
    curr_goals_lit = current.features.goals_lit_by_beam
    ctx["prev_beam_coverage"] = prev_goals_lit
    ctx["beam_coverage_change"] = curr_goals_lit - prev_goals_lit
    ctx["total_goals"] = len(current.goal_positions)
    prev_prox = previous.features.beam_min_dist_to_goal
    curr_prox = current.features.beam_min_dist_to_goal
    ctx["beam_proximity_change"] = prev_prox - curr_prox
    if previous.player_pos is not None and current.player_pos is not None:
        px, py = previous.player_pos
        cx, cy = current.player_pos
        dx = cx - px
        dy = cy - py
        if abs(dx) > 0:
            ctx["last_move_direction"] = "right" if dx > 0 else "left"
        elif abs(dy) > 0:
            ctx["last_move_direction"] = "down" if dy > 0 else "up"
    if (ctx.get("beam_coverage_change", 0) > 0 or ctx.get("beam_proximity_change", 0) > 0) and ctx.get("last_move_direction"):
        wm[_BEAM_DIR_KEY] = ctx["last_move_direction"]
    wm[_BEAM_COVERAGE_KEY] = curr_goals_lit
    wm[_BEAM_PROX_KEY] = curr_prox
    best_dir = wm.get(_BEAM_DIR_KEY)
    if best_dir:
        ctx["beam_best_direction"] = best_dir


def _get_board_data(observation) -> list[list[int]] | None:
    derived = getattr(observation, "derived_features", {})
    board_grid = [
        [int(p) for p in row]
        for row in (derived.get("board_grid", []) if derived else [])
    ]
    if board_grid:
        return board_grid
    image = getattr(observation, "image_input", None)
    if image is None:
        return None
    board = getattr(image, "board", None)
    if board is not None and len(board) > 0 and len(board[0]) > 0:
        return [[int(v) for v in row] for row in board]
    return None
