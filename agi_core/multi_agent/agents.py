from __future__ import annotations

from typing import Any

from agi_core.multi_agent.payload import AgentFrameSnapshot, sanitize_payload
from agi_core.multi_agent.strategy_agent import StrategyAgent
from agi_core.perception.shape_recognition import ensure_shape_recognition
from agi_core.types import ExecutiveContext

AgentFrameInput = ExecutiveContext | AgentFrameSnapshot


def _frame_cache(frame: AgentFrameInput) -> dict[str, Any]:
    if isinstance(frame, AgentFrameSnapshot):
        return frame.frame_cache
    return frame.multi_agent_frame_cache


def _derived_features(frame: AgentFrameInput) -> dict[str, Any]:
    if isinstance(frame, AgentFrameSnapshot):
        return frame.derived_features
    return getattr(frame.observation, "derived_features", {}) or {}


def _normalize_single_stage_vertical_target(
    mirror_puzzle: dict[str, Any],
    goals: list[tuple[int, int]],
    width: int,
    height: int,
) -> tuple[int, int] | None:
    if not isinstance(mirror_puzzle, dict) or mirror_puzzle.get("orientation") != "vertical":
        return None
    stage_targets = mirror_puzzle.get("stage_targets", [])
    if not isinstance(stage_targets, list) or len(stage_targets) != 1:
        return None
    axis = mirror_puzzle.get("axis_pixel")
    if not isinstance(axis, int) or not goals:
        return None
    goal_xs = [int(goal[0]) for goal in goals]
    goal_ys = [int(goal[1]) for goal in goals]
    target_x = max(0, min(int(width) - 1, 2 * int(axis) - max(goal_xs)))
    target_y = max(0, min(int(height) - 1, min(goal_ys)))
    return (target_x, target_y)


def prepare_frame_cache(context: AgentFrameInput) -> dict[str, Any]:
    """Precompute shared per-frame agent features once per observation."""
    cache = _frame_cache(context)
    # Other frame-scoped systems (notably Hyperon synchronization) may seed
    # this dictionary before the strategy-agent fields are prepared.
    if "legal_set" in cache and "legal_actions" in cache:
        return cache
    if isinstance(context, AgentFrameSnapshot):
        return cache
    observation = context.observation
    legal_actions = tuple(getattr(observation, "legal_actions", ()) or ())
    legal_set = set(legal_actions)
    features = getattr(observation, "derived_features", {}) or {}
    shape_recognition = ensure_shape_recognition(getattr(observation, "image_input", None), features)
    mirror_puzzle = shape_recognition.get("mirror_puzzle", {}) if isinstance(shape_recognition, dict) else {}
    stage_targets = mirror_puzzle.get("stage_targets", []) if isinstance(mirror_puzzle, dict) else []
    active_stage = max(1, int(mirror_puzzle.get("active_stage", 1) or 1))
    active_target = None
    if stage_targets:
        stage_index = min(active_stage - 1, len(stage_targets) - 1)
        active_target = tuple(stage_targets[stage_index].get("target_anchor", ())) or None
    player = features.get("player_position")
    if player is None:
        active_anchor = shape_recognition.get("active_anchor")
        player = tuple(active_anchor) if isinstance(active_anchor, list) and len(active_anchor) == 2 else None
    goals = features.get("goal_positions", [])
    if not goals:
        goals = [
            tuple(cluster.get("center", ()))
            for cluster in shape_recognition.get("goal_clusters", [])
            if cluster.get("center")
        ]
    normalized_target = _normalize_single_stage_vertical_target(
        mirror_puzzle,
        goals,
        getattr(observation, "image_input", None).width if not isinstance(context, AgentFrameSnapshot) and getattr(observation, "image_input", None) is not None else 64,
        getattr(observation, "image_input", None).height if not isinstance(context, AgentFrameSnapshot) and getattr(observation, "image_input", None) is not None else 64,
    )
    if normalized_target is not None:
        active_target = normalized_target
    goal_direction = None
    target_for_direction = active_target
    if player is not None and target_for_direction is not None:
        px, py = player
        gx, gy = target_for_direction
        if abs(gx - px) > abs(gy - py):
            goal_direction = "ACTION4" if gx > px else "ACTION3"
        else:
            goal_direction = "ACTION2" if gy > py else "ACTION1"
    elif player is not None and goals:
        px, py = player
        gx = sum(g[0] for g in goals) / len(goals)
        gy = sum(g[1] for g in goals) / len(goals)
        if abs(gx - px) > abs(gy - py):
            goal_direction = "ACTION4" if gx > px else "ACTION3"
        else:
            goal_direction = "ACTION2" if gy > py else "ACTION1"
    analyzer_perception_failed = bool(features.get("analyzer_perception_failed"))
    has_beam_value = features.get("has_beam")
    has_beam = bool(has_beam_value) if has_beam_value is not None and not analyzer_perception_failed else False
    has_mirror = bool(shape_recognition.get("mirror", {}).get("present"))
    target_distance = None
    ready_to_switch = False
    if player is not None and active_target is not None:
        dx = abs(int(active_target[0]) - int(player[0]))
        dy = abs(int(active_target[1]) - int(player[1]))
        target_distance = dx + dy
        ready_to_switch = dx <= 1 and dy <= 1
    image = getattr(observation, "image_input", None)
    board = getattr(image, "board", None)
    beam_visible = False
    beam_evidence_uncertain = False
    if board:
        beam_visible = any(4 in row for row in board)
        if has_beam_value is None and not analyzer_perception_failed:
            has_beam = any(4 in row for row in board)
        beam_evidence_uncertain = analyzer_perception_failed and beam_visible
        if not has_mirror:
            has_mirror = any(10 in row for row in board)
    shape_agent_mode = bool(mirror_puzzle.get("active")) and active_target is not None and int(getattr(observation, "levels_completed", 0) or 0) >= 2
    cache.update(
        {
            "legal_actions": legal_actions,
            "legal_set": legal_set,
            "has_beam": has_beam,
            "beam_visible": beam_visible,
            "beam_evidence_uncertain": beam_evidence_uncertain,
            "has_mirror": has_mirror,
            "goal_direction": goal_direction,
            "shape_recognition": shape_recognition,
            "mirror_puzzle_active": bool(mirror_puzzle.get("active")),
            "shape_agent_mode": shape_agent_mode,
            "mirror_orientation": mirror_puzzle.get("orientation"),
            "active_shape_target": active_target,
            "active_shape_stage": active_stage if stage_targets else 0,
            "active_shape_target_distance": target_distance,
            "ready_to_switch": ready_to_switch,
        }
    )
    return cache


def build_frame_snapshot(context: ExecutiveContext) -> AgentFrameSnapshot:
    cache = dict(prepare_frame_cache(context))
    return AgentFrameSnapshot(
        game_id=str(getattr(context.observation, "game_id", "") or ""),
        step_index=int(getattr(context.observation, "step_index", 0) or 0),
        levels_completed=int(getattr(context.observation, "levels_completed", 0) or 0),
        legal_actions=tuple(getattr(context.observation, "legal_actions", ()) or ()),
        derived_features=sanitize_payload(_derived_features(context)),
        frame_cache=sanitize_payload(cache),
    )


def _has_beam(context: AgentFrameInput) -> bool:
    """Check if the current board has beam (color 4)."""
    return bool(prepare_frame_cache(context).get("has_beam"))


def _has_uncertain_beam(context: AgentFrameInput) -> bool:
    """Check for visible beam evidence when analyzer confirmation failed."""
    return bool(prepare_frame_cache(context).get("beam_evidence_uncertain"))


def _has_mirror(context: AgentFrameInput) -> bool:
    """Check if the board has mirror cells (color 10)."""
    return bool(prepare_frame_cache(context).get("has_mirror"))


def _goal_direction(context: AgentFrameInput) -> str | None:
    """Return a heuristic direction toward goals, or None."""
    return prepare_frame_cache(context).get("goal_direction")


def _target_metadata(context: AgentFrameInput) -> dict[str, Any]:
    cache = prepare_frame_cache(context)
    target = cache.get("active_shape_target")
    if target is None:
        return {}
    return {
        "target_anchor": [int(target[0]), int(target[1])],
        "shape_stage": int(cache.get("active_shape_stage", 0) or 0),
        "target_distance": int(cache.get("active_shape_target_distance", 0) or 0),
        "ready_to_switch": bool(cache.get("ready_to_switch")),
    }


def _ordered_mirror_actions(context: AgentFrameInput) -> list[str]:
    cache = prepare_frame_cache(context)
    legal = cache["legal_set"]
    ordered: list[str] = []
    target = cache.get("active_shape_target")
    player = _derived_features(context).get("player_position")
    if cache.get("ready_to_switch") and "ACTION5" in legal:
        ordered.append("ACTION5")
    if (
        isinstance(player, tuple)
        and isinstance(target, tuple)
        and len(player) == 2
        and len(target) == 2
    ):
        dx = int(target[0]) - int(player[0])
        dy = int(target[1]) - int(player[1])
        primary = None
        secondary = None
        if abs(dx) >= abs(dy):
            primary = "ACTION4" if dx > 0 else "ACTION3" if dx < 0 else None
            secondary = "ACTION2" if dy > 0 else "ACTION1" if dy < 0 else None
        else:
            primary = "ACTION2" if dy > 0 else "ACTION1" if dy < 0 else None
            secondary = "ACTION4" if dx > 0 else "ACTION3" if dx < 0 else None
        for action in (primary, secondary):
            if action and action in legal and action not in ordered:
                ordered.append(action)
    goal_dir = cache.get("goal_direction")
    if goal_dir and goal_dir in legal and goal_dir not in ordered:
        ordered.append(goal_dir)
    if cache.get("shape_agent_mode"):
        if cache.get("ready_to_switch") and "ACTION5" in legal and "ACTION5" not in ordered:
            ordered.append("ACTION5")
        return ordered
    for action in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
        if action in legal and action not in ordered:
            ordered.append(action)
    if cache.get("ready_to_switch") and "ACTION5" in legal and "ACTION5" not in ordered:
        ordered.append("ACTION5")
    if "ACTION6" in legal and "ACTION6" not in ordered:
        ordered.append("ACTION6")
    return ordered


class MovementAgent(StrategyAgent):
    """Specialises in A*-based navigation and direction heuristics."""

    def __init__(self) -> None:
        super().__init__(
            agent_id="movement", wm_capacity=7,
            wm_category_max={"goal": 3, "outcome": 2, "percept": 2},
            gs_max_failures=3,
        )
        self.gs.set_root("reach-goal-area")

    def produce(self, context: ExecutiveContext) -> list:
        cache = prepare_frame_cache(context)
        legal = cache["legal_set"]
        proposals = []
        if cache.get("shape_agent_mode"):
            target_metadata = _target_metadata(context)
            ordered_actions = _ordered_mirror_actions(context)
            for index, action in enumerate(ordered_actions):
                if action not in legal or action == "ACTION6":
                    continue
                if action == "ACTION5" and target_metadata.get("ready_to_switch"):
                    confidence = 0.86
                elif index == 0:
                    confidence = 0.82
                elif index == 1:
                    confidence = 0.74
                else:
                    confidence = 0.42
                proposals.append(
                    self._make_proposal(
                        action_name=action,
                        confidence=confidence,
                        reason=f"movement:shape-stage-{target_metadata.get('shape_stage', 0)}",
                        subgoal="navigate",
                        expected_outcome="progress",
                        metadata=target_metadata,
                    )
                )
            return proposals
        goal_dir = _goal_direction(context)
        target_metadata = _target_metadata(context)
        for action in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
            if action in legal:
                confidence = 0.78 if action == goal_dir and target_metadata else 0.7 if action == goal_dir else 0.5
                proposals.append(
                    self._make_proposal(
                        action_name=action, confidence=confidence,
                        reason=f"movement:{action}", subgoal="navigate",
                        expected_outcome="progress",
                        metadata=target_metadata,
                    )
                )
        return proposals


class ExplorationAgent(StrategyAgent):
    """Specialises in board exploration via beam nav and switches."""

    def __init__(self) -> None:
        super().__init__(
            agent_id="exploration", wm_capacity=7,
            wm_category_max={"goal": 2, "outcome": 3, "percept": 2},
            gs_max_failures=3,
        )
        self.gs.set_root("explore-board")

    def produce(self, context: ExecutiveContext) -> list:
        cache = prepare_frame_cache(context)
        legal = cache["legal_set"]
        proposals = []
        if cache.get("shape_agent_mode"):
            return proposals
        has_switch = "ACTION5" in legal
        has_mirror = _has_mirror(context)
        # Only propose switch if no mirror present (PuzzleAgent handles mirror cases)
        if has_switch and not has_mirror:
            has_beam_now = _has_beam(context)
            has_uncertain_beam = _has_uncertain_beam(context)
            switch_conf = 0.65 if has_beam_now else 0.6 if has_uncertain_beam else 0.55
            proposals.append(
                self._make_proposal(
                    action_name="ACTION5", confidence=switch_conf,
                    reason="exploration:switch" + (
                        "+beam" if has_beam_now else "+beam-uncertain" if has_uncertain_beam else ""
                    ),
                    subgoal="try-switch",
                )
            )
        for action in ("ACTION2", "ACTION1", "ACTION3", "ACTION4"):
            if action in legal:
                proposals.append(
                    self._make_proposal(
                        action_name=action, confidence=0.45,
                        reason=f"exploration:{action}", subgoal="explore-area",
                    )
                )
                break
        return proposals


class InteractionAgent(StrategyAgent):
    """Specialises in clicking interactive objects and mirror positioning."""

    def __init__(self) -> None:
        super().__init__(
            agent_id="interaction", wm_capacity=7,
            wm_category_max={"goal": 2, "outcome": 2, "percept": 3},
            gs_max_failures=2,
        )
        self.gs.set_root("interact-with-board")

    def produce(self, context: ExecutiveContext) -> list:
        cache = prepare_frame_cache(context)
        legal = cache["legal_set"]
        proposals = []
        if cache.get("mirror_puzzle_active"):
            target_metadata = _target_metadata(context)
            if cache.get("ready_to_switch") and "ACTION5" in legal:
                proposals.append(
                    self._make_proposal(
                        action_name="ACTION5",
                        confidence=0.74,
                        reason="interaction:switch-stage",
                        subgoal="solve-puzzle",
                        metadata=target_metadata,
                    )
                )
            return proposals
        if "ACTION6" in legal:
            has_mirror_now = _has_mirror(context)
            mirror_puzzle_active = bool(cache.get("shape_agent_mode"))
            confidence = 0.45 if mirror_puzzle_active else 0.65 if has_mirror_now else 0.55
            proposals.append(
                self._make_proposal(
                    action_name="ACTION6", confidence=confidence,
                    reason="interaction:inspect" + ("+mirror" if has_mirror_now else ""),
                    subgoal="inspect-active-board" if mirror_puzzle_active else "interact-with-board",
                    metadata=_target_metadata(context),
                )
            )
        # Also propose movement to reach interactive areas
        goal_dir = _goal_direction(context)
        if goal_dir and goal_dir in legal:
            proposals.append(
                self._make_proposal(
                    action_name=goal_dir, confidence=0.4,
                    reason="interaction:approach", subgoal="inspect-active-board",
                    metadata=_target_metadata(context),
                )
            )
        return proposals


class PuzzleAgent(StrategyAgent):
    """Specialises in mirror reflection puzzles and beam alignment."""

    def __init__(self) -> None:
        super().__init__(
            agent_id="puzzle", wm_capacity=7,
            wm_category_max={"goal": 3, "outcome": 2, "percept": 2},
            gs_max_failures=3,
        )
        self.gs.set_root("solve-puzzle")

    def produce(self, context: ExecutiveContext) -> list:
        cache = prepare_frame_cache(context)
        legal = cache["legal_set"]
        proposals = []
        has_beam = _has_beam(context)
        has_uncertain_beam = _has_uncertain_beam(context)
        has_mirror = _has_mirror(context)
        target_metadata = _target_metadata(context)
        if cache.get("shape_agent_mode") and target_metadata:
            ordered_actions = _ordered_mirror_actions(context)
            for index, action in enumerate(ordered_actions):
                if action == "ACTION6":
                    continue
                if action in legal:
                    if action == "ACTION5" and target_metadata.get("ready_to_switch"):
                        confidence = 0.9
                    elif index == 0:
                        confidence = 0.84
                    elif index == 1:
                        confidence = 0.76
                    else:
                        confidence = 0.36 if action == "ACTION6" else 0.46
                    proposals.append(
                        self._make_proposal(
                            action_name=action,
                            confidence=confidence,
                            reason=f"puzzle:shape-stage-{target_metadata['shape_stage']}",
                            subgoal="solve-puzzle",
                            metadata=target_metadata,
                        )
                    )
            return proposals
        # When beam or mirror present, prioritise switches and strategic movement
        if has_beam or has_mirror or has_uncertain_beam:
            # Propose ALL relevant actions (not just the first match)
            for action in ("ACTION5", "ACTION4", "ACTION2", "ACTION1", "ACTION3"):
                if action in legal:
                    conf = 0.7 if action in ("ACTION5", "ACTION4", "ACTION2") else 0.5
                    proposals.append(
                        self._make_proposal(
                            action_name=action, confidence=conf,
                            reason=f"puzzle:{action}" + (
                                "+beam" if has_beam else "+beam-uncertain" if has_uncertain_beam else "+mirror"
                            ),
                            subgoal="solve-puzzle",
                        )
                    )
            if "ACTION6" in legal:
                proposals.append(
                    self._make_proposal(
                        action_name="ACTION6", confidence=0.6,
                        reason="puzzle:inspect", subgoal="inspect-active-board",
                    )
                )
        else:
            for action in ("ACTION5", "ACTION4", "ACTION2", "ACTION1", "ACTION3"):
                if action in legal:
                    proposals.append(
                        self._make_proposal(
                            action_name=action, confidence=0.5,
                            reason=f"puzzle:{action}", subgoal="solve-puzzle",
                        )
                    )
                    break
        return proposals
