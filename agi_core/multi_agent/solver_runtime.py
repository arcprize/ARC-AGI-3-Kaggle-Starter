from __future__ import annotations

from typing import Any

from agi_core.knowledge.solver_selector import SolverSelector
from agi_core.perception.shape_recognition import ensure_shape_recognition
from agi_core.types import ActionDecision, ExecutiveContext


def board_has_mirror(board: list[list[int]]) -> bool:
    return any(value == 10 for row in board for value in row)


def get_board_data(observation: Any) -> list[list[int]] | None:
    image_input = getattr(observation, "image_input", None)
    if image_input is None:
        return None
    return getattr(image_input, "board", None)


def _normalize_stage_targets(stage_targets: Any) -> list[dict[str, Any]]:
    if not isinstance(stage_targets, list):
        return []
    normalized: list[dict[str, Any]] = []
    for raw_stage in stage_targets:
        if not isinstance(raw_stage, dict):
            continue
        stage = dict(raw_stage)
        target = stage.get("target_anchor")
        if isinstance(target, (list, tuple)) and len(target) == 2:
            stage["target_anchor"] = [int(target[0]), int(target[1])]
        normalized.append(stage)
    return normalized


def sync_solver_episode_facts(context: ExecutiveContext, game_model: Any) -> None:
    try:
        episode = context.hyperon.facts.setdefault("episode", {})
        episode["board_type"] = game_model.features.board_type
        episode["primary_dynamics"] = game_model.features.primary_dynamics
        episode["dynamics"] = list(game_model.features.dynamics)
    except Exception:
        pass


def build_solver_context(
    solver: Any,
    game_model: Any,
    context: ExecutiveContext,
    supervisor: Any,
    *,
    memory_db_path: str | None,
) -> Any:
    shape_recognition = ensure_shape_recognition(
        getattr(context.observation, "image_input", None),
        getattr(context.observation, "derived_features", {}),
    )
    mirror_puzzle = (
        shape_recognition.get("mirror_puzzle", {})
        if isinstance(shape_recognition, dict)
        else {}
    )
    frame_cache = getattr(context, "multi_agent_frame_cache", {}) or {}
    cached_stage_targets = _normalize_stage_targets(frame_cache.get("shape_stage_targets"))
    shape_stage_targets = (
        cached_stage_targets
        if cached_stage_targets
        else list(mirror_puzzle.get("stage_targets", []))
        if isinstance(mirror_puzzle, dict)
        else []
    )
    plan_context: dict[str, Any] = {
        "non_beam_controllers": getattr(supervisor, "_non_beam_controllers", set()),
        "failed_directions": getattr(supervisor, "_failed_directions", set()),
        "stable_goal": getattr(supervisor, "_stable_goal", None),
        "switch_attempted": getattr(supervisor, "_switch_count", 0),
        "levels_completed": getattr(context.observation, "levels_completed", 0),
        "action_priors": {},
        "memory_db_path": memory_db_path,
        "beam_cells": [],
        "mirror_axis": (
            game_model.features.mirror_axis if hasattr(game_model, "features") else -1
        ),
        "goal_history": list(getattr(supervisor, "_goal_history", [])),
        "abstract_memories": list(
            context.memory_bundle.abstract_memories
        )
        if context.memory_bundle and context.memory_bundle.abstract_memories
        else [],
        "shape_recognition": shape_recognition,
        "shape_stage_targets": shape_stage_targets,
    }
    selector = getattr(supervisor, "_solver_selector", None)
    if selector is not None:
        return selector.build_solver_context(game_model, plan_context, context)
    selector = SolverSelector(
        context.hyperon,
        memory_db_path=memory_db_path,
    )
    return selector.build_solver_context(game_model, plan_context, context)


def stabilize_shape_frame_cache(
    sticky_shape_target: dict[str, Any] | None,
    context: ExecutiveContext,
    frame_cache: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    levels_completed = int(getattr(context.observation, "levels_completed", 0) or 0)
    mirror_active = bool(frame_cache.get("mirror_puzzle_active"))
    active_target = frame_cache.get("active_shape_target")
    active_stage = int(frame_cache.get("active_shape_stage", 0) or 0)
    stage_targets = _normalize_stage_targets(frame_cache.get("shape_stage_targets"))
    player_position = (
        getattr(context.observation, "derived_features", {}) or {}
    ).get("player_position")
    sticky = sticky_shape_target
    sticky_targets = _normalize_stage_targets(sticky.get("stage_targets")) if isinstance(sticky, dict) else []
    sticky_stage = int(sticky.get("stage", 0) or 0) if isinstance(sticky, dict) else 0
    sticky_valid = (
        sticky is not None
        and sticky.get("levels_completed") == levels_completed
        and sticky.get("orientation") == frame_cache.get("mirror_orientation")
    )
    if mirror_active and active_target:
        if sticky_valid and sticky_targets and (not stage_targets or active_stage == sticky_stage):
            frame_cache["shape_stage_targets"] = [dict(stage) for stage in sticky_targets]
            stage_targets = _normalize_stage_targets(frame_cache.get("shape_stage_targets"))
        if (
            sticky_valid
            and sticky_stage == active_stage
            and isinstance(sticky.get("target"), tuple)
        ):
            sticky_target = sticky["target"]
            target_shift = (
                abs(int(active_target[0]) - int(sticky_target[0]))
                + abs(int(active_target[1]) - int(sticky_target[1]))
            )
            sticky_distance = None
            if isinstance(player_position, tuple) and len(player_position) == 2:
                sticky_distance = (
                    abs(int(sticky_target[0]) - int(player_position[0]))
                    + abs(int(sticky_target[1]) - int(player_position[1]))
                )
            if target_shift >= 6 and (sticky_distance is None or sticky_distance > 2):
                frame_cache["active_shape_target"] = sticky_target
                active_target = sticky_target
        return frame_cache, {
            "levels_completed": levels_completed,
            "target": active_target,
            "stage": active_stage,
            "orientation": frame_cache.get("mirror_orientation"),
            "stage_targets": stage_targets,
        }
    if (
        mirror_active
        and not active_target
        and sticky_valid
    ):
        frame_cache["active_shape_target"] = sticky["target"]
        frame_cache["active_shape_stage"] = sticky["stage"]
        if sticky_targets:
            frame_cache["shape_stage_targets"] = [dict(stage) for stage in sticky_targets]
        frame_cache["shape_agent_mode"] = levels_completed >= 2
        return frame_cache, sticky
    if not mirror_active:
        return frame_cache, None
    return frame_cache, sticky


def seed_sticky_shape_target(
    sticky_shape_target: dict[str, Any] | None,
    context: ExecutiveContext,
    decision: ActionDecision,
) -> dict[str, Any] | None:
    metadata = getattr(decision, "metadata", {}) or {}
    target = metadata.get("target_anchor")
    if not isinstance(target, list) or len(target) != 2:
        return sticky_shape_target
    frame_cache = getattr(context, "multi_agent_frame_cache", {}) or {}
    shape_recognition = frame_cache.get("shape_recognition")
    if not isinstance(shape_recognition, dict):
        shape_recognition = ensure_shape_recognition(
            getattr(context.observation, "image_input", None),
            getattr(context.observation, "derived_features", {}),
        )
    mirror_puzzle = (
        shape_recognition.get("mirror_puzzle", {})
        if isinstance(shape_recognition, dict)
        else {}
    )
    if not isinstance(mirror_puzzle, dict) or not mirror_puzzle.get("active"):
        return sticky_shape_target
    stage_targets = _normalize_stage_targets(frame_cache.get("shape_stage_targets"))
    stage = int(metadata.get("shape_stage", 0) or 0)
    if stage > 0:
        while len(stage_targets) < stage:
            stage_targets.append({})
        stage_targets[stage - 1] = {
            **dict(stage_targets[stage - 1]),
            "target_anchor": [int(target[0]), int(target[1])],
        }
    return {
        "levels_completed": int(getattr(context.observation, "levels_completed", 0) or 0),
        "target": (int(target[0]), int(target[1])),
        "stage": stage,
        "orientation": mirror_puzzle.get("orientation"),
        "stage_targets": stage_targets,
    }
