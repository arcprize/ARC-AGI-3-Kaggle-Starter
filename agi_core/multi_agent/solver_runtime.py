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
        "shape_stage_targets": list(mirror_puzzle.get("stage_targets", []))
        if isinstance(mirror_puzzle, dict)
        else [],
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
    player_position = (
        getattr(context.observation, "derived_features", {}) or {}
    ).get("player_position")
    if mirror_active and active_target:
        sticky = sticky_shape_target
        if (
            sticky is not None
            and sticky.get("levels_completed") == levels_completed
            and sticky.get("orientation") == frame_cache.get("mirror_orientation")
            and int(sticky.get("stage", 0) or 0) == active_stage
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
        }
    sticky = sticky_shape_target
    if (
        mirror_active
        and not active_target
        and sticky is not None
        and sticky.get("levels_completed") == levels_completed
        and sticky.get("orientation") == frame_cache.get("mirror_orientation")
    ):
        frame_cache["active_shape_target"] = sticky["target"]
        frame_cache["active_shape_stage"] = sticky["stage"]
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
    return {
        "levels_completed": int(getattr(context.observation, "levels_completed", 0) or 0),
        "target": (int(target[0]), int(target[1])),
        "stage": int(metadata.get("shape_stage", 0) or 0),
        "orientation": mirror_puzzle.get("orientation"),
    }
