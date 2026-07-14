from __future__ import annotations

from typing import Any

from agi_core.config import RuntimeConfig
from agi_core.types import EpisodeRecord, Observation, ObservedTransition


def _has_observation_board(observation: Observation | None) -> bool:
    if observation is None:
        return False
    derived = getattr(observation, "derived_features", {}) or {}
    if bool(derived.get("has_image_input")):
        return True
    image_input = getattr(observation, "image_input", None)
    board = getattr(image_input, "board", None)
    return bool(isinstance(board, list) and board and isinstance(board[0], list) and board[0])


def _is_late_same_level_soft_reset(
    previous: Observation | None,
    current: Observation,
    reset_states: set[str],
) -> bool:
    if previous is None:
        return False
    previous_state = str(previous.game_state or "").upper()
    current_state = str(current.game_state or "").upper()
    if current_state not in reset_states:
        return False
    previous_level = int(previous.levels_completed or 0)
    current_level = int(current.levels_completed or 0)
    if previous_state not in reset_states:
        return (
            previous_level >= 3
            and current_level == previous_level
            and _has_observation_board(current)
        )
    return (
        previous_level >= 3
        and current_level < previous_level
        and _has_observation_board(previous)
        and not _has_observation_board(current)
    )


def _is_late_same_level_soft_restart(
    previous: Observation | None,
    current: Observation,
    reset_states: set[str],
) -> bool:
    if previous is None:
        return False
    previous_state = str(previous.game_state or "").upper()
    current_state = str(current.game_state or "").upper()
    if previous_state not in reset_states or current_state in reset_states:
        return False
    previous_level = int(previous.levels_completed or 0)
    current_level = int(current.levels_completed or 0)
    return (
        current_level >= 3
        and previous_level <= current_level
        and not _has_observation_board(previous)
        and _has_observation_board(current)
    )


def observation_boundary(
    previous: Observation | None,
    current: Observation,
    config: RuntimeConfig,
) -> str | None:
    if previous is None:
        return None
    if str(previous.game_id or "") != str(current.game_id or ""):
        return "game"
    state = str(current.game_state or "").upper()
    if state == str(config.win_state_name or "").upper():
        return "win"
    reset_states = {str(name or "").upper() for name in config.reset_state_names}
    if _is_late_same_level_soft_restart(previous, current, reset_states):
        return "soft_restart"
    if int(current.levels_completed or 0) > int(previous.levels_completed or 0):
        return "level"
    if _is_late_same_level_soft_reset(previous, current, reset_states):
        return "soft_reset"
    if state in reset_states:
        return "game_over"
    previous_state = str(previous.game_state or "").upper()
    if previous_state in reset_states and state not in reset_states:
        return "restart"
    if (
        bool(current.derived_features.get("step_index_reliable", True))
        and bool(previous.derived_features.get("step_index_reliable", True))
        and int(current.step_index or 0) < int(previous.step_index or 0)
    ):
        return "restart"
    return None


def observation_player_position(
    observation: Observation | None,
) -> tuple[int, int] | None:
    if observation is None:
        return None
    value = observation.derived_features.get("player_position")
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    return None


def observation_frame_signature(observation: Observation | None) -> str:
    if observation is None:
        return ""
    return str(observation.derived_features.get("frame_signature", "") or "")


def classify_observed_transition(
    previous: Observation | None,
    current: Observation,
    record: EpisodeRecord,
    config: RuntimeConfig,
) -> ObservedTransition:
    boundary = observation_boundary(previous, current, config)
    previous_position = observation_player_position(previous)
    current_position = observation_player_position(current)
    player_moved = bool(
        previous_position is not None
        and current_position is not None
        and previous_position != current_position
    )
    previous_signature = observation_frame_signature(previous)
    current_signature = observation_frame_signature(current)
    frame_changed = bool(
        previous_signature
        and current_signature
        and previous_signature != current_signature
    )

    if boundary in {"win", "level"} or int(record.progress_delta or 0) > 0:
        outcome = "progress"
    elif boundary == "game_over" or str(record.outcome or "") == "failed":
        outcome = "failed"
    elif boundary in {"restart", "game", "soft_reset", "soft_restart"}:
        outcome = "unknown"
    elif player_moved:
        outcome = "advanced"
    elif previous_signature and current_signature and not frame_changed:
        outcome = "stalled"
    elif frame_changed:
        outcome = "unknown"
    else:
        outcome = "stalled" if str(record.outcome or "") == "stalled" else "unknown"

    return ObservedTransition(
        game_id=current.game_id,
        action_name=str(record.action_name or ""),
        decision_source=str(record.decision_source or ""),
        decision_subgoal=record.decision_subgoal,
        local_outcome=outcome,
        player_moved=player_moved,
        frame_changed=frame_changed,
        boundary=boundary,
        record=record,
    )


def boundary_transition(
    observation: Observation,
    boundary: str,
) -> ObservedTransition:
    return ObservedTransition(
        game_id=observation.game_id,
        local_outcome="unknown",
        boundary=boundary,
    )
