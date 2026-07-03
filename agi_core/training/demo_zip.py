from __future__ import annotations

import hashlib
import json
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from agi_core.config import RuntimeConfig
from agi_core.storage import (
    initialize_memory_schema,
    record_transitions,
    upsert_metadata,
    upsert_training_counts,
)
from agi_core.types import EpisodeRecord

@dataclass(slots=True)
class RecordingMeta:
    recording_name: str
    game_id: str
    recording_guid: str


@dataclass(slots=True)
class TrainingStep:
    game_id: str
    recording_guid: str
    step_index: int
    timestamp: str
    frame: Any
    available_actions: list[str]
    action_id: int
    action_name: str | None
    action_data: dict[str, Any]
    levels_completed: int
    state: str
    win_levels: int
    full_reset: bool
    frame_signature: str | None


def _win_state_name() -> str:
    return _terminal_state_config()[0]


def _reset_state_names() -> set[str]:
    return set(_terminal_state_config()[1])


@lru_cache(maxsize=1)
def _terminal_state_config() -> tuple[str, tuple[str, ...]]:
    config = RuntimeConfig.from_env()
    return config.win_state_name, tuple(config.reset_state_names)


def _is_win_state(state_name: str) -> bool:
    return str(state_name or "").upper() == _win_state_name()


def _is_reset_state(state_name: str) -> bool:
    return str(state_name or "").upper() in _reset_state_names()


def normalize_action_name(action_id: Any) -> str | None:
    if isinstance(action_id, str):
        text = action_id.strip().upper()
        if text == "RESET":
            return "RESET"
        if text.startswith("ACTION") and text[6:].isdigit():
            value = int(text[6:])
            if 1 <= value <= 6:
                return f"ACTION{value}"
    try:
        value = int(action_id)
    except (TypeError, ValueError):
        return None
    if value == 0:
        return "RESET"
    if 1 <= value <= 6:
        return f"ACTION{value}"
    return None


def normalize_action_id(action_id: Any) -> int | None:
    action_name = normalize_action_name(action_id)
    if action_name is None:
        return None
    if action_name == "RESET":
        return 0
    return int(action_name.replace("ACTION", ""))


def latest_visible_board(frame: Any) -> Any:
    if not isinstance(frame, list) or not frame:
        return None
    candidate = frame[-1]
    if not isinstance(candidate, list) or not candidate:
        return None
    if not isinstance(candidate[0], list):
        return None
    return candidate


def compute_frame_signature(frame: Any) -> str | None:
    board = latest_visible_board(frame)
    if board is None:
        return None
    height = len(board)
    width = len(board[0]) if board[0] else 0
    payload = json.dumps(
        {"height": height, "width": width, "board": board},
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def runtime_frame_signature(raw_frame_ref: Any) -> str | None:
    if raw_frame_ref is None:
        return None
    for attribute in ("frame", "frames", "board", "grid"):
        value = getattr(raw_frame_ref, attribute, None)
        signature = compute_frame_signature(value)
        if signature is not None:
            return signature
    return None


def iter_recordings(zip_path: str | Path) -> Iterator[RecordingMeta]:
    archive_path = Path(zip_path)
    with zipfile.ZipFile(archive_path) as archive:
        for name in sorted(archive.namelist()):
            if not _is_recording_entry(name):
                continue
            parts = Path(name).parts
            if len(parts) < 3:
                continue
            game_id = parts[-2]
            recording_guid = Path(parts[-1]).name.replace(".recording.jsonl", "")
            yield RecordingMeta(
                recording_name=name,
                game_id=game_id,
                recording_guid=recording_guid,
            )


def iter_steps(zip_path: str | Path, recording_name: str) -> Iterator[TrainingStep]:
    archive_path = Path(zip_path)
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(recording_name, "r") as handle:
            for step_index, raw_line in enumerate(handle):
                payload = json.loads(raw_line.decode("utf-8"))
                data = payload.get("data", {})
                if not isinstance(data, dict):
                    continue
                action_input = data.get("action_input", {})
                if not isinstance(action_input, dict):
                    continue
                action_id = normalize_action_id(action_input.get("id"))
                action_name = normalize_action_name(action_input.get("id"))
                if action_id is None or action_name is None:
                    continue
                available_actions = [
                    action_name
                    for action_name in (
                        normalize_action_name(action_id)
                        for action_id in data.get("available_actions", [])
                    )
                    if action_name is not None
                ]
                frame = data.get("frame")
                yield TrainingStep(
                    game_id=str(data.get("game_id", "")),
                    recording_guid=str(data.get("guid", "")),
                    step_index=step_index,
                    timestamp=str(payload.get("timestamp", "")),
                    frame=frame,
                    available_actions=available_actions,
                    action_id=action_id,
                    action_name=action_name,
                    action_data=dict(action_input.get("data", {})),
                    levels_completed=int(data.get("levels_completed", 0) or 0),
                    state=str(data.get("state", "")),
                    win_levels=int(data.get("win_levels", 0) or 0),
                    full_reset=bool(data.get("full_reset", False)),
                    frame_signature=compute_frame_signature(frame),
                )


def build_action_prior_artifact(
    zip_path: str | Path,
    output_path: str | Path | None = None,
    games: set[str] | None = None,
    limit_recordings: int | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    archive_path = Path(zip_path)
    wanted_games = {game.strip() for game in games or set() if game.strip()}
    artifact: dict[str, Any] = {
        "source_zip": str(archive_path),
        "recordings_seen": 0,
        "steps_seen": 0,
        "steps_used": 0,
        "games": {},
    }
    selected_recordings = _select_recordings(
        archive_path,
        wanted_games=wanted_games,
        limit_recordings=limit_recordings,
    )
    training_rows: list[dict[str, Any]] = []
    memory_records: list[EpisodeRecord] = []
    for payload in _collect_recording_aggregates(
        archive_path,
        selected_recordings=selected_recordings,
        workers=workers,
    ):
        _merge_recording_payload(artifact, payload)
        memory_records.extend(payload.get("memory_records", []))
    if output_path is not None:
        destination = Path(output_path)
        if destination.exists():
            destination.unlink()
        initialize_memory_schema(destination)
        for game_id, game_entry in sorted(artifact["games"].items()):
            training_rows.extend(
                _count_rows_for_game(
                    game_id=game_id,
                    game_entry=game_entry,
                )
            )
        upsert_training_counts(destination, training_rows)
        record_transitions(destination, memory_records)
        upsert_metadata(
            destination,
            {
                "schema_version": "2",
                "source_zip": str(archive_path),
                "build_timestamp": datetime.now(timezone.utc).isoformat(),
                "recordings_seen": str(artifact["recordings_seen"]),
                "steps_seen": str(artifact["steps_seen"]),
                "steps_used": str(artifact["steps_used"]),
                "offline_memory_records": str(len(memory_records)),
            },
        )
    return artifact


def _increment_count(bucket: dict[str, int], action_name: str) -> None:
    bucket[action_name] = int(bucket.get(action_name, 0) or 0) + 1


def _increment_count_by(bucket: dict[str, int], action_name: str, amount: int) -> None:
    if amount <= 0:
        return
    bucket[action_name] = int(bucket.get(action_name, 0) or 0) + amount


def _new_game_entry() -> dict[str, Any]:
    return {
        "recordings": 0,
        "steps_used": 0,
        "actions": {},
        "action_supports": {},
        "states": {},
        "frame_signatures": {},
    }


def _select_recordings(
    zip_path: str | Path,
    *,
    wanted_games: set[str],
    limit_recordings: int | None,
) -> list[RecordingMeta]:
    selected: list[RecordingMeta] = []
    for recording in iter_recordings(zip_path):
        short_game_id = recording.game_id.split("-")[0]
        if wanted_games and short_game_id not in wanted_games:
            continue
        if limit_recordings is not None and len(selected) >= limit_recordings:
            break
        selected.append(recording)
    return selected


def _process_recording(
    zip_path: str | Path,
    recording_name: str,
    short_game_id: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "recordings_seen": 1,
        "steps_seen": 0,
        "steps_used": 0,
        "games": {short_game_id: _new_game_entry()},
        "memory_records": [],
    }
    game_entry = payload["games"][short_game_id]
    game_entry["recordings"] = 1
    steps = list(iter_steps(zip_path, recording_name))
    recording_bonus = _recording_quality_bonus(steps)
    recording_penalty = _recording_drag_penalty(steps)
    seen_context_actions: dict[tuple[str, str | None, str], int] = {}
    seen_game_actions: set[str] = set()
    seen_state_actions: set[tuple[str, str]] = set()
    seen_frame_actions: set[tuple[str, str]] = set()
    for index, step in enumerate(steps):
        previous_step = steps[index - 1] if index > 0 else None
        next_step = steps[index + 1] if index + 1 < len(steps) else None
        future_steps = steps[index + 1 : index + 4]
        stall_streak = _preceding_stall_streak(steps, index)
        payload["steps_seen"] += 1
        action_name = step.action_name
        if action_name is None:
            continue
        if action_name != "RESET" and action_name not in step.available_actions:
            continue
        repetition_count = _count_repeated_context_action(
            seen_context_actions,
            state_name=step.state,
            frame_signature=step.frame_signature,
            action_name=action_name,
        )
        weight = _training_weight(
            step,
            previous_step,
            next_step,
            future_steps=future_steps,
            repetition_count=repetition_count,
            recording_bonus=recording_bonus,
            recording_penalty=recording_penalty,
            stall_streak=stall_streak,
        )
        if weight <= 0:
            continue
        payload["steps_used"] += 1
        game_entry["steps_used"] += 1
        payload["memory_records"].append(
            _build_memory_record(
                step=step,
                next_step=next_step,
                future_steps=future_steps,
                stall_streak=stall_streak,
            )
        )
        _increment_count_by(game_entry["actions"], action_name, weight)
        _increment_support_once(game_entry["action_supports"], seen_game_actions, action_name)
        state_entry = game_entry["states"].setdefault(step.state, {"actions": {}})
        state_entry.setdefault("action_supports", {})
        _increment_count_by(state_entry["actions"], action_name, weight)
        _increment_support_once(
            state_entry["action_supports"],
            seen_state_actions,
            (step.state, action_name),
        )
        if step.frame_signature:
            frame_entry = game_entry["frame_signatures"].setdefault(
                step.frame_signature,
                {"actions": {}},
            )
            frame_entry.setdefault("action_supports", {})
            _increment_count_by(frame_entry["actions"], action_name, weight)
            _increment_support_once(
                frame_entry["action_supports"],
                seen_frame_actions,
                (step.frame_signature, action_name),
            )
    return payload


def _process_recording_job(job: tuple[str, str, str]) -> dict[str, Any]:
    zip_path, recording_name, short_game_id = job
    return _process_recording(zip_path, recording_name, short_game_id)


def _collect_recording_aggregates(
    zip_path: str | Path,
    *,
    selected_recordings: list[RecordingMeta],
    workers: int,
) -> list[dict[str, Any]]:
    jobs = [
        (str(Path(zip_path)), recording.recording_name, recording.game_id.split("-")[0])
        for recording in selected_recordings
    ]
    if workers <= 1 or len(jobs) <= 1:
        return [_process_recording_job(job) for job in jobs]
    max_workers = min(len(jobs), max(1, workers))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_process_recording_job, jobs))


def default_worker_count() -> int:
    return max(1, (os.cpu_count() or 1) - 1)


def _training_weight(
    step: TrainingStep,
    previous_step: TrainingStep | None,
    next_step: TrainingStep | None,
    *,
    future_steps: list[TrainingStep] | None = None,
    repetition_count: int = 0,
    recording_bonus: int = 0,
    recording_penalty: int = 0,
    stall_streak: int = 0,
) -> int:
    future_steps = future_steps or []
    outcome = _infer_step_outcome(step, next_step)
    future_progress_delta, progress_distance = _lookahead_progress(step, future_steps)
    future_failure_distance = _lookahead_failure_distance(step, future_steps)
    branching_bonus = _branching_difficulty_bonus(step)
    fragility_penalty = _fragile_progress_penalty(future_failure_distance)
    endgame_bonus = _endgame_conversion_bonus(step, next_step, future_steps)
    momentum_bonus = _progress_momentum_bonus(step, next_step, future_steps)
    stability_bonus = _progress_stability_bonus(step, next_step, future_steps)
    regression_penalty = _progress_regression_penalty(step, next_step, future_steps)
    pre_win_setup_bonus = _pre_win_setup_bonus(next_step, future_steps)
    win_bonus = _terminal_win_bonus(next_step, future_steps)
    break_stall_bonus = _break_stall_bonus(stall_streak, outcome, future_progress_delta)
    recovery_bonus = _post_reset_recovery_bonus(previous_step, outcome, future_progress_delta)
    if step.action_name == "RESET":
        if not _is_meaningful_reset(step):
            return 0
        base_weight = 2 if repetition_count == 0 else 1
        return max(1, base_weight + min(1, recording_bonus) - recording_penalty)
    if outcome == "failed" and future_progress_delta <= 0:
        return 0
    if outcome == "progress":
        progress_delta = max(0, _step_progress_delta(step, next_step))
        base_weight = 3 + min(2, progress_delta)
        return max(
            1,
            (base_weight - repetition_count)
            + recording_bonus
            - recording_penalty
            + branching_bonus
            + endgame_bonus
            + momentum_bonus
            + stability_bonus
            + win_bonus
            + break_stall_bonus
            + recovery_bonus
            - fragility_penalty
            - regression_penalty,
        )
    if future_progress_delta > 0 and (
        future_failure_distance is None or progress_distance <= future_failure_distance
    ):
        base_weight = 2 + min(2, future_progress_delta)
        distance_penalty = max(0, progress_distance - 1)
        return max(
            1,
            (base_weight - repetition_count - distance_penalty)
            + recording_bonus
            - recording_penalty
            + branching_bonus
            + endgame_bonus
            + momentum_bonus
            + pre_win_setup_bonus
            + win_bonus
            + break_stall_bonus
            + recovery_bonus
            - fragility_penalty,
        )
    if _is_noop_transition(step, next_step):
        return 0
    if repetition_count > 0:
        return 0
    return max(0, 1 - recording_penalty)


def _recording_quality_bonus(steps: list[TrainingStep]) -> int:
    if not steps:
        return 0
    if any(_is_win_state(step.state) for step in steps):
        return 2
    best_levels_completed = max(int(step.levels_completed or 0) for step in steps)
    max_win_levels = max(int(step.win_levels or 0) for step in steps)
    if best_levels_completed <= 0 or max_win_levels <= 0:
        return 0
    progress_ratio = best_levels_completed / max_win_levels
    if progress_ratio >= 0.5:
        return 2
    if progress_ratio >= 0.2:
        return 1
    return 0


def _recording_drag_penalty(steps: list[TrainingStep]) -> int:
    if len(steps) < 6:
        return 0
    best_levels_completed = max(int(step.levels_completed or 0) for step in steps) if steps else 0
    if any(_is_win_state(step.state) for step in steps):
        return 0
    if best_levels_completed <= 0:
        return 1
    if len(steps) > best_levels_completed * 6:
        return 1
    return 0


def _branching_difficulty_bonus(step: TrainingStep) -> int:
    branch_count = len(step.available_actions)
    if branch_count >= 5:
        return 1
    return 0


def _fragile_progress_penalty(future_failure_distance: int | None) -> int:
    if future_failure_distance is None:
        return 0
    if future_failure_distance <= 2:
        return 2
    if future_failure_distance == 3:
        return 1
    return 0


def _endgame_conversion_bonus(
    step: TrainingStep,
    next_step: TrainingStep | None,
    future_steps: list[TrainingStep],
) -> int:
    if step.win_levels <= 0:
        return 0
    candidate_steps = [candidate for candidate in [next_step, *future_steps] if candidate is not None]
    if any(_is_win_state(candidate.state) for candidate in candidate_steps):
        return 2
    best_future_levels = max(
        [int(step.levels_completed or 0)]
        + [int(candidate.levels_completed or 0) for candidate in candidate_steps]
    )
    if best_future_levels >= max(1, step.win_levels - 1):
        return 1
    return 0


def _progress_momentum_bonus(
    step: TrainingStep,
    next_step: TrainingStep | None,
    future_steps: list[TrainingStep],
) -> int:
    candidate_steps = [candidate for candidate in [next_step, *future_steps] if candidate is not None]
    progress_hits = 0
    best_seen = int(step.levels_completed or 0)
    for candidate in candidate_steps:
        levels_completed = int(candidate.levels_completed or 0)
        if _is_win_state(candidate.state) or levels_completed > best_seen:
            progress_hits += 1
            best_seen = max(best_seen, levels_completed)
    if progress_hits >= 2:
        return 1
    return 0


def _progress_stability_bonus(
    step: TrainingStep,
    next_step: TrainingStep | None,
    future_steps: list[TrainingStep],
) -> int:
    if next_step is None or _step_progress_delta(step, next_step) <= 0:
        return 0
    sustained_level = int(next_step.levels_completed or 0)
    stable_follow_ups = 0
    for candidate in future_steps:
        if _is_reset_state(candidate.state) or candidate.full_reset:
            break
        if int(candidate.levels_completed or 0) < sustained_level:
            break
        stable_follow_ups += 1
    if stable_follow_ups >= 2:
        return 1
    return 0


def _progress_regression_penalty(
    step: TrainingStep,
    next_step: TrainingStep | None,
    future_steps: list[TrainingStep],
) -> int:
    if next_step is None or _step_progress_delta(step, next_step) <= 0:
        return 0
    gained_level = int(next_step.levels_completed or 0)
    for distance, candidate in enumerate(future_steps, start=1):
        if _is_win_state(candidate.state):
            return 0
        if _is_reset_state(candidate.state) or candidate.full_reset:
            return 0
        if int(candidate.levels_completed or 0) < gained_level:
            if distance <= 2:
                return 2
            return 1
    return 0


def _pre_win_setup_bonus(
    next_step: TrainingStep | None,
    future_steps: list[TrainingStep],
) -> int:
    if next_step is not None and _is_win_state(next_step.state):
        return 0
    for distance, candidate in enumerate(future_steps, start=1):
        if _is_win_state(candidate.state):
            if distance <= 2:
                return 1
            return 0
    return 0


def _terminal_win_bonus(
    next_step: TrainingStep | None,
    future_steps: list[TrainingStep],
) -> int:
    candidate_steps = [candidate for candidate in [next_step, *future_steps] if candidate is not None]
    if any(_is_win_state(candidate.state) for candidate in candidate_steps):
        return 1
    return 0


def _break_stall_bonus(
    stall_streak: int,
    outcome: str,
    future_progress_delta: int,
) -> int:
    if outcome != "progress" and future_progress_delta <= 0:
        return 0
    if stall_streak >= 4:
        return 2
    if stall_streak >= 2:
        return 1
    return 0


def _preceding_stall_streak(
    steps: list[TrainingStep],
    index: int,
) -> int:
    if index <= 0:
        return 0
    current = steps[index]
    streak = 0
    cursor = index - 1
    while cursor >= 0:
        previous = steps[cursor]
        if _is_win_state(previous.state) or _is_reset_state(previous.state):
            break
        if int(previous.levels_completed or 0) != int(current.levels_completed or 0):
            break
        streak += 1
        cursor -= 1
    return streak


def _post_reset_recovery_bonus(
    previous_step: TrainingStep | None,
    outcome: str,
    future_progress_delta: int,
) -> int:
    if previous_step is None or previous_step.action_name != "RESET":
        return 0
    if outcome == "progress" or future_progress_delta > 0:
        return 1
    return 0


def _count_repeated_context_action(
    seen_context_actions: dict[tuple[str, str | None, str], int],
    *,
    state_name: str,
    frame_signature: str | None,
    action_name: str,
) -> int:
    key = (state_name, frame_signature, action_name)
    current = int(seen_context_actions.get(key, 0) or 0)
    seen_context_actions[key] = current + 1
    return current


def _increment_support_once(
    bucket: dict[str, int],
    seen_keys: set[Any],
    key: Any,
) -> None:
    if key in seen_keys:
        return
    seen_keys.add(key)
    action_name = key if isinstance(key, str) else str(key[-1])
    bucket[action_name] = int(bucket.get(action_name, 0) or 0) + 1


def _infer_step_outcome(
    step: TrainingStep,
    next_step: TrainingStep | None,
) -> str:
    progress_delta = _step_progress_delta(step, next_step)
    state_name = next_step.state if next_step is not None else step.state
    full_reset = bool(next_step.full_reset if next_step is not None else step.full_reset)
    if _is_win_state(state_name) or progress_delta > 0:
        return "progress"
    if _is_reset_state(state_name) or full_reset:
        return "failed"
    return "stalled"


def _step_progress_delta(
    step: TrainingStep,
    next_step: TrainingStep | None,
) -> int:
    later_levels = next_step.levels_completed if next_step is not None else step.levels_completed
    return int(later_levels) - int(step.levels_completed)


def _lookahead_progress(
    step: TrainingStep,
    future_steps: list[TrainingStep],
) -> tuple[int, int]:
    for distance, future_step in enumerate(future_steps, start=1):
        progress_delta = int(future_step.levels_completed) - int(step.levels_completed)
        if _is_win_state(future_step.state) or progress_delta > 0:
            return progress_delta, distance
    return 0, 0


def _lookahead_failure_distance(
    step: TrainingStep,
    future_steps: list[TrainingStep],
) -> int | None:
    for distance, future_step in enumerate(future_steps, start=1):
        if _is_reset_state(future_step.state) or future_step.full_reset:
            return distance
    return None


def _is_meaningful_reset(step: TrainingStep) -> bool:
    return _is_reset_state(step.state) or step.full_reset


def _is_noop_transition(
    step: TrainingStep,
    next_step: TrainingStep | None,
) -> bool:
    if next_step is None:
        return False
    return (
        step.state == next_step.state
        and step.levels_completed == next_step.levels_completed
        and step.frame_signature is not None
        and step.frame_signature == next_step.frame_signature
    )


def _merge_recording_payload(
    artifact: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    artifact["recordings_seen"] += int(payload.get("recordings_seen", 0) or 0)
    artifact["steps_seen"] += int(payload.get("steps_seen", 0) or 0)
    artifact["steps_used"] += int(payload.get("steps_used", 0) or 0)
    artifact.setdefault("offline_memory_records", 0)
    artifact["offline_memory_records"] += len(payload.get("memory_records", []))
    for game_id, game_payload in sorted(payload.get("games", {}).items()):
        target_game_entry = artifact["games"].setdefault(game_id, _new_game_entry())
        target_game_entry["recordings"] += int(game_payload.get("recordings", 0) or 0)
        target_game_entry["steps_used"] += int(game_payload.get("steps_used", 0) or 0)
        for action_name, count in sorted(game_payload.get("actions", {}).items()):
            target_game_entry["actions"][action_name] = (
                int(target_game_entry["actions"].get(action_name, 0) or 0) + int(count or 0)
            )
        for action_name, count in sorted(game_payload.get("action_supports", {}).items()):
            target_game_entry["action_supports"][action_name] = (
                int(target_game_entry["action_supports"].get(action_name, 0) or 0)
                + int(count or 0)
            )
        for state_name, state_entry in sorted(game_payload.get("states", {}).items()):
            target_state_entry = target_game_entry["states"].setdefault(state_name, {"actions": {}})
            target_state_entry.setdefault("action_supports", {})
            for action_name, count in sorted(state_entry.get("actions", {}).items()):
                target_state_entry["actions"][action_name] = (
                    int(target_state_entry["actions"].get(action_name, 0) or 0)
                    + int(count or 0)
                )
            for action_name, count in sorted(state_entry.get("action_supports", {}).items()):
                target_state_entry["action_supports"][action_name] = (
                    int(target_state_entry["action_supports"].get(action_name, 0) or 0)
                    + int(count or 0)
                )
        for frame_signature, frame_entry in sorted(game_payload.get("frame_signatures", {}).items()):
            target_frame_entry = target_game_entry["frame_signatures"].setdefault(
                frame_signature,
                {"actions": {}},
            )
            target_frame_entry.setdefault("action_supports", {})
            for action_name, count in sorted(frame_entry.get("actions", {}).items()):
                target_frame_entry["actions"][action_name] = (
                    int(target_frame_entry["actions"].get(action_name, 0) or 0)
                    + int(count or 0)
                )
            for action_name, count in sorted(frame_entry.get("action_supports", {}).items()):
                target_frame_entry["action_supports"][action_name] = (
                    int(target_frame_entry["action_supports"].get(action_name, 0) or 0)
                    + int(count or 0)
                )


def _build_memory_record(
    *,
    step: TrainingStep,
    next_step: TrainingStep | None,
    future_steps: list[TrainingStep],
    stall_streak: int,
) -> EpisodeRecord:
    outcome = _infer_step_outcome(step, next_step)
    progress_delta = _step_progress_delta(step, next_step)
    next_state = next_step.state if next_step is not None else step.state
    next_frame_signature = next_step.frame_signature if next_step is not None else step.frame_signature
    return EpisodeRecord(
        game_id=step.game_id.split("-")[0],
        step_index=int(step.step_index),
        action_name=str(step.action_name or ""),
        outcome=outcome,
        progress_delta=int(progress_delta),
        game_state=next_state,
        prior_game_state=step.state,
        levels_completed=int(next_step.levels_completed if next_step is not None else step.levels_completed),
        prior_frame_signature=step.frame_signature,
        next_game_state=next_state,
        next_frame_signature=next_frame_signature,
        prior_legal_actions=list(step.available_actions),
        stagnation_before=max(0, stall_streak),
        stagnation_after=max(0, stall_streak + (0 if outcome == "progress" else 1)),
        decision_subgoal=_infer_training_subgoal(
            step=step,
            next_step=next_step,
            future_steps=future_steps,
            outcome=outcome,
            stall_streak=stall_streak,
        ),
        decision_source="offline-demo-trainer",
    )


def _infer_training_subgoal(
    *,
    step: TrainingStep,
    next_step: TrainingStep | None,
    future_steps: list[TrainingStep],
    outcome: str,
    stall_streak: int,
) -> str:
    if step.action_name == "RESET" or _is_reset_state(step.state):
        return "re-enter-level"
    if outcome == "progress":
        return "advance-level"
    if stall_streak >= 2:
        return "break-stall"
    future_progress_delta, _ = _lookahead_progress(step, future_steps)
    if future_progress_delta > 0:
        return "advance-level"
    if next_step is not None and _is_win_state(next_step.state):
        return "hold-win"
    return "break-stall"


def _count_rows_for_game(
    game_id: str,
    game_entry: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for action_name, count in sorted(
        _apply_recording_support_bonus(
            game_entry.get("actions", {}),
            game_entry.get("action_supports", {}),
        ).items()
    ):
        rows.append(
            {
                "scope": "game",
                "game_id": game_id,
                "action_name": action_name,
                "count": count,
            }
        )
    for state_name, state_entry in sorted(game_entry.get("states", {}).items()):
        for action_name, count in sorted(
            _consensus_filter_actions(
                _apply_recording_support_bonus(
                    state_entry.get("actions", {}),
                    state_entry.get("action_supports", {}),
                )
            ).items()
        ):
            rows.append(
                {
                    "scope": "state",
                    "game_id": game_id,
                    "state": state_name,
                    "action_name": action_name,
                    "count": count,
                }
            )
    for frame_signature, frame_entry in sorted(game_entry.get("frame_signatures", {}).items()):
        for action_name, count in sorted(
            _consensus_filter_actions(
                _apply_recording_support_bonus(
                    frame_entry.get("actions", {}),
                    frame_entry.get("action_supports", {}),
                )
            ).items()
        ):
            rows.append(
                {
                    "scope": "frame",
                    "game_id": game_id,
                    "frame_signature": frame_signature,
                    "action_name": action_name,
                    "count": count,
                }
            )
    return rows


def _apply_recording_support_bonus(
    actions: dict[str, int],
    supports: dict[str, int],
) -> dict[str, int]:
    adjusted: dict[str, int] = {}
    for action_name, count in actions.items():
        base = int(count or 0)
        if base <= 0:
            continue
        support_count = int(supports.get(action_name, 0) or 0)
        adjusted[str(action_name)] = base + min(2, max(0, support_count - 1))
    return adjusted


def _consensus_filter_actions(actions: dict[str, int]) -> dict[str, int]:
    normalized = {
        str(action_name): int(count or 0)
        for action_name, count in actions.items()
        if int(count or 0) > 0
    }
    if len(normalized) <= 1:
        return normalized
    ranked = sorted(
        normalized.items(),
        key=lambda item: (-int(item[1]), str(item[0])),
    )
    leader_name, leader_count = ranked[0]
    runner_up_count = int(ranked[1][1])
    margin = int(leader_count) - runner_up_count
    if margin <= 0:
        return {}
    return {str(leader_name): margin}


def _is_recording_entry(name: str) -> bool:
    if name.startswith("__MACOSX/"):
        return False
    if "/__MACOSX/" in name:
        return False
    return name.endswith(".recording.jsonl")
