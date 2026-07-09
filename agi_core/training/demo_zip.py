"""Offline training pipeline for ARC-AGI-3.

Processes human-demo recording ZIP archives into SQLite training artifacts:
action-count priors (4 scopes), episodic memory records, abstract memories,
and board-policy parameters.  Designed for ``scripts/train_offline_from_zip.py``.

Key functions:
    ``build_action_prior_artifact()`` — main entry point.
    ``iter_recordings()`` / ``iter_steps()`` — low-level ZIP traversal.
    ``normalize_action_name()`` / ``normalize_action_id()`` — action ID normalization.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import zipfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from datetime import datetime, timezone
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

from agi_core.config import RuntimeConfig
from agi_core.image_input import (
    BoardImageInput,
    board_image_input_from_source,
    default_click_target,
    extract_board_motif_tags,
    extract_board_image_input,
)
from agi_core.storage import (
    initialize_memory_schema,
    record_transitions,
    upsert_abstract_memory_rows,
    upsert_board_policy_params,
    upsert_chunk,
    upsert_cognition_traces,
    upsert_metadata,
    upsert_training_counts,
)
from agi_core.types import EpisodeRecord
from agi_core.cognition import (
    AttentionFocus,
    ChunkLibrary,
    ConsolidatedMemory,
    GoalStack,
    MetaCognition,
    WorkingMemory,
)

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
    image_input: BoardImageInput | None = None


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
    return normalize_visible_board(frame)


def compute_frame_signature(frame: Any) -> str | None:
    image_input = board_image_input_from_source(frame, source_attr="frame")
    if image_input is None:
        return None
    return image_input.frame_signature


def runtime_frame_signature(raw_frame_ref: Any) -> str | None:
    image_input = extract_board_image_input(raw_frame_ref)
    if image_input is None:
        return None
    return image_input.frame_signature


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
                action_name = normalize_action_name(action_input.get("id"))
                if action_name is None:
                    continue
                action_id = normalize_action_id(action_name)
                if action_id is None:
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
                image_input = board_image_input_from_source(frame, source_attr="frame")
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
                    frame_signature=image_input.frame_signature if image_input is not None else None,
                    image_input=image_input,
                )


def build_action_prior_artifact(
    zip_path: str | Path,
    output_path: str | Path | None = None,
    games: set[str] | None = None,
    limit_recordings: int | None = None,
    workers: int = 1,
    build_abstractions: bool = True,
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
    abstract_rows: list[dict[str, Any]] = []
    for payload in _collect_recording_aggregates(
        archive_path,
        selected_recordings=selected_recordings,
        workers=workers,
    ):
        _merge_recording_payload(artifact, payload)
        memory_records.extend(payload.get("memory_records", []))
        if build_abstractions:
            abstract_rows.extend(payload.get("abstract_rows", []))
    if output_path is not None:
        destination = Path(output_path)
        _assert_safe_output_database(destination)
        initialize_memory_schema(destination)
        _reset_output_database(destination)
        for game_id, game_entry in sorted(artifact["games"].items()):
            training_rows.extend(
                _count_rows_for_game(
                    game_id=game_id,
                    game_entry=game_entry,
                )
            )
        upsert_training_counts(destination, training_rows)
        record_transitions(destination, memory_records)
        if build_abstractions:
            upsert_abstract_memory_rows(destination, abstract_rows)
        policy_rows = _extract_board_policy_params(artifact["games"])
        upsert_board_policy_params(destination, policy_rows)
        upsert_metadata(
            destination,
            {
                "schema_version": "2",
                "source_zip": str(archive_path),
                "build_timestamp": datetime.now(timezone.utc).isoformat(),
                "recordings_seen": artifact["recordings_seen"],
                "steps_seen": artifact["steps_seen"],
                "steps_used": artifact["steps_used"],
                "offline_memory_records": len(memory_records),
                "abstract_memory_rows": len(abstract_rows),
            },
        )
    artifact["abstract_memory_rows"] = len(abstract_rows)

    # ── Post-hoc cognition pass (sequential) ──────────────────────
    if output_path is not None:
        traces, chunks_by_name = _post_hoc_cognition_pass(
            archive_path, selected_recordings,
        )
        if traces:
            upsert_cognition_traces(destination, traces)
        if chunks_by_name:
            for name, data in chunks_by_name.items():
                upsert_chunk(destination, name, data)
        artifact["cognition_traces_count"] = len(traces)
        artifact["chunk_count"] = len(chunks_by_name)
        if traces or chunks_by_name:
            existing_meta = {}
            with closing(sqlite3.connect(Path(destination))) as conn:
                try:
                    rows = conn.execute("SELECT key, value FROM metadata").fetchall()
                    existing_meta = {str(r[0]): str(r[1]) for r in rows}
                except Exception:
                    pass
            existing_meta["cognition_traces_count"] = str(len(traces))
            existing_meta["chunk_count"] = str(len(chunks_by_name))
            upsert_metadata(destination, existing_meta)

    return artifact


def _assert_safe_output_database(output_path: str | Path) -> None:
    destination = Path(output_path).resolve()
    config = RuntimeConfig.from_env(Path(__file__).resolve().parents[2])
    protected_paths = {
        Path(config.memory_db_path).resolve(),
        Path(config.archive_memory_db_path).resolve(),
    }
    if destination in protected_paths:
        raise ValueError(
            "refusing to overwrite live memory database; choose a dedicated training artifact path"
        )


def _reset_output_database(db_path: str | Path) -> None:
    """Drop all data rows (preserves schema) via per-table DELETE."""
    _TABLES_TO_CLEAR = [
        "game_action_priors", "state_action_priors", "frame_action_priors", "subgoal_action_priors",
        "live_episode_records", "live_action_stats", "live_state_action_stats",
        "live_frame_action_stats", "live_subgoal_action_stats", "live_basin_stats",
        "live_episode_summaries", "live_pattern_stats", "live_sequence_stats",
        "live_board_cluster_stats", "live_semantic_memory", "live_procedural_memory",
        "abstract_memories", "cognition_traces", "chunks", "metadata",
    ]
    with closing(sqlite3.connect(Path(db_path))) as connection:
        for table in _TABLES_TO_CLEAR:
            connection.execute(f"DELETE FROM {table}")
        connection.commit()


def _increment_count(bucket: dict[str, int], action_name: str) -> None:
    bucket[action_name] = _safe_int(bucket.get(action_name)) + 1


def _increment_count_by(bucket: dict[str, int], action_name: str, amount: int) -> None:
    if amount <= 0:
        return
    bucket[action_name] = _safe_int(bucket.get(action_name)) + amount


def _new_game_entry() -> dict[str, Any]:
    return {
        "recordings": 0,
        "steps_used": 0,
        "actions": {},
        "action_supports": {},
        "states": {},
        "subgoals": {},
        "frame_signatures": {},
    }


def _select_recordings(
    zip_path: str | Path,
    *,
    wanted_games: set[str],
    limit_recordings: int | None,
) -> list[RecordingMeta]:
    """Select recordings, distributing ``limit_recordings`` evenly across games."""
    all_recordings: dict[str, list[RecordingMeta]] = {}
    for recording in iter_recordings(zip_path):
        short_game_id = recording.game_id.split("-")[0]
        if wanted_games and short_game_id not in wanted_games:
            continue
        all_recordings.setdefault(short_game_id, []).append(recording)
    if limit_recordings is None or not all_recordings:
        return [rec for recs in all_recordings.values() for rec in recs]
    per_game = max(1, limit_recordings // len(all_recordings))
    selected: list[RecordingMeta] = []
    for game_id in sorted(all_recordings):
        selected.extend(all_recordings[game_id][:per_game])
        if len(selected) >= limit_recordings:
            break
    return selected[:limit_recordings]


def _apply_step_counts(
    game_entry: dict[str, Any],
    action_name: str,
    weight: int,
    step: TrainingStep,
    decision_subgoal: str | None,
    seen: dict[str, set],
) -> None:
    """Record weighted action counts at game, state, subgoal, and frame levels."""
    _increment_count_by(game_entry["actions"], action_name, weight)
    _increment_support_once(game_entry["action_supports"], seen["game"], action_name)
    state_entry = game_entry["states"].setdefault(step.state, {"actions": {}, "action_supports": {}})
    _increment_count_by(state_entry["actions"], action_name, weight)
    _increment_support_once(state_entry["action_supports"], seen["state"], (step.state, action_name))
    if decision_subgoal:
        sg_entry = game_entry["subgoals"].setdefault(decision_subgoal, {"actions": {}, "action_supports": {}})
        _increment_count_by(sg_entry["actions"], action_name, weight)
        _increment_support_once(sg_entry["action_supports"], seen["subgoal"], (decision_subgoal, action_name))
    if step.frame_signature:
        fs_entry = game_entry["frame_signatures"].setdefault(step.frame_signature, {"actions": {}, "action_supports": {}})
        _increment_count_by(fs_entry["actions"], action_name, weight)
        _increment_support_once(fs_entry["action_supports"], seen["frame"], (step.frame_signature, action_name))


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
        "abstract_rows": [],
    }
    game_entry = payload["games"][short_game_id]
    game_entry["recordings"] = 1
    steps = list(iter_steps(zip_path, recording_name))
    recording_bonus = _recording_quality_bonus(steps)
    recording_penalty = _recording_drag_penalty(steps)
    seen_context_actions: dict[tuple[str, str | None, str], int] = {}
    seen: dict[str, set] = {"game": set(), "state": set(), "subgoal": set(), "frame": set()}
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
            step, previous_step, next_step,
            future_steps=future_steps,
            repetition_count=repetition_count,
            recording_bonus=recording_bonus,
            recording_penalty=recording_penalty,
            stall_streak=stall_streak,
        )
        if weight <= 0:
            continue
        decision_subgoal = _infer_training_subgoal(
            step=step, next_step=next_step, future_steps=future_steps,
            outcome=_infer_step_outcome(step, next_step),
            stall_streak=stall_streak,
        )
        payload["steps_used"] += 1
        game_entry["steps_used"] += 1
        payload["memory_records"].append(
            _build_memory_record(step=step, next_step=next_step, future_steps=future_steps,
                                 stall_streak=stall_streak, decision_subgoal=decision_subgoal)
        )
        abstract_row = _build_abstract_memory_row(
            short_game_id=short_game_id, steps=steps, index=index,
            step=step, next_step=next_step, future_steps=future_steps,
            decision_subgoal=decision_subgoal,
        )
        if abstract_row is not None:
            payload["abstract_rows"].append(abstract_row)
        _apply_step_counts(game_entry, action_name, weight, step, decision_subgoal, seen)
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


def _compute_training_bonuses(
    step: TrainingStep,
    previous_step: TrainingStep | None,
    next_step: TrainingStep | None,
    future_steps: list[TrainingStep],
    stall_streak: int,
    outcome: str,
    future_progress_delta: int,
) -> dict[str, Any]:
    """Compute all per-step bonuses and penalties for training weight calculation."""
    future_failure_distance = _lookahead_failure_distance(step, future_steps)
    return {
        "branching_bonus": _branching_difficulty_bonus(step),
        "fragility_penalty": _fragile_progress_penalty(future_failure_distance),
        "endgame_bonus": _endgame_conversion_bonus(step, next_step, future_steps),
        "momentum_bonus": _progress_momentum_bonus(step, next_step, future_steps),
        "stability_bonus": _progress_stability_bonus(step, next_step, future_steps),
        "regression_penalty": _progress_regression_penalty(step, next_step, future_steps),
        "pre_win_setup_bonus": _pre_win_setup_bonus(next_step, future_steps),
        "win_bonus": _terminal_win_bonus(next_step, future_steps),
        "break_stall_bonus": _break_stall_bonus(stall_streak, outcome, future_progress_delta),
        "recovery_bonus": _post_reset_recovery_bonus(previous_step, outcome, future_progress_delta),
        "future_failure_distance": future_failure_distance,
    }


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
    b = _compute_training_bonuses(step, previous_step, next_step, future_steps, stall_streak, outcome, future_progress_delta)
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
            + recording_bonus - recording_penalty
            + b["branching_bonus"] + b["endgame_bonus"] + b["momentum_bonus"]
            + b["stability_bonus"] + b["win_bonus"] + b["break_stall_bonus"]
            + b["recovery_bonus"] - b["fragility_penalty"] - b["regression_penalty"],
        )
    if future_progress_delta > 0 and (
        b["future_failure_distance"] is None or progress_distance <= b["future_failure_distance"]
    ):
        base_weight = 2 + min(2, future_progress_delta)
        distance_penalty = max(0, progress_distance - 1)
        return max(
            1,
            (base_weight - repetition_count - distance_penalty)
            + recording_bonus - recording_penalty
            + b["branching_bonus"] + b["endgame_bonus"] + b["momentum_bonus"]
            + b["pre_win_setup_bonus"] + b["win_bonus"] + b["break_stall_bonus"]
            + b["recovery_bonus"] - b["fragility_penalty"],
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
    best_levels_completed = max(_safe_int(step.levels_completed) for step in steps)
    max_win_levels = max(_safe_int(step.win_levels) for step in steps)
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
    best_levels_completed = max(_safe_int(step.levels_completed) for step in steps) if steps else 0
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


def _candidate_steps(next_step: TrainingStep | None, future_steps: list[TrainingStep]) -> list[TrainingStep]:
    """Build list of non-None candidates from next_step + future_steps."""
    return [c for c in ([next_step] + future_steps) if c is not None]


def _endgame_conversion_bonus(
    step: TrainingStep,
    next_step: TrainingStep | None,
    future_steps: list[TrainingStep],
) -> int:
    if step.win_levels <= 0:
        return 0
    candidate_steps = _candidate_steps(next_step, future_steps)
    if any(_is_win_state(candidate.state) for candidate in candidate_steps):
        return 2
    best_future_levels = max(
        [_safe_int(step.levels_completed)]
        + [_safe_int(c.levels_completed) for c in candidate_steps]
    )
    if best_future_levels >= max(1, step.win_levels - 1):
        return 1
    return 0


def _progress_momentum_bonus(
    step: TrainingStep,
    next_step: TrainingStep | None,
    future_steps: list[TrainingStep],
) -> int:
    candidate_steps = _candidate_steps(next_step, future_steps)
    progress_hits = 0
    best_seen = _safe_int(step.levels_completed)
    for candidate in candidate_steps:
        levels_completed = _safe_int(candidate.levels_completed)
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
    sustained_level = _safe_int(next_step.levels_completed)
    stable_follow_ups = 0
    for candidate in future_steps:
        if _is_terminal_step(candidate):
            break
        if _safe_int(candidate.levels_completed) < sustained_level:
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
    gained_level = _safe_int(next_step.levels_completed)
    for distance, candidate in enumerate(future_steps, start=1):
        if _is_terminal_step(candidate):
            return 0
        if _safe_int(candidate.levels_completed) < gained_level:
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
    candidate_steps = _candidate_steps(next_step, future_steps)
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
        if _is_terminal_step(previous):
            break
        if _safe_int(previous.levels_completed) != _safe_int(current.levels_completed):
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
    current = _safe_int(seen_context_actions.get(key))
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
    bucket[action_name] = _safe_int(bucket.get(action_name)) + 1


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
    later_levels = _next_or_current("levels_completed", step, next_step)
    return _safe_int(later_levels) - _safe_int(step.levels_completed)


def _lookahead_progress(
    step: TrainingStep,
    future_steps: list[TrainingStep],
) -> tuple[int, int]:
    base = _safe_int(step.levels_completed)
    for distance, future_step in enumerate(future_steps, start=1):
        progress_delta = _safe_int(future_step.levels_completed) - base
        if _is_win_state(future_step.state) or progress_delta > 0:
            return progress_delta, distance
    return 0, 0


def _lookahead_failure_distance(
    step: TrainingStep,
    future_steps: list[TrainingStep],
) -> int | None:
    for distance, future_step in enumerate(future_steps, start=1):
        if _is_terminal_step(future_step):
            return distance
    return None


def _is_terminal_step(step: TrainingStep) -> bool:
    """True if the step is a win or reset state (recording boundary)."""
    return _is_win_state(step.state) or _is_reset_state(step.state) or step.full_reset


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
        and _safe_int(step.levels_completed) == _safe_int(next_step.levels_completed)
        and step.frame_signature is not None
        and step.frame_signature == next_step.frame_signature
    )


def _merge_nested_counts(target: dict[str, Any], source: dict[str, Any], *keys: str) -> None:
    """Merge action counts from ``source[*keys]`` into ``target[*keys]``.
    
    Handles both flat dicts (``{"ACTION1": 5, ...}`` at the top level)
    and nested dicts (``{"state1": {"actions": {...}, "action_supports": {...}}}``).
    """
    for key in keys:
        items = source.get(key, {})
        if not isinstance(items, dict):
            continue
        for name, entry in sorted(items.items()):
            if isinstance(entry, int):
                # Flat dict: entry is the count directly
                target.setdefault(key, {})[name] = _safe_int(target.get(key, {}).get(name, 0)) + entry
            elif isinstance(entry, dict):
                # Nested dict
                target_entry = target.setdefault(key, {}).setdefault(name, {"actions": {}, "action_supports": {}})
                for bucket in ("actions", "action_supports"):
                    for action_name, count in sorted(entry.get(bucket, {}).items()):
                        target_entry[bucket][action_name] = (
                            _safe_int(target_entry[bucket].get(action_name)) + _safe_int(count)
                        )


def _safe_int(value: Any) -> int:
    """Safely coerce to int, treating None/empty as 0."""
    return int(value or 0)


def _safe_add(target: dict[str, Any], key: str, value: int) -> None:
    target[key] = _safe_int(target.get(key)) + value


def _merge_recording_payload(
    artifact: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    _safe_add(artifact, "recordings_seen", _safe_int(payload.get("recordings_seen")))
    _safe_add(artifact, "steps_seen", _safe_int(payload.get("steps_seen")))
    _safe_add(artifact, "steps_used", _safe_int(payload.get("steps_used")))
    _safe_add(artifact, "offline_memory_records", len(payload.get("memory_records", [])))
    _safe_add(artifact, "abstract_memory_rows", len(payload.get("abstract_rows", [])))
    for game_id, game_payload in sorted(payload.get("games", {}).items()):
        target_game_entry = artifact["games"].setdefault(game_id, _new_game_entry())
        target_game_entry["recordings"] += _safe_int(game_payload.get("recordings"))
        target_game_entry["steps_used"] += _safe_int(game_payload.get("steps_used"))
        # Merge top-level game actions
        _merge_nested_counts(target_game_entry, game_payload, "actions", "action_supports")
        # Merge per-state, per-subgoal, per-frame actions
        _merge_nested_counts(target_game_entry, game_payload, "states", "subgoals", "frame_signatures")


def _count_active_cells(image_input: BoardImageInput | None) -> int:
    """Count non-zero cells in a board image."""
    if image_input is None:
        return 0
    return sum(1 for row in image_input.board for value in row if _safe_int(value) != 0)


def _next_or_current(attr: str, step: TrainingStep, next_step: TrainingStep | None) -> Any:
    """Return ``next_step.attr`` if available, else ``step.attr``."""
    return getattr(next_step, attr) if next_step is not None else getattr(step, attr)


def _build_memory_record(
    *,
    step: TrainingStep,
    next_step: TrainingStep | None,
    future_steps: list[TrainingStep],
    stall_streak: int,
    decision_subgoal: str | None = None,
) -> EpisodeRecord:
    outcome = _infer_step_outcome(step, next_step)
    progress_delta = _step_progress_delta(step, next_step)
    next_state = _next_or_current("state", step, next_step)
    next_frame_signature = _next_or_current("frame_signature", step, next_step)
    img = step.image_input
    board_height = _safe_int(img.height if img is not None else 0)
    board_width = _safe_int(img.width if img is not None else 0)
    board_active_cells = _count_active_cells(img)
    board_source_attr = None
    board_default_probe: dict[str, int] = {}
    if img is not None:
        board_source_attr = img.source_attr
        board_default_probe = default_click_target(img)
    return EpisodeRecord(
        game_id=step.game_id.split("-")[0],
        step_index=int(step.step_index),
        action_name=str(step.action_name or ""),
        outcome=outcome,
        progress_delta=int(progress_delta),
        game_state=next_state,
        prior_game_state=step.state,
        levels_completed=int(_next_or_current("levels_completed", step, next_step)),
        prior_frame_signature=step.frame_signature,
        next_game_state=next_state,
        next_frame_signature=next_frame_signature,
        prior_legal_actions=list(step.available_actions),
        stagnation_before=max(0, stall_streak),
        stagnation_after=max(0, stall_streak + (0 if outcome == "progress" else 1)),
        decision_subgoal=decision_subgoal
        or _infer_training_subgoal(
            step=step,
            next_step=next_step,
            future_steps=future_steps,
            outcome=outcome,
            stall_streak=stall_streak,
        ),
        decision_source="offline-demo-trainer",
        prior_board_height=board_height,
        prior_board_width=board_width,
        prior_board_active_cells=board_active_cells,
        prior_board_source_attr=board_source_attr,
        prior_board_default_probe=board_default_probe,
    )


def _build_abstract_memory_row(
    *,
    short_game_id: str,
    steps: list[TrainingStep],
    index: int,
    step: TrainingStep,
    next_step: TrainingStep | None,
    future_steps: list[TrainingStep],
    decision_subgoal: str | None,
) -> dict[str, Any] | None:
    outcome = _infer_step_outcome(step, next_step)
    future_progress_delta, _ = _lookahead_progress(step, future_steps)
    if outcome != "progress" and future_progress_delta <= 0:
        return None
    img = step.image_input
    board_height = _safe_int(img.height if img is not None else 0)
    board_width = _safe_int(img.width if img is not None else 0)
    board_active_cells = _count_active_cells(img)
    board_source_attr = str(img.source_attr or "") if img is not None else ""
    motif_tags = extract_board_motif_tags(
        step.image_input,
        legal_actions=step.available_actions,
        levels_completed=step.levels_completed,
        next_levels_completed=(
            next_step.levels_completed if next_step is not None else step.levels_completed
        ),
    )
    is_interaction = str(step.action_name or "") == "ACTION6"
    if is_interaction:
        if "interaction_step" not in motif_tags:
            motif_tags = list(motif_tags) + ["interaction_step"]
    lead_in_actions = [
        str(prior_step.action_name or "")
        for prior_step in steps[max(0, index - 3) : index]
        if str(prior_step.action_name or "")
    ]
    progress = 1 if outcome == "progress" else 0
    stalled = 1 if outcome == "stalled" else 0
    return {
        "subgoal": str(decision_subgoal or ""),
        "state": str(step.state or ""),
        "board_height": board_height,
        "board_width": board_width,
        "board_active_cells": board_active_cells,
        "board_source_attr": board_source_attr,
        "motif_tags": motif_tags,
        "action_name": str(step.action_name or ""),
        "lead_in_actions": lead_in_actions,
        "lesson_text": _abstract_lesson_text(
            action_name=str(step.action_name or ""),
            subgoal=str(decision_subgoal or ""),
            motif_tags=motif_tags,
            action_data=step.action_data,
        ),
        "salience": 1.0 + (0.5 if outcome == "progress" else 0.2 * min(3, future_progress_delta)),
        "confidence": 0.8 if outcome == "progress" else 0.62,
        "progress": progress,
        "failed": 0,
        "stalled": stalled,
        "total": 1,
        "last_outcome": outcome,
        "last_step_index": int(step.step_index),
        "source_games": [short_game_id],
    }


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
    future_progress_delta, _ = _lookahead_progress(step, future_steps)
    if step.action_name == "ACTION6" and _count_active_cells(step.image_input) > 0:
        if outcome == "progress" or future_progress_delta > 0:
            return "inspect-active-board"
    if next_step is not None and _is_win_state(next_step.state):
        return "hold-win"
    if outcome == "progress":
        return "advance-level"
    if stall_streak >= 2:
        return "break-stall"
    if future_progress_delta > 0:
        return "advance-level"
    return "break-stall"


def _abstract_lesson_text(
    *,
    action_name: str,
    subgoal: str,
    motif_tags: list[str],
    action_data: dict[str, Any] | None = None,
) -> str:
    motif_text = ", ".join(motif_tags[:4]) if motif_tags else "generic-board"
    if action_name == "ACTION6" and action_data:
        cx = action_data.get("x", "")
        cy = action_data.get("y", "")
        return f"{action_name} transfers for {subgoal or 'current-subgoal'} under {motif_text} at ({cx},{cy})"
    return f"{action_name} transfers for {subgoal or 'current-subgoal'} under {motif_text}"


# Thresholds for dynamics inference from action distributions
_DYNAMICS_THRESHOLDS = {
    "beam_switch_min": 0.15,
    "beam_click_min": 0.15,
    "signal_switch_min": 0.12,
    "signal_click_min": 0.08,
    "multi_switch_min": 0.10,
    "multi_click_max": 0.05,
    "push_click_min": 0.12,
    "push_switch_max": 0.05,
    "gravity_up_min": 0.18,
    "gravity_down_max": 0.10,
    "sliding_horiz_ratio": 3.0,
    "inventory_click_min": 0.05,
    "inventory_click_max": 0.12,
    "inventory_switch_max": 0.05,
    "nav_switch_max": 0.08,
    "nav_click_max": 0.08,
    "pattern_switch_min": 0.05,
    "pattern_click_min": 0.05,
    "cellular_move_max_ratio": 0.1,
}


def _infer_dynamics_from_actions(game_actions: dict[str, int], thresholds: dict[str, float] | None = None) -> str:
    """Infer the primary dynamics category from action distribution.

    Returns one of the 12 DynamicsCategory values based on action ratios.
    """
    t = thresholds or _DYNAMICS_THRESHOLDS
    total = sum(game_actions.values())
    if total <= 0:
        return "static"
    sr = game_actions.get("ACTION5", 0) / total
    cr = game_actions.get("ACTION6", 0) / total
    ur = game_actions.get("ACTION1", 0) / total
    dr = game_actions.get("ACTION2", 0) / total
    vert = game_actions.get("ACTION1", 0) + game_actions.get("ACTION2", 0)
    horiz = game_actions.get("ACTION3", 0) + game_actions.get("ACTION4", 0)
    tm = vert + horiz
    if sr > t["beam_switch_min"] and cr > t["beam_click_min"]:
        return "beam"
    if sr > t["signal_switch_min"] and cr > t["signal_click_min"]:
        return "signal"
    if sr > t["multi_switch_min"] and cr < t["multi_click_max"] and tm > 0:
        return "multi-agent"
    if cr > t["push_click_min"] and sr < t["push_switch_max"]:
        return "push-pull"
    if ur > t["gravity_up_min"] and dr < t["gravity_down_max"] and tm > 0:
        return "gravity"
    if horiz > vert * t["sliding_horiz_ratio"] and tm > 0:
        return "sliding"
    if cr > t["inventory_click_min"] and cr < t["inventory_click_max"] and sr < t["inventory_switch_max"] and tm > 0:
        return "inventory"
    if tm > 0 and sr < t["nav_switch_max"] and cr < t["nav_click_max"]:
        return "navigation"
    if sr > t["pattern_switch_min"] or cr > t["pattern_click_min"]:
        return "pattern-transform"
    if tm < total * t["cellular_move_max_ratio"]:
        return "cellular-automata"
    return "navigation"


def _infer_board_type(game_actions: dict[str, int]) -> str:
    """Legacy wrapper — returns 'puzzle', 'platformer', or 'unknown' for backward compat."""
    dynamics = _infer_dynamics_from_actions(game_actions)
    puzzle_like = {"beam", "signal", "push-pull", "multi-agent", "pattern-transform", "hybrid"}
    platformer_like = {"gravity", "sliding", "navigation", "inventory"}
    if dynamics in puzzle_like:
        return "puzzle"
    if dynamics in platformer_like:
        return "platformer"
    if dynamics == "static":
        return "unknown"
    return "unknown"


# ── Cognition context (post-hoc integration) ──────────────────────────


@dataclass
class _CognitionContext:
    """Per-game cognition module bundle used during post-hoc cognition pass."""
    working_memory: WorkingMemory
    meta_cognition: MetaCognition
    goal_stack: GoalStack
    chunk_library: ChunkLibrary
    consolidated_memory: ConsolidatedMemory
    attention: AttentionFocus | None
    game_id: str = ""


def _create_cognition_context() -> _CognitionContext:
    """Create a fresh set of cognition modules for a game."""
    return _CognitionContext(
        working_memory=WorkingMemory(),
        meta_cognition=MetaCognition(),
        goal_stack=GoalStack(),
        chunk_library=ChunkLibrary(),
        consolidated_memory=ConsolidatedMemory(),
        attention=AttentionFocus(),
    )


def _detect_bg_color(board: list[list[int]]) -> int:
    """Detect background color as the most common border cell value."""
    from collections import Counter
    h, w = len(board), len(board[0]) if board else 0
    border: list[int] = []
    if h > 0:
        border.extend(board[0])
        if h > 1:
            border.extend(board[h - 1])
    for y in range(1, h - 1):
        if w > 0:
            border.append(board[y][0])
            if w > 1:
                border.append(board[y][w - 1])
    return Counter(border).most_common(1)[0][0] if border else 9


def _infer_player_position(image_input: BoardImageInput) -> tuple[int, int] | None:
    """Infer player position from board: first non-bg non-zero cell near center."""
    board = image_input.board
    bg = _detect_bg_color(board)
    h, w = len(board), len(board[0]) if board else 0
    cx, cy = w // 2, h // 2
    # Spiral search outward from center
    for r in range(0, max(h, w)):
        for dy in range(-r, r + 1):
            y = cy + dy
            if y < 0 or y >= h:
                continue
            for dx in range(-r, r + 1):
                x = cx + dx
                if x < 0 or x >= w:
                    continue
                val = board[y][x]
                if val not in (bg, 0, 9):
                    return (x, y)
    return None


def _infer_goal_positions(image_input: BoardImageInput) -> list[tuple[int, int]]:
    """Find goal-colored cells (color 11) on the board."""
    board = image_input.board
    goals: list[tuple[int, int]] = []
    for y, row in enumerate(board):
        for x, val in enumerate(row):
            if val == 11:  # standard goal color
                goals.append((x, y))
    return goals


def _update_cognition(
    ctx: _CognitionContext,
    step: TrainingStep,
    next_step: TrainingStep | None,
    steps: list[TrainingStep],
    index: int,
    stall_streak: int,
) -> dict[str, Any]:
    """Run one step through the cognition pipeline. Returns cognition metrics."""
    action_name = str(step.action_name or "")

    # WorkingMemory
    ctx.working_memory.record_action(action_name)
    ctx.working_memory.tick()

    # MetaCognition
    recent = ctx.working_memory.recent_actions(5)
    outcomes = ["stalled" if a in ("RESET", "ACTION6") else "progress" for a in recent]
    wm_load = ctx.working_memory.load
    goal_depth = ctx.goal_stack.depth
    goal_progress = ctx.goal_stack.progress_ratio if ctx.goal_stack.root else 0.0
    confidence = ctx.meta_cognition.estimate_confidence(
        plan_steps=[], goal_stack_depth=goal_depth,
        working_memory_load=wm_load, stagnation_steps=stall_streak,
        recent_outcomes=outcomes,
        consecutive_failures=ctx.meta_cognition._consecutive_failures,
    )
    strategy = ctx.meta_cognition.select_strategy(
        confidence=confidence, stagnation_steps=stall_streak,
        has_plan=False, working_memory_load=wm_load,
        goal_progress=goal_progress,
    )

    # GoalStack
    outcome = _infer_step_outcome(step, next_step)
    future_steps = steps[index + 1:index + 4] if index + 1 < len(steps) else []
    subgoal = _infer_training_subgoal(
        step=step, next_step=next_step, future_steps=future_steps,
        outcome=outcome, stall_streak=stall_streak,
    )
    if not ctx.goal_stack.find(subgoal):
        ctx.goal_stack.push(subgoal)
    ctx.goal_stack.mark_active()

    # AttentionFocus
    mean_sal = 0.0
    max_sal = 0.0
    salient_count = 0
    if step.image_input is not None and ctx.attention is not None:
        player_pos = _infer_player_position(step.image_input)
        goal_pos = _infer_goal_positions(step.image_input)
        sal = ctx.attention.compute(step.image_input.board, player_pos, goal_pos)
        cells = [sal.data[y][x] for y in range(min(64, len(sal.data))) for x in range(min(64, len(sal.data[0])))]
        if cells:
            mean_sal = sum(cells) / len(cells)
            max_sal = max(cells)
            salient_count = sum(1 for v in cells if v >= sal.threshold)

    return {
        "cognition_strategy": strategy.value,
        "cognition_confidence": round(confidence, 3),
        "wm_load": round(wm_load, 3),
        "goal_depth": goal_depth,
        "goal_progress": round(goal_progress, 3),
        "mean_salience": round(mean_sal, 4),
        "max_salience": round(max_sal, 4),
        "salient_cell_count": salient_count,
    }


def _process_recording_cognition(
    zip_path: Path,
    recording: RecordingMeta,
    ctx: _CognitionContext,
) -> list[dict[str, Any]]:
    """Replay one recording through cognition modules. Returns trace dicts."""
    traces: list[dict[str, Any]] = []
    steps = list(iter_steps(zip_path, recording.recording_name))
    if not steps:
        return traces
    for index, step in enumerate(steps):
        next_step = steps[index + 1] if index + 1 < len(steps) else None
        stall_streak = _preceding_stall_streak(steps, index)
        action_name = str(step.action_name or "")
        if not action_name:
            continue
        metrics = _update_cognition(ctx, step, next_step, steps, index, stall_streak)
        metrics["game_id"] = ctx.game_id
        metrics["recording_guid"] = recording.recording_guid
        metrics["step_index"] = step.step_index
        metrics["action_name"] = action_name
        traces.append(metrics)
        # On win: extract chunk + record episode
        if _is_win_state(str(next_step.state if next_step else step.state)):
            win_actions = [s.action_name for s in steps if s.action_name and s.action_name != "RESET"]
            if win_actions:
                dyn = _infer_dynamics_from_actions({a: 1 for a in win_actions})
                ctx.chunk_library.extract(win_actions[-20:], dyn, 0, "win")
            board_hash = hash(str(getattr(step, 'image_input', None))[-256:]) if step.image_input else 0
            ctx.consolidated_memory.record(
                dynamics=dyn if 'dyn' in dir() else "unknown",
                action_sequence=win_actions[-20:] if win_actions else [],
                outcome="win", goal_count=0,
                player_start=(0, 0), board_hash=board_hash,
            )
    return traces


def _post_hoc_cognition_pass(
    zip_path: Path,
    selected_recordings: list[RecordingMeta],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Post-hoc cognition pass: replay all recordings through cognition modules.
    
    Returns (all_traces, all_chunks_by_name).
    Cognition is per-game — modules are reset between games but persist across
    recordings of the same game (so ChunkLibrary and ConsolidatedMemory accumulate).
    """
    all_traces: list[dict[str, Any]] = []
    all_chunks: dict[str, dict[str, Any]] = {}
    per_game: dict[str, _CognitionContext] = {}

    for recording in selected_recordings:
        sgid = recording.game_id.split("-")[0]
        if sgid not in per_game:
            per_game[sgid] = _create_cognition_context()
            per_game[sgid].game_id = sgid
        ctx = per_game[sgid]
        recording_traces = _process_recording_cognition(zip_path, recording, ctx)
        all_traces.extend(recording_traces)

    # End of each game: consolidate + collect chunks
    for sgid, ctx in per_game.items():
        ctx.consolidated_memory.consolidate()
        for name, chunk in ctx.chunk_library._chunks.items():
            if name not in all_chunks:
                all_chunks[name] = {
                    "dynamics": chunk.preconditions.get("dynamics", ""),
                    "goal_count": chunk.preconditions.get("goal_count", 0),
                    "action_prefix": chunk.preconditions.get("action_prefix", []),
                    "actions": chunk.actions,
                    "frequency": chunk.frequency,
                    "reliability": chunk.reliability,
                    "last_used": chunk.last_used,
                    "source_game_ids": [],
                }
            if sgid not in all_chunks[name]["source_game_ids"]:
                all_chunks[name]["source_game_ids"].append(sgid)

    return all_traces, all_chunks


# ── Board policy extraction ───────────────────────────────────────────


def _extract_board_policy_params(
    artifact_games: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    # Bucket by dynamics category (12 categories) and by legacy board_type.
    dyn_actions: dict[str, dict[str, int]] = {}
    dyn_games: dict[str, set[str]] = {}
    bt_actions: dict[str, dict[str, int]] = {}
    bt_games: dict[str, set[str]] = {}
    for game_id, game_entry in artifact_games.items():
        actions = game_entry.get("actions", {})
        dynamics = _infer_dynamics_from_actions(actions)
        bt = _infer_board_type(actions)
        dyn_games.setdefault(dynamics, set()).add(game_id)
        bt_games.setdefault(bt, set()).add(game_id)
        target_d = dyn_actions.setdefault(dynamics, {})
        target_b = bt_actions.setdefault(bt, {})
        for action_name, count in actions.items():
            c = int(count or 0)
            target_d[action_name] = target_d.get(action_name, 0) + c
            target_b[action_name] = target_b.get(action_name, 0) + c
    rows: list[dict[str, Any]] = []

    _PARAM_DEFS = [
        ("explore-bias", 0.7, lambda a, t, v, h, t3, s: "vertical" if v > h else "horizontal"),
        ("interact-range", 0.65, lambda a, t, v, h, t3, s: "4" if a.get("ACTION6", 0) / t > 0.08 else "2"),
        ("switch-urgency", 0.7, lambda a, t, v, h, t3, s: "high" if "ACTION5" in t3 else "low"),
        ("path-preference", 0.6, lambda a, t, v, h, t3, s: "direct" if h > v * 2 else "safe"),
        ("entity-count", 0.7, lambda a, t, v, h, t3, s: "multiple" if a.get("ACTION5", 0) > 0 else "single"),
        ("goal-proximity", 0.6, lambda a, t, v, h, t3, s: "1" if "ACTION6" in t3 else "3"),
        ("retry-tolerance", 0.6, lambda a, t, v, h, t3, s: "2" if a.get("ACTION5", 0) / t > 0.1 else "5"),
        ("stagnation-threshold", 0.6, lambda a, t, v, h, t3, s: "2" if a.get("ACTION5", 0) / t > 0.1 else "5"),
    ]

    def _extract_rows(
        key: str,
        action_counts: dict[str, int],
        game_count: int,
    ) -> list[dict[str, Any]]:
        if not action_counts:
            return []
        total = sum(action_counts.values())
        if total <= 0:
            return []
        vert = action_counts.get("ACTION1", 0) + action_counts.get("ACTION2", 0)
        horiz = action_counts.get("ACTION3", 0) + action_counts.get("ACTION4", 0)
        sorted_actions = sorted(action_counts.items(), key=lambda x: -x[1])
        top3 = {a for a, _ in sorted_actions[:3]}
        sample_count = game_count
        result: list[dict[str, Any]] = []

        for name, confidence, resolver in _PARAM_DEFS:
            result.append({
                "board_type": key,
                "param_name": name,
                "param_value": resolver(action_counts, total, vert, horiz, top3, sample_count),
                "confidence": confidence,
                "sample_count": sample_count,
            })

        # move-bias
        move_actions = {
            "right": action_counts.get("ACTION4", 0),
            "down": action_counts.get("ACTION2", 0),
            "left": action_counts.get("ACTION3", 0),
            "up": action_counts.get("ACTION1", 0),
        }
        if move_actions:
            dominant_dir = max(move_actions, key=move_actions.get)
            if move_actions[dominant_dir] > 0:
                result.append({"board_type": key, "param_name": "move-bias", "param_value": dominant_dir, "confidence": 0.65, "sample_count": sample_count})

        # beam-priority
        a6p = action_counts.get("ACTION6", 0)
        s6c = action_counts.get("ACTION5", 0) + a6p
        if s6c > total * 0.2:
            result.append({"board_type": key, "param_name": "beam-priority", "param_value": "high", "confidence": 0.6, "sample_count": sample_count})
        elif s6c > total * 0.1:
            result.append({"board_type": key, "param_name": "beam-priority", "param_value": "medium", "confidence": 0.6, "sample_count": sample_count})
        else:
            result.append({"board_type": key, "param_name": "beam-priority", "param_value": "low", "confidence": 0.6, "sample_count": sample_count})

        # click-action-priority
        a6pct = a6p / total
        if a6pct > 0.15:
            result.append({"board_type": key, "param_name": "click-action-priority", "param_value": "high", "confidence": 0.65, "sample_count": sample_count})
        elif a6pct > 0.05:
            result.append({"board_type": key, "param_name": "click-action-priority", "param_value": "medium", "confidence": 0.65, "sample_count": sample_count})
        else:
            result.append({"board_type": key, "param_name": "click-action-priority", "param_value": "low", "confidence": 0.65, "sample_count": sample_count})

        # beam-reflection-type
        beam_actions = action_counts.get("ACTION5", 0) + a6p
        if beam_actions > 0 and horiz > vert:
            result.append({"board_type": key, "param_name": "beam-reflection-type", "param_value": "horizontal", "confidence": 0.6, "sample_count": sample_count})
        elif beam_actions > 0 and vert > horiz:
            result.append({"board_type": key, "param_name": "beam-reflection-type", "param_value": "vertical", "confidence": 0.6, "sample_count": sample_count})
        else:
            result.append({"board_type": key, "param_name": "beam-reflection-type", "param_value": "dual", "confidence": 0.6, "sample_count": sample_count})

        # path-restricted
        a3pct = action_counts.get("ACTION3", 0) / total
        if a3pct > 0.15 and a6pct > 0.08:
            result.append({"board_type": key, "param_name": "path-restricted", "param_value": "true", "confidence": 0.55, "sample_count": sample_count})
        else:
            result.append({"board_type": key, "param_name": "path-restricted", "param_value": "false", "confidence": 0.55, "sample_count": sample_count})

        return result

    # Extract for each dynamics category
    for dyn_key, dyn_counts in dyn_actions.items():
        rows.extend(_extract_rows(dyn_key, dyn_counts, len(dyn_games.get(dyn_key, set()))))
    # Also extract for legacy board types
    for bt_key, bt_counts in bt_actions.items():
        rows.extend(_extract_rows(bt_key, bt_counts, len(bt_games.get(bt_key, set()))))

    return rows


def _count_rows_for_scope(
    game_id: str,
    scope: str,
    entries: dict[str, Any],
    key_field: str | None = None,
) -> list[dict[str, Any]]:
    """Build training-count rows for a single scope level."""
    rows: list[dict[str, Any]] = []
    for entry_key, entry in sorted(entries.items()):
        for action_name, count in sorted(
            _consensus_filter_actions(
                _apply_recording_support_bonus(
                    entry.get("actions", {}),
                    entry.get("action_supports", {}),
                )
            ).items()
        ):
            row: dict[str, Any] = {
                "scope": scope,
                "game_id": game_id,
                "action_name": action_name,
                "count": count,
            }
            if key_field:
                row[key_field] = entry_key
            rows.append(row)
    return rows


def _count_rows_for_game(
    game_id: str,
    game_entry: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(_count_rows_for_scope(game_id, "game", {"_": game_entry}))
    rows.extend(_count_rows_for_scope(game_id, "state", game_entry.get("states", {}), key_field="state"))
    rows.extend(_count_rows_for_scope(game_id, "subgoal", game_entry.get("subgoals", {}), key_field="subgoal"))
    rows.extend(_count_rows_for_scope(game_id, "frame", game_entry.get("frame_signatures", {}), key_field="frame_signature"))
    return rows


def _apply_recording_support_bonus(
    actions: dict[str, int],
    supports: dict[str, int],
) -> dict[str, int]:
    adjusted: dict[str, int] = {}
    for action_name, count in actions.items():
        base = _safe_int(count)
        if base <= 0:
            continue
        support_count = _safe_int(supports.get(action_name))
        adjusted[str(action_name)] = base + min(2, max(0, support_count - 1))
    return adjusted


def _consensus_filter_actions(actions: dict[str, int]) -> dict[str, int]:
    normalized = {
        str(action_name): _safe_int(count)
        for action_name, count in actions.items()
        if _safe_int(count) > 0
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
        margin = 0
    return {str(leader_name): max(1, margin)}


def _is_recording_entry(name: str) -> bool:
    if "__MACOSX" in name:
        return False
    return name.endswith(".recording.jsonl")
