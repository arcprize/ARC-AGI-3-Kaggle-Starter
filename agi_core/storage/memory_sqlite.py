from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from agi_core.image_input import board_context_match_score, extract_board_motif_tags
from agi_core.storage.memory_cache import get_cache
from agi_core.storage.priors_sqlite import (
    game_id_aliases,
    initialize_schema,
    load_best_prior_bucket,
)
from agi_core.types import EpisodeRecord

DEFAULT_RECENT_RECORD_LIMIT = 24
DEFAULT_BOARD_CONTEXT_HISTORY_LIMIT = 16
DEFAULT_EPISODE_SUMMARY_LIMIT = 8

# -- In-memory cache helper ---------------------------------------------

_MEM_CACHE: dict[str, object] = {}


def _refresh_cache(path: str | Path) -> None:
    """Refresh the in-memory cache after a disk write."""
    cache = get_cache(str(Path(path).resolve()))
    if cache is not None:
        cache.refresh()


def _connect(path: str | Path, *, write: bool = False) -> sqlite3.Connection:
    """Return an SQLite connection.

    *write=True*  → direct disk connection (cache bypassed, no auto-refresh).
    *write=False* → in-memory cache connection (sub-ms reads).

    After any write operation callers should refresh the cache via
    ``get_cache(path).refresh()`` so subsequent reads see the new data.
    """
    if write:
        return sqlite3.connect(str(path))
    resolved = str(Path(path).resolve())
    cache = _MEM_CACHE.get(resolved) or get_cache(resolved)
    if cache is not None:
        _MEM_CACHE[resolved] = cache
        return cache.connect(write=False)
    return sqlite3.connect(str(path))


def initialize_memory_schema(db_path: str | Path) -> None:
    path = Path(db_path)
    initialize_schema(path)
    # Refresh in-memory cache so subsequent reads see the created tables.
    cache = get_cache(str(path))
    if cache is not None:
        cache.refresh()
    with closing(_connect(path, write=True)) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS live_episode_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                action_name TEXT NOT NULL,
                outcome TEXT NOT NULL,
                progress_delta INTEGER NOT NULL,
                prior_state TEXT NOT NULL,
                next_state TEXT NOT NULL,
                prior_frame_signature TEXT NOT NULL,
                next_frame_signature TEXT NOT NULL,
                subgoal TEXT NOT NULL,
                decision_source TEXT NOT NULL,
                stagnation_before INTEGER NOT NULL,
                stagnation_after INTEGER NOT NULL,
                legal_actions_json TEXT NOT NULL,
                prior_board_height INTEGER NOT NULL DEFAULT 0,
                prior_board_width INTEGER NOT NULL DEFAULT 0,
                prior_board_active_cells INTEGER NOT NULL DEFAULT 0,
                prior_board_source_attr TEXT NOT NULL DEFAULT '',
                prior_board_default_probe_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        _ensure_live_episode_record_columns(connection)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS live_action_stats (
                game_id TEXT NOT NULL,
                action_name TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                failed INTEGER NOT NULL DEFAULT 0,
                stalled INTEGER NOT NULL DEFAULT 0,
                total INTEGER NOT NULL DEFAULT 0,
                last_outcome TEXT,
                PRIMARY KEY (game_id, action_name)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS live_state_action_stats (
                game_id TEXT NOT NULL,
                state TEXT NOT NULL,
                action_name TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                failed INTEGER NOT NULL DEFAULT 0,
                stalled INTEGER NOT NULL DEFAULT 0,
                total INTEGER NOT NULL DEFAULT 0,
                last_outcome TEXT,
                PRIMARY KEY (game_id, state, action_name)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS live_frame_action_stats (
                game_id TEXT NOT NULL,
                frame_signature TEXT NOT NULL,
                action_name TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                failed INTEGER NOT NULL DEFAULT 0,
                stalled INTEGER NOT NULL DEFAULT 0,
                total INTEGER NOT NULL DEFAULT 0,
                last_outcome TEXT,
                PRIMARY KEY (game_id, frame_signature, action_name)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS live_subgoal_action_stats (
                game_id TEXT NOT NULL,
                subgoal TEXT NOT NULL,
                action_name TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                failed INTEGER NOT NULL DEFAULT 0,
                stalled INTEGER NOT NULL DEFAULT 0,
                total INTEGER NOT NULL DEFAULT 0,
                last_outcome TEXT,
                PRIMARY KEY (game_id, subgoal, action_name)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS live_basin_stats (
                game_id TEXT NOT NULL,
                state TEXT NOT NULL,
                frame_signature TEXT NOT NULL,
                visits INTEGER NOT NULL DEFAULT 0,
                actions_attempted INTEGER NOT NULL DEFAULT 0,
                distinct_actions_tried INTEGER NOT NULL DEFAULT 0,
                progress_events INTEGER NOT NULL DEFAULT 0,
                failed_events INTEGER NOT NULL DEFAULT 0,
                stalled_events INTEGER NOT NULL DEFAULT 0,
                last_action_name TEXT,
                last_progress_step INTEGER,
                is_exhausted INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (game_id, state, frame_signature)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS live_episode_summaries (
                game_id TEXT NOT NULL,
                prior_state TEXT NOT NULL,
                prior_frame_signature TEXT NOT NULL,
                subgoal TEXT NOT NULL,
                board_height INTEGER NOT NULL DEFAULT 0,
                board_width INTEGER NOT NULL DEFAULT 0,
                board_active_cells INTEGER NOT NULL DEFAULT 0,
                board_source_attr TEXT NOT NULL DEFAULT '',
                board_default_probe_json TEXT NOT NULL DEFAULT '{}',
                action_name TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                failed INTEGER NOT NULL DEFAULT 0,
                stalled INTEGER NOT NULL DEFAULT 0,
                total INTEGER NOT NULL DEFAULT 0,
                last_outcome TEXT,
                first_step_index INTEGER NOT NULL DEFAULT 0,
                last_step_index INTEGER NOT NULL DEFAULT 0,
                recent_action_sequence_json TEXT NOT NULL DEFAULT '[]',
                recent_outcome_sequence_json TEXT NOT NULL DEFAULT '[]',
                decision_sources_json TEXT NOT NULL DEFAULT '{}',
                max_stagnation_after INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (
                    game_id,
                    prior_state,
                    prior_frame_signature,
                    subgoal,
                    board_height,
                    board_width,
                    board_active_cells,
                    board_source_attr,
                    board_default_probe_json,
                    action_name
                )
            )
            """
        )
        _ensure_live_episode_summary_columns(connection)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS live_pattern_stats (
                game_id TEXT NOT NULL,
                state TEXT NOT NULL,
                subgoal TEXT NOT NULL,
                board_height INTEGER NOT NULL DEFAULT 0,
                board_width INTEGER NOT NULL DEFAULT 0,
                board_active_cells INTEGER NOT NULL DEFAULT 0,
                board_source_attr TEXT NOT NULL DEFAULT '',
                action_name TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                failed INTEGER NOT NULL DEFAULT 0,
                stalled INTEGER NOT NULL DEFAULT 0,
                total INTEGER NOT NULL DEFAULT 0,
                last_outcome TEXT,
                PRIMARY KEY (
                    game_id,
                    state,
                    subgoal,
                    board_height,
                    board_width,
                    board_active_cells,
                    board_source_attr,
                    action_name
                )
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS live_sequence_stats (
                game_id TEXT NOT NULL,
                state TEXT NOT NULL,
                subgoal TEXT NOT NULL,
                previous_action_name TEXT NOT NULL,
                action_name TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                failed INTEGER NOT NULL DEFAULT 0,
                stalled INTEGER NOT NULL DEFAULT 0,
                total INTEGER NOT NULL DEFAULT 0,
                last_outcome TEXT,
                PRIMARY KEY (
                    game_id,
                    state,
                    subgoal,
                    previous_action_name,
                    action_name
                )
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS live_board_cluster_stats (
                game_id TEXT NOT NULL,
                board_height INTEGER NOT NULL DEFAULT 0,
                board_width INTEGER NOT NULL DEFAULT 0,
                board_active_cells INTEGER NOT NULL DEFAULT 0,
                board_source_attr TEXT NOT NULL DEFAULT '',
                action_name TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                failed INTEGER NOT NULL DEFAULT 0,
                stalled INTEGER NOT NULL DEFAULT 0,
                total INTEGER NOT NULL DEFAULT 0,
                last_outcome TEXT,
                PRIMARY KEY (
                    game_id,
                    board_height,
                    board_width,
                    board_active_cells,
                    board_source_attr,
                    action_name
                )
                )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS live_semantic_memory (
                game_id TEXT NOT NULL,
                state TEXT NOT NULL,
                subgoal TEXT NOT NULL,
                board_height INTEGER NOT NULL DEFAULT 0,
                board_width INTEGER NOT NULL DEFAULT 0,
                board_active_cells INTEGER NOT NULL DEFAULT 0,
                board_source_attr TEXT NOT NULL DEFAULT '',
                action_name TEXT NOT NULL,
                lesson_text TEXT NOT NULL DEFAULT '',
                salience REAL NOT NULL DEFAULT 0.0,
                confidence REAL NOT NULL DEFAULT 0.0,
                progress INTEGER NOT NULL DEFAULT 0,
                failed INTEGER NOT NULL DEFAULT 0,
                stalled INTEGER NOT NULL DEFAULT 0,
                total INTEGER NOT NULL DEFAULT 0,
                last_outcome TEXT,
                last_step_index INTEGER NOT NULL DEFAULT 0,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_access_step INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (
                    game_id,
                    state,
                    subgoal,
                    board_height,
                    board_width,
                    board_active_cells,
                    board_source_attr,
                    action_name
                )
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS live_procedural_memory (
                game_id TEXT NOT NULL,
                state TEXT NOT NULL,
                subgoal TEXT NOT NULL,
                previous_action_name TEXT NOT NULL,
                action_name TEXT NOT NULL,
                strategy_tag TEXT NOT NULL DEFAULT '',
                salience REAL NOT NULL DEFAULT 0.0,
                confidence REAL NOT NULL DEFAULT 0.0,
                progress INTEGER NOT NULL DEFAULT 0,
                failed INTEGER NOT NULL DEFAULT 0,
                stalled INTEGER NOT NULL DEFAULT 0,
                total INTEGER NOT NULL DEFAULT 0,
                last_outcome TEXT,
                last_step_index INTEGER NOT NULL DEFAULT 0,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_access_step INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (
                    game_id,
                    state,
                    subgoal,
                    previous_action_name,
                    action_name
                )
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS abstract_memories (
                subgoal TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT '',
                board_height INTEGER NOT NULL DEFAULT 0,
                board_width INTEGER NOT NULL DEFAULT 0,
                board_active_cells INTEGER NOT NULL DEFAULT 0,
                board_source_attr TEXT NOT NULL DEFAULT '',
                motif_tags_json TEXT NOT NULL DEFAULT '[]',
                action_name TEXT NOT NULL,
                lead_in_actions_json TEXT NOT NULL DEFAULT '[]',
                lesson_text TEXT NOT NULL DEFAULT '',
                salience REAL NOT NULL DEFAULT 0.0,
                confidence REAL NOT NULL DEFAULT 0.0,
                progress INTEGER NOT NULL DEFAULT 0,
                failed INTEGER NOT NULL DEFAULT 0,
                stalled INTEGER NOT NULL DEFAULT 0,
                total INTEGER NOT NULL DEFAULT 0,
                last_outcome TEXT,
                last_step_index INTEGER NOT NULL DEFAULT 0,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_access_step INTEGER NOT NULL DEFAULT 0,
                source_game_count INTEGER NOT NULL DEFAULT 0,
                source_games_json TEXT NOT NULL DEFAULT '[]',
                PRIMARY KEY (
                    subgoal,
                    state,
                    board_height,
                    board_width,
                    board_active_cells,
                    board_source_attr,
                    motif_tags_json,
                    action_name
                )
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_live_records_game_step
            ON live_episode_records(game_id, step_index)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_live_state_stats
            ON live_state_action_stats(game_id, state)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_live_frame_stats
            ON live_frame_action_stats(game_id, frame_signature)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_live_subgoal_stats
            ON live_subgoal_action_stats(game_id, subgoal)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_live_episode_summaries_game_context
            ON live_episode_summaries(game_id, prior_state, subgoal, last_step_index)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_live_pattern_stats_context
            ON live_pattern_stats(game_id, state, subgoal)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_live_sequence_stats_context
            ON live_sequence_stats(game_id, state, subgoal, previous_action_name)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_live_board_cluster_stats_context
            ON live_board_cluster_stats(game_id, board_height, board_width, board_active_cells)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_live_semantic_memory_context
            ON live_semantic_memory(game_id, state, subgoal, board_height, board_width, board_active_cells)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_live_procedural_memory_context
            ON live_procedural_memory(game_id, state, subgoal, previous_action_name)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_abstract_memories_context
            ON abstract_memories(subgoal, state, board_height, board_width, board_active_cells)
            """
        )
        # Cognition traces table (post-hoc cognition pass during training)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS cognition_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                recording_guid TEXT NOT NULL DEFAULT '',
                step_index INTEGER NOT NULL,
                action_name TEXT NOT NULL DEFAULT '',
                cognition_strategy TEXT NOT NULL DEFAULT '',
                cognition_confidence REAL NOT NULL DEFAULT 0.0,
                wm_load REAL NOT NULL DEFAULT 0.0,
                goal_depth INTEGER NOT NULL DEFAULT 0,
                goal_progress REAL NOT NULL DEFAULT 0.0,
                mean_salience REAL NOT NULL DEFAULT 0.0,
                max_salience REAL NOT NULL DEFAULT 0.0,
                salient_cell_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cognition_traces_lookup
            ON cognition_traces(game_id, recording_guid, step_index)
            """
        )
        # Chunks table (persisted ChunkLibrary chunks from cognition pass)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                dynamics TEXT NOT NULL DEFAULT '',
                goal_count INTEGER NOT NULL DEFAULT 0,
                action_prefix_json TEXT NOT NULL DEFAULT '[]',
                actions_json TEXT NOT NULL DEFAULT '[]',
                frequency INTEGER NOT NULL DEFAULT 1,
                reliability REAL NOT NULL DEFAULT 1.0,
                last_used REAL NOT NULL DEFAULT 0.0,
                source_game_ids_json TEXT NOT NULL DEFAULT '[]'
            )
            """
        )
        connection.commit()


def import_legacy_learning_json(db_path: str | Path, json_path: str | Path) -> None:
    db = Path(db_path)
    path = Path(json_path)
    if not path.exists():
        return
    initialize_memory_schema(db)
    with closing(_connect(db, write=True)) as connection:
        if _metadata_value(connection, "legacy_import_completed") == "1":
            return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    games = payload.get("games", {}) if isinstance(payload, dict) else {}
    if not isinstance(games, dict):
        return
    with closing(_connect(db, write=True)) as connection:
        for game_id, game_entry in sorted(games.items()):
            if not isinstance(game_entry, dict):
                continue
            _import_action_scope(
                connection,
                table="live_action_stats",
                scope_fields=(str(game_id),),
                actions=game_entry.get("actions", {}),
            )
            for state_name, state_entry in sorted((game_entry.get("states", {}) or {}).items()):
                if not isinstance(state_entry, dict):
                    continue
                _import_action_scope(
                    connection,
                    table="live_state_action_stats",
                    scope_fields=(str(game_id), str(state_name)),
                    actions=state_entry.get("actions", {}),
                )
                _seed_basin_from_aggregate_actions(
                    connection,
                    game_id=str(game_id),
                    state=str(state_name),
                    frame_signature="",
                    actions=state_entry.get("actions", {}),
                )
            for frame_signature, frame_entry in sorted(
                (game_entry.get("frame_signatures", {}) or {}).items()
            ):
                if not isinstance(frame_entry, dict):
                    continue
                _import_action_scope(
                    connection,
                    table="live_frame_action_stats",
                    scope_fields=(str(game_id), str(frame_signature)),
                    actions=frame_entry.get("actions", {}),
                )
            for subgoal_name, subgoal_entry in sorted(
                (game_entry.get("subgoals", {}) or {}).items()
            ):
                if not isinstance(subgoal_entry, dict):
                    continue
                _import_action_scope(
                    connection,
                    table="live_subgoal_action_stats",
                    scope_fields=(str(game_id), str(subgoal_name)),
                    actions=subgoal_entry.get("actions", {}),
                )
        connection.execute(
            """
            INSERT INTO metadata(key, value)
            VALUES('legacy_import_completed', '1')
            ON CONFLICT(key) DO UPDATE SET value='1'
            """
        )
        connection.commit()


def record_transition(db_path: str | Path, record: EpisodeRecord) -> None:
    db = Path(db_path)
    initialize_memory_schema(db)
    with closing(_connect(db, write=True)) as connection:
        _record_transition_with_connection(connection, record)
        connection.commit()
    _refresh_cache(db)


def record_transitions(db_path: str | Path, records: list[EpisodeRecord]) -> None:
    if not records:
        return
    db = Path(db_path)
    initialize_memory_schema(db)
    with closing(_connect(db, write=True)) as connection:
        for record in records:
            _record_transition_with_connection(connection, record)
        connection.commit()
    _refresh_cache(db)


def upsert_abstract_memory_rows(
    db_path: str | Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    db = Path(db_path)
    initialize_memory_schema(db)
    with closing(_connect(db, write=True)) as connection:
        for row in rows:
            _upsert_abstract_memory_row(connection, row)
        connection.commit()
    _refresh_cache(db)


def upsert_cognition_traces(db_path: str | Path, traces: list[dict[str, Any]]) -> None:
    """Batch insert cognition traces into ``cognition_traces`` table."""
    if not traces:
        return
    db = Path(db_path)
    initialize_memory_schema(db)
    with closing(_connect(db, write=True)) as connection:
        for trace in traces:
            connection.execute(
                """
                INSERT INTO cognition_traces(
                    game_id, recording_guid, step_index, action_name,
                    cognition_strategy, cognition_confidence,
                    wm_load, goal_depth, goal_progress,
                    mean_salience, max_salience, salient_cell_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    str(trace.get("game_id", "")),
                    str(trace.get("recording_guid", "")),
                    int(trace.get("step_index", 0)),
                    str(trace.get("action_name", "")),
                    str(trace.get("cognition_strategy", "")),
                    float(trace.get("cognition_confidence", 0.0)),
                    float(trace.get("wm_load", 0.0)),
                    int(trace.get("goal_depth", 0)),
                    float(trace.get("goal_progress", 0.0)),
                    float(trace.get("mean_salience", 0.0)),
                    float(trace.get("max_salience", 0.0)),
                    int(trace.get("salient_cell_count", 0)),
                ],
            )
        connection.commit()
    _refresh_cache(db)


def upsert_chunk(db_path: str | Path, chunk_name: str, chunk_data: dict[str, Any]) -> None:
    """Insert or update a chunk row in the ``chunks`` table."""
    db = Path(db_path)
    initialize_memory_schema(db)
    with closing(_connect(db, write=True)) as connection:
        connection.execute(
            """
            INSERT INTO chunks (name, dynamics, goal_count, action_prefix_json, actions_json,
                                frequency, reliability, last_used, source_game_ids_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                frequency = excluded.frequency,
                actions_json = excluded.actions_json,
                reliability = CASE
                    WHEN chunks.actions_json = excluded.actions_json
                    THEN MIN(1.0, chunks.reliability + 0.05)
                    ELSE MAX(0.0, chunks.reliability - 0.1)
                END,
                last_used = MAX(chunks.last_used, excluded.last_used)
            """,
            [
                chunk_name,
                str(chunk_data.get("dynamics", "")),
                int(chunk_data.get("goal_count", 0)),
                json.dumps(chunk_data.get("action_prefix", [])),
                json.dumps(chunk_data.get("actions", [])),
                int(chunk_data.get("frequency", 1)),
                float(chunk_data.get("reliability", 1.0)),
                float(chunk_data.get("last_used", 0.0)),
                json.dumps(chunk_data.get("source_game_ids", [])),
            ],
        )
        connection.commit()
    _refresh_cache(db)


def _record_transition_with_connection(
    connection: sqlite3.Connection,
    record: EpisodeRecord,
) -> None:
    connection.execute(
        """
        INSERT INTO live_episode_records(
            game_id,
            step_index,
            action_name,
            outcome,
            progress_delta,
            prior_state,
            next_state,
            prior_frame_signature,
            next_frame_signature,
            subgoal,
            decision_source,
            stagnation_before,
            stagnation_after,
            legal_actions_json,
            prior_board_height,
            prior_board_width,
            prior_board_active_cells,
            prior_board_source_attr,
            prior_board_default_probe_json
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            record.game_id,
            int(record.step_index),
            record.action_name,
            record.outcome,
            int(record.progress_delta),
            record.prior_game_state,
            record.next_game_state,
            str(record.prior_frame_signature or ""),
            str(record.next_frame_signature or ""),
            str(record.decision_subgoal or ""),
            str(record.decision_source or ""),
            int(record.stagnation_before),
            int(record.stagnation_after),
            json.dumps(list(record.prior_legal_actions), sort_keys=True),
            int(record.prior_board_height),
            int(record.prior_board_width),
            int(record.prior_board_active_cells),
            str(record.prior_board_source_attr or ""),
            json.dumps(dict(record.prior_board_default_probe or {}), sort_keys=True),
        ],
    )
    _upsert_outcome_row(
        connection,
        table="live_action_stats",
        scope_fields=[record.game_id],
        key_value=record.action_name,
        outcome=record.outcome,
    )
    if record.prior_game_state:
        _upsert_outcome_row(
            connection,
            table="live_state_action_stats",
            scope_fields=[record.game_id, record.prior_game_state],
            key_value=record.action_name,
            outcome=record.outcome,
        )
    if record.prior_frame_signature:
        _upsert_outcome_row(
            connection,
            table="live_frame_action_stats",
            scope_fields=[record.game_id, record.prior_frame_signature],
            key_value=record.action_name,
            outcome=record.outcome,
        )
    if record.decision_subgoal:
        _upsert_outcome_row(
            connection,
            table="live_subgoal_action_stats",
            scope_fields=[record.game_id, record.decision_subgoal],
            key_value=record.action_name,
            outcome=record.outcome,
        )
    _refresh_basin_row(
        connection,
        record.game_id,
        record.prior_game_state,
        str(record.prior_frame_signature or ""),
        record.action_name,
    )
    _upsert_pattern_stats_row(connection, record)
    _upsert_sequence_stats_row(connection, record)
    _upsert_board_cluster_stats_row(connection, record)
    _upsert_episode_summary_row(connection, record)
    _upsert_semantic_memory_row(connection, record)
    _upsert_procedural_memory_row(connection, record)


def load_basin_status(
    db_path: str | Path,
    game_id: str,
    state: str,
    frame_signature: str,
) -> dict[str, Any]:
    path = Path(db_path)
    if not path.exists() or not state:
        return {
            "state": state,
            "frame_signature": frame_signature,
            "visits": 0,
            "actions_attempted": 0,
            "distinct_actions_tried": 0,
            "progress_events": 0,
            "failed_events": 0,
            "stalled_events": 0,
            "last_action_name": None,
            "last_progress_step": None,
            "is_exhausted": False,
            "action_outcomes": {},
        }
    with closing(_connect(path)) as connection:
        lookup_game_id = _resolve_memory_lookup_game_id(connection, game_id)
        basin_frame_signature = frame_signature
        row = _load_basin_row(connection, lookup_game_id, state, basin_frame_signature)
        action_outcomes = _load_basin_action_outcomes(
            connection,
            game_id=lookup_game_id,
            state=state,
            frame_signature=basin_frame_signature,
        )
        if row is None and frame_signature:
            basin_frame_signature = ""
            row = _load_basin_row(connection, lookup_game_id, state, basin_frame_signature)
            action_outcomes = _load_basin_action_outcomes(
                connection,
                game_id=lookup_game_id,
                state=state,
                frame_signature=basin_frame_signature,
            )
        if row is None:
            return {
                "state": state,
                "frame_signature": basin_frame_signature,
                "visits": 0,
                "actions_attempted": 0,
                "distinct_actions_tried": 0,
                "progress_events": 0,
                "failed_events": 0,
                "stalled_events": 0,
                "last_action_name": None,
                "last_progress_step": None,
                "is_exhausted": False,
                "action_outcomes": action_outcomes,
            }
        return {
            "state": state,
            "frame_signature": basin_frame_signature,
            "visits": int(row[0] or 0),
            "actions_attempted": int(row[1] or 0),
            "distinct_actions_tried": int(row[2] or 0),
            "progress_events": int(row[3] or 0),
            "failed_events": int(row[4] or 0),
            "stalled_events": int(row[5] or 0),
            "last_action_name": row[6],
            "last_progress_step": row[7],
            "is_exhausted": bool(row[8]),
            "action_outcomes": action_outcomes,
        }


def load_memory_bundle(
    db_path: str | Path,
    game_id: str,
    state: str,
    frame_signature: str,
    subgoal: str | None,
    legal_actions: list[str],
    board_context: dict[str, Any] | None = None,
    board_motif_tags: list[str] | None = None,
    last_action: str | None = None,
    touch_retrieved: bool = True,
    retrieval_step: int | None = None,
    include_retrieval_metadata: bool = False,
) -> dict[str, Any]:
    path = Path(db_path)
    if not path.exists():
        bundle = _empty_bundle()
        bundle["board_context_summary"] = _normalized_board_context(board_context)
        return bundle
    with closing(_connect(path)) as connection:
        lookup_game_id = _resolve_memory_lookup_game_id(connection, game_id)
        action_stats = _query_action_stats(
            connection,
            """
            SELECT action_name, progress, failed, stalled, total, last_outcome
            FROM live_action_stats
            WHERE game_id = ?
            ORDER BY action_name
            """,
            [lookup_game_id],
        )
        state_stats = _query_action_stats(
            connection,
            """
            SELECT action_name, progress, failed, stalled, total, last_outcome
            FROM live_state_action_stats
            WHERE game_id = ? AND state = ?
            ORDER BY action_name
            """,
            [lookup_game_id, state],
        )
        frame_stats = _query_action_stats(
            connection,
            """
            SELECT action_name, progress, failed, stalled, total, last_outcome
            FROM live_frame_action_stats
            WHERE game_id = ? AND frame_signature = ?
            ORDER BY action_name
            """,
            [lookup_game_id, frame_signature],
        )
        subgoal_stats = _query_action_stats(
            connection,
            """
            SELECT action_name, progress, failed, stalled, total, last_outcome
            FROM live_subgoal_action_stats
            WHERE game_id = ? AND subgoal = ?
            ORDER BY action_name
            """,
            [lookup_game_id, str(subgoal or "")],
        )
        basin_status = load_basin_status(path, lookup_game_id, state, frame_signature)
        basin_action_stats = dict(basin_status.get("action_outcomes", {}))
        pattern_stats = _load_pattern_action_stats(
            connection,
            game_id=lookup_game_id,
            state=state,
            subgoal=subgoal,
            board_context=board_context,
        )
        sequence_stats = _load_sequence_action_stats(
            connection,
            game_id=lookup_game_id,
            state=state,
            subgoal=subgoal,
            last_action=last_action,
        )
        board_cluster_stats = _load_board_cluster_action_stats(
            connection,
            game_id=lookup_game_id,
            board_context=board_context,
        )
        semantic_memories = _load_semantic_memories(
            connection,
            game_id=lookup_game_id,
            state=state,
            subgoal=subgoal,
            board_context=board_context,
        )
        procedural_memories = _load_procedural_memories(
            connection,
            game_id=lookup_game_id,
            state=state,
            subgoal=subgoal,
            last_action=last_action,
        )
        abstract_memories = _load_abstract_memories(
            connection,
            state=state,
            subgoal=subgoal,
            board_context=board_context,
            board_motif_tags=board_motif_tags,
            legal_actions=legal_actions,
        )
        episode_summaries = _load_episode_summaries(
            connection,
            game_id=lookup_game_id,
            state=state,
            frame_signature=frame_signature,
            subgoal=subgoal,
            board_context=board_context,
            last_action=last_action,
        )
        recent_records = _load_recent_records(
            connection,
            game_id=lookup_game_id,
            state=state,
            frame_signature=frame_signature,
            subgoal=subgoal,
            board_context=board_context,
            last_action=last_action,
        )
        if touch_retrieved:
            _touch_retrieved_memory_rows(
                connection,
                semantic_memories=semantic_memories,
                procedural_memories=procedural_memories,
                abstract_memories=abstract_memories,
                step_index_hint=_resolve_retrieval_step(
                    retrieval_step=retrieval_step,
                    semantic_memories=semantic_memories,
                    procedural_memories=procedural_memories,
                    abstract_memories=abstract_memories,
                    recent_records=recent_records,
                ),
            )
    trained_prior_bucket = load_best_prior_bucket(
        path,
        game_id=game_id,
        state=state,
        frame_signature=frame_signature,
        legal_actions=legal_actions,
        subgoal=subgoal,
    )
    suppressed_actions = _derive_suppressed_actions(
        legal_actions=legal_actions,
        basin_action_stats=basin_action_stats,
        state_stats=state_stats,
        subgoal_stats=subgoal_stats,
        basin_status=basin_status,
    )
    board_context_action_stats = _derive_board_context_action_stats(
        board_context=board_context,
        recent_records=recent_records,
    )
    unknown_game_context = _is_unknown_game_context(
        trained_prior_bucket=trained_prior_bucket,
        recent_records=recent_records,
        state_stats=state_stats,
        frame_stats=frame_stats,
        subgoal_stats=subgoal_stats,
    )
    recommended_actions = _derive_recommended_actions(
        legal_actions=legal_actions,
        action_stats=action_stats,
        state_stats=state_stats,
        basin_action_stats=basin_action_stats,
        frame_stats=frame_stats,
        subgoal_stats=subgoal_stats,
        pattern_stats=pattern_stats,
        sequence_stats=sequence_stats,
        board_cluster_stats=board_cluster_stats,
        board_context_stats=board_context_action_stats,
        trained_prior_bucket=trained_prior_bucket,
        abstract_memories=abstract_memories,
        recent_records=recent_records,
        suppressed_actions=suppressed_actions,
        unknown_game_context=unknown_game_context,
    )
    basin_status["is_exhausted"] = _is_basin_exhausted(
        basin_status=basin_status,
        legal_actions=legal_actions,
        suppressed_actions=suppressed_actions,
    )
    working_memory = _build_working_memory(
        state=state,
        frame_signature=frame_signature,
        subgoal=subgoal,
        board_context=board_context,
        legal_actions=legal_actions,
        recent_records=recent_records,
        recommended_actions=recommended_actions,
        trained_prior_bucket=trained_prior_bucket,
        suppressed_actions=suppressed_actions,
        semantic_memories=semantic_memories,
        procedural_memories=procedural_memories,
        abstract_memories=abstract_memories,
        basin_status=basin_status,
    )
    episodic_recalls = _build_episodic_recalls(
        recent_records=recent_records,
        episode_summaries=episode_summaries,
        state=state,
        frame_signature=frame_signature,
        subgoal=subgoal,
    )
    retrieval_summary = _build_retrieval_summary(
        state=state,
        frame_signature=frame_signature,
        subgoal=subgoal,
        board_context=board_context,
        legal_actions=legal_actions,
        recent_records=recent_records,
        episode_summaries=episode_summaries,
        semantic_memories=semantic_memories,
        procedural_memories=procedural_memories,
        abstract_memories=abstract_memories,
        recommended_actions=recommended_actions,
        trained_prior_bucket=trained_prior_bucket,
        suppressed_actions=suppressed_actions,
        unknown_game_context=unknown_game_context,
    )
    return {
        "working_memory": working_memory,
        "episodic_recalls": episodic_recalls,
        "semantic_memories": (
            semantic_memories
            if include_retrieval_metadata
            else _strip_retrieval_metadata(semantic_memories)
        ),
        "procedural_memories": (
            procedural_memories
            if include_retrieval_metadata
            else _strip_retrieval_metadata(procedural_memories)
        ),
        "abstract_memories": (
            abstract_memories
            if include_retrieval_metadata
            else _strip_retrieval_metadata(abstract_memories)
        ),
        "retrieval_summary": retrieval_summary,
        "board_context_summary": _normalized_board_context(board_context),
        "board_context_action_stats": board_context_action_stats,
        "pattern_action_stats": pattern_stats,
        "sequence_action_stats": sequence_stats,
        "board_cluster_action_stats": board_cluster_stats,
        "episode_summaries": episode_summaries,
        "recent_records": recent_records,
        "basin_status": basin_status,
        "basin_action_stats": basin_action_stats,
        "action_stats": action_stats,
        "state_action_stats": state_stats,
        "frame_action_stats": frame_stats,
        "subgoal_action_stats": subgoal_stats,
        "trained_prior_bucket": trained_prior_bucket,
        "recommended_actions": recommended_actions,
        "suppressed_actions": suppressed_actions,
    }


def _strip_retrieval_metadata(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stripped: list[dict[str, Any]] = []
    for row in list(rows or []):
        normalized = dict(row or {})
        normalized.pop("_rowid", None)
        normalized.pop("retrieval_store", None)
        stripped.append(normalized)
    return stripped


def _touch_retrieved_memory_rows(
    connection: sqlite3.Connection,
    *,
    semantic_memories: list[dict[str, Any]],
    procedural_memories: list[dict[str, Any]],
    abstract_memories: list[dict[str, Any]],
    step_index_hint: int,
) -> None:
    semantic_rowids = [
        int(memory.get("_rowid", 0) or 0)
        for memory in semantic_memories
        if int(memory.get("_rowid", 0) or 0) > 0
    ]
    procedural_rowids = [
        int(memory.get("_rowid", 0) or 0)
        for memory in procedural_memories
        if int(memory.get("_rowid", 0) or 0) > 0
    ]
    abstract_rowids = [
        int(memory.get("_rowid", 0) or 0)
        for memory in abstract_memories
        if int(memory.get("_rowid", 0) or 0) > 0
    ]
    if not semantic_rowids and not procedural_rowids and not abstract_rowids:
        return
    try:
        if semantic_rowids:
            placeholders = ",".join("?" for _ in semantic_rowids)
            connection.execute(
                f"""
                UPDATE live_semantic_memory
                SET access_count = access_count + 1,
                    last_access_step = ?
                WHERE rowid IN ({placeholders})
                """,
                [int(step_index_hint), *semantic_rowids],
            )
        if procedural_rowids:
            placeholders = ",".join("?" for _ in procedural_rowids)
            connection.execute(
                f"""
                UPDATE live_procedural_memory
                SET access_count = access_count + 1,
                    last_access_step = ?
                WHERE rowid IN ({placeholders})
                """,
                [int(step_index_hint), *procedural_rowids],
            )
        if abstract_rowids and _table_exists(connection, "abstract_memories"):
            placeholders = ",".join("?" for _ in abstract_rowids)
            connection.execute(
                f"""
                UPDATE abstract_memories
                SET access_count = access_count + 1,
                    last_access_step = ?
                WHERE rowid IN ({placeholders})
                """,
                [int(step_index_hint), *abstract_rowids],
            )
        connection.commit()
    except sqlite3.Error:
        connection.rollback()


def _resolve_retrieval_step(
    *,
    retrieval_step: int | None,
    semantic_memories: list[dict[str, Any]],
    procedural_memories: list[dict[str, Any]],
    abstract_memories: list[dict[str, Any]],
    recent_records: list[dict[str, Any]],
) -> int:
    if retrieval_step is not None:
        return int(retrieval_step)
    return max(
        [0]
        + [int(memory.get("last_step_index", 0) or 0) for memory in semantic_memories]
        + [int(memory.get("last_step_index", 0) or 0) for memory in procedural_memories]
        + [int(memory.get("last_step_index", 0) or 0) for memory in abstract_memories]
        + [int(record.get("step_index", 0) or 0) for record in recent_records[:1]]
    )


def load_memory_bundle_from_stores(
    *,
    hot_db_path: str | Path | None,
    archive_db_path: str | Path | None = None,
    trained_db_path: str | Path | None = None,
    game_id: str,
    state: str,
    frame_signature: str,
    subgoal: str | None,
    legal_actions: list[str],
    board_context: dict[str, Any] | None = None,
    board_motif_tags: list[str] | None = None,
    last_action: str | None = None,
    touch_retrieved: bool = True,
    retrieval_step: int | None = None,
) -> dict[str, Any]:
    include_retrieval_metadata = bool(touch_retrieved)
    _hot_kw = dict(
        game_id=game_id, state=state, frame_signature=frame_signature,
        subgoal=subgoal, legal_actions=legal_actions,
        board_context=board_context, board_motif_tags=board_motif_tags,
        last_action=last_action,
        touch_retrieved=False if include_retrieval_metadata else touch_retrieved,
        retrieval_step=retrieval_step,
        include_retrieval_metadata=include_retrieval_metadata,
        retrieval_store="hot",
    )
    _archive_kw = dict(
        game_id=game_id, state=state, frame_signature=frame_signature,
        subgoal=subgoal, legal_actions=legal_actions,
        board_context=board_context, board_motif_tags=board_motif_tags,
        last_action=last_action,
        allow_same_as=hot_db_path,
        touch_retrieved=False if include_retrieval_metadata else touch_retrieved,
        retrieval_step=retrieval_step,
        include_retrieval_metadata=include_retrieval_metadata,
        retrieval_store="archive",
    )
    _trained_kw = dict(
        game_id=game_id, state=state, frame_signature=frame_signature,
        subgoal=subgoal, legal_actions=legal_actions,
        board_context=board_context, board_motif_tags=board_motif_tags,
        last_action=last_action,
        allow_same_as=hot_db_path,
        touch_retrieved=False,
        retrieval_step=retrieval_step,
        include_retrieval_metadata=False,
        retrieval_store="trained",
    )
    try:
        from agi_core.orch_or.parallel import ThreadPoolManager
        _futures = {
            ThreadPoolManager.submit(
                _safe_load_memory_bundle, hot_db_path, **_hot_kw,
            ): "hot",
            ThreadPoolManager.submit(
                _safe_load_memory_bundle, archive_db_path, **_archive_kw,
            ): "archive",
            ThreadPoolManager.submit(
                _safe_load_memory_bundle, trained_db_path, **_trained_kw,
            ): "trained",
        }
        _results: dict[str, tuple] = {}
        from concurrent.futures import as_completed
        for _f in as_completed(_futures):
            _results[_futures[_f]] = _f.result()
        hot_bundle, hot_error = _results.get("hot", (None, None))
        archive_bundle, archive_error = _results.get("archive", (None, None))
        trained_bundle, trained_error = _results.get("trained", (None, None))
    except Exception:
        hot_bundle, hot_error = _safe_load_memory_bundle(hot_db_path, **_hot_kw)
        archive_bundle, archive_error = _safe_load_memory_bundle(archive_db_path, **_archive_kw)
        trained_bundle, trained_error = _safe_load_memory_bundle(trained_db_path, **_trained_kw)
    if hot_bundle is None and archive_bundle is None and trained_bundle is None:
        if hot_error is not None:
            raise hot_error
        if archive_error is not None:
            raise archive_error
        bundle = _empty_bundle()
        bundle["board_context_summary"] = _normalized_board_context(board_context)
        return bundle

    def _merge_trained(base_bundle, trained_bnd):
        if trained_bnd is None:
            return base_bundle
        base_bundle["abstract_memories"] = _merge_ranked_memory_items(
            base_bundle.get("abstract_memories", []),
            trained_bnd.get("abstract_memories", []),
            limit=6,
        )
        base_bundle["trained_prior_bucket"] = _merge_prior_buckets(
            base_bundle.get("trained_prior_bucket", {}),
            trained_bnd.get("trained_prior_bucket", {}),
        )
        return base_bundle

    if hot_bundle is None and archive_bundle is None:
        bundle = _empty_bundle()
        bundle["board_context_summary"] = _normalized_board_context(board_context)
        bundle = _merge_trained(bundle, trained_bundle)
        return bundle
    if hot_bundle is None:
        base = archive_bundle or _empty_bundle()
        base = _merge_trained(base, trained_bundle)
        base["board_context_summary"] = _normalized_board_context(board_context)
        if include_retrieval_metadata:
            _touch_bundle_memory_rows(
                archive_db_path,
                base,
                retrieval_step=retrieval_step,
                retrieval_store="archive",
            )
            return _strip_bundle_retrieval_metadata(base)
        return base
    if archive_bundle is None:
        base = hot_bundle
        base = _merge_trained(base, trained_bundle)
        base["board_context_summary"] = _normalized_board_context(board_context)
        if include_retrieval_metadata:
            _touch_bundle_memory_rows(
                hot_db_path,
                base,
                retrieval_step=retrieval_step,
                retrieval_store="hot",
            )
            return _strip_bundle_retrieval_metadata(base)
        return base
    merged_bundle = _merge_memory_bundles(
        hot_bundle=hot_bundle,
        archive_bundle=archive_bundle,
        legal_actions=legal_actions,
        board_context=board_context,
    )
    merged_bundle = _merge_trained(merged_bundle, trained_bundle)
    if include_retrieval_metadata:
        _touch_bundle_memory_rows(
            hot_db_path,
            merged_bundle,
            retrieval_step=retrieval_step,
            retrieval_store="hot",
        )
        _touch_bundle_memory_rows(
            archive_db_path,
            merged_bundle,
            retrieval_step=retrieval_step,
            retrieval_store="archive",
        )
        return _strip_bundle_retrieval_metadata(merged_bundle)
    return merged_bundle


def _empty_bundle() -> dict[str, Any]:
    return {
        "working_memory": {},
        "episodic_recalls": [],
        "semantic_memories": [],
        "procedural_memories": [],
        "abstract_memories": [],
        "retrieval_summary": {},
        "board_context_summary": _normalized_board_context(None),
        "board_context_action_stats": {},
        "pattern_action_stats": {},
        "sequence_action_stats": {},
        "board_cluster_action_stats": {},
        "episode_summaries": [],
        "recent_records": [],
        "basin_status": {
            "state": "",
            "frame_signature": "",
            "visits": 0,
            "actions_attempted": 0,
            "distinct_actions_tried": 0,
            "progress_events": 0,
            "failed_events": 0,
            "stalled_events": 0,
            "last_action_name": None,
            "last_progress_step": None,
            "is_exhausted": False,
            "action_outcomes": {},
        },
        "basin_action_stats": {},
        "action_stats": {},
        "state_action_stats": {},
        "frame_action_stats": {},
        "subgoal_action_stats": {},
        "trained_prior_bucket": {},
        "recommended_actions": [],
        "suppressed_actions": [],
    }


def _resolve_memory_lookup_game_id(
    connection: sqlite3.Connection,
    game_id: str,
) -> str:
    aliases = [alias for alias in game_id_aliases(game_id) if alias]
    if not aliases:
        return str(game_id or "")
    primary = aliases[0]
    if _memory_rows_exist(connection, primary):
        return primary
    for alias in aliases[1:]:
        if _memory_rows_exist(connection, alias):
            return alias
    return primary


def _memory_rows_exist(connection: sqlite3.Connection, game_id: str) -> bool:
    if not game_id:
        return False
    for table_name in (
        "live_episode_records",
        "live_action_stats",
        "live_state_action_stats",
        "live_frame_action_stats",
        "live_subgoal_action_stats",
        "live_episode_summaries",
        "live_semantic_memory",
        "live_procedural_memory",
    ):
        if not _table_exists(connection, table_name):
            continue
        row = connection.execute(
            f"SELECT 1 FROM {table_name} WHERE game_id = ? LIMIT 1",
            [game_id],
        ).fetchone()
        if row is not None:
            return True
    return False


def _safe_load_memory_bundle(
    db_path: str | Path | None,
    *,
    game_id: str,
    state: str,
    frame_signature: str,
    subgoal: str | None,
    legal_actions: list[str],
    board_context: dict[str, Any] | None,
    board_motif_tags: list[str] | None,
    last_action: str | None,
    allow_same_as: str | Path | None = None,
    touch_retrieved: bool = True,
    retrieval_step: int | None = None,
    include_retrieval_metadata: bool = False,
    retrieval_store: str | None = None,
) -> tuple[dict[str, Any] | None, Exception | None]:
    if db_path is None:
        return None, None
    path = Path(db_path)
    if allow_same_as is not None and path == Path(allow_same_as):
        return None, None
    try:
        bundle = load_memory_bundle(
            path,
            game_id=game_id,
            state=state,
            frame_signature=frame_signature,
            subgoal=subgoal,
            legal_actions=legal_actions,
            board_context=board_context,
            board_motif_tags=board_motif_tags,
            last_action=last_action,
            touch_retrieved=touch_retrieved,
            retrieval_step=retrieval_step,
            include_retrieval_metadata=include_retrieval_metadata,
        )
        if include_retrieval_metadata and retrieval_store:
            for memory in list(bundle.get("semantic_memories", []) or []):
                if isinstance(memory, dict):
                    memory["retrieval_store"] = retrieval_store
            for memory in list(bundle.get("procedural_memories", []) or []):
                if isinstance(memory, dict):
                    memory["retrieval_store"] = retrieval_store
            for memory in list(bundle.get("abstract_memories", []) or []):
                if isinstance(memory, dict):
                    memory["retrieval_store"] = retrieval_store
        return (bundle, None)
    except Exception as exc:
        if isinstance(exc, sqlite3.OperationalError) and "no such table:" in str(exc):
            bundle = _empty_bundle()
            bundle["board_context_summary"] = _normalized_board_context(board_context)
            return bundle, None
        return None, exc


def _strip_bundle_retrieval_metadata(bundle: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(bundle or {})
    normalized["semantic_memories"] = _strip_retrieval_metadata(
        list(normalized.get("semantic_memories", []) or [])
    )
    normalized["procedural_memories"] = _strip_retrieval_metadata(
        list(normalized.get("procedural_memories", []) or [])
    )
    normalized["abstract_memories"] = _strip_retrieval_metadata(
        list(normalized.get("abstract_memories", []) or [])
    )
    return normalized


def _touch_bundle_memory_rows(
    db_path: str | Path | None,
    bundle: dict[str, Any],
    *,
    retrieval_step: int | None,
    retrieval_store: str,
) -> None:
    if db_path is None:
        return
    semantic_memories = [
        dict(memory or {})
        for memory in list(bundle.get("semantic_memories", []) or [])
        if str(dict(memory or {}).get("retrieval_store", "") or "") == retrieval_store
    ]
    procedural_memories = [
        dict(memory or {})
        for memory in list(bundle.get("procedural_memories", []) or [])
        if str(dict(memory or {}).get("retrieval_store", "") or "") == retrieval_store
    ]
    abstract_memories = [
        dict(memory or {})
        for memory in list(bundle.get("abstract_memories", []) or [])
        if str(dict(memory or {}).get("retrieval_store", "") or "") == retrieval_store
    ]
    if not semantic_memories and not procedural_memories and not abstract_memories:
        return
    path = Path(db_path)
    if not path.exists():
        return
    with closing(_connect(path, write=True)) as connection:
        _touch_retrieved_memory_rows(
            connection,
            semantic_memories=semantic_memories,
            procedural_memories=procedural_memories,
            abstract_memories=abstract_memories,
            step_index_hint=_resolve_retrieval_step(
                retrieval_step=retrieval_step,
                semantic_memories=semantic_memories,
                procedural_memories=procedural_memories,
                abstract_memories=abstract_memories,
                recent_records=[],
            ),
        )


def _merge_memory_bundles(
    *,
    hot_bundle: dict[str, Any],
    archive_bundle: dict[str, Any],
    legal_actions: list[str],
    board_context: dict[str, Any] | None,
) -> dict[str, Any]:
    merged_action_stats = _merge_action_stats_dicts(
        hot_bundle.get("action_stats", {}),
        archive_bundle.get("action_stats", {}),
    )
    merged_state_stats = _merge_action_stats_dicts(
        hot_bundle.get("state_action_stats", {}),
        archive_bundle.get("state_action_stats", {}),
    )
    merged_frame_stats = _merge_action_stats_dicts(
        hot_bundle.get("frame_action_stats", {}),
        archive_bundle.get("frame_action_stats", {}),
    )
    merged_subgoal_stats = _merge_action_stats_dicts(
        hot_bundle.get("subgoal_action_stats", {}),
        archive_bundle.get("subgoal_action_stats", {}),
    )
    merged_pattern_stats = _merge_action_stats_dicts(
        hot_bundle.get("pattern_action_stats", {}),
        archive_bundle.get("pattern_action_stats", {}),
    )
    merged_sequence_stats = _merge_action_stats_dicts(
        hot_bundle.get("sequence_action_stats", {}),
        archive_bundle.get("sequence_action_stats", {}),
    )
    merged_board_cluster_stats = _merge_action_stats_dicts(
        hot_bundle.get("board_cluster_action_stats", {}),
        archive_bundle.get("board_cluster_action_stats", {}),
    )
    merged_board_context_stats = _merge_action_stats_dicts(
        hot_bundle.get("board_context_action_stats", {}),
        archive_bundle.get("board_context_action_stats", {}),
    )
    merged_basin_action_stats = _merge_action_stats_dicts(
        hot_bundle.get("basin_action_stats", {}),
        archive_bundle.get("basin_action_stats", {}),
    )
    merged_recent_records = _merge_recent_records(
        hot_bundle.get("recent_records", []),
        archive_bundle.get("recent_records", []),
        limit=DEFAULT_RECENT_RECORD_LIMIT,
    )
    merged_episode_summaries = _merge_episode_summaries(
        hot_bundle.get("episode_summaries", []),
        archive_bundle.get("episode_summaries", []),
        limit=DEFAULT_EPISODE_SUMMARY_LIMIT,
    )
    merged_semantic_memories = _merge_ranked_memory_items(
        hot_bundle.get("semantic_memories", []),
        archive_bundle.get("semantic_memories", []),
        limit=6,
    )
    merged_procedural_memories = _merge_ranked_memory_items(
        hot_bundle.get("procedural_memories", []),
        archive_bundle.get("procedural_memories", []),
        limit=6,
    )
    merged_abstract_memories = _merge_ranked_memory_items(
        hot_bundle.get("abstract_memories", []),
        archive_bundle.get("abstract_memories", []),
        limit=6,
    )
    merged_trained_prior_bucket = _merge_prior_buckets(
        hot_bundle.get("trained_prior_bucket", {}),
        archive_bundle.get("trained_prior_bucket", {}),
    )
    basin_status = _select_hot_first_basin_status(
        hot_bundle.get("basin_status", {}),
        archive_bundle.get("basin_status", {}),
    )
    suppressed_actions = _derive_suppressed_actions(
        legal_actions=legal_actions,
        basin_action_stats=merged_basin_action_stats,
        state_stats=merged_state_stats,
        subgoal_stats=merged_subgoal_stats,
        basin_status=basin_status,
    )
    unknown_game_context = _is_unknown_game_context(
        trained_prior_bucket=merged_trained_prior_bucket,
        recent_records=merged_recent_records,
        state_stats=merged_state_stats,
        frame_stats=merged_frame_stats,
        subgoal_stats=merged_subgoal_stats,
    )
    recommended_actions = _derive_recommended_actions(
        legal_actions=legal_actions,
        action_stats=merged_action_stats,
        state_stats=merged_state_stats,
        basin_action_stats=merged_basin_action_stats,
        frame_stats=merged_frame_stats,
        subgoal_stats=merged_subgoal_stats,
        pattern_stats=merged_pattern_stats,
        sequence_stats=merged_sequence_stats,
        board_cluster_stats=merged_board_cluster_stats,
        board_context_stats=merged_board_context_stats,
        trained_prior_bucket=merged_trained_prior_bucket,
        abstract_memories=merged_abstract_memories,
        recent_records=merged_recent_records,
        suppressed_actions=suppressed_actions,
        unknown_game_context=unknown_game_context,
    )
    basin_status["is_exhausted"] = _is_basin_exhausted(
        basin_status=basin_status,
        legal_actions=legal_actions,
        suppressed_actions=suppressed_actions,
    )
    merged_working_memory = _merge_working_memory(
        hot_bundle.get("working_memory", {}),
        archive_bundle.get("working_memory", {}),
        recent_records=merged_recent_records,
        semantic_memories=merged_semantic_memories,
        procedural_memories=merged_procedural_memories,
        abstract_memories=merged_abstract_memories,
        recommended_actions=recommended_actions,
        trained_prior_bucket=merged_trained_prior_bucket,
        suppressed_actions=suppressed_actions,
        basin_status=basin_status,
    )
    merged_episodic_recalls = _merge_ranked_memory_items(
        hot_bundle.get("episodic_recalls", []),
        archive_bundle.get("episodic_recalls", []),
        limit=8,
    )
    retrieval_summary = _merge_retrieval_summary(
        hot_bundle.get("retrieval_summary", {}),
        archive_bundle.get("retrieval_summary", {}),
        recent_records=merged_recent_records,
        episode_summaries=merged_episode_summaries,
        episodic_recalls=merged_episodic_recalls,
        semantic_memories=merged_semantic_memories,
        procedural_memories=merged_procedural_memories,
        abstract_memories=merged_abstract_memories,
        recommended_actions=recommended_actions,
        trained_prior_bucket=merged_trained_prior_bucket,
        suppressed_actions=suppressed_actions,
        unknown_game_context=unknown_game_context,
    )
    return {
        "working_memory": merged_working_memory,
        "episodic_recalls": merged_episodic_recalls,
        "semantic_memories": merged_semantic_memories,
        "procedural_memories": merged_procedural_memories,
        "abstract_memories": merged_abstract_memories,
        "retrieval_summary": retrieval_summary,
        "board_context_summary": dict(
            hot_bundle.get("board_context_summary")
            or archive_bundle.get("board_context_summary")
            or _normalized_board_context(board_context)
        ),
        "board_context_action_stats": merged_board_context_stats,
        "pattern_action_stats": merged_pattern_stats,
        "sequence_action_stats": merged_sequence_stats,
        "board_cluster_action_stats": merged_board_cluster_stats,
        "episode_summaries": merged_episode_summaries,
        "recent_records": merged_recent_records,
        "basin_status": basin_status,
        "basin_action_stats": merged_basin_action_stats,
        "action_stats": merged_action_stats,
        "state_action_stats": merged_state_stats,
        "frame_action_stats": merged_frame_stats,
        "subgoal_action_stats": merged_subgoal_stats,
        "trained_prior_bucket": merged_trained_prior_bucket,
        "recommended_actions": recommended_actions,
        "suppressed_actions": suppressed_actions,
    }


def _load_recent_records(
    connection: sqlite3.Connection,
    *,
    game_id: str,
    state: str,
    frame_signature: str,
    subgoal: str | None,
    board_context: dict[str, Any] | None,
    last_action: str | None = None,
    limit: int = DEFAULT_RECENT_RECORD_LIMIT,
) -> list[dict[str, Any]]:
    select_clause = """
        SELECT
            id,
            game_id,
            step_index,
            action_name,
            outcome,
            progress_delta,
            next_state,
            prior_state,
            prior_frame_signature,
            next_frame_signature,
            subgoal,
            decision_source,
            stagnation_before,
            stagnation_after,
            legal_actions_json,
            prior_board_height,
            prior_board_width,
            prior_board_active_cells,
            prior_board_source_attr,
            prior_board_default_probe_json
        FROM live_episode_records
    """
    candidate_rows: list[tuple[Any, ...]] = []
    seen_ids: set[int] = set()

    def extend_candidates(where_clause: str, parameters: list[Any], row_limit: int) -> None:
        rows = connection.execute(
            f"""
            {select_clause}
            WHERE {where_clause}
            ORDER BY id DESC
            LIMIT {int(row_limit)}
            """,
            parameters,
        ).fetchall()
        for row in rows:
            row_id = int(row[0] or 0)
            if row_id in seen_ids:
                continue
            seen_ids.add(row_id)
            candidate_rows.append(row)

    if state and frame_signature:
        extend_candidates(
            "game_id = ? AND prior_state = ? AND prior_frame_signature = ?",
            [game_id, state, frame_signature],
            max(limit * 2, 16),
        )
    if state:
        extend_candidates(
            "game_id = ? AND prior_state = ?",
            [game_id, state],
            max(limit * 2, 16),
        )
    current_subgoal = str(subgoal or "")
    current_board_context = _normalized_board_context(board_context)
    current_board_family = _board_family_signature(current_board_context)
    current_last_action = str(last_action or "")
    if current_subgoal:
        extend_candidates(
            "game_id = ? AND subgoal = ?",
            [game_id, current_subgoal],
            max(limit * 2, 16),
        )
    if frame_signature:
        extend_candidates(
            "game_id = ? AND prior_frame_signature = ?",
            [game_id, frame_signature],
            max(limit * 2, 16),
        )
    extend_candidates(
        "game_id = ?",
        [game_id],
        max(limit * 4, 24),
    )
    scored_rows: list[tuple[tuple[int, int, int, int, int, int, int, int], tuple[Any, ...]]] = []
    for row in candidate_rows:
        prior_state = str(row[7] or "")
        prior_frame_signature = str(row[8] or "")
        record_subgoal = str(row[10] or "")
        record_action_name = str(row[3] or "")
        row_board_summary = {
            "board_height": int(row[15] or 0),
            "board_width": int(row[16] or 0),
            "board_active_cells": int(row[17] or 0),
            "board_source_attr": str(row[18] or "") or None,
            "board_default_probe": _deserialize_probe_json(row[19]),
        }
        exact_basin = 1 if state and frame_signature and prior_state == state and prior_frame_signature == frame_signature else 0
        same_state = 1 if state and prior_state == state else 0
        same_frame = 1 if frame_signature and prior_frame_signature == frame_signature else 0
        same_subgoal = 1 if current_subgoal and record_subgoal == current_subgoal else 0
        same_board_family = 1 if current_board_family and current_board_family == _board_family_signature(row_board_summary) else 0
        same_last_action = 1 if current_last_action and record_action_name == current_last_action else 0
        board_match = board_context_match_score(current_board_context, row_board_summary)
        scored_rows.append(
            (
                (
                    exact_basin,
                    same_state,
                    same_subgoal,
                    same_board_family,
                    same_last_action,
                    board_match,
                    same_frame,
                    int(row[0] or 0),
                ),
                row,
            )
        )
    scored_rows.sort(key=lambda item: item[0], reverse=True)
    recent_records: list[dict[str, Any]] = []
    for _, row in scored_rows[:limit]:
        recent_records.append(
            {
                "game_id": str(row[1]),
                "step_index": int(row[2] or 0),
                "action_name": str(row[3]),
                "outcome": str(row[4]),
                "progress_delta": int(row[5] or 0),
                "game_state": str(row[6] or ""),
                "prior_game_state": str(row[7] or ""),
                "prior_frame_signature": str(row[8] or ""),
                "next_frame_signature": str(row[9] or ""),
                "decision_subgoal": str(row[10] or ""),
                "decision_source": str(row[11] or ""),
                "stagnation_before": int(row[12] or 0),
                "stagnation_after": int(row[13] or 0),
                "prior_legal_actions": json.loads(row[14] or "[]"),
                "board_context_summary": {
                    "board_height": int(row[15] or 0),
                    "board_width": int(row[16] or 0),
                    "board_active_cells": int(row[17] or 0),
                    "board_source_attr": str(row[18] or "") or None,
                    "board_default_probe": _deserialize_probe_json(row[19]),
                },
            }
        )
    return recent_records


def _load_episode_summaries(
    connection: sqlite3.Connection,
    *,
    game_id: str,
    state: str,
    frame_signature: str,
    subgoal: str | None,
    board_context: dict[str, Any] | None,
    last_action: str | None = None,
    limit: int = DEFAULT_EPISODE_SUMMARY_LIMIT,
) -> list[dict[str, Any]]:
    select_clause = """
        SELECT
            game_id,
            prior_state,
            prior_frame_signature,
            subgoal,
            board_height,
            board_width,
            board_active_cells,
            board_source_attr,
            board_default_probe_json,
            action_name,
            progress,
            failed,
            stalled,
            total,
            last_outcome,
            first_step_index,
            last_step_index,
            recent_action_sequence_json,
            recent_outcome_sequence_json,
            decision_sources_json,
            max_stagnation_after
        FROM live_episode_summaries
    """
    candidate_rows: list[tuple[Any, ...]] = []
    seen_keys: set[tuple[str, ...]] = set()

    def extend_candidates(where_clause: str, parameters: list[Any], row_limit: int) -> None:
        rows = connection.execute(
            f"""
            {select_clause}
            WHERE {where_clause}
            ORDER BY last_step_index DESC
            LIMIT {int(row_limit)}
            """,
            parameters,
        ).fetchall()
        for row in rows:
            dedupe_key = (
                str(row[0] or ""),
                str(row[1] or ""),
                str(row[2] or ""),
                str(row[3] or ""),
                str(row[4] or 0),
                str(row[5] or 0),
                str(row[6] or 0),
                str(row[7] or ""),
                str(row[8] or ""),
                str(row[9] or ""),
            )
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            candidate_rows.append(row)

    current_subgoal = str(subgoal or "")
    current_board_context = _normalized_board_context(board_context)
    current_board_family = _board_family_signature(current_board_context)
    current_last_action = str(last_action or "")
    if state and frame_signature:
        extend_candidates(
            "game_id = ? AND prior_state = ? AND prior_frame_signature = ?",
            [game_id, state, frame_signature],
            max(limit * 6, 48),
        )
    if state:
        extend_candidates(
            "game_id = ? AND prior_state = ?",
            [game_id, state],
            max(limit * 6, 48),
        )
    if current_subgoal:
        extend_candidates(
            "game_id = ? AND subgoal = ?",
            [game_id, current_subgoal],
            max(limit * 6, 48),
        )
    if frame_signature:
        extend_candidates(
            "game_id = ? AND prior_frame_signature = ?",
            [game_id, frame_signature],
            max(limit * 6, 48),
        )
    extend_candidates(
        "game_id = ?",
        [game_id],
        max(limit * 12, 96),
    )
    grouped: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in candidate_rows:
        prior_state = str(row[1] or "")
        prior_frame_signature = str(row[2] or "")
        record_subgoal = str(row[3] or "")
        board_summary = {
            "board_height": int(row[4] or 0),
            "board_width": int(row[5] or 0),
            "board_active_cells": int(row[6] or 0),
            "board_source_attr": str(row[7] or "") or None,
            "board_default_probe": _deserialize_probe_json(row[8]),
        }
        score = (
            1 if state and frame_signature and prior_state == state and prior_frame_signature == frame_signature else 0,
            1 if state and prior_state == state else 0,
            1 if current_subgoal and record_subgoal == current_subgoal else 0,
            1 if current_board_family and current_board_family == _board_family_signature(board_summary) else 0,
            1 if current_last_action and _summary_sequence_matches_last_action(_deserialize_json_list(row[17]), current_last_action) else 0,
            board_context_match_score(current_board_context, board_summary),
            1 if frame_signature and prior_frame_signature == frame_signature else 0,
            int(row[16] or 0),
        )
        group_key = (
            str(row[0] or ""),
            prior_state,
            prior_frame_signature,
            record_subgoal,
            str(board_summary["board_height"]),
            str(board_summary["board_width"]),
            str(board_summary["board_active_cells"]),
            str(board_summary["board_source_attr"] or ""),
            json.dumps(dict(board_summary["board_default_probe"] or {}), sort_keys=True),
        )
        summary = grouped.get(group_key)
        if summary is None:
            summary = {
                "game_id": str(row[0] or ""),
                "prior_game_state": prior_state,
                "prior_frame_signature": prior_frame_signature,
                "decision_subgoal": record_subgoal,
                "board_context_summary": board_summary,
                "action_outcomes": {},
                "first_step_index": int(row[15] or 0),
                "total_events": 0,
                "last_step_index": int(row[16] or 0),
                "recent_action_sequence": _deserialize_json_list(row[17]),
                "recent_outcome_sequence": _deserialize_json_list(row[18]),
                "decision_sources": _deserialize_json_object(row[19]),
                "max_stagnation_after": int(row[20] or 0),
                "_score": score,
            }
            grouped[group_key] = summary
        elif score > tuple(summary["_score"]):
            summary["_score"] = score
            summary["first_step_index"] = min(
                int(summary.get("first_step_index", 0) or 0) or int(row[15] or 0),
                int(row[15] or 0),
            )
            summary["last_step_index"] = max(
                int(summary.get("last_step_index", 0) or 0),
                int(row[16] or 0),
            )
            summary["max_stagnation_after"] = max(
                int(summary.get("max_stagnation_after", 0) or 0),
                int(row[20] or 0),
            )
        action_name = str(row[9] or "")
        if action_name:
            summary["action_outcomes"][action_name] = {
                "progress": int(row[10] or 0),
                "failed": int(row[11] or 0),
                "stalled": int(row[12] or 0),
                "total": int(row[13] or 0),
                "last_outcome": str(row[14] or "") or None,
            }
        if not summary.get("recent_action_sequence"):
            summary["recent_action_sequence"] = _deserialize_json_list(row[17])
        if not summary.get("recent_outcome_sequence"):
            summary["recent_outcome_sequence"] = _deserialize_json_list(row[18])
        if not summary.get("decision_sources"):
            summary["decision_sources"] = _deserialize_json_object(row[19])
        summary["total_events"] = int(summary.get("total_events", 0) or 0) + int(row[13] or 0)
    ranked = sorted(
        grouped.values(),
        key=lambda item: (
            tuple(item.get("_score", ())),
            int(item.get("total_events", 0) or 0),
            int(item.get("last_step_index", 0) or 0),
        ),
        reverse=True,
    )
    summaries: list[dict[str, Any]] = []
    for summary in ranked[:limit]:
        summaries.append(
            {
                "game_id": str(summary.get("game_id", "") or ""),
                "prior_game_state": str(summary.get("prior_game_state", "") or ""),
                "prior_frame_signature": str(summary.get("prior_frame_signature", "") or ""),
                "decision_subgoal": str(summary.get("decision_subgoal", "") or ""),
                "board_context_summary": dict(summary.get("board_context_summary", {}) or {}),
                "action_outcomes": dict(summary.get("action_outcomes", {}) or {}),
                "first_step_index": int(summary.get("first_step_index", 0) or 0),
                "total_events": int(summary.get("total_events", 0) or 0),
                "last_step_index": int(summary.get("last_step_index", 0) or 0),
                "recent_action_sequence": list(summary.get("recent_action_sequence", []) or []),
                "recent_outcome_sequence": list(summary.get("recent_outcome_sequence", []) or []),
                "decision_sources": dict(summary.get("decision_sources", {}) or {}),
                "max_stagnation_after": int(summary.get("max_stagnation_after", 0) or 0),
            }
        )
    return summaries


def _load_pattern_action_stats(
    connection: sqlite3.Connection,
    *,
    game_id: str,
    state: str,
    subgoal: str | None,
    board_context: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    current_board_context = _normalized_board_context(board_context)
    return _query_action_stats(
        connection,
        """
        SELECT action_name, progress, failed, stalled, total, last_outcome
        FROM live_pattern_stats
        WHERE game_id = ? AND state = ? AND subgoal = ?
          AND board_height = ? AND board_width = ? AND board_active_cells = ?
          AND board_source_attr = ?
        ORDER BY action_name
        """,
        [
            game_id,
            state,
            str(subgoal or ""),
            int(current_board_context.get("board_height", 0) or 0),
            int(current_board_context.get("board_width", 0) or 0),
            int(current_board_context.get("board_active_cells", 0) or 0),
            str(current_board_context.get("board_source_attr") or ""),
        ],
    )


def _load_sequence_action_stats(
    connection: sqlite3.Connection,
    *,
    game_id: str,
    state: str,
    subgoal: str | None,
    last_action: str | None,
) -> dict[str, dict[str, Any]]:
    if not last_action:
        return {}
    return _query_action_stats(
        connection,
        """
        SELECT action_name, progress, failed, stalled, total, last_outcome
        FROM live_sequence_stats
        WHERE game_id = ? AND state = ? AND subgoal = ? AND previous_action_name = ?
        ORDER BY action_name
        """,
        [game_id, state, str(subgoal or ""), str(last_action or "")],
    )


def _load_board_cluster_action_stats(
    connection: sqlite3.Connection,
    *,
    game_id: str,
    board_context: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    current_board_context = _normalized_board_context(board_context)
    return _query_action_stats(
        connection,
        """
        SELECT action_name, progress, failed, stalled, total, last_outcome
        FROM live_board_cluster_stats
        WHERE game_id = ?
          AND board_height = ? AND board_width = ? AND board_active_cells = ?
          AND board_source_attr = ?
        ORDER BY action_name
        """,
        [
            game_id,
            int(current_board_context.get("board_height", 0) or 0),
            int(current_board_context.get("board_width", 0) or 0),
            int(current_board_context.get("board_active_cells", 0) or 0),
            str(current_board_context.get("board_source_attr") or ""),
        ],
    )


def _load_semantic_memories(
    connection: sqlite3.Connection,
    *,
    game_id: str,
    state: str,
    subgoal: str | None,
    board_context: dict[str, Any] | None,
    limit: int = 6,
) -> list[dict[str, Any]]:
    if not _table_exists(connection, "live_semantic_memory"):
        return []
    current_board_context = _normalized_board_context(board_context)
    current_board_family = _board_family_signature(current_board_context)
    current_subgoal = str(subgoal or "")
    select_clause = """
        SELECT
            rowid,
            state,
            subgoal,
            board_height,
            board_width,
            board_active_cells,
            board_source_attr,
            action_name,
            lesson_text,
            salience,
            confidence,
            progress,
            failed,
            stalled,
            total,
            last_outcome,
            last_step_index,
            access_count,
            last_access_step
        FROM live_semantic_memory
    """
    candidate_rows: list[tuple[Any, ...]] = []
    seen_ids: set[int] = set()

    def extend_candidates(where_clause: str, parameters: list[Any], row_limit: int) -> None:
        rows = connection.execute(
            f"""
            {select_clause}
            WHERE {where_clause}
            ORDER BY last_step_index DESC, salience DESC
            LIMIT {int(row_limit)}
            """,
            parameters,
        ).fetchall()
        for row in rows:
            row_id = int(row[0] or 0)
            if row_id in seen_ids:
                continue
            seen_ids.add(row_id)
            candidate_rows.append(row)

    board_height = int(current_board_context.get("board_height", 0) or 0)
    board_width = int(current_board_context.get("board_width", 0) or 0)
    board_active_cells = int(current_board_context.get("board_active_cells", 0) or 0)
    board_source_attr = str(current_board_context.get("board_source_attr") or "")
    if state and current_subgoal and board_height > 0 and board_width > 0:
        extend_candidates(
            """
            game_id = ? AND state = ? AND subgoal = ?
              AND board_height = ? AND board_width = ?
              AND board_active_cells = ? AND board_source_attr = ?
            """,
            [
                game_id,
                state,
                current_subgoal,
                board_height,
                board_width,
                board_active_cells,
                board_source_attr,
            ],
            max(limit * 2, 16),
        )
    if state and current_subgoal:
        extend_candidates(
            "game_id = ? AND state = ? AND subgoal = ?",
            [game_id, state, current_subgoal],
            max(limit * 2, 16),
        )
    if state and board_height > 0 and board_width > 0:
        extend_candidates(
            """
            game_id = ? AND state = ?
              AND board_height = ? AND board_width = ?
              AND board_active_cells = ? AND board_source_attr = ?
            """,
            [
                game_id,
                state,
                board_height,
                board_width,
                board_active_cells,
                board_source_attr,
            ],
            max(limit * 2, 16),
        )
    if current_subgoal and board_height > 0 and board_width > 0:
        extend_candidates(
            """
            game_id = ? AND subgoal = ?
              AND board_height = ? AND board_width = ?
              AND board_active_cells = ? AND board_source_attr = ?
            """,
            [
                game_id,
                current_subgoal,
                board_height,
                board_width,
                board_active_cells,
                board_source_attr,
            ],
            max(limit * 2, 16),
        )
    if state:
        extend_candidates(
            "game_id = ? AND state = ?",
            [game_id, state],
            max(limit * 4, 24),
        )
    if current_subgoal:
        extend_candidates(
            "game_id = ? AND subgoal = ?",
            [game_id, current_subgoal],
            max(limit * 4, 24),
        )
    extend_candidates(
        "game_id = ?",
        [game_id],
        max(limit * 8, 48),
    )
    ranked: list[tuple[tuple[float, ...], dict[str, Any], int]] = []
    for row in candidate_rows:
        board_summary = {
            "board_height": int(row[3] or 0),
            "board_width": int(row[4] or 0),
            "board_active_cells": int(row[5] or 0),
            "board_source_attr": str(row[6] or "") or None,
            "board_default_probe": {},
        }
        progress = int(row[11] or 0)
        failed = int(row[12] or 0)
        stalled = int(row[13] or 0)
        total = int(row[14] or 0)
        salience = float(row[9] or 0.0)
        confidence = float(row[10] or 0.0)
        access_count = int(row[17] or 0)
        last_access_step = int(row[18] or 0)
        score = (
            1.0 if state and str(row[1] or "") == state else 0.0,
            1.0 if current_subgoal and str(row[2] or "") == current_subgoal else 0.0,
            float(board_context_match_score(current_board_context, board_summary)),
            1.0 if current_board_family and current_board_family == _board_family_signature(board_summary) else 0.0,
            _success_bias(progress=progress, failed=failed, stalled=stalled, total=total),
            salience,
            confidence,
            min(1.0, float(access_count) / 8.0),
            min(1.0, float(last_access_step) / 1000.0),
            float(int(row[16] or 0)),
        )
        retrieval_rank = (
            score[0],
            score[1],
            score[2],
            score[3],
            score[4],
            score[5],
            score[6],
            score[7],
            score[8],
            min(1.0, float(int(row[16] or 0)) / 1000.0),
        )
        memory = {
            "memory_type": "semantic",
            "_rowid": int(row[0] or 0),
            "state": str(row[1] or ""),
            "subgoal": str(row[2] or ""),
            "board_context_summary": board_summary,
            "action_name": str(row[7] or ""),
            "lesson": str(row[8] or ""),
            "salience": salience,
            "confidence": confidence,
            "progress": progress,
            "failed": failed,
            "stalled": stalled,
            "total": total,
            "last_outcome": str(row[15] or "") or None,
            "last_step_index": int(row[16] or 0),
            "access_count": access_count,
            "last_access_step": last_access_step,
            "score": round(sum(retrieval_rank[:-1]), 4),
            "retrieval_rank": list(retrieval_rank),
            "retrieval_provenance": "semantic-store",
        }
        ranked.append((score, memory, int(row[0] or 0)))
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected_rows = ranked[:limit]
    return [memory for _, memory, _ in selected_rows]


def _load_procedural_memories(
    connection: sqlite3.Connection,
    *,
    game_id: str,
    state: str,
    subgoal: str | None,
    last_action: str | None,
    limit: int = 6,
) -> list[dict[str, Any]]:
    if not _table_exists(connection, "live_procedural_memory"):
        return []
    current_subgoal = str(subgoal or "")
    current_last_action = str(last_action or "")
    select_clause = """
        SELECT
            rowid,
            state,
            subgoal,
            previous_action_name,
            action_name,
            strategy_tag,
            salience,
            confidence,
            progress,
            failed,
            stalled,
            total,
            last_outcome,
            last_step_index,
            access_count,
            last_access_step
        FROM live_procedural_memory
    """
    candidate_rows: list[tuple[Any, ...]] = []
    seen_ids: set[int] = set()

    def extend_candidates(where_clause: str, parameters: list[Any], row_limit: int) -> None:
        rows = connection.execute(
            f"""
            {select_clause}
            WHERE {where_clause}
            ORDER BY last_step_index DESC, salience DESC
            LIMIT {int(row_limit)}
            """,
            parameters,
        ).fetchall()
        for row in rows:
            row_id = int(row[0] or 0)
            if row_id in seen_ids:
                continue
            seen_ids.add(row_id)
            candidate_rows.append(row)

    if state and current_subgoal and current_last_action:
        extend_candidates(
            "game_id = ? AND state = ? AND subgoal = ? AND previous_action_name = ?",
            [game_id, state, current_subgoal, current_last_action],
            max(limit * 2, 16),
        )
    if state and current_subgoal:
        extend_candidates(
            "game_id = ? AND state = ? AND subgoal = ?",
            [game_id, state, current_subgoal],
            max(limit * 2, 16),
        )
    if state and current_last_action:
        extend_candidates(
            "game_id = ? AND state = ? AND previous_action_name = ?",
            [game_id, state, current_last_action],
            max(limit * 2, 16),
        )
    if current_subgoal and current_last_action:
        extend_candidates(
            "game_id = ? AND subgoal = ? AND previous_action_name = ?",
            [game_id, current_subgoal, current_last_action],
            max(limit * 2, 16),
        )
    if state:
        extend_candidates(
            "game_id = ? AND state = ?",
            [game_id, state],
            max(limit * 4, 24),
        )
    if current_subgoal:
        extend_candidates(
            "game_id = ? AND subgoal = ?",
            [game_id, current_subgoal],
            max(limit * 4, 24),
        )
    if current_last_action:
        extend_candidates(
            "game_id = ? AND previous_action_name = ?",
            [game_id, current_last_action],
            max(limit * 4, 24),
        )
    extend_candidates(
        "game_id = ?",
        [game_id],
        max(limit * 8, 48),
    )
    ranked: list[tuple[tuple[float, ...], dict[str, Any], int]] = []
    for row in candidate_rows:
        progress = int(row[8] or 0)
        failed = int(row[9] or 0)
        stalled = int(row[10] or 0)
        total = int(row[11] or 0)
        salience = float(row[6] or 0.0)
        confidence = float(row[7] or 0.0)
        access_count = int(row[14] or 0)
        last_access_step = int(row[15] or 0)
        preferred = progress > max(failed, stalled)
        score = (
            1.0 if state and str(row[1] or "") == state else 0.0,
            1.0 if current_subgoal and str(row[2] or "") == current_subgoal else 0.0,
            1.0 if current_last_action and str(row[3] or "") == current_last_action else 0.0,
            _success_bias(progress=progress, failed=failed, stalled=stalled, total=total),
            salience,
            confidence,
            min(1.0, float(access_count) / 8.0),
            min(1.0, float(last_access_step) / 1000.0),
            float(int(row[13] or 0)),
        )
        retrieval_rank = (
            score[0],
            score[1],
            score[2],
            score[3],
            score[4],
            score[5],
            score[6],
            score[7],
            min(1.0, float(int(row[13] or 0)) / 1000.0),
        )
        ranked.append(
            (
                score,
                {
                    "memory_type": "procedural",
                    "_rowid": int(row[0] or 0),
                    "state": str(row[1] or ""),
                    "subgoal": str(row[2] or ""),
                    "previous_action_name": str(row[3] or ""),
                    "action_name": str(row[4] or ""),
                    "strategy_tag": str(row[5] or ""),
                    "salience": salience,
                    "confidence": confidence,
                    "progress": progress,
                    "failed": failed,
                    "stalled": stalled,
                    "total": total,
                    "last_outcome": str(row[12] or "") or None,
                    "last_step_index": int(row[13] or 0),
                    "access_count": access_count,
                    "last_access_step": last_access_step,
                    "preferred": preferred,
                    "score": round(sum(retrieval_rank[:-1]), 4),
                    "retrieval_rank": list(retrieval_rank),
                    "retrieval_provenance": "procedural-store",
                },
                int(row[0] or 0),
            )
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected_rows = ranked[:limit]
    return [memory for _, memory, _ in selected_rows]


def _load_abstract_memories(
    connection: sqlite3.Connection,
    *,
    state: str,
    subgoal: str | None,
    board_context: dict[str, Any] | None,
    board_motif_tags: list[str] | None,
    legal_actions: list[str],
    limit: int = 6,
) -> list[dict[str, Any]]:
    if not _table_exists(connection, "abstract_memories"):
        return []
    current_board_context = _normalized_board_context(board_context)
    current_subgoal = str(subgoal or "")
    current_motifs = _normalized_motif_tags(board_motif_tags)
    legal = {str(action_name or "") for action_name in list(legal_actions or [])}
    board_height = int(current_board_context.get("board_height", 0) or 0)
    board_width = int(current_board_context.get("board_width", 0) or 0)
    board_active_cells = int(current_board_context.get("board_active_cells", 0) or 0)
    board_source_attr = str(current_board_context.get("board_source_attr") or "")
    candidate_rows: list[tuple[Any, ...]] = []
    seen_ids: set[int] = set()

    def extend_candidates(where_clause: str, parameters: list[Any], row_limit: int) -> None:
        rows = connection.execute(
            f"""
            SELECT
                rowid,
                subgoal,
                state,
                board_height,
                board_width,
                board_active_cells,
                board_source_attr,
                motif_tags_json,
                action_name,
                lead_in_actions_json,
                lesson_text,
                salience,
                confidence,
                progress,
                failed,
                stalled,
                total,
                last_outcome,
                last_step_index,
                access_count,
                last_access_step,
                source_game_count,
                source_games_json
            FROM abstract_memories
            WHERE {where_clause}
            ORDER BY last_step_index DESC, salience DESC
            LIMIT {int(row_limit)}
            """,
            parameters,
        ).fetchall()
        for row in rows:
            row_id = int(row[0] or 0)
            if row_id in seen_ids:
                continue
            seen_ids.add(row_id)
            candidate_rows.append(row)

    if current_subgoal and board_height > 0 and board_width > 0:
        extend_candidates(
            """
            subgoal = ? AND board_height = ? AND board_width = ?
              AND board_active_cells = ? AND board_source_attr = ?
            """,
            [current_subgoal, board_height, board_width, board_active_cells, board_source_attr],
            max(limit * 2, 16),
        )
    if current_subgoal:
        extend_candidates("subgoal = ?", [current_subgoal], max(limit * 4, 24))
    if state:
        extend_candidates("state = ?", [state], max(limit * 3, 18))
    extend_candidates("1 = 1", [], max(limit * 8, 48))

    ranked: list[tuple[tuple[float, ...], dict[str, Any], int]] = []
    for row in candidate_rows:
        action_name = str(row[8] or "")
        if legal and action_name not in legal:
            continue
        candidate_board_context = {
            "board_height": int(row[3] or 0),
            "board_width": int(row[4] or 0),
            "board_active_cells": int(row[5] or 0),
            "board_source_attr": str(row[6] or "") or None,
            "board_default_probe": {},
        }
        motif_tags = _normalized_motif_tags(json.loads(str(row[7] or "[]")))
        progress = int(row[13] or 0)
        failed = int(row[14] or 0)
        stalled = int(row[15] or 0)
        total = int(row[16] or 0)
        access_count = int(row[19] or 0)
        last_access_step = int(row[20] or 0)
        motif_overlap = len(set(current_motifs).intersection(motif_tags))
        score = (
            1.0 if current_subgoal and str(row[1] or "") == current_subgoal else 0.0,
            float(board_context_match_score(current_board_context, candidate_board_context)),
            float(motif_overlap),
            _success_bias(progress=progress, failed=failed, stalled=stalled, total=total),
            min(1.0, float(access_count) / 8.0),
            min(1.0, float(last_access_step) / 1000.0),
            float(int(row[18] or 0)),
        )
        retrieval_rank = (
            score[0],
            score[1],
            score[2],
            score[3],
            score[4],
            score[5],
            min(1.0, float(int(row[18] or 0)) / 1000.0),
        )
        ranked.append(
            (
                score,
                {
                    "memory_type": "abstract",
                    "_rowid": int(row[0] or 0),
                    "subgoal": str(row[1] or ""),
                    "state": str(row[2] or ""),
                    "board_context_summary": candidate_board_context,
                    "motif_tags": motif_tags,
                    "action_name": action_name,
                    "lead_in_actions": list(json.loads(str(row[9] or "[]"))),
                    "lesson": str(row[10] or ""),
                    "salience": float(row[11] or 0.0),
                    "confidence": float(row[12] or 0.0),
                    "progress": progress,
                    "failed": failed,
                    "stalled": stalled,
                    "total": total,
                    "last_outcome": str(row[17] or "") or None,
                    "last_step_index": int(row[18] or 0),
                    "access_count": access_count,
                    "last_access_step": last_access_step,
                    "source_game_count": int(row[21] or 0),
                    "source_games": list(json.loads(str(row[22] or "[]"))),
                    "motif_overlap": motif_overlap,
                    "score": round(sum(retrieval_rank), 4),
                    "retrieval_rank": list(retrieval_rank),
                    "retrieval_provenance": "abstract-store",
                },
                int(row[0] or 0),
            )
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [memory for _, memory, _ in ranked[:limit]]


def _build_working_memory(
    *,
    state: str,
    frame_signature: str,
    subgoal: str | None,
    board_context: dict[str, Any] | None,
    legal_actions: list[str],
    recent_records: list[dict[str, Any]],
    recommended_actions: list[str],
    trained_prior_bucket: dict[str, int],
    suppressed_actions: list[str],
    semantic_memories: list[dict[str, Any]],
    procedural_memories: list[dict[str, Any]],
    abstract_memories: list[dict[str, Any]],
    basin_status: dict[str, Any],
) -> dict[str, Any]:
    normalized_board_context = _normalized_board_context(board_context)
    recent_failures = _working_memory_recent_failures(recent_records)
    open_failure_modes = _working_memory_open_failure_modes(recent_records)
    failure_streaks = _working_memory_failure_streaks(recent_records)
    semantic_focus_actions = _working_memory_focus_actions(
        semantic_memories,
        limit=3,
    )
    procedural_focus_actions = _working_memory_focus_actions(
        procedural_memories,
        limit=3,
    )
    abstract_focus_actions = _compact_abstract_focus_actions(
        abstract_memories,
        legal_actions=legal_actions,
    )
    recent_action_window = _working_memory_recent_action_window(recent_records)
    trained_prior_focus = _compact_trained_prior_focus(
        trained_prior_bucket,
        legal_actions=legal_actions,
    )
    return {
        "active_state": str(state or ""),
        "active_frame_signature": str(frame_signature or ""),
        "active_subgoal": str(subgoal or ""),
        "current_board_context": normalized_board_context,
        "board_context_summary": normalized_board_context,
        "legal_actions": list(legal_actions),
        "recent_actions": recent_action_window,
        "recent_action_window": recent_action_window,
        "recommended_actions": list(recommended_actions[:4]),
        "trained_prior_focus": trained_prior_focus,
        "suppressed_actions": list(suppressed_actions[:6]),
        "recent_failures": recent_failures,
        "open_failure_modes": open_failure_modes,
        "failure_streaks": failure_streaks,
        "semantic_focus_actions": semantic_focus_actions,
        "procedural_focus_actions": procedural_focus_actions,
        "abstract_focus_actions": abstract_focus_actions,
        "basin_exhausted": bool(basin_status.get("is_exhausted", False)),
        "basin_visits": int(basin_status.get("visits", 0) or 0),
    }


def _working_memory_recent_action_window(
    recent_records: list[dict[str, Any]],
    *,
    limit: int = 8,
) -> list[str]:
    recent_action_window = [
        str(record.get("action_name", "") or "")
        for record in recent_records[:limit]
        if str(record.get("action_name", "") or "")
    ]
    return recent_action_window


def _working_memory_open_failure_modes(
    recent_records: list[dict[str, Any]],
    *,
    limit: int = 4,
) -> list[dict[str, Any]]:
    open_failure_modes: list[dict[str, Any]] = []
    seen_failures: set[tuple[str, str]] = set()
    for record in recent_records:
        outcome = str(record.get("outcome", "") or "")
        if outcome == "progress":
            continue
        key = (
            str(record.get("action_name", "") or ""),
            str(record.get("decision_subgoal", "") or ""),
        )
        if key in seen_failures:
            continue
        seen_failures.add(key)
        open_failure_modes.append(
            {
                "action_name": key[0],
                "subgoal": key[1],
                "outcome": outcome,
                "stagnation_after": int(record.get("stagnation_after", 0) or 0),
                "frame_signature": str(record.get("prior_frame_signature", "") or ""),
            }
        )
        if len(open_failure_modes) >= limit:
            break
    return open_failure_modes


def _working_memory_recent_failures(
    recent_records: list[dict[str, Any]],
    *,
    limit: int = 4,
) -> list[dict[str, Any]]:
    recent_failures: list[dict[str, Any]] = []
    for record in recent_records:
        outcome = str(record.get("outcome", "") or "")
        if outcome == "progress":
            continue
        recent_failures.append(
            {
                "action_name": str(record.get("action_name", "") or ""),
                "outcome": outcome,
                "prior_game_state": str(record.get("prior_game_state", "") or ""),
                "prior_frame_signature": str(
                    record.get("prior_frame_signature", "") or ""
                ),
                "decision_subgoal": str(record.get("decision_subgoal", "") or ""),
                "subgoal": str(record.get("decision_subgoal", "") or ""),
                "stagnation_after": int(record.get("stagnation_after", 0) or 0),
                "frame_signature": str(record.get("prior_frame_signature", "") or ""),
            }
        )
        if len(recent_failures) >= limit:
            break
    return recent_failures


def _working_memory_failure_streaks(
    recent_records: list[dict[str, Any]],
    *,
    limit: int = 6,
) -> list[dict[str, Any]]:
    unresolved_counts: dict[str, int] = {}
    closed_actions: set[str] = set()
    for record in recent_records:
        action_name = str(record.get("action_name", "") or "")
        if not action_name or action_name in closed_actions:
            continue
        outcome = str(record.get("outcome", "") or "")
        if outcome == "progress":
            closed_actions.add(action_name)
            continue
        unresolved_counts[action_name] = int(unresolved_counts.get(action_name, 0) or 0) + 1
    ranked = sorted(
        (
            {
                "action_name": action_name,
                "count": int(count or 0),
            }
            for action_name, count in unresolved_counts.items()
            if int(count or 0) > 0
        ),
        key=lambda item: (-int(item["count"]), str(item["action_name"])),
    )
    return ranked[:limit]


def _working_memory_focus_actions(
    memories: list[dict[str, Any]],
    *,
    limit: int,
) -> list[str]:
    focus_actions: list[str] = []
    for memory in memories:
        action_name = str(memory.get("action_name", "") or "")
        if not action_name or action_name in focus_actions:
            continue
        focus_actions.append(action_name)
        if len(focus_actions) >= limit:
            break
    return focus_actions


def _build_episodic_recalls(
    *,
    recent_records: list[dict[str, Any]],
    episode_summaries: list[dict[str, Any]],
    state: str,
    frame_signature: str,
    subgoal: str | None,
    limit: int = 8,
) -> list[dict[str, Any]]:
    recalls: list[dict[str, Any]] = []
    current_subgoal = str(subgoal or "")
    for record in recent_records[:4]:
        score = _episodic_record_score(record, state=state, frame_signature=frame_signature, subgoal=current_subgoal)
        recalls.append(
            {
                "recall_type": "record",
                "action_name": str(record.get("action_name", "") or ""),
                "outcome": str(record.get("outcome", "") or ""),
                "prior_game_state": str(record.get("prior_game_state", "") or ""),
                "prior_frame_signature": str(record.get("prior_frame_signature", "") or ""),
                "decision_subgoal": str(record.get("decision_subgoal", "") or ""),
                "step_index": int(record.get("step_index", 0) or 0),
                "board_context_summary": dict(record.get("board_context_summary", {}) or {}),
                "salience": round(max(0.2, score), 4),
                "confidence": round(max(0.2, score * 0.75), 4),
                "score": round(score, 4),
                "retrieval_provenance": "recent-record",
            }
        )
    for summary in episode_summaries[:4]:
        score = _episodic_summary_score(summary, state=state, frame_signature=frame_signature, subgoal=current_subgoal)
        recalls.append(
            {
                "recall_type": "summary",
                "action_name": "",
                "outcome": "",
                "prior_game_state": str(summary.get("prior_game_state", "") or ""),
                "prior_frame_signature": str(summary.get("prior_frame_signature", "") or ""),
                "decision_subgoal": str(summary.get("decision_subgoal", "") or ""),
                "step_index": int(summary.get("last_step_index", 0) or 0),
                "board_context_summary": dict(summary.get("board_context_summary", {}) or {}),
                "salience": round(max(0.25, score), 4),
                "confidence": round(max(0.25, min(1.0, score * 0.8)), 4),
                "score": round(score, 4),
                "retrieval_provenance": "episode-summary",
                "recent_action_sequence": list(summary.get("recent_action_sequence", []) or []),
                "recent_outcome_sequence": list(summary.get("recent_outcome_sequence", []) or []),
                "total_events": int(summary.get("total_events", 0) or 0),
            }
        )
    recalls.sort(
        key=lambda item: (
            float(item.get("score", 0.0) or 0.0),
            float(item.get("salience", 0.0) or 0.0),
            int(item.get("step_index", 0) or 0),
        ),
        reverse=True,
    )
    return recalls[:limit]


def _build_retrieval_summary(
    *,
    state: str,
    frame_signature: str,
    subgoal: str | None,
    board_context: dict[str, Any] | None,
    legal_actions: list[str],
    recent_records: list[dict[str, Any]],
    episode_summaries: list[dict[str, Any]],
    semantic_memories: list[dict[str, Any]],
    procedural_memories: list[dict[str, Any]],
    abstract_memories: list[dict[str, Any]],
    recommended_actions: list[str],
    trained_prior_bucket: dict[str, int],
    suppressed_actions: list[str],
    unknown_game_context: bool = False,
) -> dict[str, Any]:
    trained_prior_focus = _compact_trained_prior_focus(
        trained_prior_bucket,
        legal_actions=legal_actions,
    )
    abstract_focus_actions = _compact_abstract_focus_actions(
        abstract_memories,
        legal_actions=legal_actions,
    )
    top_recall_sources = [
        source
        for source in [
            "recent-record" if recent_records else "",
            "episode-summary" if episode_summaries else "",
            "semantic-store" if semantic_memories else "",
            "procedural-store" if procedural_memories else "",
            "abstract-store" if abstract_memories else "",
            "abstraction-transfer" if unknown_game_context and abstract_memories else "",
        ]
        if source
    ]
    return {
        "retrieval_version": "hybrid-multistore-v1",
        "state": str(state or ""),
        "frame_signature": str(frame_signature or ""),
        "subgoal": str(subgoal or ""),
        "board_context_summary": _normalized_board_context(board_context),
        "legal_actions": list(legal_actions),
        "recent_record_count": len(recent_records),
        "episode_summary_count": len(episode_summaries),
        "semantic_memory_count": len(semantic_memories),
        "procedural_memory_count": len(procedural_memories),
        "abstract_memory_count": len(abstract_memories),
        "top_recall_sources": top_recall_sources,
        "suggested_actions": list(recommended_actions[:4]),
        "trained_prior_focus": trained_prior_focus,
        "abstract_focus_actions": abstract_focus_actions,
        "suppressed_actions": list(suppressed_actions[:6]),
    }


def _ensure_live_episode_record_columns(connection: sqlite3.Connection) -> None:
    columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(live_episode_records)").fetchall()
    }
    expected = {
        "prior_board_height": "INTEGER NOT NULL DEFAULT 0",
        "prior_board_width": "INTEGER NOT NULL DEFAULT 0",
        "prior_board_active_cells": "INTEGER NOT NULL DEFAULT 0",
        "prior_board_source_attr": "TEXT NOT NULL DEFAULT ''",
        "prior_board_default_probe_json": "TEXT NOT NULL DEFAULT '{}'",
    }
    for column_name, column_def in expected.items():
        if column_name in columns:
            continue
        connection.execute(
            f"ALTER TABLE live_episode_records ADD COLUMN {column_name} {column_def}"
        )


def _ensure_live_episode_summary_columns(connection: sqlite3.Connection) -> None:
    columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(live_episode_summaries)").fetchall()
    }
    expected = {
        "first_step_index": "INTEGER NOT NULL DEFAULT 0",
        "recent_action_sequence_json": "TEXT NOT NULL DEFAULT '[]'",
        "recent_outcome_sequence_json": "TEXT NOT NULL DEFAULT '[]'",
        "decision_sources_json": "TEXT NOT NULL DEFAULT '{}'",
        "max_stagnation_after": "INTEGER NOT NULL DEFAULT 0",
    }
    for column_name, column_def in expected.items():
        if column_name in columns:
            continue
        connection.execute(
            f"ALTER TABLE live_episode_summaries ADD COLUMN {column_name} {column_def}"
        )


def _table_has_rows(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone()
    return row is not None


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        [table],
    ).fetchone()
    return row is not None


def _metadata_value(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute(
        "SELECT value FROM metadata WHERE key = ?",
        [key],
    ).fetchone()
    if row is None:
        return None
    return str(row[0])


def _import_action_scope(
    connection: sqlite3.Connection,
    *,
    table: str,
    scope_fields: tuple[str, ...],
    actions: Any,
) -> None:
    if not isinstance(actions, dict):
        return
    for action_name, stats in sorted(actions.items()):
        if not isinstance(stats, dict):
            continue
        _seed_stats_row(
            connection,
            table=table,
            scope_fields=list(scope_fields),
            key_value=str(action_name),
            stats={
                "progress": int(stats.get("progress", 0) or 0),
                "failed": int(stats.get("failed", 0) or 0),
                "stalled": int(stats.get("stalled", 0) or 0),
                "total": int(stats.get("total", 0) or 0),
                "last_outcome": str(stats.get("last_outcome") or ""),
            },
        )


def _upsert_outcome_row(
    connection: sqlite3.Connection,
    *,
    table: str,
    scope_fields: list[str],
    key_value: str,
    outcome: str,
) -> None:
    field_names = _scope_field_names(table)
    placeholders = ",".join("?" for _ in range(len(field_names) + 5))
    columns = ", ".join([*field_names, "progress", "failed", "stalled", "total", "last_outcome"])
    values = [
        *scope_fields,
        key_value,
        1 if outcome == "progress" else 0,
        1 if outcome == "failed" else 0,
        1 if outcome == "stalled" else 0,
        1,
        outcome,
    ]
    connection.execute(
        f"""
        INSERT INTO {table}({columns})
        VALUES({placeholders})
        ON CONFLICT({", ".join(field_names)})
        DO UPDATE SET
            progress={table}.progress + excluded.progress,
            failed={table}.failed + excluded.failed,
            stalled={table}.stalled + excluded.stalled,
            total={table}.total + excluded.total,
            last_outcome=excluded.last_outcome
        """,
        values,
    )


def _upsert_stats_row(
    connection: sqlite3.Connection,
    *,
    table: str,
    scope_fields: list[str],
    key_value: str,
    stats: dict[str, Any],
) -> None:
    field_names = _scope_field_names(table)
    placeholders = ",".join("?" for _ in range(len(field_names) + 5))
    columns = ", ".join([*field_names, "progress", "failed", "stalled", "total", "last_outcome"])
    values = [
        *scope_fields,
        key_value,
        int(stats.get("progress", 0) or 0),
        int(stats.get("failed", 0) or 0),
        int(stats.get("stalled", 0) or 0),
        int(stats.get("total", 0) or 0),
        str(stats.get("last_outcome") or ""),
    ]
    connection.execute(
        f"""
        INSERT INTO {table}({columns})
        VALUES({placeholders})
        ON CONFLICT({", ".join(field_names)})
        DO UPDATE SET
            progress={table}.progress + excluded.progress,
            failed={table}.failed + excluded.failed,
            stalled={table}.stalled + excluded.stalled,
            total={table}.total + excluded.total,
            last_outcome=excluded.last_outcome
        """,
        values,
    )


def _seed_stats_row(
    connection: sqlite3.Connection,
    *,
    table: str,
    scope_fields: list[str],
    key_value: str,
    stats: dict[str, Any],
) -> None:
    field_names = _scope_field_names(table)
    placeholders = ",".join("?" for _ in range(len(field_names) + 5))
    columns = ", ".join([*field_names, "progress", "failed", "stalled", "total", "last_outcome"])
    values = [
        *scope_fields,
        key_value,
        int(stats.get("progress", 0) or 0),
        int(stats.get("failed", 0) or 0),
        int(stats.get("stalled", 0) or 0),
        int(stats.get("total", 0) or 0),
        str(stats.get("last_outcome") or ""),
    ]
    connection.execute(
        f"""
        INSERT INTO {table}({columns})
        VALUES({placeholders})
        ON CONFLICT({", ".join(field_names)}) DO NOTHING
        """,
        values,
    )


def _scope_field_names(table: str) -> list[str]:
    if table == "live_action_stats":
        return ["game_id", "action_name"]
    if table == "live_state_action_stats":
        return ["game_id", "state", "action_name"]
    if table == "live_frame_action_stats":
        return ["game_id", "frame_signature", "action_name"]
    if table == "live_subgoal_action_stats":
        return ["game_id", "subgoal", "action_name"]
    raise ValueError(f"unknown stats table: {table}")


def _query_action_stats(
    connection: sqlite3.Connection,
    query: str,
    parameters: list[Any],
    last_outcome_query: str | None = None,
    last_outcome_scope: list[Any] | None = None,
) -> dict[str, dict[str, Any]]:
    rows = connection.execute(query, parameters).fetchall()
    stats: dict[str, dict[str, Any]] = {}
    for row in rows:
        action_name = str(row[0])
        progress = int(row[1] or 0)
        failed = int(row[2] or 0)
        stalled = int(row[3] or 0)
        total = int(row[4] or 0)
        last_outcome = ""
        if len(row) > 5 and row[5] is not None and last_outcome_query is None:
            last_outcome = str(row[5] or "")
        elif last_outcome_query is not None and last_outcome_scope is not None:
            last_row = connection.execute(
                last_outcome_query,
                [*last_outcome_scope, action_name],
            ).fetchone()
            if last_row is not None:
                last_outcome = str(last_row[0] or "")
        stats[action_name] = {
            "progress": progress,
            "failed": failed,
            "stalled": stalled,
            "total": total,
            "last_outcome": last_outcome,
        }
    return stats


def _normalized_board_context(board_context: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(board_context, dict):
        return {
            "board_height": 0,
            "board_width": 0,
            "board_active_cells": 0,
            "board_source_attr": None,
            "board_default_probe": {},
        }
    return {
        "board_height": int(board_context.get("board_height", 0) or 0),
        "board_width": int(board_context.get("board_width", 0) or 0),
        "board_active_cells": int(board_context.get("board_active_cells", 0) or 0),
        "board_source_attr": str(board_context.get("board_source_attr") or "") or None,
        "board_default_probe": dict(board_context.get("board_default_probe") or {}),
    }


def _normalized_motif_tags(tags: Any) -> list[str]:
    normalized = sorted(
        {
            str(tag or "").strip()
            for tag in list(tags or [])
            if str(tag or "").strip()
        }
    )
    return normalized


def _deserialize_probe_json(value: Any) -> dict[str, int]:
    try:
        payload = json.loads(value or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    result: dict[str, int] = {}
    for key in ("x", "y"):
        if key not in payload:
            continue
        try:
            result[key] = int(payload[key])
        except (TypeError, ValueError):
            continue
    return result


def _deserialize_json_list(value: Any) -> list[str]:
    try:
        payload = json.loads(value or "[]")
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [str(item or "") for item in payload if str(item or "")]


def _deserialize_json_object(value: Any) -> dict[str, int]:
    try:
        payload = json.loads(value or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    result: dict[str, int] = {}
    for key, raw_value in payload.items():
        name = str(key or "")
        if not name:
            continue
        try:
            result[name] = int(raw_value or 0)
        except (TypeError, ValueError):
            continue
    return result


def _board_family_signature(board_context: dict[str, Any] | None) -> str:
    summary = _normalized_board_context(board_context)
    height = int(summary.get("board_height", 0) or 0)
    width = int(summary.get("board_width", 0) or 0)
    active_cells = int(summary.get("board_active_cells", 0) or 0)
    source_attr = str(summary.get("board_source_attr") or "")
    total_cells = max(height * width, 0)
    density_bucket = "empty"
    if total_cells > 0 and active_cells > 0:
        density = active_cells / total_cells
        if density < 0.15:
            density_bucket = "sparse"
        elif density < 0.60:
            density_bucket = "medium"
        else:
            density_bucket = "dense"
    probe_region = _probe_region(
        width=width,
        height=height,
        probe=dict(summary.get("board_default_probe") or {}),
    )
    return f"{width}x{height}:{density_bucket}:{source_attr or '-'}:{probe_region}"


def _probe_region(
    *,
    width: int,
    height: int,
    probe: dict[str, Any] | None,
) -> str:
    if width <= 0 or height <= 0 or not isinstance(probe, dict):
        return "-"
    try:
        x = int(probe.get("x"))
        y = int(probe.get("y"))
    except (TypeError, ValueError):
        return "-"
    horizontal = "c"
    vertical = "c"
    if x < width / 3:
        horizontal = "l"
    elif x >= (2 * width) / 3:
        horizontal = "r"
    if y < height / 3:
        vertical = "t"
    elif y >= (2 * height) / 3:
        vertical = "b"
    return vertical + horizontal


def _summary_sequence_matches_last_action(
    recent_action_sequence: list[str],
    last_action: str,
) -> bool:
    if not recent_action_sequence or not last_action:
        return False
    return str(recent_action_sequence[-1] or "") == str(last_action or "")


def _refresh_basin_row(
    connection: sqlite3.Connection,
    game_id: str,
    state: str,
    frame_signature: str,
    last_action_name: str | None,
) -> None:
    if not state:
        return
    row = connection.execute(
        """
        SELECT
            COUNT(*),
            COUNT(DISTINCT action_name),
            SUM(CASE WHEN outcome = 'progress' THEN 1 ELSE 0 END),
            SUM(CASE WHEN outcome = 'failed' THEN 1 ELSE 0 END),
            SUM(CASE WHEN outcome = 'stalled' THEN 1 ELSE 0 END)
        FROM live_episode_records
        WHERE game_id = ? AND prior_state = ? AND prior_frame_signature = ?
        """,
        [game_id, state, frame_signature],
    ).fetchone()
    last_action_row = connection.execute(
        """
        SELECT action_name
        FROM live_episode_records
        WHERE game_id = ? AND prior_state = ? AND prior_frame_signature = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        [game_id, state, frame_signature],
    ).fetchone()
    last_progress_row = connection.execute(
        """
        SELECT MAX(step_index)
        FROM live_episode_records
        WHERE game_id = ? AND prior_state = ? AND prior_frame_signature = ? AND outcome = 'progress'
        """,
        [game_id, state, frame_signature],
    ).fetchone()
    actions_attempted = int(row[0] or 0)
    distinct_actions = int(row[1] or 0)
    progress_events = int(row[2] or 0)
    failed_events = int(row[3] or 0)
    stalled_events = int(row[4] or 0)
    visits = progress_events + failed_events + stalled_events
    is_exhausted = 1 if progress_events == 0 and distinct_actions > 0 else 0
    connection.execute(
        """
        INSERT INTO live_basin_stats(
            game_id,
            state,
            frame_signature,
            visits,
            actions_attempted,
            distinct_actions_tried,
            progress_events,
            failed_events,
            stalled_events,
            last_action_name,
            last_progress_step,
            is_exhausted
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(game_id, state, frame_signature)
        DO UPDATE SET
            visits=excluded.visits,
            actions_attempted=excluded.actions_attempted,
            distinct_actions_tried=excluded.distinct_actions_tried,
            progress_events=excluded.progress_events,
            failed_events=excluded.failed_events,
            stalled_events=excluded.stalled_events,
            last_action_name=excluded.last_action_name,
            last_progress_step=excluded.last_progress_step,
            is_exhausted=excluded.is_exhausted
        """,
        [
            game_id,
            state,
            frame_signature,
            visits,
            actions_attempted,
            distinct_actions,
            progress_events,
            failed_events,
            stalled_events,
            last_action_row[0] if last_action_row is not None else last_action_name,
            last_progress_row[0] if last_progress_row is not None else None,
            is_exhausted,
        ],
    )


def _load_basin_row(
    connection: sqlite3.Connection,
    game_id: str,
    state: str,
    frame_signature: str,
):
    return connection.execute(
        """
        SELECT
            visits,
            actions_attempted,
            distinct_actions_tried,
            progress_events,
            failed_events,
            stalled_events,
            last_action_name,
            last_progress_step,
            is_exhausted
        FROM live_basin_stats
        WHERE game_id = ? AND state = ? AND frame_signature = ?
        """,
        [game_id, state, frame_signature],
    ).fetchone()


def _load_basin_action_outcomes(
    connection: sqlite3.Connection,
    *,
    game_id: str,
    state: str,
    frame_signature: str,
) -> dict[str, dict[str, Any]]:
    if frame_signature:
        return _query_action_stats(
            connection,
            """
            SELECT
                action_name,
                SUM(CASE WHEN outcome = 'progress' THEN 1 ELSE 0 END) AS progress,
                SUM(CASE WHEN outcome = 'failed' THEN 1 ELSE 0 END) AS failed,
                SUM(CASE WHEN outcome = 'stalled' THEN 1 ELSE 0 END) AS stalled,
                COUNT(*) AS total,
                MAX(id) AS last_record_id
            FROM live_episode_records
            WHERE game_id = ? AND prior_state = ? AND prior_frame_signature = ?
            GROUP BY action_name
            ORDER BY action_name
            """,
            [game_id, state, frame_signature],
            last_outcome_query="""
                SELECT outcome
                FROM live_episode_records
                WHERE game_id = ? AND prior_state = ? AND prior_frame_signature = ? AND action_name = ?
                ORDER BY id DESC
                LIMIT 1
            """,
            last_outcome_scope=[game_id, state, frame_signature],
        )
    return _query_action_stats(
        connection,
        """
        SELECT action_name, progress, failed, stalled, total, last_outcome
        FROM live_state_action_stats
        WHERE game_id = ? AND state = ?
        ORDER BY action_name
        """,
        [game_id, state],
    )


def _seed_basin_from_aggregate_actions(
    connection: sqlite3.Connection,
    *,
    game_id: str,
    state: str,
    frame_signature: str,
    actions: Any,
) -> None:
    if not state or not isinstance(actions, dict):
        return
    normalized: list[tuple[str, dict[str, Any]]] = []
    for action_name, stats in sorted(actions.items()):
        if not isinstance(stats, dict):
            continue
        normalized.append((str(action_name), dict(stats)))
    if not normalized:
        return
    progress_events = sum(int(stats.get("progress", 0) or 0) for _, stats in normalized)
    failed_events = sum(int(stats.get("failed", 0) or 0) for _, stats in normalized)
    stalled_events = sum(int(stats.get("stalled", 0) or 0) for _, stats in normalized)
    actions_attempted = sum(int(stats.get("total", 0) or 0) for _, stats in normalized)
    distinct_actions = sum(1 for _, stats in normalized if int(stats.get("total", 0) or 0) > 0)
    visits = progress_events + failed_events + stalled_events
    last_action_name = None
    if normalized:
        last_action_name = max(
            normalized,
            key=lambda item: (
                int(item[1].get("total", 0) or 0),
                int(item[1].get("progress", 0) or 0),
                item[0],
            ),
        )[0]
    connection.execute(
        """
        INSERT INTO live_basin_stats(
            game_id,
            state,
            frame_signature,
            visits,
            actions_attempted,
            distinct_actions_tried,
            progress_events,
            failed_events,
            stalled_events,
            last_action_name,
            last_progress_step,
            is_exhausted
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(game_id, state, frame_signature)
        DO UPDATE SET
            visits=excluded.visits,
            actions_attempted=excluded.actions_attempted,
            distinct_actions_tried=excluded.distinct_actions_tried,
            progress_events=excluded.progress_events,
            failed_events=excluded.failed_events,
            stalled_events=excluded.stalled_events,
            last_action_name=excluded.last_action_name,
            last_progress_step=excluded.last_progress_step,
            is_exhausted=excluded.is_exhausted
        """,
        [
            game_id,
            state,
            frame_signature,
            visits,
            actions_attempted,
            distinct_actions,
            progress_events,
            failed_events,
            stalled_events,
            last_action_name,
            None,
            1 if progress_events == 0 and distinct_actions > 0 else 0,
        ],
    )


def _derive_suppressed_actions(
    *,
    legal_actions: list[str],
    basin_action_stats: dict[str, dict[str, Any]],
    state_stats: dict[str, dict[str, Any]],
    subgoal_stats: dict[str, dict[str, Any]],
    basin_status: dict[str, Any],
) -> list[str]:
    suppressed: list[str] = []
    for action_name in legal_actions:
        if action_name == "RESET":
            continue
        basin = basin_action_stats.get(action_name, {})
        if int(basin.get("progress", 0) or 0) > 0:
            continue
        if int(basin.get("failed", 0) or 0) > 0 or int(basin.get("stalled", 0) or 0) > 0:
            suppressed.append(action_name)
            continue
        subgoal = subgoal_stats.get(action_name, {})
        if _has_context_failure_without_progress(subgoal, stalled_threshold=2):
            suppressed.append(action_name)
            continue
        state = state_stats.get(action_name, {})
        if _has_context_failure_without_progress(state, stalled_threshold=2):
            suppressed.append(action_name)
    if _is_basin_exhausted(
        basin_status=basin_status,
        legal_actions=legal_actions,
        suppressed_actions=suppressed,
    ):
        return [action for action in legal_actions if action != "RESET"]
    return suppressed


def _has_context_failure_without_progress(
    stats: dict[str, Any],
    *,
    stalled_threshold: int,
) -> bool:
    if not stats:
        return False
    if int(stats.get("progress", 0) or 0) > 0:
        return False
    if int(stats.get("failed", 0) or 0) > 0:
        return True
    return int(stats.get("stalled", 0) or 0) >= stalled_threshold


def _derive_recommended_actions(
    *,
    legal_actions: list[str],
    action_stats: dict[str, dict[str, Any]],
    state_stats: dict[str, dict[str, Any]],
    basin_action_stats: dict[str, dict[str, Any]],
    frame_stats: dict[str, dict[str, Any]],
    subgoal_stats: dict[str, dict[str, Any]],
    pattern_stats: dict[str, dict[str, Any]],
    sequence_stats: dict[str, dict[str, Any]],
    board_cluster_stats: dict[str, dict[str, Any]],
    board_context_stats: dict[str, dict[str, Any]],
    trained_prior_bucket: dict[str, int],
    abstract_memories: list[dict[str, Any]],
    recent_records: list[dict[str, Any]],
    suppressed_actions: list[str],
    unknown_game_context: bool = False,
) -> list[str]:
    ranked: list[tuple[float, int, int, str]] = []
    suppressed = set(suppressed_actions)
    legal_action_order = {
        str(action_name or "").upper(): index
        for index, action_name in enumerate(legal_actions)
    }
    recent_action = None
    recent_outcomes: dict[str, str] = {}
    board_context_attempted_actions = {
        str(action_name or "").upper()
        for action_name in board_context_stats.keys()
        if str(action_name or "").upper()
    }
    board_context_attempt_totals = {
        str(action_name or "").upper(): int(dict(stats or {}).get("total", 0) or 0)
        for action_name, stats in board_context_stats.items()
    }
    board_context_progress_actions = {
        action_name
        for action_name, stats in board_context_stats.items()
        if int(dict(stats or {}).get("progress", 0) or 0) > 0
    }
    board_context_recovery_mode = (
        len(board_context_attempted_actions) >= 2
        and not board_context_progress_actions
    )
    if recent_records:
        recent_action = str(recent_records[0].get("action_name") or "") or None
        for record in recent_records[:4]:
            action_name = str(record.get("action_name") or "").upper()
            outcome = str(record.get("outcome") or "")
            if action_name and action_name not in recent_outcomes:
                recent_outcomes[action_name] = outcome
    prior_peak = max((int(value or 0) for value in trained_prior_bucket.values()), default=0)
    abstract_by_action = _aggregate_abstract_action_guidance(abstract_memories)
    for action_name in legal_actions:
        if action_name == "RESET" or action_name in suppressed:
            continue
        basin = basin_action_stats.get(action_name, {})
        frame = frame_stats.get(action_name, {})
        state = state_stats.get(action_name, {})
        subgoal = subgoal_stats.get(action_name, {})
        pattern = pattern_stats.get(action_name, {})
        sequence = sequence_stats.get(action_name, {})
        board_cluster = board_cluster_stats.get(action_name, {})
        board_context = board_context_stats.get(action_name, {})
        game = action_stats.get(action_name, {})
        abstract = abstract_by_action.get(action_name, {})
        score = 0.0
        score += 5.0 * int(basin.get("progress", 0) or 0)
        score += 4.5 * int(sequence.get("progress", 0) or 0)
        score += 4.0 * int(board_context.get("progress", 0) or 0)
        score += 3.75 * int(pattern.get("progress", 0) or 0)
        score += 3.5 * int(state.get("progress", 0) or 0)
        score += 3.0 * int(board_cluster.get("progress", 0) or 0)
        score += 2.5 * int(subgoal.get("progress", 0) or 0)
        score += 1.5 * int(frame.get("progress", 0) or 0)
        score += 1.0 * int(game.get("progress", 0) or 0)
        score += _abstract_action_score(
            abstract,
            unknown_game_context=unknown_game_context,
        )
        score -= 2.5 * int(basin.get("failed", 0) or 0)
        score -= 1.60 * int(sequence.get("failed", 0) or 0)
        score -= 0.85 * int(sequence.get("stalled", 0) or 0)
        score -= 1.25 * int(basin.get("stalled", 0) or 0)
        score -= 1.25 * int(board_context.get("failed", 0) or 0)
        score -= 0.60 * int(board_context.get("stalled", 0) or 0)
        score -= 1.10 * int(pattern.get("failed", 0) or 0)
        score -= 0.50 * int(pattern.get("stalled", 0) or 0)
        score -= 0.90 * int(board_cluster.get("failed", 0) or 0)
        score -= 0.45 * int(board_cluster.get("stalled", 0) or 0)
        score -= 0.75 * int(frame.get("failed", 0) or 0)
        score -= 0.35 * int(frame.get("stalled", 0) or 0)
        score -= 0.75 * int(state.get("failed", 0) or 0)
        score -= 0.35 * int(state.get("stalled", 0) or 0)
        score -= 0.75 * int(subgoal.get("failed", 0) or 0)
        score -= 0.35 * int(subgoal.get("stalled", 0) or 0)
        if prior_peak > 0:
            score += 1.75 * (int(trained_prior_bucket.get(action_name, 0) or 0) / prior_peak)
        if recent_outcomes.get(action_name) in {"failed", "stalled"}:
            score -= 0.75
        if action_name == recent_action:
            score -= 0.35
        if board_context_recovery_mode:
            if action_name not in board_context_attempted_actions:
                score += 1.25
            elif action_name not in board_context_progress_actions:
                score -= 0.20
        ranked.append(
            (
                score,
                int(board_context_attempt_totals.get(action_name, 0) or 0),
                int(legal_action_order.get(action_name, len(legal_actions))),
                action_name,
            )
        )
    ranked.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    positive = [action_name for score, _, _, action_name in ranked if score > 0]
    if positive:
        return positive[:3]
    return [action_name for _, _, _, action_name in ranked[:3]]


def _compact_trained_prior_focus(
    trained_prior_bucket: dict[str, int],
    *,
    legal_actions: list[str],
    limit: int = 4,
) -> list[dict[str, int | str]]:
    legal = {str(action_name or "") for action_name in list(legal_actions or [])}
    ranked = sorted(
        (
            (str(action_name or ""), int(count or 0))
            for action_name, count in dict(trained_prior_bucket or {}).items()
            if str(action_name or "") in legal and int(count or 0) > 0
        ),
        key=lambda item: (-item[1], item[0]),
    )
    return [
        {"action_name": action_name, "count": count}
        for action_name, count in ranked[:limit]
    ]


def _compact_abstract_focus_actions(
    abstract_memories: list[dict[str, Any]],
    *,
    legal_actions: list[str],
    limit: int = 4,
) -> list[dict[str, Any]]:
    legal = {str(action_name or "") for action_name in list(legal_actions or [])}
    ranked: list[tuple[float, int, str, dict[str, Any]]] = []
    for memory in list(abstract_memories or []):
        action_name = str(dict(memory or {}).get("action_name", "") or "")
        if not action_name or (legal and action_name not in legal):
            continue
        ranked.append(
            (
                float(dict(memory or {}).get("score", 0.0) or 0.0),
                int(dict(memory or {}).get("source_game_count", 0) or 0),
                action_name,
                {
                    "action_name": action_name,
                    "motif_overlap": int(dict(memory or {}).get("motif_overlap", 0) or 0),
                    "source_game_count": int(
                        dict(memory or {}).get("source_game_count", 0) or 0
                    ),
                },
            )
        )
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [payload for _, _, _, payload in ranked[:limit]]


def _aggregate_abstract_action_guidance(
    abstract_memories: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    for memory in list(abstract_memories or []):
        action_name = str(dict(memory or {}).get("action_name", "") or "")
        if not action_name:
            continue
        entry = aggregated.setdefault(
            action_name,
            {
                "progress": 0,
                "failed": 0,
                "stalled": 0,
                "total": 0,
                "motif_overlap": 0,
                "source_game_count": 0,
                "score": 0.0,
            },
        )
        entry["progress"] += int(dict(memory or {}).get("progress", 0) or 0)
        entry["failed"] += int(dict(memory or {}).get("failed", 0) or 0)
        entry["stalled"] += int(dict(memory or {}).get("stalled", 0) or 0)
        entry["total"] += int(dict(memory or {}).get("total", 0) or 0)
        entry["motif_overlap"] = max(
            int(entry.get("motif_overlap", 0) or 0),
            int(dict(memory or {}).get("motif_overlap", 0) or 0),
        )
        entry["source_game_count"] = max(
            int(entry.get("source_game_count", 0) or 0),
            int(dict(memory or {}).get("source_game_count", 0) or 0),
        )
        entry["score"] = max(
            float(entry.get("score", 0.0) or 0.0),
            float(dict(memory or {}).get("score", 0.0) or 0.0),
        )
    return aggregated


def _abstract_action_score(
    abstract_stats: dict[str, Any],
    *,
    unknown_game_context: bool,
) -> float:
    if not abstract_stats:
        return 0.0
    progress = int(abstract_stats.get("progress", 0) or 0)
    failed = int(abstract_stats.get("failed", 0) or 0)
    stalled = int(abstract_stats.get("stalled", 0) or 0)
    motif_overlap = int(abstract_stats.get("motif_overlap", 0) or 0)
    source_game_count = int(abstract_stats.get("source_game_count", 0) or 0)
    base = (
        0.12 * motif_overlap
        + 0.10 * min(3, source_game_count)
        + 0.45 * _success_bias(
            progress=progress,
            failed=failed,
            stalled=stalled,
            total=int(abstract_stats.get("total", 0) or 0),
        )
    )
    if unknown_game_context:
        base += 1.35 + (0.45 * motif_overlap)
    return base


def _derive_board_context_action_stats(
    *,
    board_context: dict[str, Any] | None,
    recent_records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    current_board_context = _normalized_board_context(board_context)
    if not recent_records:
        return {}
    stats: dict[str, dict[str, Any]] = {}
    for record in recent_records[:DEFAULT_BOARD_CONTEXT_HISTORY_LIMIT]:
        action_name = str(record.get("action_name") or "").upper()
        if not action_name:
            continue
        if (
            board_context_match_score(
                current_board_context,
                dict(record.get("board_context_summary", {}) or {}),
            )
            < 2
        ):
            continue
        entry = stats.setdefault(
            action_name,
            {"progress": 0, "failed": 0, "stalled": 0, "total": 0, "last_outcome": None},
        )
        outcome = str(record.get("outcome") or "")
        if outcome == "progress":
            entry["progress"] = int(entry.get("progress", 0) or 0) + 1
        elif outcome == "failed":
            entry["failed"] = int(entry.get("failed", 0) or 0) + 1
        elif outcome == "stalled":
            entry["stalled"] = int(entry.get("stalled", 0) or 0) + 1
        entry["total"] = int(entry.get("total", 0) or 0) + 1
        entry["last_outcome"] = outcome
    return stats


def _upsert_episode_summary_row(
    connection: sqlite3.Connection,
    record: EpisodeRecord,
) -> None:
    horizon = _episode_summary_horizon(connection, record)
    board_default_probe_json = json.dumps(
        dict(record.prior_board_default_probe or {}),
        sort_keys=True,
    )
    columns = [
        "game_id",
        "prior_state",
        "prior_frame_signature",
        "subgoal",
        "board_height",
        "board_width",
        "board_active_cells",
        "board_source_attr",
        "board_default_probe_json",
        "action_name",
        "progress",
        "failed",
        "stalled",
        "total",
        "last_outcome",
        "first_step_index",
        "last_step_index",
        "recent_action_sequence_json",
        "recent_outcome_sequence_json",
        "decision_sources_json",
        "max_stagnation_after",
    ]
    values = [
        record.game_id,
        str(record.prior_game_state or ""),
        str(record.prior_frame_signature or ""),
        str(record.decision_subgoal or ""),
        int(record.prior_board_height),
        int(record.prior_board_width),
        int(record.prior_board_active_cells),
        str(record.prior_board_source_attr or ""),
        board_default_probe_json,
        record.action_name,
        1 if record.outcome == "progress" else 0,
        1 if record.outcome == "failed" else 0,
        1 if record.outcome == "stalled" else 0,
        1,
        str(record.outcome or ""),
        int(horizon["first_step_index"]),
        int(record.step_index),
        json.dumps(list(horizon["recent_action_sequence"]), sort_keys=False),
        json.dumps(list(horizon["recent_outcome_sequence"]), sort_keys=False),
        json.dumps(dict(horizon["decision_sources"]), sort_keys=True),
        int(horizon["max_stagnation_after"]),
    ]
    update_assignments = [
        "progress = progress + excluded.progress",
        "failed = failed + excluded.failed",
        "stalled = stalled + excluded.stalled",
        "total = total + excluded.total",
        "last_outcome = excluded.last_outcome",
        "first_step_index = CASE"
        " WHEN first_step_index = 0 THEN excluded.first_step_index"
        " WHEN excluded.first_step_index > 0 AND excluded.first_step_index < first_step_index"
        " THEN excluded.first_step_index"
        " ELSE first_step_index END",
        "last_step_index = CASE"
        " WHEN excluded.last_step_index > last_step_index"
        " THEN excluded.last_step_index"
        " ELSE last_step_index END",
        "recent_action_sequence_json = excluded.recent_action_sequence_json",
        "recent_outcome_sequence_json = excluded.recent_outcome_sequence_json",
        "decision_sources_json = excluded.decision_sources_json",
        "max_stagnation_after = CASE"
        " WHEN excluded.max_stagnation_after > max_stagnation_after"
        " THEN excluded.max_stagnation_after"
        " ELSE max_stagnation_after END",
    ]
    connection.execute(
        f"""
        INSERT INTO live_episode_summaries({", ".join(columns)})
        VALUES({", ".join("?" for _ in columns)})
        ON CONFLICT(
            game_id,
            prior_state,
            prior_frame_signature,
            subgoal,
            board_height,
            board_width,
            board_active_cells,
            board_source_attr,
            board_default_probe_json,
            action_name
        ) DO UPDATE SET {", ".join(update_assignments)}
        """,
        values,
    )


def _episode_summary_horizon(
    connection: sqlite3.Connection,
    record: EpisodeRecord,
    *,
    sequence_limit: int = 6,
) -> dict[str, Any]:
    scope = [
        record.game_id,
        str(record.prior_game_state or ""),
        str(record.prior_frame_signature or ""),
        str(record.decision_subgoal or ""),
        int(record.prior_board_height),
        int(record.prior_board_width),
        int(record.prior_board_active_cells),
        str(record.prior_board_source_attr or ""),
        json.dumps(dict(record.prior_board_default_probe or {}), sort_keys=True),
    ]
    row = connection.execute(
        """
        SELECT MIN(step_index), MAX(stagnation_after)
        FROM live_episode_records
        WHERE game_id = ? AND prior_state = ? AND prior_frame_signature = ? AND subgoal = ?
          AND prior_board_height = ? AND prior_board_width = ? AND prior_board_active_cells = ?
          AND prior_board_source_attr = ? AND prior_board_default_probe_json = ?
        """,
        scope,
    ).fetchone()
    recent_rows = connection.execute(
        f"""
        SELECT action_name, outcome, decision_source
        FROM live_episode_records
        WHERE game_id = ? AND prior_state = ? AND prior_frame_signature = ? AND subgoal = ?
          AND prior_board_height = ? AND prior_board_width = ? AND prior_board_active_cells = ?
          AND prior_board_source_attr = ? AND prior_board_default_probe_json = ?
        ORDER BY id DESC
        LIMIT {int(sequence_limit)}
        """,
        scope,
    ).fetchall()
    recent_rows = list(reversed(recent_rows))
    recent_actions: list[str] = []
    recent_outcomes: list[str] = []
    decision_sources: dict[str, int] = {}
    for action_name, outcome, decision_source in recent_rows:
        action_text = str(action_name or "")
        outcome_text = str(outcome or "")
        source_text = str(decision_source or "")
        if action_text:
            recent_actions.append(action_text)
        if outcome_text:
            recent_outcomes.append(outcome_text)
        if source_text:
            decision_sources[source_text] = int(decision_sources.get(source_text, 0) or 0) + 1
    return {
        "first_step_index": int((row[0] if row is not None else 0) or 0),
        "max_stagnation_after": int((row[1] if row is not None else 0) or 0),
        "recent_action_sequence": recent_actions,
        "recent_outcome_sequence": recent_outcomes,
        "decision_sources": decision_sources,
    }


def _upsert_pattern_stats_row(
    connection: sqlite3.Connection,
    record: EpisodeRecord,
) -> None:
    _upsert_hierarchical_stats_row(
        connection,
        table="live_pattern_stats",
        key_fields=[
            record.game_id,
            str(record.prior_game_state or ""),
            str(record.decision_subgoal or ""),
            int(record.prior_board_height),
            int(record.prior_board_width),
            int(record.prior_board_active_cells),
            str(record.prior_board_source_attr or ""),
        ],
        action_name=record.action_name,
        outcome=record.outcome,
    )


def _upsert_sequence_stats_row(
    connection: sqlite3.Connection,
    record: EpisodeRecord,
) -> None:
    previous_action_name = _lookup_previous_action_name(
        connection,
        game_id=record.game_id,
        step_index=int(record.step_index),
    )
    if not previous_action_name:
        return
    _upsert_hierarchical_stats_row(
        connection,
        table="live_sequence_stats",
        key_fields=[
            record.game_id,
            str(record.prior_game_state or ""),
            str(record.decision_subgoal or ""),
            str(previous_action_name),
        ],
        action_name=record.action_name,
        outcome=record.outcome,
    )


def _upsert_board_cluster_stats_row(
    connection: sqlite3.Connection,
    record: EpisodeRecord,
) -> None:
    _upsert_hierarchical_stats_row(
        connection,
        table="live_board_cluster_stats",
        key_fields=[
            record.game_id,
            int(record.prior_board_height),
            int(record.prior_board_width),
            int(record.prior_board_active_cells),
            str(record.prior_board_source_attr or ""),
        ],
        action_name=record.action_name,
        outcome=record.outcome,
    )


def _upsert_semantic_memory_row(
    connection: sqlite3.Connection,
    record: EpisodeRecord,
) -> None:
    board_context = _record_board_context_summary(record)
    lesson_text = _semantic_lesson_text(record)
    salience = _memory_salience(record)
    confidence = _memory_confidence(record)
    connection.execute(
        """
        INSERT INTO live_semantic_memory(
            game_id,
            state,
            subgoal,
            board_height,
            board_width,
            board_active_cells,
            board_source_attr,
            action_name,
            lesson_text,
            salience,
            confidence,
            progress,
            failed,
            stalled,
            total,
            last_outcome,
            last_step_index,
            access_count,
            last_access_step
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)
        ON CONFLICT(
            game_id,
            state,
            subgoal,
            board_height,
            board_width,
            board_active_cells,
            board_source_attr,
            action_name
        ) DO UPDATE SET
            lesson_text = excluded.lesson_text,
            salience = live_semantic_memory.salience + excluded.salience,
            confidence = CASE
                WHEN excluded.confidence > live_semantic_memory.confidence
                THEN excluded.confidence
                ELSE live_semantic_memory.confidence
            END,
            progress = live_semantic_memory.progress + excluded.progress,
            failed = live_semantic_memory.failed + excluded.failed,
            stalled = live_semantic_memory.stalled + excluded.stalled,
            total = live_semantic_memory.total + excluded.total,
            last_outcome = excluded.last_outcome,
            last_step_index = CASE
                WHEN excluded.last_step_index > live_semantic_memory.last_step_index
                THEN excluded.last_step_index
                ELSE live_semantic_memory.last_step_index
            END
        """,
        [
            record.game_id,
            str(record.prior_game_state or ""),
            str(record.decision_subgoal or ""),
            int(board_context["board_height"]),
            int(board_context["board_width"]),
            int(board_context["board_active_cells"]),
            str(board_context.get("board_source_attr") or ""),
            str(record.action_name or ""),
            lesson_text,
            float(salience),
            float(confidence),
            1 if record.outcome == "progress" else 0,
            1 if record.outcome == "failed" else 0,
            1 if record.outcome == "stalled" else 0,
            1,
            str(record.outcome or ""),
            int(record.step_index),
        ],
    )


def _upsert_procedural_memory_row(
    connection: sqlite3.Connection,
    record: EpisodeRecord,
) -> None:
    previous_action_name = _lookup_previous_action_name(
        connection,
        game_id=record.game_id,
        step_index=int(record.step_index),
    )
    strategy_tag = _procedural_strategy_tag(record)
    salience = _memory_salience(record)
    confidence = _memory_confidence(record)
    connection.execute(
        """
        INSERT INTO live_procedural_memory(
            game_id,
            state,
            subgoal,
            previous_action_name,
            action_name,
            strategy_tag,
            salience,
            confidence,
            progress,
            failed,
            stalled,
            total,
            last_outcome,
            last_step_index,
            access_count,
            last_access_step
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)
        ON CONFLICT(
            game_id,
            state,
            subgoal,
            previous_action_name,
            action_name
        ) DO UPDATE SET
            strategy_tag = excluded.strategy_tag,
            salience = live_procedural_memory.salience + excluded.salience,
            confidence = CASE
                WHEN excluded.confidence > live_procedural_memory.confidence
                THEN excluded.confidence
                ELSE live_procedural_memory.confidence
            END,
            progress = live_procedural_memory.progress + excluded.progress,
            failed = live_procedural_memory.failed + excluded.failed,
            stalled = live_procedural_memory.stalled + excluded.stalled,
            total = live_procedural_memory.total + excluded.total,
            last_outcome = excluded.last_outcome,
            last_step_index = CASE
                WHEN excluded.last_step_index > live_procedural_memory.last_step_index
                THEN excluded.last_step_index
                ELSE live_procedural_memory.last_step_index
            END
        """,
        [
            record.game_id,
            str(record.prior_game_state or ""),
            str(record.decision_subgoal or ""),
            str(previous_action_name or ""),
            str(record.action_name or ""),
            strategy_tag,
            float(salience),
            float(confidence),
            1 if record.outcome == "progress" else 0,
            1 if record.outcome == "failed" else 0,
            1 if record.outcome == "stalled" else 0,
            1,
            str(record.outcome or ""),
            int(record.step_index),
        ],
    )


def _upsert_abstract_memory_row(
    connection: sqlite3.Connection,
    row: dict[str, Any],
) -> None:
    motif_tags = _normalized_motif_tags(row.get("motif_tags", []))
    source_games = sorted(
        {
            str(game_id or "")
            for game_id in list(row.get("source_games", []) or [])
            if str(game_id or "")
        }
    )
    key_values = [
        str(row.get("subgoal", "") or ""),
        str(row.get("state", "") or ""),
        int(row.get("board_height", 0) or 0),
        int(row.get("board_width", 0) or 0),
        int(row.get("board_active_cells", 0) or 0),
        str(row.get("board_source_attr", "") or ""),
        json.dumps(motif_tags, sort_keys=True),
        str(row.get("action_name", "") or ""),
    ]
    existing = connection.execute(
        """
        SELECT
            source_games_json,
            source_game_count,
            access_count,
            last_access_step
        FROM abstract_memories
        WHERE subgoal = ? AND state = ? AND board_height = ? AND board_width = ?
          AND board_active_cells = ? AND board_source_attr = ? AND motif_tags_json = ?
          AND action_name = ?
        """,
        key_values,
    ).fetchone()
    existing_games = set()
    access_count = 0
    last_access_step = 0
    if existing is not None:
        existing_games = {
            str(game_id or "")
            for game_id in list(json.loads(str(existing[0] or "[]")) or [])
            if str(game_id or "")
        }
        access_count = int(existing[2] or 0)
        last_access_step = int(existing[3] or 0)
    merged_source_games = sorted(existing_games.union(source_games))
    connection.execute(
        """
        INSERT INTO abstract_memories(
            subgoal,
            state,
            board_height,
            board_width,
            board_active_cells,
            board_source_attr,
            motif_tags_json,
            action_name,
            lead_in_actions_json,
            lesson_text,
            salience,
            confidence,
            progress,
            failed,
            stalled,
            total,
            last_outcome,
            last_step_index,
            access_count,
            last_access_step,
            source_game_count,
            source_games_json
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(
            subgoal,
            state,
            board_height,
            board_width,
            board_active_cells,
            board_source_attr,
            motif_tags_json,
            action_name
        ) DO UPDATE SET
            lead_in_actions_json = excluded.lead_in_actions_json,
            lesson_text = excluded.lesson_text,
            salience = abstract_memories.salience + excluded.salience,
            confidence = CASE
                WHEN excluded.confidence > abstract_memories.confidence
                THEN excluded.confidence
                ELSE abstract_memories.confidence
            END,
            progress = abstract_memories.progress + excluded.progress,
            failed = abstract_memories.failed + excluded.failed,
            stalled = abstract_memories.stalled + excluded.stalled,
            total = abstract_memories.total + excluded.total,
            last_outcome = excluded.last_outcome,
            last_step_index = CASE
                WHEN excluded.last_step_index > abstract_memories.last_step_index
                THEN excluded.last_step_index
                ELSE abstract_memories.last_step_index
            END,
            source_game_count = excluded.source_game_count,
            source_games_json = excluded.source_games_json,
            access_count = excluded.access_count,
            last_access_step = excluded.last_access_step
        """,
        [
            *key_values,
            json.dumps(list(row.get("lead_in_actions", []) or [])[:3]),
            str(row.get("lesson_text", "") or ""),
            float(row.get("salience", 0.0) or 0.0),
            float(row.get("confidence", 0.0) or 0.0),
            int(row.get("progress", 0) or 0),
            int(row.get("failed", 0) or 0),
            int(row.get("stalled", 0) or 0),
            int(row.get("total", 0) or 0),
            str(row.get("last_outcome", "") or ""),
            int(row.get("last_step_index", 0) or 0),
            access_count,
            last_access_step,
            len(merged_source_games),
            json.dumps(merged_source_games, sort_keys=True),
        ],
    )


def _upsert_hierarchical_stats_row(
    connection: sqlite3.Connection,
    *,
    table: str,
    key_fields: list[Any],
    action_name: str,
    outcome: str,
) -> None:
    columns = [*(_hierarchical_scope_field_names(table)), "action_name", "progress", "failed", "stalled", "total", "last_outcome"]
    values = [
        *key_fields,
        action_name,
        1 if outcome == "progress" else 0,
        1 if outcome == "failed" else 0,
        1 if outcome == "stalled" else 0,
        1,
        str(outcome or ""),
    ]
    updates = [
        "progress = progress + excluded.progress",
        "failed = failed + excluded.failed",
        "stalled = stalled + excluded.stalled",
        "total = total + excluded.total",
        "last_outcome = excluded.last_outcome",
    ]
    connection.execute(
        f"""
        INSERT INTO {table}({", ".join(columns)})
        VALUES({", ".join("?" for _ in columns)})
        ON CONFLICT({", ".join(columns[:len(key_fields) + 1])})
        DO UPDATE SET {", ".join(updates)}
        """,
        values,
    )


def _hierarchical_scope_field_names(table: str) -> list[str]:
    if table == "live_pattern_stats":
        return [
            "game_id",
            "state",
            "subgoal",
            "board_height",
            "board_width",
            "board_active_cells",
            "board_source_attr",
        ]
    if table == "live_sequence_stats":
        return [
            "game_id",
            "state",
            "subgoal",
            "previous_action_name",
        ]
    if table == "live_board_cluster_stats":
        return [
            "game_id",
            "board_height",
            "board_width",
            "board_active_cells",
            "board_source_attr",
        ]
    raise ValueError(f"unknown hierarchical stats table: {table}")


def _lookup_previous_action_name(
    connection: sqlite3.Connection,
    *,
    game_id: str,
    step_index: int,
) -> str | None:
    row = connection.execute(
        """
        SELECT action_name
        FROM live_episode_records
        WHERE game_id = ? AND step_index < ?
        ORDER BY step_index DESC, id DESC
        LIMIT 1
        """,
        [game_id, int(step_index)],
    ).fetchone()
    if row is None:
        return None
    return str(row[0] or "") or None


def _is_basin_exhausted(
    *,
    basin_status: dict[str, Any],
    legal_actions: list[str],
    suppressed_actions: list[str],
) -> bool:
    candidate_actions = [action for action in legal_actions if action != "RESET"]
    if not candidate_actions:
        return False
    if int(basin_status.get("progress_events", 0) or 0) > 0:
        return False
    return all(action in set(suppressed_actions) for action in candidate_actions)


def _record_board_context_summary(record: EpisodeRecord) -> dict[str, Any]:
    return {
        "board_height": int(record.prior_board_height or 0),
        "board_width": int(record.prior_board_width or 0),
        "board_active_cells": int(record.prior_board_active_cells or 0),
        "board_source_attr": str(record.prior_board_source_attr or "") or None,
        "board_default_probe": dict(record.prior_board_default_probe or {}),
    }


def _memory_salience(record: EpisodeRecord) -> float:
    base = 0.5
    if record.outcome == "progress":
        base += 1.4 + min(2, max(0, int(record.progress_delta or 0))) * 0.35
    elif record.outcome == "failed":
        base += 0.85
    else:
        base += 0.65
    if int(record.stagnation_before or 0) >= 2 and record.outcome == "progress":
        base += 0.5
    if str(record.decision_subgoal or "") == "break-stall":
        base += 0.2
    return round(base, 4)


def _memory_confidence(record: EpisodeRecord) -> float:
    if record.outcome == "progress":
        confidence = 0.85
    elif record.outcome == "failed":
        confidence = 0.55
    else:
        confidence = 0.45
    if int(record.progress_delta or 0) > 0:
        confidence += 0.10
    return round(min(0.99, confidence), 4)


def _semantic_lesson_text(record: EpisodeRecord) -> str:
    board = _record_board_context_summary(record)
    board_shape = f"{int(board['board_width'])}x{int(board['board_height'])}"
    active_cells = int(board["board_active_cells"] or 0)
    subgoal = str(record.decision_subgoal or "default-policy")
    if record.outcome == "progress":
        return (
            f"Prefer {record.action_name} in {record.prior_game_state} during {subgoal} "
            f"on board {board_shape} with {active_cells} active cells."
        )
    return (
        f"Be cautious with {record.action_name} in {record.prior_game_state} during {subgoal} "
        f"on board {board_shape} with {active_cells} active cells."
    )


def _procedural_strategy_tag(record: EpisodeRecord) -> str:
    if record.outcome == "progress" and int(record.stagnation_before or 0) >= 2:
        return "stall-recovery"
    if record.outcome == "progress":
        return "progress-chain"
    if record.outcome == "failed":
        return "terminal-risk"
    return "exploration-chain"


def _success_bias(
    *,
    progress: int,
    failed: int,
    stalled: int,
    total: int,
) -> float:
    if total <= 0:
        return 0.0
    return round(((progress * 1.0) - (failed * 0.55) - (stalled * 0.25)) / max(1, total), 4)


def _is_unknown_game_context(
    *,
    trained_prior_bucket: dict[str, int],
    recent_records: list[dict[str, Any]],
    state_stats: dict[str, dict[str, Any]],
    frame_stats: dict[str, dict[str, Any]],
    subgoal_stats: dict[str, dict[str, Any]],
) -> bool:
    if any(int(count or 0) > 0 for count in dict(trained_prior_bucket or {}).values()):
        return False
    if recent_records:
        return False
    for stats_group in (state_stats, frame_stats, subgoal_stats):
        for stats in dict(stats_group or {}).values():
            if int(dict(stats or {}).get("total", 0) or 0) > 0:
                return False
    return True


def _episodic_record_score(
    record: dict[str, Any],
    *,
    state: str,
    frame_signature: str,
    subgoal: str,
) -> float:
    score = 0.2
    if str(record.get("prior_game_state", "") or "") == str(state or ""):
        score += 0.4
    if frame_signature and str(record.get("prior_frame_signature", "") or "") == frame_signature:
        score += 0.5
    if subgoal and str(record.get("decision_subgoal", "") or "") == subgoal:
        score += 0.35
    outcome = str(record.get("outcome", "") or "")
    if outcome == "progress":
        score += 0.45
    elif outcome == "failed":
        score += 0.15
    else:
        score += 0.10
    score += min(0.2, int(record.get("stagnation_after", 0) or 0) * 0.03)
    return score


def _episodic_summary_score(
    summary: dict[str, Any],
    *,
    state: str,
    frame_signature: str,
    subgoal: str,
) -> float:
    score = 0.25
    if str(summary.get("prior_game_state", "") or "") == str(state or ""):
        score += 0.35
    if frame_signature and str(summary.get("prior_frame_signature", "") or "") == frame_signature:
        score += 0.45
    if subgoal and str(summary.get("decision_subgoal", "") or "") == subgoal:
        score += 0.30
    score += min(0.3, int(summary.get("total_events", 0) or 0) * 0.05)
    score += min(0.2, int(summary.get("max_stagnation_after", 0) or 0) * 0.03)
    return score


def _merge_action_stats_dicts(
    primary: dict[str, Any],
    secondary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for source in (primary, secondary):
        for action_name, stats in dict(source or {}).items():
            entry = merged.setdefault(
                str(action_name),
                {"progress": 0, "failed": 0, "stalled": 0, "total": 0, "last_outcome": None},
            )
            values = dict(stats or {})
            entry["progress"] = int(entry.get("progress", 0) or 0) + int(
                values.get("progress", 0) or 0
            )
            entry["failed"] = int(entry.get("failed", 0) or 0) + int(
                values.get("failed", 0) or 0
            )
            entry["stalled"] = int(entry.get("stalled", 0) or 0) + int(
                values.get("stalled", 0) or 0
            )
            entry["total"] = int(entry.get("total", 0) or 0) + int(
                values.get("total", 0) or 0
            )
            if values.get("last_outcome"):
                entry["last_outcome"] = str(values.get("last_outcome") or "")
    return merged


def _merge_recent_records(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for source in (primary, secondary):
        for row in list(source or []):
            normalized = dict(row or {})
            key = (
                str(normalized.get("game_id", "") or ""),
                str(normalized.get("step_index", 0) or 0),
                str(normalized.get("action_name", "") or ""),
                str(normalized.get("prior_frame_signature", "") or ""),
                str(normalized.get("decision_subgoal", "") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(normalized)
            if len(merged) >= limit:
                return merged
    return merged


def _merge_ranked_memory_items(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    merged_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    for source_index, source in enumerate((primary, secondary)):
        source_name = "hot" if source_index == 0 else "archive"
        source_bias = 0.08 if source_index == 0 else 0.0
        for item in list(source or []):
            normalized = dict(item or {})
            board_context_key = json.dumps(
                dict(
                    normalized.get("board_context_summary", {})
                    or normalized.get("board_context", {})
                    or {}
                ),
                sort_keys=True,
            )
            key = (
                str(normalized.get("memory_type", "") or normalized.get("recall_type", "")),
                str(normalized.get("state", "") or normalized.get("prior_game_state", "")),
                str(normalized.get("subgoal", "") or normalized.get("decision_subgoal", "")),
                str(normalized.get("action_name", "") or ""),
                str(normalized.get("previous_action_name", "") or ""),
                str(normalized.get("prior_frame_signature", "") or ""),
                board_context_key,
            )
            normalized["retrieval_provenance"] = str(
                normalized.get("retrieval_provenance") or source_name
            )
            raw_rank = list(normalized.get("retrieval_rank", []) or [])
            if raw_rank:
                retrieval_rank = [float(value or 0.0) for value in raw_rank]
            else:
                retrieval_rank = [
                    float(normalized.get("score", 0.0) or 0.0),
                    float(normalized.get("salience", 0.0) or 0.0),
                    float(normalized.get("confidence", 0.0) or 0.0),
                    min(
                        1.0,
                        float(int(normalized.get("step_index", normalized.get("last_step_index", 0)) or 0))
                        / 1000.0,
                    ),
                ]
            retrieval_rank[0] += source_bias
            normalized["retrieval_rank"] = list(retrieval_rank)
            normalized["score"] = round(sum(retrieval_rank[:-1]), 4)
            current = merged_by_key.get(key)
            current_rank = tuple(float(value or 0.0) for value in list(current.get("retrieval_rank", []) or [])) if current is not None else ()
            new_rank = tuple(float(value or 0.0) for value in retrieval_rank)
            if current is None or new_rank > current_rank:
                merged_by_key[key] = normalized
    ranked = sorted(
        merged_by_key.values(),
        key=lambda item: (
            tuple(float(value or 0.0) for value in list(item.get("retrieval_rank", []) or [])),
            float(item.get("salience", 0.0) or 0.0),
            float(item.get("confidence", 0.0) or 0.0),
            int(item.get("step_index", item.get("last_step_index", 0)) or 0),
        ),
        reverse=True,
    )
    return ranked[:limit]


def _merge_episode_summaries(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    merged_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    for source in (primary, secondary):
        for summary in list(source or []):
            normalized = dict(summary or {})
            key = (
                str(normalized.get("game_id", "") or ""),
                str(normalized.get("prior_game_state", "") or ""),
                str(normalized.get("prior_frame_signature", "") or ""),
                str(normalized.get("decision_subgoal", "") or ""),
                json.dumps(dict(normalized.get("board_context_summary", {}) or {}), sort_keys=True),
            )
            current = merged_by_key.get(key)
            if current is None:
                merged_by_key[key] = dict(normalized)
                continue
            current["first_step_index"] = min(
                int(current.get("first_step_index", 0) or 0) or int(normalized.get("first_step_index", 0) or 0),
                int(normalized.get("first_step_index", 0) or 0),
            )
            current["last_step_index"] = max(
                int(current.get("last_step_index", 0) or 0),
                int(normalized.get("last_step_index", 0) or 0),
            )
            current["total_events"] = int(current.get("total_events", 0) or 0) + int(
                normalized.get("total_events", 0) or 0
            )
            current["max_stagnation_after"] = max(
                int(current.get("max_stagnation_after", 0) or 0),
                int(normalized.get("max_stagnation_after", 0) or 0),
            )
            current["action_outcomes"] = _merge_action_stats_dicts(
                current.get("action_outcomes", {}),
                normalized.get("action_outcomes", {}),
            )
            if not current.get("recent_action_sequence"):
                current["recent_action_sequence"] = list(
                    normalized.get("recent_action_sequence", []) or []
                )
            if not current.get("recent_outcome_sequence"):
                current["recent_outcome_sequence"] = list(
                    normalized.get("recent_outcome_sequence", []) or []
                )
            current["decision_sources"] = _merge_prior_buckets(
                current.get("decision_sources", {}),
                normalized.get("decision_sources", {}),
            )
    ranked = sorted(
        merged_by_key.values(),
        key=lambda item: (
            int(item.get("last_step_index", 0) or 0),
            int(item.get("total_events", 0) or 0),
        ),
        reverse=True,
    )
    return ranked[:limit]


def _merge_prior_buckets(
    primary: dict[str, Any],
    secondary: dict[str, Any],
) -> dict[str, int]:
    merged: dict[str, int] = {}
    for source in (primary, secondary):
        for name, value in dict(source or {}).items():
            key = str(name or "")
            if not key:
                continue
            merged[key] = max(int(merged.get(key, 0) or 0), int(value or 0))
    return merged


def _merge_working_memory(
    primary: dict[str, Any],
    secondary: dict[str, Any],
    *,
    recent_records: list[dict[str, Any]],
    semantic_memories: list[dict[str, Any]],
    procedural_memories: list[dict[str, Any]],
    abstract_memories: list[dict[str, Any]],
    recommended_actions: list[str],
    trained_prior_bucket: dict[str, int],
    suppressed_actions: list[str],
    basin_status: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(secondary or {})
    merged.update(dict(primary or {}))
    recent_action_window = _working_memory_recent_action_window(
        recent_records,
    )
    recent_failures = _working_memory_recent_failures(recent_records)
    legal_actions = [
        str(action_name or "")
        for action_name in list(merged.get("legal_actions", []) or [])
        if str(action_name or "")
    ]
    merged["recommended_actions"] = list(recommended_actions[:4])
    merged["trained_prior_focus"] = _compact_trained_prior_focus(
        trained_prior_bucket,
        legal_actions=legal_actions,
    )
    merged["suppressed_actions"] = list(suppressed_actions[:6])
    merged["recent_actions"] = recent_action_window
    merged["recent_action_window"] = recent_action_window
    merged["recent_failures"] = recent_failures
    merged["open_failure_modes"] = _working_memory_open_failure_modes(
        recent_records,
    )
    merged["failure_streaks"] = _working_memory_failure_streaks(
        recent_records,
    )
    merged["semantic_focus_actions"] = _working_memory_focus_actions(
        semantic_memories,
        limit=3,
    )
    merged["procedural_focus_actions"] = _working_memory_focus_actions(
        procedural_memories,
        limit=3,
    )
    merged["abstract_focus_actions"] = _compact_abstract_focus_actions(
        abstract_memories,
        legal_actions=legal_actions,
    )
    if "board_context_summary" in merged and "current_board_context" not in merged:
        merged["current_board_context"] = dict(merged.get("board_context_summary") or {})
    merged["basin_exhausted"] = bool(basin_status.get("is_exhausted", False))
    merged["basin_visits"] = int(basin_status.get("visits", 0) or 0)
    return merged


def _merge_retrieval_summary(
    primary: dict[str, Any],
    secondary: dict[str, Any],
    *,
    recent_records: list[dict[str, Any]],
    episode_summaries: list[dict[str, Any]],
    episodic_recalls: list[dict[str, Any]],
    semantic_memories: list[dict[str, Any]],
    procedural_memories: list[dict[str, Any]],
    abstract_memories: list[dict[str, Any]],
    recommended_actions: list[str],
    trained_prior_bucket: dict[str, int],
    suppressed_actions: list[str],
    unknown_game_context: bool = False,
) -> dict[str, Any]:
    merged = dict(secondary or {})
    merged.update(dict(primary or {}))
    legal_actions = [
        str(action_name or "")
        for action_name in list(merged.get("legal_actions", []) or [])
        if str(action_name or "")
    ]
    merged["recent_record_count"] = len(recent_records)
    merged["episode_summary_count"] = len(episode_summaries)
    merged["episodic_recall_count"] = len(episodic_recalls)
    merged["semantic_memory_count"] = len(semantic_memories)
    merged["procedural_memory_count"] = len(procedural_memories)
    merged["abstract_memory_count"] = len(abstract_memories)
    merged["suggested_actions"] = list(recommended_actions[:4])
    merged["trained_prior_focus"] = _compact_trained_prior_focus(
        trained_prior_bucket,
        legal_actions=legal_actions,
    )
    merged["suppressed_actions"] = list(suppressed_actions[:6])
    merged["abstract_focus_actions"] = _compact_abstract_focus_actions(
        abstract_memories,
        legal_actions=legal_actions,
    )
    merged["top_recall_sources"] = [
        source
        for source in [
            "recent-record" if any(item.get("retrieval_provenance") == "recent-record" for item in episodic_recalls) else "",
            "episode-summary" if any(item.get("retrieval_provenance") == "episode-summary" for item in episodic_recalls) else "",
            "semantic-store" if semantic_memories else "",
            "procedural-store" if procedural_memories else "",
            "abstract-store" if abstract_memories else "",
            "abstraction-transfer" if unknown_game_context and abstract_memories else "",
        ]
        if source
    ]
    merged.setdefault("retrieval_version", "hybrid-multistore-v1")
    return merged


def _select_hot_first_basin_status(
    hot_status: dict[str, Any],
    archive_status: dict[str, Any],
) -> dict[str, Any]:
    hot = dict(hot_status or {})
    archive = dict(archive_status or {})
    hot_strength = sum(
        int(hot.get(field, 0) or 0)
        for field in ("visits", "actions_attempted", "progress_events", "failed_events", "stalled_events")
    )
    archive_strength = sum(
        int(archive.get(field, 0) or 0)
        for field in ("visits", "actions_attempted", "progress_events", "failed_events", "stalled_events")
    )
    if hot_strength > 0 or archive_strength <= 0:
        return hot
    return archive
