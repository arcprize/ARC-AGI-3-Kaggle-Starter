from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from agi_core.image_input import board_context_match_score
from agi_core.storage.priors_sqlite import initialize_schema, load_best_prior_bucket
from agi_core.types import EpisodeRecord


def initialize_memory_schema(db_path: str | Path) -> None:
    path = Path(db_path)
    initialize_schema(path)
    with closing(sqlite3.connect(path)) as connection:
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
        connection.commit()


def import_legacy_learning_json(db_path: str | Path, json_path: str | Path) -> None:
    db = Path(db_path)
    path = Path(json_path)
    if not path.exists():
        return
    initialize_memory_schema(db)
    with closing(sqlite3.connect(db)) as connection:
        if _metadata_value(connection, "legacy_import_completed") == "1":
            return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    games = payload.get("games", {}) if isinstance(payload, dict) else {}
    if not isinstance(games, dict):
        return
    with closing(sqlite3.connect(db)) as connection:
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
    with closing(sqlite3.connect(db)) as connection:
        _record_transition_with_connection(connection, record)
        connection.commit()


def record_transitions(db_path: str | Path, records: list[EpisodeRecord]) -> None:
    if not records:
        return
    db = Path(db_path)
    initialize_memory_schema(db)
    with closing(sqlite3.connect(db)) as connection:
        for record in records:
            _record_transition_with_connection(connection, record)
        connection.commit()


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
    with closing(sqlite3.connect(path)) as connection:
        basin_frame_signature = frame_signature
        row = _load_basin_row(connection, game_id, state, basin_frame_signature)
        action_outcomes = _load_basin_action_outcomes(
            connection,
            game_id=game_id,
            state=state,
            frame_signature=basin_frame_signature,
        )
        if row is None and frame_signature:
            basin_frame_signature = ""
            row = _load_basin_row(connection, game_id, state, basin_frame_signature)
            action_outcomes = _load_basin_action_outcomes(
                connection,
                game_id=game_id,
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
) -> dict[str, Any]:
    path = Path(db_path)
    if not path.exists():
        bundle = _empty_bundle()
        bundle["board_context_summary"] = _normalized_board_context(board_context)
        return bundle
    with closing(sqlite3.connect(path)) as connection:
        action_stats = _query_action_stats(
            connection,
            """
            SELECT action_name, progress, failed, stalled, total, last_outcome
            FROM live_action_stats
            WHERE game_id = ?
            ORDER BY action_name
            """,
            [game_id],
        )
        state_stats = _query_action_stats(
            connection,
            """
            SELECT action_name, progress, failed, stalled, total, last_outcome
            FROM live_state_action_stats
            WHERE game_id = ? AND state = ?
            ORDER BY action_name
            """,
            [game_id, state],
        )
        frame_stats = _query_action_stats(
            connection,
            """
            SELECT action_name, progress, failed, stalled, total, last_outcome
            FROM live_frame_action_stats
            WHERE game_id = ? AND frame_signature = ?
            ORDER BY action_name
            """,
            [game_id, frame_signature],
        )
        subgoal_stats = _query_action_stats(
            connection,
            """
            SELECT action_name, progress, failed, stalled, total, last_outcome
            FROM live_subgoal_action_stats
            WHERE game_id = ? AND subgoal = ?
            ORDER BY action_name
            """,
            [game_id, str(subgoal or "")],
        )
        basin_status = load_basin_status(path, game_id, state, frame_signature)
        basin_action_stats = dict(basin_status.get("action_outcomes", {}))
        recent_records = _load_recent_records(
            connection,
            game_id=game_id,
            state=state,
            frame_signature=frame_signature,
            subgoal=subgoal,
            board_context=board_context,
        )
    trained_prior_bucket = load_best_prior_bucket(
        path,
        game_id=game_id,
        state=state,
        frame_signature=frame_signature,
        legal_actions=legal_actions,
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
    recommended_actions = _derive_recommended_actions(
        legal_actions=legal_actions,
        action_stats=action_stats,
        state_stats=state_stats,
        basin_action_stats=basin_action_stats,
        frame_stats=frame_stats,
        subgoal_stats=subgoal_stats,
        board_context_stats=board_context_action_stats,
        trained_prior_bucket=trained_prior_bucket,
        recent_records=recent_records,
        suppressed_actions=suppressed_actions,
    )
    basin_status["is_exhausted"] = _is_basin_exhausted(
        basin_status=basin_status,
        legal_actions=legal_actions,
        suppressed_actions=suppressed_actions,
    )
    return {
        "board_context_summary": _normalized_board_context(board_context),
        "board_context_action_stats": board_context_action_stats,
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


def _empty_bundle() -> dict[str, Any]:
    return {
        "board_context_summary": _normalized_board_context(None),
        "board_context_action_stats": {},
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


def _load_recent_records(
    connection: sqlite3.Connection,
    *,
    game_id: str,
    state: str,
    frame_signature: str,
    subgoal: str | None,
    board_context: dict[str, Any] | None,
    limit: int = 8,
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
    scored_rows: list[tuple[tuple[int, int, int, int, int, int], tuple[Any, ...]]] = []
    for row in candidate_rows:
        prior_state = str(row[7] or "")
        prior_frame_signature = str(row[8] or "")
        record_subgoal = str(row[10] or "")
        exact_basin = 1 if state and frame_signature and prior_state == state and prior_frame_signature == frame_signature else 0
        same_state = 1 if state and prior_state == state else 0
        same_frame = 1 if frame_signature and prior_frame_signature == frame_signature else 0
        same_subgoal = 1 if current_subgoal and record_subgoal == current_subgoal else 0
        board_match = board_context_match_score(
            current_board_context,
            {
                "board_height": int(row[15] or 0),
                "board_width": int(row[16] or 0),
                "board_active_cells": int(row[17] or 0),
                "board_source_attr": str(row[18] or "") or None,
                "board_default_probe": _deserialize_probe_json(row[19]),
            },
        )
        scored_rows.append(
            (
                (
                    exact_basin,
                    same_state,
                    same_subgoal,
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


def _table_has_rows(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone()
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
    board_context_stats: dict[str, dict[str, Any]],
    trained_prior_bucket: dict[str, int],
    recent_records: list[dict[str, Any]],
    suppressed_actions: list[str],
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
    for action_name in legal_actions:
        if action_name == "RESET" or action_name in suppressed:
            continue
        basin = basin_action_stats.get(action_name, {})
        frame = frame_stats.get(action_name, {})
        state = state_stats.get(action_name, {})
        subgoal = subgoal_stats.get(action_name, {})
        board_context = board_context_stats.get(action_name, {})
        game = action_stats.get(action_name, {})
        score = 0.0
        score += 5.0 * int(basin.get("progress", 0) or 0)
        score += 4.0 * int(board_context.get("progress", 0) or 0)
        score += 3.5 * int(state.get("progress", 0) or 0)
        score += 2.5 * int(subgoal.get("progress", 0) or 0)
        score += 1.5 * int(frame.get("progress", 0) or 0)
        score += 1.0 * int(game.get("progress", 0) or 0)
        score -= 2.5 * int(basin.get("failed", 0) or 0)
        score -= 1.25 * int(basin.get("stalled", 0) or 0)
        score -= 1.25 * int(board_context.get("failed", 0) or 0)
        score -= 0.60 * int(board_context.get("stalled", 0) or 0)
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


def _derive_board_context_action_stats(
    *,
    board_context: dict[str, Any] | None,
    recent_records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    current_board_context = _normalized_board_context(board_context)
    if not recent_records:
        return {}
    stats: dict[str, dict[str, Any]] = {}
    for record in recent_records[:6]:
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
