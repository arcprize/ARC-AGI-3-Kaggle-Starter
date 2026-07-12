from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable

from agi_core.storage.memory_cache import get_cache

_MEM_CACHE: dict[str, object] = {}


def _refresh_cache(path: str | Path) -> None:
    cache = get_cache(str(Path(path).resolve()))
    if cache is not None:
        cache.refresh()


def _connect(path: str | Path, *, write: bool = False) -> sqlite3.Connection:
    resolved = str(Path(path).resolve())
    cache = _MEM_CACHE.get(resolved) or get_cache(resolved)
    if cache is not None:
        _MEM_CACHE[resolved] = cache
        return cache.connect(write=write)
    return sqlite3.connect(str(path))


def game_id_aliases(game_id: str) -> list[str]:
    normalized = str(game_id or "").strip()
    if not normalized:
        return [""]
    aliases = [normalized]
    short_id = normalized.split("-", 1)[0].strip()
    if short_id and short_id not in aliases:
        aliases.append(short_id)
    return aliases


def initialize_schema(db_path: str | Path) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect(path, write=True)) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS game_action_priors (
                game_id TEXT NOT NULL,
                action_name TEXT NOT NULL,
                count INTEGER NOT NULL,
                PRIMARY KEY (game_id, action_name)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS state_action_priors (
                game_id TEXT NOT NULL,
                state TEXT NOT NULL,
                action_name TEXT NOT NULL,
                count INTEGER NOT NULL,
                PRIMARY KEY (game_id, state, action_name)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS frame_action_priors (
                game_id TEXT NOT NULL,
                frame_signature TEXT NOT NULL,
                action_name TEXT NOT NULL,
                count INTEGER NOT NULL,
                PRIMARY KEY (game_id, frame_signature, action_name)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS subgoal_action_priors (
                game_id TEXT NOT NULL,
                subgoal TEXT NOT NULL,
                action_name TEXT NOT NULL,
                count INTEGER NOT NULL,
                PRIMARY KEY (game_id, subgoal, action_name)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_state_lookup
            ON state_action_priors(game_id, state)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_frame_lookup
            ON frame_action_priors(game_id, frame_signature)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_subgoal_lookup
            ON subgoal_action_priors(game_id, subgoal)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS board_policy_params (
                board_type TEXT NOT NULL,
                param_name TEXT NOT NULL,
                param_value TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                sample_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (board_type, param_name)
            )
            """
        )
        connection.commit()
    cache = get_cache(str(path))
    if cache is not None:
        cache.refresh()


def upsert_metadata(db_path: str | Path, values: dict[str, str]) -> None:
    if not values:
        return
    with closing(_connect(db_path, write=True)) as connection:
        connection.executemany(
            """
            INSERT INTO metadata(key, value)
            VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            [(key, value) for key, value in values.items()],
        )
        connection.commit()


def upsert_training_counts(db_path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    game_rows: list[tuple[str, str, int]] = []
    state_rows: list[tuple[str, str, str, int]] = []
    frame_rows: list[tuple[str, str, str, int]] = []
    subgoal_rows: list[tuple[str, str, str, int]] = []
    for row in rows:
        scope = str(row.get("scope", ""))
        game_id = str(row.get("game_id", ""))
        action_name = str(row.get("action_name", ""))
        count = int(row.get("count", 0) or 0)
        if not game_id or not action_name or count <= 0:
            continue
        if scope == "game":
            game_rows.append((game_id, action_name, count))
        elif scope == "state":
            state_rows.append((game_id, str(row.get("state", "")), action_name, count))
        elif scope == "frame":
            frame_rows.append(
                (game_id, str(row.get("frame_signature", "")), action_name, count)
            )
        elif scope == "subgoal":
            subgoal_rows.append(
                (game_id, str(row.get("subgoal", "")), action_name, count)
            )
    with closing(_connect(db_path, write=True)) as connection:
        if game_rows:
            connection.executemany(
                """
                INSERT INTO game_action_priors(game_id, action_name, count)
                VALUES(?, ?, ?)
                ON CONFLICT(game_id, action_name)
                DO UPDATE SET count=game_action_priors.count + excluded.count
                """,
                game_rows,
            )
        if state_rows:
            connection.executemany(
                """
                INSERT INTO state_action_priors(game_id, state, action_name, count)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(game_id, state, action_name)
                DO UPDATE SET count=state_action_priors.count + excluded.count
                """,
                state_rows,
            )
        if frame_rows:
            connection.executemany(
                """
                INSERT INTO frame_action_priors(game_id, frame_signature, action_name, count)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(game_id, frame_signature, action_name)
                DO UPDATE SET count=frame_action_priors.count + excluded.count
                """,
                frame_rows,
            )
        if subgoal_rows:
            connection.executemany(
                """
                INSERT INTO subgoal_action_priors(game_id, subgoal, action_name, count)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(game_id, subgoal, action_name)
                DO UPDATE SET count=subgoal_action_priors.count + excluded.count
                """,
                subgoal_rows,
            )
        connection.commit()
    _refresh_cache(db_path)


def load_best_prior_bucket(
    db_path: str | Path,
    game_id: str,
    state: str,
    frame_signature: str,
    legal_actions: list[str],
    subgoal: str | None = None,
) -> dict[str, int]:
    path = Path(db_path)
    if not path.exists() or not legal_actions:
        return {}
    aliases = [alias for alias in game_id_aliases(game_id) if alias]
    with closing(_connect(path)) as connection:
        merged_bucket: dict[str, int] = {}
        if subgoal:
            for alias in aliases:
                subgoal_bucket = _load_bucket(
                    connection,
                    """
                    SELECT action_name, count
                    FROM subgoal_action_priors
                    WHERE game_id = ? AND subgoal = ?
                    """,
                    [alias, str(subgoal)],
                    legal_actions,
                )
                _merge_missing_actions(merged_bucket, subgoal_bucket)
        if subgoal:
            for alias in aliases:
                state_bucket = _load_bucket(
                    connection,
                    """
                    SELECT action_name, count
                    FROM state_action_priors
                    WHERE game_id = ? AND state = ?
                    """,
                    [alias, state],
                    legal_actions,
                )
                _merge_missing_actions(merged_bucket, state_bucket)
            for alias in aliases:
                frame_bucket = _load_bucket(
                    connection,
                    """
                    SELECT action_name, count
                    FROM frame_action_priors
                    WHERE game_id = ? AND frame_signature = ?
                    """,
                    [alias, frame_signature],
                    legal_actions,
                )
                _merge_missing_actions(merged_bucket, frame_bucket)
        else:
            for alias in aliases:
                frame_bucket = _load_bucket(
                    connection,
                    """
                    SELECT action_name, count
                    FROM frame_action_priors
                    WHERE game_id = ? AND frame_signature = ?
                    """,
                    [alias, frame_signature],
                    legal_actions,
                )
                _merge_missing_actions(merged_bucket, frame_bucket)
            for alias in aliases:
                state_bucket = _load_bucket(
                    connection,
                    """
                    SELECT action_name, count
                    FROM state_action_priors
                    WHERE game_id = ? AND state = ?
                    """,
                    [alias, state],
                    legal_actions,
                )
                _merge_missing_actions(merged_bucket, state_bucket)
        for alias in aliases:
            game_bucket = _load_bucket(
                connection,
                """
                SELECT action_name, count
                FROM game_action_priors
                WHERE game_id = ?
                """,
                [alias],
                legal_actions,
            )
            _merge_missing_actions(merged_bucket, game_bucket)
        return merged_bucket


def load_game_summary(db_path: str | Path, game_id: str) -> dict[str, Any]:
    path = Path(db_path)
    if not path.exists():
        return {"actions": {}, "states": {}, "frame_signatures": {}, "subgoals": {}}
    with closing(_connect(path)) as connection:
        summary = {
            "actions": _query_pairs(
                connection,
                "SELECT action_name, count FROM game_action_priors WHERE game_id = ?",
                [game_id],
            ),
            "states": {},
            "frame_signatures": {},
            "subgoals": {},
        }
        for state_name, action_name, count in connection.execute(
            """
            SELECT state, action_name, count
            FROM state_action_priors
            WHERE game_id = ?
            ORDER BY state, action_name
            """,
            [game_id],
        ):
            state_entry = summary["states"].setdefault(str(state_name), {"actions": {}})
            state_entry["actions"][str(action_name)] = int(count)
        for frame_signature, action_name, count in connection.execute(
            """
            SELECT frame_signature, action_name, count
            FROM frame_action_priors
            WHERE game_id = ?
            ORDER BY frame_signature, action_name
            """,
            [game_id],
        ):
            frame_entry = summary["frame_signatures"].setdefault(
                str(frame_signature), {"actions": {}}
            )
            frame_entry["actions"][str(action_name)] = int(count)
        for subgoal, action_name, count in connection.execute(
            """
            SELECT subgoal, action_name, count
            FROM subgoal_action_priors
            WHERE game_id = ?
            ORDER BY subgoal, action_name
            """,
            [game_id],
        ):
            subgoal_entry = summary["subgoals"].setdefault(
                str(subgoal), {"actions": {}}
            )
            subgoal_entry["actions"][str(action_name)] = int(count)
        return summary


def _load_bucket(
    connection: sqlite3.Connection,
    base_query: str,
    parameters: list[Any],
    legal_actions: list[str],
) -> dict[str, int]:
    placeholders = ",".join("?" for _ in legal_actions)
    query = (
        f"{base_query} AND action_name IN ({placeholders}) "
        "ORDER BY count DESC, action_name ASC"
    )
    return _query_pairs(connection, query, [*parameters, *legal_actions])


def _query_pairs(
    connection: sqlite3.Connection,
    query: str,
    parameters: list[Any],
) -> dict[str, int]:
    rows = connection.execute(query, parameters).fetchall()
    return {str(action_name): int(count) for action_name, count in rows}


def upsert_board_policy_params(
    db_path: str | Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    with closing(_connect(db_path, write=True)) as connection:
        connection.executemany(
            """
            INSERT INTO board_policy_params(board_type, param_name, param_value, confidence, sample_count)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(board_type, param_name)
            DO UPDATE SET
                param_value = excluded.param_value,
                confidence = excluded.confidence,
                sample_count = board_policy_params.sample_count + excluded.sample_count
            """,
            [
                (
                    str(row.get("board_type", "")),
                    str(row.get("param_name", "")),
                    str(row.get("param_value", "")),
                    float(row.get("confidence", 1.0)),
                    int(row.get("sample_count", 0)),
                )
                for row in rows
            ],
        )
        connection.commit()
    _refresh_cache(db_path)


def load_board_policy_params(
    db_path: str | Path,
    board_type: str,
) -> dict[str, str]:
    path = Path(db_path)
    if not path.exists():
        return {}
    with closing(_connect(path)) as connection:
        rows = connection.execute(
            """
            SELECT param_name, param_value
            FROM board_policy_params
            WHERE board_type = ?
            ORDER BY confidence DESC, sample_count DESC
            """,
            [board_type],
        ).fetchall()
        return {str(name): str(value) for name, value in rows}


def _merge_missing_actions(target: dict[str, int], source: dict[str, int]) -> None:
    for action_name, count in source.items():
        target.setdefault(str(action_name), int(count))
