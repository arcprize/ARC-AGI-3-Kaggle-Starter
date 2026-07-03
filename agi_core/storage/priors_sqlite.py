from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable


def initialize_schema(db_path: str | Path) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection:
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
        connection.commit()


def upsert_metadata(db_path: str | Path, values: dict[str, str]) -> None:
    if not values:
        return
    with closing(sqlite3.connect(Path(db_path))) as connection:
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
    with closing(sqlite3.connect(Path(db_path))) as connection:
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
        connection.commit()


def load_best_prior_bucket(
    db_path: str | Path,
    game_id: str,
    state: str,
    frame_signature: str,
    legal_actions: list[str],
) -> dict[str, int]:
    path = Path(db_path)
    if not path.exists() or not legal_actions:
        return {}
    with closing(sqlite3.connect(path)) as connection:
        frame_bucket = _load_bucket(
            connection,
            """
            SELECT action_name, count
            FROM frame_action_priors
            WHERE game_id = ? AND frame_signature = ?
            """,
            [game_id, frame_signature],
            legal_actions,
        )
        if frame_bucket:
            return frame_bucket
        state_bucket = _load_bucket(
            connection,
            """
            SELECT action_name, count
            FROM state_action_priors
            WHERE game_id = ? AND state = ?
            """,
            [game_id, state],
            legal_actions,
        )
        if state_bucket:
            return state_bucket
        return _load_bucket(
            connection,
            """
            SELECT action_name, count
            FROM game_action_priors
            WHERE game_id = ?
            """,
            [game_id],
            legal_actions,
        )


def load_game_summary(db_path: str | Path, game_id: str) -> dict[str, Any]:
    path = Path(db_path)
    if not path.exists():
        return {"actions": {}, "states": {}, "frame_signatures": {}}
    with closing(sqlite3.connect(path)) as connection:
        summary = {
            "actions": _query_pairs(
                connection,
                "SELECT action_name, count FROM game_action_priors WHERE game_id = ?",
                [game_id],
            ),
            "states": {},
            "frame_signatures": {},
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
