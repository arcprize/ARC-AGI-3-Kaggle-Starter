from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PuzzleBoardState:
    game_id: str = ""
    step: int = 0
    frame_signature: str = ""
    player_x: int = -1
    player_y: int = -1
    goal_positions: list[tuple[int, int]] = field(default_factory=list)
    beam_cells: list[tuple[int, int]] = field(default_factory=list)
    mirror_x: int = -1
    mirror_y: int = -1
    goals_lit: int = 0
    action_taken: str = ""
    outcome: str = ""
    progress_delta: int = 0


def initialize_puzzle_schema(db_path: str | Path) -> None:
    path = Path(db_path)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS puzzle_board_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                frame_signature TEXT NOT NULL DEFAULT '',
                player_x INTEGER NOT NULL DEFAULT -1,
                player_y INTEGER NOT NULL DEFAULT -1,
                goal_positions_json TEXT NOT NULL DEFAULT '[]',
                beam_cells_json TEXT NOT NULL DEFAULT '[]',
                mirror_x INTEGER NOT NULL DEFAULT -1,
                mirror_y INTEGER NOT NULL DEFAULT -1,
                goals_lit INTEGER NOT NULL DEFAULT 0,
                action_taken TEXT NOT NULL DEFAULT '',
                outcome TEXT NOT NULL DEFAULT '',
                progress_delta INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS puzzle_episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                episode_id TEXT NOT NULL,
                start_step INTEGER NOT NULL,
                end_step INTEGER NOT NULL,
                levels_completed INTEGER NOT NULL DEFAULT 0,
                actions_taken INTEGER NOT NULL DEFAULT 0,
                solved INTEGER NOT NULL DEFAULT 0,
                state_ids_json TEXT NOT NULL DEFAULT '[]'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS puzzle_solutions (
                game_id TEXT NOT NULL,
                board_hash TEXT NOT NULL,
                goal_hash TEXT NOT NULL,
                mirror_target_x INTEGER NOT NULL DEFAULT -1,
                switch_sequence_json TEXT NOT NULL DEFAULT '[]',
                confidence REAL NOT NULL DEFAULT 0.5,
                use_count INTEGER NOT NULL DEFAULT 0,
                last_used_at TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (game_id, board_hash, goal_hash)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_puzzle_states_game ON puzzle_board_states(game_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_puzzle_episodes_game ON puzzle_episodes(game_id)")
        conn.commit()


def _compute_board_hash(goal_positions: list[tuple[int, int]]) -> str:
    return str(sorted(goal_positions)[:5])


def store_board_state(
    db_path: str | Path,
    state: PuzzleBoardState,
) -> int:
    path = Path(db_path)
    with closing(sqlite3.connect(path)) as conn:
        cur = conn.execute(
            """INSERT INTO puzzle_board_states
               (game_id, step, frame_signature, player_x, player_y,
                goal_positions_json, beam_cells_json, mirror_x, mirror_y,
                goals_lit, action_taken, outcome, progress_delta)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                state.game_id, state.step, state.frame_signature,
                state.player_x, state.player_y,
                json.dumps(state.goal_positions),
                json.dumps(state.beam_cells),
                state.mirror_x, state.mirror_y,
                state.goals_lit, state.action_taken,
                state.outcome, state.progress_delta,
            ],
        )
        conn.commit()
        return cur.lastrowid or 0


def prune_board_states(db_path: str | Path, max_per_game: int = 200) -> None:
    path = Path(db_path)
    if not path.exists():
        return
    with closing(sqlite3.connect(path)) as conn:
        game_ids = [r[0] for r in conn.execute("SELECT DISTINCT game_id FROM puzzle_board_states").fetchall()]
        for gid in game_ids:
            conn.execute(
                """DELETE FROM puzzle_board_states WHERE id NOT IN (
                    SELECT id FROM puzzle_board_states
                    WHERE game_id = ?
                    ORDER BY step DESC LIMIT ?
                ) AND game_id = ?""",
                [gid, max_per_game, gid],
            )
        conn.commit()


def query_similar_state(
    db_path: str | Path,
    game_id: str,
    goal_positions: list[tuple[int, int]],
    max_dist: int = 5,
) -> dict[str, Any] | None:
    path = Path(db_path)
    if not path.exists():
        return None
    goal_set = set(goal_positions)
    with closing(sqlite3.connect(path)) as conn:
        rows = conn.execute(
            """SELECT action_taken, outcome, progress_delta, goals_lit, step,
                      goal_positions_json
               FROM puzzle_board_states
               WHERE game_id = ? AND outcome = 'progress'
               ORDER BY step DESC LIMIT 20""",
            [game_id],
        ).fetchall()
        best_score = -1
        best_row = None
        for row in rows:
            try:
                stored_goals = set(tuple(g) for g in json.loads(row[5]))
                overlap = len(goal_set & stored_goals)
                if overlap > best_score:
                    best_score = overlap
                    best_row = row
            except Exception:
                if best_row is None:
                    best_row = row
        if best_row is not None and best_score >= max(1, len(goal_positions) // 2):
            return {
                "action": best_row[0],
                "outcome": best_row[1],
                "progress": best_row[2],
                "goals_lit": best_row[3],
                "step": best_row[4],
                "source": "memory",
            }
    return None


def load_puzzle_solution(
    db_path: str | Path,
    game_id: str,
    goal_positions: list[tuple[int, int]],
) -> dict[str, Any] | None:
    path = Path(db_path)
    if not path.exists():
        return None
    board_hash = _compute_board_hash(goal_positions)
    goal_hash = _compute_board_hash(goal_positions)
    with closing(sqlite3.connect(path)) as conn:
        row = conn.execute(
            """SELECT mirror_target_x, switch_sequence_json, confidence
               FROM puzzle_solutions
               WHERE game_id = ? AND board_hash = ? AND goal_hash = ?
               ORDER BY confidence DESC LIMIT 1""",
            [game_id, board_hash, goal_hash],
        ).fetchone()
        if row is not None:
            return {
                "mirror_target_x": row[0],
                "switch_sequence": json.loads(row[1]),
                "confidence": row[2],
                "source": "cached_solution",
            }
    return None


def save_puzzle_solution(
    db_path: str | Path,
    game_id: str,
    goal_positions: list[tuple[int, int]],
    mirror_target_x: int,
    switch_sequence: list[str],
    confidence: float = 0.7,
) -> None:
    path = Path(db_path)
    from datetime import datetime, timezone
    board_hash = _compute_board_hash(goal_positions)
    goal_hash = _compute_board_hash(goal_positions)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO puzzle_solutions
               (game_id, board_hash, goal_hash, mirror_target_x,
                switch_sequence_json, confidence, use_count, last_used_at)
               VALUES (?, ?, ?, ?, ?, ?, COALESCE(
                   (SELECT use_count + 1 FROM puzzle_solutions
                    WHERE game_id = ? AND board_hash = ? AND goal_hash = ?), 1
               ), ?)""",
            [
                game_id, board_hash, goal_hash,
                mirror_target_x, json.dumps(switch_sequence),
                confidence,
                game_id, board_hash, goal_hash,
                datetime.now(timezone.utc).isoformat(),
            ],
        )
        conn.commit()
