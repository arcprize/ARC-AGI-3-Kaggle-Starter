from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS interaction_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                board_motif TEXT NOT NULL DEFAULT '',
                click_x INTEGER NOT NULL,
                click_y INTEGER NOT NULL,
                action_name TEXT NOT NULL DEFAULT 'ACTION6',
                outcome TEXT NOT NULL DEFAULT '',
                success_count INTEGER NOT NULL DEFAULT 0,
                fail_count INTEGER NOT NULL DEFAULT 0,
                last_used_at TEXT NOT NULL DEFAULT '',
                step INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_interaction_game
            ON interaction_records(game_id, outcome)
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
        # Prune old states if too many
        count = conn.execute(
            "SELECT COUNT(*) FROM puzzle_board_states WHERE game_id = ?",
            [state.game_id],
        ).fetchone()[0]
        if count > 500:
            conn.execute(
                """DELETE FROM puzzle_board_states WHERE id IN (
                    SELECT id FROM puzzle_board_states
                    WHERE game_id = ?
                    ORDER BY step ASC
                    LIMIT ?
                ) AND game_id = ?""",
                [state.game_id, count - 300, state.game_id],
            )
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


def query_puzzle_solution_fuzzy(
    db_path: str | Path | None,
    game_id: str,
    goal_positions: list[tuple[int, int]],
    min_overlap: float = 0.5,
) -> dict[str, Any] | None:
    """Fuzzy match puzzle solutions by goal overlap instead of exact hash."""
    if db_path is None:
        return None
    path = Path(db_path)
    if not path.exists():
        return None
    goal_set = set(goal_positions)
    with closing(sqlite3.connect(path)) as conn:
        rows = conn.execute(
            """SELECT mirror_target_x, switch_sequence_json, confidence, goal_hash
               FROM puzzle_solutions
               WHERE game_id = ?
               ORDER BY confidence DESC, use_count DESC LIMIT 20""",
            [game_id],
        ).fetchall()
        best_score = -1
        best_row = None
        for row in rows:
            try:
                import ast
                stored_hash = row[3]
                stored_goals = set(ast.literal_eval(stored_hash) if stored_hash.startswith("[") else [])
                overlap = len(goal_set & stored_goals) if stored_goals else 0
                score = overlap / max(len(goal_set), 1)
                if score > best_score and score >= min_overlap:
                    best_score = score
                    best_row = row
            except Exception:
                if best_row is None:
                    best_row = row
        if best_row is not None:
            return {
                "mirror_target_x": best_row[0],
                "switch_sequence": json.loads(best_row[1]),
                "confidence": best_row[2],
                "source": "cached_solution_fuzzy",
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
            conn.execute(
                "UPDATE puzzle_solutions SET confidence = ?, last_used_at = ? WHERE game_id = ? AND board_hash = ? AND goal_hash = ?",
                [max(row[2] * 0.98, 0.3), datetime.now(timezone.utc).isoformat(), game_id, board_hash, goal_hash],
            )
            conn.commit()
            return {
                "mirror_target_x": row[0],
                "switch_sequence": json.loads(row[1]),
                "confidence": row[2],
                "source": "cached_solution",
            }
    return None


def store_puzzle_episode(
    db_path: str | Path | None,
    game_id: str,
    episode_id: str,
    start_step: int,
    end_step: int,
    levels_completed: int,
    solved: bool,
) -> None:
    """Store episode-level puzzle solving metadata."""
    if db_path is None:
        return
    path = Path(db_path)
    try:
        with closing(sqlite3.connect(path)) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO puzzle_episodes
                   (game_id, episode_id, start_step, end_step,
                    levels_completed, actions_taken, solved)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [game_id, episode_id, start_step, end_step,
                 levels_completed, end_step - start_step, 1 if solved else 0],
            )
            conn.commit()
    except Exception:
        pass


def save_puzzle_solution(
    db_path: str | Path,
    game_id: str,
    goal_positions: list[tuple[int, int]],
    mirror_target_x: int,
    switch_sequence: list[str],
    confidence: float = 0.7,
) -> None:
    path = Path(db_path)
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


def store_interaction_record(
    db_path: str | Path | None,
    game_id: str,
    click_x: int,
    click_y: int,
    outcome: str,
    step: int = 0,
    action_name: str = "ACTION6",
    board_motif: str = "",
) -> None:
    if db_path is None:
        return
    path = Path(db_path)
    if not path.parent.exists():
        return
    with closing(sqlite3.connect(path)) as conn:
        existing = conn.execute(
            "SELECT id, success_count, fail_count FROM interaction_records "
            "WHERE game_id = ? AND click_x = ? AND click_y = ? AND action_name = ?",
            [game_id, click_x, click_y, action_name],
        ).fetchone()
        if existing:
            sid, sc, fc = existing
            if outcome in ("progress", "win"):
                conn.execute(
                    "UPDATE interaction_records SET success_count = ?, last_used_at = ?, step = ? WHERE id = ?",
                    [sc + 1, datetime.now(timezone.utc).isoformat(), step, sid],
                )
            else:
                conn.execute(
                    "UPDATE interaction_records SET fail_count = ?, last_used_at = ?, step = ? WHERE id = ?",
                    [fc + 1, datetime.now(timezone.utc).isoformat(), step, sid],
                )
        else:
            sc = 1 if outcome in ("progress", "win") else 0
            fc = 1 if outcome not in ("progress", "win") else 0
            conn.execute(
                "INSERT INTO interaction_records "
                "(game_id, board_motif, click_x, click_y, action_name, outcome, "
                "success_count, fail_count, last_used_at, step) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [game_id, board_motif, click_x, click_y, action_name, outcome,
                 sc, fc, datetime.now(timezone.utc).isoformat(), step],
            )
        conn.commit()


def query_best_interaction(
    db_path: str | Path | None,
    game_id: str,
    player_x: int,
    player_y: int,
    obstacles: list[tuple[int, int, str]],
    max_dist: int = 8,
    board_motif: str = "",
) -> tuple[int, int] | None:
    if db_path is None:
        return None
    path = Path(db_path)
    if not path.exists():
        return None
    obs_positions = {(ox, oy) for ox, oy, _ in obstacles}
    with closing(sqlite3.connect(path)) as conn:
        rows = conn.execute(
            "SELECT click_x, click_y, success_count, fail_count, board_motif, game_id "
            "FROM interaction_records "
            "ORDER BY (success_count - fail_count) DESC, success_count DESC "
            "LIMIT 40",
        ).fetchall()
        best = None
        best_score = -1
        for rx, ry, sc, fc, motif, gid in rows:
            if (rx, ry) not in obs_positions:
                continue
            dist = abs(rx - player_x) + abs(ry - player_y)
            motif_bonus = 0.3 if motif and board_motif and motif == board_motif else 0.0
            cross_game_bonus = 0.1 if gid != game_id else 0.0
            score = sc - fc - dist * 0.5 + motif_bonus + cross_game_bonus
            if score > best_score:
                best_score = score
                best = (rx, ry)
        return best
