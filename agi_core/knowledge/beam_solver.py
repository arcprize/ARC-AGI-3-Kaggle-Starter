from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def compute_reflected_cells(
    beam_cells: list[tuple[int, int]],
    mirror_pos: int,
    width: int,
    height: int,
    vertical: bool = False,
) -> set[tuple[int, int]]:
    """Compute beam cells reflected across a mirror.
    
    Horizontal (vertical=False): reflected_x = 2 * mirror_pos - beam_x, y unchanged.
    Vertical (vertical=True): reflected_y = 2 * mirror_pos - beam_y, x unchanged.
    """
    reflected: set[tuple[int, int]] = set()
    for bx, by in beam_cells:
        if vertical:
            ry = 2 * mirror_pos - by
            if 0 <= bx < width and 0 <= ry < height:
                reflected.add((bx, ry))
        else:
            rx = 2 * mirror_pos - bx
            if 0 <= rx < width and 0 <= by < height:
                reflected.add((rx, by))
    return reflected


def find_uncovered_goals(
    goal_positions: list[tuple[int, int]],
    beam_cells: list[tuple[int, int]],
    reflected: set[tuple[int, int]],
    player_pos: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    """Return goals NOT covered by beam, reflection, or player."""
    covered: set[tuple[int, int]] = set(beam_cells) | reflected
    if player_pos is not None:
        covered.add(player_pos)
    return [g for g in goal_positions if g not in covered]


def compute_mirror_targets(
    goal_positions: list[tuple[int, int]],
    beam_cells: list[tuple[int, int]],
    mirror_axis: int,
    width: int,
    height: int,
) -> tuple[int | None, list[tuple[int, int]]]:
    """Find optimal mirror x position to cover as many goals as possible.
    
    Returns (best_mirror_x, still_uncovered_goals).
    """
    from collections import Counter

    uncovered = find_uncovered_goals(goal_positions, beam_cells, set())
    if not uncovered:
        return (mirror_axis, [])

    # For each uncovered goal and each beam cell at the same y,
    # compute the mirror x that would reflect that beam cell onto the goal.
    position_scores: Counter[int] = Counter()
    for gx, gy in uncovered:
        for bx, by in beam_cells:
            if by != gy:
                continue
            for mx in ((bx + gx) // 2, (bx + gx + 1) // 2):
                if 0 <= mx < width:
                    position_scores[mx] += 1

    if not position_scores:
        return (mirror_axis, uncovered)

    best_mx = position_scores.most_common(1)[0][0]
    reflected = compute_reflected_cells(beam_cells, best_mx, width, height)
    still_uncovered = find_uncovered_goals(goal_positions, beam_cells, reflected)
    return (best_mx, still_uncovered)


def mirror_direction_to_target(
    current_mirror_pos: tuple[int, int],
    target_mirror_pos: int,
    vertical: bool = False,
) -> str | None:
    """Return direction to move mirror toward target position."""
    if vertical:
        _, py = current_mirror_pos
        if abs(py - target_mirror_pos) <= 3:
            return None
        return "up" if py > target_mirror_pos else "down"
    px, _ = current_mirror_pos
    if abs(px - target_mirror_pos) <= 3:
        return None
    return "left" if px > target_mirror_pos else "right"


def check_win_possible(
    goal_positions: list[tuple[int, int]],
    beam_cells: list[tuple[int, int]],
    mirror_axis: int,
    width: int,
    height: int,
    player_pos: tuple[int, int] | None = None,
) -> bool:
    """Check if all goals can be covered via reflection + player."""
    test_positions = set()
    for _, gy in goal_positions:
        for bx, by in beam_cells:
            if by == gy:
                test_positions.add((bx + bx) // 2)
                test_positions.add((bx + bx + 1) // 2)
    if not test_positions:
        test_positions = set(range(0, width, 4))
    for test_mx in test_positions:
        if 0 <= test_mx < width:
            reflected = compute_reflected_cells(beam_cells, test_mx, width, height)
            covered = set(beam_cells) | reflected
            if player_pos is not None:
                covered.add(player_pos)
            if all(g in covered for g in goal_positions):
                return True
    return False


def solve_beam_puzzle(
    goal_positions: list[tuple[int, int]],
    beam_cells: list[tuple[int, int]],
    mirror_axis: int,
    player_pos: tuple[int, int] | None,
    current_mirror_pos: tuple[int, int] | None,
    width: int,
    height: int,
    memory_db_path: str | Path | None = None,
    game_id: str = "",
) -> dict[str, Any]:
    """Main entry point for the beam puzzle solver.
    
    Returns dict with:
      - mirror_direction: str | None — direction to move mirror ('left'/'right')
      - best_mirror_x: int | None — target mirror position
      - uncovered_goals: list — goals still needing coverage
      - all_coverable: bool — whether all goals can be covered
      - memory_hit: bool — whether a cached solution was used
    """
    cache_key = (game_id, tuple(sorted(goal_positions)), mirror_axis)
    cached = _load_cached_solution(memory_db_path, cache_key)
    if cached is not None:
        return cached

    if not beam_cells or not goal_positions:
        return {
            "mirror_direction": None,
            "best_mirror_x": None,
            "uncovered_goals": list(goal_positions),
            "all_coverable": False,
            "memory_hit": False,
        }

    best_mx, uncovered = compute_mirror_targets(
        goal_positions, beam_cells, mirror_axis, width, height,
    )

    all_coverable = check_win_possible(
        goal_positions, beam_cells, mirror_axis, width, height, player_pos,
    )

    result: dict[str, Any] = {
        "best_mirror_x": best_mx,
        "uncovered_goals": uncovered,
        "all_coverable": all_coverable,
        "memory_hit": False,
    }

    if best_mx is not None and current_mirror_pos is not None:
        result["mirror_direction"] = mirror_direction_to_target(current_mirror_pos, best_mx)
    else:
        result["mirror_direction"] = None

    _cache_solution(memory_db_path, cache_key, result)
    return result


def _load_cached_solution(
    db_path: str | Path | None,
    cache_key: tuple,
) -> dict[str, Any] | None:
    if db_path is None:
        return None
    path = Path(db_path)
    if not path.exists():
        return None
    try:
        with closing(sqlite3.connect(path)) as conn:
            exact_hash = str(cache_key[1])
            row = conn.execute(
                "SELECT result_json FROM beam_solutions WHERE game_id=? AND goal_hash=? AND mirror_axis=?",
                [cache_key[0], exact_hash, cache_key[2]],
            ).fetchone()
            if row is not None:
                import json
                return json.loads(row[0])
            fuzzy = conn.execute(
                "SELECT result_json, goal_hash FROM beam_solutions WHERE game_id=? AND mirror_axis=? ORDER BY use_count DESC LIMIT 5",
                [cache_key[0], cache_key[2]],
            ).fetchall()
            if fuzzy:
                import ast, json
                current_goals = set(cache_key[1])
                best = None
                best_overlap = 0
                for row2 in fuzzy:
                    try:
                        stored = set(ast.literal_eval(row2[1]))
                        overlap = len(current_goals & stored)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best = json.loads(row2[0])
                    except Exception as exc:
                        logger.debug("beam_solver fuzzy cache parse error: %s", exc)
                if best is not None and best_overlap >= max(1, len(current_goals) // 2):
                    return best
    except Exception as exc:
        logger.debug("beam_solver cache load error: %s", exc)
    return None


def _cache_solution(
    db_path: str | Path | None,
    cache_key: tuple,
    result: dict[str, Any],
) -> None:
    if db_path is None:
        return
    path = Path(db_path)
    try:
        with closing(sqlite3.connect(path)) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS beam_solutions (
                    game_id TEXT,
                    goal_hash TEXT,
                    mirror_axis INTEGER,
                    result_json TEXT,
                    PRIMARY KEY (game_id, goal_hash, mirror_axis)
                )"""
            )
            import json
            conn.execute(
                "INSERT OR REPLACE INTO beam_solutions VALUES (?, ?, ?, ?)",
                [cache_key[0], str(cache_key[1]), cache_key[2], json.dumps(result)],
            )
            conn.commit()
    except Exception as exc:
        logger.debug("beam_solver cache write error: %s", exc)
