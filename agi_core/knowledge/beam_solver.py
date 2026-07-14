from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def compute_reflected_cells(
    beam_cells: list[tuple[int, int]],
    mirror_x: int,
    width: int,
    height: int,
    mirror_y: int | None = None,
) -> set[tuple[int, int]]:
    """Compute beam cells reflected across a player/mirror.
    
    The player acts as both horizontal and vertical mirror simultaneously:
    reflected_x = 2 * mirror_x - beam_x
    reflected_y = 2 * mirror_y - beam_y  (if mirror_y provided)
    If mirror_y is None, only horizontal reflection is done (y unchanged).
    """
    reflected: set[tuple[int, int]] = set()
    for bx, by in beam_cells:
        rx = 2 * mirror_x - bx
        if mirror_y is not None:
            ry = 2 * mirror_y - by
        else:
            ry = by
        if 0 <= rx < width and 0 <= ry < height:
            reflected.add((rx, ry))
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
    mirror_x: int,
    width: int,
    height: int,
    mirror_y: int | None = None,
) -> tuple[int | None, int | None, list[tuple[int, int]]]:
    """Find optimal mirror (player) position to cover as many goals as possible.
    
    Uses smart scoring: weights x and y targets by coverage score,
    preferring positions that cover more goals simultaneously via reflection.
    
    Returns (best_mirror_x, best_mirror_y, still_uncovered_goals).
    """
    from collections import Counter

    uncovered = find_uncovered_goals(goal_positions, beam_cells, set())
    if not uncovered:
        return (mirror_x, mirror_y, [])

    x_scores: Counter[int] = Counter()
    y_scores: Counter[int] = Counter()
    pair_scores: dict[tuple[int, int], int] = {}
    for gx, gy in uncovered:
        for bx, by in beam_cells:
            mx1 = (bx + gx) // 2
            mx2 = (bx + gx + 1) // 2
            for mx in (mx1, mx2):
                if 0 <= mx < width:
                    x_scores[mx] += 1
            if mirror_y is not None:
                my1 = (by + gy) // 2
                my2 = (by + gy + 1) // 2
                for my in (my1, my2):
                    if 0 <= my < height:
                        y_scores[my] += 1
                        pair_key = ((bx + gx) // 2 * height + (by + gy) // 2) if mirror_y is not None else mx * height
                        pair_scores[(mx if mirror_y is None else (bx + gx) // 2, my if mirror_y is not None else 0)] = \
                            pair_scores.get((mx if mirror_y is None else (bx + gx) // 2, my if mirror_y is not None else 0), 0) + 1

    best_mx = x_scores.most_common(1)[0][0] if x_scores else mirror_x
    best_my = y_scores.most_common(1)[0][0] if y_scores else (mirror_y if mirror_y is not None else 0)
    # Use pair_scores to validate combined position
    combined_score = pair_scores.get((best_mx, best_my), 0)
    # If combined score is low, try alternative pairs
    if combined_score < len(uncovered) // 2 and pair_scores:
        top_pair = max(pair_scores, key=lambda k: pair_scores[k])
        best_mx, best_my = top_pair
    reflected = compute_reflected_cells(beam_cells, best_mx, width, height, mirror_y=best_my if mirror_y is not None else None)
    still_uncovered = find_uncovered_goals(goal_positions, beam_cells, reflected)
    return (best_mx, best_my, still_uncovered)


def mirror_direction_to_target(
    current_mirror_pos: tuple[int, int],
    target_mirror_pos: int,
) -> str | None:
    """Return direction to move mirror toward target x."""
    px, _ = current_mirror_pos
    if abs(px - target_mirror_pos) <= 3:
        return None
    return "left" if px > target_mirror_pos else "right"


def check_win_possible(
    goal_positions: list[tuple[int, int]],
    beam_cells: list[tuple[int, int]],
    mirror_x: int,
    width: int,
    height: int,
    player_pos: tuple[int, int] | None = None,
    mirror_y: int | None = None,
) -> bool:
    """Check if all goals can be covered via dual-axis reflection + player."""
    covered = set(beam_cells)
    if player_pos is not None:
        covered.add(player_pos)
    if all(g in covered for g in goal_positions):
        return True
    for test_mx in range(0, width, 4):
        for test_my in range(0, height, 4) if mirror_y is not None else [0]:
            reflected = compute_reflected_cells(beam_cells, test_mx, width, height, mirror_y=test_my if mirror_y is not None else None)
            c = set(beam_cells) | reflected
            if player_pos is not None:
                c.add(player_pos)
            if all(g in c for g in goal_positions):
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
      - mirror_direction: str | None — direction to move mirror
      - best_mirror_x: int | None — target mirror x position
      - best_mirror_y: int | None — target mirror y position
      - uncovered_goals: list — goals still needing coverage
      - all_coverable: bool — whether all goals can be covered
      - memory_hit: bool — whether a cached solution was used
      - alternative_positions: list — alternative (mx, my) positions if primary fails
    """
    px, py = player_pos if player_pos else (mirror_axis, 0)
    if memory_db_path is not None:
        _init_beam_schema(memory_db_path)
    cache_key = (game_id, tuple(sorted(goal_positions)), mirror_axis)
    cached = _load_cached_solution(memory_db_path, cache_key)
    if cached is not None:
        return cached

    if not beam_cells or not goal_positions:
        return {
            "mirror_direction": None,
            "best_mirror_x": None,
            "best_mirror_y": None,
            "uncovered_goals": list(goal_positions),
            "all_coverable": False,
            "memory_hit": False,
            "alternative_positions": [],
        }

    best_mx, best_my, uncovered = compute_mirror_targets(
        goal_positions, beam_cells, px, width, height, mirror_y=py,
    )

    all_coverable = check_win_possible(
        goal_positions, beam_cells, px, width, height, player_pos, mirror_y=py,
    )

    result: dict[str, Any] = {
        "best_mirror_x": best_mx,
        "best_mirror_y": best_my,
        "uncovered_goals": uncovered,
        "all_coverable": all_coverable,
        "memory_hit": False,
        "alternative_positions": [],
    }

    if current_mirror_pos is not None:
        dx = abs(current_mirror_pos[0] - best_mx)
        dy = abs(current_mirror_pos[1] - best_my) if best_my != py else abs(current_mirror_pos[1] - py)
        if dx >= dy:
            result["mirror_direction"] = mirror_direction_to_target(current_mirror_pos, best_mx)
        else:
            result["mirror_direction"] = "up" if current_mirror_pos[1] > best_my else "down" if best_my != py else mirror_direction_to_target(current_mirror_pos, best_mx)
    else:
        result["mirror_direction"] = None

    # Generate alternative positions for fallback
    from collections import Counter
    x_scores: Counter[int] = Counter()
    y_scores: Counter[int] = Counter()
    for gx, gy in goal_positions:
        for bx, by in beam_cells:
            for mx in ((bx + gx) // 2, (bx + gx + 1) // 2):
                if 0 <= mx < width:
                    x_scores[mx] += 1
            if py is not None:
                for my in ((by + gy) // 2, (by + gy + 1) // 2):
                    if 0 <= my < height:
                        y_scores[my] += 1
    top_x = [x for x, _ in x_scores.most_common(3)]
    top_y = [y for y, _ in y_scores.most_common(3)]
    alternatives = []
    for mx in top_x[:2]:
        for my in top_y[:2]:
            if (mx, my) != (best_mx, best_my):
                alternatives.append((mx, my))
    result["alternative_positions"] = alternatives[:3]
    _cache_solution(memory_db_path, cache_key, result)
    return result


def _init_beam_schema(db_path: str | Path) -> None:
    """Initialize the beam_solutions table."""
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
            conn.commit()
    except Exception as exc:
        logger.debug("beam_solver schema init error: %s", exc)


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
                "INSERT OR REPLACE INTO beam_solutions VALUES (?, ?, ?, ?)",
                [cache_key[0], str(cache_key[1]), cache_key[2], __import__('json').dumps(result)],
            )
            conn.commit()
    except Exception as exc:
        logger.debug("beam_solver cache write error: %s", exc)
