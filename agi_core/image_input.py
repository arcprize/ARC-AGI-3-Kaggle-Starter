from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(slots=True)
class BoardImageInput:
    board: list[list[int]]
    height: int
    width: int
    frame_signature: str | None
    source_attr: str | None


def board_context_summary(
    image_input: BoardImageInput | None,
    *,
    derived_features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    features = derived_features or {}
    default_probe = features.get("board_default_probe")
    if not isinstance(default_probe, dict):
        default_probe = default_click_target(image_input) if image_input is not None else {}
    return {
        "board_height": int(
            features.get("board_height", image_input.height if image_input is not None else 0) or 0
        ),
        "board_width": int(
            features.get("board_width", image_input.width if image_input is not None else 0) or 0
        ),
        "board_active_cells": int(
            features.get("board_active_cells", active_cell_count(image_input)) or 0
        ),
        "board_source_attr": features.get(
            "board_source_attr",
            image_input.source_attr if image_input is not None else None,
        ),
        "board_default_probe": dict(default_probe or {}),
    }


def board_context_match_score(
    current: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> int:
    current_summary = _normalized_board_context_summary(current)
    candidate_summary = _normalized_board_context_summary(candidate)
    if not current_summary or not candidate_summary:
        return 0
    score = 0
    current_height = int(current_summary.get("board_height", 0) or 0)
    current_width = int(current_summary.get("board_width", 0) or 0)
    candidate_height = int(candidate_summary.get("board_height", 0) or 0)
    candidate_width = int(candidate_summary.get("board_width", 0) or 0)
    current_active = int(current_summary.get("board_active_cells", 0) or 0)
    candidate_active = int(candidate_summary.get("board_active_cells", 0) or 0)
    current_source = str(current_summary.get("board_source_attr") or "")
    candidate_source = str(candidate_summary.get("board_source_attr") or "")
    current_probe = dict(current_summary.get("board_default_probe") or {})
    candidate_probe = dict(candidate_summary.get("board_default_probe") or {})
    if current_height > 0 and current_width > 0:
        if current_height == candidate_height and current_width == candidate_width:
            score += 2
    if current_active > 0:
        if current_active == candidate_active:
            score += 2
    elif (
        current_height > 0
        and current_width > 0
        and candidate_height == current_height
        and candidate_width == current_width
        and candidate_active == 0
    ):
        score += 1
    if current_source and current_source == candidate_source:
        score += 1
    if current_probe and current_probe == candidate_probe:
        score += 1
    return score


def extract_board_motif_tags(
    image_input: BoardImageInput | None,
    *,
    board_context: dict[str, Any] | None = None,
    legal_actions: list[str] | None = None,
    levels_completed: int | None = None,
    next_levels_completed: int | None = None,
) -> list[str]:
    summary = _normalized_board_context_summary(
        board_context if board_context is not None else board_context_summary(image_input)
    )
    tags: set[str] = set()
    board = image_input.board if image_input is not None else []
    active_cells = _active_cells(board) if board else []
    active_cell_set = set(active_cells)
    default_probe = dict(summary.get("board_default_probe") or {})
    probe_cell = (
        int(default_probe.get("x", -1) or -1),
        int(default_probe.get("y", -1) or -1),
    )
    components = _connected_active_components(board) if board else []
    legal = {str(action_name or "").upper() for action_name in list(legal_actions or [])}
    board_height = int(summary.get("board_height", 0) or 0)
    board_width = int(summary.get("board_width", 0) or 0)
    board_active_cells = int(summary.get("board_active_cells", 0) or 0)

    if board and _is_horizontally_symmetric(board):
        tags.add("axis_horizontal")
    if board and _is_vertically_symmetric(board):
        tags.add("axis_vertical")
    if len(active_cells) == 1:
        tags.add("single_control_selected")
    if len(components) >= 2:
        tags.add("multiple_piece_candidates")
        tags.add("multi_piece_level")
    elif len({int(cell or 0) for row in board for cell in row if int(cell or 0) != 0}) >= 2:
        tags.add("multi_piece_level")
    if len(active_cells) == 1 and probe_cell in active_cell_set:
        tags.add("selection_on_control")
    if active_cells and probe_cell in active_cell_set:
        tags.add("selection_on_piece")
    if board_active_cells > 1 and len(components) >= 1:
        tags.add("piece_translation_required")
    if active_cells and probe_cell not in active_cell_set:
        tags.add("control_translation_required")
    if (
        "axis_horizontal" in tags or "axis_vertical" in tags
    ) and board_active_cells >= 2:
        tags.add("mirror_alignment_candidate")
    if "ACTION6" in legal and board_height > 0 and board_width > 0:
        area = max(1, board_height * board_width)
        if board_active_cells >= max(4, area // 3):
            tags.add("probe_legal_and_dense_board")
    if (
        levels_completed is not None
        and next_levels_completed is not None
        and int(next_levels_completed) > int(levels_completed)
    ):
        tags.add("post_progress_new_level")

    # ── Spatial and colour-presence tags ──────────────────────────────

    # Colour presence (fast pixel scan)
    if any(10 in row for row in board):
        tags.add("mirror_present")
    beam_present, beam_evidence_uncertain = _beam_presence_state(board)
    if beam_present:
        tags.add("beam_present")
    elif beam_evidence_uncertain:
        tags.add("beam_evidence_uncertain")

    # Component spatial analysis
    if components:
        all_cells = [cell for comp in components for cell in comp]
        xs = [x for x, _ in all_cells]
        ys = [y for _, y in all_cells]

        # Strip layouts
        bw = max(board_width, 1)
        bh = max(board_height, 1)
        if max(xs) - min(xs) < bw * 0.2 and max(ys) - min(ys) > bh * 0.5:
            tags.add("components_vertical_strip")
        if max(ys) - min(ys) < bh * 0.2 and max(xs) - min(xs) > bw * 0.5:
            tags.add("components_horizontal_strip")

        # Player isolation: only one non-trivial component
        if len(components) <= 2:
            tags.add("player_isolated")

        # Large central component (> 30 % board area)
        for comp in components:
            cx1 = min(c[0] for c in comp)
            cx2 = max(c[0] for c in comp)
            cy1 = min(c[1] for c in comp)
            cy2 = max(c[1] for c in comp)
            cw = cx2 - cx1 + 1
            ch = cy2 - cy1 + 1
            if cw * ch >= 0.30 * bw * bh:
                tags.add("large_central_component")
                break

        # Active cells near board edge
        for x, y in all_cells:
            if x < 3 or x >= board_width - 3 or y < 3 or y >= board_height - 3:
                tags.add("active_at_edge")
                break

        # Goal cells in corner regions
        for comp in components:
            color = board[comp[0][1]][comp[0][0]] if comp else -1
            if color == 11:
                for x, y in comp:
                    in_corner = (
                        (x < 8 and y < 8)
                        or (x < 8 and y >= board_height - 8)
                        or (x >= board_width - 8 and y < 8)
                        or (x >= board_width - 8 and y >= board_height - 8)
                    )
                    if in_corner:
                        tags.add("goals_in_corner")
                        break
                else:
                    continue
                break

    return sorted(tags)


def _beam_presence_state(board: list[list[int]]) -> tuple[bool, bool]:
    if not board:
        return False, False
    if not any(4 in row for row in board):
        return False, False
    try:
        from agi_core.orch_or.analyzer import build_game_model
        return bool(build_game_model(board).features.has_beam), False
    except Exception:
        return False, True


def default_click_target(image_input: BoardImageInput | None) -> dict[str, int]:
    if image_input is None:
        return {"x": 32, "y": 32}
    active_cells: list[tuple[int, int]] = []
    for y, row in enumerate(image_input.board):
        for x, value in enumerate(row):
            if int(value) != 0:
                active_cells.append((x, y))
    if active_cells:
        dense_target = _dense_board_contrast_target(image_input, active_cells)
        if dense_target is not None:
            return dense_target
        avg_x = _round_half_up(sum(x for x, _ in active_cells) / len(active_cells))
        avg_y = _round_half_up(sum(y for _, y in active_cells) / len(active_cells))
        return {"x": avg_x, "y": avg_y}
    return {
        "x": image_input.width // 2,
        "y": image_input.height // 2,
    }


def board_probe_click_target(
    image_input: BoardImageInput | None,
    attempt_index: int = 0,
) -> dict[str, int]:
    if image_input is None:
        return {"x": 32, "y": 32}
    active_cells = _active_cells(image_input.board)
    if not active_cells:
        return default_click_target(image_input)
    centroid = default_click_target(image_input)
    min_x = min(x for x, _ in active_cells)
    max_x = max(x for x, _ in active_cells)
    min_y = min(y for _, y in active_cells)
    max_y = max(y for _, y in active_cells)
    targets: list[dict[str, int]] = []
    for x, y in (
        (centroid["x"], centroid["y"]),
        (min_x, min_y),
        (max_x, min_y),
        (min_x, max_y),
        (max_x, max_y),
    ):
        candidate = {"x": int(x), "y": int(y)}
        if candidate not in targets:
            targets.append(candidate)
    if not targets:
        return centroid
    normalized_index = max(0, int(attempt_index or 0)) % len(targets)
    return targets[normalized_index]


def click_target_bias(
    image_input: BoardImageInput | None,
    metadata: dict[str, object] | None,
) -> float:
    if image_input is None or not metadata:
        return 0.0
    x = _coerce_int(metadata.get("x"))
    y = _coerce_int(metadata.get("y"))
    if x is None or y is None:
        return 0.0
    if x < 0 or y < 0 or x >= image_input.width or y >= image_input.height:
        return -0.12
    cell_value = int(image_input.board[y][x])
    if cell_value != 0:
        return 0.10
    if _has_active_cells(image_input.board):
        return -0.04
    return 0.02


def canonical_click_target_bonus(
    image_input: BoardImageInput | None,
    metadata: dict[str, object] | None,
) -> float:
    if image_input is None or not metadata:
        return 0.0
    x = _coerce_int(metadata.get("x"))
    y = _coerce_int(metadata.get("y"))
    if x is None or y is None:
        return 0.0
    target = default_click_target(image_input)
    if x == target["x"] and y == target["y"]:
        return 0.04
    return 0.0


def active_cell_count(image_input: BoardImageInput | None) -> int:
    if image_input is None:
        return 0
    return sum(
        1
        for row in image_input.board
        for cell in row
        if int(cell) != 0
    )


def preferred_board_probe_action(
    legal_actions: Iterable[str],
    image_input: BoardImageInput | None,
    *,
    suppressed_actions: set[str] | None = None,
) -> str | None:
    if active_cell_count(image_input) <= 0:
        return None
    blocked = suppressed_actions or set()
    for action_name in legal_actions:
        if action_name == "ACTION6" and action_name not in blocked:
            return action_name
    return None


def board_image_input_from_source(
    source: Any,
    source_attr: str | None = None,
) -> BoardImageInput | None:
    board = normalize_visible_board(source)
    if board is None:
        return None
    height = len(board)
    width = len(board[0]) if board and board[0] else 0
    return BoardImageInput(
        board=board,
        height=height,
        width=width,
        frame_signature=compute_board_signature(board),
        source_attr=source_attr,
    )


def extract_board_image_input(raw_frame_ref: Any) -> BoardImageInput | None:
    if raw_frame_ref is None:
        return None
    direct = board_image_input_from_source(raw_frame_ref, source_attr=None)
    if direct is not None:
        return direct
    for attribute in ("frame", "frames", "board", "grid"):
        value = getattr(raw_frame_ref, attribute, None)
        image_input = board_image_input_from_source(value, source_attr=attribute)
        if image_input is not None:
            return image_input
    return None


def normalize_visible_board(frame: Any) -> list[list[int]] | None:
    if isinstance(frame, list) and frame:
        if _is_int_grid(frame):
            return _copy_int_grid(frame)
        candidate = frame[-1]
        if _is_int_grid(candidate):
            return _copy_int_grid(candidate)
    return None


def compute_board_signature(board: list[list[int]] | None) -> str | None:
    if not board:
        return None
    height = len(board)
    width = len(board[0]) if board[0] else 0
    payload = json.dumps(
        {"height": height, "width": width, "board": board},
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _is_int_grid(candidate: Any) -> bool:
    if not isinstance(candidate, list) or not candidate:
        return False
    for row in candidate:
        if not isinstance(row, list):
            return False
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, int):
                return False
    return True


def _copy_int_grid(board: list[list[int]]) -> list[list[int]]:
    return [[int(cell) for cell in row] for row in board]


def _has_active_cells(board: list[list[int]]) -> bool:
    for row in board:
        for cell in row:
            if int(cell) != 0:
                return True
    return False


def _active_cells(board: list[list[int]]) -> list[tuple[int, int]]:
    active: list[tuple[int, int]] = []
    for y, row in enumerate(board):
        for x, cell in enumerate(row):
            if int(cell) != 0:
                active.append((x, y))
    return active


def _connected_active_components(board: list[list[int]]) -> list[list[tuple[int, int]]]:
    active = set(_active_cells(board))
    if not active:
        return []
    components: list[list[tuple[int, int]]] = []
    while active:
        start = active.pop()
        stack = [start]
        component = [start]
        while stack:
            x, y = stack.pop()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (x + dx, y + dy)
                if neighbor not in active:
                    continue
                active.remove(neighbor)
                stack.append(neighbor)
                component.append(neighbor)
        components.append(component)
    return components


def _is_horizontally_symmetric(board: list[list[int]]) -> bool:
    if not board:
        return False
    height = len(board)
    for y in range(height):
        if board[y] != board[height - 1 - y]:
            return False
    return True


def _is_vertically_symmetric(board: list[list[int]]) -> bool:
    if not board:
        return False
    for row in board:
        if row != list(reversed(row)):
            return False
    return True


def _dense_board_contrast_target(
    image_input: BoardImageInput,
    active_cells: list[tuple[int, int]],
) -> dict[str, int] | None:
    total_cells = int(image_input.width) * int(image_input.height)
    active_count = len(active_cells)
    zero_count = total_cells - active_count
    if total_cells <= 0 or zero_count <= 0:
        return None
    if (active_count / total_cells) < 0.75:
        return None
    center_x = (image_input.width - 1) / 2.0
    center_y = (image_input.height - 1) / 2.0
    best_cell: tuple[int, int] | None = None
    best_key: tuple[int, float, int, int] | None = None
    for x, y in active_cells:
        zero_neighbors = _zero_neighbor_count(image_input.board, x, y)
        if zero_neighbors <= 0:
            continue
        center_distance = abs(x - center_x) + abs(y - center_y)
        candidate_key = (
            zero_neighbors,
            -center_distance,
            -y,
            -x,
        )
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_cell = (x, y)
    if best_cell is None:
        return None
    return {"x": int(best_cell[0]), "y": int(best_cell[1])}


def _zero_neighbor_count(board: list[list[int]], x: int, y: int) -> int:
    count = 0
    height = len(board)
    width = len(board[0]) if height > 0 and board[0] else 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx = x + dx
            ny = y + dy
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            if int(board[ny][nx]) == 0:
                count += 1
    return count


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _round_half_up(value: float) -> int:
    return int(value + 0.5)


def _normalized_board_context_summary(
    summary: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    return {
        "board_height": int(summary.get("board_height", 0) or 0),
        "board_width": int(summary.get("board_width", 0) or 0),
        "board_active_cells": int(summary.get("board_active_cells", 0) or 0),
        "board_source_attr": str(summary.get("board_source_attr") or "") or None,
        "board_default_probe": dict(summary.get("board_default_probe") or {}),
    }
