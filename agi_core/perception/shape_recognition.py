from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agi_core.image_input import BoardImageInput

try:
    from agi_core.cuda import detect_components_gpu as _detect_components_gpu
    from agi_core.cuda import best_reflection_placement_gpu as _best_reflection_placement_gpu
    _CUDA_COMPONENTS_AVAILABLE = True
except ImportError:
    _CUDA_COMPONENTS_AVAILABLE = False


@dataclass(slots=True)
class _ColorComponent:
    color: int
    cells: list[tuple[int, int]]
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    touches_border: bool


@dataclass(slots=True)
class _Placement:
    entity_index: int
    fresh_hits: int
    cluster_hits: int
    center: tuple[int, int]
    bbox: tuple[int, int, int, int]
    reflected_hits: list[tuple[int, int]]
    goal_cluster_ids: list[str]


def ensure_shape_recognition(
    image_input: BoardImageInput | None,
    derived_features: dict[str, Any] | None,
) -> dict[str, Any]:
    features = derived_features if isinstance(derived_features, dict) else {}
    cached = features.get("shape_recognition")
    current_signature = image_input.frame_signature if image_input is not None else None
    player_hint = _coerce_player_hint(features.get("player_position"))
    cached_player_hint = _coerce_player_hint(cached.get("player_hint")) if isinstance(cached, dict) else None
    if (
        isinstance(cached, dict)
        and cached.get("version") == 1
        and cached.get("frame_signature") == current_signature
        and (
            cached_player_hint == player_hint
            or (
                cached_player_hint is None
                and image_input is None
            )
        )
    ):
        return cached
    artifact = recognize_shape_recognition(image_input, player_hint=player_hint)
    if features is not None:
        features["shape_recognition"] = artifact
    return artifact


def recognize_shape_recognition(
    image_input: BoardImageInput | None,
    *,
    player_hint: tuple[int, int] | None = None,
) -> dict[str, Any]:
    if image_input is None or not image_input.board:
        return {
            "version": 1,
            "frame_signature": image_input.frame_signature if image_input is not None else None,
            "board_size": {"width": 0, "height": 0},
            "components": [],
            "goal_clusters": [],
            "shape_entities": [],
            "equivalence_classes": [],
            "mirror": {"present": False},
            "mirror_puzzle": {"active": False},
        }
    board = image_input.board
    height = len(board)
    width = len(board[0]) if board and board[0] else 0
    background = _detect_background(board)
    components = _color_components(board, background)
    component_payload = [_serialize_component(component, width, height) for component in components]
    mirror_component = _largest_component([component for component in components if component.color == 10])
    goal_components = [
        component
        for component in components
        if (
            component.color == 11
            and not component.touches_border
            and any(x < width - 1 for x, _ in component.cells)
        )
    ]
    goal_clusters = _serialize_goal_clusters(goal_components)
    shape_components = [
        component
        for component in components
        if component.color in (0, 5)
        and 4 <= len(component.cells) <= 100
        and not component.touches_border
    ]
    ordered_shape_components = sorted(
        shape_components,
        key=lambda item: (item.center[1], item.center[0], item.color),
    )
    shape_entities, equivalence_classes = _serialize_shape_entities(ordered_shape_components)
    mirror_payload = _serialize_mirror(mirror_component)
    active_shape_index = _active_shape_index(ordered_shape_components, player_hint=player_hint)
    active_anchor = None
    if active_shape_index is not None and 0 <= active_shape_index < len(ordered_shape_components):
        active_anchor = list(ordered_shape_components[active_shape_index].center)
    mirror_puzzle = _build_mirror_puzzle(
        width=width,
        height=height,
        goal_components=goal_components,
        shape_components=ordered_shape_components,
        active_shape_index=active_shape_index,
        mirror_component=mirror_component,
    )
    return {
        "version": 1,
        "frame_signature": image_input.frame_signature,
        "player_hint": list(player_hint) if player_hint is not None else None,
        "board_size": {"width": width, "height": height},
        "background_color": background,
        "components": component_payload,
        "goal_clusters": goal_clusters,
        "shape_entities": shape_entities,
        "equivalence_classes": equivalence_classes,
        "mirror": mirror_payload,
        "active_shape_entity_id": (
            shape_entities[active_shape_index]["entity_id"]
            if active_shape_index is not None and active_shape_index < len(shape_entities)
            else None
        ),
        "active_anchor": active_anchor,
        "mirror_puzzle": mirror_puzzle,
    }


def _detect_background(board: list[list[int]]) -> int:
    height = len(board)
    width = len(board[0]) if height and board[0] else 0
    if width <= 0 or height <= 0:
        return 0
    from agi_core.orch_or.analyzer import _detect_background_color
    return _detect_background_color(board, width, height)


def _color_components(board: list[list[int]], background_color: int) -> list[_ColorComponent]:
    # Try GPU-accelerated path first
    if _CUDA_COMPONENTS_AVAILABLE:
        gpu_result = _detect_components_gpu(board, bg_color=background_color)
        if gpu_result is not None:
            return [
                _ColorComponent(
                    color=entry["color"],
                    cells=entry["cells"],
                    bbox=entry["bbox"],
                    center=entry["center"],
                    touches_border=entry["touches_border"],
                )
                for entry in gpu_result["components"]
            ]

    # CPU fallback
    height = len(board)
    width = len(board[0]) if height and board[0] else 0
    visited: set[tuple[int, int]] = set()
    components: list[_ColorComponent] = []
    for y in range(height):
        for x in range(width):
            if (x, y) in visited:
                continue
            color = int(board[y][x])
            if color == background_color or color < 0:
                visited.add((x, y))
                continue
            stack = [(x, y)]
            visited.add((x, y))
            cells: list[tuple[int, int]] = []
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx = cx + dx
                    ny = cy + dy
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    if (nx, ny) in visited or int(board[ny][nx]) != color:
                        continue
                    visited.add((nx, ny))
                    stack.append((nx, ny))
            components.append(
                _ColorComponent(
                    color=color,
                    cells=cells,
                    bbox=_bbox(cells),
                    center=_center(cells),
                    touches_border=any(
                        px == 0 or py == 0 or px == width - 1 or py == height - 1
                        for px, py in cells
                    ),
                )
            )
    return components


def _serialize_component(component: _ColorComponent, width: int, height: int) -> dict[str, Any]:
    mask = _normalized_mask(component.cells)
    return {
        "color": component.color,
        "size": len(component.cells),
        "bbox": list(component.bbox),
        "center": list(component.center),
        "touches_border": component.touches_border,
        "normalized_mask": [[x, y] for x, y in mask],
        "dominant_color": component.color,
        "overlay": component.color == 0,
        "interior": (
            component.bbox[0] > 0
            and component.bbox[1] > 0
            and component.bbox[2] < width - 1
            and component.bbox[3] < height - 1
        ),
    }


def _serialize_goal_clusters(goal_components: list[_ColorComponent]) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for index, component in enumerate(
        sorted(goal_components, key=lambda item: (item.center[1], item.center[0]))
    ):
        clusters.append(
            {
                "cluster_id": f"goal-{index + 1}",
                "size": len(component.cells),
                "bbox": list(component.bbox),
                "center": list(component.center),
                "anchor": [component.bbox[0], component.bbox[1]],
                "cells": [[x, y] for x, y in sorted(component.cells)],
            }
        )
    return clusters


def _serialize_shape_entities(
    shape_components: list[_ColorComponent],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(shape_components, key=lambda item: (item.center[1], item.center[0], item.color))
    mask_groups: dict[tuple[tuple[int, int], ...], list[int]] = {}
    entities: list[dict[str, Any]] = []
    for index, component in enumerate(ordered):
        mask = tuple(_normalized_mask(component.cells))
        mask_groups.setdefault(mask, []).append(index)
        entities.append(
            {
                "entity_id": f"shape-{index + 1}",
                "order": index + 1,
                "color": component.color,
                "size": len(component.cells),
                "bbox": list(component.bbox),
                "center": list(component.center),
                "anchor": [component.center[0], component.center[1]],
                "normalized_mask": [[x, y] for x, y in mask],
                "overlay": component.color == 0,
            }
        )
    equivalence_classes: list[dict[str, Any]] = []
    for class_index, (mask, member_indices) in enumerate(
        sorted(mask_groups.items(), key=lambda item: (len(item[1]), item[0]))
    ):
        group_id = f"class-{class_index + 1}"
        for member_index in member_indices:
            entities[member_index]["equivalence_class"] = group_id
        equivalence_classes.append(
            {
                "equivalence_class": group_id,
                "members": [entities[member_index]["entity_id"] for member_index in member_indices],
                "normalized_mask": [[x, y] for x, y in mask],
            }
        )
    return entities, equivalence_classes


def _serialize_mirror(mirror_component: _ColorComponent | None) -> dict[str, Any]:
    if mirror_component is None:
        return {"present": False}
    x1, y1, x2, y2 = mirror_component.bbox
    horizontal = (x2 - x1) > (y2 - y1)
    axis = mirror_component.center[1] if horizontal else mirror_component.center[0]
    return {
        "present": True,
        "orientation": "horizontal" if horizontal else "vertical",
        "bbox": list(mirror_component.bbox),
        "center": list(mirror_component.center),
        "axis_pixel": axis,
        "axis_orientation": "y" if horizontal else "x",
    }


def _build_mirror_puzzle(
    *,
    width: int,
    height: int,
    goal_components: list[_ColorComponent],
    shape_components: list[_ColorComponent],
    active_shape_index: int | None,
    mirror_component: _ColorComponent | None,
) -> dict[str, Any]:
    if mirror_component is None or not goal_components or not shape_components:
        return {"active": False}
    x1, y1, x2, y2 = mirror_component.bbox
    horizontal = (x2 - x1) > (y2 - y1)
    axis = mirror_component.center[1] if horizontal else mirror_component.center[0]
    goal_cells = {cell for component in goal_components for cell in component.cells}
    remaining_goals = set(goal_cells)
    goal_cluster_map = {
        f"goal-{index + 1}": set(component.cells)
        for index, component in enumerate(sorted(goal_components, key=lambda item: (item.center[1], item.center[0])))
    }
    ordered_shapes = sorted(shape_components, key=lambda item: (item.center[1], item.center[0], item.color))
    placements: list[_Placement] = []
    assigned_indices: set[int] = set()
    while remaining_goals and len(assigned_indices) < len(ordered_shapes):
        best_candidate: _Placement | None = None
        best_candidate_key: tuple[int, int, int] | None = None
        for entity_index, component in enumerate(ordered_shapes):
            if entity_index in assigned_indices:
                continue
            placement = _best_reflection_placement(
                width=width,
                height=height,
                component=component,
                remaining_goals=remaining_goals,
                goal_cluster_map=goal_cluster_map,
                horizontal=horizontal,
                axis=axis,
            )
            if placement is None:
                continue
            candidate_key = (
                placement.fresh_hits,
                placement.cluster_hits,
                -abs(placement.center[0] - component.center[0]) - abs(placement.center[1] - component.center[1]),
            )
            candidate = _Placement(
                entity_index=entity_index,
                fresh_hits=placement.fresh_hits,
                cluster_hits=placement.cluster_hits,
                center=placement.center,
                bbox=placement.bbox,
                reflected_hits=placement.reflected_hits,
                goal_cluster_ids=placement.goal_cluster_ids,
            )
            if best_candidate is None or best_candidate_key is None or candidate_key > best_candidate_key:
                best_candidate = candidate
                best_candidate_key = candidate_key
        if best_candidate is None:
            break
        placements.append(best_candidate)
        assigned_indices.add(best_candidate.entity_index)
        remaining_goals -= set(best_candidate.reflected_hits)
    stage_targets: list[dict[str, Any]] = []
    recommended_order: list[str] = []
    for stage_index, placement in enumerate(placements, start=1):
        entity = ordered_shapes[placement.entity_index]
        entity_id = f"shape-{placement.entity_index + 1}"
        recommended_order.append(entity_id)
        stage_targets.append(
            {
                "switch_stage": stage_index,
                "entity_id": entity_id,
                "entity_anchor": [entity.center[0], entity.center[1]],
                "target_anchor": [placement.center[0], placement.center[1]],
                "target_bbox": list(placement.bbox),
                "goal_cluster_ids": list(placement.goal_cluster_ids),
                "reflected_coverage": [[x, y] for x, y in sorted(placement.reflected_hits)],
                "fresh_hits": placement.fresh_hits,
            }
        )
    if not stage_targets:
        return {
            "active": False,
            "orientation": "horizontal" if horizontal else "vertical",
            "axis_pixel": axis,
            "player_side": _side_name(
                ordered_shapes[active_shape_index].center
                if active_shape_index is not None
                else ordered_shapes[0].center,
                horizontal,
                axis,
            ),
            "active_stage": 0,
            "recommended_switch_order": [],
            "stage_targets": [],
            "goal_cluster_count": len(goal_cluster_map),
            "shape_entity_count": len(ordered_shapes),
        }

    active_stage = 1
    if active_shape_index is not None:
        active_entity_id = f"shape-{active_shape_index + 1}"
        for stage in stage_targets:
            if stage["entity_id"] == active_entity_id:
                active_stage = int(stage["switch_stage"])
                break
    return {
        "active": True,
        "orientation": "horizontal" if horizontal else "vertical",
        "axis_pixel": axis,
        "player_side": _side_name(ordered_shapes[active_shape_index].center if active_shape_index is not None else ordered_shapes[0].center, horizontal, axis),
        "active_stage": active_stage,
        "recommended_switch_order": recommended_order,
        "stage_targets": stage_targets,
        "goal_cluster_count": len(goal_cluster_map),
        "shape_entity_count": len(ordered_shapes),
    }


def _best_reflection_placement(
    *,
    width: int,
    height: int,
    component: _ColorComponent,
    remaining_goals: set[tuple[int, int]],
    goal_cluster_map: dict[str, set[tuple[int, int]]],
    horizontal: bool,
    axis: int,
) -> _Placement | None:
    if not remaining_goals:
        return None
    # Try GPU-accelerated path first
    if _CUDA_COMPONENTS_AVAILABLE:
        gpu_result = _best_reflection_placement_gpu(
            width=width, height=height,
            component_cells=component.cells,
            component_bbox=component.bbox,
            remaining_goals=remaining_goals,
            horizontal=horizontal, axis=axis,
        )
        if gpu_result is not None:
            gpu_hits, gpu_cx, gpu_cy = gpu_result
            # Compute side-constrained bbox and cluster IDs on CPU
            x1, y1, x2, y2 = component.bbox
            rel_cells = _normalized_mask(component.cells)
            shape_w = x2 - x1 + 1
            shape_h = y2 - y1 + 1
            tx = gpu_cx - shape_w // 2
            ty = gpu_cy - shape_h // 2
            placed = {(tx + dx, ty + dy) for dx, dy in rel_cells}
            reflected = _reflect_cells(placed, horizontal, axis)
            hits_set = reflected & remaining_goals
            cluster_ids = [
                cid for cid, ccells in goal_cluster_map.items()
                if reflected & ccells
            ]
            return _Placement(
                entity_index=-1,
                fresh_hits=gpu_hits,
                cluster_hits=len(cluster_ids),
                center=(gpu_cx, gpu_cy),
                bbox=(tx, ty, tx + shape_w - 1, ty + shape_h - 1),
                reflected_hits=sorted(hits_set),
                goal_cluster_ids=cluster_ids,
            )

    x1, y1, x2, y2 = component.bbox
    rel_cells = _normalized_mask(component.cells)
    shape_w = x2 - x1 + 1
    shape_h = y2 - y1 + 1
    current_center = component.center
    current_side = _side_sign(current_center, horizontal, axis)
    best: tuple[int, int, int] | None = None
    best_placement: _Placement | None = None
    for tx in range(0, max(0, width - shape_w + 1)):
        for ty in range(0, max(0, height - shape_h + 1)):
            placed = {(tx + dx, ty + dy) for dx, dy in rel_cells}
            if current_side != 0 and any(_side_sign(cell, horizontal, axis) not in (0, current_side) for cell in placed):
                continue
            reflected = _reflect_cells(placed, horizontal, axis)
            if any(px < 0 or py < 0 or px >= width or py >= height for px, py in reflected):
                continue
            hits = reflected & remaining_goals
            if not hits:
                continue
            center = _center(list(placed))
            cluster_ids = [
                cluster_id
                for cluster_id, cluster_cells in goal_cluster_map.items()
                if reflected & cluster_cells
            ]
            score = (
                len(hits),
                len(cluster_ids),
                -(abs(center[0] - current_center[0]) + abs(center[1] - current_center[1])),
            )
            if best is not None and score <= best:
                continue
            best = score
            best_placement = _Placement(
                entity_index=-1,
                fresh_hits=len(hits),
                cluster_hits=len(cluster_ids),
                center=center,
                bbox=(tx, ty, tx + shape_w - 1, ty + shape_h - 1),
                reflected_hits=sorted(hits),
                goal_cluster_ids=cluster_ids,
            )
    if best_placement is None:
        return None
    return _Placement(
        entity_index=-1,
        fresh_hits=best_placement.fresh_hits,
        cluster_hits=best_placement.cluster_hits,
        center=best_placement.center,
        bbox=best_placement.bbox,
        reflected_hits=best_placement.reflected_hits,
        goal_cluster_ids=best_placement.goal_cluster_ids,
    )


def _largest_component(components: list[_ColorComponent]) -> _ColorComponent | None:
    if not components:
        return None
    return max(components, key=lambda item: len(item.cells))


def _active_shape_index(
    shape_components: list[_ColorComponent],
    *,
    player_hint: tuple[int, int] | None = None,
) -> int | None:
    if not shape_components:
        return None
    ordered = sorted(shape_components, key=lambda item: (item.center[1], item.center[0], item.color))
    for index, component in enumerate(ordered):
        if component.color == 0:
            return index
    if player_hint is not None:
        return min(
            range(len(ordered)),
            key=lambda index: _shape_player_distance(ordered[index], player_hint),
        )
    return 0


def _shape_player_distance(component: _ColorComponent, player: tuple[int, int]) -> tuple[int, int]:
    px, py = player
    x1, y1, x2, y2 = component.bbox
    dx = 0 if x1 <= px <= x2 else min(abs(px - x1), abs(px - x2))
    dy = 0 if y1 <= py <= y2 else min(abs(py - y1), abs(py - y2))
    center_dx = abs(component.center[0] - px) + abs(component.center[1] - py)
    return (dx + dy, center_dx)


def _coerce_player_hint(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    try:
        return (int(value[0]), int(value[1]))
    except (TypeError, ValueError):
        return None


def _normalized_mask(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_x = min(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    return sorted((x - min_x, y - min_y) for x, y in cells)


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    return (
        min(x for x, _ in cells),
        min(y for _, y in cells),
        max(x for x, _ in cells),
        max(y for _, y in cells),
    )


def _center(cells: list[tuple[int, int]]) -> tuple[int, int]:
    return (
        sum(x for x, _ in cells) // len(cells),
        sum(y for _, y in cells) // len(cells),
    )


def _reflect_cells(
    cells: set[tuple[int, int]],
    horizontal: bool,
    axis: int,
) -> set[tuple[int, int]]:
    if horizontal:
        return {(x, 2 * axis - y) for x, y in cells}
    return {(2 * axis - x, y) for x, y in cells}


def _side_sign(cell: tuple[int, int], horizontal: bool, axis: int) -> int:
    coord = cell[1] if horizontal else cell[0]
    if coord < axis:
        return -1
    if coord > axis:
        return 1
    return 0


def _side_name(cell: tuple[int, int], horizontal: bool, axis: int) -> str:
    sign = _side_sign(cell, horizontal, axis)
    if horizontal:
        if sign < 0:
            return "above"
        if sign > 0:
            return "below"
        return "on-axis"
    if sign < 0:
        return "left"
    if sign > 0:
        return "right"
    return "on-axis"
