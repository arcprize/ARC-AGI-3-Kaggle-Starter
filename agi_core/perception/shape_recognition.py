from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agi_core.image_input import BoardImageInput
from agi_core.perception.gpu_perception import get_cached_perception, gpu_perceive

try:
    from agi_core.cuda import detect_components_gpu as _detect_components_gpu
    from agi_core.cuda import best_reflection_placement_gpu as _best_reflection_placement_gpu
    from agi_core.cuda import kernel_available as _cuda_kernel_available
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


def _is_compatible_shape_recognition_cache(cached: Any) -> bool:
    if not isinstance(cached, dict):
        return False
    if cached.get("version") == 2:
        return True
    return "mirror_puzzle" in cached


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
        _is_compatible_shape_recognition_cache(cached)
        and (
            image_input is None
            or (
                cached.get("frame_signature") == current_signature
                and cached_player_hint == player_hint
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
            "version": 2,
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
        "version": 2,
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
    if _CUDA_COMPONENTS_AVAILABLE and _cuda_kernel_available("components"):
        cached_perception = get_cached_perception(board, bg_color=background_color)
        if cached_perception is None:
            cached_perception = gpu_perceive(board, bg_color=background_color)
        if cached_perception is not None and cached_perception.raw_components:
            return [
                _ColorComponent(
                    color=entry["color"],
                    cells=entry["cells"],
                    bbox=entry["bbox"],
                    center=entry["center"],
                    touches_border=entry["touches_border"],
                )
                for entry in cached_perception.raw_components
            ]

    # Try GPU-accelerated path first
    if _CUDA_COMPONENTS_AVAILABLE and _cuda_kernel_available("components"):
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
    contour = _extract_contour(component.cells)
    perimeter = _compute_perimeter(contour)
    hu_moments = _compute_hu_moments(component.cells)
    features = _compute_shape_features(component.cells, contour, perimeter, component.bbox)
    rotation_estimate = _estimate_rotation_angle(component.cells)
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
        "contour": [[x, y] for x, y in contour],
        "perimeter": perimeter,
        "hu_moments": hu_moments,
        "features": features,
        "rotation_estimate": rotation_estimate,
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


_HU_SIMILARITY_THRESHOLD = 5.0
_FEATURE_SIMILARITY_WEIGHT = 0.3


def _estimate_rotation_angle(cells: list[tuple[int, int]]) -> float:
    """Estimate rotation angle (radians) from central moments.

    Uses the orientation of the best-fit ellipse:
    θ = 0.5 * atan2(2 * μ11, μ20 - μ02)
    Returns angle in [-π/2, π/2] radians.
    """
    if len(cells) < 3:
        return 0.0
    cx = sum(x for x, _ in cells) / len(cells)
    cy = sum(y for _, y in cells) / len(cells)
    mu20 = mu02 = mu11 = 0.0
    for x, y in cells:
        dx = x - cx
        dy = y - cy
        mu20 += dx * dx
        mu02 += dy * dy
        mu11 += dx * dy
    denom = mu20 - mu02
    if abs(denom) < _EPSILON and abs(mu11) < _EPSILON:
        return 0.0
    return 0.5 * math.atan2(2.0 * mu11, denom)


def _feature_vector_distance(f1: dict[str, float], f2: dict[str, float]) -> float:
    """Normalized Euclidean distance between two shape feature vectors.

    Uses subset of features: aspect_ratio, compactness, solidity, extent, convexity.
    Returns distance in [0, 1] (0 = identical, 1 = maximally different).
    """
    keys = ["aspect_ratio", "compactness", "solidity", "extent", "convexity"]
    eps = 1e-6
    dist_sum = 0.0
    count = 0
    for key in keys:
        a = f1.get(key, 0.0)
        b = f2.get(key, 0.0)
        # Normalize difference by max value to get [0, 1] range per feature
        max_val = max(abs(a), abs(b), 1.0)
        dist_sum += abs(a - b) / max_val
        count += 1
    return dist_sum / max(count, 1)


def _serialize_shape_entities(
    shape_components: list[_ColorComponent],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(shape_components, key=lambda item: (item.center[1], item.center[0], item.color))
    mask_groups: dict[tuple[tuple[int, int], ...], list[int]] = {}
    entities: list[dict[str, Any]] = []
    all_hu: list[list[float]] = []
    all_features: list[dict[str, float]] = []
    for index, component in enumerate(ordered):
        mask = tuple(_normalized_mask(component.cells))
        mask_groups.setdefault(mask, []).append(index)
        contour = _extract_contour(component.cells)
        hu_moments = _compute_hu_moments(component.cells)
        all_hu.append(hu_moments)
        features = _compute_shape_features(
            component.cells, contour, _compute_perimeter(contour), component.bbox
        )
        all_features.append(features)
        rotation_estimate = _estimate_rotation_angle(component.cells)
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
                "contour": [[x, y] for x, y in contour],
                "perimeter": _compute_perimeter(contour),
                "hu_moments": hu_moments,
                "features": features,
                "rotation_estimate": rotation_estimate,
            }
        )

    # Exact-match equivalence classes (backward compatible)
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

    # Hu-moment-based descriptor classes (rotation-invariant similarity)
    descriptor_groups: list[list[int]] = _cluster_by_hu_descriptor(all_hu, all_features)
    descriptor_classes: list[dict[str, Any]] = []
    for class_index, member_indices in enumerate(descriptor_groups):
        group_id = f"desc-{class_index + 1}"
        for member_index in member_indices:
            entities[member_index]["descriptor_class"] = group_id
        descriptor_classes.append(
            {
                "descriptor_class": group_id,
                "members": [entities[member_index]["entity_id"] for member_index in member_indices],
            }
        )

    return entities, [*equivalence_classes, *descriptor_classes]


def _cluster_by_hu_descriptor(
    all_hu: list[list[float]],
    all_features: list[dict[str, float]] | None = None,
) -> list[list[int]]:
    """Cluster shape indices by combined Hu moment and feature vector similarity.

    Uses greedy single-linkage clustering.
    Combined distance = (1 - w) * hu_dist + w * feature_dist.
    Returns list of index groups, each group containing similar shapes.
    """
    n = len(all_hu)
    if n <= 1:
        return [list(range(n))]

    assigned = [False] * n
    groups: list[list[int]] = []
    w = _FEATURE_SIMILARITY_WEIGHT

    for i in range(n):
        if assigned[i]:
            continue
        group = [i]
        assigned[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(n):
                if assigned[j]:
                    continue
                for k in group:
                    hu_dist = _hu_log_distance(all_hu[j], all_hu[k])
                    feat_dist = (
                        _feature_vector_distance(all_features[j], all_features[k])
                        if all_features is not None
                        else 0.0
                    )
                    combined = (1.0 - w) * hu_dist + w * feat_dist
                    if combined < _HU_SIMILARITY_THRESHOLD:
                        group.append(j)
                        assigned[j] = True
                        changed = True
                        break
        groups.append(group)

    return groups


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
    mirror_cells = list(mirror_component.cells) if mirror_component else None
    goal_cells = [cell for comp in goal_components for cell in comp.cells] if goal_components else None
    for stage_index, placement in enumerate(placements, start=1):
        entity = ordered_shapes[placement.entity_index]
        entity_id = f"shape-{placement.entity_index + 1}"
        recommended_order.append(entity_id)
        other_shape_cells = [
            c.cells for i, c in enumerate(ordered_shapes) if i != placement.entity_index
        ]
        spatial = _compute_spatial_descriptors(
            list(entity.cells),
            mirror_cells=mirror_cells,
            goal_cells=goal_cells,
            other_shapes_cells=other_shape_cells,
        )
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
                "spatial": spatial,
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
            "goal_cells": [[x, y] for x, y in sorted(goal_cells or set())],
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
        "goal_cells": [[x, y] for x, y in sorted(goal_cells or [])],
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


def _extract_contour(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Extract boundary pixels sorted by polar angle around centroid.

    Returns the outer boundary pixels as an ordered list of (x, y) tuples.
    For small shapes (< 3 cells), returns all cells.
    """
    if len(cells) <= 2:
        return sorted(cells)

    cell_set = set(cells)
    # Boundary: cells with at least one 4-connected neighbor NOT in the shape
    boundary = sorted(
        (x, y) for x, y in cells
        if (x + 1, y) not in cell_set
        or (x - 1, y) not in cell_set
        or (x, y + 1) not in cell_set
        or (x, y - 1) not in cell_set
    )
    if len(boundary) <= 2:
        return boundary

    cx = sum(x for x, _ in boundary) / len(boundary)
    cy = sum(y for _, y in boundary) / len(boundary)
    return sorted(boundary, key=lambda c: (math.atan2(c[1] - cy, c[0] - cx), c[0], c[1]))


def _compute_perimeter(contour: list[tuple[int, int]]) -> int:
    """Compute perimeter as sum of consecutive edge pixel distances.

    8-connectivity: horizontal/vertical steps = 1, diagonal steps = sqrt(2).
    Returns integer perimeter (rounded).
    """
    if len(contour) <= 1:
        return 0
    total = 0.0
    for i in range(len(contour)):
        x1, y1 = contour[i]
        x2, y2 = contour[(i + 1) % len(contour)]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx == 0 or dy == 0:
            total += 1.0
        else:
            total += math.sqrt(2.0)
    return int(round(total))


def _compute_area(cells: list[tuple[int, int]]) -> int:
    """Compute area as number of pixels in the shape."""
    return len(cells)


_EPSILON = 1e-12


def _compute_hu_moments(cells: list[tuple[int, int]]) -> list[float]:
    """Compute 7 rotation-invariant Hu moments for a set of pixel coordinates.

    Uses image moment formulas for binary shapes (I(x,y) = 1 for shape pixels).
    Returns 7 Hu moment values (as floats, clamped to avoid zero/overflow).
    """
    if not cells:
        return [0.0] * 7

    # Raw moments M_{pq} = sum x^p y^q
    m00 = float(len(cells))
    if m00 < _EPSILON:
        return [0.0] * 7

    m10 = sum(x for x, _ in cells)
    m01 = sum(y for _, y in cells)
    cx = m10 / m00
    cy = m01 / m00

    # Central moments mu_{pq} = sum (x-cx)^p (y-cy)^q
    mu20 = mu02 = mu11 = mu30 = mu03 = mu12 = mu21 = 0.0
    for x, y in cells:
        dx = x - cx
        dy = y - cy
        dx2 = dx * dx
        dy2 = dy * dy
        mu20 += dx2
        mu02 += dy2
        mu11 += dx * dy
        dx3 = dx2 * dx
        dy3 = dy2 * dy
        mu30 += dx3
        mu03 += dy3
        mu12 += dx * dy2
        mu21 += dx2 * dy

    # Normalized central moments eta_{pq} = mu_{pq} / mu_{00}^{(1 + (p+q)/2)}
    denom = m00  # mu00 = m00 (already normalized)
    denom2 = denom * denom
    denom_1_5 = denom * math.sqrt(denom)
    denom2_sqrt = denom2 * denom

    eta20 = mu20 / denom2 if denom2 > _EPSILON else 0.0
    eta02 = mu02 / denom2 if denom2 > _EPSILON else 0.0
    eta11 = mu11 / denom2 if denom2 > _EPSILON else 0.0
    eta30 = mu30 / denom_1_5 if denom_1_5 > _EPSILON else 0.0
    eta03 = mu03 / denom_1_5 if denom_1_5 > _EPSILON else 0.0
    eta12 = mu12 / denom_1_5 if denom_1_5 > _EPSILON else 0.0
    eta21 = mu21 / denom_1_5 if denom_1_5 > _EPSILON else 0.0

    # Hu moments (7 invariants)
    h1 = eta20 + eta02
    h2 = (eta20 - eta02) ** 2 + 4 * eta11 ** 2
    h3 = (eta30 - 3 * eta12) ** 2 + (3 * eta21 - eta03) ** 2
    h4 = (eta30 + eta12) ** 2 + (eta21 + eta03) ** 2
    h5 = (
        (eta30 - 3 * eta12) * (eta30 + eta12) * ((eta30 + eta12) ** 2 - 3 * (eta21 + eta03) ** 2)
        + (3 * eta21 - eta03) * (eta21 + eta03) * (3 * (eta30 + eta12) ** 2 - (eta21 + eta03) ** 2)
    )
    h6 = (
        (eta20 - eta02) * ((eta30 + eta12) ** 2 - (eta21 + eta03) ** 2)
        + 4 * eta11 * (eta30 + eta12) * (eta21 + eta03)
    )
    h7 = (
        (3 * eta21 - eta03) * (eta30 + eta12) * ((eta30 + eta12) ** 2 - 3 * (eta21 + eta03) ** 2)
        - (eta30 - 3 * eta12) * (eta21 + eta03) * (3 * (eta30 + eta12) ** 2 - (eta21 + eta03) ** 2)
    )

    # Clamp near-zero values and use absolute value (sign can vary with rotation)
    moments = [h1, h2, h3, h4, h5, h6, h7]
    clamped = [abs(m) if abs(m) > _EPSILON else 0.0 for m in moments]
    return clamped


def _hu_log_distance(m1: list[float], m2: list[float]) -> float:
    """Compute log-scaled distance between two sets of 7 Hu moments.

    Distance = sum |log(|m1_i| + 1e-12) - log(|m2_i| + 1e-12)| for i = 1..7
    Lower = more similar.
    """
    dist = 0.0
    for a, b in zip(m1, m2):
        log_a = math.log(abs(a) + 1e-12)
        log_b = math.log(abs(b) + 1e-12)
        dist += abs(log_a - log_b)
    return dist


def _convex_hull(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Compute convex hull using Andrew's monotone chain algorithm.

    Returns hull points in counter-clockwise order, without collinear points.
    Returns empty list for < 3 points.
    """
    if len(points) < 3:
        return []

    pts = sorted(set(points))

    def cross(o: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> int:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[int, int]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[int, int]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _compute_shape_features(
    cells: list[tuple[int, int]],
    contour: list[tuple[int, int]],
    perimeter: int,
    bbox: tuple[int, int, int, int],
) -> dict[str, float]:
    """Compute structural shape features.

    Returns dict with: aspect_ratio, compactness, solidity, extent.
    """
    features: dict[str, float] = {}
    area = float(len(cells))
    x1, y1, x2, y2 = bbox
    bbox_w = float(max(1, x2 - x1 + 1))
    bbox_h = float(max(1, y2 - y1 + 1))

    # Aspect ratio
    features["aspect_ratio"] = bbox_w / bbox_h if bbox_h > _EPSILON else 0.0

    # Extent = area / bbox area
    bbox_area = bbox_w * bbox_h
    features["extent"] = area / bbox_area if bbox_area > _EPSILON else 0.0

    # Compactness = perimeter² / area (circle ≈ 12.6)
    if area > _EPSILON:
        features["compactness"] = (float(perimeter) ** 2) / area
    else:
        features["compactness"] = 0.0

    # Convex hull derived features
    hull = _convex_hull(contour if len(contour) >= 3 else cells)
    if len(hull) >= 3:
        hull_perimeter = _compute_perimeter(hull)
        hull_area = _compute_hull_area(hull)
        # Solidity = area / convex hull area
        features["solidity"] = area / hull_area if hull_area > _EPSILON else 1.0
        # Convexity = convex hull perimeter / actual perimeter
        features["convexity"] = hull_perimeter / float(perimeter) if perimeter > 0 else 1.0
    else:
        features["solidity"] = 1.0
        features["convexity"] = 1.0

    return features


def _compute_hull_area(hull: list[tuple[int, int]]) -> float:
    """Compute area of a convex polygon in pixel units.

    Counts integer-coordinate pixels inside or on the hull polygon.
    """
    if len(hull) < 3:
        return 0.0
    x1 = min(x for x, _ in hull)
    y1 = min(y for _, y in hull)
    x2 = max(x for x, _ in hull)
    y2 = max(y for _, y in hull)

    count = 0.0
    for y in range(y1, y2 + 1):
        for x in range(x1, x2 + 1):
            if _point_in_convex_polygon((x, y), hull):
                count += 1.0
    return count


def _point_in_convex_polygon(point: tuple[int, int], hull: list[tuple[int, int]]) -> bool:
    """Test if point is inside or on a convex polygon using cross product signs."""
    x, y = point
    n = len(hull)
    sign: bool | None = None
    for i in range(n):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % n]
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if abs(cross) < _EPSILON:
            continue
        if sign is None:
            sign = cross > 0
        elif (cross > 0) != sign:
            return False
    return True


def match_shape_to_template(
    cells: list[tuple[int, int]],
    template_hu_moments: list[float],
    threshold: float = 0.1,
) -> tuple[float, float]:
    """Compare a shape against a template using Hu moment log-distance.

    Args:
        cells: The shape's pixel coordinates.
        template_hu_moments: 7 Hu moments of the template.
        threshold: Distance threshold below which shapes are considered matching.

    Returns:
        (similarity_score, rotation_estimate)
        similarity_score in [0, 1] where 1.0 = identical, 0.0 = very different.
    """
    shape_hu = _compute_hu_moments(cells)
    hu_dist = _hu_log_distance(shape_hu, template_hu_moments)
    # Convert distance to similarity: 1 / (1 + dist)
    similarity = 1.0 / (1.0 + hu_dist)
    rotation = _estimate_rotation_angle(cells)
    return similarity, rotation


def find_best_template_match(
    cells: list[tuple[int, int]],
    templates: list[dict[str, Any]],
    min_score: float = 0.7,
) -> tuple[int | None, float]:
    """Find the best matching template for a shape from a list of templates.

    Args:
        cells: The shape's pixel coordinates.
        templates: List of template dicts, each must have "hu_moments" key.
        min_score: Minimum similarity score required.

    Returns:
        (best_template_index, similarity_score) or (None, 0.0) if no match found.
    """
    if not templates or not cells:
        return None, 0.0

    best_idx: int | None = None
    best_score = 0.0

    for i, template in enumerate(templates):
        tm = template.get("hu_moments")
        if tm is None:
            continue
        score, _ = match_shape_to_template(cells, tm, threshold=0.1)
        if score > best_score and score >= min_score:
            best_score = score
            best_idx = i

    return best_idx, best_score


# ---------------------------------------------------------------------------
# Iteration 7: Template storage from solved puzzles
# ---------------------------------------------------------------------------

_SHAPE_TEMPLATES_PATH = Path("shape_templates.json")


def store_shape_template(
    cells: list[tuple[int, int]],
    hu_moments: list[float] | None = None,
    features: dict[str, float] | None = None,
    placement: dict[str, Any] | None = None,
    goal_coverage: list[tuple[int, int]] | None = None,
    filepath: str | Path | None = None,
) -> None:
    """Store a shape template to a JSON file.

    Each template stores the shape's Hu moments, features, and optional
    placement metadata (mirror orientation, pixel target, goal coverage).
    """
    if hu_moments is None:
        hu_moments = _compute_hu_moments(cells)
    path = Path(filepath) if filepath else _SHAPE_TEMPLATES_PATH
    templates = load_shape_templates(filepath)
    template = {
        "hu_moments": hu_moments,
        "features": features or {},
        "placement": placement or {},
        "goal_coverage": sorted(goal_coverage) if goal_coverage else [],
        "cell_count": len(cells),
        "normalized_mask": _normalized_mask(cells),
    }
    # Check for near-duplicate (same Hu moments, close enough)
    for existing in templates:
        existing_hu = existing.get("hu_moments")
        if existing_hu and _hu_log_distance(hu_moments, existing_hu) < 0.5:
            # Merge: keep the one with more complete data
            if placement and not existing.get("placement"):
                existing["placement"] = placement
            if goal_coverage and not existing.get("goal_coverage"):
                existing["goal_coverage"] = sorted(goal_coverage)
            if features and not existing.get("features"):
                existing["features"] = features
            _write_templates(templates, path)
            return
    templates.append(template)
    _write_templates(templates, path)


def _write_templates(templates: list[dict[str, Any]], path: Path) -> None:
    """Write templates list to JSON file."""
    try:
        with open(path, "w") as f:
            json.dump(templates, f, indent=2)
    except (OSError, PermissionError):
        pass


def load_shape_templates(filepath: str | Path | None = None) -> list[dict[str, Any]]:
    """Load shape templates from a JSON file.

    Returns empty list if file doesn't exist or is corrupt.
    """
    path = Path(filepath) if filepath else _SHAPE_TEMPLATES_PATH
    if not path.exists():
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, OSError):
        return []


# ---------------------------------------------------------------------------
# Iteration 8: Template-aware mirror placement
# ---------------------------------------------------------------------------

def _template_aware_placement(
    component: _ColorComponent,
    remaining_goals: set[tuple[int, int]],
    goal_cluster_map: dict[str, set[tuple[int, int]]],
    horizontal: bool,
    axis: int,
    *,
    width: int,
    height: int,
    templates: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Try template-aware placement before brute-force search.

    If a shape matches a stored template with the same mirror orientation,
    use the template's cached target. Otherwise returns None to fall back
    to brute-force.
    """
    if not templates:
        return None

    cells = component.cells
    best_idx, score = find_best_template_match(cells, templates, min_score=0.8)
    if best_idx is None:
        return None

    template = templates[best_idx]
    tp = template.get("placement", {})
    template_horiz = tp.get("horizontal")
    if template_horiz is not None and template_horiz != horizontal:
        return None  # mirror orientation mismatch
    return {
        "template_index": best_idx,
        "score": score,
        "target_anchor": tp.get("target_anchor"),
        "placement": tp,
    }


# ---------------------------------------------------------------------------
# Iteration 9: Partial overlap / fuzzy mask matching
# ---------------------------------------------------------------------------

def _mask_overlap_score(
    cells1: list[tuple[int, int]],
    cells2: list[tuple[int, int]],
) -> float:
    """Compute IoU (Intersection over Union) between two masks.

    Returns value in [0, 1] where 1.0 = identical masks.
    No alignment — masks are compared at their raw positions.
    Use this to detect the same shape appearing at the same location.
    """
    if not cells1 or not cells2:
        return 0.0

    set_a = set(cells1)
    set_b = set(cells2)

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return intersection / union


# ---------------------------------------------------------------------------
# Iteration 10: Spatial relationship descriptors
# ---------------------------------------------------------------------------

def _compute_spatial_descriptors(
    cells: list[tuple[int, int]],
    mirror_cells: list[tuple[int, int]] | None = None,
    goal_cells: list[tuple[int, int]] | None = None,
    other_shapes_cells: list[list[tuple[int, int]]] | None = None,
) -> dict[str, Any]:
    """Compute spatial relationship descriptors for a shape.

    Describes the shape's position relative to mirror, goals, and other shapes.
    """
    descriptors: dict[str, Any] = {}
    shape_center = _center(cells)

    if mirror_cells:
        mirror_center = _center(mirror_cells)
        descriptors["mirror_distance"] = abs(shape_center[0] - mirror_center[0]) + abs(shape_center[1] - mirror_center[1])
        descriptors["mirror_side"] = "left" if shape_center[0] < mirror_center[0] else "right"

    if goal_cells:
        goal_centers = _center(goal_cells)
        descriptors["goal_distance"] = abs(shape_center[0] - goal_centers[0]) + abs(shape_center[1] - goal_centers[1])
        # Distance to nearest goal
        min_gdist = min(abs(shape_center[0] - gx) + abs(shape_center[1] - gy) for gx, gy in goal_cells)
        descriptors["nearest_goal_distance"] = min_gdist

    if other_shapes_cells:
        other_centers = [_center(s) for s in other_shapes_cells]
        descriptors["other_shape_count"] = len(other_shapes_cells)
        if other_centers:
            min_sdist = min(
                abs(shape_center[0] - cx) + abs(shape_center[1] - cy)
                for cx, cy in other_centers
            )
            descriptors["nearest_shape_distance"] = min_sdist

    return descriptors


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
