"""
Scene Graph Module for HyPrune.

This module converts raw ARC grids (2D numpy arrays of ints 0-9) into structured object representations.
"""

from __future__ import annotations
import dataclasses
import hashlib
from collections import deque
from typing import Optional

import numpy as np


@dataclasses.dataclass
class ArcObject:
    id: int
    pixels: frozenset[tuple[int, int]]  # (row, col) coordinates
    color: int  # ARC color 0-9
    bounding_box: tuple[int, int, int, int]  # (r_min, c_min, r_max, c_max)
    shape_hash: str  # Canonical shape signature (translation-invariant)
    symmetry: dict  # {horizontal: bool, vertical: bool, rotational_90: bool, rotational_180: bool}
    is_background: bool

    @property
    def width(self) -> int:
        return self.bounding_box[3] - self.bounding_box[1] + 1

    @property
    def height(self) -> int:
        return self.bounding_box[2] - self.bounding_box[0] + 1

    @property
    def area(self) -> int:
        return len(self.pixels)

    @property
    def centroid(self) -> tuple[float, float]:
        r_sum = sum(p[0] for p in self.pixels)
        c_sum = sum(p[1] for p in self.pixels)
        return (r_sum / self.area, c_sum / self.area)

    def as_mask(self, grid_shape: tuple[int, int]) -> np.ndarray:
        """Return boolean mask of this object on a grid."""
        mask = np.zeros(grid_shape, dtype=bool)
        for r, c in self.pixels:
            if 0 <= r < grid_shape[0] and 0 <= c < grid_shape[1]:
                mask[r, c] = True
        return mask

    def normalized_shape(self) -> np.ndarray:
        """Translation-invariant shape representation (cropped to bounding box)."""
        shape = np.zeros((self.height, self.width), dtype=bool)
        r_min, c_min, _, _ = self.bounding_box
        for r, c in self.pixels:
            shape[r - r_min, c - c_min] = True
        return shape


@dataclasses.dataclass
class SpatialRelation:
    obj1_id: int
    relation: str  # 'above', 'below', 'left_of', 'right_of', 'adjacent', 'contains', 'inside', 'overlaps'
    obj2_id: int


@dataclasses.dataclass
class SceneGraph:
    objects: list[ArcObject]
    grid_size: tuple[int, int]  # (rows, cols)
    background_color: int
    spatial_relations: list[SpatialRelation]
    color_histogram: dict[int, int]
    grid_symmetry: dict  # Grid-level symmetry

    def get_object(self, obj_id: int) -> Optional[ArcObject]:
        for obj in self.objects:
            if obj.id == obj_id:
                return obj
        return None

    def objects_by_color(self, color: int) -> list[ArcObject]:
        return [obj for obj in self.objects if obj.color == color]

    def non_background_objects(self) -> list[ArcObject]:
        return [obj for obj in self.objects if not obj.is_background]

    def adjacency_matrix(self) -> np.ndarray:
        num_objs = len(self.objects)
        adj = np.zeros((num_objs, num_objs), dtype=bool)
        id_to_idx = {obj.id: i for i, obj in enumerate(self.objects)}
        for rel in self.spatial_relations:
            if rel.relation in ('adjacent', 'overlaps'):
                idx1 = id_to_idx.get(rel.obj1_id)
                idx2 = id_to_idx.get(rel.obj2_id)
                if idx1 is not None and idx2 is not None:
                    adj[idx1, idx2] = True
                    adj[idx2, idx1] = True
        return adj


class ObjectParser:
    """Parses ARC grids into SceneGraph representations."""

    def __init__(self, connectivity: int = 4, min_object_size: int = 1):
        if connectivity not in (4, 8):
            raise ValueError("Connectivity must be 4 or 8")
        self.connectivity = connectivity
        self.min_object_size = min_object_size

    def parse(self, grid: np.ndarray) -> SceneGraph:
        """Parse a grid into a full SceneGraph."""
        if grid.size == 0:
            return SceneGraph(
                objects=[],
                grid_size=grid.shape,
                background_color=0,
                spatial_relations=[],
                color_histogram={},
                grid_symmetry={
                    "horizontal": False,
                    "vertical": False,
                    "diagonal_main": False,
                    "diagonal_anti": False,
                },
            )

        background_color = self._find_background_color(grid)

        vals, counts = np.unique(grid, return_counts=True)
        color_histogram = {int(v): int(c) for v, c in zip(vals, counts)}

        objects = self._extract_components(grid, background_color)
        # Compute spatial relations only on non-background objects to keep parsing ultra-fast
        non_bg_objects = [o for o in objects if not o.is_background][:15]
        spatial_relations = self._compute_spatial_relations(non_bg_objects)
        grid_symmetry = self._detect_grid_symmetry(grid)

        return SceneGraph(
            objects=objects,
            grid_size=grid.shape,
            background_color=background_color,
            spatial_relations=spatial_relations,
            color_histogram=color_histogram,
            grid_symmetry=grid_symmetry,
        )

    def _extract_components(self, grid: np.ndarray, background_color: int) -> list[ArcObject]:
        """BFS-based connected component extraction."""
        rows, cols = grid.shape
        visited = np.zeros((rows, cols), dtype=bool)
        objects = []
        obj_id = 0

        if self.connectivity == 4:
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for r in range(rows):
            for c in range(cols):
                if not visited[r, c]:
                    color = int(grid[r, c])
                    pixels = []
                    q = deque([(r, c)])
                    visited[r, c] = True

                    while q:
                        cr, cc = q.popleft()
                        pixels.append((cr, cc))
                        for dr, dc in dirs:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                if not visited[nr, nc] and grid[nr, nc] == color:
                                    visited[nr, nc] = True
                                    q.append((nr, nc))

                    if len(pixels) >= self.min_object_size:
                        frozen_pixels = frozenset(pixels)
                        r_min = min(p[0] for p in pixels)
                        r_max = max(p[0] for p in pixels)
                        c_min = min(p[1] for p in pixels)
                        c_max = max(p[1] for p in pixels)
                        bbox = (r_min, c_min, r_max, c_max)
                        shape_hash = self._compute_shape_hash(frozen_pixels)

                        # Create temporary object to compute symmetry
                        temp_obj = ArcObject(
                            id=obj_id,
                            pixels=frozen_pixels,
                            color=color,
                            bounding_box=bbox,
                            shape_hash=shape_hash,
                            symmetry={},
                            is_background=(color == background_color),
                        )
                        temp_obj.symmetry = self._detect_symmetry(temp_obj)

                        objects.append(temp_obj)
                        obj_id += 1

        return objects

    def _compute_shape_hash(self, pixels: frozenset[tuple[int, int]]) -> str:
        """Translation-invariant shape hash."""
        if not pixels:
            return ""
        r_min = min(p[0] for p in pixels)
        c_min = min(p[1] for p in pixels)
        normalized = sorted([(p[0] - r_min, p[1] - c_min) for p in pixels])
        hash_str = str(normalized).encode('utf-8')
        return hashlib.md5(hash_str).hexdigest()

    def _detect_symmetry(self, obj: ArcObject) -> dict:
        """Check horizontal, vertical, 90° and 180° rotational symmetry."""
        shape = obj.normalized_shape()
        symmetry = {
            "horizontal": False,
            "vertical": False,
            "rotational_90": False,
            "rotational_180": False,
        }

        if shape.size == 0:
            return symmetry

        symmetry["horizontal"] = np.array_equal(shape, np.flipud(shape))
        symmetry["vertical"] = np.array_equal(shape, np.fliplr(shape))

        rot90 = np.rot90(shape)
        if shape.shape == rot90.shape:
            symmetry["rotational_90"] = np.array_equal(shape, rot90)

        rot180 = np.rot90(shape, 2)
        if shape.shape == rot180.shape:
            symmetry["rotational_180"] = np.array_equal(shape, rot180)

        return symmetry

    def _compute_spatial_relations(self, objects: list[ArcObject]) -> list[SpatialRelation]:
        """Compute pairwise spatial relations between objects."""
        relations = []
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i == j:
                    continue

                r1_min, c1_min, r1_max, c1_max = obj1.bounding_box
                r2_min, c2_min, r2_max, c2_max = obj2.bounding_box

                # Above / Below
                if r1_max < r2_min:
                    relations.append(SpatialRelation(obj1.id, "above", obj2.id))
                elif r1_min > r2_max:
                    relations.append(SpatialRelation(obj1.id, "below", obj2.id))

                # Left / Right
                if c1_max < c2_min:
                    relations.append(SpatialRelation(obj1.id, "left_of", obj2.id))
                elif c1_min > c2_max:
                    relations.append(SpatialRelation(obj1.id, "right_of", obj2.id))

                # Contains (obj1 fully contains obj2 bbox)
                if r1_min <= r2_min and r1_max >= r2_max and c1_min <= c2_min and c1_max >= c2_max:
                    relations.append(SpatialRelation(obj1.id, "contains", obj2.id))
                    relations.append(SpatialRelation(obj2.id, "inside", obj1.id))

                # Overlaps (bounding box overlap)
                if not (r1_max < r2_min or r1_min > r2_max or c1_max < c2_min or c1_min > c2_max):
                    # Check pixel overlap for strict overlapping (excluding same object which is avoided by i==j)
                    # For ARC grids objects usually don't have overlapping pixels if they are extracted disjointly,
                    # but we can record bounding box overlap if needed. Skipping for now as it's not strictly specified, 
                    # except maybe "overlaps". We'll just define adjacency below.
                    pass

                # Adjacent
                adjacent = False
                for p1 in obj1.pixels:
                    for p2 in obj2.pixels:
                        if abs(p1[0] - p2[0]) <= 1 and abs(p1[1] - p2[1]) <= 1:
                            adjacent = True
                            break
                    if adjacent:
                        break
                if adjacent:
                    relations.append(SpatialRelation(obj1.id, "adjacent", obj2.id))

        return relations

    def _detect_grid_symmetry(self, grid: np.ndarray) -> dict:
        """Check grid-level symmetry properties."""
        sym = {
            "horizontal": False,
            "vertical": False,
            "diagonal_main": False,
            "diagonal_anti": False,
        }
        if grid.size == 0:
            return sym

        sym["horizontal"] = np.array_equal(grid, np.flipud(grid))
        sym["vertical"] = np.array_equal(grid, np.fliplr(grid))

        if grid.shape[0] == grid.shape[1]:
            sym["diagonal_main"] = np.array_equal(grid, grid.T)
            # Anti-diagonal symmetry: equivalent to rotating 90, flipping, etc.
            # Easiest way: flip left-right, then transpose, then flip left-right
            sym["diagonal_anti"] = np.array_equal(grid, np.rot90(np.fliplr(grid)))

        return sym

    @staticmethod
    def _find_background_color(grid: np.ndarray) -> int:
        """Most frequent color in the grid."""
        if grid.size == 0:
            return 0
        unique, counts = np.unique(grid, return_counts=True)
        return int(unique[np.argmax(counts)])
