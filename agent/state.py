"""Compact object-centric state and transition types."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from hashlib import blake2b

Cell = tuple[int, int]
BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class ObjectFeature:
    object_id: str
    color: int
    cells: frozenset[Cell]
    bbox: BBox
    centroid: tuple[float, float]
    size: int
    shape_signature: tuple[Cell, ...]
    touches_border: bool = False

    def compact(self) -> dict[str, object]:
        return {
            "id": self.object_id,
            "color": self.color,
            "bbox": self.bbox,
            "centroid": tuple(round(v, 2) for v in self.centroid),
            "size": self.size,
            "shape": self.shape_signature,
        }


@dataclass(frozen=True)
class ObjectMotion:
    before_id: str
    after_id: str
    color: int
    delta: tuple[int, int]
    before_centroid: tuple[float, float]
    after_centroid: tuple[float, float]


@dataclass(frozen=True)
class ContactEvent:
    first_id: str
    second_id: str
    kind: str


@dataclass(frozen=True)
class StructuredState:
    width: int
    height: int
    background: int
    objects: tuple[ObjectFeature, ...]
    color_histogram: tuple[tuple[int, int], ...]
    raw_fingerprint: str
    fingerprint: str
    grid: tuple[tuple[int, ...], ...] = field(repr=False, compare=False)
    abstraction: str = "interior-objects-v1"

    def compact(self) -> dict[str, object]:
        return {
            "fingerprint": self.fingerprint,
            "raw_fingerprint": self.raw_fingerprint,
            "abstraction": self.abstraction,
            "size": [self.width, self.height],
            "background": self.background,
            "objects": [obj.compact() for obj in self.objects],
        }


@dataclass(frozen=True)
class StateDiff:
    changed_cells: frozenset[Cell] = frozenset()
    changed_bbox: BBox | None = None
    appeared: tuple[ObjectFeature, ...] = ()
    disappeared: tuple[ObjectFeature, ...] = ()
    motions: tuple[ObjectMotion, ...] = ()
    transformed: tuple[tuple[str, str], ...] = ()
    contacts: tuple[ContactEvent, ...] = ()
    progress_delta: int = 0
    terminal: str = "NOT_FINISHED"

    @property
    def changed(self) -> bool:
        return bool(
            self.changed_cells
            or self.appeared
            or self.disappeared
            or self.motions
            or self.transformed
            or self.progress_delta
        )

    @property
    def affected_colors(self) -> tuple[int, ...]:
        colors = {o.color for o in self.appeared + self.disappeared}
        colors.update(m.color for m in self.motions)
        return tuple(sorted(colors))

    def outcome_signature(self) -> tuple[object, ...]:
        motion = tuple(sorted(m.delta for m in self.motions))
        return (
            bool(self.changed_cells),
            self.progress_delta,
            self.terminal,
            motion,
            len(self.appeared),
            len(self.disappeared),
            self.affected_colors,
        )

    def compact(self) -> dict[str, object]:
        return {
            "changed_cells": len(self.changed_cells),
            "changed_bbox": self.changed_bbox,
            "appeared": [o.object_id for o in self.appeared],
            "disappeared": [o.object_id for o in self.disappeared],
            "motions": [{"object": m.before_id, "delta": m.delta} for m in self.motions],
            "transformed": list(self.transformed),
            "contacts": [c.__dict__ for c in self.contacts],
            "progress_delta": self.progress_delta,
            "terminal": self.terminal,
        }


def stable_fingerprint(rows: Iterable[Iterable[int]], background: int) -> str:
    digest = blake2b(digest_size=10)
    digest.update(bytes([background]))
    for row in rows:
        digest.update(bytes(row))
    return digest.hexdigest()


def abstract_fingerprint(
    objects: Iterable[ObjectFeature],
    *,
    width: int,
    height: int,
    background: int,
    levels_completed: int = 0,
) -> str:
    """Hash task-relevant interior objects while ignoring likely border UI.

    Border-connected components remain available to perception and coordinate
    generation; they are excluded only from the learning key. Pairwise spatial
    relations make the key object-centric without discarding exact positions.
    """
    interior = sorted(
        (obj for obj in objects if not obj.touches_border),
        key=lambda obj: (obj.color, obj.bbox, obj.size, obj.shape_signature),
    )
    descriptors = tuple(
        (obj.color, obj.size, obj.bbox, obj.shape_signature) for obj in interior
    )
    relations = []
    for index, first in enumerate(interior):
        for second in interior[index + 1 :]:
            dx = (
                -1
                if first.centroid[0] < second.centroid[0]
                else int(first.centroid[0] > second.centroid[0])
            )
            dy = (
                -1
                if first.centroid[1] < second.centroid[1]
                else int(first.centroid[1] > second.centroid[1])
            )
            relations.append((first.color, second.color, dx, dy))
    payload = (
        "interior-objects-v1",
        width,
        height,
        background,
        levels_completed,
        descriptors,
        tuple(relations),
    )
    return blake2b(repr(payload).encode(), digest_size=10).hexdigest()
