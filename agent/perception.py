"""Dependency-free object extraction and consecutive-frame differencing."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Sequence
from hashlib import blake2b
from math import hypot
from typing import Any

from .state import (
    ContactEvent,
    ObjectFeature,
    ObjectMotion,
    StateDiff,
    StructuredState,
    abstract_fingerprint,
    stable_fingerprint,
)


def latest_grid(frame: Any) -> tuple[tuple[int, ...], ...]:
    """Return the final animation grid from FrameData or a raw 2D/3D value."""
    raw = getattr(frame, "frame", frame)
    if raw is None or len(raw) == 0:
        return ()
    # FrameData.frame is normally [animation_frame][row][column]. Tests and
    # callers may pass a single [row][column] grid directly.
    first = raw[0]
    grid = raw if len(first) > 0 and isinstance(first[0], (int, float)) else raw[-1]
    rows = tuple(tuple(int(value) for value in row) for row in grid)
    if rows and any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("ARC frame rows must have equal width")
    return rows


def infer_background(grid: Sequence[Sequence[int]]) -> int:
    """Infer background by frequency, breaking ties using border support."""
    if not grid or not grid[0]:
        return 0
    counts = Counter(value for row in grid for value in row)
    border = Counter(grid[0]) + Counter(grid[-1])
    for row in grid[1:-1]:
        border[row[0]] += 1
        border[row[-1]] += 1
    return max(counts, key=lambda color: (counts[color], border[color], -color))


def _component(
    grid: Sequence[Sequence[int]], start: tuple[int, int], seen: set[tuple[int, int]]
) -> frozenset[tuple[int, int]]:
    height, width = len(grid), len(grid[0])
    color = grid[start[1]][start[0]]
    queue = deque([start])
    seen.add(start)
    cells: set[tuple[int, int]] = set()
    while queue:
        x, y = queue.popleft()
        cells.add((x, y))
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if (
                0 <= nx < width
                and 0 <= ny < height
                and (nx, ny) not in seen
                and grid[ny][nx] == color
            ):
                seen.add((nx, ny))
                queue.append((nx, ny))
    return frozenset(cells)


def extract_objects(
    grid: Sequence[Sequence[int]], background: int | None = None
) -> tuple[ObjectFeature, ...]:
    """Extract 4-connected, same-colour non-background components."""
    if not grid or not grid[0]:
        return ()
    height, width = len(grid), len(grid[0])
    bg = infer_background(grid) if background is None else background
    seen: set[tuple[int, int]] = set()
    objects: list[ObjectFeature] = []
    for y, row in enumerate(grid):
        for x, color in enumerate(row):
            if color == bg or (x, y) in seen:
                continue
            cells = _component(grid, (x, y), seen)
            xs = [cell[0] for cell in cells]
            ys = [cell[1] for cell in cells]
            min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
            shape = tuple(sorted((cx - min_x, cy - min_y) for cx, cy in cells))
            shape_digest = blake2b(repr(shape).encode(), digest_size=2).hexdigest()
            object_id = f"c{color}:{min_x},{min_y}:{len(cells)}:{shape_digest}"
            objects.append(
                ObjectFeature(
                    object_id=object_id,
                    color=int(color),
                    cells=cells,
                    bbox=(min_x, min_y, max_x, max_y),
                    centroid=(sum(xs) / len(xs), sum(ys) / len(ys)),
                    size=len(cells),
                    shape_signature=shape,
                    touches_border=(
                        min_x == 0 or min_y == 0 or max_x == width - 1 or max_y == height - 1
                    ),
                )
            )
    return tuple(sorted(objects, key=lambda obj: (obj.color, obj.bbox, obj.size)))


def perceive(frame: Any) -> StructuredState:
    grid = latest_grid(frame)
    height = len(grid)
    width = len(grid[0]) if grid else 0
    background = infer_background(grid)
    histogram = tuple(sorted(Counter(v for row in grid for v in row).items()))
    objects = extract_objects(grid, background)
    levels_completed = int(getattr(frame, "levels_completed", 0) or 0)
    return StructuredState(
        width=width,
        height=height,
        background=background,
        objects=objects,
        color_histogram=histogram,
        raw_fingerprint=stable_fingerprint(grid, background),
        fingerprint=abstract_fingerprint(
            objects,
            width=width,
            height=height,
            background=background,
            levels_completed=levels_completed,
        ),
        grid=grid,
    )


def _match_objects(
    before: tuple[ObjectFeature, ...], after: tuple[ObjectFeature, ...]
) -> tuple[list[tuple[ObjectFeature, ObjectFeature]], list[ObjectFeature], list[ObjectFeature]]:
    """Greedily match components using colour, shape, overlap, and proximity."""
    candidates: list[tuple[float, int, int]] = []
    for i, old in enumerate(before):
        for j, new in enumerate(after):
            overlap = len(old.cells & new.cells) / max(old.size, new.size)
            same_shape = old.shape_signature == new.shape_signature
            same_color = old.color == new.color
            if not same_color and overlap == 0:
                continue
            distance = hypot(
                old.centroid[0] - new.centroid[0], old.centroid[1] - new.centroid[1]
            )
            size_delta = abs(old.size - new.size) / max(old.size, new.size)
            score = (
                4.0 * overlap
                + 2.0 * same_shape
                + 1.5 * same_color
                - 0.08 * distance
                - size_delta
            )
            if score > 0.25:
                candidates.append((score, i, j))
    matched_old: set[int] = set()
    matched_new: set[int] = set()
    matches: list[tuple[ObjectFeature, ObjectFeature]] = []
    for _, i, j in sorted(candidates, reverse=True):
        if i not in matched_old and j not in matched_new:
            matched_old.add(i)
            matched_new.add(j)
            matches.append((before[i], after[j]))
    disappeared = [obj for i, obj in enumerate(before) if i not in matched_old]
    appeared = [obj for j, obj in enumerate(after) if j not in matched_new]
    return matches, disappeared, appeared


def _adjacent(first: ObjectFeature, second: ObjectFeature) -> bool:
    if first.cells & second.cells:
        return True
    second_cells = second.cells
    return any(
        (x + dx, y + dy) in second_cells
        for x, y in first.cells
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))
    )


def diff_states(
    before: StructuredState,
    after: StructuredState,
    *,
    progress_delta: int = 0,
    terminal: str = "NOT_FINISHED",
) -> StateDiff:
    if (before.width, before.height) != (after.width, after.height):
        changed = frozenset(
            (x, y) for y, row in enumerate(after.grid) for x, _ in enumerate(row)
        )
    else:
        changed = frozenset(
            (x, y)
            for y in range(after.height)
            for x in range(after.width)
            if before.grid[y][x] != after.grid[y][x]
        )
    changed_bbox = None
    if changed:
        xs, ys = zip(*changed, strict=False)
        changed_bbox = (min(xs), min(ys), max(xs), max(ys))

    matches, disappeared, appeared = _match_objects(before.objects, after.objects)
    motions: list[ObjectMotion] = []
    transformed: list[tuple[str, str]] = []
    for old, new in matches:
        dx = round(new.centroid[0] - old.centroid[0])
        dy = round(new.centroid[1] - old.centroid[1])
        if (dx or dy) and old.shape_signature == new.shape_signature:
            motions.append(
                ObjectMotion(
                    old.object_id,
                    new.object_id,
                    new.color,
                    (dx, dy),
                    old.centroid,
                    new.centroid,
                )
            )
        if old.shape_signature != new.shape_signature or old.color != new.color:
            transformed.append((old.object_id, new.object_id))

    contacts: list[ContactEvent] = []
    old_by_new = {new.object_id: old for old, new in matches}
    for moving in motions:
        moved = next(obj for obj in after.objects if obj.object_id == moving.after_id)
        old_moved = old_by_new[moved.object_id]
        for other in after.objects:
            if other.object_id == moved.object_id or not _adjacent(moved, other):
                continue
            previous_other = old_by_new.get(other.object_id)
            if previous_other is None or not _adjacent(old_moved, previous_other):
                contacts.append(ContactEvent(moved.object_id, other.object_id, "new-adjacency"))

    return StateDiff(
        changed_cells=changed,
        changed_bbox=changed_bbox,
        appeared=tuple(appeared),
        disappeared=tuple(disappeared),
        motions=tuple(motions),
        transformed=tuple(transformed),
        contacts=tuple(contacts),
        progress_delta=progress_delta,
        terminal=terminal,
    )
