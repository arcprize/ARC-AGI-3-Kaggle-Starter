"""Temporal entity tracking across frames.

Tracks components detected by ``build_game_model`` across consecutive frames,
providing persistent entity IDs, velocity estimation, and lifecycle management.
"""

from __future__ import annotations

from typing import Any

from agi_core.orch_or.game_model import EntityTrack

_MAX_HISTORY = 16
_STALE_FRAMES = 30
_MATCH_THRESHOLD = 20


class EntityTracker:
    """Matches components across frames and maintains persistent tracks.

    Usage::

        tracker = EntityTracker()
        for each frame:
            tracks = tracker.update(components, frame_index)
            # tracks now have persistent entity_ids
    """

    def __init__(self) -> None:
        self._tracks: dict[int, EntityTrack] = {}
        self._next_id: int = 1
        self._frame_cursor: int = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._frame_cursor = 0

    def update(
        self,
        components: list[Any],
        frame: int | None = None,
    ) -> list[EntityTrack]:
        """Match *components* (from ``build_game_model``) to existing tracks.

        Returns a list of ``EntityTrack`` objects for this frame, ordered
        to match the input *components* list (``tracks[i]`` corresponds to
        ``components[i]``, or *None* if no match was found).
        """
        current_frame = self._frame_cursor if frame is None else max(0, int(frame))
        if frame is None:
            self._frame_cursor = current_frame + 1
        else:
            self._frame_cursor = max(self._frame_cursor, current_frame + 1)

        if not components:
            # No components this frame — mark all tracks stale
            stale: list[int] = []
            for eid, track in self._tracks.items():
                if current_frame - track.last_seen > _STALE_FRAMES:
                    stale.append(eid)
                else:
                    track.moved_this_frame = False
                    track.stationary_streak += 1
            for eid in stale:
                del self._tracks[eid]
            return []

        # Build centroid list for input components
        comp_centers = [_component_center(c) for c in components]
        comp_colors = [_component_color(c) for c in components]

        # Greedy matching: assign each component to best available track
        available_tracks = dict(self._tracks)
        used_tracks: set[int] = set()
        result: list[EntityTrack | None] = [None] * len(components)

        # Sort by track age so older tracks get first pick
        sorted_track_ids = sorted(
            available_tracks.keys(),
            key=lambda eid: (-available_tracks[eid].age, -available_tracks[eid].last_seen),
        )

        for ci, (center, color) in enumerate(zip(comp_centers, comp_colors)):
            best_id: int | None = None
            best_cost = _MATCH_THRESHOLD + 1

            for eid in sorted_track_ids:
                if eid in used_tracks:
                    continue
                track = available_tracks[eid]
                if not track.positions:
                    continue
                cost = abs(track.color - color) * 100 + _manhattan(center, track.positions[-1])
                if cost < best_cost:
                    best_cost = cost
                    best_id = eid

            if best_id is not None:
                used_tracks.add(best_id)
                track = available_tracks[best_id]
                self._update_track(track, color, center, _component_bbox(components[ci]), current_frame)
                result[ci] = track
            else:
                # New track
                eid = self._next_id
                self._next_id += 1
                track = EntityTrack(entity_id=eid, color=color)
                self._update_track(track, color, center, _component_bbox(components[ci]), current_frame)
                self._tracks[eid] = track
                used_tracks.add(eid)
                result[ci] = track

        # Mark unmatched existing tracks as stale
        for eid, track in list(self._tracks.items()):
            if eid not in used_tracks:
                track.moved_this_frame = False
                track.stationary_streak += 1
                if current_frame - track.last_seen > _STALE_FRAMES:
                    del self._tracks[eid]

        return [r for r in result if r is not None]

    def _update_track(
        self,
        track: EntityTrack,
        color: int,
        center: tuple[int, int],
        bbox: tuple[int, int, int, int],
        frame: int,
    ) -> None:
        """Update a single track with a new observation."""
        # Update velocity
        if track.positions:
            prev = track.positions[-1]
            raw_vx = center[0] - prev[0]
            raw_vy = center[1] - prev[1]
            vx = int(0.5 * track.velocity[0] + 0.5 * raw_vx)
            vy = int(0.5 * track.velocity[1] + 0.5 * raw_vy)
            track.velocity = (vx, vy)
            moved = raw_vx != 0 or raw_vy != 0
            track.moved_this_frame = moved
            track.stationary_streak = 0 if moved else track.stationary_streak + 1
        else:
            track.velocity = (0, 0)
            track.moved_this_frame = True
            track.stationary_streak = 0

        track.color = color
        track.positions.append(center)
        track.bboxes.append(bbox)
        if len(track.positions) > _MAX_HISTORY:
            track.positions.pop(0)
            track.bboxes.pop(0)
        track.age += 1
        track.last_seen = frame

    def get_track(self, entity_id: int) -> EntityTrack | None:
        return self._tracks.get(entity_id)

    def all_tracks(self) -> list[EntityTrack]:
        return list(self._tracks.values())


# ── Helpers ──────────────────────────────────────────────────────────────

def _component_center(comp: Any) -> tuple[int, int]:
    """Extract centroid from a Component or similar object."""
    if hasattr(comp, "center"):
        c = comp.center
        if isinstance(c, (list, tuple)) and len(c) >= 2:
            return (int(c[0]), int(c[1]))
    if hasattr(comp, "bbox"):
        x1, y1, x2, y2 = comp.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    if hasattr(comp, "cells") and comp.cells:
        xs = [p[0] for p in comp.cells]
        ys = [p[1] for p in comp.cells]
        return (sum(xs) // len(xs), sum(ys) // len(ys))
    return (0, 0)


def _component_color(comp: Any) -> int:
    if hasattr(comp, "dominant_color"):
        return int(comp.dominant_color)
    if hasattr(comp, "color"):
        return int(comp.color)
    return -1


def _component_bbox(comp: Any) -> tuple[int, int, int, int]:
    if hasattr(comp, "bbox"):
        return comp.bbox
    return (0, 0, 0, 0)


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
