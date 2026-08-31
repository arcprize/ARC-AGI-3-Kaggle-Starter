"""Process-global cross-clone family store (battle-plan transfer core).

Engine facts this graft is built on (verified against the vendored
sources and ``arc_agi`` 0.9.8 / ``arcengine`` 0.9.3):

- All 110 competition runs share ONE process and ONE ThreadPoolExecutor
  (``HarnessSolver`` fans games out under a single ``Semaphore(28)``), so
  a module-level singleton guarded by a lock is the whole coordination
  substrate — no IPC, no files.
- The only trustworthy clone signal is the initial visible frame: private
  tags are stripped at the gateway (``arc_agi/api.py`` frame projection)
  and clone-index math is not sanctioned. A fingerprint that misses makes
  the entire transfer stack a measured no-op (fingerprint ``None`` ->
  every store call a no-op), which is the design's provability guarantee.

Invariants enforced here:

- The store holds ONLY plain data (grids, action ids, dicts) copied on
  publish. It never retains a game/env/session reference, so a family key
  cannot resurrect engine state or pin memory.
- ``publish_segment`` / ``publish_win`` are MONOTONIC: a longer plan can
  never displace a shorter one. Transfer only ever gets cheaper.
- ``note_grid_divergence`` is per-family and, once tripped, STICKY: after
  ``_GRID_FLIP_STRIKES`` grid replay mismatches that each kept levels
  advancing consistently (the cosmetic-RNG / lf52 class), that family
  verifies replay by level count only, forever, for every sibling clone.
  Requiring more than one corroborating divergence keeps a lone misleading
  sibling from enabling unverified 'level'-mode replay onto a scored play.
- Every public store method is blanket ``try/except`` -> ``None``/no-op
  and consults the module-level :data:`ENABLED` flag, so any store fault
  or a disabled build degrades to stock (no transfer, no crash).

The trace-walking that turns a recorded trace into per-level
:class:`Segment` objects lives in ``trace_utils`` (W8); this module only
stores, copies, and hands back whatever segments it is given, plus a
self-contained splitter for a stored full-win plan.
"""

from __future__ import annotations

import hashlib
import struct
import threading
from dataclasses import dataclass, field
from typing import Any

import arcengine

from inference.framework.solver import _grid_from_state

Grid = tuple[tuple[int, ...], ...]

# Global kill switch. Consulted by every public store method; flip to
# False (tests, flag-off builds) to make the whole store an inert no-op.
ENABLED = True

# Consistent-levels grid divergences required before a family flips from
# 'grid' to the unverified 'level' mode (see note_grid_divergence). >1 so a
# single misleading sibling cannot enable blind level-mode replay onto the
# scored play.
_GRID_FLIP_STRIKES = 2


@dataclass(frozen=True)
class SegmentStep:
    """One replayable action and the state it produced.

    ``action_id`` is an :class:`arcengine.GameAction` (replayed directly
    via ``arcengine.ActionInput(id=action_id, ...)``); ``action_data`` is
    the click/argument payload; ``post_grid`` / ``post_levels`` are the
    verification anchors checked after the action during replay.
    """

    action_id: arcengine.GameAction
    action_data: dict[str, Any]
    post_grid: Grid
    post_levels: int


@dataclass(frozen=True)
class Segment:
    """The replayable plan for a single level.

    ``level_start_grid`` is the visible frame at the moment this level
    began (used by adoption to gate a mid-game replay against the live
    board); ``steps`` are the kept actions that advance that level to the
    next. Both fields are immutable — grids are tuples, ``steps`` is a
    tuple, and :func:`_copy_segment` copies the per-step ``action_data``
    dicts on publish so the store owns an isolated snapshot.
    """

    level_start_grid: Grid
    steps: tuple[SegmentStep, ...]


@dataclass
class _Family:
    """Per-fingerprint transfer state (plain data only)."""

    segments: dict[int, Segment] = field(default_factory=dict)
    win_plan: tuple[SegmentStep, ...] | None = None
    verify_mode: str = "grid"
    divergence_strikes: int = 0
    stats: dict[str, int] = field(default_factory=dict)


def _serialize_fingerprint_inputs(
    grid: Grid, levels: int, actions: list[int]
) -> bytes:
    """Deterministic, cross-process-stable byte encoding of the fingerprint
    inputs. Explicit framing (never ``repr``/``pickle``) so the digest is
    identical for any two processes that observe the same initial state."""
    chunks: list[bytes] = [struct.pack(">II", len(grid), int(levels))]
    for row in grid:
        chunks.append(struct.pack(">I", len(row)))
        chunks.append(struct.pack(f">{len(row)}i", *(int(c) for c in row)))
    chunks.append(struct.pack(">I", len(actions)))
    chunks.append(struct.pack(f">{len(actions)}i", *actions))
    return b"".join(chunks)


def fingerprint(game: Any) -> bytes | None:
    """Stable family key for ``game`` from its initial state, or ``None``.

    ``blake2b`` over the serialized initial grid, level count, and sorted
    available action ids. ANY exception (unstarted game, missing state,
    exotic frame) -> ``None`` -> the game joins no family -> transfer is a
    no-op for it. Stable across processes within a run (deterministic
    serialization); it need not survive a process restart.
    """
    try:
        state = game.current_state
        grid = _grid_from_state(state)
        levels = int(game.number_of_levels)
        actions = sorted(int(a) for a in state.available_actions)
        payload = _serialize_fingerprint_inputs(grid, levels, actions)
        return hashlib.blake2b(payload, digest_size=16).digest()
    except Exception:
        return None


def _copy_segment(segment: Segment) -> Segment:
    """Copy-on-publish snapshot: fresh ``action_data`` dicts, everything
    else already immutable."""
    steps = tuple(
        SegmentStep(
            action_id=s.action_id,
            action_data=dict(s.action_data),
            post_grid=s.post_grid,
            post_levels=int(s.post_levels),
        )
        for s in segment.steps
    )
    return Segment(level_start_grid=segment.level_start_grid, steps=steps)


def _plan_to_steps(plan: Any) -> tuple[SegmentStep, ...]:
    """Convert a pruned winning trace (``list[TraceStep]``) to immutable
    :class:`SegmentStep` records via duck typing (no ``banking_solver``
    import, no cycle)."""
    return tuple(
        SegmentStep(
            action_id=step.action_id,
            action_data=dict(step.action_data),
            post_grid=step.grid,
            post_levels=int(step.levels_completed),
        )
        for step in plan
    )


def _split_plan_into_segments(
    plan: tuple[SegmentStep, ...]
) -> dict[int, Segment]:
    """Split a pruned full-win plan into per-level segments.

    Keyed by the ``levels_completed`` value DURING which the actions were
    taken (level 0's actions are keyed 0, and advance the count to 1). A
    pruned plan carries no RESETs and no silent no-ops, so a level break
    is exactly a ``post_levels`` increment. ``level_start_grid`` for a
    level is the ``post_grid`` of the previous level's final action; level
    0 has no observed pre-first-action frame, so it starts ``()``.
    """
    segments: dict[int, Segment] = {}
    cur_level = 0
    cur_start: Grid = ()
    cur_steps: list[SegmentStep] = []
    for step in plan:
        cur_steps.append(step)
        if step.post_levels > cur_level:
            segments[cur_level] = Segment(
                level_start_grid=cur_start, steps=tuple(cur_steps)
            )
            cur_level = int(step.post_levels)
            cur_start = step.post_grid
            cur_steps = []
    return segments


class _FamilyStore:
    """Thread-safe process-global store of per-family transfer plans.

    One lock serializes every mutation and read; families are created
    lazily. Every public method is blanket ``try/except`` -> ``None``/
    no-op and short-circuits when :data:`ENABLED` is False or the key is
    ``None`` (a fingerprint miss).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._families: dict[bytes, _Family] = {}

    def _family(self, key: bytes) -> _Family:
        fam = self._families.get(key)
        if fam is None:
            fam = _Family()
            self._families[key] = fam
        return fam

    def publish_segment(
        self, key: bytes | None, level: int, segment: Segment
    ) -> None:
        if not ENABLED or key is None:
            return
        try:
            level = int(level)
            snapshot = _copy_segment(segment)
            with self._lock:
                fam = self._family(key)
                existing = fam.segments.get(level)
                if existing is None or len(snapshot.steps) < len(existing.steps):
                    fam.segments[level] = snapshot
                    fam.stats["segment_publishes"] = (
                        fam.stats.get("segment_publishes", 0) + 1
                    )
        except Exception:
            return

    def publish_win(self, key: bytes | None, plan: Any) -> None:
        if not ENABLED or key is None:
            return
        try:
            steps = _plan_to_steps(plan)
            if not steps:
                return
            with self._lock:
                fam = self._family(key)
                if fam.win_plan is None or len(steps) < len(fam.win_plan):
                    fam.win_plan = steps
                    fam.stats["win_publishes"] = (
                        fam.stats.get("win_publishes", 0) + 1
                    )
        except Exception:
            return

    def best_prefix(self, key: bytes | None) -> list[Segment] | None:
        if not ENABLED:
            return None
        if key is None:
            return []
        try:
            with self._lock:
                fam = self._families.get(key)
                if fam is None:
                    return []
                if fam.win_plan is not None:
                    split = _split_plan_into_segments(fam.win_plan)
                    return [split[lvl] for lvl in sorted(split)]
                prefix: list[Segment] = []
                level = 0
                while level in fam.segments:
                    prefix.append(fam.segments[level])
                    level += 1
                return prefix
        except Exception:
            return None

    def segment_for_level(
        self, key: bytes | None, level: int
    ) -> Segment | None:
        if not ENABLED or key is None:
            return None
        try:
            level = int(level)
            with self._lock:
                fam = self._families.get(key)
                if fam is None:
                    return None
                seg = fam.segments.get(level)
                if seg is not None:
                    return seg
                if fam.win_plan is not None:
                    return _split_plan_into_segments(fam.win_plan).get(level)
                return None
        except Exception:
            return None

    def note_grid_divergence(self, key: bytes | None) -> None:
        """Record one consistent-levels grid divergence, flipping the family to
        ``'level'`` verification only after ``_GRID_FLIP_STRIKES`` of them.

        A single sibling's grid divergence is weak evidence: it distinguishes a
        cosmetic-RNG family (every sibling's board animates, logic identical)
        from one whose RNG actually drives logic — but only in aggregate. In
        ``'grid'`` mode every replay step is verified and the first mismatch
        aborts after one wasted action, so staying in ``'grid'`` longer is the
        safe default; ``'level'`` mode skips that check and commits unverified
        actions to the clone's single scored play (no max-over-plays net, unlike
        banking's second-play replay). Requiring corroboration from more than
        one observation before entering the unverified mode keeps a lone
        misleading signal from letting a later sibling blind-replay onto a
        divergent board and inflate its scored action counts."""
        if not ENABLED or key is None:
            return
        try:
            with self._lock:
                fam = self._family(key)
                fam.divergence_strikes += 1
                if fam.divergence_strikes >= _GRID_FLIP_STRIKES:
                    fam.verify_mode = "level"
        except Exception:
            return

    def get_verify_mode(self, key: bytes | None) -> str | None:
        if not ENABLED:
            return None
        if key is None:
            return "grid"
        try:
            with self._lock:
                fam = self._families.get(key)
                return fam.verify_mode if fam is not None else "grid"
        except Exception:
            return "grid"


_STORE: _FamilyStore | None = None
_STORE_LOCK = threading.Lock()


def get_store() -> _FamilyStore:
    """The process-global store singleton (created lazily)."""
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = _FamilyStore()
        return _STORE


def reset_store() -> None:
    """Drop all family state — tests only (never called in the harness)."""
    global _STORE
    with _STORE_LOCK:
        _STORE = _FamilyStore()
