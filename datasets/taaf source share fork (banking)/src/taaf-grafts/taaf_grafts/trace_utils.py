"""Per-level prefix pruning, generalized from banking's whole-win pruner.

``prune_winning_trace`` (in ``banking_solver``) turns a *complete* winning
trace into one flat replay plan and raises on anything it cannot verify —
correct for banking, where a partial plan is worthless. The cross-clone
transfer stack needs the same reduction applied *per level* to *partial*
traces: a scout that clears levels 0..k (and may still be mid-level, dead,
or given up) must publish exactly the k fully-completed level segments so a
sibling clone can replay them.

``prune_level_segments`` is that generalization. It walks the trace with
banking's byte-for-byte semantics:

- a ``RESET`` voids the pending buffer and rebases the running grid to the
  post-RESET frame (a mid-level RESET — including the harness auto-RESET
  after GAME_OVER — restarts the current level from its fresh board);
- a step that changes neither the visible frame nor ``levels_completed`` is
  a prunable no-op and is dropped;
- a step that advances ``levels_completed`` is always kept.

The two differences from the banking pruner, both required for partial
transfer, are:

1. at EVERY ``levels_completed`` advance it flushes the pending buffer as a
   completed :class:`Segment` keyed by the just-finished level index, and
2. it never raises — the trailing pending buffer of the in-progress level
   is simply discarded (no WIN, no full-coverage requirement).

INVARIANT (proven by the equivalence property test): on a genuine winning
trace, concatenating the segment steps in ascending level order yields
exactly ``prune_winning_trace``'s plan. The two pruners are the same walk;
this one just cuts the plan at each level boundary and tolerates a partial
tail.

A ``Segment``'s ``level_start_grid`` is the fresh board that level began on
— the initial grid for level 0, the completing step's post-frame for a
level entered by advancing, or the post-RESET frame for a level last
restarted by a RESET. Transfer adoption gates replay on this grid matching
the live post-auto-RESET board, so it must be the clean level-start frame.

Pure functions, no I/O, no engine or game references retained. ``TraceStep``
is imported from ``banking_solver`` (never duplicated); the produced
``Segment``/``SegmentStep`` carry only plain data (copy-on-produce) so the
family store can hold them without pinning any live state.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import arcengine

from taaf_grafts.banking_solver import TraceStep

Grid = tuple[tuple[int, ...], ...]


class SegmentStep(NamedTuple):
    """One kept action of a completed level, minus banking's engine-state
    field (a stored segment holds only replayable data)."""

    action_id: arcengine.GameAction
    action_data: dict[str, Any]
    post_grid: Grid
    post_levels: int


class Segment(NamedTuple):
    """A fully-completed level's pruned replay: the board it started on and
    the ordered kept steps that clear it."""

    level_start_grid: Grid
    steps: tuple[SegmentStep, ...]


def _segment_step(step: TraceStep) -> SegmentStep:
    return SegmentStep(
        action_id=step.action_id,
        action_data=dict(step.action_data),
        post_grid=step.grid,
        post_levels=step.levels_completed,
    )


def prune_level_segments(
    trace: list[TraceStep],
    initial_grid: Grid,
) -> dict[int, Segment]:
    """Return one pruned :class:`Segment` per fully-completed level.

    Walks ``trace`` with banking's exact prune semantics, flushing a segment
    at every ``levels_completed`` advance and discarding the trailing
    in-progress buffer. Works on partial traces and never raises — an empty
    or all-pruned trace yields ``{}``. Only levels observed to complete are
    returned, keyed by their zero-based level index.
    """
    segments: dict[int, Segment] = {}
    pending: list[SegmentStep] = []
    prev_grid = initial_grid
    prev_levels = 0
    level_start_grid = initial_grid
    for step in trace:
        if step.action_id == arcengine.GameAction.RESET:
            pending = []
            prev_grid = step.grid
            level_start_grid = step.grid
            continue
        if step.levels_completed > prev_levels:
            pending.append(_segment_step(step))
            segments[prev_levels] = Segment(
                level_start_grid=level_start_grid,
                steps=tuple(pending),
            )
            pending = []
            prev_levels = step.levels_completed
            level_start_grid = step.grid
        elif step.grid != prev_grid:
            pending.append(_segment_step(step))
        # else: no frame change, no level progress — prunable no-op.
        prev_grid = step.grid
    return segments
