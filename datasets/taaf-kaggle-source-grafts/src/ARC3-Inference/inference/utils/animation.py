"""Animation awareness (Experiment 3).

``arcengine`` renders a frame after every internal ``step()``, so one action
can come back as a short animation. TAAF exposes the whole list
(``GameState.all_frames``), but the harness only ever consumed
``raw.frame[-1]`` -- so every intermediate frame was discarded before the
agent could see it.

Measured over 24 games (12 multi-frame responses each), 13 games return
multi-frame responses, in two distinct shapes:

- **type 1** (ft09, sb26): first and last frame are identical, all information
  -- a rejected click, a consumed attempt -- lives only in between.
- **type 2** (r11l, sk48): no transient pixels at all, the intermediate frames
  are pure motion interpolation and carry nothing the final frame does not.

The metadata built here is deliberately tiny (a few dozen tokens, and nothing
at all for the single-frame case) because it goes into every action result.
"""
from __future__ import annotations

from typing import Any, Sequence

from inference.utils.grid_utils import ARC_COLOR_CHARS, format_grid_ascii

Grid = tuple[tuple[int, ...], ...]

# --- Stage 2 retrieval budget ------------------------------------------------
#
# Why a diff timeline and not the frames themselves: ``format_grid_ascii``
# renders one 64x64 grid as 4159 characters, roughly 1400-2000 tokens, while
# the whole tool response budget is ~1024 tokens. sb26 returns up to 42 frames
# for a single action -- 60-80k tokens raw, still 24-34k after deduplication.
# Full frames are simply never affordable, so retrieval collapses consecutive
# identical frames and reports only the cells that changed between the
# remaining ones, under a hard global budget. A single frame can still be read
# verbatim, but only cropped to a region.
ANIMATION_MAX_STEPS = 8
ANIMATION_MAX_CELLS_PER_STEP = 24
ANIMATION_MAX_TOTAL_CELLS = 80
ANIMATION_MAX_CROP_CELLS = 1024
ANIMATION_CROP_PADDING = 2

# --- Stage 3 proactive hint --------------------------------------------------
#
# Deliberately conservative: the hint spends prompt tokens and competes with
# the model's own plan, so it only fires after several turns of genuinely no
# progress on a level AND only where this level's animations actually hide
# something (transient pixels) -- the type-1 case, where reading the final
# frame cannot possibly work. Type-2 games (pure motion interpolation) never
# trigger it, because there is nothing in those frames to find.
ANIMATION_HINT_MIN_TURNS_WITHOUT_PROGRESS = 6
ANIMATION_HINT_MIN_TRANSIENT_PIXELS = 8
ANIMATION_HINT_MIN_TRANSIENT_ANIMATIONS = 2
ANIMATION_HINT_COOLDOWN_TURNS = 6
ANIMATION_HINT_FOLLOW_WINDOW_TURNS = 3


def normalize_frames(raw_frames: Any) -> list[Grid]:
    """Convert an engine frame list into hashable, comparable grids."""
    frames: list[Grid] = []
    for frame in raw_frames or ():
        rows = frame.tolist() if hasattr(frame, "tolist") else frame
        frames.append(tuple(tuple(int(cell) for cell in row) for row in rows or ()))
    return frames


def _transient_cells(frames: Sequence[Grid]) -> list[tuple[int, int]]:
    """Cells that some intermediate frame shows differently from the final one.

    These are exactly the pixels an agent reading only the final frame can
    never see -- the type-1 signal.
    """
    final = frames[-1]
    cells: set[tuple[int, int]] = set()
    for frame in frames[:-1]:
        for row_index, row in enumerate(frame):
            final_row = final[row_index] if row_index < len(final) else ()
            for col_index, value in enumerate(row):
                final_value = final_row[col_index] if col_index < len(final_row) else None
                if value != final_value:
                    cells.add((row_index, col_index))
    return sorted(cells)


def summarize_animation(frames: Sequence[Grid], *, board_changed: bool) -> dict[str, Any] | None:
    """Compact per-action animation metadata, or ``None`` if there was none.

    Returns ``None`` for the single-frame case so ordinary actions carry no
    extra tokens at all.
    """
    if len(frames) <= 1:
        return None

    transient = _transient_cells(frames)
    summary: dict[str, Any] = {
        "frames": len(frames),
        "unique_frames": len(dict.fromkeys(frames)),
        # The headline signal: the action animated, yet the board the agent
        # gets to see is byte-identical to the one before it.
        "board_unchanged": not board_changed,
        "transient_pixels": len(transient),
    }
    if transient:
        rows = [cell[0] for cell in transient]
        cols = [cell[1] for cell in transient]
        # Inclusive [top, left, bottom, right]; enough to point at the region
        # without spending tokens on a coordinate list.
        summary["transient_bbox"] = [min(rows), min(cols), max(rows), max(cols)]
    return summary


def pick_animation(summaries: Sequence[dict[str, Any] | None]) -> dict[str, Any] | None:
    """The single most informative animation out of a batch of actions.

    A batch reports one ``animation`` block, not one per action, so pick the
    richest: most transient pixels first (that is what the agent cannot see
    otherwise), then most frames. Summing across actions would be meaningless
    -- the blocks describe different points in time.
    """
    candidates = [item for item in summaries if isinstance(item, dict) and item]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (int(item.get("transient_pixels") or 0), int(item.get("frames") or 0)),
    )


def _color_char(value: int | None) -> str:
    if value is None:
        return "?"
    return ARC_COLOR_CHARS[max(0, min(15, int(value)))]


def _cell_at(grid: Grid, row: int, col: int) -> int | None:
    if row >= len(grid):
        return None
    line = grid[row]
    return line[col] if col < len(line) else None


def _diff_cells(before: Grid, after: Grid) -> list[tuple[int, int, int | None, int | None]]:
    rows = max(len(before), len(after))
    changes: list[tuple[int, int, int | None, int | None]] = []
    for row in range(rows):
        cols = max(len(before[row]) if row < len(before) else 0, len(after[row]) if row < len(after) else 0)
        for col in range(cols):
            old = _cell_at(before, row, col)
            new = _cell_at(after, row, col)
            if old != new:
                changes.append((row, col, old, new))
    return changes


def _bbox_text(cells: Sequence[tuple[int, int, Any, Any]]) -> str:
    rows = [cell[0] for cell in cells]
    cols = [cell[1] for cell in cells]
    return f"rows {min(rows)}-{max(rows)}, cols {min(cols)}-{max(cols)}"


def _format_changes(cells: Sequence[tuple[int, int, int | None, int | None]], budget: int) -> list[str]:
    """Group changed cells by colour transition, one compact line each.

    Strings rather than nested coordinate lists because the caller renders
    with ``json.dumps(indent=2)``, which would put every ``[row, col]`` pair on
    four separate lines.
    """
    grouped: dict[str, list[str]] = {}
    for row, col, old, new in cells[:budget]:
        key = f"{_color_char(old)}>{_color_char(new)}"
        grouped.setdefault(key, []).append(f"({row},{col})")
    lines = [f"{key} @ {' '.join(points)}" for key, points in grouped.items()]
    if len(cells) > budget:
        lines.append(f"... {len(cells) - budget} further changed cells omitted")
    return lines


def _dedupe_frames(frames: Sequence[Grid]) -> list[tuple[Grid, int]]:
    """Collapse runs of identical frames into ``(grid, repeat)`` pairs."""
    collapsed: list[tuple[Grid, int]] = []
    for grid in frames:
        if collapsed and collapsed[-1][0] == grid:
            previous, count = collapsed[-1]
            collapsed[-1] = (previous, count + 1)
            continue
        collapsed.append((grid, 1))
    return collapsed


def _crop_ascii(grid: Grid, region: Sequence[int]) -> dict[str, Any]:
    top, left, bottom, right = (int(value) for value in region)
    top = max(0, top)
    left = max(0, left)
    bottom = min(len(grid) - 1, bottom)
    right = min(max((len(row) for row in grid), default=1) - 1, right)
    if bottom < top or right < left:
        return {"error": "Empty region."}
    cells = (bottom - top + 1) * (right - left + 1)
    if cells > ANIMATION_MAX_CROP_CELLS:
        return {
            "error": (
                f"Region covers {cells} cells, over the {ANIMATION_MAX_CROP_CELLS}-cell limit. "
                "Pass a smaller region=(top, left, bottom, right)."
            )
        }
    cropped = tuple(tuple(row[left : right + 1]) for row in grid[top : bottom + 1])
    return {
        "region": f"rows {top}-{bottom}, cols {left}-{right}",
        "ascii": format_grid_ascii(cropped),
    }


def build_animation_view(
    record: dict[str, Any] | None,
    *,
    frame: int | None = None,
    region: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Render one stored animation for the agent, within the token budget.

    ``frame=None`` gives the diff timeline (the default and the cheap path);
    ``frame=k`` reads frame ``k`` verbatim, cropped to ``region`` or, failing
    that, to the transient bounding box.
    """
    if not record:
        return {"error": "No animation is available; the last action returned a single frame."}
    frames: list[Grid] = list(record.get("frames") or ())
    if len(frames) <= 1:
        return {"error": "No animation is available; the last action returned a single frame."}

    header: dict[str, Any] = {
        "action": record.get("action_display"),
        "action_num": record.get("action_num"),
        "frames": len(frames),
    }

    if frame is not None:
        try:
            index = int(frame)
        except (TypeError, ValueError):
            return {**header, "error": "frame must be an integer."}
        if index < 0:
            index += len(frames)
        if not 0 <= index < len(frames):
            return {**header, "error": f"frame must be between 0 and {len(frames) - 1}."}
        grid = frames[index]
        if region is None:
            summary = record.get("summary") or {}
            bbox = summary.get("transient_bbox")
            if bbox:
                region = [
                    bbox[0] - ANIMATION_CROP_PADDING,
                    bbox[1] - ANIMATION_CROP_PADDING,
                    bbox[2] + ANIMATION_CROP_PADDING,
                    bbox[3] + ANIMATION_CROP_PADDING,
                ]
            else:
                return {
                    **header,
                    "error": (
                        "This animation has no transient region to crop to. "
                        "Pass region=(top, left, bottom, right) explicitly."
                    ),
                }
        return {**header, "frame": index, **_crop_ascii(grid, region)}

    collapsed = _dedupe_frames(frames)
    before = record.get("before")
    chain: list[Grid] = [before if before is not None else collapsed[0][0]]
    chain.extend(grid for grid, _ in collapsed)

    steps: list[dict[str, Any]] = []
    for position, (grid, repeat) in enumerate(collapsed, start=1):
        cells = _diff_cells(chain[position - 1], grid)
        if not cells:
            continue
        steps.append({"_position": position, "_repeat": repeat, "_cells": cells})

    # Over budget: keep the steps with the most change (that is where the
    # information is), then restore chronological order so the sequence still
    # reads as a sequence.
    omitted = 0
    if len(steps) > ANIMATION_MAX_STEPS:
        omitted = len(steps) - ANIMATION_MAX_STEPS
        steps = sorted(steps, key=lambda item: len(item["_cells"]), reverse=True)[:ANIMATION_MAX_STEPS]
        steps.sort(key=lambda item: item["_position"])

    rendered: list[dict[str, Any]] = []
    remaining = ANIMATION_MAX_TOTAL_CELLS
    for step in steps:
        cells = step["_cells"]
        entry: dict[str, Any] = {
            "step": step["_position"],
            "changed": len(cells),
            "bbox": _bbox_text(cells),
        }
        if step["_repeat"] > 1:
            entry["held_for_frames"] = step["_repeat"]
        budget = min(ANIMATION_MAX_CELLS_PER_STEP, max(0, remaining))
        if len(cells) > ANIMATION_MAX_CELLS_PER_STEP or budget == 0:
            # Too broad to enumerate usefully -- a colour-transition census
            # says more per token than an arbitrary truncated cell list.
            census: dict[str, int] = {}
            for _, _, old, new in cells:
                key = f"{_color_char(old)}>{_color_char(new)}"
                census[key] = census.get(key, 0) + 1
            entry["transitions"] = census
        else:
            entry["changes"] = _format_changes(cells, budget)
            remaining -= min(len(cells), budget)
        rendered.append(entry)

    summary = record.get("summary") or {}
    view = {
        **header,
        "unique_frames": summary.get("unique_frames"),
        "board_unchanged": summary.get("board_unchanged"),
        "steps": rendered,
        "note": (
            "Diff timeline: step 1 is the change from the board before the action to the first "
            "returned frame, each later step the change from the previous distinct frame. "
            "Identical consecutive frames are collapsed into held_for_frames. "
            "Use animation(frame=k) to read one frame around the transient region verbatim."
        ),
    }
    if omitted:
        view["omitted_steps"] = omitted
    return view


def should_suggest_animation(
    *,
    turns_without_progress: int,
    transient_animations: int,
    turns_since_last_hint: int,
) -> bool:
    """Whether to proactively point the agent at ``animation()`` this turn."""
    return (
        turns_without_progress >= ANIMATION_HINT_MIN_TURNS_WITHOUT_PROGRESS
        and transient_animations >= ANIMATION_HINT_MIN_TRANSIENT_ANIMATIONS
        and turns_since_last_hint >= ANIMATION_HINT_COOLDOWN_TURNS
    )


def animation_hint_text(turns_without_progress: int, transient_animations: int) -> str:
    return (
        f"You have spent {turns_without_progress} turns on this level without completing it, "
        f"and {transient_animations} action(s) here returned animations whose intermediate frames "
        "differ from the board you were shown. Whatever those actions did was displayed only in "
        "frames you never saw. Call `animation()` to read that diff timeline; it executes nothing "
        "and spends no action budget."
    )


def describe_animation(summary: dict[str, Any] | None) -> str:
    """One-line prompt rendering of ``summarize_animation`` output."""
    if not summary:
        return ""
    frames = summary.get("frames")
    unique = summary.get("unique_frames")
    transient = summary.get("transient_pixels") or 0
    pieces = [f"That sequence animated: {frames} frames ({unique} distinct)"]
    if summary.get("board_unchanged"):
        pieces.append(
            "and the final board is identical to the one before the action -- "
            "the action was not a no-op, its effect is visible only in the intermediate frames"
        )
    if transient:
        bbox = summary.get("transient_bbox")
        region = f" around rows {bbox[0]}-{bbox[2]}, cols {bbox[1]}-{bbox[3]}" if bbox else ""
        pieces.append(f"with {transient} transient pixels{region}")
    else:
        pieces.append("with no transient pixels (pure motion interpolation)")
    return ", ".join(pieces) + "."
