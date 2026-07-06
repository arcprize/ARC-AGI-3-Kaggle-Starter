## Goal
- Make ORCH-OR solve AR25 level 1 and level 2 through general puzzle-solving mechanisms without fast paths to the reactive specialist.

## Constraints & Preferences
- No fast paths: ORCH-OR framework must drive decisions for all games, no game-specific shortcuts
- Run `python -m unittest tests.test_orch_or` after every change to verify no regressions (110 tests, all pass)
- 164 executive tests also all pass
- 2 pre-existing failures in `test_hyperon_bridge` (KeyError: 'ACTION1', AssertionError: False is not true) are unrelated

## Progress
### Done
- **Color 0 in walkable set** — `_find_walkable` in `analyzer.py` now always includes `0` in `walkable_colors` (alongside bg_color, player_color, goal_colors); this fixes movement through indicator pixels that sit on the shape piece so the system can move into and through them
- **Y‑diff threshold (12 px)** — `_mirror_axis_direction` returns y-direction only when `abs(by - gy) >= 12`; below that threshold, it falls through to x-direction (horizontal adjustment) to avoid vertical oscillation
- **Beam‑y‑range overlap check** — When the beam's y-range doesn't intersect the goal's y-range, `_mirror_axis_direction` always returns toward_y (ignoring the ydiff threshold). This handles cases where the beam is far above/below the goal but the ydiff happens to be small (e.g., test scenario uses 9‑pixel beam vs 3‑pixel goal). Covers the stuck‑at‑left‑border case in level 1: beam_y_max=44 < goal_y=47 → forces downward movement.
- **Reflection‑over‑goal‑x guard** — When `mirror_axis >= 0` and `2 * mirror_axis - player_x >= goal_x`, the function prefers y-direction even if ydiff < 12; this prevents getting stuck at the left border when the reflection x already covers the goal x
- **Fallback vertical‑first order** — When `has_beam=True`, the final fallback loop tries `[(0,1), (0,-1), (1,0), (-1,0)]` (vertical before horizontal) instead of the default `[(1,0), (-1,0), (0,1), (0,-1)]`
- **Beam‑position mirror‑inversion** — In the beam‑position navigation fallback, when `mirror_axis >= 0`, the x-direction is mirrored (`"left" ↔ "right"`) so the player moves away from the beam (toward the mirror) instead of toward it
- **Gradient opposite‑direction** — `_beam_gradient_step`: when `proximity_change < 0`, tries the opposite of `last_dir` before trying perpendicular directions; fixes the old bug where the system would try left/right after overshooting the goal y instead of backtracking

### Level 1 Completed
- **Step 14** — `levels completed 1` — first AR25 level completion by the ORCH-OR system
- Path: down from y=16→37 (steps 1‑6), then left from x=20→2 (steps 7‑12), then the reflection‑over‑goal‑x guard + beam‑y‑overlap check + vertical‑first fallback drove downward through the stuck area until the beam reached the goal
- After completion: system transitions to level 2 (puzzle mode, not platformer)

### In Progress
- **Level 2** — After transition to level 2, the system enters puzzle mode (navigation‑to‑goal logic). Early steps show the player stuck moving down from (28, 2) without progress — the retry loop fires repeatedly. Likely needs the movement system to handle puzzle‑mode obstacles.
- **Test fix** — `test_mirror_axis_direction_for_level1` was failing because the test scenario (beam y=5‑6, goal y=15‑17) needed "down" but the code returned "left" (ydiff=9 < 12). Fixed with beam‑y‑overlap check: beam_max_y=6 < goal_min_y=15 → returns "down".

### Blocked
- *(none)*

## Key Decisions
- **Color 0 cells are walkable**: The indicator pixels (color 0) that sit on top of the controlled entity are added to the walkable color set. This is safe because color 0 universally marks selection indicators in ARC‑AGI, not obstacles.
- **Vertical‑then‑horizontal switching**: Use beam‑to‑goal y‑distance threshold (12 px) as the switching criterion—when close enough in y, focus on x. The 12‑px threshold was chosen because the 3‑pixel step size plus the beam fan means 12 px is ~4 game‑cell units of tolerance.
- **Beam‑y‑overlap check**: When the beam doesn't yet reach the goal's y-range, always prioritize y movement; the threshold only applies once the beam spans the goal y. This is the right semantic: the beam needs to be at the right y before fine‑tuning x.
- **Reflection‑over‑goal‑x overrides x‑priority**: When the player's reflection already crosses the goal x-coordinate, the x alignment is "done" and the system should return to adjusting y. This prevents being stuck at the left wall while the y is still wrong.
- **Vertical‑first fallback for beam puzzles**: When no strategy yields a valid move, try up/down before left/right. Horizontal reversal is rarely helpful in mirror puzzles (moving back the way you came), but vertical movement often is.
- **Beam‑position invert for mirrors**: The beam position navigation was designed for direct light‑source puzzles; for mirror puzzles the player moves opposite to the raw direction, so invert x.

## Next Steps
1. **Investigate level 2**: The system enters puzzle mode for level 2 but gets stuck. Analyze level 2 layout, obstacles, and geometry to understand what's blocking movement.
2. **Run full end‑to‑end test**: `$env:AGI_DEBUG_DECISIONS="1"; .venv\Scripts\python scripts/play_local.py --game ar25 --max-steps 200` to verify level 2 behavior
3. **Run regression suite**: `python -m unittest tests.test_orch_or` to ensure all 110 tests still pass

## Critical Context
- **AR25 rendering**: `frame.frame[0]` is 64×64 numpy int8. Color 0 = selection indicators, color 4 = beam, color 11 = goals, color 5 = shape piece, color 10 = mirror, color 9 = background.
- **Level 1**: Shape piece at pixel (20,16) ≈ game (6,5), mirror axis at pixel 28, goal at pixel (53,47). **Solved at step 14.**
- **Level 2**: Different mirror axis, goal positions, initial player placement. Puzzle mode (not platformer).
- **Movement**: ACTION1=‑y up, ACTION2=+y down, ACTION3=‑x left, ACTION4=+x right, +3 pixels per step.
- **Beam composition**: The beam component (color 4) is the union of stationary border‑sprite reflections and the moving shape‑piece reflection. The beam extends in y as the player moves down. Beam y‑range = player_y‑1 to player_y+7 (9 pixel height of the shape piece).
- **Oscillation (resolved)**: The previous vertical oscillation (y=43‑49) is fixed by the 12‑px ydiff threshold: the system switches from y to x when |beam_y‑goal_y| < 12 and the beam already spans the goal y-range.
- **walkable colors** now = `{9 (bg), 0 (indicators), 5 (shape piece), 11 (goals)}` — all cells with these colors that are not wall obstacles.
- **Tests**: 110 ORCH‑OR tests pass, 164 executive tests pass, 2 pre‑existing hyperon failures.

## Relevant Files
- **agi_core/orch_or/strategies/exploration.py**: `_mirror_axis_direction` — uses ydiff threshold (12 px), beam‑y‑overlap check, and reflection‑over‑goal‑x guard to decide y vs x priority; `_beam_gradient_step` — opposite‑direction on proximity decrease; fallback loop order changed for beam puzzles; beam‑position navigation inverts x for mirrors
- **agi_core/orch_or/analyzer.py**: `_find_walkable` — added color 0 to walkable_colors
- **scripts/play_local.py**: Run `$env:AGI_DEBUG_DECISIONS="1"; .venv\Scripts\python scripts/play_local.py --game ar25 --max-steps 200` for end‑to‑end testing
- **tests/test_orch_or.py**: 110 tests
