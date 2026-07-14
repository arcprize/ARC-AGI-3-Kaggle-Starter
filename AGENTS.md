# ARC-AGI-3 Kaggle Starter

## Overview
ARC-AGI-3 Kaggle competition starter kit. The agent system uses ORCH-OR (Orchestrator) — a general puzzle-solving framework — with a vendored ARC-AGI-3-Agents framework at `vendor/ARC-AGI-3-Agents/`. The main agent is at `agent/my_agent.py`. Custom code lives in `agi_core/` (especially `agi_core/orch_or/` for the ORCH-OR planner) and `agent/my_agent.py`.

## Python Environment
- Use the local `.venv` Python interpreter: `.venv/Scripts/python` (Windows) or `.venv/bin/python` (Linux/Mac)
- The `.venv` has `arc-agi` and other dependencies installed — the global `python` command may not have them
- All commands below should be prefixed with the venv Python path, or activate the venv first

## Key Commands
- **Run all ORCH-OR unit tests**: `.venv/Scripts/python -m unittest tests.test_orch_or` (110 tests, must pass)
- **Run executive tests**: `.venv/Scripts/python -m unittest tests.test_executive` (163 tests, must pass)
- **Run hyperon bridge tests** (pre-existing failures unrelated): `.venv/Scripts/python -m unittest tests.test_hyperon_bridge` (3 known failures)
- **Run priors SQLite tests**: `.venv/Scripts/python -m unittest tests.test_priors_sqlite`
- **Run a specific test**: `.venv/Scripts/python -m unittest tests.test_orch_or.TestOrchOr.test_ar25_level2_beam_alignment`
- **Play a game locally**: `.venv/Scripts/python scripts/play_local.py --game ar25 --max-steps 200`
- **Quick smoke test (multiple games, 50 steps)**: `.venv/Scripts/python scripts/play_local.py --game ls20,vc33 --max-steps 50`
- **Run training from demo zip**: `.venv/Scripts/python scripts/train_offline_from_zip.py`

## Architecture

### ORCH-OR (`agi_core/orch_or/`)
- **`orchestrator.py`**: Top-level decision loop. Maintains `_stable_goal`, `_failed_directions`, `_non_beam_controllers`, `_switch_count`. Generates plans via `generate_plan()`, falls back when no plan steps exist.
- **`plan_generator.py`**: Routes objectives to strategies. Objectives: `navigate-to-goal` → movement strategy, `explore-board` → exploration strategy, `interaction` → interaction strategy, `puzzle` → puzzle strategy.
- **`plan_executor.py`**: Converts `PlanStep` actions to game actions (`ACTION1`-`ACTION6`).
- **`monitor.py`**: Detects stalled progress. Keeps track of `_failed_directions` (which movement directions failed).
- **`analyzer.py`**: Builds `GameModel` from the game board (64×64 int8 grid). Finds components, obstacles, walkable cells, beam, goals.
- **`reducer.py`**: Selects the leaf node from the plan tree.
- **`game_model.py`**: Data classes: `GameModel`, `Component`, `GameFeatures`, `PlanStep`.
- **`search.py`**: A* pathfinding over walkable cells.

### Strategies (`agi_core/orch_or/strategies/`)
- **`movement.py`**: `movement_strategy` — A* pathfinding + direction-based fallback. Contains `_is_beam_controller()` to avoid beam-nav for non-beam entities.
- **`exploration.py`**: `exploration_strategy` — beam navigation logic, stable-goal navigation, switch trigger as last resort. Also has `_is_beam_controller()`.
- **`interaction.py`**: `interaction_strategy` — proposes ACTION6 click on nearby interactive obstacles, and ACTION5 switch when beam lights goals.
- **`puzzle.py`**: `puzzle_strategy` — defers to exploration when beam is present.

### Hyperon Bridge (`agi_core/knowledge/`)
- **`hyperon_bridge.py`**: Wraps symbolic rules (MeTTa), trained priors (SQLite), and live memory (SQLite) into unified action confidence computation. `_run_shadow_rules()` mirrors ~30 board-type policy queries. All 22 board-type handlers check `_load_board_policy_overrides()` for trained params before falling back to hardcoded defaults. No game-specific overrides (removed).
- **`rules/arc.metta`**: 200-line MeTTa symbolic policy scaffold — board-type strategies (puzzle/platformer/unknown). No game-specific rules (removed).
- **`_load_board_policy_overrides()`**: Loads `board_policy_params` from trained DB to override hardcoded MeTTa values with data-driven parameters. Supports 22 param names including gravity-enabled, horizontal-scroll, collision-type, exit-type, probe-strategy, focus-mode, nav-style, danger-detection, viewport-style.
- **`_load_trained_prior_bucket()`**: Loads action count priors from trained DB.
- **`_load_persisted_memory_bundle()`**: Now loads from hot + archive + trained DBs, merging abstract memories and prior buckets from all three sources.

### Training (`agi_core/training/`)
- **`demo_zip.py`**: Processes `arc_agi_3_public_demo_human_testing.zip` (26 games, 680 recordings, 180K steps). Produces: action counts (4 scopes), memory records (84K), abstract memories (cross-game), and `board_policy_params` from action distributions.
- **`_extract_board_policy_params()`**: Infers board type (puzzle/platformer/unknown) from action distribution, then derives 10 policy dimensions (explore-bias, interact-range, switch-urgency, path-preference, move-bias, beam-priority, entity-count, goal-proximity, retry-tolerance, stagnation-threshold).

### Storage (`agi_core/storage/`)
- **`priors_sqlite.py`**: 4 prior tables (game/state/frame/subgoal) + `board_policy_params` table. `load_memory_bundle_from_stores()` now accepts `trained_db_path` — loads abstract memories and prior buckets from trained DB alongside hot and archive.
- **`memory_sqlite.py`**: 13 live memory tables (episode records, action stats, basin stats, semantic/procedural/abstract memories). Dual-store (hot + archive) loading with hot bias.

### Configuration (`agi_core/config.py`)
- **`RuntimeConfig`**: `@dataclass(slots=True)` with ~35 fields from `AGI_*` env vars. Key fields:
  - `llm_enabled: bool = False` (env `AGI_LLM_ENABLE`, default `"0"`) — when False, LLM components are not created
  - `planner_enabled: bool = True` (env `AGI_PLANNER_ENABLE`, default `"1"`)
  - `planner_parallel: bool = True` (env `AGI_PLANNER_PARALLEL`, default `"1"`) — parallel strategy execution
  - `planner_workers: int = 0` (env `AGI_PLANNER_WORKERS`, default `"0"`) — `0` = auto (`cpu_count`, capped at 32)
  - `source_bias` — `{"reactive": 0.25, "symbolic": 0.15}` (LLM removed)
- **LLM disabled by default**: ORCH-OR is the primary backend. When ORCH-OR fails to plan, the Executive fallback uses reactive + symbolic specialists + critic only (no LLM).
- **Re-enable LLM**: set `AGI_LLM_ENABLE=1` (and optionally `AGI_LLM_MAX_CALLS=2`).

### Other key files
- **`agi_core/executive.py`**: Top-level executive. Guards all LLM code paths with `self.llm is not None` checks.
- **`agi_core/specialists/reactive.py`**: Reactive specialist (fast-path for AR25 removed).
- **`agent/my_agent.py`**: Conditionally creates LLMSpecialist and llm_backend only when `config.llm_enabled` is True.

## Important Conventions
- **No fast paths**: ORCH-OR must drive decisions for all games — NO game-specific hardcoded logic.
- **Movement step = 3 grid cells** in AR25 (3px per step in 64×64 grid).
- **ACTION6 = click/interact** — used to break interactive obstacles.
- **Color conventions**: 0 = selection indicator, 4 = beam, 5 = shape piece, 9 = background, 10 = mirror, 11 = goals.
- **`_is_beam_controller`**: Checks proximity to beam cells (≤6px) to determine if the player entity controls the beam. Prevents beam-nav from applying to shape piece.
- **`_failed_directions`**: Set of direction strings that failed (detected by monitor). Must be checked in all strategy code.
- **`non_beam_controllers`**: Set of component colors that are NOT beam controllers (e.g., `{5}` for shape piece).
- **`_switch_count`**: Tracks number of entity switches; reset on level completion.
- **`stable_goal`**: Persisted goal position used across frames for platformer boards (goal doesn't move).
- **A* walkable set**: Always includes bg_color, player_color, goal_colors, and color 0. Obstacles are excluded.
- **Tests**: Run both `test_orch_or` and `test_executive` after every change.

## Current State
- AR25 has 8 levels (no "level 0" — first level is Level 1).
- **Level 1** (mirror reflection puzzle): Shape piece must position itself so its reflection across a static mirror (color 10, grid x=10) covers all 5 goal markers at grid positions (19,15), (17,17), (17,16), (17,15), (18,15).
- **Level 1 COMPLETING**: The agent navigates the shape piece to the computed target pixel (4, 46) via A* pathfinding in `_try_mirror_placement()` and the game engine's `vplrhaovhr()` detects the win. Verified to achieve `levels_completed=1` within ~15 steps.
- The game engine renders with letterbox padding (color 5) and a step counter energy bar (color 11) that can confuse component detection. Fixed by: (1) excluding the rightmost column and bottom row from background color sampling in `_detect_background_color()`, (2) selecting the correct goal component (not the energy bar) in `_try_mirror_placement()`.

### Key Fixes Applied
- **False beam detection fixed**: The game engine renders mirror reflections as color-4 cells (`pbtdgroplk = wxztsweyvs = 4` in `ar25.py:1599`). Fixed in `_detect_beam()` (analyzer.py) by checking if ALL color-4 cells are on the opposite side of a mirror from the player.
- **Background color detection**: `_detect_background_color()` now excludes the rightmost column (energy bar, color 11) and bottom row (letterbox padding, color 5) from border sampling.
- **Goal component vs energy bar**: The energy bar (color 11) at column 63 was selected as the goal component. Fixed `_try_mirror_placement()` to select the color-11 component that has cells in interior columns (not column 63).
- **NameError in `_navigate_to_goal`**: The `goal_score` closure referenced undefined `features` and `beam_cells`. Fixed by using `game_model.features` and computing `beam_cells` inside the method.
- **Infinite ACTION5 loop from cached solutions fixed**: `_action_from_memory()` returned ACTION5 (`cached_solution_switch_back`) when the player was within 3px of a cached mirror position. Changed to return `_fallback()` instead.
- **`_switch_count` protected from entity jump resets**: Added `last_action != "ACTION5"` guard.
- **PuzzleSystem activation for mirror puzzles**: `should_activate()` explicitly checks for mirror components (color 10) with goals on the opposite side.
- **Heuristic fallback navigation**: Added heuristic movement toward target when A* fails in `_try_mirror_placement()`.

### Level 1 Target Computation
- Mirror at grid x=10 (pixel x=31). Goals span grid (17-19, 15-17) → pixels (52-58, 46-52).
- Target: grid (1, 15) → pixel (4, 46). Reflection covers (17-19, 15-17) → all 5 goals.
- Reflection formula (game engine): `reflected_x = 2 * mirror_sprite_x - original_x`.

## Cleanup History
- **All game-specific hardcoding removed**: `_GAME_PRIORS`, `_GAME_RESET_NEEDED`, `_BEAM_STRATEGY` dicts (ar25) removed from `hyperon_bridge.py:385-470`. These are now fully hyperon memory managed via `_load_trained_prior_bucket()` and `_load_board_policy_overrides()`.
- **`beam_strategy` dead code removed**: Queried from hyperon bridge but never consumed by any strategy. Removed from `orchestrator.py` and `hyperon_bridge.py`.
- **ar25 game-specific rules removed from `arc.metta`**: Game priors, reset overrides, beam strategies, subgoal overrides.
- **12 board-type handlers extended**: `board-gravity-enabled`, `board-horizontal-scroll`, `board-collision-type`, `board-exit-type`, `board-probe-strategy`, `board-focus-mode`, `board-nav-style`, `board-danger-detection`, `board-viewport-style` now check trained overrides before falling back to hardcoded defaults.
- **ar25-specific comment** removed from `interaction.py:42`.
- **ar25 test data removed**: `test_beam_strategy_ar25_level_1`, `test_beam_strategy_ar25_level_2`, `test_game_action_priors_ar25_overrides_defaults` deleted.
