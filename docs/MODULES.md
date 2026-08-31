# Modules

# MODULES.md — arc3-duck-v12

One entry per file actually touched or read this session. Depth varies —
files marked **(surface only)** were read at header/docstring level, not
line-by-line; treat those as starting points, not confirmed understanding.
See QUESTIONS.md for gaps.

## tufa-arc-agi-framework/src/taaf/ — the harness

| File | Purpose |
|---|---|
| `game.py` | Core `Game`/`GameState` abstractions. Owns `_compute_final_score` — the quadratic `min(115, (baseline/actions)²×100)` formula. Also owns an `arcengine` enum-pickling fix. |
| `game_api.py` | `arcengine`-backed concrete `Game` subclass. Reconciles baseline actions from the API response into `base_actions_per_level`. `ArcadeSpec` is the picklable Arcade description; `RunSession` caches one Arcade per unique spec. |
| `solver.py` | Abstract `Solver` contract: `setup()`/`teardown()`, `run_games(list[Game])`, cancellation semantics. **(surface only)** |
| `benchmark.py` | `Benchmark` dataclass + orchestration — the thing that gets pickled/unpickled between the deploy step and the notebook. |
| `deploy_kaggle.py` | Writes the two Kaggle bundles (pickled benchmark + source snapshots as a dataset; rendered notebook kernel). Also where `submission.parquet` gets written after a run. |
| `competition_arcade.py` | Local WSGI simulator of the real Kaggle gateway Arcade (same constraints: one scorecard, hidden baselines, no repeated game IDs) — lets failures be reproduced before spending a submission. **(surface only)** |
| `diagnostics.py` | `generate_run_html`, `generate_run_summary_txt`, cross-run comparison. PNGs inline base64, MP4s linked. **(surface only)** |
| `game_examples.py`, `solver_examples.py`, `standard_benchmarks.py` | Reference/example implementations — not part of the production duck-harness path. **(not read)** |
| `kaggle/`, `kaggle_random.py`, `deploy.py`, `deploy_inline.py`, `deploy_slurm.py`, `support.py` | Deployment-target variants and shared support code. **(not read)** |

## ARC3-Inference/inference/agent/ — the agent

| File | Purpose |
|---|---|
| `tool_agent.py` | **The core orchestrator.** Direct OpenAI-compatible tool-calling analyzer. Builds requests, calls vLLM, parses tool calls. This is the file every graft wraps or subclasses. **Good Stage 4 candidate.** |
| `prompts.py` | All prompt template text — `GAME_OVERVIEW_ADDENDUM`, `VISUAL_GAME_ADDENDUM` (contains the explicit HUD-timer-bar warning), `PYTHON_ADDENDUM`, tool-call format guidance, etc. |
| `python_tool_sandbox.py` | Isolated subprocess runner for the model's Python tool calls. Splices in `segmentation.py` at bootstrap since project imports aren't available inside the sandbox. |
| `vision_context.py` | Optional multimodal image context — renders the current grid as an actual image (ARC color map → RGB) for models that take image input. Gated by a `MULTIMODAL_CONTEXT` env var. |
| `action_names.py` | Bidirectional mapping between model-facing action names (`UP`/`DOWN`/`MOUSE`...) and engine names (`ACTION1`.../`RESET`). |
| `runtime_state.py` | `Frame` (grid/step/level) and `HistoryEntry` dataclasses — the structured state shared with the Python sandbox tools. Serialized to `tool_runtime_state.json`. |

## ARC3-Inference/inference/framework/ — the bridge

| File | Purpose |
|---|---|
| `solver.py` | `HarnessSolver` — adapts TAAF's abstract `Solver` contract to this specific `tool_agent`-based agent. Contains `ANALYZER_RETRY_BACKOFF_SECONDS` and the unbounded-retry bug that `retry_guard.py` was built to fix. |
| `kaggle.py` | Kaggle-specific constants: `DUCK_HARNESS_PUBLIC_GAME_IDS` (the 25 official games), model/wheelhouse dataset sources, `DEFAULT_VLLM_MAX_MODEL_LEN=65536`, served model name. |
| `run.py` | CLI entry point for running the harness locally through TAAF's Benchmark/GameAPI stack (used outside the notebook, e.g. for dev iteration). **(surface only)** |

## ARC3-Inference/inference/utils/ — shared utilities

| File | Purpose |
|---|---|
| `segmentation.py` | Connected-component segmentation of a single frame layer. Deliberately stdlib-only with no project imports so it can be spliced into the sandbox bootstrap. Includes Moore-neighbour contour tracing. |
| `grid_utils.py` | ARC color legend/chars, ASCII grid formatting. **(surface only)** |
| `openai_compat.py` | Builds chat payloads/headers for the OpenAI-compatible vLLM endpoint. **(surface only)** |
| `rearc_baselines.py`, `rearc_version.py`, `run_artifacts.py`, `viewer_artifacts.py` | **(not read)** |

## taaf-grafts/taaf_grafts/ — the patch layer

| File | Flag | Purpose |
|---|---|---|
| `composite.py` | — | The installer. Reads the flags dict, wires everything else together, prints the confirmation banner. Read in full this session. |
| `agent_ext.py` | `efficiency` | `EfficiencyToolAgent` — report-only per-turn budget note. Pure, LLM-free waste detectors (net-zero cycles, stagnation, revisits). Falls back to a heuristic baseline proxy when the real baseline is hidden (real submission). |
| `retry_guard.py` | `retry_guard` | Bounded-retry + vLLM health-probe wrapper. Fixes an unbounded 1-request/second retry loop against a dead local vLLM server. Transparent pass-through on every healthy turn. |
| `shortcircuit_solver.py` | `shortcircuit` | Trims genuinely wasted repeated/no-op actions. **Not behaviorally inert** — directly improves the quadratic efficiency score on levels already being cleared. |
| `recovery.py` | `recovery` | R1 refresh (free, death-spiral/lock-in detection + context wipe), R2 probe (costly, ≤16 actions, fires on flatline/dominance signal after `PROBE_MIN_ACTS`), R3 handoff (free, writes a cross-level note that survives the engine's level-transition wipe). **Tested locally — see EXPERIMENT_LOG.md, net negative as shipped.** |
| `banking_solver.py` | `banking` | Win-then-replay: caches a winning action sequence for reuse. **Untested this session.** |
| `transfer_solver.py` | `transfer` | Cross-clone replay via `family_store` — lets a later clone of the same game family skip to the deepest level a sibling already solved. Built specifically around the real competition's ~110-clone structure. **Untested this session.** |
| `family_store.py` | (used by transfer) | The shared store transfer reads/writes. **(surface only)** |
| `solver_base.py` | — | Base class shared by the solver-replacement grafts. **(not read)** |

## Flags confirmed present (corrected 2026-07-24)
`schema_void`, `schema_notes`, `schema_helpers` — all three modules exist in the deployed dataset bundle AND `composite.py` handles them explicitly. Previous analysis that called them "missing/inert" was incorrect. See IDEAS.md for experiment plans.

| File | Flag | Purpose |
|---|---|---|
| `schema_void.py` | `schema_void` | Surprise-abort on mixed commit batches: trims un-executed tails when valid-actions collapse or grid oscillation is detected. Composes OVER shortcircuit. |
| `schema_notes.py` | `schema_notes` | Static ~100-token "SCHEMA LOOP" prompt note appended every turn: probe→observe→commit deliberation nudge. Subclasses EfficiencyToolAgent. |
| `schema_helpers.py` | `schema_helpers` | Preloads `grid_diff`, `connected_components`, `action_effect_summary`, `recent_history` into the Python sandbox. Subclasses EfficiencyToolAgent. **Mutually exclusive with schema_notes.** |