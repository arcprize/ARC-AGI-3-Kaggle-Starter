# ARC-AGI-3 Duck v12 - Comprehensive Technical Guide

**Last Updated**: 2026-07-21  
**Current Score**: 1.33 (local validation)  
**Model**: Qwen3.6-27B-FP8  
**Submission Name**: arc3-duck-v12

---

## Table of Contents
1. [High-Level Architecture](#high-level-architecture)
2. [The Notebook (arc3-duck-v12.ipynb)](#the-notebook)
3. [The Scoring Formula (Most Critical!)](#the-scoring-formula)
4. [The Per-Turn Loop](#the-per-turn-loop)
5. [Key Files & How They Interact](#key-files--how-they-interact)
6. [The Grafts System (Patches)](#the-grafts-system-patches)
7. [How to Make Edits](#how-to-make-edits)
8. [Performance Analysis](#performance-analysis)

---

## High-Level Architecture

### 4 Independent Codebases

| Repo | Role | Path in Datasets |
|------|------|------------------|
| **TAAF** | Game harness: scoring, state management, orchestration | `taaf source share fork (banking)/src/tufa-arc-agi-framework` |
| **Inference** | LLM agent: prompts, tool-calling, Python sandbox | `taaf source share fork (banking)/src/ARC3-Inference` |
| **Grafts** | Patches: efficiency, recovery, retry logic | `taaf source share fork (banking)/src/taaf-grafts` |
| **Notebook** | Orchestration + experiment hooks | `1.33 scored in arc agi 3 competiotn in kaggle/arc3-duck-v12 (1).ipynb` |

### Information Flow

```
Kaggle Notebook (arc3-duck-v12.ipynb)
         ↓
[CELL 1-5] Boot environment, load code
         ↓
[CELL 6] Install grafts (composite.py flags dict)
         ↓
[CELL 7] Run benchmark → bm.play()
         ↓
Per-game loop:
   Build Frame + History
         ↓
   Prompt Builder (prompts.py templates)
         ↓
   Graft Chain [recovery → retry_guard]
         ↓
   vLLM Call (Qwen3.6-27B-FP8)
         ↓
   Parse tool call (JSON/XML format)
         ↓
   Execute in sandbox (python_tool_sandbox.py) OR engine action
         ↓
   Game engine (arcengine.step_env)
         ↓
   Score update (THE FORMULA)
         ↓
   Save diagnostics
         ↓
[CELL 8] Render diagnostics.html, submission.parquet
```

---

## The Notebook

**File**: `1.33 scored in arc agi 3 competiotn in kaggle/arc3-duck-v12 (1).ipynb`

### Cell Breakdown

| Cell | Purpose | Editable? |
|------|---------|-----------|
| **1** | Environment detection (`TRUE_SUBMISSION` flag) | ⚠️ Only if you understand consequences |
| **2** | Install ARC runtime from wheels | ❌ No |
| **3-4** | Locate source bundle, import repos to sys.path | ❌ No |
| **5** | Load pickled benchmark & deployment target | ❌ No |
| **6** | **EXPERIMENT HOOK** — install grafts via flags dict | ✅ **YES** |
| **7** | Play games (main loop) | ❌ No |
| **8** | Render diagnostics | ❌ No |

### Cell 6: The Graft Install (THE HOOK)

This is where you experiment. Currently:

```python
try:
    from taaf_grafts.composite import install
    
    install(bm, flags={
        "efficiency": True,
        "retry_guard": True,
        "shortcircuit": True,
        "recovery": True
    })
except Exception as exc:
    print(f"[taaf_grafts] cell-12 graft failed, running stock: {type(exc).__name__}: {exc}")
```

**What each flag does**:

| Flag | Effect | Status |
|------|--------|--------|
| `efficiency` | Per-turn budget note (report-only) | Active ✓ |
| `retry_guard` | Bounded retry + vLLM health probe | Active ✓ |
| `shortcircuit` | Trims no-op actions | Active ✓ |
| `recovery` | Stall detection + context refresh | **Active but net-negative** ⚠️ |
| `context_window` | Widen context (not a flag, set via import) | Not in current flags |
| `banking` | Cache + replay winning sequences | Untested ❓ |
| `transfer` | Cross-clone replay (for real competition) | Untested ❓ |

---

## The Scoring Formula

### **THIS IS THE MOST IMPORTANT LINE IN THE ENTIRE SYSTEM**

Located in: `datasets/taaf source share fork (banking)/src/tufa-arc-agi-framework/src/taaf/game.py`

```python
per_level_score = min(115, (baseline_actions / actions_used)² × 100)  [if completed]
per_level_score = 0                                                    [if not completed]
```

### Why This Matters

- **QUADRATIC** relationship between efficiency and score
- Complete a level 33% slower = **50% score loss** (not 33% loss)
- **EVERY OPTIMIZATION TARGETS ACTION COUNT**, not reasoning quality

### Examples

| Baseline | Your Actions | Score Calculation | Result |
|----------|-------------|-------------------|--------|
| 10 | 10 | min(115, (10/10)² × 100) | **100** |
| 10 | 13 | min(115, (10/13)² × 100) | **59** |
| 10 | 20 | min(115, (10/20)² × 100) | **25** |

→ **Every wasted action has a quadratic penalty**

### Real Submission vs. Local Run

- **Local**: baselines visible → can tune exactly
- **Real**: baselines hidden → agent_ext.py uses heuristic proxy

---

## The Per-Turn Loop

### Sequence (Inside `HarnessSolver.run_games()`)

1. **Build Runtime State** (`runtime_state.py`)
   - `Frame`: current grid as int8 numpy array (ARC palette 0-15)
   - `HistoryEntry`: list of past actions + frames
   - `valid_actions`: legal moves from engine

2. **Prompt Builder** (`prompts.py`)
   - Base system prompt
   - Add `GAME_OVERVIEW_ADDENDUM` (multi-level, optimization for efficiency)
   - Add `VISUAL_GAME_ADDENDUM` (entities, no HUD confusion warning)
   - Add `PYTHON_ADDENDUM` (sandbox syntax, segmentation, action() call)
   - Add `STRUCTURED_RUNTIME_STATE_ADDENDUM` (runtime vars: `current_frame`, `history`, `valid_actions`)
   - Add history (last 30 turns, compressed)
   - Current frame as ASCII + optional image

3. **Graft Chain** (optional interception)
   - **Recovery** (if enabled): stall detector, context wipe trigger, cross-level notes
   - **RetryGuard** (if enabled): bounded retry, vLLM health probe
   - Both degrade to stock on any error
   - **Outermost layer first**: RetryGuard catches failures inside Recovery

4. **vLLM Call** (`openai_compat.py`)
   - Endpoint: `http://127.0.0.1:1234/v1` (local) or env-resolved
   - Model: `vrfai/Qwen3.6-27B-FP8`
   - Max tokens per turn: ~16K (configurable)
   - Temperature: 0.6, top_p: 0.95, top_k: 20
   - Timeout: per env var (default infinite in local, safety cap in real submission)

5. **Tool Call Parsing** (`tool_agent.py`)
   - Regex extract `<tool_call><function=...><parameter=...>` blocks
   - Tool name: usually `python` or `execute_action` / `action`
   - Parameters: `code` (for python), `action` (for moves)

6. **Execute** (two paths)
   - **Path A (Python)**: Run in isolated subprocess (`python_tool_sandbox.py`)
     - Preload: `current_frame`, `history`, `previous_frame`, `valid_actions`, `last_action_result`
     - Available: `action()` function to step engine in sandbox
     - Available: `segmentation` for connected-component tracing
     - 30-second timeout, ~1K output tokens
   - **Path B (Direct Action)**: Parse action name → engine action → step

7. **Engine Step** (`arcengine.step_env`)
   - Advance game state one frame (may have animation frames inside)
   - Return new frame, available actions, level metadata

8. **Score Update**
   - Apply THE FORMULA
   - Check if `level_completed` → track per-level action count
   - Check if `game_over` or `done` → game complete

---

## Key Files & How They Interact

### TAAF (Harness)

#### `game.py` — **THE CORE SCORING**

**Key Classes**:
- `Frame`: 2D int8 grid (0-15, ARC colors)
- `GameState`: wraps arcengine frame + metadata
- `ActionRecord`: one action with token cost
- `GameRun`: per-game state machine + history
- `Game` (abstract), `GameAPI` (concrete with arcengine)

**Key Function** (line ~350):
```python
def _compute_final_score(self):
    if levels_completed == number_of_levels:
        for each level:
            score += min(115, (baseline_actions[level] / actions_used[level])² × 100)
    return score
```

**Why you care**: This formula is why every edit to the agent should reduce actions, not increase reasoning steps.

#### `game_api.py` — Bridge to arcengine

- `ArcadeSpec`: picklable Arcade description (operation mode, base URL, env dir)
- `GameAPI`: concrete Game using arcengine
- `GameAPI.step_env()`: calls arcengine, reconciles baselines

#### `benchmark.py` — The Container

- `Benchmark`: dataclass holding games list, solver, job_dir
- `bm.run()`: async orchestration loop
- `bm.play()` → `Benchmark.run()`

---

### Inference (Agent)

#### `tool_agent.py` — **THE AGENT LOGIC**

**Key Function**: `ToolAgent.analyze_frame()`
- Builds prompt via `prompts.py` templates
- Calls vLLM via `openai_compat.py`
- Parses tool calls (regex extract XML blocks)
- Dispatches to sandbox or direct action

**Key Class**: `ToolAgent`
```python
class ToolAgent:
    def __init__(self, model, timeout, api_key, base_url, provider):
        # Store LLM connection details
        
    async def analyze_frame(self, frame, history, valid_actions):
        # Build prompt
        # Call vLLM
        # Parse tool call
        # Execute sandbox or action
        # Return next action + tokens spent
```

**Context Window Tuning** (Line ~90):
```python
_LOCAL_ANALYZER_CONTEXT_WINDOW = _get_env_int("LOCAL_ANALYZER_CONTEXT_WINDOW", 32768)
```
- Default: 32K (conservative)
- Server max: 65K
- **Bottleneck**: repeated hypotheses caused by context overflow
- **Fix candidate**: widen to 50K-60K (see IDEAS.md)

#### `prompts.py` — Template Library

**Addenda** (concatenated into prompt):
- `GAME_OVERVIEW_ADDENDUM`: multi-level, efficiency emphasis
- `VISUAL_GAME_ADDENDUM`: object parsing, **HUD-timer warning** ← common failure mode
- `PYTHON_ADDENDUM`: sandbox syntax, segmentation usage
- `STRUCTURED_RUNTIME_STATE_ADDENDUM`: runtime var semantics
- `COMPACT_TOOL_SESSION_ADDENDUM`: tool call rules

**What you can edit**: Addenda text to steer agent behavior (e.g. add more examples, clarify ambiguous cases)

#### `python_tool_sandbox.py` — Isolated Execution

- Spawns subprocess
- Preloads globals: `current_frame`, `history`, `valid_actions`, etc.
- Splices in `segmentation.py` (no imports needed)
- Captures stdout/stderr, JSON result
- 30-second timeout

**What you can edit**: Preloaded globals, default imports, timeout logic

#### `segmentation.py` — Connected Components

- 4-connected object tracing
- Returns: nodes (with id, color, hash, pixels, boundary, children), adjacency list
- Deliberately stdlib-only (splices into sandbox at bootstrap)

**What you can edit**: Algorithm (e.g. 8-connected instead of 4-connected)

---

### Grafts (Patches)

#### `composite.py` — The Installer

**Key Function**: `install(bm, flags)`
- Reads flags dict
- Swaps solver instance (if `banking`/`transfer`/`shortcircuit`)
- Chains analyzers (if `recovery`/`retry_guard`)
- Patches module globals (if `context_window`)
- All wrapped in try/except that restores stock solver on any error

**What you can edit**:
- Add new flags
- Tune analyzer chain order
- Add new grafts

#### `agent_ext.py` — Efficiency Warnings (efficiency flag)

- Per-turn budget note: "You've used X% of efficient baseline."
- Net-zero waste detection: cycles, stagnation, revisits
- Report-only: can't stop wasted actions
- Heuristic proxy when baselines hidden (real submission)

**What you can edit**: Budget thresholds, waste detection heuristics

#### `retry_guard.py` — Bounded Retry (retry_guard flag)

- Fixes unbounded 1-request/second retry loop against dead vLLM
- Health probe: checks if server is responding
- Transparent pass-through on healthy turns
- Proven effective ✓

**What you can edit**: Retry backoff schedule, probe logic

#### `recovery.py` — Stall Detection (recovery flag)

**Three recovery modes**:
1. **R1** (free): context wipe on flatline/lock-in
2. **R2** (costly): ≤16-action probe on stall signal
3. **R3** (free): cross-level notes surviving engine reset

**Status**: Tested locally, **net negative** (EXPERIMENT_LOG.md)
- Probe misfires on levels that are actually progressing slowly
- Costs more than it saves

**What you can edit**: Stall detection thresholds, probe budget

#### `shortcircuit_solver.py` — No-Op Trimmer (shortcircuit flag)

- Analyzes transcript
- Trims genuinely wasted repeated/no-op actions
- **Not inert**: directly improves score on already-completed levels
- Active and working ✓

#### `banking_solver.py` — Win Cache (banking flag)

- Cache winning action sequence for reuse
- Untested this session
- May interfere with multi-level puzzle dynamics

#### `transfer_solver.py` — Cross-Clone Replay (transfer flag)

- For competition's ~110-clone structure
- Later clones skip to deepest level sibling already solved
- Untested this session
- Built specifically for real competition cloning

---

## The Grafts System (Patches)

### How Grafts Work

1. **Solver Replacement** (if flag in `_SOLVER_FLAGS`)
   - Subclass `HarnessSolver`
   - Override `run_game_internal()`
   - Store in `bm.solver`
   - `Benchmark.run()` deepcopies it → survives deepcopy

2. **Analyzer Chaining** (if flag in `_CHAIN_LAYERS`)
   - Wrap `ToolAgent` with a decorator
   - Each layer: intercept, maybe mutate, call inner, handle errors
   - Outermost: `RetryGuard`
   - Innermost: stock `ToolAgent`

### Error Handling

**Golden Rule**: Any graft error → stock behavior, no crash

```python
try:
    from taaf_grafts.composite import install
    install(bm, flags={...})
except Exception:
    # bm.solver already restored to stock by composite's internal try/except
    # Log the error, move on
```

### Current Active Grafts (Cell 6 Flags)

```python
install(bm, flags={
    "efficiency": True,        # Report-only budget warnings ✓
    "retry_guard": True,       # Bounded retry + health probe ✓
    "shortcircuit": True,      # No-op trimmer ✓
    "recovery": True,          # Stall detection (net negative?) ⚠️
})
```

### Potential New Grafts

- `context_window`: widen to 50K+ (untested, risk: slower per-turn)
- `banking` + `transfer`: for real competition structure (untested)
- `vision_context`: multimodal image input (optional, gated)

---

## How to Make Edits

### Safe Edits (Low Risk)

1. **Tune flags in Cell 6**
   - Add/remove `recovery`, `banking`, `transfer`
   - Current: `efficiency`, `retry_guard`, `shortcircuit` are proven working
   
2. **Edit prompts** (`prompts.py`)
   - Add clarifications to `GAME_OVERVIEW_ADDENDUM`
   - Add examples to `VISUAL_GAME_ADDENDUM`
   - Steer agent without changing core logic
   
3. **Tune constants**
   - `tool_agent.py`: `_LOCAL_ANALYZER_CONTEXT_WINDOW` (default 32K → try 50K)
   - `recovery.py`: stall thresholds
   - `agent_ext.py`: budget percentages

### Medium-Risk Edits (Requires Testing)

4. **New graft flag**
   - Add to `_CHAIN_LAYERS` in `composite.py`
   - Implement as analyzer wrapper
   - Test locally before real submission

5. **Modify sandbox preloads** (`python_tool_sandbox.py`)
   - Add new globals (be careful: namespace pollution)
   - Remove globals (break existing agents)
   
6. **Change segmentation** (`segmentation.py`)
   - Switch to 8-connected (currently 4-connected)
   - Add new return fields
   - May change agent behavior unpredictably

### High-Risk Edits (Avoid Unless Certain)

7. **Modify vLLM config** (`tool_agent.py`)
   - Temperature, top_p, top_k
   - Output token limits
   - These affect randomness/reasoning significantly

8. **Change prompt templates fundamentally**
   - Reorder addenda
   - Remove warnings
   - May break hard-coded agent expectations

9. **Modify the scoring formula** (`game.py`)
   - This is sacred; only touch if you understand every consequence

---

## Performance Analysis

### Current State (1.33 score)

| Component | Status | Notes |
|-----------|--------|-------|
| **Model** | vLLM + Qwen3.6-27B-FP8 | Working well |
| **Core Agent Loop** | Solid | Prompt structure proven |
| **Scoring** | Quadratic formula | Understood, immutable |
| **Efficiency Graft** | Report-only | Works as designed |
| **Retry Guard** | Proven effective | Fixed unbounded retry bug |
| **Shortcircuit** | Active | Trims wasted actions |
| **Recovery** | **Net Negative** | Stall detection too aggressive |
| **Context Window** | Bottleneck | 32K vs 65K server capacity |

### Known Bottlenecks (from ARCHITECTURE.md)

1. **Context Window** (High confidence)
   - Current: 32K configured
   - Server: 65K available
   - Symptom: repeated hypotheses, self-contradiction in transcripts
   - Fix: `context_window` flag or direct env var patch

2. **Recovery Graft** (High confidence, tested negative)
   - R2 probe misfires on slow-but-working levels
   - Costs more than it saves
   - Recommendation: disable or tune thresholds much higher

3. **Vision/Segmentation Misreads** (Medium confidence)
   - Agents mistake HUD timer bars for puzzle pieces
   - Prompts.py has explicit warning already
   - Root cause: unclear object boundaries in some games

4. **Efficiency Graft** (Medium confidence)
   - Report-only, can't enforce
   - May be too noisy if baseline estimate is bad

### Next Steps (from IDEAS.md)

- [ ] Widen `context_window` to 50K-60K (untested)
- [ ] Test `banking` + `transfer` in isolation
- [ ] Investigate heuristic baseline proxy accuracy
- [ ] Clarify level-transition state wipe mechanics

---

## File Reference Map

### Direct Paths in Workspace

| File | Path | Editable? | Impact |
|------|------|-----------|--------|
| **Notebook** | `1.33 scored in arc agi 3 competiotn in kaggle/arc3-duck-v12 (1).ipynb` | ✅ Cell 6 only | Experiment hook |
| **Game Formula** | `datasets/taaf source share fork (banking)/src/tufa-arc-agi-framework/src/taaf/game.py` | ❌ No | Scoring (sacred) |
| **Agent Core** | `datasets/taaf source share fork (banking)/src/ARC3-Inference/inference/agent/tool_agent.py` | ⚠️ Constants | Main loop |
| **Prompts** | `datasets/taaf source share fork (banking)/src/ARC3-Inference/inference/agent/prompts.py` | ✅ Yes | Agent steering |
| **Sandbox** | `datasets/taaf source share fork (banking)/src/ARC3-Inference/inference/agent/python_tool_sandbox.py` | ⚠️ Preloads | Sandbox globals |
| **Segmentation** | `datasets/taaf source share fork (banking)/src/ARC3-Inference/inference/utils/segmentation.py` | ⚠️ Algorithm | Object detection |
| **Graft Installer** | `datasets/taaf source share fork (banking)/src/taaf-grafts/taaf_grafts/composite.py` | ✅ Flags, layers | Patch system |
| **Efficiency** | `datasets/taaf source share fork (banking)/src/taaf-grafts/taaf_grafts/agent_ext.py` | ⚠️ Thresholds | Budget warnings |
| **Retry Guard** | `datasets/taaf source share fork (banking)/src/taaf-grafts/taaf_grafts/retry_guard.py` | ⚠️ Backoff | Retry logic |
| **Recovery** | `datasets/taaf source share fork (banking)/src/taaf-grafts/taaf_grafts/recovery.py` | ⚠️ Thresholds | Stall detection |
| **Shortcircuit** | `datasets/taaf source share fork (banking)/src/taaf-grafts/taaf_grafts/shortcircuit_solver.py` | ⚠️ Logic | No-op trimmer |
| **Banking** | `datasets/taaf source share fork (banking)/src/taaf-grafts/taaf_grafts/banking_solver.py` | ⚠️ Logic | Win cache (untested) |
| **Transfer** | `datasets/taaf source share fork (banking)/src/taaf-grafts/taaf_grafts/transfer_solver.py` | ⚠️ Logic | Cross-clone (untested) |

---

## Quick Reference: Common Edits

### Disable Recovery Graft (High Confidence Fix)

**File**: Notebook Cell 6

**Current**:
```python
install(bm, flags={
    "efficiency": True,
    "retry_guard": True,
    "shortcircuit": True,
    "recovery": True,  # ← Remove this line
})
```

**After**:
```python
install(bm, flags={
    "efficiency": True,
    "retry_guard": True,
    "shortcircuit": True,
})
```

### Widen Context Window (Medium Confidence)

**File**: `datasets/taaf source share fork (banking)/src/ARC3-Inference/inference/agent/tool_agent.py` (line ~90)

**Current**:
```python
_LOCAL_ANALYZER_CONTEXT_WINDOW = _get_env_int("LOCAL_ANALYZER_CONTEXT_WINDOW", 32768)
```

**After** (try 50K):
```python
_LOCAL_ANALYZER_CONTEXT_WINDOW = _get_env_int("LOCAL_ANALYZER_CONTEXT_WINDOW", 51200)
```

### Enable Transfer Graft (Untested)

**File**: Notebook Cell 6

**Add to flags**:
```python
install(bm, flags={
    "efficiency": True,
    "retry_guard": True,
    "shortcircuit": True,
    "transfer": True,  # ← Enables cross-clone replay
})
```

---

## Debug / Diagnostics

### Run Locally (Non-Submission)

1. Save notebook as `test-run.ipynb`
2. Set Cell 6 flags to test variant
3. Click "Run All"
4. Check `diagnostics.html` in output
5. Review `submission.parquet` (not scored locally)

### Check Active Grafts

Look for output banner after Cell 6 runs:
```
TAAF_GRAFTS FEATURES={'efficiency': True, 'retry_guard': True, ...} API_VERSION=1
[efficiency] armed
[retry_guard] armed
[shortcircuit] armed
```

### Understand a Failed Run

- Check `summary.txt` in working dir
- Check `git_status.txt` (version info)
- Check per-game transcripts in HTML
- Look for `[taaf_grafts]` error messages

---

## Summary: What You Need to Know to Make Edits

1. **The Scoring Formula** (quadratic in action count)
   - Wasted actions have exponential cost
   - Every optimization targets reducing actions
   
2. **The Per-Turn Loop** (frame → prompt → vLLM → parse → execute → score)
   - Each turn is one observe-plan-act cycle
   - Grafts wrap the LLM call (RetryGuard outer, Recovery inner)

3. **Cell 6 is Your Experiment Hook**
   - Flags dict controls which grafts activate
   - Safe to toggle, always falls back to stock on error

4. **The Grafts System** (analyzers wrap ToolAgent)
   - Recovery: stall detection (currently net-negative ⚠️)
   - RetryGuard: fixes unbounded retry (proven ✓)
   - Shortcircuit: trims no-ops (proven ✓)
   - Efficiency: budget warnings (proven ✓)

5. **Known Bottlenecks**
   - Context window: 32K vs 65K available (widening untested but likely positive)
   - Recovery: too aggressive (disable it)
   - Segmentation: occasional HUD misreads (fixable via prompts)

---

**You're now equipped to understand and edit the codebase.** Start with low-risk edits (disable recovery, widen context, tune prompts) and validate locally before real submission.
