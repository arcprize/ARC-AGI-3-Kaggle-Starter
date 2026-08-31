# Arc3-Duck-v12 Notebook Execution Walkthrough

**Complete step-by-step breakdown of what happens when you run the notebook.**

---

## Cell 1: Environment & Submission Mode Detection

**File Location**: `arc3-duck-v12.ipynb` cells 1-2 combined

### What It Does

```python
TRUE_SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
os.environ["TAAF_RUN_AS_SUBMISSION"] = "1" if TRUE_SUBMISSION else "0"
os.environ["TAAF_MINIMAL_DIAGNOSTICS"] = "1" if TRUE_SUBMISSION else "0"
```

### Key Variable: `TRUE_SUBMISSION`

| Scenario | `TRUE_SUBMISSION` | Diagnostics | Games | Baselines | Deadline |
|----------|------------------|-------------|-------|-----------|----------|
| **Local Run** (Save & Run All) | `False` | Full | 4 (offline bundled) | Visible | ~10 min before budget |
| **Real Submission** (Kaggle rerun) | `True` | Minimal | 25 official (~110 clones) | Hidden | 11h 20m cap |

### Critical Constant Names (set here, read later)

- `TAAF_RUN_AS_SUBMISSION` — read by framework to gate diagnostics
- `TAAF_MINIMAL_DIAGNOSTICS` — read by framework to skip periodic saves
- `MPLBACKEND=Agg` — matplotlib uses non-interactive backend
- `ONLY_RESET_LEVELS=true` — tells arcengine to keep level on RESET

### Output

```
taaf.kaggle: TRUE_SUBMISSION=False
```

If `True` → you're in official competition rerun (real submission, real score counted)

---

## Cell 2: Install ARC Runtime

**Purpose**: Install `arc-agi` from offline wheelhouse (Kaggle has no internet)

### What It Does

```python
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "--quiet", "--no-index", "--no-warn-conflicts",
    "--find-links", "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",
    "arc-agi",
])
```

### Files Extracted

From: `arc-agi-0.9.8-py3-none-any.whl` + dependencies
- arcengine, blinker, certifi, charset_normalizer, click, contourpy, cycler, flask, fonttools, idna, itsdangerous, jinja2, kiwisolver, ... (full wheel list in workspace)

### Output

Usually silent (stdout=DEVNULL), or error if wheels are corrupt

---

## Cell 3-4: Locate Source Bundle & Start vLLM Server

**Purpose**: Mount Kaggle datasets, add source repos to sys.path, run setup commands

### Key Constants

```python
DATASET_SOURCES = [
    "thtennant/taaf-kaggle-source-share-fork",           # [0] THE SOURCE BUNDLE
    "driessmit1/arc3-vllm-h100-wheelhouse-v3",
    "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot",
]
DATASET_BUNDLE_MARKER = "taaf-kaggle-bundle.json"
```

### What It Does

1. **Locate the source bundle** by finding `taaf-kaggle-bundle.json` marker file
   - Returns path to `taaf source share fork (banking)/` (from your datasets folder)

2. **Map Kaggle mount paths**
   - Each dataset mounts at `/kaggle/input/<slug>` or `/kaggle/input/datasets/<owner>/<slug>`
   - Stores mapping in `TAAF_KAGGLE_INPUT_PATHS` JSON env var

3. **Add source repos to sys.path**
   - Looks in `<bundle>/src/` for subdirectories
   - For each: tries `<repo>/src` then `<repo>` as import root
   - Inserts all in reverse order (so `taaf-grafts` is earliest, most specific first)

4. **Create `.pth` file** for child processes
   - Writes `/usr/local/lib/python3.*/site-packages/taaf_kaggle_sources.pth`
   - Contains all source paths, one per line

5. **Run setup commands**
   - Reads `<bundle>/setup_commands.json`
   - Commands run serially with env vars set
   - **IMPORTANT**: This is where vLLM server starts (likely first command)

### Environment Variables Published

```python
{
    "TAAF_KAGGLE_INPUT_PATHS": "{...mount paths JSON...}",
    "TAAF_KAGGLE_DATASET_SOURCES": "['thtennant/...', 'driessmit1/...', ...]",
    "TAAF_KAGGLE_KERNEL_SOURCES": "[]",
    "PYTHON": "/usr/bin/python3.11",
    "TAAF_KAGGLE_BUNDLE_DIR": "/kaggle/input/.../taaf source share fork (banking)",
    "TAAF_KAGGLE_WORKING_DIR": "/kaggle/working",
}
```

### Setup Commands Likely Include

1. Install vLLM + torch GPU packages
2. Start vLLM server on `http://127.0.0.1:1234/v1`
3. Load Qwen3.6-27B-FP8 model weights

### Output

```
taaf.kaggle: source bundle = /kaggle/input/.../taaf source share fork (banking)
taaf.kaggle: input paths = {...}
taaf.kaggle: wrote .../taaf_kaggle_sources.pth (3 source roots)
taaf.kaggle: setup command: <command>
...
```

---

## Cell 5: Load Pickled Benchmark

**Purpose**: Restore the pre-built benchmark configuration and solver

### What It Does

```python
# Restore deployment target (metadata + runtime info)
with open(BUNDLE_DIR / "deploy_target.pkl", "rb") as file:
    target = pickle.load(file)
target.actual_run_as_submission = TRUE_SUBMISSION
target.is_competition_rerun = TRUE_SUBMISSION

# Restore benchmark (games list, solver, job_dir)
with open(BUNDLE_DIR / "benchmark_initial.pkl", "rb") as file:
    bm = pickle.load(file)
bm.job_dir = WORKING_DIR
```

### Objects Loaded

**`target`** (deployment target):
- Contains metadata about the run
- `max_runtime_s`: total budget (e.g., 43200 for 12 hours)
- `actual_run_as_submission`: set to `TRUE_SUBMISSION`

**`bm`** (benchmark):
- `games`: list of `Game` objects (initially 4 offline ones, swapped in Cell 7)
- `solver`: `HarnessSolver` instance (will be wrapped/modified in Cell 6)
- `job_dir`: changed to `/kaggle/working`
- `n_passes`: number of passes per game (usually 1)
- `label`: run label for diagnostics

### Post-Unpickle State

- vLLM server should be running (started in Cell 4)
- Source repos importable (sys.path set in Cell 4)
- `bm.solver` is stock `HarnessSolver` from pickled state
- `bm.games` are offline test games (not the real 25 yet)

---

## Cell 6: Customization & Graft Install

**File**: `arc3-duck-v12.ipynb` Cell 6 (`"## 6. Customization hook"`)

**THIS IS THE EXPERIMENT HOOK — THE ONLY CELL YOU SHOULD EDIT DURING ITERATION**

### What Stock Code Does

```python
# Make one-off changes to `bm`, `bm.games`, or `bm.solver` here before the run starts.
# Example:
# bm.label = f"{bm.label}-debug"

try:
    from taaf_grafts.composite import install
    
    install(bm, flags={
        "efficiency": True,
        "retry_guard": True,
        "shortcircuit": True,
        "recovery": True,
    })
except Exception as exc:
    print(f"[taaf_grafts] cell-12 graft failed, running stock: {type(exc).__name__}: {exc}")
```

### What `install()` Does

Located in: `taaf_grafts/composite.py`

```python
def install(bm, flags):
    # [1] Restore original solver in case of error (try/except wraps entire body)
    
    # [2] If solver-replacement flag set (banking/transfer/shortcircuit):
    #     - Create new solver subclass instance
    #     - Replace bm.solver
    
    # [3] If analyzer-chain flag set (recovery/retry_guard):
    #     - Build analyzer wrapping chain
    #     - Store factory on bm.solver.analyzer_factory
    
    # [4] If context_window flag set:
    #     - Patch module global _LOCAL_ANALYZER_CONTEXT_WINDOW
    
    # [5] Print machine-parseable banner:
    #     TAAF_GRAFTS FEATURES={...} API_VERSION=1
    #     [efficiency] armed
    #     [retry_guard] armed
    #     ...
```

### Current Flags Explained

| Flag | Module | What It Does | Current | Status |
|------|--------|-------------|---------|--------|
| `efficiency` | `agent_ext.py` | Per-turn budget warnings (report-only) | **True** | ✓ Proven |
| `retry_guard` | `retry_guard.py` | Bounded retry + vLLM health check | **True** | ✓ Proven |
| `shortcircuit` | `shortcircuit_solver.py` | Trim no-op actions in transcript | **True** | ✓ Proven |
| `recovery` | `recovery.py` | Stall detection + context reset | **True** | ⚠️ Net-negative |
| `banking` | `banking_solver.py` | Win cache + replay | **Not set** | ❓ Untested |
| `transfer` | `transfer_solver.py` | Cross-clone replay | **Not set** | ❓ Untested |
| `context_window` | `tool_agent.py` (global) | Widen context (not a flag, import-time only) | **Not tuned** | 🔄 Potential |

### Graft Chain Architecture

If `recovery` and `retry_guard` both enabled, the analyzer chain looks like:

```
vLLM Request
     ↓
┌─ RetryGuard (outer layer, first to catch errors)
│  └─ RecoveryLayer (inner layer)
│     └─ Stock ToolAgent
│
└─ On any error in Recovery/ToolAgent:
   RetryGuard backs off and retries, or falls back
```

**Why this order?** RetryGuard must govern the entire recovery cycle, including any probes Recovery might trigger.

### Possible Experiments

1. **Disable recovery** (medium confidence fix)
   ```python
   flags={"efficiency": True, "retry_guard": True, "shortcircuit": True}
   ```

2. **Widen context window** (requires separate edit to tool_agent.py)
   ```python
   # This alone won't work; you also need to edit tool_agent.py line ~90
   flags={..., "context_window": 51200}  # Not actually used; just docs
   ```

3. **Enable transfer** (untested, for real competition)
   ```python
   flags={..., "transfer": True}  # Also enables banking automatically
   ```

### Output if Successful

```
TAAF_GRAFTS FEATURES={'efficiency': True, 'retry_guard': True, 'shortcircuit': True, 'recovery': True} API_VERSION=1
[efficiency] armed
[retry_guard] armed
[recovery] armed (R1+R2+R3)
[shortcircuit] armed
```

---

## Cell 7: Play Benchmark

**The Main Loop** — This is where all the action happens

### Flow Overview

```python
if TRUE_SUBMISSION:
    # Wait for Kaggle gateway (competitions arcade)
    # Load live 25-game list from gateway
else:
    # Load bundled offline 4-game list from environment_files/
    
# CELL-6 REAASIGNS bm.games (offline dup-gate + competition list swap)

# Benchmark.run(soft_end_time, runtime_environment, minimal_diagnostics)
#   └─ HarnessSolver.run_games()
#        └─ for each game:
#             └─ for each pass (usually 1):
#                  └─ game.start_game()  [initialize level, frame, actions]
#                  └─ while not done:
#                      └─ ToolAgent.analyze_frame()  [main per-turn loop]
#                      └─ parse tool call
#                      └─ execute action or python sandbox
#                      └─ game.step_env()  [advance engine]
#                      └─ score update (THE FORMULA)
#                  └─ game.finish_game()  [tally final score]
```

### Offline vs. Live Arcade

#### Offline Mode (Local Run, TRUE_SUBMISSION=False)

1. **Load environment files** from `arc_agi_3_wheels/../environment_files/`
2. Create `GameAPI` objects pointing to offline `Arcade(OFFLINE mode)`
3. Take up to 4 games
4. Add 1 duplicate (for transfer graft testing)
5. Set `soft_end = now + budget - 10 min` (graceful shutdown buffer)

#### Competition Mode (Real Submission, TRUE_SUBMISSION=True)

1. **Wait for Kaggle gateway** at `http://gateway:8001/` (5-sec poll, 10-min timeout)
2. Create `GameAPI` objects pointing to live `Arcade(COMPETITION mode, gateway URL)`
3. All 25 official games + later clones (~110 total)
4. Set `soft_end = now + 11h 20m` (safety cap, emerge before 12h hard kill)

### Per-Game Loop Detail

**For each game**:

1. **`game.start_game()`**
   - Initialize arcengine game state
   - Load level 0
   - Fetch baseline actions (if visible)
   - Create `GameRun` object (empty history, score=0)

2. **Per-turn loop** (while not `game_over` and not `soft_end` and not timeout)
   - **Build frame**: `GameState` from engine
   - **Get history**: last 30 turns (compressed)
   - **Get valid actions**: engine-provided legal moves
   - **Call `ToolAgent.analyze_frame()`** ← main LLM call, see detailed breakdown below
   - **Parse tool call**: extract action name or python code
   - **Execute**:
     - If python code: run in sandbox with preloaded globals
     - If action: map to engine action (UP, DOWN, MOUSE, ACTION1-6, RESET)
   - **`game.step_env(action)`**: engine advances, new frame returned
   - **Score update**: apply quadratic formula, update `GameRun.actions_per_level`
   - **Check termination**: level complete? game won? invalid action?

3. **`game.finish_game()`**
   - Finalize score (quadratic formula over all levels)
   - Store in `GameRun.final_score`
   - Write intermediate states (if enabled)

### Per-Turn Loop: Deep Dive (ToolAgent.analyze_frame)

**Called once per turn, the heart of the agent**

#### Step 1: Build Prompt

From `inference/agent/prompts.py`:

```python
prompt_parts = [
    SYSTEM_PROMPT,
    GAME_OVERVIEW_ADDENDUM,        # "Multi-level, optimize actions"
    VISUAL_GAME_ADDENDUM,           # "Objects, no HUD confusion"
    PYTHON_ADDENDUM,                # "Sandbox syntax"
    STRUCTURED_RUNTIME_STATE_ADDENDUM,  # "Runtime vars: current_frame, history, ..."
    MULTIMODAL_CONTEXT_ADDENDUM,    # "Image provided"
    TOOL_CALL_FORMAT_GUIDANCE,      # "<tool_call><function=...>"
    COMPACT_TOOL_SESSION_ADDENDUM,  # "One tool, python"
    
    # History (last 30 turns, compressed)
    "\n[Turn history]...",
    
    # Current frame as ASCII art
    "\n[Current frame]\n",
    current_frame.ascii,
    
    # Optional: image
    "\n[Current grid image]\n<image>...",
]
```

#### Step 2: Call vLLM

Via `openai_compat.py`:

```python
response = requests.post(
    f"{base_url}/v1/chat/completions",
    headers=build_headers(api_key),
    json=build_chat_payload(
        messages=prompt_parts,
        model="vrfai/Qwen3.6-27B-FP8",
        max_tokens=16384,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        tools=[{"type": "function", "function": {"name": "python", "description": "..."}}],
        tool_choice="auto",
    ),
    timeout=timeout_seconds,
)
```

**Configured in**: `tool_agent.py` lines ~80-100

#### Step 3: Parse Response

Regex extraction (`tool_agent.py` line ~40-80):

```python
# If response contains "<tool_call><function=python>...</function></tool_call>":
#   Extract code
# Elif contains "<tool_call><function=action>...</function></tool_call>":
#   Extract action
# Else:
#   Return text response (unusual, treated as context-only)
```

#### Step 4: Execute

**Path A: Python Tool**

```python
# code = extracted code string
# Run in subprocess via python_tool_sandbox.py:
#   Preload globals:
#     current_frame    [Frame object: .ascii, .segmentation, .step, .level, .shape]
#     history          [List of HistoryEntry: .action, .frame]
#     previous_frame   [Frame before last action, or None]
#     valid_actions    [List of legal action names]
#     last_action_result  [Dict from last action(...) call]
#     action(actions)  [Function to step engine]
#   
#   Available imports: bisect, collections, copy, itertools, json, math, ...
#   
#   available segmentation module (stdlib-only, spliced at bootstrap)
#
# Capture stdout/stderr + return value
# 30-second timeout
# ~1K tokens output cap
```

**Path B: Direct Action**

```python
# action = extracted action name (e.g., "LEFT", "UP", "MOUSE 4 7")
# Map to engine action via action_names.py
# Call game.step_env(engine_action)
# Return new frame
```

### After Per-Turn Loop

When game ends (level complete or game over):

1. Score updated (quadratic formula)
2. Next level auto-loads (if multi-level game)
3. History carried forward (but level-specific state resets)
4. Diagnostics saved periodically

### Soft Deadline Enforcement

Every turn checks:

```python
if soft_end_time and datetime.now() > soft_end_time:
    # Graceful shutdown: finish current game, exit loop
    break
```

**Local run** (FALSE_SUBMISSION): `soft_end = now + 9h 50m` (budget 10h, buffer 10m)

**Real submission** (TRUE_SUBMISSION): `soft_end = now + 11h 20m` (hard wall 12h, buffer 40m)

### Output During Play

```
[game_id: train_abc123]
  [level 0]
    turn 1: action UP
    turn 2: python (segmentation analysis)
    turn 3: action MOUSE 5 8
    [level complete, score 100/115]
  [level 1]
    turn 4: action LEFT
    ...
```

Plus per-turn diagnostics writes (if not `TAAF_MINIMAL_DIAGNOSTICS`).

---

## Cell 7b: Dup-Game Gate & Game List Reassignment

**Part of Cell 7, runs after offline/live branch but before play**

### Purpose

- **Offline**: ensure transfer graft test can fire (needs 2 games of same family)
- **Real submission**: inert (gated on `not TRUE_SUBMISSION`)

### What It Does

If offline (not TRUE_SUBMISSION):

```python
try:
    first = bm.games[0]  # Load first game
    bm.games = bm.games[:3] + [
        GameAPI(
            env_name=first.env_name,              # Same environment
            arcade_spec=first.arcade_spec,        # Same spec (fingerprint family)
            external_game_id=f"{first.env_name}-dup",  # Distinct ID
        ),
    ]
except:
    # Fallback: just take first 4
    bm.games = bm.games[:4]
```

**Result**: 4 games, with games[0] and games[3] being the same family (clones)

**Why**: `transfer` graft checks if a sibling clone already solved deeper levels; the dup gives it something to replay

---

## Cell 8: Show Diagnostics

**Render HTML + display inline**

### What It Does

```python
from IPython.display import HTML, display

diagnostics_html = WORKING_DIR / "diagnostics.html"
if diagnostics_html.is_file():
    # Render diagnostics.html in an iframe (styles isolated)
    display(HTML(...))
else:
    print("No diagnostics.html — minimal diagnostics (real submission) suppresses it.")
```

### What's in diagnostics.html

- Run summary (total score, games played, time)
- Per-game transcript (moves, frames, scores)
- Inline base64-encoded frame images
- Linked MP4 videos (if created)
- Per-game analysis notes

### Submission Files

Written to `/kaggle/working/`:
- `submission.parquet` — **THE ONLY FILE KAGGLE READS**
- `diagnostics.html` — for your analysis
- `summary.txt` — plain-text summary
- `git_status.txt` — version tracking
- `tool_runtime_state.json` — last frame's runtime vars
- Intermediate state pickles (if enabled)

---

## Summary: Execution Timeline

```
┌─────────────────────────────────────────────────────────────┐
│ Cell 1: Set TRUE_SUBMISSION, env vars                       │
├─────────────────────────────────────────────────────────────┤
│ Cell 2: pip install arc-agi from wheels                     │
├─────────────────────────────────────────────────────────────┤
│ Cell 3-4: Mount datasets, add repos to sys.path             │
│           Run setup commands (START vLLM server here)       │
├─────────────────────────────────────────────────────────────┤
│ Cell 5: Unpickle benchmark & deployment target              │
├─────────────────────────────────────────────────────────────┤
│ Cell 6: EXPERIMENT HOOK — install grafts                    │
├─────────────────────────────────────────────────────────────┤
│ Cell 7: MAIN LOOP                                           │
│   [7a] Build game list (offline or live)                    │
│   [7b] Add dup-game (offline only)                          │
│   [7c] For each game, for each level:                       │
│        Per-turn loop: prompt → vLLM → parse → execute       │
│        Score update (THE FORMULA)                           │
│   [7d] Teardown (stop vLLM, etc.)                           │
│   [7e] Write submission.parquet                             │
├─────────────────────────────────────────────────────────────┤
│ Cell 8: Render diagnostics.html inline                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways for Editing

1. **Cell 6 is safe**: Toggle flags, customize before run
2. **Cell 7 is complex**: Read-only unless you understand the whole flow
3. **vLLM must be running**: Cell 4 starts it; if it fails, everything downstream fails
4. **TRUE_SUBMISSION is immutable**: Only Kaggle sets it; local runs always False
5. **Soft deadline protects against timeout**: Real submission has 40-min safety buffer
6. **Dup-game gate is inert on real submission**: Only fires locally
7. **Diagnostics tell you everything**: Check `diagnostics.html` for frame-by-frame transcript
