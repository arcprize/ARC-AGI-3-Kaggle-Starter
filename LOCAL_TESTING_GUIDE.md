# How to Test Your Edits Locally with Full Diagnostics

**Complete workflow from edit → local test → diagnostics review**

---

## Overview

You can test edits **locally on your machine** (not Kaggle) with full competition dataset:

- **4 offline games** (from bundled environment files)
- **Full diagnostics** (HTML with transcripts, frame images, etc.)
- **Free** — doesn't spend submissions
- **Fast iteration** — run end-to-end in ~10-20 minutes (GPU dependent)

---

## Step 1: Set Up Your Python Environment

### One-Time Setup

**Terminal**:
```powershell
cd d:\AI_ARMY\arc_agi3_solver
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt  # If exists
# OR manually:
pip install jupyter numpy matplotlib pandas
```

You already have `.venv` activated (I can see it in terminal context), so skip this.

### Verify Activation

```powershell
python --version
pip --version
```

Should show Python 3.10+ and pip from `.venv`.

---

## Step 2: Make a Simple Edit (Example: Disable Recovery Graft)

### File: `1.33 scored in arc agi 3 competiotn in kaggle/arc3-duck-v12 (1).ipynb`

**Find Cell 6** (contains "Customization hook" or "install(bm, flags=")

**Current code**:
```python
install(bm, flags={
    "efficiency": True,
    "retry_guard": True,
    "shortcircuit": True,
    "recovery": True,  # ← REMOVE THIS LINE
})
```

**After edit**:
```python
install(bm, flags={
    "efficiency": True,
    "retry_guard": True,
    "shortcircuit": True,
})
```

**Save the notebook** (Ctrl+S in VS Code)

---

## Step 3: Convert Notebook to Python Script (Optional but Easier)

### Why?
- Jupyter can be slow locally
- Python script is faster to iterate
- Easier to debug

### Do This

**Terminal**:
```powershell
jupyter nbconvert --to script "1.33 scored in arc agi 3 competiotn in kaggle/arc3-duck-v12 (1).ipynb" --output=test_run.py
```

This creates `test_run.py` in the same folder.

**Alternative**: Keep it as notebook, just open in Jupyter and run.

---

## Step 4: Prepare Your Environment

### Key: Tell the Script This is a Local Test (Not Real Submission)

**Terminal** (set env var):
```powershell
$env:KAGGLE_IS_COMPETITION_RERUN = "0"
$env:TAAF_RUN_AS_SUBMISSION = "0"
$env:TAAF_MINIMAL_DIAGNOSTICS = "0"
```

This tells the system:
- `TRUE_SUBMISSION = False` (uses 4 offline games, not 25 competition games)
- Full diagnostics are written
- Baselines are visible

### Verify

```powershell
echo $env:KAGGLE_IS_COMPETITION_RERUN
# Should output: 0
```

---

## Step 5: Run Locally

### Option A: Run Jupyter Notebook (Easiest)

**Terminal**:
```powershell
cd "d:\AI_ARMY\arc_agi3_solver\1.33 scored in arc agi 3 competiotn in kaggle"
jupyter notebook "arc3-duck-v12 (1).ipynb"
```

Then:
1. VS Code opens Jupyter UI
2. Click "Run All" or run cell-by-cell
3. Watch output in terminal

### Option B: Run Python Script (Faster, Less Memory)

**Terminal**:
```powershell
cd "d:\AI_ARMY\arc_agi3_solver\1.33 scored in arc agi 3 competiotn in kaggle"
python test_run.py
```

---

## Step 6: Wait for Completion

### What to Expect

**Terminal output** (while running):

```
taaf.kaggle: TRUE_SUBMISSION=False
taaf.kaggle: source bundle = <path to datasets folder>
taaf.kaggle: input paths = {...}
taaf.kaggle: wrote taaf_kaggle_sources.pth (3 source roots)
[setup command] ...installing wheels...
[setup command] ...starting vLLM server...

TAAF_GRAFTS FEATURES={'efficiency': True, 'retry_guard': True, 'shortcircuit': True} API_VERSION=1
[efficiency] armed
[retry_guard] armed
[shortcircuit] armed

[game_id: train_abc123]
  [level 0]
    turn 1: analyzing...
    turn 2: executing action UP
    turn 3: analyzing...
    ...
[final score for train_abc123: 87/115]

[game_id: train_def456]
  ...

[Finished all games]
```

### Runtime

- **First run**: 15-30 min (downloading model, compiling)
- **Subsequent runs**: 5-15 min (model cached)
- **GPU**: Much faster (required, won't work on CPU alone)

### Output Location

```
<notebook folder>/
├── diagnostics.html       ← OPEN THIS IN BROWSER
├── summary.txt            ← Plain text summary
├── submission.parquet     ← Not scored locally
├── git_status.txt
└── server_recording/      ← Video recordings (if enabled)
```

---

## Step 7: View Diagnostics

### Open diagnostics.html in Browser

```powershell
# Windows
start diagnostics.html

# Or manually navigate to:
# d:\AI_ARMY\arc_agi3_solver\1.33 scored in arc agi 3 competiotn in kaggle\diagnostics.html
```

Browser opens, shows:

```
┌─────────────────────────────────┐
│ ARC-AGI-3 Run Diagnostics       │
├─────────────────────────────────┤
│ Run: arc3-duck-v12              │
│ Status: Completed               │
│ Total Score: 287/460 (62%)      │  ← YOUR AGGREGATE SCORE
├─────────────────────────────────┤
│ [Game 1: train_abc123]          │
│   Level 0: 100/115 (10 actions) │  ← Per-level breakdown
│   Level 1: 87/115  (13 actions) │
│   Total: 187/230               │
│                                 │
│   [Click for transcript]        │  ← Links to full transcript
│   [Click to view frames]        │
├─────────────────────────────────┤
│ [Game 2: train_def456]          │
│   ...                           │
└─────────────────────────────────┘
```

### Click into a Game

Shows:

```
Turn-by-turn transcript:
  Turn 1: Agent analyzed frame
    Action: UP
    Result: Board changed, no level complete
    Tokens: 450 input, 120 output
    
  Turn 2: Agent analyzed frame, ran Python
    Python code: segmentation analysis
    Output: [3 objects detected]
    Action: MOUSE 4 7
    Result: Level completed! Score: 100/115
    
  Turn 3-5: Level 1 attempts...
    [similar detail]
```

Plus frame images (ARC grids rendered with colors).

---

## Step 8: Compare Results

### Before & After Your Edit

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Total Score | 287/460 | ? | ? |
| Turn Count | 142 | ? | ? |
| Recovery Probes Fired | 12 | 0 (disabled) | ? |
| Efficiency Warnings | 15 | ? | ? |

Look for:
- **Score improvement** ✓ (goal)
- **Fewer actions per level** ✓ (efficiency)
- **Fewer stalls/recoveries** ✓ (if disabled recovery)

---

## Common Issues & Fixes

### Issue: `ModuleNotFoundError: arc_agi not installed`

**Fix**: The setup commands in Cell 4 install it. If they fail:

```powershell
pip install arc-agi  # Try manual install
```

### Issue: vLLM Server Doesn't Start

**Fix**: Check if port 1234 is in use:

```powershell
netstat -ano | findstr :1234
# If in use, kill it:
taskkill /PID <PID> /F
```

Then rerun.

### Issue: `CUDA out of memory`

**Fix**: You need an NVIDIA GPU with ≥8GB VRAM. This won't run on CPU alone.

Check GPU:
```powershell
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

### Issue: Takes Too Long / Hangs

**Fix**: Timeout is set to ~10 min soft deadline (local) vs 11h 20m (real).

If it hangs after that:
1. Press Ctrl+C in terminal
2. Check last frame in `diagnostics.html`
3. Look for infinite loops in agent transcript

---

## Step 9: Iterate

Once you see diagnostics:

1. **If score improved**: Great! Try next edit
2. **If score worsened**: Revert edit
3. **If mixed (some games better, some worse)**: Investigate via transcripts

### Make Another Edit

1. Edit Cell 6 or other code
2. Save notebook / Python file
3. Rerun (`python test_run.py` or Run All in Jupyter)
4. Check new `diagnostics.html`

---

## Full Example Workflow

**Terminal**:
```powershell
# Setup
cd "d:\AI_ARMY\arc_agi3_solver\1.33 scored in arc agi 3 competiotn in kaggle"

# Set local test mode
$env:KAGGLE_IS_COMPETITION_RERUN = "0"
$env:TAAF_RUN_AS_SUBMISSION = "0"
$env:TAAF_MINIMAL_DIAGNOSTICS = "0"

# Convert to script (optional)
jupyter nbconvert --to script "arc3-duck-v12 (1).ipynb" --output=test_run.py

# Run it
python test_run.py

# Wait 10-20 minutes...
# When done:

# Open diagnostics in browser
start diagnostics.html

# Check summary
cat summary.txt

# If good, edit Cell 6 and rerun
# OR check transcripts for clues if bad
```

---

## What the 4 Offline Games Are

From: `datasets/arc-prize-2026-arc-agi-3/environment_files/`

These are the **competition's official games**, running locally (no Kaggle server needed):

- Identical to real competition
- Baselines visible (can tune exactly)
- Smaller batch (4 vs 25) for fast iteration
- Real scoring applied

---

## Diagnostic HTML Key Sections

| Section | What It Shows |
|---------|---------------|
| **Run Summary** | Total score, games played, time |
| **Per-Game Summary** | Score per game |
| **Game Transcript** | Turn-by-turn: action, result, tokens |
| **Frame Images** | ARC grids (colors rendered) |
| **Agent Notes** | Efficiency warnings, recovery probes |
| **Timings** | Per-turn, per-level, per-game |

### Most Useful for Debugging

Click into a game → scroll to a bad level → see:
- What actions agent took
- What frame changes occurred
- Whether level was completed
- Token usage per turn
- Any errors/timeouts

---

## Quick Checklist

- [ ] Edit Cell 6 (or other code)
- [ ] Save notebook
- [ ] Set env vars: `KAGGLE_IS_COMPETITION_RERUN=0`, `TAAF_MINIMAL_DIAGNOSTICS=0`
- [ ] Run: `python test_run.py` or Jupyter notebook
- [ ] Wait for completion (~15 min first run, ~5 min cached)
- [ ] Open `diagnostics.html` in browser
- [ ] Compare scores to baseline (1.33 local, 1.28 real submission)
- [ ] If improved: iterate; if worsened: revert

---

## To Compare Against Baseline

Your current local baseline:

```
Score from DETAILED_ANALYSIS.md:
- Current best local: 1.33 (arc3-duck-v12)
- Current best real submission: 1.28
```

After your edit, check if:
- New score > 1.33 (improvement) ✓
- New score < 1.33 (regression) ✗

If regression, check `diagnostics.html` to see which game(s) degraded.

---

## Next: Your First Edit

1. **Disable recovery graft** (recommended first test)
   - Edit Cell 6, remove `"recovery": True`
   - Run locally
   - Compare score in diagnostics.html

2. **Expected result**
   - May see slight improvement (recovery was net-negative)
   - OR no change (recovery was doing nothing)
   - Should NOT see huge decline (recovery rarely helped)

3. **If improved**
   - Keep it
   - Try next edit (widen context window)

4. **If worsened**
   - Revert
   - Try different edit

---

## TL;DR — Commands to Copy-Paste

```powershell
# Set local mode
$env:KAGGLE_IS_COMPETITION_RERUN = "0"
$env:TAAF_MINIMAL_DIAGNOSTICS = "0"

# Go to notebook folder
cd "d:\AI_ARMY\arc_agi3_solver\1.33 scored in arc agi 3 competiotn in kaggle"

# Convert to script
jupyter nbconvert --to script "arc3-duck-v12 (1).ipynb" --output=test_run.py

# Run
python test_run.py

# Open results (when done)
start diagnostics.html
```

Done! You now have:
- ✅ Local test environment
- ✅ Diagnostics (HTML transcripts, scores, frame images)
- ✅ Fast iteration loop (~5 min per test after first)
- ✅ Risk-free (no Kaggle quota spent)
