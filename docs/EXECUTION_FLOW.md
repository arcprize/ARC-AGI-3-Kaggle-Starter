# Execution Flow

# EXECUTION_FLOW.md — arc3-duck-v12

Chronological walkthrough of one run, tied to actual notebook cell numbers.
See ARCHITECTURE.md for the structural (box) view.

## 1. Environment & submission mode (notebook cell, "## 1")
```python
TRUE_SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
os.environ["TAAF_RUN_AS_SUBMISSION"] = "1" if TRUE_SUBMISSION else "0"
os.environ["TAAF_MINIMAL_DIAGNOSTICS"] = "1" if TRUE_SUBMISSION else "0"
```
This single boolean, set only by Kaggle's own grading system (never by us clicking Save & Run All), controls everything downstream:
- whether baselines are visible to the efficiency graft
- whether periodic diagnostics writes happen
- which game list gets played (see step 6 below)
- soft-deadline behavior (step 7)

**This is why local "Save & Run All" runs are safe** — `TRUE_SUBMISSION` is always `False` unless Kaggle itself is doing the official rerun after "Submit to Competition" is clicked.

## 2. Install ARC runtime
Installs the `arc-agi` package and dependencies from the bundled offline wheelhouse (no internet in accelerated Kaggle sessions).

## 3-4. Locate and import source bundle
Finds the attached Kaggle dataset by marker file (mount path varies), adds `taaf`, `inference`, and `taaf_grafts` to `sys.path`, then runs each repo's setup commands — this is where the vLLM server actually boots and the Qwen3.6-27B-FP8 model loads (`max_model_len=65536`).

## 5. Load the benchmark
Unpickles the `Benchmark` object (`bm`) — the games list, solver config, job_dir — and repoints its output paths at `/kaggle/working`.

## 6. Customization hook / graft install
```python
install(bm, flags={...})
```
This is the cell we've been editing all session. `install()` is defined in `taaf-grafts/composite.py`; it reads the flags dict and:
- swaps in `EfficiencyToolAgent` if `efficiency: True`
- builds the analyzer chain (`recovery` → `retry_guard`) if those flags are set
- swaps the solver class if `banking`/`transfer`/`shortcircuit` are set
- patches `_LOCAL_ANALYZER_CONTEXT_WINDOW` if `context_window` is set
- prints a `TAAF_GRAFTS FEATURES={...}` banner and per-graft `[x] armed` lines — **this banner is the only reliable confirmation a flag actually took**, since any internal failure here degrades silently to stock (wrapped in try/except at the cell level).

**Right after this**, a separate "dup-game commit gate" reassigns `bm.games` — but only when `not TRUE_SUBMISSION`:
- it clones `games[0]` into a second `Game` object sharing the same `arcade_spec` but a distinct `game_id` (via `GameAPI.external_game_id`)
- this exists specifically so the `transfer` graft's family-store logic has something to fire against locally, since a real submission naturally has clones (see step 6b) but a local run otherwise wouldn't
- on any construction fault it falls back to `bm.games[:4]`
- **this entire block is provably inert on a real submission** — it's gated on `not TRUE_SUBMISSION`

## 6b. What the real competition rerun plays instead
Per `inference/framework/kaggle.py`, the full public set is `DUCK_HARNESS_PUBLIC_GAME_IDS` (25 games). The actual private evaluation clones these into ~110 runs (~4.4 clones per game family) — this is the structural reason the `transfer` graft exists at all: later clones of the same family can skip to the deepest solved level for free if a sibling already solved part of it.

## 7. Play the benchmark
`bm.play()` → `HarnessSolver.run_games()` → for each game, for each pass:
- loop: build prompt → graft chain → vLLM call → parse tool call → (optional) Python sandbox detour → engine action → score update
- outside a real submission, the run stops ~10 minutes before the wall-clock budget for a graceful exit; inside a real submission it runs to the actual deadline
- teardown commands run even if the run raises

## 8. Show the diagnostics
`taaf.diagnostics.generate_run_html` / `generate_run_summary_txt` build `diagnostics.html` and `summary.txt` from the finished benchmark state. `taaf.deploy_kaggle` writes `submission.parquet` — the only file Kaggle's grading actually reads.

## 9. (Manual, outside the notebook) Submit to Competition
Clicking this button on the kernel page is the action that:
- sets `KAGGLE_IS_COMPETITION_RERUN=1` for a fresh rerun of this exact notebook version
- makes `TRUE_SUBMISSION` evaluate `True` this time, switching to the real 25-game / ~110-clone set with hidden baselines
- spends one of the daily submissions
- produces the actual leaderboard score

This is the only step in the whole pipeline that costs anything real — every step 1-8 above is repeatable for free (aside from Kaggle's weekly GPU-hour quota).