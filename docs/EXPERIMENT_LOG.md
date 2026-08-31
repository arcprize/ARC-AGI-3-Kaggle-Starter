# Experiment Log
# EXPERIMENT_LOG.md — arc3-duck-v12

Format: config -> result -> what we learned. Only real runs, real numbers.

---

## Baseline progression (leaderboard, pre-this-session)
0.86 (initial forge-based agent) -> 0.50 (vLLM liveness bug) -> 0.62 (liveness+BFS fix)
-> 0.35 (reflection fix backfired under latency contention) -> 0.96 (switched to
Tufa Labs TAAF harness) -> 1.32 -> **1.33 (starting point for this log)**

Config at 1.33: `{"efficiency": True, "retry_guard": True, "shortcircuit": True, "context_window": 57344}`

---

## Experiment 1 — Local validation of the 1.33 config
**Config:** `{"efficiency": True, "retry_guard": True, "shortcircuit": True}` (unchanged)
**Result:** mean 0.89

---

## Experiment 2 — `recovery: True` added
**Result:** mean 0.03 (regression vs. 0.89 due to R2 probe tax at action 120)

---

## Experiment 3 — Real submission: recovery + banking + transfer + schema flags
**Kaggle Leaderboard Result: 0.82** (regression due to `recovery` probe tax)

---

## Experiment 4 — Context Window 57344 (without schema grafts)
**Kaggle Leaderboard Result: 0.91** (regression due to 57K token latency penalty)

---

## Experiment 5 (Experiment B Folder) — Context 57344 + Schema Notes + Schema Void + Transfer
**Kaggle Leaderboard Result: 0.66**

---

## Experiment 6 (Experiment C Folder) — Stock 32K Speed + Schema Helpers + Schema Void + Transfer
**Kaggle Leaderboard Result: 1.06**
**Local 4-Game Mean Score: 2.5101**

---

## Experiment 7 (Experiment D Folder) — Context 45056 + Schema Helpers + Schema Void + Transfer
**Kaggle Leaderboard Result: 0.95**

---

## Experiment 8 (Experiment F Folder) — Inline State Deduplication Graft (`state_dedup`)
**Kaggle Leaderboard Result: 0.77**

---

## Experiment 9 (Results 4) — Schema Notes Only
**Kaggle Leaderboard Result: 0.47** (destroyed reasoning flow)

---

## Experiment 10 (Results 5) — Banking Only
**Kaggle Leaderboard Result: 1.10**
**Local 4-Game Mean Score: 1.3190**

---

## Experiment 11 (New Folder / sam agi) — Level 2+3 Tools/Architecture + Banking + 57K Context
**Config:** `find_path` + `find_objects` + `death_memory` + post-death cleanup + `banking` + `schema_helpers` (57K context)

**Kaggle Leaderboard Result: 0.60** (Kaggle vLLM latency timeout regression)
**Local 4-Game Mean Score: 2.6848 (HIGHEST LOCAL SCORE EVER! 🏆)**

| Game | Local Score | Levels | Actions | Key Breakthrough |
|---|---|---|---|---|
| `tn36` | **10.71** | **2/7** | 467 | Record high score on `tn36`! |
| `m0r0` | **0.02** | **1/6** | 477 | **UNLOCKED!** First time `m0r0` cleared Level 0! |
| `sk48` | 0.00 | 0/8 | **105** | Actions reduced from 940+ to 105! |
| `sk48-dup` | 0.00 | 0/8 | 934 | |

**Key Diagnostic:**
Experiment 11 delivered our **highest local score ever (2.6848)** and unlocked `m0r0` for the first time. However, combining 57K context with verbose prompt notes caused vLLM prefill timeouts on Kaggle's shared GPUs, stranding games at 0 score.

**Calibration Fix:** Keep `find_path`, `find_objects`, `death_memory`, and post-death cleanup, but use **Stock 32K Context** (`32768`) and a 1-line prompt note to eliminate Kaggle latency.
