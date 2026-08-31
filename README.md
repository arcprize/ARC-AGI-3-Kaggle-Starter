# ARC-AGI-3 Agent Optimization Research

> A Qwen-led ARC-AGI-3 solver with a deliberately bounded online world-model sidecar, plus the historical graft experiments that informed it.

## Research Overview

This repository documents our end-to-end research pipeline for building, analyzing, and iteratively improving an autonomous AI agent that solves interactive video-game-style reasoning puzzles in the [ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3) competition. Our approach treats agent performance as a function of controllable variables — context memory capacity, prompt-level deliberation strategies, action-batch discipline, and cross-instance knowledge transfer — and applies rigorous single-variable ablation methodology to isolate causal effects on the competition scoring metric.

### Research Methodology

Our work follows a structured experimental pipeline:

1. **Hypothesis Formation**: Each experiment targets a specific bottleneck identified through transcript forensics, scoring formula analysis, or competitive intelligence from top-performing agents.
2. **Single-Variable Isolation**: We change exactly one configuration parameter per experiment to establish causal attribution (not just correlation) between changes and score movements.
3. **Quantitative Evaluation**: Every experiment is evaluated against the quadratic RHAE scoring formula: $\text{Score}_{\text{level}} = \min\bigl(115,\;(\text{baseline}/\text{actions})^2 \times 100\bigr)$, ensuring we measure what matters.
4. **Post-Mortem Analysis**: Failed experiments are analyzed for root cause — we extract more signal from regressions than from successes.

### Key Research Findings

| Finding | Method | Impact |
|---|---|---|
| **Recovery R2 probe is net-negative** | Controlled ablation (Exp 1 vs 2) | Injecting 16 scripted probe actions at `PROBE_MIN_ACTS=120` destroys ~97% of level score due to quadratic penalty. Proven causal. |
| **Context window 57344 is too aggressive** | Real Kaggle submission (Exp 4) | 88% context utilization causes throughput degradation. Optimal calibration: 51200 tokens (78% of vLLM server capacity). |
| **Schema grafts ARE active** | Source code audit (corrected prior assumption) | `schema_void.py`, `schema_notes.py`, `schema_helpers.py` all exist in the deployed dataset and are processed by `composite.py`. Previous documentation incorrectly classified them as inert. |
| **Multi-variable stacking masks causal signals** | Post-mortem of Exp 3 (0.82 regression) | Stacking `recovery` + `banking` + `transfer` + 3 schema flags in one submission made it impossible to attribute the 1.33→0.82 regression to any single change. |

## Experiment Results

| # | Configuration Delta | Kaggle Score | Direction | Key Observation |
|---|---|---|---|---|
| Baseline | `{efficiency, retry_guard, shortcircuit}` | **1.33** | — | Floor configuration |
| Exp 2 | + `recovery: True` | (local only) | ↓ 0.89→0.03 | R2 probe tax on tn36: 183→244 actions |
| Exp 3 | + `recovery` + `banking` + `transfer` + schema flags | **0.82** | ↓ 39% | Multi-variable confound; `recovery` dominant |
| Exp 4 | + `context_window: 57344` | **0.91** | ↓ 32% | Context too aggressive; throughput degraded |
| Exp 5 | + `context_window: 51200` + `schema_notes` + `schema_void` + `transfer` (NO recovery) | **PENDING** | Expected ↑ | Removes proven negatives, adds validated positives |

## Technical Architecture

The agent is built on the [Tufa Labs TAAF framework](https://www.kaggle.com/datasets/thtennant/taaf-kaggle-source-share-fork) with a modular graft composition system:

```
┌─────────────────────────────────────────────────┐
│  Benchmark.run()                                │
│  ├── ShortCircuitHarnessSolver (no-op trimmer)  │
│  │   └── SchemaVoidMixin (batch tail trimmer)   │
│  ├── TransferHarnessSolver (cross-clone replay) │
│  │   └── BankingHarnessSolver (win-then-replay) │
│  └── Analyzer Chain:                            │
│      ├── RetryGuard (outermost)                 │
│      └── EfficiencyToolAgent (innermost)        │
│          └── SchemaNotesToolAgent (prompt note)  │
│              └── ToolAgent + Qwen3.6-27B-FP8    │
└─────────────────────────────────────────────────┘
```

Each graft is individually flag-gated, blanket-guarded (any error degrades to stock behavior), and composable via the `composite.install(bm, flags={})` entry point.

### Current v20/v21 submission paths

The diagram above describes the historical graft research. The active Kaggle
candidate keeps the exact, measured v12 Qwen run as its primary planner and
adds no dependency on the unavailable `taaf_grafts` package.

```text
Qwen primary planner
        │ action/frame history
        ▼
online world model ──► movement + passability + click semantics + HUD filtering
        │
        ├── verified cross-level route/click: at most 4 single actions per level
        └── otherwise: return control to Qwen unchanged
```

The v20 sidecar waits for at least 24 history entries, then learns only from
observed transitions. At level boundaries it retains stable mechanics but clears
board-specific state. It can act only when it has strong evidence from an
earlier completed level: replaying a route toward a proven goal color, or
issuing one central click after a high-confidence `step` classification.

The separate v21 candidate adds the intended mental loop: it grades a learned
forward model against real, held-out movement transitions; searches candidate
routes in that model for free; and intervenes only if a complete rollout lowers
a learned objective. It then executes **one** predicted action, observes its
outcome, and re-plans. v20 skips even this mental-model observation path, so it
remains the conservative control. Neither candidate is a claim of private-score
improvement until independently submitted to Kaggle.

## Repository Structure

```
├── docs/
│   ├── IDEAS.md              # Prioritized experiment roadmap (11 experiments, 4 tiers)
│   ├── EXPERIMENT_LOG.md     # Chronological experiment results with quantitative analysis
│   ├── ARCHITECTURE.md       # Four-codebase architectural breakdown
│   ├── MODULES.md            # Per-file module documentation with flag mappings
│   └── QUESTIONS.md          # Open research questions with resolution tracking
├── experiment a/             # Exp 4: context_window=57344 (scored 0.91)
│   ├── sam-learning.ipynb    # Submission notebook
│   └── results (3).zip      # Full benchmark output (events, prompts, diagnostics)
├── experiment e/             # Exp 3: full stack with recovery (scored 0.82)
│   ├── sam-agent-33.ipynb    # Submission notebook
│   └── results (3).zip      # Full benchmark output
├── 1.33 scored in .../       # Reference baseline notebook (scored 1.33)
│   ├── arc3-duck-v12 (1).ipynb
│   └── archive (2).zip      # Source bundle with all graft modules
├── datasets/                 # Deployed Kaggle dataset bundle (taaf source share fork)
├── arc3x/                    # Online world-model sidecar
│   ├── percept.py            # Frame deltas, object masks, HUD filtering
│   ├── mind.py               # Movement and passability hypotheses
│   ├── clicks.py             # Learned click semantics
│   ├── dream.py              # Self-grading forward model and free rollouts
│   └── autopilot.py          # Conservative Qwen-compatible controller
├── tools/                    # Notebook build and validation tooling
├── create_experiment_notebook.py  # Automated notebook generator for experiments A-E
├── verify_grafts.py          # Local graft installation verification suite
└── parse_exp_*.py            # Benchmark result analysis scripts
```

## Scoring Formula & Optimization Target

The ARC-AGI-3 competition scores agents using a **quadratic efficiency-weighted** formula:

$$\text{Score}_{\text{level}} = \min\left(115,\;\left(\frac{\text{human\_baseline\_actions}}{\text{ai\_actions}}\right)^2 \times 100\right)$$

This means:
- Reducing actions from 200→100 yields a **4× score improvement**
- Reducing actions from 100→50 yields another **4× improvement**
- A single wasted action batch can destroy an entire level's score contribution

Our optimization strategy targets all three levers:
1. **Reduce wasted actions** (context window, schema void, shortcircuit)
2. **Improve action quality** (schema notes prompt steering)
3. **Multiply efficiency across clones** (transfer/banking cross-instance replay)

## Tools & Infrastructure

| Tool | Purpose |
|---|---|
| `create_experiment_notebook.py` | One-command notebook generation for any experiment config (A-E) |
| `verify_grafts.py` | Offline graft installation verification with mocked Kaggle dependencies |
| `parse_exp_a.py` / `parse_exp_e.py` | Automated benchmark.json score extraction and analysis |

## Competition Context

- **Competition**: [ARC Prize 2026 — ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
- **Current v12-derived candidates**: Qwen3.8-27B-FP8 served through the original Kaggle/vLLM setup
- **Agent Framework**: Tufa Labs TAAF (The Agent Architecture Framework)
- **Runtime**: ~2h 20m per submission, ~110 clones across ~25 game families

## Current state and local validation

The checked-in `arc3x` solver is the active research implementation. The
frame-0 marker detector remains experimental: its diagnostic currently passes
23/25 coverage but fails the source-grounded `tu93` check, so it is not wired
into the planner. This is intentional until an objective detector is validated.
The human-mind learner in `arc3x/mindgraft.py` now treats level changes as scene
cuts and excludes them from motion and geometry learning, preventing a new board
from poisoning the previous level's internal model.
`arc3x/pilot.py` also maintains a history-only progress ledger: colors that
ratchet down or up across ordinary play can become objective evidence without
hard-coding a game or color. This is guarded against large floods and clock-like
HUD counters, and has regression tests in `tests/test_progress_pilot.py`.
On laboratory levels it now also covers the learned-reachable map and tries
context-sensitive non-movement actions at adjacent unknown objects. Both rules
use only observed controls, geometry, and transitions. Board-specific state is
cleared at a level transition while the learned mechanics model is retained for
the next level.

The current local pilot benchmark is a development diagnostic, not a claim of
general solving. Under the fixed 25-game public suite and a 60-turn/3,000-action
pilot harness, it clears at least one level in **6/25** games and at least two
levels in **1/25**, for a mean score of **0.662**. The cleared games are
`lp85`, `ls20`, `m0r0`, `r11l`, `sc25`, and `tn36`; `tn36` clears two levels.
This is up from 3/25 and 0.156 before online click semantics. The pre-defined
public split is 4/17 clears on tuning games and 2/8 on the holdout, so the
change is not confined to the games used for iteration, but the private
competition score and all-games coverage remain unknown until Kaggle runs the
notebook. It does not yet meet a 10-game target.

Run the fast validation checks from the repository root:

```powershell
$env:PYTHONPATH = "."
.venv\Scripts\python.exe -m compileall -q arc3x tools
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe arc3x\why_markers.py --steps 400
```

The full scored suite is intentionally separate because it is CPU-intensive:
`arc3x\suite.py --split both -w 10 --budget 3000`.

To regenerate the v20/v21 Kaggle notebooks from the known 2.14-scoring v12 base:

```powershell
.venv\Scripts\python.exe tools\build_mind_notebook.py
```

This produces three notebooks:

- `1.33 scored in arc agi 3 competiotn in kaggle\arc3-duck-v20-v12-baseline-safety.ipynb`
  is code-identical to the v12 benchmark configuration, with only historic
  execution outputs cleared. Use this if maintaining the measured 2.14 private
  baseline is the priority.
- `1.33 scored in arc agi 3 competiotn in kaggle\arc3-duck-v20-v12-sidecar.ipynb`
  retains the same v12 Qwen3.8 model attachment metadata and source-bundle
  configuration, and adds a cautious sidecar after the first 24 observed
  transitions. The sidecar is limited to four single-action, high-confidence
  interventions per level and has no dependency on the later `taaf_grafts`
  import, which failed in the recorded v15 run.
- `1.33 scored in arc agi 3 competiotn in kaggle\arc3-duck-v21-v12-mental-simulation.ipynb`
  is the separate mental-simulation experiment. It preserves Qwen's opening,
  requires either eight held-out movement predictions or four held-out
  paint/teleport click predictions at >=80% accuracy, and executes only the
  first action of an objective-improving simulated plan before observing and
  re-planning.

The known private Kaggle score for v12 is **2.14**. The separate 5.04 value is
an offline six-game, four-pass diagnostic mean, not a private-leaderboard
result. The sidecar's private score remains unmeasured and must not be treated
as an improvement in advance. Submit the source-equivalent safety notebook
first, then run v20 and v21 as separate A/B experiments.

For a local, no-LLM diagnostic of the production `Pilot` path (not comparable
to Qwen or Kaggle scoring), run:

```powershell
.venv\Scripts\python.exe arc3x\pilot_harness.py tn36 --budget 300
```

It records how often the mental model has enough held-out predictive evidence,
how often it finds an internal plan, and why it abstains. This is the correct
place to improve observation/model learning before loosening the action gate.

## Kaggle submission description (623/750 characters)

> Architecture: a Qwen3.8 planning agent is the primary solver, wrapped by a lightweight online world-model sidecar. From action/frame history it induces movement rules, passability, click semantics (teleport, paint, widget, or inert), and masks HUD/countdown noise. It carries only stable mechanics across level boundaries while clearing board-specific state. The sidecar waits through the opening, then can take only a few single, model-verified actions toward a goal proven by an earlier level; otherwise Qwen continues unchanged. The aim is not a hard-coded game solver, but reusable causal knowledge learned during play.
