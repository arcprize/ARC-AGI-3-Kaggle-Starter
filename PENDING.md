# ARC-AGI-3 continuation handoff

Last updated: 2026-08-31

## Completed checkpoint

The guarded mental-simulation implementation is complete for this checkpoint
and was committed and pushed to both remotes.

- Commit: `ce97c67 feat: add guarded mental simulation solver`
- Branch: `codex/control-001-seed`
- Pushed to both `origin` and `personal` listed below.
- Working tree was clean immediately after the commit. Temporary local
  diagnostic JSON files were removed; their relevant findings are summarized in
  this document.

Completed in this checkpoint:

- Added v21 mental-simulation notebook generation and artifact.
- Wired the self-grading `Dream` model into the production `Pilot`.
- Added a one-step, observe-and-replan movement planner.
- Added held-out validation and one-step planning for predictable paint/teleport
  clicks.
- Preserved v20 as a conservative control by skipping mental observation there.
- Added a local Pilot-only diagnostic harness, unit tests, README documentation,
  and reproducible notebook safety/control checks.

## Goal

Build and evaluate the user's core loop for ARC-AGI-3:

```text
observe -> form hypotheses -> probe sparingly -> build a world model
-> simulate candidate plans internally -> execute a verified action
-> compare prediction with reality -> revise the model
```

The target is stronger generalization across unseen games and later levels, not
fitting to the 25 public games. Do not claim AGI, 10+ solved games, or a Kaggle
improvement until a scored evaluation supports it.

## Ground truth and controls

- Repository: `D:\AI_ARMY\arc_agi3_solver`
- Branch: `codex/control-001-seed`
- Remotes:
  - `origin`: `https://github.com/samrishtt/ARC-AGI-3-Kaggle-Starter.git`
  - `personal`: `https://github.com/samrishtt/arc-agi-3-kaggle-competition.git`
- Exact measured private Kaggle baseline: **v12, score 2.14**.
- Source notebook:
  `1.33 scored in arc agi 3 competiotn in kaggle/arc3-duck-v12-with-qwen-3-8-27b this one scored 2.14 .ipynb`
- The 5.04 figure is an offline six-game/four-pass diagnostic mean, **not** a
  private leaderboard score.
- `taaf_grafts` failed in a later recorded run. New notebook candidates must
  not depend on it.

## Submission artifacts

All are derived by `tools/build_mind_notebook.py` from the exact v12 source.

1. `arc3-duck-v20-v12-baseline-safety.ipynb`
   - Source-equivalent v12 with old outputs cleared.
   - Use this as the non-regression control.
2. `arc3-duck-v20-v12-sidecar.ipynb`
   - Conservative sidecar control.
   - Qwen keeps the opening; after 24 history entries it may issue at most four
     one-action interventions per level, only for a known prior-level goal route
     or a >=90% coordinate-free click rule.
3. `arc3-duck-v21-v12-mental-simulation.ipynb`
   - New experimental candidate implementing mental simulation.
   - Qwen keeps the opening. The sidecar acts only after its movement copy has
     at least 8 held-out, >=80% accurate movement predictions, or its
     paint/teleport click copy has at least 4 held-out, >=80% accurate effects.
     A complete internal rollout must lower a learned objective. It executes
     **one** action, observes, and replans.

Do not replace v20 or safety with v21. Submit/evaluate v21 as a separate A/B
candidate.

## Current implementation

- `arc3x/pilot.py`
  - Holds `Dream(self.mind.mech)` so the world model and mechanics learner share
    the same observed rules.
  - Grades each new history transition in the dream before `Mind.absorb` learns
    from it, preventing leakage from the answer into its confidence score.
  - Adds `_imagined_plan`: free search -> full rollout -> objective-improvement
    check -> route only if the model is confident.
  - In active mode `decide`, mental plans are allowed before heuristic routes.
  - In mental sidecar mode, `assist(... allow_imagination=True)` emits only the
    first action of a verified plan and then re-plans next turn.
  - Tracks mental checks, accepted plans, and rejection reasons in the per-game
    summary. v20 can explicitly skip dream observation, preserving it as the
    conservative control.
- `arc3x/autopilot.py`
  - Adds `ARC3X_PILOT_MODE=mental`; `sidecar` remains frozen v20 behavior.
- `arc3x/percept.py`
  - Makes `Volatility` grid-shape-safe for local harness/test boards rather than
    assuming 64x64 forever.
- `tools/build_mind_notebook.py`
  - Embeds `dream.py` and produces the v21 mental notebook in addition to v20
    and safety.
- `tests/test_mental_pilot.py`
  - Tests shared model wiring, confidence refusal, simulated objective-improving
    plan, and one-step/replan sidecar behavior.
- `arc3x/pilot_harness.py`
  - A local, no-LLM adapter that feeds the production `Pilot` only frame/action
    history and declared actions. It does not expose game snapshots or concrete
    engine action lists to the pilot. Use it to measure activation/abstention,
    not as a Kaggle score proxy.
- `arc3x/clicks.py`
  - Adds a held-out prediction ledger for paint/teleport effects. v21 can use a
    click simulation only after four predictions meet the same >=80% threshold;
    widgets, toggles, selects, and inert clicks still decline to predict.

## Validation already completed

```powershell
$env:PYTHONPATH = "."
.venv\Scripts\python.exe -m compileall -q arc3x tests
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Result: **23 tests passed**.

Static notebook check passed:

```text
v12 safety equivalence; v20 control isolation; v21 mental configuration: PASS
```

This proves notebook/source structure and unit-level gates only. It does not
prove that v21 improves Kaggle score or solves any number of games.

Pilot-only diagnostic completed on `tn36`, `lp85`, and `r11l` at a 300-action
cap: each cleared level 0 (17--24 actions), but that sample made **0/10** mental
plans. `tn36` is click-only, so its movement model has no avatar by design; it
needs a validated click rule before the v21 click planner can activate.

A complete 25-game, 300-action/60-turn pilot-only diagnostic also finished on
the movement-only v21 revision that preceded the click-prediction addition:
**5/25** games cleared level 0, mean **0.509**, and the mental planner proposed
**49/568** plans. Those plans were concentrated in `ka59` (39/59), `ls20`
(6/21), and `wa30` (4/16). This is useful activation evidence only: it used no
Qwen and predates the final click extension, so it must not be presented as a
Kaggle score or as a result for the final v21 notebook. A current-code rerun
confirmed `ka59` still proposes 39/59 mental plans before the interactive run
timed out on later games.

## Immediate next work

1. Run the pilot harness in bounded chunks for v21-related regression evidence
   without tuning on the holdout split. Start with tune games, preserve holdout
   as the final check:

   ```powershell
   $env:PYTHONPATH = "."
   .venv\Scripts\python.exe arc3x\pilot_harness.py --split tune --budget 300 --turns 60 --json pilot-tune.json
   .venv\Scripts\python.exe arc3x\pilot_harness.py --split hold --budget 300 --turns 60 --json pilot-hold.json
   ```

   If terminal time limits interfere, run small named batches and combine only
   completed reports. Do not change policy after inspecting holdout results.

2. Run the older full local suite separately for broad standalone-agent
   regression evidence (it does not directly exercise the wrapped `Pilot`):

   ```powershell
   $env:PYTHONPATH = "."
   .venv\Scripts\python.exe arc3x\suite.py --split both -w 10 --budget 3000
   ```

   Record tune (17 games) and holdout (8 games) separately. Compare against the
   existing documented active-pilot diagnostic: 6/25 >=1-level clears, 1/25 >=2,
   mean 0.662. That historical figure is not a Qwen/Kaggle comparison.

3. Inspect the pilot-harness traces for three questions before changing policy:
   - How often do the movement or click model become predictive from Qwen/pilot
     history?
   - How often does `_imagined_plan` find a route and how often is its first
     predicted action correct?
   - Which games fail because there is no learned objective, a wrong world model,
     or dynamics that make multi-step planning unsafe?

4. If the dream rarely activates, improve *evidence collection*, not confidence
   thresholds. Potential safe additions:
   - Record prediction metrics per game/level in pilot summaries.
   - Let Qwen's own action history yield labelled non-movement/use effects.
   - Add modelled click/use effects only after a held-out prediction test.

5. If the dream activates but first-step accuracy is poor, keep one-step replan
   and improve `Dream.project`/dynamics recognition; do not enable multi-action
   blind execution.

6. Re-run unit tests and notebook static equivalence after each code change.
   Commit only task-related files and push both remotes after a clean diff.

## Safety and communication rules

- Preserve user changes and never reset/overwrite the v12 source notebook.
- Use `apply_patch` for edits.
- The v12 safety notebook must remain source-equivalent to v12 (apart from
  cleared historic outputs/execution counts).
- State experimental claims precisely: local diagnostics, Kaggle score, and
  private generalization are different measures.
- The research-summit description should describe this as a model-based,
  hypothesis-driven interaction study—not as achieved AGI.
