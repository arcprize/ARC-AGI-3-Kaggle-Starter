# HypothesisWorld research plan

## Primary question

Can explicit hypothesis competition and active falsification improve completion and
human-relative action efficiency in unseen ARC-AGI-3 environments over novelty-only and
non-hypothesis world-model baselines?

## Testable claims

1. Prediction disagreement produces more diagnostic early actions than novelty alone.
2. Explicit contradiction tracking avoids repeating disproven action-effect rules.
3. Compact object-level outcomes transfer across visually distinct states better than
   exact frame transitions.
4. Switching from experiments to short-horizon planning after confidence rises improves
   action efficiency without catastrophically reducing exploration.
5. Salience-conditioned coordinate actions dominate uniformly sampled coordinates.

## Measurement

Primary outcomes are levels completed and official Relative Human Action Efficiency.
Secondary outcomes are actions to first progress, deaths, unique compact states,
hypotheses falsified, calibration of predicted outcomes, prediction entropy, and the
fraction of actions chosen primarily for information versus progress.

Compare all six ablations with fixed seeds and identical game/version/action budgets.
Report per-game results rather than only an aggregate, and keep public-game tuning
separate from held-out conclusions. Re-run any stochastic baseline across multiple seeds.

## What is research and what is scaffolding?

Research mechanisms:

- simultaneous falsifiable outcome hypotheses with explicit contradiction records;
- prediction-disagreement as an actionable approximation to information gain;
- confidence-dependent arbitration between experimentation and planning;
- object/context-conditioned proposals for coordinate experiments;
- rejected-hypothesis memory with evidence-based reconsideration.

Engineering scaffolding:

- connected-component perception and greedy object matching;
- bounded counters, hashes, event logs, configuration flags, and notebook packaging;
- an empirical transition table and depth-two graph planner;
- official framework lifecycle handling and synthetic regression tests.

The scaffolding is intentionally replaceable. A useful experiment should change one
research mechanism while leaving the official evaluation path and measurements stable.

## Known failure modes to watch

- A dominant colour is not always semantic background; camouflage or large objects can
  be mis-segmented.
- Same-colour touching objects merge under connected-component perception.
- Animation timing may make a transient effect look like a stable causal outcome.
- Exact state fingerprints fragment experience when counters or cosmetic pixels change;
  object abstraction can create the opposite failure by aliasing semantically different
  screens.
- Greedy matching is ambiguous among repeated identical objects.
- Outcome hypotheses describe correlations and may confuse preconditions with effects.
- Sparse reward and irreversible actions can make active experiments too expensive under
  the squared action-efficiency metric.
- The depth-two planner cannot compose unseen transformations or infer distant goals.

## Next five experiments

The first stable-state pilot replaced exact learning keys with dual raw/abstract
identities and added alias-aware planning confidence. It improved the three-seed
`vc33` full-agent completion result, but is not evidence of broad generalization; see
[`experiments/reports/2026-09-15-stable-state-pilot.md`](experiments/reports/2026-09-15-stable-state-pilot.md).

1. Replace the fixed border heuristic with learned stable masks, evaluated on more
   games with a preregistered alias-risk threshold.
2. Add relational hypothesis templates (`touching`, `inside`, `aligned`, `same colour`)
   and compare causal prediction calibration against outcome-only hypotheses.
3. Estimate expected information gain by posterior rollouts rather than entropy of point
   predictions; compare diagnostic actions per unit of RHAE cost.
4. Learn reversible action models and use `ACTION7` or safe reset-aware counterfactual
   experiments where the framework exposes them.
5. Add latent-state finite-state hypotheses for history-dependent effects, evaluated on
   held-out games with preregistered thresholds and seeds.
