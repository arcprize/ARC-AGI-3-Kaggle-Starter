# Stable-state abstraction pilot — 2026-09-15

## Question

Can an object-centric learning key reuse transitions across cosmetic frame changes
without causing the planner to overgeneralize?

## Change

Each observation now has two identities:

- a raw full-grid fingerprint retained for audit and alias measurement;
- an abstract fingerprint containing dimensions, inferred background, completed-level
  count, and exact descriptors and relations for non-border-connected objects.

Border-connected objects remain available to perception and action generation. They
are omitted only from the learning key. Planning confidence is the empirical transition
confidence multiplied by `1 / sqrt(raw variants represented by the abstract state)`.
Predictions below the existing 0.68 threshold are not used for planning.

The ablation runner now records the seed in every run and preserves stdout/stderr event
order. Selection and outcome records include the game ID, ablation, and seed.

## Protocol

- Official local ARC-AGI-3 service and scorecard
- Games: `ls20`, `vc33`
- Budget: 100 actions per game
- Agent seeds: 0, 1, 2
- Raw logs: generated under `experiments/results/` and intentionally gitignored
- Command:

```bash
.venv/bin/python experiments/run_ablations.py \
  --games ls20,vc33 --steps 100 --ablations F --seeds 0,1,2
```

## Results

The initial abstraction made planning overconfident. Before alias-aware confidence, the
full agent completed `vc33` in only one of three seeds:

| Full agent, before alias gate | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| `vc33` levels | 1 | 0 | 0 |
| `ls20` levels | 0 | 0 | 0 |

After adding alias-aware confidence:

| Full agent, after alias gate | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| `vc33` levels | 2 | 1 | 1 |
| `ls20` levels | 0 | 0 | 0 |
| Aggregate score | 2.3507 | 0.0547 | 0.1400 |

The active-hypothesis variant without planning completed `vc33` in all three seeds
(2, 1, and 1 levels), confirming that the pre-gate regression was introduced by
planning rather than the experiment selector alone.

Representative post-gate seed-0 diagnostics:

| Game | Raw states | Abstract states | Largest raw alias set | Decisions with nonzero planning value |
|---|---:|---:|---:|---:|
| `vc33` | 91 | 13 | 47 | 11 / 100 |
| `ls20` | 95 | 51 | 2 | 49 / 100 |

## Interpretation

The pilot supports two narrow conclusions:

1. Cosmetic or interface variation was fragmenting `vc33` experience: the abstract
   key compressed 91 observed raw states to 13 learning states in the representative
   run.
2. Transition consistency alone was insufficient evidence for planning. A heavily
   aliased self-transition could appear perfectly predictable and trap the policy;
   representation confidence removed that failure in this pilot.

It does **not** establish general ARC-AGI-3 improvement. The sample contains only two
public games and three internal agent seeds. The fixed border heuristic may discard a
task-relevant border object, and `ls20` still has zero completed levels. The next
evaluation should expand the game set and compare fixed-border abstraction with a
learned temporal stability mask using a preregistered alias-risk criterion.
