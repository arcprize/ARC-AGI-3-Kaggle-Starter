# tufa-arc-agi-framework (TAAF)

ARC-AGI3 orchestration framework — Tufa Labs.

The driver document for this repository is [`docs/requirements.md`](docs/requirements.md).

## Quick start

```bash
make install-dev      # set up .venv + dev deps
make prepare          # ruff format + lint + pyright + pytest
```

See [`notebooks/demo_workflow.ipynb`](notebooks/demo_workflow.ipynb) for a happy-path tour of `Game`, `Solver`, and `Benchmark`.

## Kaggle Random Smoke

`SolverRandom` is TAAF's built-in solver example, so its Kaggle launcher lives
in this repo. The default Make target runs the 25 official games once,
CPU-only, with no action cap, and a 10-minute notebook runtime.

```bash
make kaggle-random \
  KAGGLE_RANDOM_RUN_NAME=taaf-random-smoke \
  KAGGLE_RANDOM_KERNEL_SLUG=taaf-random-smoke \
  KAGGLE_RANDOM_DATASET_REF=driessmit1/taaf-kaggle-source-random-smoke
```

For a local package check without uploading to Kaggle:

```bash
make kaggle-random KAGGLE_RANDOM_DRY_RUN=true KAGGLE_RANDOM_JOB_DIR=/tmp/taaf-random-dry-run
```

The direct CLI is also available:

```bash
.venv/bin/python -m taaf.kaggle_random \
  --kernel-slug taaf-random-smoke \
  --dataset-ref driessmit1/taaf-kaggle-source-random-smoke
```

Example Kaggle outputs:

- [Random smoke notebook](https://www.kaggle.com/code/driessmit1/taaf-random-kaggle)
- [Duck-Tok solver notebook](https://www.kaggle.com/code/driessmit1/taaf-duck-tok-0527-1712)

## re-arc dependency

`pyproject.toml` depends on `arc-agi-3-local` from the `re-arc-3` Git
repository, so CI and package installs do not require a sibling checkout:

```bash
make install-dev
```

Developer sandboxes that need local `re-arc-3` changes can override that Git
dependency in their own environment.
