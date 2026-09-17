# ARC Prize 2026 — Local Dev Starter

**Go from zero to your first Kaggle submission in about 10 minutes, without
ever opening the Kaggle notebook editor.**

This is a starter kit for the [ARC Prize 2026 — ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
competition. You'll edit one Python file on your laptop, see it actually play
the real game environments locally, and push it to Kaggle as a submission with
a single command.

No Docker. No `submission.json` to hand-write. No copy-pasting between your
editor and a notebook.

---

## What you need before you start

- **Python 3.12** (the competition's `arc-agi` package requires it)
  - macOS: `brew install python@3.12`
  - Ubuntu: `sudo apt install python3.12 python3.12-venv`
  - Windows: install from [python.org](https://www.python.org/downloads/)
- **git** (to clone the official agent framework)
- **A Kaggle account** with the competition rules accepted
  ([accept here](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/rules))

That's it. No GPU required for the starter agent.

---

## Quick start

```bash
# 1.  Clone this repo and step in
git clone https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter.git
cd ARC-AGI-3-Kaggle-Starter

# 2.  Drop your Kaggle API token (kaggle.com → Settings → Create New Token)
#     into the project-local .kaggle/ folder (NOT your home directory)
mkdir -p .kaggle && echo "KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" > .kaggle/access_token
chmod 600 .kaggle/access_token

# 3.  One-time setup: venv, dependencies, framework
make setup

# 4.  Open agent/my_agent.py to see the random-action starter, then edit
#     it to make a better submission. Add more agent/*.py modules if needed.

# 5.  Run it locally against every game in the competition (takes seconds)
make play-local

# 6.  Push it to Kaggle as a submission notebook
make submit

# 7.  Watch the run
make status

# 8.  When status shows "complete", open the notebook on kaggle.com,
#     find your kernel, click "Submit to Competition" in the top
#     right, and pick `submission.parquet` from the Output File
#     dropdown. That's one of your 5 daily submissions.
```

That's the entire loop. Steps 4–7 are what you'll repeat as you iterate;
step 8 is the deliberate moment when you spend a daily submission.

---

## The one file you edit: `agent/my_agent.py`

This is the only file you normally touch. It defines a class called `MyAgent`
with two methods:

```python
class MyAgent(Agent):
    def is_done(self, frames, latest_frame) -> bool:
        """Return True when your agent wants to stop playing."""
        ...

    def choose_action(self, frames, latest_frame) -> GameAction:
        """Look at the game state and return the next action."""
        ...
```

The starter version picks random actions — a baseline that proves your whole
pipeline works end-to-end. Replace the body of `choose_action` with your
strategy. Everything else (Kaggle plumbing, submission file format, game
orchestration) is handled for you.

For larger agents, place supporting Python modules beside `my_agent.py` and
use package-relative imports, for example `from .memory import Memory`. The
notebook builder packages every `agent/*.py` file into an offline
`submission_agent` package while retaining the same `MyAgent` entry point.

---

## What happens when you run `make submit`

The competition is a *code* competition: you submit a notebook, Kaggle runs it
twice.

```diagram
   make submit
       │
       ▼
   ┌─────────────────────────────────────┐
   │  Kaggle Phase A: Save & Run All     │
   │  ─ Runs your notebook in their      │
   │    real environment                 │
   │  ─ Validates that your code         │
   │    executes without errors          │
   │  ─ make status shows "complete"     │
   └─────────────────┬───────────────────┘
                     │
                     │  You click "Submit to Competition"
                     │  on the kernel page
                     ▼
   ┌─────────────────────────────────────┐
   │  Kaggle Phase B: Competition Rerun  │
   │  ─ Your agent actually plays the    │
   │    hidden game set                  │
   │  ─ Your leaderboard score appears   │
   └─────────────────────────────────────┘
```

`make submit` builds and uploads the notebook (Phase A). After
`make status` reports `complete`, open the kernel on kaggle.com and click
**"Submit to Competition"** to enter Phase B and get a leaderboard score.

> **You only get 5 official submissions per day**, so it pays to be
> confident before you submit: get `make play-local` passing, then submit.

> **Heads up:** Before your first `make submit`, open
> [`notebooks/kernel-metadata.json`](notebooks/kernel-metadata.json) and
> replace `REPLACE_WITH_YOUR_USERNAME` with your Kaggle handle. The Makefile
> will refuse to push until you do.

### Choosing an accelerator

The notebook is generated with a **T4 GPU** by default (matches Kaggle's
sample submission). To change it, open
[`scripts/build_notebook.py`](scripts/build_notebook.py) and edit **one
line** near the top:

```python
ACCELERATOR = "t4"     # change "t4" to one of: cpu, t4, p100, rtx6000
```

Then re-run `make submit`. That's it — both the notebook metadata and
[`notebooks/kernel-metadata.json`](notebooks/kernel-metadata.json) get
updated automatically.

| Value | Hardware | When to use |
|---|---|---|
| `"cpu"` | No GPU | The random starter, or any non-ML agent |
| `"t4"` | Nvidia T4 ×2 | **Default.** Small models, fast iteration |
| `"p100"` | Nvidia P100 | Single big-memory GPU |
| `"rtx6000"` | Nvidia RTX 6000 (`g4-standard-48`) | Heavy ML; **ARC-AGI-3 exclusive**, burns GPU quota faster |

RTX 6000 is reserved for ARC-AGI-3 notebooks only — don't use it for early
iteration. All accelerated Kaggle sessions have internet disabled, which is
already the default in this kit.

---

## All the commands

| Command | What it does |
|---|---|
| `make setup` | One-time install: Python venv, `arc-agi`, `kaggle` CLI, clones the framework |
| `make play-local` | Runs your agent against every game in the dataset, locally |
| `make play-local GAME=ls20` | Same, but only one game (faster while debugging) |
| `make verify-local` | 30-second smoke test on two games |
| `make list-games` | Print every game id available |
| `make pull-sample` | Download the official sample agent for reference |
| `make notebook` | Build the Kaggle notebook from your agent (no push) |
| `make submit` | Build the notebook **and** push it to Kaggle |
| `make status` | Check the status of your most recent Kaggle run |
| `make clean` | Remove the venv, downloads, and generated notebook |

---

## Why this setup, instead of editing in the Kaggle notebook?

Three reasons:

1. **Iteration speed.** Editing in your normal IDE, then `make play-local`,
   gives you a real-game-engine feedback loop in seconds. The Kaggle editor's
   loop is *minutes* per change.
2. **No environment surprises.** The local `arc-agi` PyPI package hosts the
   same game engine the Kaggle gateway runs. If it works locally, it works on
   Kaggle.
3. **Your code stays in git.** Notebooks are awful for diffs and code review.
   Here your real work lives in [`agent/my_agent.py`](agent/my_agent.py); the
   notebook is just an auto-generated deployment artifact.

---

## Project layout

```
.
├── agent/
│   ├── my_agent.py             ★ Required MyAgent entry point
│   └── *.py                    Optional supporting modules
├── scripts/
│   ├── play_local.py           Runs your agent against real games
│   ├── build_notebook.py       Packages your agent into a Kaggle notebook
│   └── slim_framework.py       Trims framework deps so install is light
├── notebooks/
│   ├── kernel-metadata.json    Edit once: your Kaggle username
│   └── submission.ipynb        Auto-generated, never edit by hand
├── vendor/                     Cloned framework (gitignored)
├── .venv/                      Python 3.12 venv (gitignored)
├── .kaggle/                    Your project-local Kaggle token (gitignored)
└── Makefile
```

---

## Troubleshooting

**`make setup` fails: `python3.12: command not found`**
Install Python 3.12 — the `arc-agi` package requires it. macOS:
`brew install python@3.12`.

**`make submit` says "edit kernel-metadata.json"**
You haven't replaced `REPLACE_WITH_YOUR_USERNAME` in
[`notebooks/kernel-metadata.json`](notebooks/kernel-metadata.json) yet.

**`make submit` says `401 Unauthorized`**
Your Kaggle token is missing or invalid. Generate a fresh one from your
[Kaggle Settings page](https://www.kaggle.com/settings) and overwrite
`.kaggle/access_token`.

**`make play-local` says "Could not create environment"**
Your machine couldn't reach the ARC-AGI API to download the game source on
first run. Check your internet, then try again — once downloaded, games are
cached in `environment_files/` and you're fully offline.

**My local score is 0.0**
That's expected for the random starter agent. Your job is to make it
non-zero. 🙂

---

## Where to go next

- Read the [ARC-AGI-3 docs](https://docs.arcprize.org/) to understand the
  benchmark.
- `make pull-sample` to study Kaggle's reference agent (the same one
  currently sitting on the leaderboard).
- The competition's [discussion forum](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion)
  for community Q&A.

Good luck. Looking forward to seeing what you build.
