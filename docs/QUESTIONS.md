# Questions

# QUESTIONS.md — arc3-duck-v12

Genuinely open. Not rhetorical, not solved elsewhere in these docs.

- ~~**Where did `schema_void`, `schema_notes`, `schema_helpers` come from?**~~
  **RESOLVED (2026-07-24):** All three modules exist in the deployed dataset
  bundle (`taaf source share fork (banking)/src/taaf-grafts/taaf_grafts/`):
  - `schema_void.py` (16KB) — surprise-abort on mixed commit batches
  - `schema_notes.py` (7KB) — static probe→observe→commit prompt note
  - `schema_helpers.py` (20KB) — preloaded sandbox grid-analysis helpers
  And `composite.py` handles all three explicitly (lines 208-213, 300-313,
  335-340). The earlier conclusion that they were "inert/missing" was wrong.
  See IDEAS.md for experiment plans using each one.

- **What exactly does the game engine's level-transition wipe reset vs.
  preserve?** `recovery.py`'s R3 handoff assumes `cross_level_notes` is the
  one surviving channel, but the actual wipe logic lives in `game.py` /
  `game_api.py` and hasn't been read in detail. If more state survives than
  assumed, R3 might be solving a smaller problem than it thinks.

- **Is `agent_ext.py`'s heuristic baseline proxy well-calibrated?** On the
  real submission, `_resolve_baselines` returns `None` for every game and the
  efficiency note falls back to a synthesized target. Never checked whether
  that heuristic is a reasonable stand-in for the real (hidden) baseline, or
  systematically off in one direction.

- **What does `family_store.py` actually store, and is it keyed correctly
  across the real ~110-clone structure?** Read at docstring level only.
  Before trusting a `transfer` test result, worth confirming the store's key
  scheme actually matches how clones are identified on the real graded run
  (vs. the manufactured `-dup` game used for local testing).

- **Why are m0r0 and sk48 structurally hard**, beyond "the agent stalls"?
  You've played both and called them easy — worth writing down *why* they're
  easy for a human (what's the actual mechanic/insight) as a concrete
  contrast against what the transcripts show the agent trying. That gap is
  probably more informative than any graft-level fix.

- **`retry_guard`'s health probe** — confirmed it exists and what bug it
  fixes (unbounded 1s retry against a dead local vLLM server), but the exact
  probe/backoff behavior under partial degradation (slow-but-alive server,
  not fully dead) hasn't been traced through.

- **What does `python_tool_sandbox.py` actually expose to the model** beyond
  `segmentation.py`? Only the bootstrap/isolation mechanics were read, not
  the full surface of what state/helpers the model can call each turn.

## Stage 4 candidate

`inference/agent/tool_agent.py` is the strongest pick for the "understand one
module completely" exercise — it's the actual orchestrator every graft wraps
or subclasses, and understanding its turn loop precisely (what state it reads,
what it writes back, exactly where a graft can and can't intervene) would
make every future graft experiment easier to reason about in advance instead
of after the fact.