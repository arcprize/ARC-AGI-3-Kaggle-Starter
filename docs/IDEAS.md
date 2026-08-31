# Ideas & Experiment Roadmap

# IDEAS.md — arc3-duck-v12 (v2, 2026-07-24)

**Goal:** Push the leaderboard score from **1.33** to **2.0+**.

**Principle:** Score = $\min\bigl(115,\;(\text{baseline}/\text{actions})^2 \times 100\bigr)$ per completed level.
Every experiment must either (a) reduce actions on already-solved levels, or (b) unlock
new level completions. The quadratic penalty makes action reduction **2× more
valuable** per unit of effort than reasoning improvements that don't translate
to fewer actions.

**Experiment discipline:** ONE variable per submission. Validate locally first.
Move to EXPERIMENT_LOG.md once actually run.

---

## ⚡ Tier 1 — High Confidence, Zero-Risk (notebook Cell 6 only)

> These require no source edits and no new dataset version. They change only the
> `flags={}` dict in Cell 6 of the notebook. Each can be tested locally for free.

### Experiment A: Context Window Expansion (HIGHEST PRIORITY)

**What:** Set `"context_window": 51200` (or `57344`) in the flags dict.

**Why it works:**
- `_LOCAL_ANALYZER_CONTEXT_WINDOW` defaults to `32768` tokens while vLLM serves
  `max_model_len=65536`. The agent is running at **half its available context**.
- Every losing transcript shows the same pattern: re-testing an action it already
  tried, or contradicting an earlier conclusion — classic context truncation
  artifacts (see ARCHITECTURE.md Stage 3 bottleneck notes).
- `composite.py` line 270-274 handles this: `_patch_context_window(resolved)` directly
  reassigns `tool_agent._LOCAL_ANALYZER_CONTEXT_WINDOW` at graft install time.
  This is read at every per-game `ToolAgent` construction, so it lands.
- **No new actions injected.** More context → fewer repeated hypotheses → fewer
  wasted probes → lower action count on games the agent already mostly solves.

**Risk:**
- Larger context = slower per-turn generation → fewer total turns within the
  11h 20m wall clock. Monitor `tokens/sec` and `total_actions_reached`.
- Mitigation: try `51200` first (78% of server max), not the full `65536`.

**Cell 6 code:**
```python
install(bm, flags={
    "efficiency": True,
    "retry_guard": True,
    "shortcircuit": True,
    "context_window": 51200,
})
```

**Success signal:** Locally, watch for fewer `[repeated hypothesis]` patterns in
transcripts and a higher mean score than the Experiment 1 baseline (0.89).

---

### Experiment B: Schema Notes — Probe→Observe→Commit Prompt Pressure

**What:** Add `"schema_notes": True` to flags (requires `"efficiency": True`).

**Why it works:**
- `schema_notes.py` exists (verified in the deployed dataset bundle) and is
  handled by `composite.py` lines 300-306. It subclasses `EfficiencyToolAgent`
  and appends a ~100-token **SCHEMA LOOP** note to every prompt turn:
  ```
  1) Unsure what a control does? Commit only 1-3 probe actions, then STOP.
  2) Prefer one short planned sequence over raster scans or long repeats.
  3) If the last batch mostly no-op'd or oscillated, change your hypothesis.
  4) Before a long sequence, state the expected board change in one line.
  ```
- This directly addresses the "raster scanning" and "brute-force one action"
  failure modes visible in the m0r0 and sk48 transcripts.
- **Report-only** — never injects or aborts actions. Pure prompt steering.
- Top competitors ("The Duck", "Forge") explicitly credit structured
  explore→verify→commit loops as their primary score driver.

**Risk:** Very low — adds ~100 tokens/turn, never touches the action path.
Worst case: neutral. Cannot cause a regression.

**Cell 6 code:**
```python
install(bm, flags={
    "efficiency": True,
    "retry_guard": True,
    "shortcircuit": True,
    "context_window": 51200,
    "schema_notes": True,
})
```

**Success signal:** Transcripts show shorter probe-then-act sequences instead of
long homogeneous action repeats. Per-game action count drops.

---

### Experiment C: Schema Void — Surprise-Abort Mixed Batches

**What:** Add `"schema_void": True` to flags.

**Why it works:**
- `schema_void.py` exists (verified) and is handled by `composite.py` lines
  208-213. It composes OVER `shortcircuit` in the session MRO.
- The vendored game loop runs mixed-action batches to the end even when a
  mid-batch action invalidates the plan's premise. Schema void adds two
  mechanical trim heuristics:
  **(D) Valid-actions collapse:** If any remaining action in the tail is no
  longer valid after a mid-batch engine step, drop the tail immediately
  instead of marching into guaranteed `invalid_action` breaks.
  **(E) Oscillation detection:** If a board returns to a grid already seen
  earlier in the same batch (after ≥2 changes), the plan is thrashing A→B→A
  and the tail is dropped.
- Both heuristics are **prefix-consistent**: they only drop un-executed tails,
  never inject, reorder, or replay actions.
- Each trimmed tail saves actions that would otherwise feed the quadratic
  penalty. This is the mechanical equivalent of what "The Duck" achieves via
  its world-model verification loop, but implemented at the harness level.

**Risk:** Low — proven prefix-consistent by unit tests. Worst case: no eligible
mixed batches occur and it's neutral.

**Cell 6 code:**
```python
install(bm, flags={
    "efficiency": True,
    "retry_guard": True,
    "shortcircuit": True,
    "context_window": 51200,
    "schema_notes": True,
    "schema_void": True,
})
```

**Success signal:** Transcript shows `[schema_void] trimmed N actions` lines.
Per-level action count decreases on games with mixed-batch plans.

---

## 🔧 Tier 2 — High Confidence, Requires Isolated Testing

> These involve grafts that have known positive mechanics but haven't been
> tested in isolation from the known-negative `recovery` graft.

### Experiment D: Banking — Win-then-Replay (Isolated)

**What:** Add `"banking": True` to flags (recovery stays OFF).

**Why it works:**
- `banking_solver.py` caches a winning action sequence when a level is
  completed. It then prunes intermediate exploratory actions from the trace
  and replays just the minimal winning plan on a fresh play of the same card.
- Directly targets RHAE: if a level was solved in 50 actions but the minimal
  winning sequence is 12 actions, the replayed plan scores
  $\min(115, (B/12)^2 \times 100)$ instead of $\min(115, (B/50)^2 \times 100)$.
- The 0.82 submission (Experiment 3 in EXPERIMENT_LOG.md) tested banking + recovery
  + transfer simultaneously, so banking's **individual effect is unknown**.

**Risk:** Medium — the replay mechanism could interfere with multi-level puzzle
dynamics where later levels depend on state built during earlier levels.

**Cell 6 code:**
```python
install(bm, flags={
    "efficiency": True,
    "retry_guard": True,
    "shortcircuit": True,
    "context_window": 51200,
    "schema_notes": True,
    "schema_void": True,
    "banking": True,
})
```

**Success signal:** Locally, `tn36` should show `[banking] armed` and a replayed
winning sequence with fewer actions than the initial solve. Look for
`[banking]` log lines in the transcript.

---

### Experiment E: Transfer — Cross-Clone Replay (For Real Submission)

**What:** Add `"transfer": True` to flags (implies banking automatically).

**Why it works:**
- The real competition runs ~110 clones (~4.4 per game family). Transfer
  round-robins clones: the first-solving "scout" publishes its pruned action
  segments to `_FamilyStore`; sibling clones adopt and replay them.
- If the scout solved levels 0-3 in 200 actions, siblings replay the pruned
  segments in ~40 actions total → massive RHAE improvement across 3-4 clones.
- This is the **highest theoretical leverage** graft — it multiplies a single
  solve's efficiency across the entire clone family.

**Risk:** Medium — `family_store.py` keying must match how clones are identified
on the real graded run. The local `sk48-dup` game exists specifically to test
transfer; check for `[transfer] adopted level...` in the transcript.

**Cell 6 code:**
```python
install(bm, flags={
    "efficiency": True,
    "retry_guard": True,
    "shortcircuit": True,
    "context_window": 51200,
    "schema_notes": True,
    "schema_void": True,
    "transfer": True,  # implies banking
})
```

**Success signal:** Locally, the `sk48-dup` game should show `[transfer] adopted
level 0 from sk48-d8078629` or similar. Action count for the dup game should be
dramatically lower than the original.

---

## 🧪 Tier 3 — Medium Confidence, Needs Source Edits (New Dataset Version)

> These require editing files in the source bundle and re-uploading as a new
> Kaggle dataset version. Higher effort, but address root causes.

### Experiment F: Fix Recovery R2 Probe Threshold

**What:** Edit `recovery.py` line 80: `PROBE_MIN_ACTS = 120` → `PROBE_MIN_ACTS = 400`

**Why it works:**
- EXPERIMENT_LOG.md proved that R2 probes fire on `tn36` around action 120 —
  a game that naturally solves by ~183 actions. The probe's 16 extra actions
  (183→244) cost ~97% of the level's score due to quadratic penalty.
- Raising the threshold to 400+ ensures probes only fire on genuinely stuck
  games (m0r0 at 883 actions, sk48 at 317+ actions) where the quadratic factor
  is already near-zero and even a small unlock would be net positive.
- R1 refresh (free, zero actions) and R3 handoff (free, zero actions) remain
  active regardless of this change.

**Risk:** Low — just moves the probe trigger further out. Probe still fires on
truly stuck games, just won't interfere with slow-but-working ones.

**After edit, Cell 6:**
```python
install(bm, flags={
    "efficiency": True,
    "retry_guard": True,
    "shortcircuit": True,
    "context_window": 51200,
    "schema_notes": True,
    "schema_void": True,
    "transfer": True,
    "recovery": True,  # NOW safe with raised threshold
})
```

**Success signal:** `tn36` transcript shows no R2 probe firing. `m0r0` or `sk48`
may still show probes, and if one of those probes unlocks a level, it's a huge
win (each level is worth an entire level weight in the score formula).

---

### Experiment G: Schema Helpers — Preloaded Sandbox Analysis Functions

**What:** Use `"schema_helpers": True` instead of `"schema_notes": True`.

**Why it works:**
- `schema_helpers.py` injects four pure analysis functions (`grid_diff`,
  `connected_components`, `action_effect_summary`, `recent_history`) into
  every Python sandbox call as a prelude.
- The model currently wastes tokens and tool turns rewriting the same grid
  plumbing from scratch every game (often with bugs). Pre-loaded helpers
  eliminate this overhead.
- Top competitors emphasize that structured grid analysis via Python REPL is
  the single highest-value capability. Making it frictionless should improve
  both reasoning quality and action economy.

**Risk:** Medium — the prelude shifts line numbers in tracebacks (cosmetic), and
`schema_helpers` and `schema_notes` are mutually exclusive (`composite.py`
lines 300-313: `elif`, not `if`). Must pick one.

> **Recommendation:** Try `schema_notes` first (Experiment B — prompt-only, zero
> risk), then swap to `schema_helpers` if the model is already probing well but
> writing buggy analysis code.

---

### Experiment H: State-Deduplication Action Filter (NEW GRAFT)

**What:** Build a new graft that maintains a hash table of observed board states
and prevents the agent from re-executing actions that produced no state change.

**Why it works (competitive intelligence):**
- Multiple top-scoring agents use hash-table deduplication to avoid revisiting
  unproductive sub-paths. The "Forge" agent tracks transition graphs to avoid
  repeating unsuccessful moves.
- Our `shortcircuit` graft already trims *consecutive* no-ops, but doesn't
  prevent the model from trying the same action again 50 turns later (after
  context truncation has forgotten the earlier failure).
- A state hash table would survive context truncation and mechanically block
  provably-wasted actions without consuming LLM tokens.

**Risk:** Higher — new code, new graft module. Requires careful composition
with existing solver grafts. Start with a read-only version that only WARNS
(like efficiency) before adding action blocking.

**Implementation sketch:**
```python
# In a new graft module, e.g. state_dedup.py
class StateDeduplication:
    def __init__(self):
        self._seen_states = {}  # hash(grid) → set of actions tried

    def should_skip(self, grid_hash, action):
        return action in self._seen_states.get(grid_hash, set())
```

---

## 🎯 Tier 4 — Speculative / Needs Investigation

### Experiment I: Heuristic Baseline Proxy Calibration

**Question:** Is `agent_ext.py`'s heuristic baseline proxy well-calibrated for
the real (hidden) baselines on the graded competition run?

**Why it matters:** If the proxy is systematically off (e.g. 2× too generous),
the efficiency nudge fires at the wrong times on the real run versus local
validation. This could explain why local-vs-real scores diverge (1.33 local
vs 1.28 real, then 0.89 local → 0.82 real).

**Investigation steps:**
1. Read `agent_ext.py`'s `_resolve_baselines` and the heuristic fallback in detail.
2. On the local offline games (where real baselines ARE visible), run with
   efficiency on and compare the heuristic proxy's per-level targets against the
   real baselines. Log both values.
3. If the proxy is >30% off on more than half the levels, recalibrate.

---

### Experiment J: Level-Transition State Wipe Analysis

**Question:** What exactly does the game engine's level-transition wipe reset
vs. preserve? `recovery.py`'s R3 handoff assumes `cross_level_notes` is the
one surviving channel, but the actual wipe logic in `game.py`/`game_api.py`
hasn't been read in detail.

**Why it matters:** If more state survives than assumed, there's a cheaper/bigger
lever for cross-level knowledge transfer than R3's current mechanism.

**Investigation steps:**
1. Read `tool_agent.py`'s `_update_summarized_knowledge_from_step_summary` in
   full to trace exactly which keys survive and which are wiped.
2. Map this against `recovery.py`'s `WIPED_KNOWLEDGE_KEYS` tuple.
3. If additional keys survive, consider using them for richer cross-level context.

---

### Experiment K: Multimodal Vision Context (Image Input)

**Question:** Does adding the rendered grid image alongside ASCII improve
reasoning on visually complex games?

**Why it matters:** The "Reki" agent (2nd place milestone) renders frames as
images and feeds them to the model. `vision_context.py` already exists in the
inference codebase, gated by a `MULTIMODAL_CONTEXT` env var.

**Risk:** High — Qwen3.6-27B-FP8 may not support or benefit from image input.
The extra tokens per turn could slow generation significantly. Only viable if
the model architecture supports multimodal input.

---

## 📋 Recommended Submission Order

Based on risk analysis, expected impact, and experiment isolation:

| Order | Experiment | Risk  | Expected Impact | What Changes |
|-------|-----------|-------|-----------------|--------------|
| **1** | **A** (context_window=51200) | Low | **High** — fixes root cause of wasted actions | Cell 6 flag only |
| **2** | **B** (schema_notes) | Very Low | **Medium** — nudges toward structured exploration | Cell 6 flag only |
| **3** | **C** (schema_void) | Low | **Medium** — mechanically trims wasted batch tails | Cell 6 flag only |
| **4** | **D** (banking) | Medium | **High** — replays minimal winning sequence | Cell 6 flag only |
| **5** | **E** (transfer) | Medium | **Very High** — multiplies efficiency across clones | Cell 6 flag only |
| **6** | **F** (fixed recovery) | Low | **Medium** — safe R2 probes on truly stuck games | Source edit |
| **7** | **G** (schema_helpers) | Medium | **Medium** — pre-loaded analysis code | Cell 6 flag swap |

> **Critical correction to previous docs:** QUESTIONS.md stated `schema_void`,
> `schema_notes`, and `schema_helpers` "don't exist in composite.py or any graft
> file." **This was WRONG.** All three modules exist in both the extracted
> archive AND the deployed dataset bundle, and `composite.py` handles all three
> explicitly (lines 208-213, 300-313, 335-340). The flags are NOT inert — they
> were inert only in the previous dataset version that was being analyzed.

---

## 🏆 The "Full Stack" Target Configuration

Once experiments A-E are validated individually, the optimal submission config is:

```python
install(bm, flags={
    "efficiency": True,
    "retry_guard": True,
    "shortcircuit": True,
    "context_window": 51200,
    "schema_notes": True,
    "schema_void": True,
    "transfer": True,      # implies banking
})
```

**Why NOT include:**
- `recovery`: Leave OFF unless Experiment F's threshold fix is validated.
- `schema_helpers`: Mutually exclusive with `schema_notes` (try after B is validated).

This configuration stacks **5 independent score-improvement mechanisms**:
1. Wider context → fewer repeated hypotheses → fewer wasted actions
2. Schema loop prompting → structured exploration → shorter action sequences
3. Schema void → mechanical batch trimming → saved actions on invalid plans
4. Banking → win-then-replay → minimal action sequences on solved levels
5. Transfer → cross-clone replay → multiplied efficiency across clone families
