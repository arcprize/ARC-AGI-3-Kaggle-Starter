# Next session — queued 2026-08-23

State: uncommitted work on `codex/control-001-seed`, off commit `1944e2a`.

## 0. Run these four things first, in this order

**Nothing below §1 has executed.** The Bash tool's safety classifier was
unavailable for essentially the whole 2026-08-23 session (one command in ~15 got
through), so `relive.py`, the `agent.py` edits and `smoke_relive.py` have not even
been import-checked. No claim is made for any of them.

```bash
# (a) does it even import
PYTHONPATH=. .venv/Scripts/python.exe -c "import arc3x.agent, arc3x.relive, arc3x.smoke_relive; print('import ok')"

# (b) is the mechanism doing what it says - 6 games, ~4 min, not a score measurement
PYTHONPATH=. .venv/Scripts/python.exe arc3x/smoke_relive.py

# (c) is max-pooling the right coarsening, or did we just assume it
PYTHONPATH=. .venv/Scripts/python.exe arc3x/why_cells.py --steps 200

# (d) the only thing allowed to make a claim about score
PYTHONPATH=. .venv/Scripts/python.exe arc3x/suite.py --split both -w 10 --budget 3000 > scratch/suite_relive.log

# (e) the objective-detector claim - INDEPENDENT of relive, run it even if (a) fails
PYTHONPATH=. .venv/Scripts/python.exe arc3x/why_markers.py --steps 400
```

(e) touches none of `relive.py`, `agent.py` or the search, so it is the one
measurement above that survives (a) failing. Read three numbers from it, in this
order of importance:

1. **"reactive never fired at all in: N/25 games."** Those are games where the
   planner has no destination for the entire budget, and a frame-0 proposal is the
   only thing that can change it. This is the number the whole idea rests on.
2. **The `tu93` row must not say `FAIL`.** It is the only row graded against
   pixels read from the game's own source (a 3×3 of colour 14, `tu93.py` 396-445).
   Coverage elsewhere does not redeem a failure there.
3. **`agrees with the reactive path`.** Corroboration by an independent route, not
   proof — `Dream.target_colors` is the thing being replaced and has its own
   documented phantom-success history. `cd82` and `re86` are expected to read
   `n/a region-match`; a detector that appeared to succeed on those would be
   matching something it does not understand.

What to read out of (b): `cells/action` **must be well under 0.7** — at 1.0 the key
is bijective, novelty carries no signal and the search is a paid random walk, which
is the measured failure `coarsen` exists to fix. Then `restarts/action` (high means
the budget went on rewinding, not searching), `drift` (should be ~0), `pool=`,
`coarsen=`, and the `x base` column per cleared level. It prints a loud warning if
the baselines came back as the silent `[100] * n` fallback, in which case every
ratio it prints is meaningless.

Compare (d) against **tune 0.177 / hold 0.005** (ratio 0.03). A mechanism that helps
tune and not hold has been fitted to games it was allowed to see.

## 1. The rule discovered on 2026-08-23, because it reorders everything below

> **An action spent on a level you go on to clear costs score quadratically. An
> action spent on a level you never clear costs nothing but budget.**

Measured from the notebook's own event log. On tn36, `exp_11` split its 583 events
**31 / 69 / 483** across levels 0 / 1 / 2: cleared the first two off 100 actions,
then poured 483 into a level it never cleared, and scored **10.71 — exactly the
2-of-7 completion cap**. `exp_e`, same game and same depth, spent **fewer** total
actions (329) and scored **1.36**, because it dithered through the two it did clear.
More actions, 7.9× the score. m0r0 is the rule inverted: 477 actions, level 0
*cleared*, score 0.02 of an available 4.76.

An earlier version of this file and of `ARCHITECTURE.md` carried an "implied speed"
column back-derived from the scores. It was wrong — the measured action counts run
the other way — and it has been deleted everywhere it appeared.

Three things follow, all now written into `PLAN.md` Phase 3 and `ARCHITECTURE.md`
§2(b):

1. **Burn level 0, and burn any level you are failing.** Level 0's weight is 1 of
   21–55, at most 4.8% of a game. A level you never clear costs nothing at all.
2. **Be crisp only where it pays** — levels 1 and 2 must land at ≤1.4× baseline,
   which means the *model* has to be doing the work by then. That is what
   `on_new_level`'s carry-over was built for and it has never been exercised.
3. **`agent.py`'s search throttle keys on the wrong variable.** It reads
   `RELIVE_ACTIONS if L == 0 else RELIVE_ACTIONS // 4`, keying on the level *index*,
   when the deciding quantity is what the level has *already* cost: adding `m`
   actions to a level `n` old multiplies its score by `(n/(n+m))²`, baseline-free,
   marginal cost falling as `1/n³`. So it throttles a hopeless level 2000 actions
   deep exactly as hard as a fresh one. Replacement shape:
   `m = clamp(ALPHA/(L+1) × spent_on_level, floor/(L+1), RELIVE_ACTIONS)`.
   **Left unimplemented on purpose** — ALPHA and the floor are guesses and
   `relive.py` has not run once. **This is the first A/B after (d) above.**

## 1b. The win-condition census, read 2026-08-23 — what the games actually want

All 25 games ship as source, so every guard on a `self.next_level()` was read
rather than guessed. 13 are readable predicates and **10 of the 13 are one shape**:

> every object of kind A must be co-located with an object of kind B.

`ka59` (two pairs at once), `s5i5`, `lp85` (two), `tu93`, `wa30`, `dc22`, `m0r0`,
`ls20` (sequenced). `cd82` and `re86` are the same idea at pixel granularity — a
canvas must equal a reference picture. The rest: eliminate a colour (`cn04`,
`r11l`), a local neighbour constraint (`ft09`), or a flag set elsewhere.

**So the destination is drawn on the board before the first action** — and
`Dream.target_colors` is `(collectible | prog.consumed) - retired`, both of which
are *reactive* and stay empty until the agent has accidentally succeeded twice.
That is the failure `progress.py` opens with: `dc22 move=100%(153) collect=[]
thought=0 levels=0`. A perfect forward model with nowhere to go. New
[markers.py](arc3x/markers.py) proposes a destination from frame 0; new
[why_markers.py](arc3x/why_markers.py) is (e) above. **Not wired into
`target_colors`, on purpose, until (e) has run.**

Two things this census already settled without spending an action:

- **`tu93` is not sokoban.** Tag `0017unajnymcki` is sprite `0016ihgrljrgpq`, a
  3×3 of colour 9 with a `4` at `[0][1]`, taken as `[0]` with `set_rotation()`
  called on it — it is the *avatar with a facing marker*. So tu93 is "avatar
  reaches an exit", one mover to N exits. Direct source confirmation that facing
  sprites rotate, which is the rotating-sprite item in §3.
- **The queued cd82 plan was wrong.** See §3.

## 2. Never judge a change on the 4-game benchmark

`exp_e` (1.36) and `exp_11` (10.71) ran on **identical repo commits** and
**byte-identical `taaf_setup_env.json`**. Worse, the graft flags live only in
notebook cell 6, and `context_window: 57344` is applied **in-process** — so the
env json reports `32768` for every run regardless. The knob most suspected of
causing wasted actions is the one the bundle never recorded.

**Fixed:** cell 6 now defines `GRAFT_FLAGS` once and writes `graft_flags.json` into
`/kaggle/working`. Consequence for planning: "graft tuning is 0-for-11" was an
underpowered experiment, not eleven refuted ideas. Use `suite.py --split both`, or
the leaderboard.

## 3. Still open from the previous queue

- ~~**ACT buttons plannable.**~~ **Done.** `Mechanics.acts` exposes buttons that
  change the board without moving the avatar; `Mechanics.moves` filtered
  `d != (0,0)` and threw all of them away. 12 of 25 games have one; `cd82` and
  `tr87` have **nothing else** (`planner-sees=[]`). Both consumers now exist:
  `act_round` (agent.py:580) and the use-branch of `push_frontier`
  (agent.py:718), which is *walk there then press use*.
  ~~*press-and-see* in frame space for the two avatar-less games~~ — **retracted
  2026-08-23 after reading `cd82.py` in full (782 lines).** cd82 is not a
  press-and-see game, it is a *paint-the-canvas* game, and press-and-see would
  have burned the budget on it:
  - a 10×10 all-zero **canvas** at (27,34); a 10×10 **target pattern** at (3,3),
    different per level; win is `np.array_equal(canvas[mask], target[mask])` with
    `mask` zeroing **both diagonals`.
  - 5×5 palette swatches set the paint colour; 8 basket slots move around a 3×3
    ring; **ACTION5 stamps a half-plane** (`[0:5,:]`, `[5:10,:]`, `[:,0:5]`,
    `[:,5:10]`) or a triangle for diagonal slots; ACTION6 on a bucket stamps a
    3×4 edge strip.
  - `_get_valid_actions` is **state-gated** — just clicked → ACTION5 only;
    counter ≥6 → palette clicks only; ≥3 → moves + palette + bucket. **ACTION5 is
    not in the default list**, so a press-and-see sweep never even sees the stamp.
  - a 100-action limit calls `self.lose()`, and `render_interface` writes
    `frame[63, x] = 4 or 5` — **direct source confirmation of the draining
    progress bar** that defeats `Volatility.hud_mask`'s 90%-frequency test and
    produced cd82's 240 phantom progress clicks.

  cd82 needs the **region-match** detector (static reference picture vs changing
  workspace, objective = mismatch count over `Volatility.changes`), not a button
  sweep. Specified, not written; see §1b.
- **Rotating-sprite matching.** `wa30` and `sc25` have four working MOVE buttons and
  the planner sees `[2, 4]`. `percept.moved_objects` requires an exact rigid shift,
  so a sprite that turns to face its direction of travel produces *zero* votes on
  the axis it is not facing. Match on centroid/bbox displacement with a shape-change
  tolerance. Same root cause as the worst move accuracy on the set (43%, 38%).
  **Deprioritised 2026-08-23:** `mind.py` already carries two workarounds for this
  same gap — `_convention()` (line 362) and `_fill_deltas()` (line 410) — and the
  button convention they encode was measured at **exactly 0.00**. Controls were
  never the bottleneck; the destination was. `why_moves.py` is still un-run, so the
  wa30/sc25 attribution remains unconfirmed even though tu93's source proves facing
  sprites exist in the set.
- **Hypothesis elimination.** Factorise rather than enumerate: 6 maps × 8 goals × 10
  block-sets × 5 collision rules × 4 act-effects = 9,600 candidates held as ~33
  independent facts, refuted per billed action. Live-pool size then scales with how
  ambiguous the game still is. **CPU candidates at ~1 ms, not LLM subagents** — an
  LLM at 17.6 s/action × 110 games does not fit the 9 h budget, and that is exactly
  what turned experiment 11's 2.68 local into 0.60 on Kaggle.
- **Convergence — but for the corrected reason.** All 25 games spend the full
  3000-action cap. Per §1 that is not itself the waste: the cost is only the *later*
  levels those actions could have bought. So the missing test is not "spend less",
  it is "**stop spending on a hypothesis already known to be dead, and re-aim**".
  Two games show the shape: one drains its objective to **zero** and then dies,
  another converts one colour into another two pixels per click for hundreds of
  actions. Both make monotone progress on a quantity that is not the win condition.

## Constraints that still hold

- Build from 2.14; do not restart from scratch. arc3x standalone is **0.142** and
  ships as a component source, never as the agent.
- No per-game hardcoding.
- Always report coverage out of 25 next to any mean — 23 zeros hide inside 0.142.
- The holdout (`cd82 ft09 lf52 m0r0 s5i5 sk48 tn36 vc33`) is fixed and never
  re-drawn.
- Only graft into the notebook if the local number beats 2.14.
