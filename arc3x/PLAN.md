# ARC-AGI-3 solver — the plan, and where it stands

Companion to [ARCHITECTURE.md](ARCHITECTURE.md), which describes how the system
works. This one says what we are doing next, why that and not something else, how
long it should take, and how we will know if it worked.

Written 2026-08-23. Status lines are as measured, including the bad ones.

---

## 1. The one number that sets the whole plan

Everything below follows from the scoring formula, so it goes first.

A game's score is

```
level i score = min(115, (baseline_actions_i / actions_spent_i)² × 100)   if cleared
              = 0                                                        if not
game score    = min( Σ(score_i × i) / Σ i ,  (Σ cleared i / Σ all i) × 100 )
leaderboard   = mean over games
```

Level *i* carries weight *i*, and **every action spent while stuck on a level is
billed to that level and to no other.**

At exactly human-baseline speed each cleared level scores 100, so the weighted
mean equals the completion cap and **score = the completion cap.** Evaluated over
the real level counts of the 25 dev games (9 games have 6 levels, 5 have 7, 6 have
8, 4 have 9, 1 has 10):

| levels cleared per game | mean score at baseline speed |
|---|---|
| level 0 only | **3.52** — a hard ceiling |
| levels 0 + 1 | **10.57** |
| levels 0 + 1 + 2 | **21.14** |

Three consequences, and they are the entire argument for the ordering of the
phases below:

1. **The target of 10+ means clearing two levels per game, not five.** Much
   nearer than "10 out of 100" sounds.
2. **A solver that clears level 0 everywhere and level 1 nowhere cannot exceed
   3.52.** No amount of breadth or efficiency work gets past that. This is the
   single most important fact in the project and we computed it late.
3. **Depth buys efficiency slack quadratically.** Two levels needs ≤1.1×
   baseline to reach 10.57; three levels reaches 10.8 even at 1.4× baseline.
   Aiming one level deeper than strictly necessary is the cheaper route to the
   same score.

The current leaderboard entry is **2.14**, which is close to the level-0-only
ceiling. That is consistent with two very different regimes — "level 0 on ~60% of
games" or "two levels on ~5 games and nothing on 20" — and those need opposite
fixes. Resolving which is task 0c below.

## 2. Total cost

**~25–35 working hours**, plus ~9 h of Kaggle wall-clock per submission attempt.
Phases 0, 1, 3 and 4 are ordinary engineering and estimate reliably. **Phase 2 is
research and its hours could be 10 or 40.**

---

## Phase 0 — Instrumentation · 2 h · ✅ done

The highest value per hour in the project, and it paid off immediately.

| task | status |
|---|---|
| 0a. Parallelise the suite | ✅ 25 games at a 3000-action budget: ~55 min serial → ~8 min on 10 workers. Exact, not approximate: every game has its own engine and there are no writes. |
| 0b. A "why did it stop" channel | ✅ every strategy records its exit reason against the level it started on. |
| 0c. Same report against the 2.14 notebook | ⬜ **not done — the one Phase 0 item still open.** |

**0a had a trap worth recording.** The first parallel version regressed the
numbers rather than just the clock: one game managed 32 actions where serial did
401. Six workers had each sized a BLAS thread pool to all 12 cores, so 72 threads
thrashed over 12. Because `spawn` re-imports the module in the child, the thread
limits have to be set in the **parent, before the pool exists** — setting them
inside the worker is too late. After the fix, parallel reproduces serial action for
action.

Related: wall-clock cutoffs are now treated as **invalidating rather than
degrading**. Kaggle bills actions, not seconds, so a run cut short by a deadline is
a machine-dependent experiment and is not comparable to another run. The summary
says so loudly instead of averaging it in.

**What 0b found on its first run, which is the point of building it.** The stall
reasons said `click:change = 386`, then after a fix `click:progress = 13,393` on
the 8 holdout games — five of eight were doing nothing else. Tracing the objective
term by term on one action-by-action play found this:

```
a 23 act6(39,2)  consumed=[4]:150  obj=614400  gain=True
a 24 act6(39,2)  consumed=[4]:149  obj=610304  gain=True
a 26 act6(41,2)  consumed=[4]:148  obj=606208  gain=True
...  field=4096   ← the HUD mask is completely empty
```

Colour 4 fell by **exactly one pixel on every click**, 154 → 126. It is a draining
step-budget bar, and the agent was clicking at random, watching its own remaining
steps tick away, and scoring every tick as progress. Roughly 12,000 actions across
the holdout spent on that.

The failure arrived through the one door the existing guard did not cover.
`progress.py` already warned about exactly this in prose and defended it with the
volatility HUD mask — but volatility asks *"does this pixel change often"*, and a
bar that loses one pixel per action changes each individual pixel on about 1/154 of
frames. It never crosses the threshold. **cd82's mask is empty.**

Fixed with a second, count-based guard: a colour that moves on ≥55% of actions
**and** moves by exactly one pixel ≥75% of the time is a counter, not a set of
objects. Both conditions are needed — "moves nearly every frame" alone would reject
a collectible in a game where every move picks something up, and "moves by one"
alone would reject single-pixel pickups. Objects go away in object-sized chunks, on
the few actions that touch one.

Result on the games that were farming: cd82 240 → 4, tn36 → 10, lf52 365 → 56 (and
it now steers, because the wasted clicks were crowding out movement).

Two further leaks found and fixed in the same pass:

- **An unbounded reward.** The objective was `count(consumed) − count(built)`,
  and that subtraction has no floor: on a game where clicking paints a few more
  pixels, it falls a little *every click, for ever*. Assembly is now measured as
  *distance below its own record*, which has a floor at zero and cannot be farmed
  — painting more raises the record by the same amount. Consumption is scaled so
  one collectible outranks any amount of assembly; assembly is a tie-break, never
  a gradient.
- **Zero-action rounds.** `navigate` reported success for *standing still* on a
  target, having spent nothing, and the main loop went round again with the same
  target list — 190 empty rounds on one game, about half a level's budget spent
  deciding not to act. Standing on a target with nothing happening now means the
  target is not one: strike it and re-aim. Plus a guard that escalates after three
  consecutive rounds that spend ≤1 action.

## Phase 1 — Breadth: clear level 0 everywhere · 6–8 h · ⏳ in progress · ceiling 3.52

| task | status |
|---|---|
| The click branch judges clicks by the objective, not by pixel change | ✅ above |
| **Give the planner a destination on frame 0** (`markers.py`) | ◐ written, never run, not wired |
| Make ACT buttons plannable | ⬜ next |
| Fix rotating-sprite matching | ⬜ deprioritised — see below |
| Stop burning the whole action cap | ◐ partial (idle-round guard); no real convergence test yet |

**A destination on frame 0 (new, and now the top item).** Read the win condition
of all 25 games from their own source on 2026-08-23. 13 are readable predicates
and **10 of the 13 are one shape: every object of kind A must be co-located with
an object of kind B** (`ka59` twice over, `s5i5`, `lp85` twice, `tu93`, `wa30`,
`dc22`, `m0r0`, `ls20` sequenced). `cd82` and `re86` are the same idea at pixel
granularity. The rest eliminate a colour (`cn04`, `r11l`), constrain neighbours
(`ft09`), or set a flag elsewhere.

So the destination is **drawn on the board before the first action** — while
`Dream.target_colors` is `(collectible | prog.consumed) - retired`, both of which
are reactive and stay empty until the agent has accidentally succeeded twice.
That is the exact failure `progress.py` opens with: `dc22 move=100%(153)
collect=[] thought=0 levels=0` — a perfect forward model with nowhere to go, and
the reason breadth work keeps not converting into level-0 clears.

[markers.py](arc3x/markers.py) proposes target sets from one frame: group
components by `(colour, bbox h, bbox w, pixel count)`, keep groups with ≥2
members, drop background, floods (`MAX_SHARE`) and colours seen to move, rank by
member count and smallness. It is a **proposer, not an oracle** — safe because
`Dream.retire` already handles a wrong destination and because by Phase 3's rule
an action on a level never cleared costs zero score.

[why_markers.py](arc3x/why_markers.py) is the gate, and it needs no completed
level and none of `relive.py`: it reports how many actions the reactive path takes
to reach what the detector says at frame 0, and how often they agree. **Do not
wire `marker_colors` into `target_colors` until it has run.** `tu93` is the one row
graded against pixels traced from source (a 3×3 of colour 14); `cd82` and `re86`
must read `n/a region-match`.

**Still to write: the region-match variant** for `cd82` and `re86`, where the
objective is the mismatch between a static reference picture and a changing
workspace. Take the bbox of `Volatility.changes > 0` as the workspace, slide a
same-size window over never-changed pixels, pick the most picture-like match, and
let `objective()` be the count of differing cells. Two games, and it needs a
change history rather than one frame, so it is deliberately separate.

**ACT buttons (3 h).** `Mechanics.moves` filters for `delta != (0,0)`, which
silently discards every button that changes the board without moving the avatar — a
use, grab, drop, select or rotate. **Twelve of 25 dev games have one, and two games
have nothing else**: five working buttons and four working buttons respectively,
all invisible to the planner. They are now exposed as `Mechanics.acts` and this
item is **done**: `act_round` (agent.py:580) presses each act button at most once,
ordered by how often it has worked, and `push_frontier` (agent.py:718) tries them
against a blocking colour once shoving it has failed — the *walk there, then press
use* capability, keeping any press that moves the ratchet.

*Press-and-see for the two avatar-less games has been **retracted**.* Reading
`cd82.py` in full (782 lines) on 2026-08-23 showed it is a **paint-the-canvas**
game, not a press-and-see one: a 10×10 zero canvas at (27,34), a per-level 10×10
target at (3,3), win = equality with **both diagonals masked out**; palette
swatches set the colour, 8 basket slots ride a 3×3 ring, **ACTION5 stamps a whole
half-plane** and ACTION6 stamps a 3×4 edge strip. And `_get_valid_actions` is
**state-gated** — ACTION5 is not in the default list at all, so a button sweep
never sees the stamp. It also confirmed cd82's 100-action limit and the
`frame[63, x]` progress bar that defeats the 90%-frequency HUD mask. cd82 needs the
region-match detector above; a press-and-see sweep would have burned its budget.

**Rotating sprites (2 h) — deprioritised 2026-08-23.** `moved_objects` requires an
exact rigid shift, so a sprite that **turns to face** the way it is moving produces
*zero* votes on the axis it is not currently facing. Two dev games have four working
movement buttons and the model finds two. But `mind.py` already carries two
workarounds for this same gap — `_convention()` (line 362) and `_fill_deltas()`
(line 410) — and the button convention they encode was measured at **exactly 0.00**.
Controls were never the bottleneck; the destination was. Keep the fix queued for
model *fidelity*, not for score. `why_moves.py` remains un-run, so the wa30/sc25
attribution is still unconfirmed — though `tu93`'s source proves facing sprites
exist in the set (tag `0017unajnymcki` is a 3×3 avatar with a facing pixel, with
`set_rotation()` called on it).

**Convergence (2 h).** Every one of the 25 games currently spends its entire
3000-action budget. There is no test for *"this repertoire cannot clear this
level"*. **Revised 2026-08-23: the reason to want one is not what it looks like.**
By Phase 3's measured rule, actions spent on a level that is never cleared cost
**zero** score, so burning the cap is not itself the waste — the cost is only the
*later levels those actions could have bought*. So the goal is not "spend less", it
is "**stop spending on a hypothesis already known to be dead**". Two games now show
the shape of what is missing: one drains its objective all the way to **zero** and
then dies, and another converts one colour into another two pixels per click for
hundreds of actions. Both are making real, monotone progress on a quantity that is
not the win condition — so the test needed is "the objective bottomed out and
nothing happened", which should retire that objective and re-aim, rather than either
keeping on pushing it or falling silent.

## Phase 2 — Depth: levels 1 and 2 · 10–15 h · ⬜ · the only route past 3.52

This is where 10+ is won or lost, and the only phase whose hours are a guess.

- **Model carry-over is already the design.** On a level change the agent keeps
  the entire learned model and resets only the frame-to-frame counters. It has
  never been exercised, because almost nothing clears level 0 — which is what
  makes Phase 1 a prerequisite rather than a nice-to-have.
- **Hypothesis elimination.** Hold a *factorised* space of candidate game-models —
  maps × goals × obstacle-sets × collision rules × button-effects — as about 33
  independent facts rather than 9,600 enumerated products, and refute per billed
  action. The number of surviving candidates then scales with how ambiguous the
  game still is, which is the natural way to spend more thought on a harder game.
  **CPU candidates at ~1 ms, not LLM calls** — see the compute note below.
- **Why depth and not more breadth:** getting level 0 on the remaining games moves
  the mean by about +1.4 and then stops for ever. Getting level 1 on half the games
  moves it by about +3.3 and opens the road to 21.

## Phase 3 — Efficiency · 4–6 h · ⬜ · **not last, and not what it sounds like**

**Revised 2026-08-23 after reading the notebook's own logs.** This was scheduled
last on the reasoning that efficiency "only matters once levels actually complete".
The logs say levels *do* complete and the points are being thrown away anyway. Six
runs of the notebook on tn36, at the **same depth** (2 of 7 levels):

| run | total actions | score | cap at that depth | kept |
|---|---|---|---|---|
| exp_6 | — | 0.10 | 3.57 (1 level) | 2.8% |
| exp_e | **329** | 1.36 | 10.71 | 13% |
| exp_5 | — | 5.28 | 10.71 | 49% |
| exp_11 | **467** | **10.71** | 10.71 | **100%** |

**Identical depth, 7.9× apart in score — and the winner spent 42% *more* actions.**
So the lever is not actions-per-run. `exp_11` shipped a per-action event log; its 583
events split **31 / 69 / 483** over levels 0 / 1 / 2. It cleared 0 and 1 off just 100
actions (at ≈1.0–1.2× baseline — derivable from the cap: hitting it forces
`score_0 + 2·score_1 ≥ 300`, and the 115 ceiling then forces `score_1 ≥ 92.5`), then
spent the other **483 failing level 2, at a cost of exactly zero.**

> **An action spent on a level you go on to clear costs score quadratically. An
> action spent on a level you never clear costs nothing but budget.**

m0r0 is the same rule from the other side: 477 actions, level 0 *cleared*, score
**0.02 of an available 4.76** — 99.6% discarded precisely *because* it cleared the
level it dawdled on. Locally, one dev game clears level 0 in 47 actions against a
baseline of 22 (2.1×), throwing away 78%.

On a 7-level game clearing levels 0 and 1: **1.0× = 10.71, 1.1× = 8.85, 1.4× =
5.46, 2.8× = 1.37.** Two levels needs ~1.05× to reach 10; three levels reaches 10.8
even at 1.4×.

**So the ordering that follows is not "depth then efficiency" but a split by
level.** Burn level 0 — its weight is 1 of 21–55, at most 4.8% of the game, so a
four-hundred-action search there is nearly free. Then levels 1 and 2 must come in at
≤1.4× baseline, which means the **model** has to be doing the work by then; a search
still groping on level 1 has already lost the points. The ~25-action opening wiggle
is billed to level 0, where it is free, and must not repeat on later levels.

And the half that inverts the usual instinct: **never ration actions on a level you
are failing.** The only budget discipline that pays is on levels you are about to
clear. The only reason to abandon a level early is to buy a *later* clear with the
budget — never to protect the score of the level you are on, which is already zero.

**The first concrete change this implies, and it is queued rather than done.**
`agent.py`'s search throttle reads

```python
out = self.relive.run_level(RELIVE_ACTIONS if L == 0 else RELIVE_ACTIONS // 4)
```

which keys on the **level index**. The rule above says the deciding quantity is how
much that level has *already* cost, because adding `m` actions to a level `n` old
multiplies its score by `(n/(n+m))²` — baseline-free, and falling as `1/n³` at the
margin. Adding 500 actions to a level 50 old keeps 0.8% of its points; adding 500 to
one already 2000 old keeps 64%. So the current line throttles a hopeless level that
is 2000 actions deep exactly as hard as a fresh one, which buys nothing at all,
since an uncleared level scores 0 either way.

The replacement shape is `m = clamp(ALPHA/(L+1) × spent_on_level, floor/(L+1),
RELIVE_ACTIONS)` — holding the *fraction* of level score surrendered roughly
constant instead of unknown, while still protecting later levels more because their
absolute points are worth more. **Deliberately not implemented yet:** `ALPHA` and
the floor are guesses, `relive.py` has not executed once, and stacking a tuned
budget policy onto an unmeasured search is exactly how the previous eleven grafts
earned their reputation. It is the first A/B after the suite runs.

That same rule sets the success test for the search component itself: **not** "does
it clear level 0" but *"does a level 0 cleared by search leave a model that clears
level 1 fast"* — which is what `arc3x/smoke_relive.py` prints as its `x base`
column, per cleared level and never averaged.

**One more thing that table settles: never judge a change on the 4-game
benchmark.** One game produced 0.00, 0.00, 0.10, 1.36, 5.28 and 10.71 across runs —
and `exp_e` (1.36) and `exp_11` (10.71) ran on **identical repo commits** with
**byte-identical env config** (temperature 0.6, no seed). Nothing captured differs.
The noise floor there exceeds any effect worth shipping, so the "0 for 11 grafts"
record is an underpowered experiment, not eleven refuted ideas. Use
`suite.py --split both`, or the leaderboard. **And add the graft-flags dict to the
artifact bundle** — the project's most expensive measurement is unattributable for
want of one `json.dump`.

## Phase 4 — Graft and submit · 3 h + ~9 h Kaggle per attempt · ⬜

**Only if the local number beats 2.14.** Lift the objective, the ACT-button
handling and the imagination search into the notebook's solver cell and let the LLM
name the genre. Each submission is ~9 h of wall-clock over 110 games with one
graded play each, so budget two or three attempts, not ten.

---

## 3. Verification, and the gates

Run the full suite after every phase, **always reporting coverage out of 25
alongside the mean** — a mean of 0.142 over 25 games is 23 zeros and one game
working, and reporting the mean alone made that look like a small positive result
for weeks. Success is level-clears rising, not the mean rising.

Each gate is falsifiable and was written before the work:

| gate | test |
|---|---|
| Phase 1 done | level 0 cleared on ≥20/25 games, mean ≥ 2.5 |
| Phase 2 done | level 1 cleared on ≥12/25 games, mean ≥ 6 |
| Phase 3 done | mean ≥ 10 locally, at ≤1.4× baseline on cleared levels |
| Phase 4 | Kaggle > 2.14, else revert and keep the 2.14 submission |

**The overfitting control.** The evaluation set is 110 private games. The danger is
not that a rule fails on the dev set — it is that a rule *succeeds* here for a
reason that does not exist there. So the 25 games are split every-third
alphabetically into 17 to iterate on and **8 that are never tuned against**, fixed
and never re-drawn (a holdout re-rolled after a disappointing result is a lottery,
not a control). `--split both` prints the ratio; a mechanism that helps the tune
half and not the hold half has been fitted to games it was allowed to see.

The clock-bar fix above is the first real test of that discipline and it passed for
the right reason: the bug was *found* on a holdout game and the fix is a general
statistical test on a colour's behaviour, with no game named anywhere in it.

## 4. Two things established, both negative, both load-bearing

**Prediction accuracy is not the bottleneck.** Two games whose forward model
predicts the next frame at 100% complete zero levels. A third predicts at 80%, is
the only one with a known goal colour, and is the only one that plans and completes
a level. Accuracy without an objective buys nothing — which is why the objective,
not the model, is where the work goes.

**Compute is not the bottleneck.** The search engine contains no neural network at
all — it is numpy on CPU. And an LLM cannot afford to *play*: at ~17.6 s/action
over ~300 actions × 110 games it needs 161 hours against a ~9 hour limit. One
earlier experiment scored 2.68 locally and **0.60** on Kaggle purely because vLLM
prefill timed out on a shared GPU. This is the reason hypothesis elimination in
Phase 2 is specified as CPU candidates rather than model calls.

## 5. The honest risk

Phases 0, 1, 3 and 4 will land in roughly the hours given. **Phase 2 is research.**
If depth does not come, the score sits near 3.5 and no amount of further tuning
moves it — that is a property of the scoring formula, not of the effort. The
mitigation is task 0c: find out which regime the existing 2.14 is in *before*
spending 15 hours on the wrong half of the problem.
