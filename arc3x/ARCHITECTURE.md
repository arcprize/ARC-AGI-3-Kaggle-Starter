# ARC-AGI-3 solver — architecture and method

A working note for another researcher. Written 2026-08-23, revised the same day at
commit `1944e2a`. Current numbers are stated as measured, including the bad ones,
and there are more bad ones than good.

---

## 1. The problem, precisely

ARC-AGI-3 is a set of interactive grid games. The agent is shown a **64×64 array
of integers 0–15** and may press up to six buttons plus a click with an (x, y)
coordinate. There is:

- no reward signal,
- no instruction, no goal statement, no genre label,
- no reset-and-retry-cheaply: **one graded play per game**,
- and no access to the game's source at evaluation time.

The only feedback is a `levels_completed` counter and a state flag
(`NOT_FINISHED` / `GAME_OVER` / `WIN`). `level_reset` costs one action and clears
`GAME_OVER`, so death is cheap.

**Evaluation is on 110 private games nobody has seen** (55 public leaderboard, 55
private). Twenty-five games ship with the engine as a development set. This
asymmetry is the single most important constraint on the design, and it is sharper
than "don't overfit" — the dev games ship as *source*, and the scored games ship as
nothing at all:

| | 25 dev games | 110 scored games |
|---|---|---|
| engine source | yes | **no** |
| `copy.deepcopy(game)` — free snapshot and rewind | free | impossible |
| `game._get_valid_actions()` — the legal *click coordinates* | readable | not exposed |
| `baseline_actions` — the human reference score | readable | stripped |
| plays per game | unlimited | **one** |

So the 25 are a laboratory, not a sample to fit. Any mechanism that works because
of a property of these 25 is worthless, and — the trap that actually caught us
repeatedly — any *measurement* that used one of the three privileges above is
inflated by an unknown factor. The withholding is enforced in code (§5), not by
discipline.

## 2. What the scoring formula demands

This deserves its own section because it inverts the obvious priorities. From the
competition scorer (replicated verbatim in `arc3x/graded.py`):

```
level i score  = min(115, (baseline_actions_i / actions_spent_i)² × 100)   if cleared
               = 0                                                         if not
game score     = min( Σ(score_i × i) / Σ i ,  (Σ cleared i / Σ all i) × 100 )
leaderboard    = mean over games
```

Level *i* carries weight *i*, and **every action spent while stuck on a level is
billed to that level and to no other**.

Two consequences:

**(a) At baseline speed, score equals the completion cap.** If each cleared level
scores 100, the weighted mean equals `(Σ cleared weights / Σ all weights) × 100`.
Evaluated over the actual level counts of the 25 dev games (9 games have 6
levels, 5 have 7, 6 have 8, 4 have 9, 1 has 10):

| levels cleared per game | mean score at baseline speed |
|-------------------------|------------------------------|
| level 0 only            | **3.52** — a hard ceiling    |
| levels 0 + 1            | **10.57**                    |
| levels 0 + 1 + 2        | **21.14**                    |

So a solver that reliably clears the first level of every game and never the
second cannot exceed 3.52, no matter how efficient it becomes. **Depth is the
only axis that matters.** This is not obvious from the leaderboard and it took us
a while to work out.

**(b) Efficiency is a multiplier on depth — but it is per level, not per run.**
Level score is `(baseline/actions)² × 100`, so depth reaches the completion cap and
speed decides what fraction of the cap survives. We had this as "optimise last"
until the LLM notebook's own logs were read properly. Six runs of it on the same
game, at the same depth:

| run | tn36 levels | total actions | score | cap at that depth | kept |
|---|---|---|---|---|---|
| exp_4 | 0/7 | — | 0.00 | — | — |
| exp_f | 0/7 | — | 0.00 | — | — |
| exp_6 | 1/7 | — | 0.10 | 3.57 | 2.8% |
| exp_e | 2/7 | **329** | 1.36 | 10.71 | 13% |
| exp_5 | 2/7 | — | 5.28 | 10.71 | 49% |
| exp_11 | 2/7 | **467** | **10.71** | 10.71 | **100%** |

**Identical depth, 7.9× apart in score — and the winner spent 42% *more* actions.**
That kills the obvious reading. An earlier draft of this section carried an
"implied speed" column back-derived from the score; the measured action counts run
the other way, so the column was deleted. The mechanism is the level split, and
`exp_11` shipped the per-action event log that shows it. Its 583 events divide
across levels as:

| level | actions | cleared? | cost |
|---|---|---|---|
| 0 | **31** | yes | scored ≥70 of 115 → baseline ≥ ~26, so ≈1.0–1.2× |
| 1 | **69** | yes | scored ≥92.5 → baseline ≥ ~66, so ≈1.0× |
| 2 | **483** | **no** | **nothing** |

(The two bounds are derived, not read: hitting the cap exactly forces
`score_0 + 2·score_1 ≥ 300`, and the 115 ceiling then forces `score_1 ≥ 92.5` and
`score_0 ≥ 70`.)

So **83% of the run was spent stuck on a level it never cleared, and that cost
exactly zero.** Uncleared levels contribute 0 to the numerator of the weighted mean
whatever you spend on them; they only ever cost you *budget*. Meanwhile `exp_e`
spent fewer actions overall and scored 7.9× less, because it dithered through the
two levels it did clear.

The rule, stated once, because everything downstream depends on it:

> **An action spent on a level you go on to clear costs score quadratically. An
> action spent on a level you never clear costs nothing but budget.**

m0r0 is the same rule from the other side: 477 actions, level 0 *cleared*, score
**0.02 of an available 4.76** — 99.6% thrown away, precisely because it did clear
the level it dawdled on. exp_11 dawdled just as long and lost nothing, because it
dawdled on a level it failed.

The arithmetic on a 7-level game clearing levels 0 and 1: **1.0× baseline = 10.71,
1.1× = 8.85, 1.4× = 5.46, 2.8× = 1.37.** So two levels needs ~1.05× to reach 10,
while three levels reaches 10.8 even at 1.4×. **Aiming one level deeper is much
the easier route to the same number.**

**(b′) That table is also a warning about measurement, and a sharper one than it
looks.** One game produced 0.00, 0.00, 0.10, 1.36, 5.28 and 10.71 across runs — and
the two extremes we can inspect, `exp_e` (1.36) and `exp_11` (10.71), ran on
**identical repo commits** (`ARC3-Inference aa69123`, `taaf fe9f7c4`, `re-arc-3`
pinned) with **byte-identical `taaf_setup_env.json`**, including
`temperature 0.6 / top_p 0.95 / top_k 20` and no seed. Nothing in either run's
captured configuration differs.

That does not prove the 7.9× is sampling noise — the runs are 17 days apart and the
graft flags live in a notebook cell that is not captured in either artifact set, so
the cause is genuinely unidentified. Worse than unidentified: the flags dict is

```python
{"efficiency": True, "retry_guard": True, "shortcircuit": True, "context_window": 57344}
```

and `context_window` is applied **in-process**, so `taaf_setup_env.json` reports the
env var `32768` for every run regardless of what the graft actually used. The single
knob most suspected of causing wasted actions — the notebook's own architecture notes
call the context window "the recurring root cause of wasted actions, not reasoning
quality" — is precisely the one the artifact bundle is blind to. So the identical
config files establish *less* than they appear to.

Practical consequences:

1. The noise floor on a 4-game benchmark exceeds any effect worth shipping, so
   "eleven grafts changed nothing" was an underpowered experiment, not eleven
   refuted ideas. Judge changes on the 25-game split or on the leaderboard, never
   on four games.
2. **Capture the graft flags in the artifact bundle.** Done: cell 6 now defines
   `GRAFT_FLAGS` once and writes `graft_flags.json` beside the other artifacts. The
   most expensive measurement in the project was unattributable for want of one
   `json.dump` of a dict already in memory.
3. Variance reduction and efficiency are the *same lever* here. An agent that
   reliably played like its own best draw would score several times its mean
   without any new capability.

**(c) Level 0 is nearly free to explore.** On a 9-level game the weights sum to
45, so level 0 caps at 2.2% of the achievable score. Hundreds of actions can be
spent there learning the game for almost nothing — but the human baselines for
level 0 are only 22–59 actions, so "free" means free in *score*, not in
information.

**(d) Putting (b) and (c) together gives the strategy, and it is not the obvious
one.** *Burn level 0 for information; win on levels 1 and 2 with the model.* Level
0's own score is at most 4.8% of a game, so a four-hundred-action search there is
cheap. Levels 1 and 2 are worth 2–3× more each and have to be cleared at ≤1.4×
baseline, which means the **model** must be doing the work by then — a search that
is still groping on level 1 has already lost the points. That is exactly what the
model carry-over in `on_new_level` (§3.5) is for, and it has never been exercised,
because almost nothing clears level 0 yet.

The measured level split adds a second half to this that is worth stating on its
own, because it inverts the usual instinct about giving up:

> **Never ration actions on a level you are failing.** exp_11 spent 483 of 583
> actions on level 2, failed it, and still scored the full cap. The only budget
> discipline that pays is on levels you are about to *clear* — and the only reason
> to abandon a level early is to buy a *later* clear with the budget, never to
> protect the score of the level you are on.

So the question to ask of any search component is not "does it clear level 0" but
**"does a level 0 cleared by search leave a model that clears level 1 fast"**. That
is what `arc3x/smoke_relive.py` measures.

## 3. Design thesis

The system is built around one claim: **a person who has never seen the game
builds a small causal model of it within about twenty presses, and thereafter
plans inside that model rather than against the game.** So the solver should
learn a model, form a goal without being told one, and search in imagination
because imagination is free while actions are billed quadratically.

Five components, each with a single job.

### 3.1 `percept.py` — vision, pure numpy

Connected-component labelling by explicit-stack flood fill (no `scipy` in the
Kaggle image), rigid-shift matching between consecutive frames, and a
**volatility mask**: pixels that change on nearly every action regardless of what
was pressed. That mask is the HUD — score digits, timers, level indicators.

Isolating the HUD turned out to be load-bearing. An earlier version treated
frames as states for a Go-Explore-style archive; a one-pixel clock in the corner
made every frame unique, so the archive never merged two visits to the same
place and the search silently degenerated. Nothing crashed; the score was just 0.

### 3.2 `mind.py` — `Mechanics`, the learned model

Answers five questions from observation alone:

| question | mechanism |
|---|---|
| which colour am I? | see below |
| which button moves me where? | `votes[(action, colour, delta)]`, consensus |
| what blocks me? | a refused move blames the cells its footprint would have entered |
| what kills me? | on death, blame only *novel* ground |
| what looks like a goal? | on `levels_completed++`, credit what vanished / what we stepped on |

**Avatar identification by reversibility.** Many things move; only one is under
control. Candidates are scored on whether their observed displacement set contains
both `(dy,dx)` and `(-dy,-dx)` under different buttons. A real control scheme
contains opposite pairs; a coincidental shape-match across frames almost never
does. Secondary criteria: axis-alignment, consistent step magnitudes, number of
distinct buttons, total vote weight. This is cheap, needs no labels, and has been
the most reliable single prior in the system.

Whatever moves *identically* to the avatar under shared buttons is folded into a
`body` set — multi-colour sprites are one object. Getting this wrong makes every
collision test wrong, because the footprint used to sample "what am I standing on"
is read from the body mask.

Two known defects, both measured:

- `moved_objects` requires an exact rigid shift, so a sprite that **rotates as it
  turns** produces zero votes on the axis it is not facing. Two dev games have
  four working movement buttons and the model finds two.
- Buttons that change the board *without* moving the avatar were discarded as
  no-ops. Twelve of 25 games have such a button (a use / grab / select / rotate);
  two games have *nothing else* — five and four working buttons respectively, all
  invisible to the planner. Now exposed as `Mechanics.acts` and used in two
  places: `act_round` ([agent.py:580](agent.py#L580)) presses each one at most
  once, ordered by how often it has been seen to do anything, and
  `push_frontier` ([agent.py:718](agent.py#L718)) tries them against a blocking
  colour after shoving it fails — *walk there, then press use*.

### 3.3 `progress.py` — `Progress`, an objective with no labels

The hardest part. `goal_colors` above only learns from a completed level, so it
needs a win to learn what a win looks like — useless on the first level of the
first game, which is the only situation that exists.

The rule, which we call the **ratchet**:

> Count every colour outside the HUD mask, every frame. A colour that **goes down
> and stays down** is being consumed — that is the objective. A colour that only
> goes up is being built.

Formally, a colour qualifies as consumed if it fell at least twice and fell at
least three times more often than it rose. The asymmetry is the whole content: a
count that oscillates is animation; a count that ratchets is progress.

One rule covers collecting pickups, clearing markers, painting a region, and
Sokoban — a crate parked on a target *occludes* the target pixel, so the target
colour ratchets down — with no genre-specific code.

Two guards, both added after the rule failed:

- **Flood rejection.** Any colour that ever covered more than 25% of the playing
  field is excluded. One game offered a board-sized background colour with a
  clean downward ratchet; chasing it replaces "no destination" with "an
  unreachable one".
- **Frozen mask.** The HUD mask is computed once and never revised. Recomputing
  it per frame looks more responsive and wipes the ledger, because the mask
  wiggles as the observation count grows. One game finished a 240-step run with a
  perfectly empty ratchet ledger despite having planned 41 times.

And a third, added after the rule failed a third time and worse:

- **Clock rejection.** A colour that moves on ≥55% of actions *and* moves by
  exactly one pixel ≥75% of the time is a counter — a step budget, a health bar, a
  fuel gauge — not a set of objects. Both conditions are needed: "moves nearly
  every frame" alone would reject a collectible in a game where every move picks
  something up, and "moves by one" alone would reject single-pixel pickups.

  This is the worst failure the ratchet rule admits, because "make this number go
  down" is satisfied by doing *anything*. The frozen-mask guard was supposed to
  cover it and cannot: volatility asks "does this pixel change often", and a bar
  that loses one pixel per action changes each individual pixel on about 1/154 of
  frames. One game's mask came out **completely empty**, its 154-pixel bar counted
  as playfield, and the agent reported progress on 240 consecutive random clicks.
  Five of the eight holdout games were doing nothing else — about 12,000 actions.

The pattern across all three is worth stating on its own, because it is the
generalisable lesson: **the ratchet rule is strong enough to find a monotone
quantity in almost any game, and not every monotone quantity is the point of the
game.** Floods, clocks and paint are all monotone. Any unsupervised objective needs
an explicit, tested list of what satisfies it without solving anything.

### 3.4 `dream.py` — `Dream`, the imagination

A forward model plus search. `predict(frame, action)` produces the imagined next
frame by translating the body mask; `observe()` grades its own prediction against
what actually happened before the model is updated, so accuracy is measured on
the prediction the agent actually acted on.

Search happens in two spaces, cheap first:

1. **Position space** — breadth-first over the learned deltas, ~8000 nodes,
   blocked by the learned obstacle set. Deep and fast, but only answers "can I
   get to that cell".
2. **Frame space** — actually simulates the next frame for each button and
   searches until `objective()` strictly drops. Answers "does this *do* anything",
   but ~600 nodes over four buttons is only about five moves of lookahead.

The abstention discipline matters: `observe()` can return `abstain` when the model
has no opinion, and abstentions are counted separately from errors. Without that
split, a game where most moves are blocked scores a free 100% by predicting
"nothing happened".

**The gap this leaves, and it is the largest one in the system.**
`Dream.target_colors` ([dream.py:512](arc3x/dream.py#L512)) is
`(collectible | prog.consumed) - retired`, and **both sources are reactive**:
`collectible` needs a colour to have already vanished under the avatar's feet,
and `prog.consumed` needs a count already ratcheted down twice
(`progress.MIN_CLICKS = 2`). So the destination set is empty until the agent has
*accidentally succeeded twice*. That is precisely the failure table
[progress.py](arc3x/progress.py) opens with — `dc22 move=100%(153) collect=[]
thought=0 levels=0`, and the same for `ka59` and `m0r0`. A perfect forward model
with nowhere to go.

### 3.4b `markers.py` — where the goal is, read off the first frame

Because all 25 dev games ship as source, their win conditions can simply be read
rather than guessed. Every guard on a `self.next_level()` was read on
2026-08-23; 13 are readable predicates and **10 of those 13 are the same shape**:

> every object of kind A must be co-located with an object of kind B.

`ka59` (two such pairs at once), `s5i5`, `lp85` (two), `tu93`, `wa30`, `dc22`,
`m0r0`, `ls20` (same, sequenced). `cd82` and `re86` are that same idea at pixel
granularity — a canvas must equal a reference picture. The remainder are
eliminate-a-colour (`cn04`, `r11l`), a local constraint (`ft09`), or a flag set
elsewhere.

**The destination is therefore drawn on the board, and it was drawn before the
first action.** A target marker in such a game is *repeated* (a game draws its N
targets identically), *static* (not a colour seen to move), and *small* (it marks
a place, so the `MAX_SHARE` judgement `progress.py` already makes applies). Group
connected components by `(colour, bbox h, bbox w, pixel count)`, keep groups with
≥2 members, drop the background, drop floods, drop movers, rank by member count
and smallness. None of those tests names a game, a genre or a colour.

Two properties make a *speculative* source safe here, and both come from
measurement rather than hope: `Dream.retire` already exists because "a wrong
destination costs more than no destination" was learned the expensive way; and by
§2(b) an action spent on a level never cleared costs no score at all, so being
wrong on level 0 is nearly free.

**Status: written, never executed, and not wired into `target_colors`.** The
measurement that decides it is [why_markers.py](arc3x/why_markers.py), which
reports how many actions the reactive path needs to reach the same answer the
detector gives at frame 0 — and the row that can genuinely fail is `tu93`, whose
exit sprite was traced to pixels (a 3×3 of colour 14, `tu93.py` 396-445). `cd82`
and `re86` are expected to fail honestly: the region-match variant is specified
in [PLAN.md](arc3x/PLAN.md) and deliberately not implemented.

### 3.5 `agent.py` — the policy

```
WIGGLE     press every button ~4×; ~25 actions, billed to level 0 where it is nearly free
then loop:
  IMAGINE      plan toward a falling ratchet colour; spend ONE action; re-plan
  GOAL         walk to any colour that has previously ended a level
  COVER        stand on every reachable square (frontier-directed, O(n) not O(n²))
  FRONTIER     walk into each *kind* of thing that has refused us, once
  CLICK        rank blob centres by rarity, click until something changes
  FLAIL        random over non-dead actions; alternate with level_reset
on level up:  keep the entire model; reset only the frame-to-frame counters
```

Two deliberate choices worth calling out:

**Re-plan after every single action, never execute a route.** The model starts out
believing nothing blocks it, so a route planned once walks into the first wall and
wastes the remainder. Planning is free; one billed action is the cheapest possible
test of a hypothesis. Each refusal teaches an obstacle, so the route repairs
itself as it is walked, and a level gets mapped in O(path) billed actions rather
than O(area).

**`cover` is deliberately dumber than the targeting heuristic.** Most levels
complete when the avatar touches the right thing, and enumerating everything
walkable is the only method that does not require guessing which thing that is.

### 3.6 `relive.py` — search, when the model has nothing to say

The stall report (§5) says the failure in one line. On every game that scores
zero, `imagine:noplan` fires once per round, in lockstep with `steer`, and
`cover:covered` fires the same number of times:

```
ka59  0/7 levels   steerx1673, imagine:noplanx1673, cover:coveredx1673
cn04  0/6 levels   steerx2648, imagine:noplanx2648, cover:coveredx2647
su15  0/9 levels   nosteerx2864, noobjectivex2842
```

In English: *the whole reachable board has been walked and there is nothing to aim
at.* `Progress` never ratchets during aimless wandering, so `objective()` returns
`None`, so the imagination has no gradient, so the round falls through to a random
walk for the rest of the budget.

Meanwhile free search in the local twin clears level 0 on **17 of 24** games in
**4 to 35 actions**. So level 0 is not deep. The agent is not failing for lack of
a reachable answer; it is failing for lack of a method that can **return to a
promising state**. A random walk cannot: if the interesting thing is twenty steps
down a corridor, drifting away from it is the overwhelmingly likely next event and
there is no way back.

**The rewind is real, and it is not `deepcopy`.** Two facts about the API combine:

- RESET restores the *current level* from its pristine clone and costs **exactly
  one action** (`handle_reset` routes to `level_reset` once play has begun).
- 23 of 25 games use no randomness at all, and no game constructor takes a seed,
  so the same action prefix reproduces the same frames.

Therefore any state reachable by a plan `p` can be re-reached for `1 + len(p)`
billed actions. That is Go-Explore's "return to cell" step, priced. It is also
what a person does: die, start the level again, walk straight back to the bit you
had not tried yet.

Three things change because the rewind is billed rather than free.

**Selection becomes promise *per action*.**

```
score(node) = promise(node) / (1 + restart_cost(node))
restart_cost = 0  for the node we are already standing on
             = 1 + depth  for any other
```

That zero term is the whole character of the search: it expands where it is until
that place is exhausted or unproductive, and only then pays to jump. What falls
out is depth-first with cost-aware backtracking, which is the right shape when a
rewind costs money. Twin Go-Explore teleports for free and therefore ignores
depth entirely — a difference that is invisible until you price it.

**The cell key has to be learned from billed frames.** The agent already keeps a
novelty set keyed on `fingerprint(frame, Volatility.live_mask)` and it does
nothing, because `hud_mask` is a *frequency* test at 90%: a pixel must change on
almost every action to be discarded. tn36's row-1 bar drains six pixels per
action, but *which* six moves along the bar, so each individual pixel changes on
about 1/49 of frames and survives. cd82's 154-pixel bar is 1/154. Both bars enter
the key, and with a counter in the key every state is novel for ever — measured:
60 distinct keys in 60 steps.

`cell.py` has the rule that fixes it — a pixel that moves **monotonically with the
action count** is a clock, because state revisits values and a counter never does
— but it learns that from deepcopied probe walks, which do not exist behind the
gateway. `Clockless` learns the identical rule from the frames of ordinary billed
play, at zero extra cost, by treating **the stretch between two RESETs as one
walk**. The cut at each reset is load-bearing: a RESET restores the counter too,
so a clock read across a restoration is *not* monotone and would be filed as
state. Held as six 64×64 accumulators, so memory is flat in run length.

The mask is frozen once and never revised, and the *first* freeze clears the
archive. Both are scars. `progress.py` records what a moving mask does — "ka59
ended a 240-step run with a perfectly empty ledger that way" — and the first
smoke test here showed the other half: a key computed under a provisional mask is
a *different key*, so nodes archived before the freeze can never be matched, and
**481 of 500 actions went on RESETs to cells that no longer existed.**

**The key coarsens itself, with no constant to tune.** Even clock-free, the key
came out nearly bijective:

```
ka59  informative=514 px    128 cells in ~430 actions
bp35  informative=1542 px   227 cells
cn04  informative=1890 px   295 cells
```

About one new cell per action means novelty is not a signal and the search
degenerates into a paid random walk. The fix measures the problem instead of
guessing at the answer: the rate at which cells are **created per action spent**
is exactly "does anything ever collapse", and it needs no per-game knowledge. When
that rate stays above 0.7 over 96 actions, the pool size doubles — each 2×2 block
of *informative* pixels is max-pooled into one — which removes sub-tile rendering
detail (a sprite's facing, an animation phase) before it removes anything
positional. Blanking non-informative pixels to −1 *before* pooling matters: pool
the raw frame and a clock inside a block dominates the max, quietly reintroducing
what the mask was built to remove.

The archive is **re-keyed rather than cleared** on coarsening, keeping the
shortest plan per merged cell and summing the evidence counters. Clearing is
simpler and much worse: the plans are the only thing that makes the rewind
affordable, and a coarser key does not change what an action does, so they are
still valid.

**Routes are recorded by `Agent.act`, not by the search.** `act` is the only place
that sees every billed action, so it appends each one to the current plan. Two
consequences: a plan is always a true recording of the prefix since the last
restoration, and the walking, imagining and clicking branches archive returnable
cells *for free*. The earlier version, where only the search maintained the plan,
had a subtle failure — a search starting after a round of ordinary play recorded
the current frame under whatever stale plan was left over, and every attempt to
return to it billed a RESET and arrived somewhere else.

**This is the fallback, not the plan.** `Dream` + `Mechanics` is strictly better
whenever it has an objective, because it walks to a place it can *prove* is better
and does it in the fewest actions — and the score is quadratic in actions. Search
is what the agent does when the model has nothing to say, which today is most of
the time on most games. Every action still flows through `Agent.act`, so
`Mechanics`, `Dream` and `Progress` keep learning from it: a level cleared by blind
search is a level whose model is then carried forward by `on_new_level`.

**Status: written, reviewed, not yet measured.** The self-calibrating key and the
`act`-side route recording have never executed. Nothing is claimed for them.

## 4. One game, start to finish

What actually happens, in order, from the first frame to the last action. Line
references are to the files above.

**Action 1 — `reset()`.** The only thing known is a 64×64 array of integers and a
list of button ids, typically `[1,2,3,4,5,6]`. Not which of them do anything, not
which colour is us, not whether there *is* an us. The frame is fingerprinted into
a novelty set and nothing else is assumed.

**Actions 2–25 — `wiggle()`: press everything, about four times each.** The
cheapest information in the game, and it is billed to level 0 where the scoring
formula makes it nearly free (on a 9-level game level 0 is 2.2% of the achievable
score). Every press feeds three consumers at once:

- `Volatility` accumulates a per-pixel change frequency. Pixels that change on
  more than half of all actions become the **HUD mask** — score digits, timers,
  level indicators — and are excluded from everything downstream.
- `Mechanics.observe` diffs the before/after frames, finds objects that moved as a
  rigid unit, and files a vote `(button, colour, (dy,dx))`. A press that changed
  nothing is *also* recorded, as a `noop` and as evidence about walls.
- `Progress.add` folds the frame into per-colour pixel counts.

The loop exits early once there is an avatar and at least two movement buttons,
because more wiggling would be paying for information already held.

**`settle()` — decide who we are.** Candidate colours are scored on
**reversibility**: does the observed displacement set contain both `(dy,dx)` and
`(−dy,−dx)` under different buttons? A real control scheme does; a coincidental
shape match across two frames almost never does. Tie-breaks are axis-alignment,
consistent step magnitude, how many distinct buttons move it, and total vote
weight. Anything that moves *identically* to the winner under shared buttons is
folded into a `body` set, so a multi-colour sprite is one object — getting this
wrong makes every collision test wrong, because "what am I standing on" is sampled
through the body mask.

At this point the agent has either a steerable avatar with a delta per button, or
it does not — and that single fact routes the rest of the play.

**Then a loop, until the budget runs out or the game is won.** Each round records
`steer`/`nosteer` and whether there is an objective at all, because *"did this
agent ever have a destination on this level"* is the first question to ask of a
zero and the score cannot answer it.

**Round step 1 — `imagine()`.** Ask the forward model for the shortest imagined
sequence that makes `objective()` strictly fall. `objective` is the ratchet: how
many pixels of the colours that are being *consumed* remain. Search is
breadth-first over predicted frames, and imagination costs nothing while actions
are billed quadratically — so this outranks every heuristic below whenever it has
an opinion.

Then the crucial discipline: **execute exactly one action and re-plan.** Never run
a route. The model starts out believing nothing blocks it, so a route planned once
walks into the first wall and wastes its whole tail. One billed action is the
cheapest possible test of a hypothesis, each refusal teaches an obstacle, and the
route repairs itself as it is walked. A level gets mapped in O(path) actions
instead of O(area).

**Round step 2 — `goal`.** If some colour has previously coincided with a level
completion, walk straight to it. This is the transfer that repays the level-0
exploration: levels 1..n are worth 2..n times as much and cost a fraction, because
the model arrives already built.

**Round step 3 — `cover()`.** Stand on every square the model believes is
reachable, frontier-directed. Deliberately dumber than the targeting heuristic:
most levels complete when the avatar touches the right thing, and enumerating
everything walkable is the only method that does not require guessing which thing
that is. Directed frontier walking makes it affordable — O(n) actions for n
squares, where a random walk is O(n²).

**Round step 4 — `push_frontier()`.** Walk into each *kind* of thing that has
refused us, once. A blocked colour might be a wall, a door, a hazard or the goal,
and from outside those look identical. One attempt per colour, not per tile.

**Round step 5 — `click_round()`.** Rank blob centres (and the corners of large
blobs) by: has a click here already done nothing, has a click here already helped,
is this colour one the ratchet says is being consumed, then rarity, then size.
Click down the list. A click that beats the best objective ever reached on this
level is `progress` and earns another round; a click that merely moves pixels is
`change`, which is information only while there is still no objective and is mere
animation once there is. **Those two must not share a name** — conflating them is
what produced 13,000 wasted actions across the holdout.

**Round step 6 — the round failed.** Re-run `settle` with weaker evidence, forget
where we have been, and either flail randomly over non-dead actions or
`level_reset` (one action) for a pristine board. A round that ran the entire
repertoire and spent ≤1 action is not caution, it is a stall; three in a row
escalates straight to flailing.

**On a level change — keep everything.** The whole learned model survives: avatar,
deltas, obstacles, hazards, goal colours, ratchet evidence. Only the frame-to-frame
counters and the coordinate-keyed memories are cleared, because coordinates do not
survive a new board and comparing a fresh level's colour counts against the
previous one's would read the restoration as a giant increase and reject every real
collectible. **This carry-over is the entire reason the design should scale with
depth** — and it has barely been exercised, because almost nothing clears level 0
yet.

## 5. Method — how claims get established

This is the part we would most want critiqued.

**An honest offline replica.** `arc3x/graded.py` reproduces the competition
scorer verbatim and deliberately withholds three powers the local engine offers
but the graded gateway does not:

1. `copy.deepcopy(game)` — free snapshots, free rewind. Behind the gateway there
   is no game object, only HTTP.
2. `game._get_valid_actions()` — its own docstring says it is never exposed to
   users. It returns the *state-dependent* legal set including, for the click
   action, the concrete legal coordinates. An agent reading it is being told
   which of 4,096 clicks are live.
3. `baseline_actions` — present locally, stripped from the API. Used to *score*
   the run, never shown to the agent.

Every earlier local measurement in this project was inflated by one of these.

**Coverage travels with the mean, always.** A mean of 0.142 over 25 games is 23
zeros and one game working. Reporting the mean alone made that look like a small
positive result for weeks.

**A "why did it stop" channel.** `agent_fn` returns nothing, so the score was the
only output, and a score cannot distinguish *"never had a destination"* from
*"had one and could not reach it"* — which need opposite fixes. Every strategy
now records its exit reason against the level it started on.

**A fixed train/holdout split.** 17 games to iterate on, 8 held clean, chosen
every-third alphabetically and **never re-drawn** — a holdout re-rolled after a
disappointing result is a lottery, not a control. `--split both` prints the ratio.
A mechanism that helps tune and not hold has been fitted to games it was allowed
to see, and the 110 private games will behave like hold.

**Wall-clock cutoffs are treated as invalidating, not degrading.** Kaggle bills
actions, not seconds. A run cut short by a deadline is a machine-dependent
experiment and is not comparable to another run; the summary says so loudly
rather than averaging it in. This caught a real error: naive parallelisation had
each worker size a BLAS thread pool to all 12 cores, and one game managed 32
actions where serial did 401.

**Negative results are kept.** Two examples:
- We hypothesised that the click games are reference-panel + canvas, and tested
  half-board agreement. 11/11 click games landed in the predicted band — which
  looked like a clean win until the control gave **25/25**, including three
  avatar games at >90%. The boards are mostly empty and empty agrees with empty;
  the test measured emptiness. Discarded.
- Graft-style tuning of the LLM notebook is **0 for 11** experiments. The only
  change that ever moved the score was swapping the model.

## 6. Current results, as measured

| system | measured on | mean | level-clears |
|---|---|---|---|
| LLM notebook (the submitted one) | **110 unseen games**, Kaggle | **2.14** | not instrumented |
| `arc3x` standalone | 25 dev games | **0.142** | 2 across 25 |
| `arc3x`, tune split | 17 games it was built on | **0.177** | 2 of 17 |
| `arc3x`, holdout split | 8 games never tuned on | **0.005** | 3 of 8 reach level 1, none finish |
| Go-Explore in the local twin | 25 dev games | **8.592** | 32 levels, 18/25 games |

Four things to read out of that table, because three of them are traps.

**2.14 is the only number that measures the actual task.** Everything else is a
lab instrument. `arc3x` is 15× *worse* than the notebook and does not ship as the
solver; it is a measurement rig and a source of components.

**The tune:holdout ratio is 0.03.** For every unit of score built on games whose
source we can read, about 3% survives to games we cannot. That single number is
the honest answer to "how much does the 25-game dev set tell us about the 110".

**And the reason the notebook wins is that it never learned from the 25.** It
plays blind, so it has no transfer gap by construction. This is the argument for
grafting rather than shipping: move over only what is *structural* — the scoring
arithmetic, the RESET rewind, the ratchet objective, the button convention — and
leave behind everything that works only because the source was readable.

**The 8.592 is inflated and must not be quoted bare.** It comes from a search with
both local privileges. `_get_valid_actions()` hands it the concrete legal click
coordinates, which on the 16 click games shrinks the branching factor by 20–83×
(bp35 6.3→258 legal-vs-declared, ft09 8.0→256). Movement-only games are 1×, so
their numbers are honest and the click games' are not. Read 8.592 as *"level 0 is
shallow when you can search"* — which is a real and encouraging fact, and is what
`relive.py` is trying to cash — and not as a score.

Its one clean unprivileged success: a game with no avatar at all clears level 0 in
**6 actions against a human baseline of 17**, driven purely by the ratchet
objective.

Four results we consider established. All four are negative, and all four saved
more time than a positive would have:

**Prediction accuracy is not the bottleneck.** Two games whose forward model
predicts at 100% complete zero levels. A third predicts at 80%, is the only one
with a known goal colour, and is the only one that plans and completes a level.
Accuracy without an objective buys nothing.

**Compute is not the bottleneck.** `arc3x` contains no neural network and no LLM
at all — it is numpy on CPU. And an LLM cannot afford to play: at ~17.6 s/action
over ~300 actions × 110 games it needs 161 hours against a ~9 hour limit. A prior
experiment scored 2.68 locally and **0.60** on Kaggle purely because vLLM prefill
timed out on a shared GPU.

**Distilling a policy from the dev games does not transfer.** This is the largest
single result and it closed off a whole intended direction. Leave-*games*-out (not
leave-frames-out — that leaks, because consecutive frames of one game are nearly
identical): the learned policy picks the search-preferred action **3.75× better
than random on tune** and **1.10× on holdout**. Played as an actual agent on the 8
holdout games it scored **0.000 against random's 0.000**, clearing zero levels on
all eight.

Three independent checks confirm it is the target function and not the setup:

- **Three encodings all fail**: raw colour 1.10×, colour-by-rank 1.05×,
  colour-by-learned-role 0.86×. If it were a representation problem, one of them
  would have moved.
- **The learning curve is flat and non-monotone** over a 3.3× range of training
  data: 1.14 → 0.76 → 0.76 → 1.10. Not data-starved.
- **Random is a real control here** and was run, because "better than chance on a
  6-way choice" is 1.17× and eyeballing cannot distinguish that from 1.10×.

The conclusion we drew: **transfer what is procedural, never what is parametric.**
A search *procedure* runs on a game it has never seen; weights fitted to 25 games
describe those 25 games.

**A correct control model is worth nothing without an objective.** The one prior
that *does* transfer is the button convention: measured over all 25 games,
ACTION1 = north in 9/10 games that move under it, ACTION2 = south 12/13,
ACTION3 = west 8/8, ACTION4 = east 11/12 — mean **85%**, with tu93 as the lone
inverter. It is a fact about the competition's action protocol, not about any
game, so it is exactly the kind of thing that should transfer. Implemented with
two guards to keep it a prior rather than a hardcoding: it fires only where the
avatar was *seen to move* under a button but the displacement could not be read,
and every already-observed button must agree or nothing is filled (on tu93,
nothing is).

It gave two games all four directions where they had two, at zero extra billed
actions. **Effect on score: none** — tune 0.177, holdout 0.005, byte-identical to
the run before. The lesson is not that the convention is wrong; it is that those
two games had nowhere to *go*. Together with the prediction-accuracy result above,
the same conclusion twice: **the binding constraint is the objective, not the
controls.** No further hours go to button, delta or sprite accuracy until
something ratchets during aimless wandering.

## 7. What is open

Costed in [PLAN.md](PLAN.md). Ordered by what the measurements say, not by what
looks tractable.

**The distance, stated once.** 2.14 today; 10+ is the target; 10.57 is what
clearing levels 0 and 1 on every game is worth at baseline speed. So the whole
remaining job is **one extra level per game**, and the efficiency slack at that
depth is generous (≤1.1× baseline for 10.57, and three levels reaches 10.8 even at
1.4×). Breadth work that never clears level 1 is capped at 3.52 and cannot get
there no matter how well it is done.

1. **The objective is the binding constraint.** Established twice independently
   (§6): two games predict their next frame at 100% and clear zero levels, and the
   button convention fixed two games' control models for exactly 0.00 score. On
   every zero-scoring movement game the stall report shows `imagine:noplan` firing
   once per round in lockstep with `steer` — the board is walked and there is
   nothing to aim at. `Progress` ratchets on *level completions and consumption*,
   and nothing ratchets during aimless wandering, so `Dream.objective` returns
   `None`. (Note for anyone reading `dream.py`: this is *not* a bootstrap deadlock
   via `goal_colors` — `objective` never reads it. `progress.py` exists precisely
   because that route cannot bootstrap.)
2. **Search when there is no objective** — `relive.py`, §3.6. The current bet, and
   the reason to believe it: free search clears level 0 on 17 of 24 games in 4–35
   actions, so the answers are shallow and the missing capability is *returning to
   a promising state*. **Written, reviewed, unmeasured.** The honest risk is
   quantified: twin Go-Explore needed 100k–600k simulated steps per level on these
   games, and the billed budget is ~3000 actions — 30–200× less. The
   counter-argument is that Go-Explore has no `Mechanics`, no walking planner and
   no objective, whereas `cover()` walks the reachable set in O(n) rather than
   O(n²) — but that is an argument, not a measurement, and it will be reported as
   whatever it turns out to be.
3. **Depth.** The level-to-level model carry-over that the whole design is built
   around has never been exercised, because almost nothing clears level 0. Items 1
   and 2 are what make this testable at all. 3.52 versus 21.14.
4. ~~**The click branch has no objective.**~~ **Fixed.** It returned success on
   *any* pixel change; it now requires beating the best objective reached on this
   level. Two bugs fell out of measuring it properly, both worth recording because
   both are traps the ratchet rule invites:
   - The objective's *assembly* term was unbounded — `count(consumed) −
     count(built)` falls a little on every click that paints anything, for ever.
     Assembly is now distance below its own record, which has a floor and cannot
     be farmed.
   - A **draining step-budget bar** passed the ratchet test perfectly and the HUD
     mask did not catch it, because a bar that loses one pixel per action changes
     each individual pixel far too rarely to look volatile. Five of the eight
     holdout games were doing nothing but watching their own budget tick down —
     about 12,000 actions. Now rejected by a count-based test: moves on ≥55% of
     actions *and* by exactly one pixel ≥75% of the time ⇒ counter, not objects.
5. **Non-movement buttons.** Twelve of 25 games have a use / grab / select button
   and none are planned with; two games have *nothing else*. Deprioritised behind
   item 1 for the reason item 1 gives — more buttons is more control, and control
   is not what is missing.
6. **Convergence — and note *why*, because the obvious reason is wrong.** All 25
   games currently spend their entire action budget, and there is no test for "this
   repertoire cannot clear this level". The instinct is that this is waste. It is
   not: by §2(b), actions poured into a level that is never cleared cost **zero**
   score — the best notebook run on tn36 put 483 of 583 actions into a level it
   failed and still took the full completion cap. So the cost of burning the budget
   is *never the current level's score*; it is only ever the **later levels it
   could have bought**. Which makes the missing test not "stop spending" but
   "**stop spending on a hypothesis already known to be dead, and spend on the next
   one**". Two games show the exact shape: one drains its objective all the way to
   **zero** and then dies, another converts one colour into another two pixels per
   click for hundreds of actions. Both make real monotone progress on a quantity
   that is not the win condition, so the test is "the objective bottomed out and
   nothing happened" ⇒ retire *that objective* and re-aim, rather than either
   pushing it or going quiet.
7. **Hypothesis elimination.** The intended mechanism after item 2: hold a
   factorised space of candidate game-models (maps × goals × obstacle-sets ×
   collision rules × button-effects) as ~33 independent facts rather than 9,600
   enumerated products, and refute per billed action. The number of surviving
   candidates then scales with how ambiguous the game still is. Deliberately
   CPU-side at ~1 ms per candidate, for the timeout reason in §6. This is
   *procedural* transfer, which is the only kind §6 found that works.
8. **Shape-changing sprites** break rigid-shift matching. Largely superseded by the
   button convention, which fills the unread axes without needing to match them.
9. **The graft.** `arc3x` at 0.142 does not ship. If item 2 measures positive on
   holdout, it moves into the 2.14 notebook — the structural parts only, per §6.
   Each Kaggle attempt is ~9 h of wall clock and one play per game, so budget two
   or three, not ten.

## 8. Things we would do differently from the start

- Replicate the scorer **first**. The 3.52 ceiling reframes everything, and we
  computed it late.
- Report coverage with every mean from day one.
- Build the "why did it stop" channel before the first optimisation, not after
  several. Its first run found the largest waste in the system, and the diagnosis
  took an action-by-action trace of the objective's own arithmetic — reasoning
  about the formula produced two confident wrong answers first.
- Treat any per-frame recomputation of a learned mask as suspect. Two separate
  silent failures traced to a mask that was allowed to move.
- **Every unsupervised objective needs an explicit list of what would satisfy it
  without solving the game**, written down and tested, not argued in prose. The
  ratchet rule is strong enough to find a monotone quantity in almost any game,
  and floods, clocks and paint are all monotone. Two of those three guards were
  added only after the failure.
- **Withhold the local privileges in code from day one.** We knew "don't overfit"
  and still spent weeks on numbers inflated by `deepcopy` and by a click oracle,
  because the privileges were available by default and avoiding them was a matter
  of remembering to. `graded.py` now makes them unavailable, and every measurement
  taken before it existed had to be thrown away.
- **Price the primitives before designing around them.** Go-Explore was built
  twice: once assuming a free rewind, then again once RESET was read properly and
  the rewind turned out to cost `1 + depth` actions. Pricing it changes the
  *selection rule*, not a constant — cost zero for the node already stood on is
  what makes the search stick rather than thrash — so it could not be retrofitted.
- **A "does this abstraction collapse anything" check belongs next to every state
  key.** Two separate silent zeros came from a key that was effectively bijective:
  once from a clock pixel, once from clock-free frames that were still too fine.
  Both times nothing crashed and the score was simply 0. Cells created per action
  spent is one line to compute and would have caught both immediately.
- **Dump the whole configuration into every artifact bundle, including the parts
  that live in a notebook cell.** The largest score swing in the project's records —
  1.36 versus 10.71 on the same game — cannot be attributed, because the repo
  commits and `taaf_setup_env.json` are byte-identical between the two runs and the
  graft-flags dict was never written down. It was in memory at the time. One
  `json.dump` would have turned the most expensive measurement we have into an
  answer instead of a question.
- **Derive the score consequence of a knob before tuning the knob.** Two examples,
  same mistake in opposite directions. The search throttle was written as
  `RELIVE_ACTIONS if L == 0 else RELIVE_ACTIONS // 4` on the reasoning that later
  levels must be protected — but the score cost of `m` extra actions on a level `n`
  old is `(n/(n+m))²`, so the deciding variable is what the level has *already*
  cost, not its index. And `STALE_LIMIT` was justified by "a level worth 1/45th of
  the score should not absorb all of it", when a level that is never cleared scores
  0 regardless and the real cost is the *later* levels the actions could have
  bought. Both knobs were reasonable; both rationales were wrong, and a wrong
  rationale is what makes the next change wrong too.

