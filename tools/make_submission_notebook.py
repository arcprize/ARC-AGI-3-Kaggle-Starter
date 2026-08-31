"""Build the submittable notebook: the 2.14 notebook plus this session's two grafts.

WHY A GENERATOR AND NOT A HAND-EDIT
-----------------------------------
The source notebook is a 700-line JSON document with one long line per cell. Editing
it by hand is how you silently corrupt a nine-hour run. This script does exactly
three things, all mechanical and all re-runnable:

  1. loads ``arc3-duck-v12 (1).ipynb``,
  2. inserts one markdown cell + one code cell immediately after the cell that
     defines ``GRAFT_FLAGS`` (the notebook's own designated customization hook, and
     therefore the last point before ``bm.run()`` where ``bm``/the source tree are
     loaded but nothing has been played),
  3. strips stale outputs, so the local ``CalledProcessError`` traceback from a
     Windows dry-run does not travel to Kaggle looking like a real failure.

WHY THESE TWO GRAFTS AND NOT THE arc3x CODE
-------------------------------------------
The notebook imports its solver from a **read-only Kaggle dataset**
(``thtennant/taaf-kaggle-source-share-fork``). Nothing in this repo can change that
dataset, so any graft has to be expressible as inline notebook code. That excludes
the arc3x modules — they need the local twin engine and score 0.142 over 25 games
standalone — and it includes exactly the two changes below, which are a wrapper and
a string.

  * **Graft A, the level probe.** ``taaf.game.Game.finish_game`` already computes
    ``actions_per_level`` and ``base_actions_per_level`` and prints them
    (``game.py`` 636-650), but a true submission sets
    ``TAAF_MINIMAL_DIAGNOSTICS=1``, which suppresses the JSON/HTML artifacts, so
    the numbers only ever existed in a log nobody kept. This appends one JSON line
    per finished game to ``/kaggle/working/level_probe.jsonl``. It reads fields that
    the framework has already filled in and writes ~110 short lines over nine
    hours. It cannot change what the solver does.

    This is the measurement the whole plan has been blocked on: 2.14 is equally
    consistent with "level 0 on 60% of games" and "two levels on 5 games, zero on
    20", and those two need opposite fixes.

  * **Graft B, the family priors.** ``inference.agent.tool_agent._build_system_prompt``
    concatenates addendum strings; this appends one more. The text carries the two
    findings from this session that generalise past the 25 readable dev games:
    the win condition is usually a *cover* predicate and is therefore drawn on
    frame 0, and action-efficiency only costs score on levels you actually clear.
    ``benchmark.py`` runs games with ``asyncio.create_task`` in a single process
    (163-199), so patching the module attribute reaches every game.

  * **Graft C, recovery minus its action tax.** ``GRAFT_FLAGS`` now sets
    ``recovery: True``. ``docs/EXPERIMENT_LOG.md`` regressed on recovery twice
    (Exp 2 local 0.03, Exp 3 Kaggle 0.82) and both times names ONE cause: "the R2
    probe tax". R2 spends up to 16 real actions per stalled level; R1 (clear a
    death-spiralled chat history, rewrite the world model) and R3 (distil mechanics
    into ``cross_level_notes``, the one knowledge key the vendor level wipe spares)
    spend **zero** actions by construction (``recovery.py`` 51-56). R2 is switchable
    off through a guard that already exists — ``build_probe_plan`` returns
    ``plan[:PROBE_MAX_ACTIONS]`` (284) and ``_do_probe`` returns on an empty plan
    (607-608) — so ``PROBE_MAX_ACTIONS = 0`` keeps the two free mechanisms and drops
    the charged one. R3 is the only thing in this bundle aimed at *depth*, and depth
    is the sole route past the 3.52 level-0 ceiling.

  * **Graft D, an economically true efficiency note.** ``build_efficiency_note``
    (``agent_ext.py`` 265-269) opens every turn with "every wasted action costs you
    quadratically" and then tells the model "you are 38.0x over the target" on a
    level it has not cleared. Measured, that is false in the direction that loses
    score: an uncleared level contributes zero whatever you spent there, so those
    actions are sunk and free. This is the most salient channel the model has — a
    dynamic per-turn user-prompt line — and it currently invites the model to write
    off exactly the levels the holdout data says it clears at 38-45x baseline. One
    appended clause keeps the pressure and blocks the write-off inference.

WHAT IS DELIBERATELY NOT IN HERE
--------------------------------
  * ``arc3x/markers.py``. Its own measurement (``why_markers.py``, run today)
    scored ``source-graded rows passing: 0/1`` — on the one row with ground truth
    read from the game's source it proposed the floor tiling instead of the exit.
    Ranking candidate destinations by repeat-count promotes scenery. Graft B states
    the *prior* the census established and lets the model apply it, which is the
    part that measured out true; the detector that failed stays home.
  * The ACTION1-4 = N/S/W/E button convention. True in 90-100% of the dev games,
    and measured at **exactly 0.00** when seeded into the arc3x planner. Controls
    were never the bottleneck, an LLM finds them in a few actions anyway, and a
    third section would dilute the two that address something the model lacks.
  * Any change to ``context_window``. It is the one knob with a real argument on
    both sides. Against 57344: three independent latency diagnoses in
    ``EXPERIMENT_LOG.md`` (Exp 4 "57K token latency penalty", Exp 11 "vLLM prefill
    timeouts on Kaggle's shared GPUs, stranding games at 0", and the 0.35 entry in
    the baseline progression), plus the only monotone trend in the whole log — the
    same flag set scored 1.06 at 32768 (Exp 6) and 0.95 at 45056 (Exp 7). For it:
    2.14 was measured at 57344, and shrinking retained history works against depth,
    which is the actual goal. Graft C's R1 resolves the tension without the tradeoff
    — it clears history only on a detected death spiral, which is the pathological
    case those latency reports describe, instead of capping every game always.
  * ``banking``, ``transfer``, ``state_dedup``, ``schema_notes``, ``schema_helpers``,
    ``schema_void``. All seven unused flags were tested on the real leaderboard over
    Exp 3-11 and every combination scored below the three-flag baseline: banking-only
    1.10, schema_helpers+schema_void+transfer 1.06, state_dedup 0.77, schema_notes
    0.47. Exp 11 is the instructive one — it took the **highest local score ever
    recorded** (2.6848, and the first ever m0r0 level-0 clear) to 0.60 on Kaggle.

Run:
    .venv/Scripts/python.exe tools/make_submission_notebook.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "1.33 scored in arc agi 3 competiotn in kaggle" / "arc3-duck-v12 (1).ipynb"
OUT = REPO / "1.33 scored in arc agi 3 competiotn in kaggle" / "arc3-duck-v14-recovery.ipynb"

# The anchor. Chosen because it is inside the notebook's own "## 6. Customization
# hook" cell, documented there as "the safe place for one-off experiments once the
# deployed bundle has loaded" — after the source tree is importable and `bm` is
# unpickled, before cell 7 reassigns `bm.games` and awaits `bm.run()`.
ANCHOR = "GRAFT_FLAGS = {"

# The one edit made to a vendor cell, rather than added after it. `install()` is
# called in the same cell immediately below the dict, so `recovery` has to be in the
# dict itself to be wired; nothing else in that cell is touched.
FLAGS_OLD = (
    'GRAFT_FLAGS = {"efficiency": True, "retry_guard": True, "shortcircuit": True, "context_window": 57344}'
)
FLAGS_NEW = '''\
# v14 (2026-08-23): "recovery" armed, but ONLY with its action-spending mechanism
# provably dead. docs/EXPERIMENT_LOG.md regressed on recovery twice (Exp 2 local
# 0.03, Exp 3 Kaggle 0.82) and attributes both to R2's probe tax ALONE: R2 spends up
# to PROBE_MAX_ACTIONS=16 real actions on any level past 120. Its two siblings cost
# ZERO actions by construction (recovery.py:51-56) — R1 clears a death-spiralled
# chat history and rewrites the world model, R3 carries the mechanics of a cleared
# level across the vendor level-transition wipe via cross_level_notes. R3 is the
# only mechanism in this bundle aimed at level 1+, and level-0 breadth alone is
# capped at 3.52 by the scoring formula.
#
# The order below is load-bearing. install() wires the chain, so R2 must be dead
# BEFORE recovery is armed — arming it with a live R2 reproduces the 0.82 run
# exactly. build_probe_plan returns plan[:PROBE_MAX_ACTIONS] (recovery.py:284) and
# _do_probe returns on an empty plan before it even counts the probe (607-608), so 0
# disables R2 through a guard that already exists. If ANY of this fails, recovery is
# never added and what runs is the byte-exact three-flag config that scored 2.14.
#
# The other six unused flags stay off: every leaderboard run that armed them scored
# below this baseline (banking 1.10, helpers+void+transfer 1.06, state_dedup 0.77,
# schema_notes 0.47). context_window stays 57344 because 2.14 was measured with it,
# and R1 attacks the history bloat that the log's latency reports describe without
# shrinking retained history on games that are doing fine.
GRAFT_FLAGS = {"efficiency": True, "retry_guard": True, "shortcircuit": True, "context_window": 57344}

try:
    import taaf_grafts.recovery as _recovery

    _probe_before = _recovery.PROBE_MAX_ACTIONS
    _recovery.PROBE_MAX_ACTIONS = 0
    # Verify with the real pure function rather than trusting the constant.
    if _recovery.build_probe_plan(["ACTION1", "ACTION2", "ACTION6"]):
        _recovery.PROBE_MAX_ACTIONS = _probe_before
        raise RuntimeError("probe plan still non-empty at PROBE_MAX_ACTIONS=0")
    GRAFT_FLAGS["recovery"] = True
    print(f"[taaf_grafts] recovery armed, R2 disabled (PROBE_MAX_ACTIONS {_probe_before} -> 0); R1+R3 cost 0 actions")
except Exception as _exc:  # noqa: BLE001 — never arm recovery with a live probe tax
    print(f"[taaf_grafts] recovery NOT armed ({type(_exc).__name__}: {_exc}); running the 3-flag 2.14 config")'''

MARKDOWN = """\
## 6b. Session grafts — level probe, priors, recovery-minus-tax, honest budget note

Four independent grafts, added 2026-08-23. Each is wrapped so that **any** failure
leaves the stock path untouched, and each is reported by name below so the run log
says which ones actually installed. Set any `INSTALL_*` flag to `False` to ship
without it.

**A. Level probe (inert).** `taaf.game.Game.finish_game` already computes
`actions_per_level` and `base_actions_per_level`; a true submission sets
`TAAF_MINIMAL_DIAGNOSTICS=1` and never persists them. This appends one JSON line per
finished game to `/kaggle/working/level_probe.jsonl`. It reads values the framework
has already filled in, so it cannot change what the solver does. It exists because
a score of 2.14 is equally consistent with *level 0 on most games* and *two levels
on a handful*, and those two states need opposite fixes. It also records `turns` and
`wallclock_s`, which together settle the latency question that three separate
entries in `EXPERIMENT_LOG.md` blame for a regression without ever measuring.

**B. Family priors (behavioural).** Appends one addendum to the agent's system
prompt, carrying two findings that generalise beyond any single game:

1. *The goal is usually drawn on frame 0.* Across the readable dev games, 10 of 13
   win conditions are one predicate — every object of kind A co-located with an
   object of kind B — and kind B is on the board before the first action, as a set
   of small, identical, static objects. The agent already has each object's shape
   hash from `segmentation`; it was never told what to look for with it.
2. *Efficiency only costs score where you clear.* Level score is
   `(baseline/actions)²`, awarded only for cleared levels, so actions on a level you
   never clear are free — explore level 0, execute crisply later, and never give up
   early.

**C. Recovery, minus the mechanism that regressed it.** `GRAFT_FLAGS` above now sets
`recovery: True`. The experiment log tried recovery twice and lost both times — Exp 2
at local 0.03, Exp 3 at Kaggle 0.82 — and attributes both to *one* cause, "the R2
probe tax": R2 spends up to 16 real actions on any level past 120 actions. Its two
siblings cost **zero** actions by construction: R1 clears a chat history that has
death-spiralled and rewrites the world model in place, and R3 distils the mechanics
learned on a cleared level into `cross_level_notes` — the only knowledge key the
vendor's level-transition wipe spares. This graft sets `PROBE_MAX_ACTIONS = 0`,
which empties the probe plan and makes `_do_probe` return at its existing guard.

R3 is the reason this is worth doing. Level-0 breadth is capped at 3.52 by the
scoring formula; every point above that has to come from level 1+, and the
documented sk48 failure is precisely "level 0 cleared, then level 1 stalled to the
wall because the wipe discarded every mechanic learned".

**D. An efficiency note that is economically true.** `build_efficiency_note` opens
every turn with *"every wasted action costs you quadratically"* and then reports
*"you are 38.0x over the target"*. On a level the agent has not cleared that is
false in the direction that loses score: an uncleared level contributes zero no
matter what was spent, so those actions are sunk and free. This is the most salient
text the model sees — dynamic, per-turn, in the user prompt — and as written it
invites writing off exactly the levels the holdout data says get cleared at 38-45x
baseline. One appended clause keeps the pressure and removes the wrong inference.

Priors 2 and D are measured from this notebook's own event log; prior 1 was read from
the 25 dev games' source. All are stated as defaults that live evidence overrides.
"""

CODE = '''\
# --- session grafts, 2026-08-23 -----------------------------------------------
# A: level probe (inert instrumentation).  B: family priors (system-prompt text).
# Both blanket-guarded: on ANY exception the stock 2.14 behaviour is what runs.
# Set either flag to False to ship without it.
INSTALL_LEVEL_PROBE = True
INSTALL_FAMILY_PRIORS = True
INSTALL_EFFICIENCY_RIDER = True

LEVEL_PROBE_PATH = WORKING_DIR / "level_probe.jsonl"


# --- Graft A: one JSON line per finished game --------------------------------
# `Game.finish_game` (taaf/game.py:596) already fills in every field read here and
# prints them at 645-650; TAAF_MINIMAL_DIAGNOSTICS=1 is why a real submission keeps
# no artifact of them. Appending is O(1) per game, ~110 lines over nine hours.
def _install_level_probe() -> str:
    import taaf.game

    game_cls = taaf.game.Game
    if getattr(game_cls.finish_game, "_level_probe", False):
        return "already installed"
    original = game_cls.finish_game

    def finish_game(self, generated_tokens: int = 0, uncached_input_tokens: int = 0) -> None:
        # finish_game is idempotent by an early return on final_score (game.py:615),
        # so it can legitimately be called twice; only the call that actually scores
        # the game should emit a row.
        run_before = getattr(self, "game_run", None)
        first_call = run_before is not None and run_before.final_score is None
        original(self, generated_tokens=generated_tokens, uncached_input_tokens=uncached_input_tokens)
        if not first_call:
            return
        try:
            run = self.game_run
            base = run.base_actions_per_level
            row = {
                "game_id": run.game_id,
                "state": run.state,
                "levels_completed": run.levels_completed,
                "number_of_levels": run.number_of_levels,
                "final_score": run.final_score,
                "actions_per_level": list(run.actions_per_level),
                # None here means the framework hit its documented silent fallback
                # (game_api.py:232) and every efficiency ratio in the run is unusable.
                "base_actions_per_level": list(base) if base is not None else None,
                "wallclock_s": round(float(run.final_wallclock_seconds or 0.0), 1),
                "turns": len(run.history),
                "generated_tokens": sum(r.generated_tokens for r in run.history) + run.final_generated_tokens,
                "note": run.solver_note,
            }
            with open(LEVEL_PROBE_PATH, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, default=str) + "\\n")
        except Exception as exc:  # noqa: BLE001 — a diagnostics write must never sink the run
            print(f"[graft] level probe row failed: {type(exc).__name__}: {exc}")

    finish_game._level_probe = True
    game_cls.finish_game = finish_game
    return f"wrapped taaf.game.Game.finish_game -> {LEVEL_PROBE_PATH}"


# --- Graft B: the two family priors ------------------------------------------
# Phrased as defaults rather than rules, and grounded in what the agent can already
# see: `segmentation` gives every object a colour+shape hash (prompts.py:43), which
# is exactly the key that groups "several identical things drawn on the board".
FAMILY_PRIORS_ADDENDUM = """

# Two priors about this family of games

These hold across the family these puzzles are drawn from, not just this one. Treat
them as strong starting defaults: evidence from THIS game always overrides them.

## 1. The goal is usually already drawn on the first frame

In most of these games the win condition is a *cover* predicate:

    every object of kind A must end up co-located with an object of kind B.

Kind B is on the board before you take a single action, and it looks like this: a
group of two or more objects that are (a) identical to each other, so same colour
and same shape and therefore the same segmentation hash, (b) small, because they
mark a place rather than being scenery or wall, and (c) static, because they do not
move when you act. Kind A is what you move or push.

So in your first turns, before experimenting: group the segmentation nodes of the
current frame by hash. Any group of 2 or more small identical shapes is a candidate
destination set, and the size of that group tells you how many things you probably
have to deliver. That reframes the game from "what does this button do" to "get
each mover onto one destination" - and it means you can often read the objective
off the board instead of inferring it from a reward you have not seen yet.

Recognise the variants:
- the destination may be a single object, not a group, so a lone small static shape
  distinct from everything else is also a candidate;
- the predicate may be at pixel level - a workspace region must be made equal to a
  reference region drawn elsewhere on the board, in which case your progress meter
  is the count of mismatched pixels between the two;
- it may be "remove every object of colour X", in which case the remaining count of
  X is your progress meter;
- it may be ordered, so covering the destinations in the wrong sequence undoes
  earlier progress.

If the first frame has no repeated small static group and no distinct lone marker,
this prior does not apply - fall back to experimenting.

## 2. Where being efficient actually pays, and where it does not

A level scores (baseline_actions / your_actions) squared, and it scores that ONLY if
you clear it. A level you never clear contributes zero no matter what you did on it,
so the actions you spent there cost you nothing.

Two consequences, and they point in opposite directions:

- Exploring is cheap on the first level, and free on any level you end up not
  clearing. Spend actions to establish the mechanics: which button moves what, what
  blocks movement, what the objective is, what an action does when nothing appears
  to change. The first level also carries the smallest weight of any level in the
  game. But "cheap" here means cheap *relative to the later levels*: if the first
  level turns out to be the only one you clear, then its efficiency is your entire
  score for this game. So explore it freely while you still do not understand it,
  and the moment you can see the objective, go and complete it rather than
  continuing to explore for its own sake.
- Being crisp pays on later levels. Once you know the mechanics, execute. Later
  levels carry more weight and charge every wasted action quadratically, and you
  should re-derive nothing you already established earlier in the same game - the
  mechanics do not change between levels, only the layout does.

And never stop trying because you have used a lot of actions. Exhausting your
actions on a level you cannot solve costs exactly zero, so there is no situation in
which giving up scores better than continuing to try.

## 3. RESET is a cheap rewind, not a failure

RESET costs ONE action and restores the level you are on to its opening position.
These games are almost always deterministic, so the same actions from the opening
position produce the same board again.

That makes RESET the cheapest tool you have, and it is worth using deliberately:

- If you have made the level unsolvable - pushed something into a corner, consumed
  something you needed, blocked the only corridor - do not try to undo it move by
  move and do not keep playing a dead position. RESET costs 1 action; grinding on a
  ruined board costs hundreds and cannot succeed.
- If you have learned the layout but executed badly, RESET and walk the route you
  now know. One rewind plus a clean run is usually far fewer actions than repairing
  a mess, and only the actions you actually spend are counted.
- Use it to run a controlled experiment: try something destructive on purpose to
  learn what a mechanic does, then RESET and apply what you learned. The knowledge
  survives the reset even though the board does not.

Be aware of one thing: a reset restarts the level, so anything you did on this level
is undone. It does not undo levels you have already completed.

## 4. Default control conventions

Start from these and let evidence correct them - one probe of each button tells you
whether they hold in this game.

- The four directional buttons are usually up, down, left, right in that order. If
  the game has a movable avatar, expect the first four actions to move it.
- If none of the directional buttons move anything, this is probably not an avatar
  game. Look for two other patterns: a separate interact/use button that acts on
  whatever you are standing on or next to, and direct clicking, where you select an
  object and then a destination rather than walking.
- An action that appears to do nothing is still information. It may be blocked by a
  wall, it may need a precondition you have not met, or it may change something
  off-screen or in a counter rather than on the board. Check the whole frame for a
  small change before concluding a button is dead, and do not press a dead-looking
  button many times in a row to be sure - twice is enough.
"""


def _install_family_priors() -> str:
    import inference.agent.tool_agent as tool_agent

    original = tool_agent._build_system_prompt
    if getattr(original, "_family_priors", False):
        return "already installed"

    def _build_system_prompt(*, tool_output_tokens: int) -> str:
        return original(tool_output_tokens=tool_output_tokens) + FAMILY_PRIORS_ADDENDUM

    _build_system_prompt._family_priors = True
    # benchmark.py plays games with asyncio.create_task in this same process
    # (benchmark.py:163-199), and tool_agent.py:941 resolves this name from module
    # globals at call time, so rebinding the module attribute reaches every game.
    tool_agent._build_system_prompt = _build_system_prompt
    return f"appended {len(FAMILY_PRIORS_ADDENDUM)} chars to the agent system prompt"


# --- Graft D: make the per-turn efficiency note economically true --------------
# build_efficiency_note opens with "every wasted action costs you quadratically"
# (agent_ext.py:265-269) and then reports "Level N: you have used 1633 actions; a
# strong score needs about 43 or fewer. You are 38.0x over the target." (286-292).
# On an UNCLEARED level that framing is false in the direction that loses score: the
# level contributes zero whatever was spent, so the actions are sunk and free. The
# holdout measurement is unambiguous - ft09 38x, cd82 45x, vc33 41x, m0r0 39x all
# CLEARED level 0 and scored ~0.06 each, which is strictly better than the 0.00 they
# score by abandoning it. This note is the most salient channel there is (dynamic,
# per-turn, user prompt), so the wrong inference here is expensive. Appending rather
# than rewriting keeps every stall line and the commit-and-stop reminder intact.
EFFICIENCY_NOTE_RIDER = (
    "SCORING FACT — read the ratio above correctly: it only costs you score on a "
    "level you actually CLEAR. A level you never clear scores zero no matter how "
    "few actions you spent, so every action already spent on this level is sunk and "
    "cost you nothing. Being far over target is therefore never a reason to give up "
    "on this level or to stop exploring it: clearing it slowly beats not clearing "
    "it, and clearing it also unlocks the later levels, which are worth more. Be "
    "efficient because the NEXT levels are expensive, not because this one is lost."
)


def _install_efficiency_rider() -> str:
    import taaf_grafts.agent_ext as agent_ext

    original = agent_ext.build_efficiency_note
    if getattr(original, "_rider", False):
        return "already installed"

    def build_efficiency_note(**kwargs) -> str:
        note = original(**kwargs)
        # An empty note means "nothing to report"; stay silent exactly as stock does,
        # so turns that were quiet before are still byte-identical.
        return f"{note}\\n{EFFICIENCY_NOTE_RIDER}" if note else note

    build_efficiency_note._rider = True
    # EfficiencyToolAgent._efficiency_note calls this by bare name (agent_ext.py:490),
    # which resolves through agent_ext's module dict at call time, so rebinding the
    # module attribute reaches the live agent even though it was built by install().
    agent_ext.build_efficiency_note = build_efficiency_note
    return f"appended {len(EFFICIENCY_NOTE_RIDER)} chars to the per-turn efficiency note"


SESSION_GRAFTS = {}
for _name, _enabled, _installer in [
    ("level_probe", INSTALL_LEVEL_PROBE, _install_level_probe),
    ("family_priors", INSTALL_FAMILY_PRIORS, _install_family_priors),
    ("efficiency_rider", INSTALL_EFFICIENCY_RIDER, _install_efficiency_rider),
]:
    if not _enabled:
        SESSION_GRAFTS[_name] = "disabled"
        continue
    try:
        SESSION_GRAFTS[_name] = _installer()
    except Exception as exc:  # noqa: BLE001 — any graft failure falls back to stock
        SESSION_GRAFTS[_name] = f"FAILED {type(exc).__name__}: {exc}"
    print(f"[graft] {_name}: {SESSION_GRAFTS[_name]}")

# Record what this run configured, next to graft_flags.json, so the result is
# attributable — the single lesson from two runs that scored 1.36 and 10.71 on the
# same game with no record of what differed.
try:
    (WORKING_DIR / "session_grafts.json").write_text(json.dumps(SESSION_GRAFTS, indent=2, sort_keys=True) + "\\n")
except Exception as exc:  # noqa: BLE001
    print(f"[graft] could not persist session graft status: {type(exc).__name__}: {exc}")
'''


def main() -> int:
    if not SRC.is_file():
        print(f"source notebook not found: {SRC}")
        return 1
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    cells = nb["cells"]

    anchor_at = next(
        (i for i, c in enumerate(cells) if c.get("cell_type") == "code" and ANCHOR in "".join(c.get("source", []))),
        None,
    )
    if anchor_at is None:
        print(f"anchor {ANCHOR!r} not found — refusing to guess an insertion point")
        return 1

    def lines(text: str) -> list[str]:
        """nbformat stores source as a list of lines, each keeping its newline."""
        out = text.splitlines(keepends=True)
        if out and out[-1].endswith("\n"):
            out[-1] = out[-1][:-1]
        return out

    already = [i for i, c in enumerate(cells) if "INSTALL_FAMILY_PRIORS" in "".join(c.get("source", []))]
    for i in reversed(already):
        del cells[i]
    if already:
        anchor_at = next(
            i for i, c in enumerate(cells) if c.get("cell_type") == "code" and ANCHOR in "".join(c.get("source", []))
        )

    cells.insert(anchor_at + 1, {"cell_type": "markdown", "metadata": {}, "source": lines(MARKDOWN)})
    cells.insert(
        anchor_at + 2,
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(CODE)},
    )

    # Arm `recovery` in the vendor cell itself: `install(bm, flags=GRAFT_FLAGS)` is
    # called a few lines below the dict, so a flag added anywhere later is ignored.
    # Exact-match the whole assignment line and refuse rather than guess, so an
    # upstream edit to that line can never be silently half-applied.
    anchor_src = "".join(cells[anchor_at]["source"])
    if FLAGS_OLD not in anchor_src:
        if "recovery" in anchor_src:
            print("anchor cell already arms recovery — regenerating from a v14 notebook?")
        else:
            print(f"could not find the exact GRAFT_FLAGS line to replace:\n  {FLAGS_OLD}")
        return 1
    if anchor_src.count(FLAGS_OLD) != 1:
        print(f"GRAFT_FLAGS line appears {anchor_src.count(FLAGS_OLD)} times — refusing to edit")
        return 1
    cells[anchor_at]["source"] = lines(anchor_src.replace(FLAGS_OLD, FLAGS_NEW))

    # Strip stale outputs. The dry-run on this Windows box left a CalledProcessError
    # traceback in the pip cell, which on Kaggle would read as a real failure.
    stripped = 0
    for cell in cells:
        if cell.get("cell_type") == "code":
            stripped += 1 if cell.get("outputs") else 0
            cell["outputs"] = []
            cell["execution_count"] = None

    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"anchor cell index      : {anchor_at}")
    print(f"cells before / after   : {len(cells) - 2} / {len(cells)}")
    print(f"stale outputs stripped : {stripped}")
    print(f"wrote                  : {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
