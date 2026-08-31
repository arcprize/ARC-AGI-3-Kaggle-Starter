"""Build v15 from v14: the bundle-path bug, and the one prior the click data earned.

WHY A NEW FILE RATHER THAN AN EDIT
----------------------------------
v14 was submitted on 2026-08-23. Editing it in place would destroy the only record
of what was actually graded, and this project's most expensive lesson is exactly
that: two runs scored 1.36 and 10.71 on the same game with no record of what
differed between them. So v14 stays byte-identical and this writes v15.

PATCH A - the bundle-path bug, which cost the last run every behavioural graft
-----------------------------------------------------------------------------
``_find_bundle_dir`` returned the FIRST ``rglob`` hit for the marker file. Index 0 of
``DATASET_SOURCES`` is then forced to whatever it returns. On the 2026-08-23 run two
attached datasets carried the marker, rglob returned the one without
``src/taaf-grafts``, and ``session_grafts.json`` recorded
``ModuleNotFoundError: No module named 'taaf_grafts'``. Every entry in
``GRAFT_FLAGS`` - efficiency, retry_guard, shortcircuit, context_window - plus the
recovery layer and the efficiency rider were silently lost, and the run played stock
for 2h12m. ``rglob`` order is filesystem order, so this was not reproducible and a
rerun could not have caught it.

The fix sorts the hits (deterministic), prefers the bundle that actually contains the
graft repo (correct), and prints an unmissable warning when none does (diagnosable).
It deliberately does NOT raise: on a real submission, falling back to stock scores
about 2.14, and raising scores zero.

PATCH B - the click-space prior, and the sampling error it survived
-------------------------------------------------------------------
Measured with the engine as oracle: freeze a state, click a cell from that frozen
state, count how many distinct boards come back. At 25 sampled cells this said tn36
had 1.0 distinct outcomes, and the first draft of this prior therefore told the agent
that three agreeing frames mean "the coordinate is decoration, stop searching".

The exhaustive sweep over all 4096 cells refuted that. tn36 has ELEVEN distinct
outcomes with 96% of cells on one of them; the 5x5 lattice had sampled only the
boring 96% and inverted the conclusion. Frame 0, every cell:

    m0r0 1   sp80 1   ka59 2   vc33 3   lf52 3   dc22 3   cn04 3
    s5i5 5   tn36 11   su15 144   r11l 3096          (of 4096, modal 24-100%)

So the truth is neither 4096 nor 1. The click space is a handful of outcomes
dominated by one "nothing special" result, plus a few rare cells that do everything
interesting. That inverts the advice: a couple of agreeing frames is the EXPECTED
observation even when special cells exist, so it can never license "stop searching" -
and because the special cells are rare, they must be hunted deliberately rather than
by sampling the board evenly, which would spend the whole budget missing them.

The prior stays a *procedure the agent runs*, names no game, and is overridden by
evidence from the live game either way. What changed is the conclusion it draws.

PATCH C - the batching prior, because the run is bound by a clock and not by actions
------------------------------------------------------------------------------------
``level_probe.jsonl`` from the 2026-08-23 run reports ``wallclock_s`` of 7920.2 for all
four games, all four ``"state": "gave_up"``, and ``total wallclock`` of 31680.9s =
4 x 7920. The games ran concurrently and every one was cut off by the job deadline
mid-play. Nothing stopped for a strategic reason, and no action cap was reached.

Actions per game were therefore set by token cost alone:

    m0r0  111007 tok / 168 act = 661 tok/act   0 levels
    sk48  204241 tok / 417 act = 490 tok/act   0 levels
    dup   177493 tok / 428 act = 415 tok/act   1 level
    tn36  155803 tok / 395 act = 394 tok/act   1 level   (118 LLM turns = 3.35 act/turn)

The harness already batches - tool_agent.py:1248 tells the model to "prefer batching it
in one call" - so the lever is the ratio, not new machinery. What the model cannot know,
and what this prior supplies, is the two costs of a batch, both read out of
``Solver.step_env`` (solver.py:604-661):

  * the loop breaks on stopped / invalid_action / action_error / run_complete /
    game_over / level_completed, and on NOTHING else - in particular not on an
    unchanged board, so a batch walked into a wall bills every action in it;
  * ``final_payload = dict(executed_payloads[-1])`` returns only the LAST frame, so
    intermediate observations are discarded and only ``board_changed`` is OR-ed.

Hence the rule is "batch what you can predict, single-step what you are probing",
which is a decision procedure rather than a length target. The level_completed break is
worth telling the model too: it is a guarantee that a long batch cannot spill actions
across a level boundary, where they would start costing score.

Run:
    .venv/Scripts/python.exe tools/make_v15.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
NB_DIR = HERE / "1.33 scored in arc agi 3 competiotn in kaggle"
SRC = NB_DIR / "arc3-duck-v14-recovery.ipynb"
DST = NB_DIR / "arc3-duck-v15-clickspace.ipynb"

# -- Patch A ------------------------------------------------------------------

OLD_BUNDLE = '''# Locate the source dataset by its marker file rather than a fixed mount path.
def _find_bundle_dir() -> Path:
    for marker in Path("/kaggle/input").rglob(DATASET_BUNDLE_MARKER):
        return marker.parent
    raise RuntimeError("TAAF source bundle not found under /kaggle/input.")
'''

NEW_BUNDLE = '''# Locate the source dataset by what it CONTAINS, not by filesystem order.
#
# v14 (2026-08-23) lost every behavioural graft to the previous version of this
# function. It returned the first Path("/kaggle/input").rglob(marker) hit, and index 0
# of DATASET_SOURCES is forced to whatever it returns. Two attached datasets carried
# the marker; rglob returned the one WITHOUT src/taaf-grafts; session_grafts.json
# recorded "ModuleNotFoundError: No module named 'taaf_grafts'"; GRAFT_FLAGS
# (efficiency, retry_guard, shortcircuit, context_window), the recovery layer and the
# efficiency rider were all silently discarded, and the run played stock for 2h12m.
# rglob order is filesystem order, so the failure was not reproducible and a rerun
# could not have found it.
#
# sorted() makes the choice deterministic; the graft-repo test makes it correct; the
# warning makes it diagnosable from the log alone. It does not raise on purpose: in a
# real rerun, stock scores about 2.14 and an exception scores zero.
GRAFT_REPO_DIR = "taaf-grafts"


def _find_bundle_dir() -> Path:
    markers = sorted(Path("/kaggle/input").rglob(DATASET_BUNDLE_MARKER))
    if not markers:
        raise RuntimeError("TAAF source bundle not found under /kaggle/input.")
    with_grafts = [m.parent for m in markers if (m.parent / "src" / GRAFT_REPO_DIR).is_dir()]
    if with_grafts:
        if len(markers) > 1:
            print(
                f"taaf.kaggle: {len(markers)} source bundles attached; chose the one "
                f"carrying src/{GRAFT_REPO_DIR}"
            )
        return with_grafts[0]
    print(
        f"taaf.kaggle: *** WARNING *** no attached bundle contains src/{GRAFT_REPO_DIR}, so "
        "every behavioural graft will fall back to stock (this is what happened on "
        "2026-08-23). Attach thtennant/taaf-kaggle-source-share-fork. Bundles seen: "
        + ", ".join(str(m.parent) for m in markers)
    )
    return markers[0].parent
'''

# -- Patch B ------------------------------------------------------------------

# Appended to FAMILY_PRIORS_ADDENDUM, so it lands in the agent's system prompt next to
# the other four priors. Anchored on the last line of section 4 rather than on the
# closing quotes, which appear elsewhere in the cell.
ANCHOR = "  button many times in a row to be sure - twice is enough.\n"

CLICK_PRIOR = '''
## 5. Clicking: find out what the coordinate is worth before you spend actions on it

A 64x64 board offers 4096 different clicks and you will never have that many actions.
So the first question about MOUSE is not "where do I click" but "what does the
coordinate do here", and three or four actions settle it.

Click one cell, then a cell far away from it, and compare the frames - not only to
each other, but each to the frame you had immediately before that click. Three things
can happen and they lead to completely different plans.

**Nothing changes at all.** Clicking is not this game's mechanism at the moment.
Perhaps nothing is selected yet, perhaps only the other buttons act. Record it and
stop spending actions on MOUSE until something else changes the situation.

**The board changes, and changes the same way wherever you click.** The coordinate is
not being read: MOUSE is a button. Use it as one - plan *sequences* of presses and
stop thinking about where.

**Different cells produce different boards.** Now position is the game, and the effect
is almost always one of four kinds. One click of each tells you which:

- it moves something to the cell you clicked - so a click is a destination, and you
  can place a thing directly instead of walking it there;
- it sets the clicked cell to a fixed colour - so there is a selected colour and you
  are drawing;
- it changes the clicked cell into something that depends on what was already there -
  a toggle, so state matters and clicking twice may undo;
- it changes the board somewhere else entirely - so that region is a button rather
  than a place, and its position is worth writing down.

**The trap is in the middle case.** In these games one single result usually covers
the great majority of the board, while a few scattered cells - sometimes only a
handful in the whole 4096 - do something else entirely, and those few are normally the
controls that matter. So two cells agreeing is exactly what you should expect even
when special cells exist. It is weak evidence that the coordinate is ignored and no
evidence at all that there is nothing to find. Never conclude from a few agreeing
clicks that the board has no special cells.

That is also why you must not hunt for them by sampling the board evenly. If a few
cells in a few thousand are special, uniform clicking will exhaust your whole budget
before it lands on one. Click deliberately instead: on the shapes, on their centres,
on small isolated markers, on anything that looks deliberately placed rather than
empty - and keep track of which colours have ever responded. A colour you have clicked
twice with no effect is not worth a third action. That one habit is what makes a click
budget affordable at all.

## 6. One reply can carry several actions, and the trade is information for time

`action(actions)` takes an ordered batch, and a run ends on a clock rather than on an
action limit: every extra reply you write costs seconds the board never gets. Batching
is the cheapest way to give yourself more actions in the same game.

It is not free, and neither cost is visible from inside the batch:

- **Only the last board comes back.** The intermediate frames are thrown away; all you
  are told is whether something changed somewhere along the way. A batch sent in order
  to see what happens teaches you almost nothing.
- **The batch does not stop when the board stops responding.** It stops if an action
  becomes invalid, if the level completes, or if the game ends - but eight moves into a
  wall are eight billed actions and one unchanged board.

So the rule is about prediction, not about length:

- **Batch when you can already say what each action will do** - walking a route you have
  mapped, repeating a press whose effect you have confirmed, carrying out a plan you
  have already checked. The intermediate frames were predictable, so losing them costs
  you nothing and you save a whole reply per action.
- **Send one action when you are trying to find something out.** Probing is exactly when
  you need the frame after each step, and a batched probe pays for observations it never
  receives.

One protection you can rely on: a batch always stops the moment a level completes, so a
long batch can never spill actions into the next level.
'''


def patch(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise SystemExit(f"{label}: expected exactly 1 match, found {source.count(old)}")
    return source.replace(old, new)


def main() -> int:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    applied = {"bundle_path": 0, "click_prior": 0}

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"])
        before = src

        if OLD_BUNDLE in src:
            src = patch(src, OLD_BUNDLE, NEW_BUNDLE, "bundle_path")
            applied["bundle_path"] += 1

        if ANCHOR in src and "## 5. Clicking: find out what the coordinate" not in src:
            src = patch(src, ANCHOR, ANCHOR + CLICK_PRIOR, "click_prior")
            applied["click_prior"] += 1

        if src != before:
            # Keep the one-string-per-line convention the rest of the file uses, so a
            # diff against v14 shows only the changed lines.
            lines = src.splitlines(keepends=True)
            cell["source"] = lines
            cell["outputs"] = []
            cell["execution_count"] = None

    missing = [k for k, v in applied.items() if v != 1]
    if missing:
        raise SystemExit(f"patches not applied exactly once: {applied}")

    DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    # Verify by reading back rather than trusting the write.
    check = json.loads(DST.read_text(encoding="utf-8"))
    blob = "\n".join("".join(c["source"]) for c in check["cells"] if c.get("cell_type") == "code")
    for needle, label in [
        ("GRAFT_REPO_DIR = \"taaf-grafts\"", "patch A marker"),
        ("with_grafts[0]", "patch A logic"),
        # Needles must not span a line break: CLICK_PRIOR is hard-wrapped, so a phrase
        # that reads as one sentence can contain a newline in the source.
        ("## 5. Clicking: find out what the coordinate", "patch B heading"),
        ("evidence at all that there is nothing to find", "patch B corrected conclusion"),
        ("## 6. One reply can carry several actions", "patch C heading"),
        ("Only the last board comes back", "patch C batch-observation cost"),
        ("does not stop when the board stops responding", "patch C batch-billing cost"),
    ]:
        if needle not in blob:
            raise SystemExit(f"verification failed: {label} missing from {DST.name}")
    if "for marker in Path(\"/kaggle/input\").rglob" in blob:
        raise SystemExit("verification failed: the old first-hit loop is still present")

    print(f"wrote {DST}")
    print(f"  patches applied: {applied}")
    print(f"  cells: {len(check['cells'])}  code cells: {sum(1 for c in check['cells'] if c.get('cell_type') == 'code')}")
    print(f"  click prior: +{len(CLICK_PRIOR)} chars to the agent system prompt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
