"""Derive the next submission notebook from v15 by removing the two addendum
sections that the run logs measure as harmful or inert.

THE MEASUREMENT THIS ENCODES
----------------------------
Twelve local runs of the same 4-game benchmark exist in this repo. Ten predate the
family-priors addendum (scratch/exp_*); two are v14 and v15. Counting RESET as a
fraction of all actions taken:

    exp_4  0.5%   exp_a  0.7%   exp_f  0.5%   exp_6  0.7%   exp_e  0.5%
    exp_b  0.5%   exp_d  0.5%   exp_5  0.7%   exp_c  0.9%   exp_11 0.6%
    ------------------------------------------------------------------
    v14    6.9%   v15    9.5%

Ten controls inside a 0.5-0.9% band, then a 10-19x jump the moment the addendum
ships. The addendum's section 3 is titled "RESET is a cheap rewind, not a failure"
and tells the model RESET is "the cheapest tool you have"; the v15 transcripts show
it obeying, verbatim: "Let me reset to understand the initial state better",
"Let me reset and explore systematically".

Why that costs score: taaf/game.py:571 bills every action to the level the run was
on when it was taken, and RESET is billed like any other action. RESET does NOT zero
the per-level counter -- it only rewinds the board. So each reset costs one action
AND discards the walk that got there, which then has to be re-paid step by step.
Per-level, from the same logs (actions / human baseline for that level):

    run     game  level    RESETs   actions/baseline   outcome
    exp_11  tn36  L1            0        52/72 = 0.7x  CLEARED
    exp_c   tn36  L1            1        79/72 = 1.1x  CLEARED
    exp_5   tn36  L1            2       178/72 = 2.5x  CLEARED
    v15     tn36  L1           22       217/72 = 3.0x  CLEARED
    v15     m0r0  L0           45       270/30 = 9.0x  CLEARED

Level score is (baseline/actions)^2 and is awarded only on levels that clear, so
the overrun is charged quadratically exactly where it hurts. v15's two cleared
levels are the two it thrashed: tn36 fell from its 10.71 completion cap to 4.89,
and m0r0 from 4.76 to 0.06. On the leaderboard v12-stock scored 2.14 with no
addendum at all; v14 scored 0.69 and v15 0.93 with one.

Section 6 ("One reply can carry several actions") is removed for a different
reason: it produced no measurable behaviour. v15's level_probe.jsonl has turns
exactly equal to actions on all four games (307/307, 310/310, 178/178, 136/136),
i.e. zero batching ever happened. It is 1.6 KB re-sent every turn for no effect,
and the run is bound by a wall clock (all four games cut at 7920s), so prompt bulk
is paid in actions the board never gets.

WHAT IS DELIBERATELY *NOT* CHANGED
----------------------------------
No replacement RESET guidance is added. The ten controls averaged 0.7% RESET with
no reset text in the prompt at all, so the cheapest way back to that behaviour is
silence, not a corrected paragraph -- any text about RESET re-anchors attention on
it, which is how this regression started.

Sections 1 (the cover-predicate goal prior), 2 (where efficiency pays), 4 (control
conventions) and 5 (clicking) are kept intact. Section 5 is the largest, and it
carries the one finding v15 was built to measure -- special cells sit on minority
colours -- which is why it stays despite the token cost.

Run:
    .venv/Scripts/python.exe tools/build_v16_notebook.py
    .venv/Scripts/python.exe tools/build_v16_notebook.py --username <kaggle-user>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NB_DIR = REPO / "1.33 scored in arc agi 3 competiotn in kaggle"
SRC_NB = NB_DIR / "arc3-duck-v15-clickspace.ipynb"
OUT_NB = NB_DIR / "arc3-duck-v16-lean.ipynb"

ADDENDUM_VAR = 'FAMILY_PRIORS_ADDENDUM = """'

# Headings to drop, matched against the "## N. title" lines of the addendum.
DROP_SECTIONS = ("RESET is a cheap rewind", "One reply can carry several actions")
# Headings that must survive, or the edit hit the wrong string.
KEEP_SECTIONS = (
    "The goal is usually already drawn on the first frame",
    "Where being efficient actually pays",
    "Default control conventions",
    "Clicking: find out what the coordinate is worth",
)

DATASET_SLUG = "taaf-kaggle-source-grafts"
OLD_SOURCE_REF = "thtennant/taaf-kaggle-source-share-fork"


def split_sections(addendum: str) -> list[tuple[str, str]]:
    """Split the addendum into (heading_line, body_including_heading) pairs.

    The preamble before the first ``## `` heading is returned with an empty
    heading so it can be reassembled verbatim.
    """
    heads = [m.start() for m in re.finditer(r"^## .*$", addendum, re.M)]
    if not heads:
        raise SystemExit("no '## ' sections found in the addendum")
    out: list[tuple[str, str]] = [("", addendum[: heads[0]])]
    for i, start in enumerate(heads):
        end = heads[i + 1] if i + 1 < len(heads) else len(addendum)
        chunk = addendum[start:end]
        out.append((chunk.splitlines()[0], chunk))
    return out


def renumber(headings_and_bodies: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Rewrite surviving '## N. ' prefixes so the numbering has no gaps."""
    n = 0
    fixed: list[tuple[str, str]] = []
    for head, body in headings_and_bodies:
        if not head:
            fixed.append((head, body))
            continue
        n += 1
        new_head = re.sub(r"^## \d+\.", f"## {n}.", head, count=1)
        fixed.append((new_head, body.replace(head, new_head, 1)))
    return fixed


def rewrite_addendum(src: str) -> tuple[str, dict[str, int]]:
    start = src.index(ADDENDUM_VAR) + len(ADDENDUM_VAR)
    end = src.index('"""', start)
    addendum = src[start:end]

    sections = split_sections(addendum)
    kept, dropped = [], []
    for head, body in sections:
        if head and any(marker in head for marker in DROP_SECTIONS):
            dropped.append((head, body))
        else:
            kept.append((head, body))

    if len(dropped) != len(DROP_SECTIONS):
        raise SystemExit(
            f"expected to drop {len(DROP_SECTIONS)} sections, dropped {len(dropped)}: "
            f"{[h for h, _ in dropped]}"
        )
    surviving = "".join(body for _head, body in renumber(kept))
    for marker in KEEP_SECTIONS:
        if marker not in surviving:
            raise SystemExit(f"section that must survive is missing: {marker!r}")
    for marker in DROP_SECTIONS:
        if marker in surviving:
            raise SystemExit(f"section that must be gone is still present: {marker!r}")

    stats = {
        "before": len(addendum),
        "after": len(surviving),
        "dropped": sum(len(b) for _h, b in dropped),
    }
    return src[:start] + surviving + src[end:], stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--username",
        help=(
            "Kaggle username owning the uploaded "
            f"{DATASET_SLUG} dataset. Omit to leave DATASET_SOURCES untouched."
        ),
    )
    args = ap.parse_args()

    if not SRC_NB.exists():
        print(f"FAIL  source notebook not found: {SRC_NB}")
        return 1

    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))

    edited_addendum = False
    edited_source = False
    stats: dict[str, int] = {}
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        changed = src

        if ADDENDUM_VAR in changed and not edited_addendum:
            changed, stats = rewrite_addendum(changed)
            edited_addendum = True

        if args.username and OLD_SOURCE_REF in changed:
            changed = changed.replace(
                OLD_SOURCE_REF, f"{args.username}/{DATASET_SLUG}"
            )
            edited_source = True

        if changed != src:
            cell["source"] = changed.splitlines(keepends=True)

    if not edited_addendum:
        print("FAIL  no cell defines FAMILY_PRIORS_ADDENDUM")
        return 1

    OUT_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"addendum: {stats['before']} -> {stats['after']} chars "
          f"(-{stats['dropped']}, -{100 * stats['dropped'] / stats['before']:.0f}%)")
    print(f"removed:  {', '.join(DROP_SECTIONS)}")
    print(f"kept:     {len(KEEP_SECTIONS)} sections, renumbered without gaps")
    if edited_source:
        print(f"dataset:  DATASET_SOURCES[0] -> {args.username}/{DATASET_SLUG}")
    else:
        print(
            f"dataset:  DATASET_SOURCES[0] still {OLD_SOURCE_REF} (private/gone) -- "
            f"rerun with --username to point it at your own {DATASET_SLUG}"
        )
    print(f"wrote     {OUT_NB.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
