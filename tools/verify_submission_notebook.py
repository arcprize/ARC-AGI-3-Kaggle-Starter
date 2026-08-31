"""Verify the generated notebook before a nine-hour run depends on it.

Five checks, each of which has a plausible way to fail that a human read-through
would miss:

  1. **Every code cell compiles.** The graft cell is ~200 lines of Python embedded in
     a Python string literal inside the generator, which means every backslash and
     every triple-quote in it had to survive two levels of escaping. A syntax error
     here would surface nine hours into a submission. Cell 7 uses top-level ``await``,
     so compilation needs ``PyCF_ALLOW_TOP_LEVEL_AWAIT``.
  2. **Cell order.** The graft has to land after ``GRAFT_FLAGS`` (source importable,
     ``bm`` unpickled) and before ``bm.run`` (nothing played yet). If it landed after
     the run cell it would install into a finished process and appear to work.
  3. **The addendum survived escaping intact.** Executed in isolation and measured,
     because a mangled ``\\n`` would silently become a literal backslash-n in the
     model's system prompt.
  4. **No outputs.** A stale traceback shipped to Kaggle reads as a real failure.
  5. **The bundle loader is contents-based.** Picking the first ``rglob`` hit cost the
     2026-08-23 run every graft, silently, for 2h12m.

Defaults to the v15 notebook. Pass a path to check a different one; v14 is expected
to fail check 5, since it is the notebook that shipped that bug.

Run:
    .venv/Scripts/python.exe tools/verify_submission_notebook.py
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NB_DIR = REPO / "1.33 scored in arc agi 3 competiotn in kaggle"

# Default to the notebook that actually ships. This file was pinned to v14 while v15
# was the upload candidate, so its four checks had never run against the file being
# submitted - which is the one way a verifier can fail silently.
DEFAULT_NB = NB_DIR / "arc3-duck-v15-clickspace.ipynb"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("notebook", nargs="?", default=str(DEFAULT_NB))
    args = ap.parse_args()
    NB = Path(args.notebook)
    if not NB.is_absolute():
        NB = NB_DIR / NB.name if (NB_DIR / NB.name).exists() else NB
    if not NB.exists():
        print(f"FAIL  no such notebook: {NB}")
        return 1
    print(f"verifying {NB.name}\n")
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    code = [(i, "".join(c["source"])) for i, c in enumerate(cells) if c.get("cell_type") == "code"]
    problems: list[str] = []

    # 1. compile everything
    for i, src in code:
        try:
            compile(src, f"cell{i}", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        except SyntaxError as exc:
            problems.append(f"cell {i} does not compile: {exc}")
    print(f"[1] compiled {len(code)} code cells")

    # 2. ordering
    def find(needle: str) -> int | None:
        return next((i for i, src in code if needle in src), None)

    flags_at, graft_at, run_at = find("GRAFT_FLAGS = {"), find("INSTALL_FAMILY_PRIORS"), find("await bm.run(")
    print(f"[2] GRAFT_FLAGS@{flags_at}  graft@{graft_at}  bm.run@{run_at}")
    if None in (flags_at, graft_at, run_at):
        problems.append("could not locate all three landmark cells")
    elif not flags_at < graft_at < run_at:
        problems.append(f"graft cell out of order: {flags_at} < {graft_at} < {run_at} is false")

    # 3. the addendum text, executed in isolation
    graft_src = dict(code)[graft_at] if graft_at is not None else ""
    tree = ast.parse(graft_src)
    addendum = None
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "FAMILY_PRIORS_ADDENDUM"
        ):
            addendum = ast.literal_eval(node.value)
    if not isinstance(addendum, str):
        problems.append("FAMILY_PRIORS_ADDENDUM is not a plain string literal")
    else:
        bad = [r"\n" in addendum, "\\\\" in addendum]
        print(
            f"[3] addendum {len(addendum)} chars, {len(addendum.splitlines())} lines, "
            f"~{len(addendum) // 4} tokens, literal-backslash-n={bad[0]}"
        )
        if any(bad):
            problems.append("addendum contains escaped sequences that should have been resolved")
        # Needles must be written exactly as they appear, markdown emphasis included.
        for want in (
            "*cover* predicate",
            "segmentation hash",
            "squared",
            "costs exactly zero",
            "RESET costs ONE action",
            "up, down, left, right",
            # Section 5, added in v15. The last two are the corrected conclusion: an
            # earlier draft told the agent that agreeing frames mean the coordinate is
            # decoration, which the exhaustive 4096-cell sweep refuted (tn36 has 11
            # distinct outcomes at 96% modal). If these phrases vanish, the wrong
            # advice is back. None may span a line break - the prior is hard-wrapped.
            "what the coordinate is worth",
            "evidence at all that there is nothing to find",
            "must not hunt for them by sampling the board evenly",
            # Section 6, added in v15: the two costs of a batch, both read out of
            # Solver.step_env. The run is clock-bound (all four games in the harvested
            # log were cut at wallclock_s 7920.2), so batching is the lever on actions -
            # but the loop does not break on an unchanged board and only the last frame
            # is returned, and the model cannot see either fact from inside.
            "Only the last board comes back",
            "does not stop when the board stops responding",
        ):
            if want not in addendum:
                problems.append(f"addendum lost the phrase {want!r}")

    # 3c. the recovery arming block: order is load-bearing, so check it explicitly.
    # Arming recovery before killing R2 is the 0.82 configuration from the log.
    flags_src = dict(code)[flags_at] if flags_at is not None else ""
    kill_at = flags_src.find("PROBE_MAX_ACTIONS = 0")
    arm_at = flags_src.find('GRAFT_FLAGS["recovery"] = True')
    install_at = flags_src.find("install(bm, flags=GRAFT_FLAGS)")
    print(f"[3c] recovery: kill R2 @{kill_at}  arm @{arm_at}  install @{install_at}")
    if -1 in (kill_at, arm_at, install_at):
        problems.append("recovery arming block is incomplete")
    elif not kill_at < arm_at < install_at:
        problems.append(f"recovery armed out of order: {kill_at} < {arm_at} < {install_at} is false")
    if "recovery" in flags_src and '"recovery": True' in flags_src.split("try:")[0]:
        problems.append("recovery is armed unconditionally in the literal dict — must be gated on the R2 kill")

    # 3d. the efficiency rider must stay silent when the stock note is empty.
    rider_src = dict(code)[graft_at] if graft_at is not None else ""
    if "if note else note" not in rider_src:
        problems.append("efficiency rider does not preserve the stock empty-note path")

    # the JSONL newline in the probe writer is the one escape most likely to break
    if graft_at is not None:
        writer = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == "\n"]
        print(f"[3b] real newline constants in graft cell: {len(writer)} (need >=1 for the JSONL writer)")
        if not writer:
            problems.append("the JSONL writer's newline did not survive escaping")

    # 4. outputs
    dirty = [i for i, c in enumerate(cells) if c.get("cell_type") == "code" and c.get("outputs")]
    print(f"[4] code cells carrying outputs: {len(dirty)}")
    if dirty:
        problems.append(f"cells still carry outputs: {dirty}")

    # 5. the bundle loader must not pick the first filesystem hit.
    #
    # On 2026-08-23 it did, two attached datasets carried the marker, and the one
    # WITHOUT src/taaf-grafts won. Every graft plus the recovery layer was silently
    # discarded and the run played stock for 2h12m. rglob order is filesystem order,
    # so a rerun could not have reproduced it. v14 fails this check on purpose - it
    # is the notebook that shipped the bug.
    blob = "\n".join(src for _, src in code)
    if 'for marker in Path("/kaggle/input").rglob' in blob:
        problems.append(
            "bundle loader still returns the first rglob hit - this is the bug that cost "
            "the 2026-08-23 run every graft"
        )
    has_pref = "with_grafts" in blob and "sorted(Path(\"/kaggle/input\").rglob" in blob
    print(f"[5] bundle loader prefers the graft-bearing bundle: {has_pref}")
    if not has_pref:
        problems.append("bundle loader does not prefer the bundle containing src/taaf-grafts")

    print()
    if problems:
        for p in problems:
            print(f"FAIL  {p}")
        return 1
    print(f"OK  {NB.name} is ready to upload ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
