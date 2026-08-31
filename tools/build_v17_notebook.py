"""Derive v17 from v16 by restoring the model that actually scored 2.14.

TWO BUILD MODES
---------------
    build_v17_notebook.py            -> arc3-duck-v17-winning-model.ipynb
    build_v17_notebook.py --control  -> arc3-duck-v17-control.ipynb

Both restore the 8-27B model that scored 2.14. They differ in what else is live:

* **--control** is the certifiable floor. It changes exactly ONE thing versus the
  run that scored 2.14 -- it restores that run's own model -- and switches every
  behavioural graft OFF (composite install, recovery, the family-priors addendum,
  the efficiency rider, and `context_window: 57344`). Only the inert level probe
  stays on, and it merely appends already-computed per-level fields to a JSONL
  file. It keeps the donor's BYTE-EXACT public config cell, so it attaches the
  same public `jakobbrggen` bundle + wheelhouse 2.14 used: nothing to upload, no
  username to fill in. Behaviourally it should reproduce 2.14; the probe is what
  makes the NEXT change attributable (does 2.14 clear level 0 broadly, or two
  levels narrowly? -- opposite fixes, and nothing else in the repo distinguishes
  them).

* the default (full) build is the SHOT at beating 2.14. It restores the model AND
  leaves v16's grafts live, pointed at our owned grafts dataset. It is NOT
  certifiable as >= 2.14, because those grafts have NEVER run on the leaderboard:
  in v14/v15 they raised ModuleNotFoundError and silently no-opped, so this would
  be the first scored run in which they actually execute -- while the model also
  changes. Two live variables, one number out. `context_window: 57344` is the
  specific hazard: the run is clock-bound, so a larger context is paid in vLLM
  prefill latency, the same mechanism that turned a 2.68 local into 0.60 on Kaggle.

WHAT THIS FIXES
---------------
v13 changed two things at once and lost the score: it added the FAMILY_PRIORS
addendum (removed in v16) AND it swapped the model. The notebook that scored 2.14
serves `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1` (an 8-27B, attached as a
Kaggle *Model*). v13/v14/v15/v16 all serve `driessmit1/vrfai-qwen3-6-27b-fp8-hf-
snapshot` (a different 6-27B, attached as a *dataset*). Because 2.14 itself came
from the model swap, swapping the model away is an independent, at-least-as-large
cause of the regression as the addendum. v16 only fixed the addendum half.

The model is NOT chosen by the notebook. It is hardcoded inside the source
bundle's `setup_commands.json`::

    MODEL_OWNER = 'driessmit1'
    MODEL_SLUG  = 'vrfai-qwen3-6-27b-fp8-hf-snapshot'
    SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'

So restoring the winning model means overriding those three assignments. The 2.14
notebook already solved this WITHOUT forking the bundle: cell `31da6b5c` attaches
the 8-27B Kaggle Model and bridges its path into the bundle's dataset resolver;
cell `eebeea13` defines `_patch_qwen38_setup_commands`, which rewrites those three
assignments in the setup here-doc in memory before it runs, then asserts the live
analyzer really is serving `Qwen/Qwen3.8-27B-FP8`. Its saved outputs prove it ran
green against the `anim-20260807` bundle family -- exactly the family our owned
`taaf-kaggle-source-grafts` dataset is built from.

HOW v17 IS BUILT
----------------
The two 2.14 cells were designed together and are self-consistent, so v17 lifts
BOTH of them wholesale in place of v16's config cell and setup-runner cell, with
exactly two edits to the config cell:

  1. DATASET_SOURCES -> [<grafts-dataset>, wheelhouse]. The 8-27B arrives as a
     Kaggle Model (bridged separately, as in 2.14), so the 6-27B snapshot is
     dropped from the dataset list entirely.
  2. `_find_taaf_bundle` -> the graft-preferring finder from v16, so behavioural
     grafts still import from our owned bundle (the whole point of "keep grafts").

Everything else in v16 is untouched -- crucially the lean addendum in the prompt
cell, which is v16's measured win.

WHAT THIS CANNOT VERIFY
-----------------------
Only a Kaggle GPU run exercises vLLM. This script asserts structure: v17 is valid
JSON, every code cell parses, the 8-27B identity and the patch mechanism are
present, the 6-27B snapshot is gone, the grafts are still preferred, and no
surviving cell references a name this splice removed. The model-attach code itself
is byte-identical to what scored 2.14, so the residual risk is the splice, not the
mechanism -- which is what the structural checks target.

Run:
    .venv/Scripts/python.exe tools/build_v17_notebook.py --control
    .venv/Scripts/python.exe tools/build_v17_notebook.py
    .venv/Scripts/python.exe tools/build_v17_notebook.py --username <kaggle-user>
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NB_DIR = REPO / "1.33 scored in arc agi 3 competiotn in kaggle"
SRC_NB = NB_DIR / "arc3-duck-v16-lean.ipynb"
DONOR_NB = NB_DIR / "arc3-duck-v12-with-qwen-3-8-27b this one scored 2.14 .ipynb"
OUT_NB = NB_DIR / "arc3-duck-v17-winning-model.ipynb"

GRAFT_SLUG = "taaf-kaggle-source-grafts"
WHEELHOUSE = "driessmit1/arc3-vllm-h100-wheelhouse-v3"
DEFAULT_USER = "USERNAME"
OUT_NB_CONTROL = NB_DIR / "arc3-duck-v17-control.ipynb"

# The 6-27B that lost, and the 8-27B that won -- used only for assertions.
LOSING_MODEL = "vrfai-qwen3-6-27b-fp8-hf-snapshot"
WINNING_OWNER = "foysalemonshanto"

# The graft-preferring bundle finder, name-compatible with the donor's
# _find_taaf_bundle so the rest of the donor config cell is untouched.
GRAFT_FINDER = '''def _find_taaf_bundle() -> Path:
    explicit = os.getenv("TAAF_KAGGLE_BUNDLE_DIR", "").strip()
    if explicit:
        path = Path(explicit)
        if (path / DATASET_BUNDLE_MARKER).is_file():
            return path

    # Pick the attached bundle by what it CONTAINS: the one carrying
    # src/taaf-grafts, so the behavioural grafts import instead of falling back
    # to stock. This is the v14 regression's root cause, fixed in v15/v16.
    markers = sorted(Path("/kaggle/input").rglob(DATASET_BUNDLE_MARKER))
    if not markers:
        raise RuntimeError("Could not find TAAF Kaggle source bundle dataset.")
    with_grafts = [m.parent for m in markers if (m.parent / "src" / "taaf-grafts").is_dir()]
    if with_grafts:
        if len(markers) > 1:
            print(
                f"taaf.kaggle: {len(markers)} source bundles attached; chose the one "
                "carrying src/taaf-grafts"
            )
        return with_grafts[0]
    print(
        "taaf.kaggle: *** WARNING *** no attached bundle carries src/taaf-grafts, so "
        "every behavioural graft falls back to stock. Bundles seen: "
        + ", ".join(str(m.parent) for m in markers)
    )
    return markers[0].parent
'''


def cell_src(cell: dict) -> str:
    return "".join(cell["source"])


def find_cell(cells: list[dict], *needles: str) -> int:
    """Index of the single cell whose source contains every needle."""
    hits = [i for i, c in enumerate(cells) if all(n in cell_src(c) for n in needles)]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly 1 cell matching {needles}, found {hits}")
    return hits[0]


def rewrite_dataset_sources(text: str, username: str) -> str:
    """Point DATASET_SOURCES at our grafts bundle + the wheelhouse only."""
    new_list = (
        "DATASET_SOURCES: list[str] = [\n"
        f'    "{username}/{GRAFT_SLUG}",\n'
        f'    "{WHEELHOUSE}",\n'
        "]"
    )
    out, n = re.subn(
        r"DATASET_SOURCES: list\[str\] = \[.*?\]",
        lambda _m: new_list,
        text,
        count=1,
        flags=re.S,
    )
    if n != 1:
        raise SystemExit("could not rewrite DATASET_SOURCES in the donor config cell")
    return out


def replace_bundle_finder(text: str) -> str:
    """Swap the donor's _find_taaf_bundle for the graft-preferring one."""
    out, n = re.subn(
        r"def _find_taaf_bundle\(\) -> Path:.*?(?=\ndef )",
        lambda _m: GRAFT_FINDER,
        text,
        count=1,
        flags=re.S,
    )
    if n != 1:
        raise SystemExit("could not locate _find_taaf_bundle in the donor config cell")
    return out


CONTROL_NOTE_MD = (
    "> **CONTROL RUN.** Grafts **B** (family priors), **C** (recovery) and **D** "
    "(efficiency rider) below are **disabled** in this build, along with the whole "
    "composite graft install. Only graft **A**, the inert level probe, is active. "
    "The single live change versus the run that scored 2.14 is that 2.14's own "
    "model is restored. The sections below are kept for reference; read them as a "
    "description of what is switched off, not of what runs.\n\n"
)

CONTROL_BLOCK = '''# CONTROL RUN (build_v17_notebook.py --control): the composite grafts are
# deliberately NOT installed, and `recovery` is not armed.
#
# WHY. The prose below this cell in earlier versions claimed the three-flag config
# was "byte-exact ... the config that scored 2.14". That claim is false: the
# notebook that actually scored 2.14 contains zero occurrences of
# GRAFT_FLAGS / taaf_grafts / FAMILY_PRIORS / level_probe. 2.14 was pure stock.
#
# It went unnoticed because in v14 and v15 these imports raised
# ModuleNotFoundError and every graft silently no-opped. This build repairs the
# bundle finder, so arming them here would make this the FIRST scored run in which
# they really execute -- while simultaneously changing the model back. Two live
# variables, one number out: exactly the confound that has made eleven previous
# experiments unattributable.
#
# `context_window: 57344` is the specific reason this is not worth the gamble.
# The run is bound by a wall clock, not an action cap (all four games in the last
# local run cut at 7920s), so a larger context is paid for in vLLM prefill latency
# and therefore in actions the board never receives. That is the same mechanism
# that turned a 2.68 local score into 0.60 on Kaggle.
#
# So this run changes ONE thing versus 2.14 -- it restores 2.14's own model -- and
# adds only the inert level probe below, which reads fields the framework has
# already computed. Behaviourally it should reproduce 2.14; the probe is what makes
# the NEXT change attributable, by settling whether 2.14 clears level 0 broadly or
# two levels narrowly. Those two states need opposite fixes and nothing in the repo
# currently distinguishes them.
GRAFT_FLAGS = {"_control_run": "composite grafts + recovery disabled deliberately"}

# Still record what this run configured, next to the other artifacts, so the
# result stays attributable.
try:
    (WORKING_DIR / "graft_flags.json").write_text(json.dumps(GRAFT_FLAGS, indent=2, sort_keys=True) + "\\n")
    print(f"taaf.kaggle: graft flags = {json.dumps(GRAFT_FLAGS, sort_keys=True)}")
except Exception as exc:  # noqa: BLE001
    print(f"[taaf_grafts] could not persist graft flags: {type(exc).__name__}: {exc}")
'''


def apply_control(text: str, cell_type: str) -> str:
    """Disable every behavioural graft, keeping only the inert level probe."""
    if cell_type == "markdown":
        # Keep the notebook self-describing: say plainly that B/C/D are off.
        return CONTROL_NOTE_MD + text if "**B. Family priors" in text else text

    # The composite-graft cell is replaced wholesale rather than patched. Its
    # leading comment block asserts "context_window stays 57344 because 2.14 was
    # measured with it", which is false -- the 2.14 notebook has no graft flags at
    # all -- so excising only the assignment would leave the wrong claim behind.
    if "from taaf_grafts.composite import install" in text:
        return CONTROL_BLOCK

    changed = text
    if "INSTALL_FAMILY_PRIORS" not in changed:
        return changed
    for var in ("INSTALL_FAMILY_PRIORS", "INSTALL_EFFICIENCY_RIDER"):
        changed, k = re.subn(rf"(?m)^{var} = True$", f"{var} = False", changed, count=1)
        if k != 1:
            raise SystemExit(f"could not disable {var}")
    return changed


def parses(text: str) -> bool:
    body = text
    if "await " in body:
        body = "async def _w():\n" + "".join("    " + ln for ln in body.splitlines(True))
    try:
        ast.parse(body)
        return True
    except SyntaxError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--username",
        default=DEFAULT_USER,
        help=f"Kaggle user owning the uploaded {GRAFT_SLUG} dataset (default: {DEFAULT_USER}).",
    )
    ap.add_argument(
        "--control",
        action="store_true",
        help=(
            "Build the controlled variant: restore the 2.14 model as the ONLY live "
            "change, disabling the composite grafts, recovery, the family-priors "
            "addendum and the efficiency rider. Keeps the inert level probe."
        ),
    )
    args = ap.parse_args()

    for nb in (SRC_NB, DONOR_NB):
        if not nb.exists():
            print(f"FAIL  notebook not found: {nb}")
            return 1

    v16 = json.loads(SRC_NB.read_text(encoding="utf-8"))
    donor = json.loads(DONOR_NB.read_text(encoding="utf-8"))
    v16_cells = v16["cells"]
    donor_cells = donor["cells"]

    # The four cells this splice touches.
    v16_cfg = find_cell(v16_cells, "DATASET_SOURCES", "_find_bundle_dir")
    v16_setup = find_cell(v16_cells, "setup_commands.json", "subprocess.run")
    d_cfg = find_cell(donor_cells, "QWEN_MODEL_OWNER", "TAAF_QWEN_SERVED_MODEL_NAME")
    d_setup = find_cell(donor_cells, "_patch_qwen38_setup_commands")

    # Build the new config cell from the donor, graft-aware and re-pointed.
    cfg_text = cell_src(donor_cells[d_cfg])
    if args.control:
        # Control mode imports no graft code at all, so it needs nothing from our
        # own bundle -- only the stock TAAF source (setup_commands.json, the two
        # pickles, preamble/teardown). So leave the donor's config cell BYTE-EXACT:
        # the public `jakobbrggen` bundle that scored 2.14, and the donor's own
        # bundle finder. Nothing to upload, no placeholder username, and no new
        # failure surface between this run and the one it is replicating.
        public_bundle = re.search(r'"([\w-]+/taaf-kaggle-source[\w-]*)"', cfg_text)
        if not public_bundle:
            raise SystemExit("could not find the donor's public source bundle ref")
        control_bundle = public_bundle.group(1)
    else:
        control_bundle = ""
        cfg_text = rewrite_dataset_sources(cfg_text, args.username)
        cfg_text = replace_bundle_finder(cfg_text)

    # The setup runner is lifted verbatim -- it and the config cell were designed
    # together and scored 2.14.
    setup_text = cell_src(donor_cells[d_setup])

    # Names v16's config/setup defined that this splice removes: any surviving cell
    # that still references them would break at runtime.
    removed_names = ("_find_bundle_dir", "GRAFT_REPO_DIR", "_command_env", "_source_path_entries")
    replaced = {v16_cfg, v16_setup}
    for i, c in enumerate(v16_cells):
        if i in replaced or c["cell_type"] != "code":
            continue
        s = cell_src(c)
        for name in removed_names:
            if name in s and name not in cfg_text and name not in setup_text:
                raise SystemExit(
                    f"cell {i} references {name!r}, which the splice removes -- unsafe"
                )

    v16_cells[v16_cfg]["source"] = cfg_text.splitlines(keepends=True)
    v16_cells[v16_setup]["source"] = setup_text.splitlines(keepends=True)

    # In control mode, silence every behavioural graft so the model is the only
    # live delta versus the 2.14 run.
    if args.control:
        for cell in v16_cells:
            s = cell_src(cell)
            changed = apply_control(s, cell["cell_type"])
            if changed != s:
                cell["source"] = changed.splitlines(keepends=True)

    # ---- structural verification ----------------------------------------
    problems: list[str] = []
    full = "\n".join(cell_src(c) for c in v16_cells)

    checks = {
        "8-27B owner present": WINNING_OWNER in full,
        "served name Qwen3.8 present": "Qwen/Qwen3.8-27B-FP8" in full,
        "patch mechanism present": "_patch_qwen38_setup_commands" in full,
        "analyzer-id assertion present": "expected {QWEN_SERVED_MODEL_NAME!r}" in full,
        "lean addendum kept (RESET text gone)": "cheap rewind" not in full,
        "6-27B snapshot removed": LOSING_MODEL not in full,
    }
    if args.control:
        # The model must be the ONLY live delta: every behavioural graft off,
        # the inert probe on, and the same public bundle 2.14 itself used.
        checks.update({
            "control: 2.14's public bundle kept (nothing to upload)": f'"{control_bundle}"' in full,
            "control: placeholder username absent": DEFAULT_USER not in full,
            "control: composite install() not armed": "from taaf_grafts.composite import install" not in full,
            "control: recovery not armed": 'GRAFT_FLAGS["recovery"] = True' not in full,
            "control: context_window 57344 not applied": '"context_window": 57344' not in full,
            "control: family-priors addendum not installed": "INSTALL_FAMILY_PRIORS = False" in full,
            "control: efficiency rider not installed": "INSTALL_EFFICIENCY_RIDER = False" in full,
            "control: inert level probe kept": "INSTALL_LEVEL_PROBE = True" in full,
        })
    else:
        checks["grafts dataset wired"] = f"{args.username}/{GRAFT_SLUG}" in full
        checks["graft preference kept"] = '"src" / "taaf-grafts"' in full or "src/taaf-grafts" in full
    for label, ok in checks.items():
        print(f"[{'x' if ok else ' '}] {label}")
        if not ok:
            problems.append(label)

    bad = [i for i, c in enumerate(v16_cells) if c["cell_type"] == "code" and not parses(cell_src(c))]
    print(f"[{'x' if not bad else ' '}] every code cell parses ({len(v16_cells)} cells, bad={bad})")
    if bad:
        problems.append(f"code cells fail to parse: {bad}")

    if problems:
        for p in problems:
            print(f"FAIL  {p}")
        return 1

    out_nb = OUT_NB_CONTROL if args.control else OUT_NB
    out_nb.write_text(json.dumps(v16, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nspliced donor cfg cell {d_cfg} -> v16 cell {v16_cfg}")
    print(f"spliced donor setup cell {d_setup} -> v16 cell {v16_setup}")
    if args.control:
        print("control mode: composite grafts, recovery, priors and rider all OFF")
        print("              only live delta vs 2.14 is the model + the inert probe")
    print(f"wrote {out_nb.name}")
    if args.control:
        print("\nBEFORE SUBMIT, attach to the notebook on Kaggle:")
        print("  - Model : foysalemonshanto/qwen3-8-27b-fp8-repacked-v1 (PyTorch / hf-fp8 / v1)")
        print(f"  - Data  : {control_bundle}   and   {WHEELHOUSE}")
        print("  Both datasets are PUBLIC and are the ones the 2.14 run used, so")
        print("  nothing has to be uploaded first.")
        return 0
    if args.username == DEFAULT_USER:
        print(
            f"\nNOTE  DATASET_SOURCES[0] is a placeholder ({DEFAULT_USER}/{GRAFT_SLUG}). "
            f"Rerun with --username <your-kaggle-user> once the dataset is uploaded."
        )
    print("\nBEFORE SUBMIT, attach to the notebook on Kaggle:")
    print("  - Model : foysalemonshanto/qwen3-8-27b-fp8-repacked-v1 (PyTorch / hf-fp8 / v1)")
    print(f"  - Data  : {args.username}/{GRAFT_SLUG}   and   {WHEELHOUSE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
