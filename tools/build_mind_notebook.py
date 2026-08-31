"""Build v12-derived ARC3 control and mental-simulation Kaggle notebooks.

This intentionally starts from the exact v12 notebook that scored 2.14 on the
private Kaggle submission.  It embeds the local mind stack immediately before
the benchmark run, but leaves the baseline Qwen agent in control until enough
history exists and the sidecar has a high-confidence learned action.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = REPO / "1.33 scored in arc agi 3 competiotn in kaggle"
DEFAULT_SOURCE = NOTEBOOK_DIR / "arc3-duck-v12-with-qwen-3-8-27b this one scored 2.14 .ipynb"
DEFAULT_OUTPUT = NOTEBOOK_DIR / "arc3-duck-v20-v12-sidecar.ipynb"
DEFAULT_SAFETY_OUTPUT = NOTEBOOK_DIR / "arc3-duck-v20-v12-baseline-safety.ipynb"
DEFAULT_MENTAL_OUTPUT = NOTEBOOK_DIR / "arc3-duck-v21-v12-mental-simulation.ipynb"
# Keep this in dependency order: notebook cells run top-to-bottom and the pilot
# imports Progress at module import time.  Omitting it lets the notebook build
# and parse but fails only when Kaggle imports ``arc3x.pilot``.
MODULES = ("percept", "mind", "progress", "dream", "mindgraft", "clicks", "pilot", "autopilot")


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
    }


SIDECAR_ARM = '''# ARC3 conservative level-solver sidecar (v20)
#
# Preserve the v12 Qwen policy's opening.  After it has supplied a meaningful
# transition history, the sidecar may contribute one action only when it can
# reuse a completed-level goal or a high-confidence coordinate-free click rule.
import os
import sys

os.environ["ARC3X_PILOT"] = "1"
os.environ["ARC3X_PILOT_MODE"] = "sidecar"
os.environ["ARC3X_PILOT_MIN_HISTORY"] = "24"
os.environ["ARC3X_PILOT_SIDECAR_ACTIONS"] = "4"
# The original v12 setup imports its TAAF source via a .pth file.  Make the
# freshly-written /kaggle/working/arc3x package explicit as well, independent of
# the notebook's current working directory.
if "/kaggle/working" not in sys.path:
    sys.path.insert(0, "/kaggle/working")
from arc3x.autopilot import arm

if arm():
    print("arc3x v20 sidecar armed; v12 Qwen policy keeps the opening")
else:
    print("arc3x v20 sidecar was not armed; continuing with the v12 baseline")
'''


SIDECAR_NOTE = """## 6c. Conservative ARC3 level-solver sidecar (experimental)

This build is based on the exact v12 private-score baseline. It embeds a
history-only mechanics learner, online click-semantics model, and fail-open
wrapper. The base Qwen agent receives the opening turns; after 24 observed
history entries the sidecar may take one verified route step toward a previously
completed-level goal, or one coordinate-free click at >=90% confidence. It caps
itself at four sidecar actions per level. A Kaggle run is still required to
measure any private-score change.
"""


MENTAL_ARM = '''# ARC3 mental-simulation sidecar (v21)
#
# This is a separately scored experimental candidate. Qwen retains the opening;
# the sidecar acts only after it has learned from real transitions, verified its
# movement predictions on held-out actions, and found a complete imagined route
# that lowers a learned objective. It executes one action, observes, then plans
# again rather than committing an unobserved route.
import os
import sys

os.environ["ARC3X_PILOT"] = "1"
os.environ["ARC3X_PILOT_MODE"] = "mental"
os.environ["ARC3X_PILOT_MIN_HISTORY"] = "24"
os.environ["ARC3X_PILOT_SIDECAR_ACTIONS"] = "4"
if "/kaggle/working" not in sys.path:
    sys.path.insert(0, "/kaggle/working")
from arc3x.autopilot import arm

if arm():
    print("arc3x v21 mental-simulation sidecar armed; Qwen keeps the opening")
else:
    print("arc3x v21 sidecar was not armed; continuing with the v12 baseline")
'''


MENTAL_NOTE = """## 6d. Mental-simulation sidecar (experimental v21)

This separate candidate implements the full observe -> predict -> simulate ->
execute loop. It first watches Qwen's real action/frame transitions and learns
movement, passability, click effects, goals, and UI volatility. It can intervene
only when either its movement copy is accurate on at least eight held-out moves,
or a paint/teleport click model is accurate on four later click effects; a free
rollout must also lower a learned objective. It executes only the first action,
observes the result, and replans. The v20 sidecar and source-equivalent v12
safety notebook remain controls; this notebook requires Kaggle A/B measurement.
"""


def build(
    source_path: Path,
    output_path: Path,
    *,
    include_sidecar: bool = True,
    arm: str = SIDECAR_ARM,
    note: str = SIDECAR_NOTE,
    kind: str = "sidecar",
) -> None:
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    cells = list(notebook.get("cells", []))
    if include_sidecar and any("arc3x human-mind pilot" in source(cell) for cell in cells):
        raise SystemExit(f"human-mind cells already present in {source_path}")

    if include_sidecar:
        # v17 labels the run as a markdown section; the original v12 baseline
        # has a compact ten-cell layout. Both have the final run cell identified
        # by the context manager that invokes ``bm.run``.
        run_index = next(
            (
                i
                for i, cell in enumerate(cells)
                if "## 7. Run the benchmark" in source(cell)
                or "run_context = contextlib.nullcontext" in source(cell)
            ),
            None,
        )
        if run_index is None:
            raise SystemExit("could not locate the benchmark run cell")

        injected = [
            markdown("## 6c. Embedded human-mind modules"),
            # ``%%writefile`` will not create missing parents. The base notebook
            # creates /kaggle/working but does not own an arc3x package directory.
            code("from pathlib import Path\nPath('/kaggle/working/arc3x').mkdir(parents=True, exist_ok=True)"),
        ]
        for name in MODULES:
            body = (REPO / "arc3x" / f"{name}.py").read_text(encoding="utf-8")
            injected.append(code(f"%%writefile /kaggle/working/arc3x/{name}.py\n{body}"))
        injected.extend([markdown(note), code(arm)])
        cells[run_index:run_index] = injected
    # Execution output from the historic v12 run is useful as evidence in the
    # repository, but should not be carried into a new Kaggle submission.  It
    # can mislead readers into treating old offline scores as this notebook's
    # result and needlessly inflates the artifact.
    for cell in cells:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    notebook["cells"] = cells

    if include_sidecar:
        for name in MODULES:
            ast.parse((REPO / "arc3x" / f"{name}.py").read_text(encoding="utf-8"))
        arm_index = next(i for i, cell in enumerate(cells) if source(cell) == arm)
        if not any(
            "## 7. Run the benchmark" in source(cell)
            or "run_context = contextlib.nullcontext" in source(cell)
            for cell in cells[arm_index + 1 :]
        ):
            raise SystemExit("pilot arm cell is not before the benchmark run")

    output_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    label = kind if include_sidecar else "baseline safety"
    print(f"wrote {label}: {output_path} ({len(cells)} cells, {output_path.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--safety-out", type=Path, default=DEFAULT_SAFETY_OUTPUT)
    parser.add_argument("--mental-out", type=Path, default=DEFAULT_MENTAL_OUTPUT)
    parser.add_argument("--no-safety", action="store_true", help="skip the source-equivalent safety notebook")
    parser.add_argument("--no-mental", action="store_true", help="skip the v21 mental-simulation candidate")
    args = parser.parse_args()
    build(args.src, args.out)
    if not args.no_safety:
        build(args.src, args.safety_out, include_sidecar=False)
    if not args.no_mental:
        build(
            args.src,
            args.mental_out,
            arm=MENTAL_ARM,
            note=MENTAL_NOTE,
            kind="mental-simulation sidecar",
        )


if __name__ == "__main__":
    main()
