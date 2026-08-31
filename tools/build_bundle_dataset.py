"""Build a self-owned Kaggle source dataset: the upstream bundle + src/taaf-grafts.

WHY THIS EXISTS
---------------
``thtennant/taaf-kaggle-source-share-fork`` went private on Kaggle. The v15 run's
``taaf_setup_env.json`` shows what Kaggle did about that::

    "thtennant/taaf-kaggle-source-share-fork":
        "/kaggle/input/datasets/jakobbrggen/taaf-kaggle-source-anim-20260807-anim"

It silently resolved the missing fork to jakobbrggen's upstream bundle, which has
no ``src/taaf-grafts``. So ``_find_bundle_dir``'s graft-preference (v15 patch A)
worked exactly as written - it found no bundle carrying the graft repo, printed
the warning, and fell back to stock. The bug was never the path. The graft source
is simply not on Kaggle any more, on either run:

    v14 graft_flags.json: ModuleNotFoundError: No module named 'taaf_grafts'
    v15 graft_flags.json: ModuleNotFoundError: No module named 'taaf_grafts'

Depending on someone else's dataset is what made this a silent failure. Owning it
makes the import deterministic.

WHAT GOES IN, AND WHY NOT SIMPLY "THE FORK"
-------------------------------------------
The obvious move - upload my local ``datasets/taaf source share fork (banking)/``
- is wrong. Diffing it against the bundle Sam downloaded today:

    only in fork : 16 files (14 are src/taaf-grafts, 2 are rearc_* helpers)
    only in zip  :  4 files (incl. inference/agent/noop_guard.py)
    both, differing: 28 files (tool_agent.py, solver.py, game_api.py, ...)

The zip is NEWER. Shipping the fork would silently roll back 28 upstream files and
drop ``noop_guard.py`` - the hard no-op guard that ``ARC3_HARD_NOOP_GUARD``
defaults to True and that was live during the v15 run. So the rule here is:

    take the zip verbatim as the base, and overlay ONLY src/taaf-grafts/ from the
    fork.

Nothing else is copied, so upstream stays byte-identical and the only delta is the
directory whose absence broke the run.

VERIFICATION
------------
The check that matters is not "did files copy" but "does the notebook's own
resolver pick this bundle". So ``_find_bundle_dir`` is re-implemented here against
the staged tree and asserted, plus every graft module is compiled and the two
import paths the notebook actually uses (``taaf_grafts.recovery``,
``taaf_grafts.composite.install``) are resolved by name.

Run:
    .venv/Scripts/python.exe tools/build_bundle_dataset.py
    .venv/Scripts/python.exe tools/build_bundle_dataset.py --zip   # also make the .zip
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NB_DIR = REPO / "1.33 scored in arc agi 3 competiotn in kaggle"
SRC_ZIP = NB_DIR / "TAAF Kaggle Source Bundle.zip"
FORK = REPO / "datasets" / "taaf source share fork (banking)"
OUT = REPO / "datasets" / "taaf-kaggle-source-grafts"

GRAFT_REL = "src/taaf-grafts"
BUNDLE_MARKER = "taaf-kaggle-bundle.json"

# Kaggle slug. Must match DATASET_SOURCES[0] in the notebook.
DATASET_SLUG = "taaf-kaggle-source-grafts"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def stage() -> tuple[int, int]:
    """Extract the zip verbatim, then overlay src/taaf-grafts from the fork."""
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    with zipfile.ZipFile(SRC_ZIP) as z:
        members = [i for i in z.infolist() if not i.is_dir()]
        z.extractall(OUT)
    n_base = len(members)

    graft_src = FORK / "src" / "taaf-grafts"
    if not graft_src.is_dir():
        raise SystemExit(f"graft repo not found at {graft_src}")
    graft_dst = OUT / "src" / "taaf-grafts"
    # __pycache__ must not ship: stale bytecode for a different Python would be
    # imported in preference to the source on some layouts.
    shutil.copytree(graft_src, graft_dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    n_graft = sum(1 for p in graft_dst.rglob("*") if p.is_file())

    return n_base, n_graft


def find_bundle_dir(root: Path) -> Path | None:
    """The notebook's own resolver, re-implemented against a staged tree.

    Kept deliberately identical to v15 cell 7 so this asserts the real thing.
    """
    markers = sorted(root.rglob(BUNDLE_MARKER))
    if not markers:
        return None
    with_grafts = [m.parent for m in markers if (m.parent / "src" / "taaf-grafts").is_dir()]
    return with_grafts[0] if with_grafts else None


def verify() -> list[str]:
    problems: list[str] = []

    # 1. upstream must be byte-identical to the zip
    with zipfile.ZipFile(SRC_ZIP) as z:
        drift = []
        for info in z.infolist():
            if info.is_dir():
                continue
            staged = OUT / info.filename
            if not staged.exists():
                drift.append(f"missing {info.filename}")
            elif sha(z.read(info.filename)) != sha(staged.read_bytes()):
                drift.append(f"modified {info.filename}")
    print(f"[1] upstream files preserved byte-for-byte: {not drift}")
    if drift:
        problems.append(f"upstream drifted: {drift[:5]}")

    # 2. the notebook's resolver must select this bundle.
    #
    # Scanned from OUT, not from OUT.parent: on Kaggle only the ATTACHED datasets
    # live under /kaggle/input, and Sam attaches one source bundle. Scanning the
    # whole datasets/ dir instead finds my local fork too, and `sorted()` prefers
    # it purely because a space sorts before a hyphen - which is a real hazard, but
    # a dev-box one. Patch A already prints a warning whenever >1 marker is seen.
    picked = find_bundle_dir(OUT)
    ok = picked is not None and picked.resolve() == OUT.resolve()
    print(f"[2] _find_bundle_dir picks this bundle: {ok}")
    if not ok:
        problems.append(f"_find_bundle_dir picked {picked}, not {OUT}")

    # 3. every graft module compiles
    mods = sorted((OUT / "src" / "taaf-grafts" / "taaf_grafts").glob("*.py"))
    bad = []
    for m in mods:
        try:
            compile(m.read_text(encoding="utf-8"), m.name, "exec")
        except SyntaxError as exc:
            bad.append(f"{m.name}: {exc}")
    print(f"[3] graft modules compiling: {len(mods) - len(bad)}/{len(mods)}")
    if bad:
        problems.append(f"graft modules failed to compile: {bad}")

    # 4. the two import paths the notebook uses must exist by name
    need = {
        "taaf_grafts/recovery.py": ["PROBE_MAX_ACTIONS"],
        "taaf_grafts/composite.py": ["def install"],
    }
    for rel, needles in need.items():
        p = OUT / "src" / "taaf-grafts" / rel
        if not p.exists():
            problems.append(f"notebook imports {rel} but it is not in the bundle")
            continue
        text = p.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                problems.append(f"{rel} lacks {needle!r}")
    print(f"[4] notebook import targets present: {all(( OUT / 'src' / 'taaf-grafts' / r).exists() for r in need)}")

    # 5. no bytecode shipped
    pyc = [p for p in OUT.rglob("*.pyc")] + [p for p in OUT.rglob("__pycache__") if p.is_dir()]
    print(f"[5] bytecode/__pycache__ entries shipped: {len(pyc)}")
    if pyc:
        problems.append(f"bytecode present in bundle: {[str(p) for p in pyc[:3]]}")

    # 6. the marker must be unique inside the bundle, or sorted() picks arbitrarily
    markers = list(OUT.rglob(BUNDLE_MARKER))
    print(f"[6] bundle markers inside the dataset: {len(markers)} (need exactly 1)")
    if len(markers) != 1:
        problems.append(f"expected exactly 1 {BUNDLE_MARKER}, found {len(markers)}")

    return problems


def write_metadata() -> None:
    """dataset-metadata.json is what `kaggle datasets create -p` reads."""
    meta = {
        "title": "TAAF Kaggle Source + Grafts",
        "id": f"USERNAME/{DATASET_SLUG}",
        "licenses": [{"name": "unknown"}],
    }
    (OUT / "dataset-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", action="store_true", help="also write a .zip next to the folder")
    args = ap.parse_args()

    if not SRC_ZIP.exists():
        print(f"FAIL  source bundle not found: {SRC_ZIP}")
        return 1

    n_base, n_graft = stage()
    print(f"staged {n_base} upstream files + {n_graft} graft files -> {OUT}\n")

    problems = verify()
    write_metadata()

    if args.zip:
        archive = OUT.parent / f"{DATASET_SLUG}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(OUT.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(OUT).as_posix())
        mb = archive.stat().st_size / 1e6
        print(f"\nwrote {archive.name} ({mb:.1f} MB)")

    print()
    if problems:
        for p in problems:
            print(f"FAIL  {p}")
        return 1
    total = sum(1 for p in OUT.rglob("*") if p.is_file())
    print(f"OK  {OUT.name} is ready to upload ({total} files)")
    print(f"    Kaggle: create a dataset from this folder, then set DATASET_SOURCES[0]")
    print(f"    to \"<your-username>/{DATASET_SLUG}\" in the notebook.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
