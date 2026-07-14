"""Splice the current `agent/my_agent.py` into `notebooks/submission.ipynb`.

The notebook follows the exact pattern used by Kaggle's official sample
("ARC3 Sample Submission - Stochastic Goose"):

  Cell 1: install the `arc-agi` wheel from the offline competition dataset.
  Cell 2: write `my_agent.py` to /kaggle/working/ — its body is THIS file.
  Cell 3: if running inside the Kaggle competition rerun, wait for the
          gateway sidecar, copy the framework into /kaggle/working/, register
          MyAgent, and run `python main.py --agent myagent`.
  Cell 4: otherwise (during commit / save-and-run-all), write a dummy
          submission.parquet so Kaggle accepts the commit.

You don't normally need to call this directly — `make submit` runs it for you.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from textwrap import dedent

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE THIS ONE LINE TO PICK YOUR KAGGLE ACCELERATOR
# Options:
#   "cpu"      — no GPU. Good for the random starter or any non-ML agent.
#   "t4"       — Nvidia T4 ×2 (default; matches Kaggle's sample submission).
#   "p100"     — Nvidia P100 (single big-memory GPU).
#   "rtx6000"  — Nvidia RTX 6000 (g4-standard-48). ARC-AGI-3 exclusive,
#                burns GPU quota faster — use only when you're confident.
# ─────────────────────────────────────────────────────────────────────────────
ACCELERATOR = "t4"

# Internal mapping; don't edit unless Kaggle adds new options.
_ACCELERATORS = {
    "cpu":     {"name": "none",            "gpu": False},
    "t4":      {"name": "nvidiaTeslaT4",   "gpu": True},
    "p100":    {"name": "nvidiaTeslaP100", "gpu": True},
    "rtx6000": {"name": "nvidiaRtx6000",   "gpu": True},
}

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "agent" / "my_agent.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "submission.ipynb"
METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"
RUNTIME_BUNDLE_DIR = ROOT / "notebooks" / "runtime_bundle"
RUNTIME_ASSET_DIRS = {
    "wheels": Path("third_party") / "wheels",
    "hyperon": Path("third_party") / "hyperon",
    "models": Path("models"),
}
RUNTIME_ASSET_SUFFIXES = {
    "wheels": {".whl"},
    "hyperon": {".whl"},
    "models": {".gguf"},
}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None,
        "source": source,
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def stage_runtime_bundle(
    root: Path = ROOT,
    bundle_dir: Path = RUNTIME_BUNDLE_DIR,
) -> dict[str, list[str]]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    for child_name in ("agi_core", "wheels", "hyperon", "models"):
        shutil.rmtree(bundle_dir / child_name, ignore_errors=True)
    try:
        (bundle_dir / "manifest.json").unlink()
    except (FileNotFoundError, PermissionError):
        pass

    manifest: dict[str, list[str]] = {
        "agi_core": [],
        "wheels": [],
        "hyperon": [],
        "models": [],
    }

    agi_core_src = root / "agi_core"
    if agi_core_src.exists():
        shutil.copytree(
            agi_core_src,
            bundle_dir / "agi_core",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        manifest["agi_core"].append("agi_core/")

    for label, relative_dir in RUNTIME_ASSET_DIRS.items():
        source_dir = root / relative_dir
        if not source_dir.exists():
            continue
        target_dir = bundle_dir / label
        for source_path in sorted(source_dir.rglob("*")):
            if not source_path.is_file():
                continue
            if source_path.name.endswith(".pyc") or "__pycache__" in source_path.parts:
                continue
            allowed_suffixes = RUNTIME_ASSET_SUFFIXES.get(label)
            if allowed_suffixes and source_path.suffix.lower() not in allowed_suffixes:
                continue
            relative = source_path.relative_to(source_dir)
            destination = target_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
            manifest[label].append(relative.as_posix())

    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def build(agent_src: Path = AGENT_SRC) -> dict:
    if not agent_src.exists():
        raise SystemExit(f"Could not find {agent_src}")
    agent_body = agent_src.read_text()

    install_cell = code_cell(
        "!pip install --no-index --find-links \\\n"
        "    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\\n"
        "    arc-agi python-dotenv"
    )
    install_cell = code_cell(
        dedent(
            """\
            import glob
            import json
            import pathlib
            import subprocess
            import sys

            competition_wheels = (
                "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"
            )
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-index",
                    "--find-links",
                    competition_wheels,
                    "arc-agi",
                    "python-dotenv",
                ]
            )

            manifest_path = pathlib.Path("runtime_bundle/manifest.json")
            manifest = {}
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())

            local_wheels = [
                str(pathlib.Path("runtime_bundle/wheels") / rel)
                for rel in manifest.get("wheels", [])
            ] + [
                str(pathlib.Path("runtime_bundle/hyperon") / rel)
                for rel in manifest.get("hyperon", [])
            ]
            if local_wheels:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "--no-index", *local_wheels]
                )
            else:
                print("No local AGI wheels staged; continuing with optional backends disabled.")
            """
        )
    )

    # We write the agent to /tmp/ (not /kaggle/working/) so it does NOT appear
    # as a notebook output. Otherwise the "Submit to Competition" UI would
    # offer it as a candidate submission file alongside submission.parquet,
    # and an unlucky default selection rejects the submission.
    write_agent_cell = code_cell(
        "%%writefile /tmp/my_agent.py\n" + agent_body
    )

    run_cell_source = dedent(
        """\
        import os

        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            # Wait for the gateway sidecar to be ready.
            !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \\
                  --retry-max-time 600 http://gateway:8001/api/games

            # Copy the framework into a writable location.
            !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents \\
                   /kaggle/working/ARC-AGI-3-Agents

            # Drop our agent in as a framework template.
            !cp /tmp/my_agent.py \\
                /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py

            # Register MyAgent in the framework's agent registry. We rewrite
            # __init__.py because the upstream version eagerly imports
            # templates with deps we don't ship (langgraph, smolagents, etc.).
            with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as f:
                f.write(\"\"\"from typing import Type
        from dotenv import load_dotenv
        from .agent import Agent, Playback
        from .swarm import Swarm
        from .templates.random_agent import Random
        from .templates.my_agent import MyAgent

        load_dotenv()

        AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
            'random': Random,
            'myagent': MyAgent,
        }
        \"\"\")

            # Point the framework at the gateway sidecar.
            with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as f:
                f.write(\"\"\"SCHEME=http
        HOST=gateway
        PORT=8001
        ARC_API_KEY=test-key-123
        ARC_BASE_URL=http://gateway:8001/
        OPERATION_MODE=online
        ENVIRONMENTS_DIR=
        RECORDINGS_DIR=/kaggle/working/server_recording
        \"\"\")

            # Run it. The gateway records every action and emits submission.parquet.
            !cd /kaggle/working/ARC-AGI-3-Agents && \\
                MPLBACKEND=agg \\
                python main.py --agent myagent
        """
    )
    run_cell = code_cell(run_cell_source)
    run_cell_source = dedent(
        """\
        import os
        import json
        import pathlib
        import shutil

        if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
            # Wait for the gateway sidecar to be ready.
            !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \\
                  --retry-max-time 600 http://gateway:8001/api/games

            # Copy the framework into a writable location.
            !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents \\
                   /kaggle/working/ARC-AGI-3-Agents

            framework_root = pathlib.Path("/kaggle/working/ARC-AGI-3-Agents")

            runtime_bundle = pathlib.Path("runtime_bundle")
            manifest_path = runtime_bundle / "manifest.json"
            manifest = {}
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
            agi_core_src = runtime_bundle / "agi_core"
            if agi_core_src.exists():
                agi_core_dst = framework_root / "agi_core"
                if agi_core_dst.exists():
                    shutil.rmtree(agi_core_dst)
                shutil.copytree(agi_core_src, agi_core_dst)

            staged_model_path = ""
            model_src = runtime_bundle / "models"
            if model_src.exists():
                model_dst = framework_root / "runtime_models"
                if model_dst.exists():
                    shutil.rmtree(model_dst)
                model_dst.mkdir(parents=True, exist_ok=True)
                staged_models = []
                for relative in manifest.get("models", []):
                    source_model = model_src / relative
                    if not source_model.exists():
                        continue
                    destination_model = model_dst / pathlib.Path(relative).name
                    shutil.copy2(source_model, destination_model)
                    staged_models.append(destination_model)
                if staged_models:
                    staged_model_path = str(sorted(staged_models)[0])

            shutil.copy2("/tmp/my_agent.py", framework_root / "agents/templates/my_agent.py")

            with open(framework_root / "agents/__init__.py", "w", encoding="utf-8") as f:
                f.write(\"\"\"from typing import Type
        from dotenv import load_dotenv
        from .agent import Agent, Playback
        from .swarm import Swarm
        from .templates.random_agent import Random
        from .templates.my_agent import MyAgent

        load_dotenv()

        AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
            'random': Random,
            'myagent': MyAgent,
        }
        \"\"\")

            env_lines = [
                "SCHEME=http",
                "HOST=gateway",
                "PORT=8001",
                "ARC_API_KEY=test-key-123",
                "ARC_BASE_URL=http://gateway:8001/",
                "OPERATION_MODE=online",
                "ENVIRONMENTS_DIR=",
                "RECORDINGS_DIR=/kaggle/working/server_recording",
                "AGI_HYPERON_RULES_PATH=/kaggle/working/ARC-AGI-3-Agents/agi_core/knowledge/rules/arc.metta",
                "AGI_HYPERON_LEARNING_PATH=/kaggle/working/ARC-AGI-3-Agents/agi_core/knowledge/rules/arc_learned.json",
                "AGI_HYPERON_SKIP_PROBE=1",
            ]
            if staged_model_path:
                env_lines.append(f"AGI_LLM_MODEL_PATH={staged_model_path}")
            with open(framework_root / ".env", "w", encoding="utf-8") as f:
                f.write("\\n".join(env_lines) + "\\n")

            !cd /kaggle/working/ARC-AGI-3-Agents && \\
                MPLBACKEND=agg \\
                python main.py --agent myagent
        """
    )
    run_cell = code_cell(run_cell_source)

    dummy_submission_cell = code_cell(
        dedent(
            """\
            import os
            if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
                # Save-and-run-all (commit) mode: emit a dummy submission so the
                # commit succeeds. The real submission.parquet is produced by the
                # gateway during competition rerun.
                import pandas as pd
                submission = pd.DataFrame(
                    data=[['1_0', '1', True, 1]],
                    columns=['row_id', 'game_id', 'end_of_game', 'score'])
                submission.to_parquet('/kaggle/working/submission.parquet', index=False)
                submission.head()
            """
        )
    )

    if ACCELERATOR not in _ACCELERATORS:
        raise SystemExit(
            f"Unknown ACCELERATOR={ACCELERATOR!r}. Pick one of: "
            f"{sorted(_ACCELERATORS)}"
        )
    accel = _ACCELERATORS[ACCELERATOR]

    notebook = {
        "metadata": {
            "kernelspec": {
                "language": "python",
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
            },
            "kaggle": {
                "accelerator": accel["name"],
                "isInternetEnabled": False,
                "isGpuEnabled": accel["gpu"],
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            markdown_cell(
                "# ARC Prize 2026 — ARC-AGI-3 Submission\n\n"
                "Built from `agent/my_agent.py` via `scripts/build_notebook.py`. "
                "Do not edit cells directly — edit the source file and re-run "
                "`make submit`."
            ),
            install_cell,
            write_agent_cell,
            run_cell,
            dummy_submission_cell,
        ],
    }
    return notebook


def main() -> None:
    manifest = stage_runtime_bundle()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1))
    staged_counts = ", ".join(f"{key}={len(value)}" for key, value in manifest.items())
    print(f"[build_notebook] Wrote {NOTEBOOK_PATH.relative_to(ROOT)}  "
          f"(accelerator: {ACCELERATOR}; staged: {staged_counts})")

    # Keep notebooks/kernel-metadata.json in sync so the user never has to
    # edit it just to flip CPU ↔ GPU.
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text())
        wanted = _ACCELERATORS[ACCELERATOR]["gpu"]
        if meta.get("enable_gpu") != wanted:
            meta["enable_gpu"] = wanted
            METADATA_PATH.write_text(json.dumps(meta, indent=2) + "\n")
            print(f"[build_notebook] Synced enable_gpu={wanted} in "
                  f"{METADATA_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
