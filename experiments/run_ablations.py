"""Run reproducible HypothesisWorld ablations through the official local harness."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", default="ls20,vc33")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--ablations", default="A,B,C,D,E,F")
    parser.add_argument(
        "--seeds",
        default="0",
        help="Comma-separated integer seeds (default: 0)",
    )
    args = parser.parse_args()

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = ROOT / "experiments" / "results" / stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    matplotlib_dir = output_dir / ".matplotlib"
    matplotlib_dir.mkdir()
    summary: list[dict[str, object]] = []
    ablations = [item.strip().upper() for item in args.ablations.split(",")]
    seeds = [int(item.strip()) for item in args.seeds.split(",")]
    for seed in seeds:
        for ablation in ablations:
            environment = os.environ.copy()
            environment["HYPOTHESISWORLD_ABLATION"] = ablation
            environment["HYPOTHESISWORLD_SEED"] = str(seed)
            environment["MPLCONFIGDIR"] = str(matplotlib_dir)
            command = [
                str(ROOT / ".venv" / "bin" / "python"),
                str(ROOT / "scripts" / "play_local.py"),
                "--game",
                args.games,
                "--max-steps",
                str(args.steps),
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            (output_dir / f"ablation-{ablation}-seed-{seed}.log").write_text(
                completed.stdout
            )
            summary.append(
                {
                    "ablation": ablation,
                    "seed": seed,
                    "returncode": completed.returncode,
                }
            )
            print(
                f"{ablation} seed={seed}: returncode={completed.returncode}",
                flush=True,
            )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "created_at": stamp,
                "games": args.games,
                "steps": args.steps,
                "seeds": seeds,
                "python": sys.version,
                "runs": summary,
            },
            indent=2,
        )
        + "\n"
    )
    if any(run["returncode"] != 0 for run in summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
