from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agi_core.training import build_action_prior_artifact, default_worker_count

DEFAULT_ZIP_PATH = ROOT / "arc_agi_3_public_demo_human_testing.zip"
DEFAULT_OUTPUT_PATH = ROOT / "training_artifacts" / "arc_demo_action_priors.sqlite3"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", default=str(DEFAULT_ZIP_PATH))
    parser.add_argument("--output", dest="output_path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument(
        "--games",
        default="",
        help="Optional comma-separated short game ids, e.g. ls20,vc33",
    )
    parser.add_argument(
        "--limit-recordings",
        type=int,
        default=None,
        help="Optional cap on processed recordings for smoke runs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_worker_count(),
        help="Worker process count for per-recording aggregation; use 1 for serial mode.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    selected_games = {
        game_id.strip()
        for game_id in str(args.games or "").split(",")
        if game_id.strip()
    }
    artifact = build_action_prior_artifact(
        zip_path=args.zip_path,
        output_path=args.output_path,
        games=selected_games,
        limit_recordings=args.limit_recordings,
        workers=args.workers,
    )
    print(f"recordings_seen={artifact['recordings_seen']}")
    print(f"steps_seen={artifact['steps_seen']}")
    print(f"steps_used={artifact['steps_used']}")
    per_game = {
        game_id: game_payload.get("steps_used", 0)
        for game_id, game_payload in sorted(artifact.get("games", {}).items())
    }
    print(f"per_game_steps={json.dumps(per_game, sort_keys=True)}")
    print(f"output_path={Path(args.output_path).resolve()}")


if __name__ == "__main__":
    main()
