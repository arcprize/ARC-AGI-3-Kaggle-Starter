from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agi_core.training import build_action_prior_artifact, default_worker_count

DEFAULT_ZIP_PATH = ROOT / "arc_agi_3_public_demo_human_testing.zip"
DEFAULT_OUTPUT_PATH = ROOT / "training_artifacts" / "arc_demo_action_priors.sqlite3"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", default=str(DEFAULT_ZIP_PATH))
    parser.add_argument(
        "--output",
        dest="output_path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Destination SQLite database for offline priors and episodic memory.",
    )
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


def load_training_metadata(db_path: str | Path) -> dict[str, str]:
    path = Path(db_path)
    if not path.exists():
        return {}
    with closing(sqlite3.connect(path)) as connection:
        rows = connection.execute("SELECT key, value FROM metadata ORDER BY key").fetchall()
    return {str(key): str(value) for key, value in rows}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_games = {
        game_id.strip()
        for game_id in str(args.games or "").split(",")
        if game_id.strip()
    }
    artifact = build_action_prior_artifact(
        zip_path=args.zip_path,
        output_path=output_path,
        games=selected_games,
        limit_recordings=args.limit_recordings,
        workers=args.workers,
    )
    metadata = load_training_metadata(output_path)
    print(f"recordings_seen={artifact['recordings_seen']}")
    print(f"steps_seen={artifact['steps_seen']}")
    print(f"steps_used={artifact['steps_used']}")
    print(f"games_seen={len(artifact.get('games', {}))}")
    print(f"offline_memory_records={metadata.get('offline_memory_records', '0')}")
    per_game = {
        game_id: game_payload.get("steps_used", 0)
        for game_id, game_payload in sorted(artifact.get("games", {}).items())
    }
    per_game_frames = {
        game_id: len((game_payload.get("frame_signatures", {}) or {}))
        for game_id, game_payload in sorted(artifact.get("games", {}).items())
    }
    print(f"per_game_steps={json.dumps(per_game, sort_keys=True)}")
    print(f"per_game_frames={json.dumps(per_game_frames, sort_keys=True)}")
    print(f"cognition_traces={artifact.get('cognition_traces_count', 0)}")
    print(f"chunks_extracted={artifact.get('chunk_count', 0)}")
    print(f"output_path={output_path}")


if __name__ == "__main__":
    main()
