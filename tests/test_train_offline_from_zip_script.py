from __future__ import annotations

import io
import json
import runpy
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path


def _make_line(
    *,
    game_id: str,
    guid: str,
    action_id: int,
    available_actions: list[int],
    state: str = "ACTIVE",
    levels_completed: int = 0,
    frame: list | None = None,
) -> str:
    board = frame or [[[1, 1], [1, 1]]]
    return json.dumps(
        {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "data": {
                "game_id": game_id,
                "frame": board,
                "state": state,
                "action_input": {"id": action_id, "data": {"game_id": game_id}},
                "guid": guid,
                "full_reset": False,
                "available_actions": available_actions,
                "levels_completed": levels_completed,
                "win_levels": 3,
            },
        }
    )


class TrainOfflineFromZipScriptTests(unittest.TestCase):
    def test_script_reports_memory_metadata_and_creates_output_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            zip_path = root / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "public_games-dataset/ls20/demo.recording.jsonl",
                    "\n".join(
                        [
                            _make_line(
                                game_id="ls20-9607627b",
                                guid="demo",
                                action_id=1,
                                available_actions=[1, 2],
                                levels_completed=0,
                                frame=[[[1, 0], [0, 1]]],
                            ),
                            _make_line(
                                game_id="ls20-9607627b",
                                guid="demo",
                                action_id=2,
                                available_actions=[1, 2],
                                levels_completed=1,
                                frame=[[[2, 2], [2, 2]]],
                            ),
                        ]
                    )
                    + "\n",
                )
            output_path = root / "nested" / "artifact.sqlite3"
            script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_offline_from_zip.py"
            stdout = io.StringIO()
            argv = [
                str(script_path),
                "--zip",
                str(zip_path),
                "--output",
                str(output_path),
                "--workers",
                "1",
            ]
            previous_argv = sys.argv[:]
            try:
                sys.argv = argv
                with redirect_stdout(stdout):
                    runpy.run_path(str(script_path), run_name="__main__")
            finally:
                sys.argv = previous_argv

            rendered = stdout.getvalue()
            self.assertTrue(output_path.exists())
            self.assertIn("games_seen=1", rendered)
            self.assertIn("offline_memory_records=2", rendered)
            self.assertIn('per_game_steps={"ls20": 2}', rendered)
            self.assertIn('per_game_frames={"ls20": 2}', rendered)
            self.assertIn(f"output_path={output_path.resolve()}", rendered)


if __name__ == "__main__":
    unittest.main()
