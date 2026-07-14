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
from unittest.mock import patch

import agi_core.training.demo_zip as demo_zip_module


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

    def test_script_refuses_live_memory_database_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            zip_path = root / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "public_games-dataset/ls20/demo.recording.jsonl",
                    _make_line(
                        game_id="ls20-9607627b",
                        guid="demo",
                        action_id=1,
                        available_actions=[1, 2],
                    )
                    + "\n",
                )
            output_path = root / "live-memory.sqlite3"
            script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_offline_from_zip.py"
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
                with patch.object(
                    demo_zip_module.RuntimeConfig,
                    "from_env",
                    return_value=demo_zip_module.RuntimeConfig(
                        project_root=root,
                        hyperon_rules_path=root / "arc.metta",
                        hyperon_learning_path=root / "arc_learned.json",
                        memory_db_path=output_path,
                        archive_memory_db_path=root / "archive.sqlite3",
                        trained_priors_db_path=root / "trained.sqlite3",
                        hyperon_wheel_dir=root / "hyperon",
                        extra_wheel_dir=root / "wheels",
                        hyperon_enabled=False,
                        hyperon_skip_probe=True,
                        hyperon_probe_timeout_seconds=0.1,
                        llm_provider="local",
                        llm_model_dir=root / "models",
                        llm_model_path=root / "models" / "executive.gguf",
                        openai_model="gpt-4.1-mini",
                        openai_base_url=None,
                        openai_timeout_seconds=None,
                        llm_max_calls_per_episode=2,
                        llm_max_tokens=192,
                        llm_temperature=0.1,
                        symbolic_confidence_gate=0.7,
                        executive_replan_stagnation_threshold=12,
                        deterministic_fallback_action="ACTION1",
                        reset_state_names=("LOSE", "GAME_OVER"),
                        win_state_name="WIN",
                    ),
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        "refusing to overwrite live memory database",
                    ):
                        runpy.run_path(str(script_path), run_name="__main__")
            finally:
                sys.argv = previous_argv


if __name__ == "__main__":
    unittest.main()
