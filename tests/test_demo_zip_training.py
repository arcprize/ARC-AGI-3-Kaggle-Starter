from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
import zipfile
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import agi_core.training.demo_zip as demo_zip_module
from agi_core.storage import load_best_prior_bucket, load_memory_bundle
from agi_core.image_input import compute_board_signature
from agi_core.training import (
    RecordingMeta,
    build_action_prior_artifact,
    compute_frame_signature,
    iter_recordings,
    iter_steps,
    normalize_action_id,
    normalize_action_name,
)


def make_line(
    *,
    game_id: str,
    guid: str,
    action_id: int,
    available_actions: list[int],
    state: str = "NOT_FINISHED",
    levels_completed: int = 0,
    frame: list | None = None,
    full_reset: bool = False,
    win_levels: int = 3,
) -> str:
    board = frame or [[[1, 1], [1, 1]]]
    return json.dumps(
        {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "data": {
                "game_id": game_id,
                "frame": board,
                "state": state,
                "action_input": {"id": action_id, "data": {"game_id": game_id}, "reasoning": None},
                "guid": guid,
                "full_reset": full_reset,
                "available_actions": available_actions,
                "levels_completed": levels_completed,
                "win_levels": win_levels,
            },
        }
    )


class DemoZipTrainingTests(unittest.TestCase):
    def _build_zip(self, root: Path) -> Path:
        zip_path = root / "demo.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr(
                "__MACOSX/public_games-dataset/ls20/._junk.recording.jsonl",
                "",
            )
            archive.writestr("public_games-dataset/readme.txt", "skip\n")
            archive.writestr(
                "public_games-dataset/ls20/demo-a.recording.jsonl",
                "\n".join(
                    [
                        make_line(
                            game_id="ls20-9607627b",
                            guid="demo-a",
                            action_id=1,
                            available_actions=[1, 2],
                            state="ACTIVE",
                            frame=[[[1, 2], [3, 4]]],
                        ),
                        make_line(
                            game_id="ls20-9607627b",
                            guid="demo-a",
                            action_id=2,
                            available_actions=[1, 2],
                            state="ACTIVE",
                            frame=[[[1, 2], [3, 5]]],
                        ),
                        make_line(
                            game_id="ls20-9607627b",
                            guid="demo-a",
                            action_id=6,
                            available_actions=[1, 2],
                            state="ACTIVE",
                            frame=[[[1, 2], [3, 5]]],
                        ),
                    ]
                )
                + "\n",
            )
            archive.writestr(
                "public_games-dataset/vc33/demo-b.recording.jsonl",
                make_line(
                    game_id="vc33-5430563c",
                    guid="demo-b",
                    action_id=0,
                    available_actions=[1, 2],
                    state="GAME_OVER",
                    frame=[[[8, 8], [8, 8]]],
                )
                + "\n",
            )
            archive.writestr(
                "public_games-dataset/ls20/demo-c.recording.jsonl",
                make_line(
                    game_id="ls20-9607627b",
                    guid="demo-c",
                    action_id=2,
                    available_actions=[1, 2],
                    state="ACTIVE",
                    frame=[[[1, 2], [3, 5]]],
                )
                + "\n",
            )
        return zip_path

    def test_iter_recordings_and_steps_stream_zip_without_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = self._build_zip(Path(tmpdir))
            recordings = list(iter_recordings(zip_path))
            self.assertEqual(
                [recording.recording_name for recording in recordings],
                [
                    "public_games-dataset/ls20/demo-a.recording.jsonl",
                    "public_games-dataset/ls20/demo-c.recording.jsonl",
                    "public_games-dataset/vc33/demo-b.recording.jsonl",
                ],
            )

            steps = list(iter_steps(zip_path, recordings[0].recording_name))
            self.assertEqual(len(steps), 3)
            self.assertEqual(steps[0].action_name, "ACTION1")
            self.assertEqual(steps[1].available_actions, ["ACTION1", "ACTION2"])
            self.assertEqual(
                steps[0].frame_signature,
                compute_frame_signature([[[1, 2], [3, 4]]]),
            )
            self.assertIsNotNone(steps[0].image_input)
            assert steps[0].image_input is not None
            self.assertEqual(steps[0].image_input.board, [[1, 2], [3, 4]])
            self.assertEqual(steps[0].image_input.source_attr, "frame")
            self.assertEqual(
                steps[0].image_input.frame_signature,
                compute_board_signature([[1, 2], [3, 4]]),
            )

    def test_normalizes_string_action_ids_and_skips_summary_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "public_games-dataset/m0r0/demo.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="m0r0-dadda488",
                                guid="demo",
                                action_id=1,
                                available_actions=[1, 2],
                            ).replace('"id": 1', '"id": "ACTION6"'),
                            json.dumps(
                                {
                                    "timestamp": "2026-01-01T00:00:01+00:00",
                                    "data": {
                                        "won": False,
                                        "played": True,
                                        "total_actions": 10,
                                        "cards": [],
                                        "levels_completed": 1,
                                    },
                                }
                            ),
                        ]
                    )
                    + "\n",
                )

            steps = list(iter_steps(zip_path, "public_games-dataset/m0r0/demo.recording.jsonl"))

            self.assertEqual(len(steps), 1)
            self.assertEqual(steps[0].action_name, "ACTION6")
            self.assertEqual(steps[0].action_id, 6)
            self.assertEqual(normalize_action_name("ACTION3"), "ACTION3")
            self.assertEqual(normalize_action_id("ACTION3"), 3)

    def test_build_action_prior_artifact_aggregates_and_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            zip_path = self._build_zip(root)
            output_path = root / "artifact.sqlite3"

            artifact = build_action_prior_artifact(zip_path, output_path=output_path)

            self.assertEqual(artifact["recordings_seen"], 3)
            self.assertEqual(artifact["steps_seen"], 5)
            self.assertEqual(artifact["steps_used"], 3)
            self.assertTrue(output_path.exists())
            self.assertEqual(artifact["games"]["ls20"]["actions"]["ACTION1"], 1)
            self.assertEqual(artifact["games"]["ls20"]["actions"]["ACTION2"], 1)
            self.assertNotIn("ACTION6", artifact["games"]["ls20"]["actions"])
            self.assertEqual(artifact["games"]["vc33"]["actions"]["RESET"], 2)
            self.assertIn("ACTIVE", artifact["games"]["ls20"]["states"])
            self.assertTrue(artifact["games"]["ls20"]["frame_signatures"])
            bucket = load_best_prior_bucket(
                output_path,
                "ls20",
                "ACTIVE",
                compute_frame_signature([[[1, 2], [3, 5]]]) or "",
                ["ACTION1", "ACTION2"],
            )
            self.assertEqual(bucket, {"ACTION2": 1})
            with closing(sqlite3.connect(output_path)) as connection:
                metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            self.assertEqual(metadata["steps_used"], "3")
            self.assertEqual(metadata["offline_memory_records"], "3")
            memory_bundle = load_memory_bundle(
                output_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature=compute_frame_signature([[[1, 2], [3, 5]]]) or "",
                subgoal=None,
                legal_actions=["ACTION1", "ACTION2"],
            )
            self.assertTrue(memory_bundle["recent_records"])
            self.assertTrue(memory_bundle["semantic_memories"])
            self.assertTrue(memory_bundle["procedural_memories"])
            self.assertTrue(memory_bundle["episodic_recalls"])

    def test_infer_training_subgoal_marks_meaningful_action6_as_inspect_active_board(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "public_games-dataset/ls20/probe.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="probe",
                                action_id=6,
                                available_actions=[1, 6],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=[[[0, 2], [0, 0]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="probe",
                                action_id=1,
                                available_actions=[1, 6],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[2, 2], [2, 2]]],
                            ),
                        ]
                    )
                    + "\n",
                )

            steps = list(iter_steps(zip_path, "public_games-dataset/ls20/probe.recording.jsonl"))
            subgoal = demo_zip_module._infer_training_subgoal(
                step=steps[0],
                next_step=steps[1],
                future_steps=[],
                outcome=demo_zip_module._infer_step_outcome(steps[0], steps[1]),
                stall_streak=0,
            )

            self.assertEqual(subgoal, "inspect-active-board")

    def test_build_action_prior_artifact_persists_action6_board_probe_subgoal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            zip_path = root / "demo.zip"
            output_path = root / "artifact.sqlite3"
            probe_frame = [[[0, 2], [0, 0]]]
            probe_signature = compute_frame_signature(probe_frame) or ""
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "public_games-dataset/ls20/probe.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="probe",
                                action_id=6,
                                available_actions=[1, 6],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=probe_frame,
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="probe",
                                action_id=1,
                                available_actions=[1, 6],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[2, 2], [2, 2]]],
                            ),
                        ]
                    )
                    + "\n",
                )

            build_action_prior_artifact(zip_path, output_path=output_path, workers=1)
            subgoal_bucket = load_best_prior_bucket(
                output_path,
                "ls20",
                "ACTIVE",
                "missing",
                ["ACTION1", "ACTION6"],
                subgoal="inspect-active-board",
            )
            bundle = load_memory_bundle(
                output_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature=probe_signature,
                subgoal="inspect-active-board",
                legal_actions=["ACTION1", "ACTION6"],
                last_action="ACTION6",
            )

            self.assertTrue(bundle["recent_records"])
            self.assertEqual(bundle["recent_records"][0]["action_name"], "ACTION6")
            self.assertEqual(subgoal_bucket, {"ACTION6": 5})
            self.assertEqual(
                bundle["recent_records"][0]["decision_subgoal"],
                "inspect-active-board",
            )
            self.assertEqual(
                bundle["recent_records"][0]["board_context_summary"]["board_default_probe"],
                {"x": 1, "y": 0},
            )
            self.assertEqual(
                bundle["working_memory"]["active_subgoal"],
                "inspect-active-board",
            )

    def test_infer_training_subgoal_uses_hold_win_for_terminal_setup_move(self) -> None:
        step = demo_zip_module.TrainingStep(
            game_id="ls20-9607627b",
            recording_guid="demo",
            step_index=1,
            timestamp="2026-01-01T00:00:00+00:00",
            frame=None,
            available_actions=["ACTION1"],
            action_id=1,
            action_name="ACTION1",
            action_data={},
            levels_completed=0,
            state="ACTIVE",
            win_levels=2,
            full_reset=False,
            frame_signature="sig-a",
            image_input=None,
        )
        win_step = demo_zip_module.TrainingStep(
            game_id="ls20-9607627b",
            recording_guid="demo",
            step_index=2,
            timestamp="2026-01-01T00:00:01+00:00",
            frame=None,
            available_actions=["ACTION1"],
            action_id=1,
            action_name="ACTION1",
            action_data={},
            levels_completed=1,
            state="WIN",
            win_levels=2,
            full_reset=False,
            frame_signature="sig-b",
            image_input=None,
        )

        subgoal = demo_zip_module._infer_training_subgoal(
            step=step,
            next_step=win_step,
            future_steps=[],
            outcome=demo_zip_module._infer_step_outcome(step, win_step),
            stall_streak=0,
        )

        self.assertEqual(subgoal, "hold-win")

    def test_custom_terminal_state_names_drive_demo_zip_outcome_helpers(self) -> None:
        step = demo_zip_module.TrainingStep(
            game_id="ls20-9607627b",
            recording_guid="demo",
            step_index=1,
            timestamp="2026-01-01T00:00:00+00:00",
            frame=None,
            available_actions=["ACTION1"],
            action_id=1,
            action_name="ACTION1",
            action_data={},
            levels_completed=0,
            state="ACTIVE",
            win_levels=2,
            full_reset=False,
            frame_signature="sig-a",
            image_input=None,
        )
        win_step = demo_zip_module.TrainingStep(
            game_id="ls20-9607627b",
            recording_guid="demo",
            step_index=2,
            timestamp="2026-01-01T00:00:01+00:00",
            frame=None,
            available_actions=["ACTION1"],
            action_id=1,
            action_name="ACTION1",
            action_data={},
            levels_completed=0,
            state="DONE",
            win_levels=2,
            full_reset=False,
            frame_signature="sig-b",
            image_input=None,
        )
        lose_step = demo_zip_module.TrainingStep(
            game_id="ls20-9607627b",
            recording_guid="demo",
            step_index=3,
            timestamp="2026-01-01T00:00:02+00:00",
            frame=None,
            available_actions=["RESET"],
            action_id=0,
            action_name="RESET",
            action_data={},
            levels_completed=0,
            state="LOSE",
            win_levels=2,
            full_reset=False,
            frame_signature="sig-c",
            image_input=None,
        )

        env = {
            "AGI_RESET_STATE_NAMES": "READY,LOSE",
            "AGI_WIN_STATE_NAME": "DONE",
        }
        demo_zip_module._terminal_state_config.cache_clear()
        with patch.dict("os.environ", env, clear=False):
            self.assertEqual(
                demo_zip_module._infer_step_outcome(step, win_step),
                "progress",
            )
            self.assertEqual(
                demo_zip_module._infer_step_outcome(step, lose_step),
                "failed",
            )
            self.assertTrue(demo_zip_module._is_meaningful_reset(lose_step))
            self.assertEqual(
                demo_zip_module._infer_training_subgoal(
                    step=lose_step,
                    next_step=win_step,
                    future_steps=[],
                    outcome="failed",
                    stall_streak=0,
                ),
                "re-enter-level",
            )
        demo_zip_module._terminal_state_config.cache_clear()

    def test_build_action_prior_artifact_respects_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = self._build_zip(Path(tmpdir))
            artifact = build_action_prior_artifact(
                zip_path,
                games={"vc33"},
                limit_recordings=1,
                workers=2,
            )

            self.assertEqual(sorted(artifact["games"]), ["vc33"])
            self.assertEqual(artifact["recordings_seen"], 1)

    def test_build_action_prior_artifact_parallel_matches_serial(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            zip_path = self._build_zip(root)
            serial_output_path = root / "serial.sqlite3"
            parallel_output_path = root / "parallel.sqlite3"

            serial_artifact = build_action_prior_artifact(
                zip_path,
                output_path=serial_output_path,
                workers=1,
            )
            parallel_artifact = build_action_prior_artifact(
                zip_path,
                output_path=parallel_output_path,
                workers=2,
            )

            self.assertEqual(serial_artifact, parallel_artifact)
            serial_bucket = load_best_prior_bucket(
                serial_output_path,
                "ls20",
                "ACTIVE",
                compute_frame_signature([[[1, 2], [3, 5]]]) or "",
                ["ACTION1", "ACTION2"],
            )
            parallel_bucket = load_best_prior_bucket(
                parallel_output_path,
                "ls20",
                "ACTIVE",
                compute_frame_signature([[[1, 2], [3, 5]]]) or "",
                ["ACTION1", "ACTION2"],
            )
            self.assertEqual(serial_bucket, parallel_bucket)
            serial_memory_bundle = load_memory_bundle(
                serial_output_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature=compute_frame_signature([[[1, 2], [3, 5]]]) or "",
                subgoal=None,
                legal_actions=["ACTION1", "ACTION2"],
            )
            parallel_memory_bundle = load_memory_bundle(
                parallel_output_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature=compute_frame_signature([[[1, 2], [3, 5]]]) or "",
                subgoal=None,
                legal_actions=["ACTION1", "ACTION2"],
            )
            self.assertEqual(serial_memory_bundle, parallel_memory_bundle)

    def test_parallel_training_merges_multiple_recordings_for_same_game(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = self._build_zip(Path(tmpdir))

            artifact = build_action_prior_artifact(
                zip_path,
                games={"ls20"},
                limit_recordings=2,
                workers=2,
            )

            self.assertEqual(artifact["recordings_seen"], 2)
            self.assertEqual(artifact["steps_seen"], 4)
            self.assertEqual(artifact["steps_used"], 2)
            self.assertEqual(artifact["games"]["ls20"]["recordings"], 2)
            self.assertEqual(artifact["games"]["ls20"]["actions"]["ACTION2"], 1)

    def test_parallel_worker_exceptions_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = self._build_zip(Path(tmpdir))
            selected_recordings = [
                RecordingMeta(
                    recording_name="public_games-dataset/ls20/missing.recording.jsonl",
                    game_id="ls20-9607627b",
                    recording_guid="missing",
                )
            ]

            with self.assertRaises(KeyError):
                demo_zip_module._collect_recording_aggregates(
                    zip_path,
                    selected_recordings=selected_recordings,
                    workers=2,
                )

    def test_training_weights_progress_and_filters_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "public_games-dataset/ls20/demo.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="demo",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=[[[1, 1], [1, 1]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="demo",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[2, 2], [2, 2]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="demo",
                                action_id=0,
                                available_actions=[1, 2],
                                state="GAME_OVER",
                                levels_completed=1,
                                frame=[[[9, 9], [9, 9]]],
                            ),
                        ]
                    )
                    + "\n",
                )

            artifact = build_action_prior_artifact(zip_path, workers=1)

            self.assertEqual(artifact["steps_seen"], 3)
            self.assertEqual(artifact["steps_used"], 2)
            self.assertEqual(artifact["games"]["ls20"]["actions"]["ACTION1"], 3)
            self.assertNotIn("ACTION2", artifact["games"]["ls20"]["actions"])
            self.assertEqual(artifact["games"]["ls20"]["actions"]["RESET"], 3)

    def test_training_ignores_noop_stall_loops(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "demo.zip"
            frame = [[[5, 5], [5, 5]]]
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "public_games-dataset/ls20/demo.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="demo",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=frame,
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="demo",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=frame,
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="demo",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=frame,
                            ),
                        ]
                    )
                    + "\n",
                )

            artifact = build_action_prior_artifact(zip_path, workers=1)

            self.assertEqual(artifact["steps_seen"], 3)
            self.assertEqual(artifact["steps_used"], 0)
            self.assertEqual(artifact["games"]["ls20"]["actions"], {})

    def test_training_downweights_repeated_context_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "demo.zip"
            repeated_frame = [[[7, 7], [7, 7]]]
            progressed_frame = [[[8, 8], [8, 8]]]
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "public_games-dataset/ls20/demo.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="demo",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=repeated_frame,
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="demo",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=progressed_frame,
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="demo",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=progressed_frame,
                            ),
                        ]
                    )
                    + "\n",
                )

            artifact = build_action_prior_artifact(zip_path, workers=1)

            self.assertEqual(artifact["steps_seen"], 3)
            self.assertEqual(artifact["steps_used"], 2)
            self.assertEqual(artifact["games"]["ls20"]["actions"]["ACTION1"], 8)

    def test_post_reset_recovery_gets_extra_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "demo.zip"
            post_reset_frame = [[[1, 0], [0, 1]]]
            normal_frame = [[[2, 0], [0, 2]]]
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "public_games-dataset/ls20/post-reset.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="post-reset",
                                action_id=0,
                                available_actions=[1, 2],
                                state="GAME_OVER",
                                levels_completed=0,
                                frame=[[[9, 9], [9, 9]]],
                                full_reset=True,
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="post-reset",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=post_reset_frame,
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="post-reset",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[3, 3], [3, 3]]],
                            ),
                        ]
                    )
                    + "\n",
                )
                archive.writestr(
                    "public_games-dataset/ls20/normal.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="normal",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=[[[4, 4], [4, 4]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="normal",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=normal_frame,
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="normal",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[5, 5], [5, 5]]],
                            ),
                        ]
                    )
                    + "\n",
                )

            artifact = build_action_prior_artifact(zip_path, workers=1)
            post_reset_signature = compute_frame_signature(post_reset_frame) or ""
            normal_signature = compute_frame_signature(normal_frame) or ""

            self.assertGreater(
                artifact["games"]["ls20"]["frame_signatures"][post_reset_signature]["actions"][
                    "ACTION1"
                ],
                artifact["games"]["ls20"]["frame_signatures"][normal_signature]["actions"][
                    "ACTION1"
                ],
            )

    def test_training_credits_short_horizon_setup_moves(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "public_games-dataset/ls20/demo.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="demo",
                                action_id=3,
                                available_actions=[1, 2, 3],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=[[[1, 0], [0, 1]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="demo",
                                action_id=2,
                                available_actions=[1, 2, 3],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=[[[1, 1], [0, 1]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="demo",
                                action_id=1,
                                available_actions=[1, 2, 3],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[2, 2], [2, 2]]],
                            ),
                        ]
                    )
                    + "\n",
                )

            artifact = build_action_prior_artifact(zip_path, workers=1)

            self.assertEqual(artifact["steps_seen"], 3)
            self.assertEqual(artifact["steps_used"], 3)
            self.assertEqual(artifact["games"]["ls20"]["actions"]["ACTION3"], 3)
            self.assertEqual(artifact["games"]["ls20"]["actions"]["ACTION2"], 5)
            self.assertEqual(artifact["games"]["ls20"]["actions"]["ACTION1"], 1)

    def test_nearer_setup_moves_outweigh_farther_setup_moves(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "public_games-dataset/ls20/demo.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="demo",
                                action_id=3,
                                available_actions=[1, 2, 3],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=[[[1, 0], [0, 1]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="demo",
                                action_id=2,
                                available_actions=[1, 2, 3],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=[[[1, 1], [0, 1]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="demo",
                                action_id=1,
                                available_actions=[1, 2, 3],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=[[[1, 1], [1, 1]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="demo",
                                action_id=1,
                                available_actions=[1, 2, 3],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[2, 2], [2, 2]]],
                            ),
                        ]
                    )
                    + "\n",
                )

            artifact = build_action_prior_artifact(zip_path, workers=1)

            self.assertEqual(artifact["steps_seen"], 4)
            self.assertEqual(artifact["steps_used"], 4)
            self.assertGreater(
                artifact["games"]["ls20"]["actions"]["ACTION1"],
                artifact["games"]["ls20"]["actions"]["ACTION2"],
            )
            self.assertGreater(
                artifact["games"]["ls20"]["actions"]["ACTION2"],
                artifact["games"]["ls20"]["actions"]["ACTION3"],
            )

    def test_stable_progress_gets_extra_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "demo.zip"
            stable_frame = [[[1, 0], [0, 1]]]
            plain_frame = [[[2, 0], [0, 2]]]
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "public_games-dataset/ls20/stable.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="stable",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=stable_frame,
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="stable",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[3, 3], [3, 3]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="stable",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[4, 4], [4, 4]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="stable",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[5, 5], [5, 5]]],
                            ),
                        ]
                    )
                    + "\n",
                )
                archive.writestr(
                    "public_games-dataset/ls20/plain.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="plain",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=plain_frame,
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="plain",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[6, 6], [6, 6]]],
                            ),
                        ]
                    )
                    + "\n",
                )

            artifact = build_action_prior_artifact(zip_path, workers=1)
            stable_signature = compute_frame_signature(stable_frame) or ""
            plain_signature = compute_frame_signature(plain_frame) or ""

            self.assertGreater(
                artifact["games"]["ls20"]["frame_signatures"][stable_signature]["actions"]["ACTION1"],
                artifact["games"]["ls20"]["frame_signatures"][plain_signature]["actions"]["ACTION1"],
            )

    def test_regressed_progress_is_discounted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "demo.zip"
            stable_frame = [[[1, 1], [0, 0]]]
            regressed_frame = [[[2, 2], [0, 0]]]
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "public_games-dataset/ls20/stable.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="stable",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=stable_frame,
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="stable",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[3, 3], [3, 3]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="stable",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[4, 4], [4, 4]]],
                            ),
                        ]
                    )
                    + "\n",
                )
                archive.writestr(
                    "public_games-dataset/ls20/regressed.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="regressed",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=regressed_frame,
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="regressed",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[5, 5], [5, 5]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="regressed",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=[[[6, 6], [6, 6]]],
                            ),
                        ]
                    )
                    + "\n",
                )

            artifact = build_action_prior_artifact(zip_path, workers=1)
            stable_signature = compute_frame_signature(stable_frame) or ""
            regressed_signature = compute_frame_signature(regressed_frame) or ""

            self.assertGreater(
                artifact["games"]["ls20"]["frame_signatures"][stable_signature]["actions"]["ACTION1"],
                artifact["games"]["ls20"]["frame_signatures"][regressed_signature]["actions"]["ACTION1"],
            )

    def test_high_branching_success_gets_extra_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "public_games-dataset/ls20/low-branch.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="low-branch",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=[[[1, 1], [0, 0]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="low-branch",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[2, 2], [2, 2]]],
                            ),
                        ]
                    )
                    + "\n",
                )
                archive.writestr(
                    "public_games-dataset/ls20/high-branch.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="high-branch",
                                action_id=1,
                                available_actions=[1, 2, 3, 4, 5, 6],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=[[[3, 3], [0, 0]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="high-branch",
                                action_id=2,
                                available_actions=[1, 2, 3, 4, 5, 6],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[4, 4], [4, 4]]],
                            ),
                        ]
                    )
                    + "\n",
                )

            artifact = build_action_prior_artifact(zip_path, workers=1)
            low_signature = compute_frame_signature([[[1, 1], [0, 0]]]) or ""
            high_signature = compute_frame_signature([[[3, 3], [0, 0]]]) or ""

            self.assertGreater(
                artifact["games"]["ls20"]["frame_signatures"][high_signature]["actions"]["ACTION1"],
                artifact["games"]["ls20"]["frame_signatures"][low_signature]["actions"]["ACTION1"],
            )

    def test_cross_recording_support_strengthens_db_prior(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            zip_path = root / "demo.zip"
            shared_frame = [[[4, 1], [1, 4]]]
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "public_games-dataset/ls20/demo-a.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="demo-a",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=shared_frame,
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="demo-a",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[8, 8], [8, 8]]],
                            ),
                        ]
                    )
                    + "\n",
                )
                archive.writestr(
                    "public_games-dataset/ls20/demo-b.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="demo-b",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=shared_frame,
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="demo-b",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[9, 9], [9, 9]]],
                            ),
                        ]
                    )
                    + "\n",
                )
            output_path = root / "artifact.sqlite3"

            artifact = build_action_prior_artifact(zip_path, output_path=output_path, workers=1)
            shared_signature = compute_frame_signature(shared_frame) or ""
            raw_count = artifact["games"]["ls20"]["frame_signatures"][shared_signature]["actions"]["ACTION1"]
            bucket = load_best_prior_bucket(
                output_path,
                "ls20",
                "ACTIVE",
                shared_signature,
                ["ACTION1", "ACTION2"],
            )

            self.assertGreater(bucket["ACTION1"], raw_count)

    def test_cross_recording_support_populates_subgoal_priors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            zip_path = root / "demo.zip"
            probe_frame = [[[0, 3], [0, 0]]]
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "public_games-dataset/ls20/probe-a.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="probe-a",
                                action_id=6,
                                available_actions=[1, 6],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=probe_frame,
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="probe-a",
                                action_id=1,
                                available_actions=[1, 6],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[5, 5], [5, 5]]],
                            ),
                        ]
                    )
                    + "\n",
                )
                archive.writestr(
                    "public_games-dataset/ls20/probe-b.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="probe-b",
                                action_id=6,
                                available_actions=[1, 6],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=probe_frame,
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="probe-b",
                                action_id=1,
                                available_actions=[1, 6],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[6, 6], [6, 6]]],
                            ),
                        ]
                    )
                    + "\n",
                )
            output_path = root / "artifact.sqlite3"

            artifact = build_action_prior_artifact(zip_path, output_path=output_path, workers=1)
            raw_count = artifact["games"]["ls20"]["subgoals"]["inspect-active-board"]["actions"]["ACTION6"]
            subgoal_bucket = load_best_prior_bucket(
                output_path,
                "ls20",
                "ACTIVE",
                "missing-frame",
                ["ACTION1", "ACTION6"],
                subgoal="inspect-active-board",
            )

            self.assertIn("inspect-active-board", artifact["games"]["ls20"]["subgoals"])
            self.assertGreater(subgoal_bucket["ACTION6"], raw_count)
            self.assertNotIn("ACTION1", subgoal_bucket)

    def test_conflicting_recordings_do_not_leak_non_consensus_game_priors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            zip_path = root / "demo.zip"
            output_path = root / "artifact.sqlite3"
            shared_frame = [[[1, 0], [0, 1]]]
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "public_games-dataset/ls20/a.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="a",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                frame=shared_frame,
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="a",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[2, 2], [2, 2]]],
                            ),
                        ]
                    )
                    + "\n",
                )
                archive.writestr(
                    "public_games-dataset/ls20/b.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="b",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                frame=shared_frame,
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="b",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[3, 3], [3, 3]]],
                            ),
                        ]
                    )
                    + "\n",
                )

            build_action_prior_artifact(zip_path, output_path=output_path, workers=1)
            bucket = load_best_prior_bucket(
                output_path,
                "ls20",
                "ACTIVE",
                "missing-frame",
                ["ACTION1", "ACTION2"],
            )

            self.assertEqual(bucket, {})

    def test_endgame_progress_gets_extra_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "public_games-dataset/ls20/early.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="early",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                win_levels=4,
                                frame=[[[1, 0], [0, 1]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="early",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                win_levels=4,
                                frame=[[[2, 2], [2, 2]]],
                            ),
                        ]
                    )
                    + "\n",
                )
                archive.writestr(
                    "public_games-dataset/ls20/endgame.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="endgame",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=2,
                                win_levels=4,
                                frame=[[[3, 0], [0, 3]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="endgame",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=3,
                                win_levels=4,
                                frame=[[[4, 4], [4, 4]]],
                            ),
                        ]
                    )
                    + "\n",
                )

            artifact = build_action_prior_artifact(zip_path, workers=1)
            early_signature = compute_frame_signature([[[1, 0], [0, 1]]]) or ""
            endgame_signature = compute_frame_signature([[[3, 0], [0, 3]]]) or ""

            self.assertGreater(
                artifact["games"]["ls20"]["frame_signatures"][endgame_signature]["actions"]["ACTION1"],
                artifact["games"]["ls20"]["frame_signatures"][early_signature]["actions"]["ACTION1"],
            )

    def test_progress_momentum_gets_extra_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "public_games-dataset/ls20/momentum.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="momentum",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=[[[1, 0], [0, 1]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="momentum",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[2, 2], [1, 1]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="momentum",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=2,
                                frame=[[[3, 3], [3, 3]]],
                            ),
                        ]
                    )
                    + "\n",
                )
                archive.writestr(
                    "public_games-dataset/ls20/isolated.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="isolated",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=[[[4, 0], [0, 4]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="isolated",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[5, 5], [1, 1]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="isolated",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[6, 6], [6, 6]]],
                            ),
                        ]
                    )
                    + "\n",
                )

            artifact = build_action_prior_artifact(zip_path, workers=1)
            momentum_signature = compute_frame_signature([[[1, 0], [0, 1]]]) or ""
            isolated_signature = compute_frame_signature([[[4, 0], [0, 4]]]) or ""

            self.assertGreater(
                artifact["games"]["ls20"]["frame_signatures"][momentum_signature]["actions"]["ACTION1"],
                artifact["games"]["ls20"]["frame_signatures"][isolated_signature]["actions"]["ACTION1"],
            )

    def test_pre_win_setup_gets_extra_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "demo.zip"
            pre_win_frame = [[[1, 0], [0, 1]]]
            plain_setup_frame = [[[2, 0], [0, 2]]]
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "public_games-dataset/ls20/pre-win.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="pre-win",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                win_levels=2,
                                frame=pre_win_frame,
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="pre-win",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                win_levels=2,
                                frame=[[[3, 3], [3, 3]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="pre-win",
                                action_id=2,
                                available_actions=[1, 2],
                                state="WIN",
                                levels_completed=1,
                                win_levels=2,
                                frame=[[[4, 4], [4, 4]]],
                            ),
                        ]
                    )
                    + "\n",
                )
                archive.writestr(
                    "public_games-dataset/ls20/plain-setup.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="plain-setup",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                win_levels=3,
                                frame=plain_setup_frame,
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="plain-setup",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                win_levels=3,
                                frame=[[[5, 5], [5, 5]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="plain-setup",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                win_levels=3,
                                frame=[[[6, 6], [6, 6]]],
                            ),
                        ]
                    )
                    + "\n",
                )

            artifact = build_action_prior_artifact(zip_path, workers=1)
            pre_win_signature = compute_frame_signature(pre_win_frame) or ""
            plain_setup_signature = compute_frame_signature(plain_setup_frame) or ""

            self.assertGreater(
                artifact["games"]["ls20"]["frame_signatures"][pre_win_signature]["actions"]["ACTION1"],
                artifact["games"]["ls20"]["frame_signatures"][plain_setup_signature]["actions"]["ACTION1"],
            )

    def test_terminal_win_gets_extra_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "public_games-dataset/ls20/near-win.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="near-win",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                win_levels=3,
                                frame=[[[1, 0], [0, 1]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="near-win",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=2,
                                win_levels=3,
                                frame=[[[2, 2], [2, 2]]],
                            ),
                        ]
                    )
                    + "\n",
                )
                archive.writestr(
                    "public_games-dataset/ls20/win.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="win",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                win_levels=3,
                                frame=[[[3, 0], [0, 3]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="win",
                                action_id=2,
                                available_actions=[1, 2],
                                state="WIN",
                                levels_completed=3,
                                win_levels=3,
                                frame=[[[4, 4], [4, 4]]],
                            ),
                        ]
                    )
                    + "\n",
                )

            artifact = build_action_prior_artifact(zip_path, workers=1)
            near_win_signature = compute_frame_signature([[[1, 0], [0, 1]]]) or ""
            win_signature = compute_frame_signature([[[3, 0], [0, 3]]]) or ""

            self.assertGreater(
                artifact["games"]["ls20"]["frame_signatures"][win_signature]["actions"]["ACTION1"],
                artifact["games"]["ls20"]["frame_signatures"][near_win_signature]["actions"]["ACTION1"],
            )

    def test_breaking_stall_gets_extra_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "public_games-dataset/ls20/immediate-progress.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="immediate-progress",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=[[[1, 0], [0, 1]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="immediate-progress",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[2, 2], [2, 2]]],
                            ),
                        ]
                    )
                    + "\n",
                )
                archive.writestr(
                    "public_games-dataset/ls20/break-stall.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="break-stall",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=[[[3, 0], [0, 3]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="break-stall",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=[[[3, 0], [0, 3]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="break-stall",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=[[[4, 0], [0, 4]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="break-stall",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[5, 5], [5, 5]]],
                            ),
                        ]
                    )
                    + "\n",
                )

            artifact = build_action_prior_artifact(zip_path, workers=1)
            immediate_signature = compute_frame_signature([[[1, 0], [0, 1]]]) or ""
            break_stall_signature = compute_frame_signature([[[4, 0], [0, 4]]]) or ""

            self.assertGreater(
                artifact["games"]["ls20"]["frame_signatures"][break_stall_signature]["actions"]["ACTION1"],
                artifact["games"]["ls20"]["frame_signatures"][immediate_signature]["actions"]["ACTION1"],
            )

    def test_fragile_progress_is_discounted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "public_games-dataset/ls20/stable.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="stable",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=[[[1, 1], [0, 0]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="stable",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[2, 2], [2, 2]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="stable",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[3, 3], [3, 3]]],
                            ),
                        ]
                    )
                    + "\n",
                )
                archive.writestr(
                    "public_games-dataset/ls20/fragile.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="fragile",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=[[[4, 4], [0, 0]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="fragile",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[5, 5], [5, 5]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="fragile",
                                action_id=0,
                                available_actions=[1, 2],
                                state="GAME_OVER",
                                levels_completed=1,
                                frame=[[[9, 9], [9, 9]]],
                            ),
                        ]
                    )
                    + "\n",
                )

            artifact = build_action_prior_artifact(zip_path, workers=1)
            stable_signature = compute_frame_signature([[[1, 1], [0, 0]]]) or ""
            fragile_signature = compute_frame_signature([[[4, 4], [0, 0]]]) or ""

            self.assertGreater(
                artifact["games"]["ls20"]["frame_signatures"][stable_signature]["actions"]["ACTION1"],
                artifact["games"]["ls20"]["frame_signatures"][fragile_signature]["actions"]["ACTION1"],
            )

    def test_higher_quality_recordings_get_bonus_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "public_games-dataset/ls20/low-quality.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="low-quality",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=[[[1, 1], [1, 1]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="low-quality",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[2, 2], [2, 2]]],
                            ),
                        ]
                    )
                    + "\n",
                )
                archive.writestr(
                    "public_games-dataset/ls20/high-quality.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="high-quality",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=[[[3, 3], [3, 3]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="high-quality",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=2,
                                frame=[[[4, 4], [4, 4]]],
                            ),
                        ]
                    )
                    + "\n",
                )

            artifact = build_action_prior_artifact(zip_path, workers=1)

            self.assertEqual(artifact["steps_seen"], 4)
            self.assertEqual(artifact["steps_used"], 4)
            self.assertEqual(artifact["games"]["ls20"]["actions"]["ACTION1"], 13)

    def test_inefficient_recordings_are_discounted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "public_games-dataset/ls20/efficient.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="efficient",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=[[[1, 1], [1, 1]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="efficient",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[2, 2], [2, 2]]],
                            ),
                        ]
                    )
                    + "\n",
                )
                archive.writestr(
                    "public_games-dataset/ls20/inefficient.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="inefficient",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=[[[3, 3], [3, 3]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="inefficient",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=[[[4, 4], [4, 4]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="inefficient",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=[[[5, 5], [5, 5]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="inefficient",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=[[[6, 6], [6, 6]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="inefficient",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=[[[7, 7], [7, 7]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="inefficient",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[8, 8], [8, 8]]],
                            ),
                        ]
                    )
                    + "\n",
                )

            artifact = build_action_prior_artifact(zip_path, workers=1)

            self.assertEqual(artifact["steps_seen"], 8)
            efficient_signature = compute_frame_signature([[[1, 1], [1, 1]]]) or ""
            inefficient_signature = compute_frame_signature([[[3, 3], [3, 3]]]) or ""
            self.assertGreater(
                artifact["games"]["ls20"]["frame_signatures"][efficient_signature]["actions"]["ACTION1"],
                artifact["games"]["ls20"]["frame_signatures"][inefficient_signature]["actions"]["ACTION1"],
            )

    def test_conflicting_frame_context_does_not_fall_back_to_non_consensus_game_prior(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            zip_path = root / "demo.zip"
            shared_frame = [[[4, 4], [4, 4]]]
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "public_games-dataset/ls20/demo-a.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="demo-a",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=shared_frame,
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="demo-a",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[9, 9], [9, 9]]],
                            ),
                        ]
                    )
                    + "\n",
                )
                archive.writestr(
                    "public_games-dataset/ls20/demo-b.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="demo-b",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=shared_frame,
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="demo-b",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[8, 8], [8, 8]]],
                            ),
                        ]
                    )
                    + "\n",
                )
            output_path = root / "artifact.sqlite3"

            artifact = build_action_prior_artifact(zip_path, output_path=output_path, workers=1)

            shared_signature = compute_frame_signature(shared_frame) or ""
            self.assertIn(shared_signature, artifact["games"]["ls20"]["frame_signatures"])
            bucket = load_best_prior_bucket(
                output_path,
                "ls20",
                "ACTIVE",
                shared_signature,
                ["ACTION1", "ACTION2"],
            )
            self.assertEqual(bucket, {})

    def test_state_consensus_keeps_only_clear_leader(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            zip_path = root / "demo.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "public_games-dataset/ls20/demo.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ls20-9607627b",
                                guid="demo",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=[[[1, 0], [0, 1]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="demo",
                                action_id=1,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[2, 2], [2, 2]]],
                            ),
                            make_line(
                                game_id="ls20-9607627b",
                                guid="demo",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=1,
                                frame=[[[3, 3], [3, 3]]],
                            ),
                        ]
                    )
                    + "\n",
                )
            output_path = root / "artifact.sqlite3"

            build_action_prior_artifact(zip_path, output_path=output_path, workers=1)

            bucket = load_best_prior_bucket(
                output_path,
                "ls20",
                "ACTIVE",
                "missing-frame",
                ["ACTION1", "ACTION2"],
            )
            self.assertEqual(bucket, {"ACTION1": 6})

    def test_build_action_prior_artifact_reuses_output_db_when_file_is_already_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            zip_path = self._build_zip(root)
            output_path = root / "artifact.sqlite3"

            build_action_prior_artifact(zip_path, output_path=output_path, workers=1)
            with closing(sqlite3.connect(output_path)) as connection:
                artifact = build_action_prior_artifact(
                    zip_path,
                    output_path=output_path,
                    workers=1,
                )
                self.assertEqual(artifact["recordings_seen"], 3)
                metadata = dict(connection.execute("SELECT key, value FROM metadata"))

            self.assertEqual(metadata["offline_memory_records"], "3")

    def test_build_action_prior_artifact_refuses_live_memory_database_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            zip_path = self._build_zip(root)
            output_path = root / "live-memory.sqlite3"

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
                with self.assertRaisesRegex(ValueError, "refusing to overwrite live memory database"):
                    build_action_prior_artifact(
                        zip_path,
                        output_path=output_path,
                        workers=1,
                    )

    def test_build_action_prior_artifact_builds_shared_abstractions_across_games(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "demo.zip"
            output_path = Path(tmpdir) / "artifact.sqlite3"
            with zipfile.ZipFile(zip_path, "w") as archive:
                for game_name, runtime_game_id in (
                    ("ar25", "ar25-11111111"),
                    ("bp35", "bp35-22222222"),
                ):
                    archive.writestr(
                        f"public_games-dataset/{game_name}/demo.recording.jsonl",
                        "\n".join(
                            [
                                make_line(
                                    game_id=runtime_game_id,
                                    guid=f"{game_name}-demo",
                                    action_id=2,
                                    available_actions=[1, 2, 6],
                                    state="ACTIVE",
                                    levels_completed=0,
                                    frame=[[[1, 0], [0, 1]]],
                                ),
                                make_line(
                                    game_id=runtime_game_id,
                                    guid=f"{game_name}-demo",
                                    action_id=1,
                                    available_actions=[1, 2, 6],
                                    state="ACTIVE",
                                    levels_completed=1,
                                    frame=[[[1, 1], [1, 1]]],
                                ),
                            ]
                        )
                        + "\n",
                    )

            artifact = build_action_prior_artifact(zip_path, output_path=output_path)
            self.assertGreaterEqual(artifact["abstract_memory_rows"], 2)
            with closing(sqlite3.connect(output_path)) as connection:
                row = connection.execute(
                    """
                    SELECT action_name, source_game_count, source_games_json
                    FROM abstract_memories
                    WHERE action_name = 'ACTION2'
                    """
                ).fetchone()
            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(row[0], "ACTION2")
            self.assertEqual(int(row[1] or 0), 2)
            self.assertEqual(sorted(json.loads(row[2])), ["ar25", "bp35"])

    def test_build_action_prior_artifact_skips_non_progress_only_abstractions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "demo.zip"
            output_path = Path(tmpdir) / "artifact.sqlite3"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "public_games-dataset/ar25/stalled.recording.jsonl",
                    "\n".join(
                        [
                            make_line(
                                game_id="ar25-11111111",
                                guid="stalled",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=[[[1, 0], [0, 1]]],
                            ),
                            make_line(
                                game_id="ar25-11111111",
                                guid="stalled",
                                action_id=2,
                                available_actions=[1, 2],
                                state="ACTIVE",
                                levels_completed=0,
                                frame=[[[1, 0], [0, 1]]],
                            ),
                        ]
                    )
                    + "\n",
                )

            artifact = build_action_prior_artifact(zip_path, output_path=output_path)
            self.assertEqual(artifact["abstract_memory_rows"], 0)
            with closing(sqlite3.connect(output_path)) as connection:
                count = connection.execute(
                    "SELECT COUNT(*) FROM abstract_memories"
                ).fetchone()[0]
            self.assertEqual(int(count or 0), 0)

    def test_infer_dynamics_beam_from_high_switch_click(self) -> None:
        self.assertEqual(
            demo_zip_module._infer_dynamics_from_actions({"ACTION5": 20, "ACTION6": 20, "ACTION1": 10}),
            "beam",
        )

    def test_infer_dynamics_signal_from_switch_click(self) -> None:
        self.assertEqual(
            demo_zip_module._infer_dynamics_from_actions({"ACTION5": 15, "ACTION6": 5, "ACTION1": 20}),
            "signal",
        )

    def test_infer_dynamics_gravity_from_high_up_ratio(self) -> None:
        self.assertEqual(
            demo_zip_module._infer_dynamics_from_actions({"ACTION1": 30, "ACTION2": 5, "ACTION3": 10, "ACTION4": 10}),
            "gravity",
        )

    def test_infer_dynamics_navigation_from_balanced_movement(self) -> None:
        self.assertEqual(
            demo_zip_module._infer_dynamics_from_actions({"ACTION1": 1, "ACTION2": 6, "ACTION3": 4, "ACTION4": 3}),
            "navigation",
        )

    def test_infer_dynamics_sliding_from_strong_horizontal(self) -> None:
        self.assertEqual(
            demo_zip_module._infer_dynamics_from_actions({"ACTION1": 1, "ACTION4": 10}),
            "sliding",
        )

    def test_infer_dynamics_static_for_empty(self) -> None:
        self.assertEqual(demo_zip_module._infer_dynamics_from_actions({}), "static")

    def test_infer_board_type_puzzle_from_switch_ratio(self) -> None:
        self.assertEqual(
            demo_zip_module._infer_board_type({"ACTION5": 20, "ACTION1": 10, "ACTION2": 10}),
            "puzzle",
        )

    def test_infer_board_type_puzzle_from_click_ratio(self) -> None:
        self.assertEqual(
            demo_zip_module._infer_board_type({"ACTION6": 30, "ACTION1": 10, "ACTION2": 10}),
            "puzzle",
        )

    def test_infer_board_type_platformer_from_up_ratio(self) -> None:
        self.assertEqual(
            demo_zip_module._infer_board_type({"ACTION1": 30, "ACTION2": 10, "ACTION3": 10, "ACTION4": 10}),
            "platformer",
        )

    def test_extract_board_policy_params_yields_ten_params(self) -> None:
        artifact_games = {
            "ls20": {"actions": {"ACTION1": 1, "ACTION2": 6, "ACTION3": 4, "ACTION4": 3, "ACTION5": 3, "ACTION6": 5}},
            "ar25": {"actions": {"ACTION1": 20, "ACTION2": 5, "ACTION3": 4, "ACTION4": 3}},
        }
        rows = demo_zip_module._extract_board_policy_params(artifact_games)
        board_types = set(r["board_type"] for r in rows)
        self.assertIn("puzzle", board_types)
        param_names = {r["param_name"] for r in rows}
        self.assertIn("explore-bias", param_names)
        self.assertIn("interact-range", param_names)
        self.assertIn("switch-urgency", param_names)
        self.assertIn("path-preference", param_names)
        self.assertIn("move-bias", param_names)
        self.assertIn("beam-priority", param_names)
        self.assertIn("entity-count", param_names)
        self.assertIn("goal-proximity", param_names)
        self.assertIn("retry-tolerance", param_names)
        self.assertIn("stagnation-threshold", param_names)
        self.assertTrue(len(rows) > 0)
        for row in rows:
            self.assertIn("board_type", row)
            self.assertIn("param_name", row)
            self.assertIn("param_value", row)
            self.assertIn("confidence", row)
            self.assertIn("sample_count", row)


if __name__ == "__main__":
    unittest.main()
