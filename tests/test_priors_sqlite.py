from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from agi_core.storage import (
    initialize_schema,
    load_best_prior_bucket,
    load_game_summary,
    upsert_metadata,
    upsert_training_counts,
)


class PriorsSqliteTests(unittest.TestCase):
    def test_initialize_schema_creates_tables_and_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "priors.sqlite3"
            initialize_schema(db_path)

            with closing(sqlite3.connect(db_path)) as connection:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                indexes = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='index'"
                    )
                }

            self.assertIn("metadata", tables)
            self.assertIn("game_action_priors", tables)
            self.assertIn("state_action_priors", tables)
            self.assertIn("frame_action_priors", tables)
            self.assertIn("subgoal_action_priors", tables)
            self.assertIn("idx_state_lookup", indexes)
            self.assertIn("idx_frame_lookup", indexes)
            self.assertIn("idx_subgoal_lookup", indexes)

    def test_upsert_and_lookup_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "priors.sqlite3"
            initialize_schema(db_path)
            upsert_training_counts(
                db_path,
                [
                    {"scope": "game", "game_id": "ls20", "action_name": "ACTION2", "count": 3},
                    {
                        "scope": "state",
                        "game_id": "ls20",
                        "state": "ACTIVE",
                        "action_name": "ACTION1",
                        "count": 5,
                    },
                    {
                        "scope": "frame",
                        "game_id": "ls20",
                        "frame_signature": "sig-1",
                        "action_name": "ACTION6",
                        "count": 7,
                    },
                    {
                        "scope": "frame",
                        "game_id": "ls20",
                        "frame_signature": "sig-1",
                        "action_name": "ACTION1",
                        "count": 2,
                    },
                    {
                        "scope": "subgoal",
                        "game_id": "ls20",
                        "subgoal": "inspect-active-board",
                        "action_name": "ACTION6",
                        "count": 6,
                    },
                ],
            )

            frame_bucket = load_best_prior_bucket(
                db_path, "ls20", "ACTIVE", "sig-1", ["ACTION1", "ACTION2", "ACTION6"]
            )
            state_bucket = load_best_prior_bucket(
                db_path, "ls20", "ACTIVE", "missing", ["ACTION1", "ACTION2"]
            )
            game_bucket = load_best_prior_bucket(
                db_path, "ls20", "UNKNOWN", "missing", ["ACTION2"]
            )
            subgoal_bucket = load_best_prior_bucket(
                db_path,
                "ls20",
                "ACTIVE",
                "missing",
                ["ACTION1", "ACTION2", "ACTION6"],
                subgoal="inspect-active-board",
            )

            self.assertEqual(frame_bucket, {"ACTION1": 2, "ACTION6": 7, "ACTION2": 3})
            self.assertEqual(state_bucket, {"ACTION1": 5, "ACTION2": 3})
            self.assertEqual(game_bucket, {"ACTION2": 3})
            self.assertEqual(subgoal_bucket, {"ACTION6": 6, "ACTION1": 5, "ACTION2": 3})

    def test_upsert_accumulates_counts_and_summary_reads_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "priors.sqlite3"
            initialize_schema(db_path)
            upsert_training_counts(
                db_path,
                [
                    {"scope": "game", "game_id": "vc33", "action_name": "RESET", "count": 1},
                    {"scope": "game", "game_id": "vc33", "action_name": "RESET", "count": 2},
                ],
            )
            upsert_metadata(db_path, {"schema_version": "1", "steps_used": "3"})

            summary = load_game_summary(db_path, "vc33")

            self.assertEqual(summary["actions"]["RESET"], 3)
            self.assertEqual(summary["subgoals"], {})

    def test_subgoal_prior_overrides_same_action_frame_prior(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "priors.sqlite3"
            initialize_schema(db_path)
            upsert_training_counts(
                db_path,
                [
                    {
                        "scope": "frame",
                        "game_id": "ls20",
                        "frame_signature": "sig-1",
                        "action_name": "ACTION6",
                        "count": 7,
                    },
                    {
                        "scope": "subgoal",
                        "game_id": "ls20",
                        "subgoal": "inspect-active-board",
                        "action_name": "ACTION6",
                        "count": 11,
                    },
                    {
                        "scope": "game",
                        "game_id": "ls20",
                        "action_name": "ACTION1",
                        "count": 3,
                    },
                ],
            )

            bucket = load_best_prior_bucket(
                db_path,
                "ls20",
                "ACTIVE",
                "sig-1",
                ["ACTION1", "ACTION6"],
                subgoal="inspect-active-board",
            )

            self.assertEqual(bucket, {"ACTION6": 11, "ACTION1": 3})

    def test_full_runtime_game_id_falls_back_to_short_offline_prior_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "priors.sqlite3"
            initialize_schema(db_path)
            upsert_training_counts(
                db_path,
                [
                    {
                        "scope": "state",
                        "game_id": "ls20",
                        "state": "ACTIVE",
                        "action_name": "ACTION1",
                        "count": 5,
                    },
                    {
                        "scope": "game",
                        "game_id": "ls20",
                        "action_name": "ACTION2",
                        "count": 3,
                    },
                ],
            )

            bucket = load_best_prior_bucket(
                db_path,
                "ls20-9607627b",
                "ACTIVE",
                "missing",
                ["ACTION1", "ACTION2"],
            )

            self.assertEqual(bucket, {"ACTION1": 5, "ACTION2": 3})

    def test_missing_db_returns_empty_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "missing.sqlite3"
            bucket = load_best_prior_bucket(
                db_path, "ls20", "ACTIVE", "sig-1", ["ACTION1", "ACTION2"]
            )
            self.assertEqual(bucket, {})


if __name__ == "__main__":
    unittest.main()
