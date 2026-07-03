from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from agi_core.storage import (
    import_legacy_learning_json,
    initialize_memory_schema,
    load_basin_status,
    load_memory_bundle,
    upsert_training_counts,
)
from agi_core.storage.memory_sqlite import record_transition
from agi_core.types import EpisodeRecord


class MemorySqliteTests(unittest.TestCase):
    def test_initialize_memory_schema_creates_live_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            initialize_memory_schema(db_path)

            with closing(sqlite3.connect(db_path)) as connection:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    ).fetchall()
                }
            self.assertIn("live_episode_records", tables)
            self.assertIn("live_action_stats", tables)
            self.assertIn("live_basin_stats", tables)

    def test_record_transition_updates_bundle_and_basin_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            record_transition(
                db_path,
                EpisodeRecord(
                    game_id="ls20",
                    step_index=1,
                    action_name="ACTION2",
                    outcome="progress",
                    progress_delta=1,
                    game_state="ACTIVE",
                    prior_game_state="ACTIVE",
                    levels_completed=1,
                    prior_frame_signature="sig-1",
                    next_game_state="ACTIVE",
                    next_frame_signature="sig-2",
                    prior_legal_actions=["ACTION1", "ACTION2"],
                    stagnation_before=0,
                    stagnation_after=0,
                    decision_subgoal="advance-level",
                    decision_source="symbolic",
                ),
            )

            bundle = load_memory_bundle(
                db_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature="sig-1",
                subgoal="advance-level",
                legal_actions=["ACTION1", "ACTION2"],
            )
            self.assertEqual(bundle["action_stats"]["ACTION2"]["progress"], 1)
            self.assertEqual(bundle["frame_action_stats"]["ACTION2"]["progress"], 1)
            self.assertEqual(bundle["subgoal_action_stats"]["ACTION2"]["progress"], 1)
            basin = load_basin_status(db_path, "ls20", "ACTIVE", "sig-1")
            self.assertEqual(basin["progress_events"], 1)
            self.assertFalse(basin["is_exhausted"])

    def test_load_memory_bundle_merges_trained_priors_and_live_suppression(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            initialize_memory_schema(db_path)
            upsert_training_counts(
                db_path,
                [
                    {
                        "scope": "frame",
                        "game_id": "ls20",
                        "frame_signature": "sig-loop",
                        "action_name": "ACTION1",
                        "count": 8,
                    },
                    {
                        "scope": "frame",
                        "game_id": "ls20",
                        "frame_signature": "sig-loop",
                        "action_name": "ACTION2",
                        "count": 1,
                    },
                ],
            )
            for step_index in range(1, 4):
                record_transition(
                    db_path,
                    EpisodeRecord(
                        game_id="ls20",
                        step_index=step_index,
                        action_name="ACTION1",
                        outcome="stalled",
                        progress_delta=0,
                        game_state="NOT_FINISHED",
                        prior_game_state="NOT_FINISHED",
                        levels_completed=0,
                        prior_frame_signature="sig-loop",
                        next_game_state="NOT_FINISHED",
                        next_frame_signature="sig-loop",
                        prior_legal_actions=["ACTION1", "ACTION2"],
                        stagnation_before=step_index - 1,
                        stagnation_after=step_index,
                        decision_subgoal="break-stall",
                        decision_source="symbolic",
                    ),
                )

            bundle = load_memory_bundle(
                db_path,
                game_id="ls20",
                state="NOT_FINISHED",
                frame_signature="sig-loop",
                subgoal="break-stall",
                legal_actions=["ACTION1", "ACTION2"],
            )
            self.assertEqual(bundle["trained_prior_bucket"]["ACTION1"], 8)
            self.assertIn("ACTION1", bundle["suppressed_actions"])
            self.assertNotIn("ACTION2", bundle["suppressed_actions"])

    def test_import_legacy_learning_json_seeds_live_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            legacy_path = Path(tmpdir) / "arc_learned.json"
            legacy_path.write_text(
                json.dumps(
                    {
                        "games": {
                            "ls20": {
                                "actions": {
                                    "ACTION2": {
                                        "progress": 2,
                                        "failed": 1,
                                        "stalled": 0,
                                        "total": 3,
                                        "last_outcome": "progress",
                                    }
                                },
                                "states": {
                                    "ACTIVE": {
                                        "actions": {
                                            "ACTION2": {
                                                "progress": 1,
                                                "failed": 0,
                                                "stalled": 0,
                                                "total": 1,
                                                "last_outcome": "progress",
                                            }
                                        }
                                    }
                                },
                                "subgoals": {},
                                "frame_signatures": {},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            import_legacy_learning_json(db_path, legacy_path)
            bundle = load_memory_bundle(
                db_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature="",
                subgoal=None,
                legal_actions=["ACTION1", "ACTION2"],
            )
            self.assertEqual(bundle["action_stats"]["ACTION2"]["progress"], 2)
            self.assertEqual(bundle["state_action_stats"]["ACTION2"]["progress"], 1)

    def test_import_legacy_learning_json_backfills_missing_tables_without_duplication(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            legacy_path = Path(tmpdir) / "arc_learned.json"
            initialize_memory_schema(db_path)
            with closing(sqlite3.connect(db_path)) as connection:
                connection.execute(
                    """
                    INSERT INTO live_action_stats(
                        game_id,
                        action_name,
                        progress,
                        failed,
                        stalled,
                        total,
                        last_outcome
                    ) VALUES(?, ?, ?, ?, ?, ?, ?)
                    """,
                    ["ls20", "ACTION2", 2, 1, 0, 3, "progress"],
                )
                connection.commit()
            legacy_path.write_text(
                json.dumps(
                    {
                        "games": {
                            "ls20": {
                                "actions": {
                                    "ACTION2": {
                                        "progress": 2,
                                        "failed": 1,
                                        "stalled": 0,
                                        "total": 3,
                                        "last_outcome": "progress",
                                    }
                                },
                                "states": {
                                    "ACTIVE": {
                                        "actions": {
                                            "ACTION2": {
                                                "progress": 1,
                                                "failed": 0,
                                                "stalled": 0,
                                                "total": 1,
                                                "last_outcome": "progress",
                                            }
                                        }
                                    }
                                },
                                "subgoals": {
                                    "advance-level": {
                                        "actions": {
                                            "ACTION2": {
                                                "progress": 1,
                                                "failed": 0,
                                                "stalled": 0,
                                                "total": 1,
                                                "last_outcome": "progress",
                                            }
                                        }
                                    }
                                },
                                "frame_signatures": {
                                    "sig-legacy": {
                                        "actions": {
                                            "ACTION2": {
                                                "progress": 1,
                                                "failed": 0,
                                                "stalled": 0,
                                                "total": 1,
                                                "last_outcome": "progress",
                                            }
                                        }
                                    }
                                },
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            import_legacy_learning_json(db_path, legacy_path)
            bundle = load_memory_bundle(
                db_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature="sig-legacy",
                subgoal="advance-level",
                legal_actions=["ACTION1", "ACTION2"],
            )

            self.assertEqual(bundle["action_stats"]["ACTION2"]["progress"], 2)
            self.assertEqual(bundle["action_stats"]["ACTION2"]["total"], 3)
            self.assertEqual(bundle["state_action_stats"]["ACTION2"]["progress"], 1)
            self.assertEqual(bundle["frame_action_stats"]["ACTION2"]["progress"], 1)
            self.assertEqual(bundle["subgoal_action_stats"]["ACTION2"]["progress"], 1)

    def test_import_legacy_learning_json_seeds_state_only_basin_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            legacy_path = Path(tmpdir) / "arc_learned.json"
            legacy_path.write_text(
                json.dumps(
                    {
                        "games": {
                            "ls20": {
                                "actions": {},
                                "states": {
                                    "ACTIVE": {
                                        "actions": {
                                            "ACTION1": {
                                                "progress": 0,
                                                "failed": 1,
                                                "stalled": 1,
                                                "total": 2,
                                                "last_outcome": "stalled",
                                            }
                                        }
                                    }
                                },
                                "subgoals": {},
                                "frame_signatures": {},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            import_legacy_learning_json(db_path, legacy_path)
            basin = load_basin_status(db_path, "ls20", "ACTIVE", "sig-missing")
            bundle = load_memory_bundle(
                db_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature="sig-missing",
                subgoal=None,
                legal_actions=["ACTION1", "ACTION2"],
            )

            self.assertEqual(basin["frame_signature"], "")
            self.assertEqual(basin["failed_events"], 1)
            self.assertEqual(basin["stalled_events"], 1)
            self.assertTrue(basin["is_exhausted"])
            self.assertEqual(basin["action_outcomes"]["ACTION1"]["failed"], 1)
            self.assertIn("ACTION1", bundle["suppressed_actions"])

    def test_load_memory_bundle_recommends_trained_prior_when_live_memory_is_sparse(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            initialize_memory_schema(db_path)
            upsert_training_counts(
                db_path,
                [
                    {
                        "scope": "state",
                        "game_id": "ls20",
                        "state": "NOT_FINISHED",
                        "action_name": "ACTION2",
                        "count": 12,
                    },
                    {
                        "scope": "state",
                        "game_id": "ls20",
                        "state": "NOT_FINISHED",
                        "action_name": "ACTION1",
                        "count": 2,
                    },
                ],
            )

            bundle = load_memory_bundle(
                db_path,
                game_id="ls20",
                state="NOT_FINISHED",
                frame_signature="sig-prior",
                subgoal="break-stall",
                legal_actions=["ACTION1", "ACTION2"],
            )

            self.assertEqual(bundle["trained_prior_bucket"]["ACTION2"], 12)
            self.assertEqual(bundle["recommended_actions"][0], "ACTION2")

    def test_load_memory_bundle_avoids_recent_stalled_repeat_in_recommendations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            initialize_memory_schema(db_path)
            record_transition(
                db_path,
                EpisodeRecord(
                    game_id="ls20",
                    step_index=1,
                    action_name="ACTION1",
                    outcome="stalled",
                    progress_delta=0,
                    game_state="NOT_FINISHED",
                    prior_game_state="NOT_FINISHED",
                    levels_completed=0,
                    prior_frame_signature="sig-recent",
                    next_game_state="NOT_FINISHED",
                    next_frame_signature="sig-recent",
                    prior_legal_actions=["ACTION1", "ACTION2"],
                    stagnation_before=0,
                    stagnation_after=1,
                    decision_subgoal="break-stall",
                    decision_source="symbolic",
                ),
            )
            upsert_training_counts(
                db_path,
                [
                    {
                        "scope": "state",
                        "game_id": "ls20",
                        "state": "NOT_FINISHED",
                        "action_name": "ACTION1",
                        "count": 10,
                    },
                    {
                        "scope": "state",
                        "game_id": "ls20",
                        "state": "NOT_FINISHED",
                        "action_name": "ACTION2",
                        "count": 8,
                    },
                ],
            )

            bundle = load_memory_bundle(
                db_path,
                game_id="ls20",
                state="NOT_FINISHED",
                frame_signature="sig-other",
                subgoal="break-stall",
                legal_actions=["ACTION1", "ACTION2"],
            )

            self.assertTrue(bundle["recent_records"])
            self.assertEqual(bundle["recent_records"][0]["action_name"], "ACTION1")
            self.assertEqual(bundle["recommended_actions"][0], "ACTION2")

    def test_basin_status_and_suppression_are_state_scoped_for_same_frame_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            initialize_memory_schema(db_path)
            record_transition(
                db_path,
                EpisodeRecord(
                    game_id="ls20",
                    step_index=1,
                    action_name="ACTION1",
                    outcome="stalled",
                    progress_delta=0,
                    game_state="NOT_FINISHED",
                    prior_game_state="NOT_FINISHED",
                    levels_completed=0,
                    prior_frame_signature="sig-shared",
                    next_game_state="NOT_FINISHED",
                    next_frame_signature="sig-shared",
                    prior_legal_actions=["ACTION1", "ACTION2"],
                    stagnation_before=0,
                    stagnation_after=1,
                    decision_subgoal="break-stall",
                    decision_source="symbolic",
                ),
            )
            record_transition(
                db_path,
                EpisodeRecord(
                    game_id="ls20",
                    step_index=2,
                    action_name="ACTION1",
                    outcome="progress",
                    progress_delta=1,
                    game_state="ACTIVE",
                    prior_game_state="ACTIVE",
                    levels_completed=1,
                    prior_frame_signature="sig-shared",
                    next_game_state="ACTIVE",
                    next_frame_signature="sig-next",
                    prior_legal_actions=["ACTION1", "ACTION2"],
                    stagnation_before=0,
                    stagnation_after=0,
                    decision_subgoal="advance-level",
                    decision_source="symbolic",
                ),
            )

            not_finished_basin = load_basin_status(
                db_path, "ls20", "NOT_FINISHED", "sig-shared"
            )
            active_basin = load_basin_status(db_path, "ls20", "ACTIVE", "sig-shared")
            not_finished_bundle = load_memory_bundle(
                db_path,
                game_id="ls20",
                state="NOT_FINISHED",
                frame_signature="sig-shared",
                subgoal="break-stall",
                legal_actions=["ACTION1", "ACTION2"],
            )
            active_bundle = load_memory_bundle(
                db_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature="sig-shared",
                subgoal="advance-level",
                legal_actions=["ACTION1", "ACTION2"],
            )

            self.assertEqual(
                not_finished_basin["action_outcomes"]["ACTION1"]["stalled"], 1
            )
            self.assertEqual(
                active_basin["action_outcomes"]["ACTION1"]["progress"], 1
            )
            self.assertIn("ACTION1", not_finished_bundle["suppressed_actions"])
            self.assertNotIn("ACTION1", active_bundle["suppressed_actions"])

    def test_state_only_basin_status_persists_without_frame_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            initialize_memory_schema(db_path)
            record_transition(
                db_path,
                EpisodeRecord(
                    game_id="ls20",
                    step_index=1,
                    action_name="ACTION1",
                    outcome="stalled",
                    progress_delta=0,
                    game_state="NOT_FINISHED",
                    prior_game_state="NOT_FINISHED",
                    levels_completed=0,
                    prior_frame_signature=None,
                    next_game_state="NOT_FINISHED",
                    next_frame_signature=None,
                    prior_legal_actions=["ACTION1", "ACTION2"],
                    stagnation_before=0,
                    stagnation_after=1,
                    decision_subgoal="break-stall",
                    decision_source="symbolic",
                ),
            )

            basin = load_basin_status(db_path, "ls20", "NOT_FINISHED", "")
            bundle = load_memory_bundle(
                db_path,
                game_id="ls20",
                state="NOT_FINISHED",
                frame_signature="",
                subgoal="break-stall",
                legal_actions=["ACTION1", "ACTION2"],
            )

            self.assertEqual(basin["action_outcomes"]["ACTION1"]["stalled"], 1)
            self.assertEqual(bundle["basin_action_stats"]["ACTION1"]["stalled"], 1)
            self.assertIn("ACTION1", bundle["suppressed_actions"])

    def test_recent_records_prioritize_current_basin_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            initialize_memory_schema(db_path)
            records = [
                EpisodeRecord(
                    game_id="ls20",
                    step_index=1,
                    action_name="ACTION1",
                    outcome="stalled",
                    progress_delta=0,
                    game_state="ACTIVE",
                    prior_game_state="ACTIVE",
                    levels_completed=0,
                    prior_frame_signature="sig-other",
                    next_game_state="ACTIVE",
                    next_frame_signature="sig-other",
                    prior_legal_actions=["ACTION1", "ACTION2"],
                    stagnation_before=0,
                    stagnation_after=1,
                    decision_subgoal="advance-level",
                    decision_source="symbolic",
                ),
                EpisodeRecord(
                    game_id="ls20",
                    step_index=2,
                    action_name="ACTION2",
                    outcome="progress",
                    progress_delta=1,
                    game_state="NOT_FINISHED",
                    prior_game_state="NOT_FINISHED",
                    levels_completed=1,
                    prior_frame_signature="sig-target",
                    next_game_state="NOT_FINISHED",
                    next_frame_signature="sig-next",
                    prior_legal_actions=["ACTION1", "ACTION2"],
                    stagnation_before=1,
                    stagnation_after=0,
                    decision_subgoal="break-stall",
                    decision_source="symbolic",
                ),
                EpisodeRecord(
                    game_id="ls20",
                    step_index=3,
                    action_name="ACTION1",
                    outcome="stalled",
                    progress_delta=0,
                    game_state="NOT_FINISHED",
                    prior_game_state="NOT_FINISHED",
                    levels_completed=1,
                    prior_frame_signature="sig-other-2",
                    next_game_state="NOT_FINISHED",
                    next_frame_signature="sig-other-2",
                    prior_legal_actions=["ACTION1", "ACTION2"],
                    stagnation_before=0,
                    stagnation_after=1,
                    decision_subgoal="break-stall",
                    decision_source="symbolic",
                ),
            ]
            for record in records:
                record_transition(db_path, record)

            bundle = load_memory_bundle(
                db_path,
                game_id="ls20",
                state="NOT_FINISHED",
                frame_signature="sig-target",
                subgoal="break-stall",
                legal_actions=["ACTION1", "ACTION2"],
            )

            self.assertTrue(bundle["recent_records"])
            self.assertEqual(bundle["recent_records"][0]["prior_game_state"], "NOT_FINISHED")
            self.assertEqual(bundle["recent_records"][0]["prior_frame_signature"], "sig-target")
            self.assertEqual(bundle["recent_records"][0]["decision_subgoal"], "break-stall")

    def test_recent_records_prefer_same_state_over_same_frame_wrong_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            initialize_memory_schema(db_path)
            records = [
                EpisodeRecord(
                    game_id="ls20",
                    step_index=1,
                    action_name="ACTION1",
                    outcome="stalled",
                    progress_delta=0,
                    game_state="ACTIVE",
                    prior_game_state="ACTIVE",
                    levels_completed=0,
                    prior_frame_signature="sig-target",
                    next_game_state="ACTIVE",
                    next_frame_signature="sig-target",
                    prior_legal_actions=["ACTION1", "ACTION2"],
                    stagnation_before=0,
                    stagnation_after=1,
                    decision_subgoal="advance-level",
                    decision_source="symbolic",
                ),
                EpisodeRecord(
                    game_id="ls20",
                    step_index=2,
                    action_name="ACTION2",
                    outcome="progress",
                    progress_delta=1,
                    game_state="NOT_FINISHED",
                    prior_game_state="NOT_FINISHED",
                    levels_completed=1,
                    prior_frame_signature="sig-other",
                    next_game_state="NOT_FINISHED",
                    next_frame_signature="sig-next",
                    prior_legal_actions=["ACTION1", "ACTION2"],
                    stagnation_before=1,
                    stagnation_after=0,
                    decision_subgoal="break-stall",
                    decision_source="symbolic",
                ),
            ]
            for record in records:
                record_transition(db_path, record)

            bundle = load_memory_bundle(
                db_path,
                game_id="ls20",
                state="NOT_FINISHED",
                frame_signature="sig-target",
                subgoal="break-stall",
                legal_actions=["ACTION1", "ACTION2"],
            )

            self.assertTrue(bundle["recent_records"])
            self.assertEqual(bundle["recent_records"][0]["prior_game_state"], "NOT_FINISHED")
            self.assertEqual(bundle["recent_records"][0]["prior_frame_signature"], "sig-other")

    def test_recent_records_can_retrieve_older_matching_context_beyond_global_recent_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            initialize_memory_schema(db_path)
            record_transition(
                db_path,
                EpisodeRecord(
                    game_id="ls20",
                    step_index=1,
                    action_name="ACTION2",
                    outcome="progress",
                    progress_delta=1,
                    game_state="NOT_FINISHED",
                    prior_game_state="NOT_FINISHED",
                    levels_completed=1,
                    prior_frame_signature="sig-target",
                    next_game_state="NOT_FINISHED",
                    next_frame_signature="sig-next",
                    prior_legal_actions=["ACTION1", "ACTION2"],
                    stagnation_before=0,
                    stagnation_after=0,
                    decision_subgoal="break-stall",
                    decision_source="symbolic",
                ),
            )
            for step_index in range(2, 42):
                record_transition(
                    db_path,
                    EpisodeRecord(
                        game_id="ls20",
                        step_index=step_index,
                        action_name="ACTION1",
                        outcome="stalled",
                        progress_delta=0,
                        game_state="ACTIVE",
                        prior_game_state="ACTIVE",
                        levels_completed=1,
                        prior_frame_signature=f"sig-noise-{step_index}",
                        next_game_state="ACTIVE",
                        next_frame_signature=f"sig-noise-{step_index}",
                        prior_legal_actions=["ACTION1", "ACTION2"],
                        stagnation_before=0,
                        stagnation_after=1,
                        decision_subgoal="advance-level",
                        decision_source="symbolic",
                    ),
                )

            bundle = load_memory_bundle(
                db_path,
                game_id="ls20",
                state="NOT_FINISHED",
                frame_signature="sig-target",
                subgoal="break-stall",
                legal_actions=["ACTION1", "ACTION2"],
            )

            self.assertTrue(bundle["recent_records"])
            self.assertEqual(bundle["recent_records"][0]["prior_game_state"], "NOT_FINISHED")
            self.assertEqual(bundle["recent_records"][0]["prior_frame_signature"], "sig-target")
            self.assertEqual(bundle["recent_records"][0]["decision_subgoal"], "break-stall")

    def test_recommended_actions_prefer_state_and_subgoal_over_coarse_frame_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            initialize_memory_schema(db_path)
            record_transition(
                db_path,
                EpisodeRecord(
                    game_id="ls20",
                    step_index=1,
                    action_name="ACTION1",
                    outcome="progress",
                    progress_delta=1,
                    game_state="ACTIVE",
                    prior_game_state="ACTIVE",
                    levels_completed=1,
                    prior_frame_signature="sig-shared",
                    next_game_state="ACTIVE",
                    next_frame_signature="sig-next",
                    prior_legal_actions=["ACTION1", "ACTION2"],
                    stagnation_before=0,
                    stagnation_after=0,
                    decision_subgoal="advance-level",
                    decision_source="symbolic",
                ),
            )
            record_transition(
                db_path,
                EpisodeRecord(
                    game_id="ls20",
                    step_index=2,
                    action_name="ACTION2",
                    outcome="progress",
                    progress_delta=1,
                    game_state="NOT_FINISHED",
                    prior_game_state="NOT_FINISHED",
                    levels_completed=1,
                    prior_frame_signature="sig-other",
                    next_game_state="NOT_FINISHED",
                    next_frame_signature="sig-next-2",
                    prior_legal_actions=["ACTION1", "ACTION2"],
                    stagnation_before=0,
                    stagnation_after=0,
                    decision_subgoal="break-stall",
                    decision_source="symbolic",
                ),
            )

            bundle = load_memory_bundle(
                db_path,
                game_id="ls20",
                state="NOT_FINISHED",
                frame_signature="sig-shared",
                subgoal="break-stall",
                legal_actions=["ACTION1", "ACTION2"],
            )

            self.assertEqual(bundle["frame_action_stats"]["ACTION1"]["progress"], 1)
            self.assertEqual(bundle["state_action_stats"]["ACTION2"]["progress"], 1)
            self.assertEqual(bundle["subgoal_action_stats"]["ACTION2"]["progress"], 1)
            self.assertEqual(bundle["recommended_actions"][0], "ACTION2")

    def test_suppressed_actions_can_come_from_state_and_subgoal_when_basin_is_new(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            initialize_memory_schema(db_path)
            record_transition(
                db_path,
                EpisodeRecord(
                    game_id="ls20",
                    step_index=1,
                    action_name="ACTION1",
                    outcome="stalled",
                    progress_delta=0,
                    game_state="NOT_FINISHED",
                    prior_game_state="NOT_FINISHED",
                    levels_completed=0,
                    prior_frame_signature="sig-old",
                    next_game_state="NOT_FINISHED",
                    next_frame_signature="sig-old",
                    prior_legal_actions=["ACTION1", "ACTION2"],
                    stagnation_before=0,
                    stagnation_after=1,
                    decision_subgoal="break-stall",
                    decision_source="symbolic",
                ),
            )
            record_transition(
                db_path,
                EpisodeRecord(
                    game_id="ls20",
                    step_index=2,
                    action_name="ACTION1",
                    outcome="stalled",
                    progress_delta=0,
                    game_state="NOT_FINISHED",
                    prior_game_state="NOT_FINISHED",
                    levels_completed=0,
                    prior_frame_signature="sig-old-2",
                    next_game_state="NOT_FINISHED",
                    next_frame_signature="sig-old-2",
                    prior_legal_actions=["ACTION1", "ACTION2"],
                    stagnation_before=1,
                    stagnation_after=2,
                    decision_subgoal="break-stall",
                    decision_source="symbolic",
                ),
            )

            bundle = load_memory_bundle(
                db_path,
                game_id="ls20",
                state="NOT_FINISHED",
                frame_signature="sig-new",
                subgoal="break-stall",
                legal_actions=["ACTION1", "ACTION2"],
            )

            self.assertEqual(bundle["basin_action_stats"]["ACTION1"]["stalled"], 2)
            self.assertEqual(bundle["state_action_stats"]["ACTION1"]["stalled"], 2)
            self.assertEqual(bundle["subgoal_action_stats"]["ACTION1"]["stalled"], 2)
            self.assertIn("ACTION1", bundle["suppressed_actions"])
            self.assertNotIn("ACTION2", bundle["suppressed_actions"])


if __name__ == "__main__":
    unittest.main()
