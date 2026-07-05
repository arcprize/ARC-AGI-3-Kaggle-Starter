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
    initialize_schema,
    load_basin_status,
    load_memory_bundle,
    load_memory_bundle_from_stores,
    upsert_abstract_memory_rows,
    upsert_training_counts,
)
from agi_core.storage.memory_sqlite import record_transition
from agi_core.storage.memory_sqlite import _merge_ranked_memory_items
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
            self.assertIn("live_episode_summaries", tables)
            self.assertIn("live_pattern_stats", tables)
            self.assertIn("live_sequence_stats", tables)
            self.assertIn("live_board_cluster_stats", tables)
            self.assertIn("live_semantic_memory", tables)
            self.assertIn("live_procedural_memory", tables)
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

    def test_load_memory_bundle_uses_short_offline_key_for_full_runtime_game_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            initialize_memory_schema(db_path)
            upsert_training_counts(
                db_path,
                [
                    {
                        "scope": "state",
                        "game_id": "ls20",
                        "state": "ACTIVE",
                        "action_name": "ACTION1",
                        "count": 6,
                    },
                ],
            )
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
                    prior_frame_signature="sig-offline",
                    next_game_state="ACTIVE",
                    next_frame_signature="sig-next",
                    prior_legal_actions=["ACTION1", "ACTION2"],
                    stagnation_before=0,
                    stagnation_after=0,
                    decision_subgoal="advance-level",
                    decision_source="offline-demo-trainer",
                ),
            )

            bundle = load_memory_bundle(
                db_path,
                game_id="ls20-9607627b",
                state="ACTIVE",
                frame_signature="sig-offline",
                subgoal="advance-level",
                legal_actions=["ACTION1", "ACTION2"],
            )

            self.assertEqual(bundle["trained_prior_bucket"]["ACTION1"], 6)
            self.assertTrue(bundle["recent_records"])
            self.assertEqual(bundle["recent_records"][0]["game_id"], "ls20")
            self.assertEqual(bundle["action_stats"]["ACTION1"]["progress"], 1)

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

    def test_load_memory_bundle_prefers_trained_subgoal_prior_when_frame_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            initialize_memory_schema(db_path)
            upsert_training_counts(
                db_path,
                [
                    {
                        "scope": "state",
                        "game_id": "ls20",
                        "state": "ACTIVE",
                        "action_name": "ACTION1",
                        "count": 8,
                    },
                    {
                        "scope": "subgoal",
                        "game_id": "ls20",
                        "subgoal": "inspect-active-board",
                        "action_name": "ACTION6",
                        "count": 11,
                    },
                ],
            )

            bundle = load_memory_bundle(
                db_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature="sig-missing",
                subgoal="inspect-active-board",
                legal_actions=["ACTION1", "ACTION6"],
            )

            self.assertEqual(bundle["trained_prior_bucket"], {"ACTION6": 11, "ACTION1": 8})
            self.assertEqual(bundle["recommended_actions"][0], "ACTION6")
            self.assertEqual(
                bundle["working_memory"]["trained_prior_focus"],
                [
                    {"action_name": "ACTION6", "count": 11},
                    {"action_name": "ACTION1", "count": 8},
                ],
            )
            self.assertEqual(
                bundle["retrieval_summary"]["trained_prior_focus"],
                [
                    {"action_name": "ACTION6", "count": 11},
                    {"action_name": "ACTION1", "count": 8},
                ],
            )

    def test_load_memory_bundle_prefers_subgoal_prior_over_same_action_frame_prior(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            initialize_memory_schema(db_path)
            upsert_training_counts(
                db_path,
                [
                    {
                        "scope": "frame",
                        "game_id": "ls20",
                        "frame_signature": "sig-probe",
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
                        "scope": "state",
                        "game_id": "ls20",
                        "state": "ACTIVE",
                        "action_name": "ACTION1",
                        "count": 8,
                    },
                ],
            )

            bundle = load_memory_bundle(
                db_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature="sig-probe",
                subgoal="inspect-active-board",
                legal_actions=["ACTION1", "ACTION6"],
            )

            self.assertEqual(bundle["trained_prior_bucket"], {"ACTION6": 11, "ACTION1": 8})
            self.assertEqual(bundle["recommended_actions"][0], "ACTION6")

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

    def test_recent_records_prefer_matching_board_context_when_state_and_subgoal_tie(self) -> None:
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
                    prior_frame_signature="sig-alpha",
                    next_game_state="ACTIVE",
                    next_frame_signature="sig-next-alpha",
                    prior_legal_actions=["ACTION1", "ACTION6"],
                    stagnation_before=0,
                    stagnation_after=0,
                    decision_subgoal="inspect-active-board",
                    decision_source="symbolic",
                    prior_board_height=3,
                    prior_board_width=3,
                    prior_board_active_cells=4,
                    prior_board_source_attr="frame",
                    prior_board_default_probe={"x": 1, "y": 1},
                ),
            )
            record_transition(
                db_path,
                EpisodeRecord(
                    game_id="ls20",
                    step_index=2,
                    action_name="ACTION6",
                    outcome="progress",
                    progress_delta=1,
                    game_state="ACTIVE",
                    prior_game_state="ACTIVE",
                    levels_completed=1,
                    prior_frame_signature="sig-beta",
                    next_game_state="ACTIVE",
                    next_frame_signature="sig-next-beta",
                    prior_legal_actions=["ACTION1", "ACTION6"],
                    stagnation_before=0,
                    stagnation_after=0,
                    decision_subgoal="inspect-active-board",
                    decision_source="symbolic",
                    prior_board_height=5,
                    prior_board_width=5,
                    prior_board_active_cells=6,
                    prior_board_source_attr="frame",
                    prior_board_default_probe={"x": 2, "y": 2},
                ),
            )

            bundle = load_memory_bundle(
                db_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature="sig-missing",
                subgoal="inspect-active-board",
                legal_actions=["ACTION1", "ACTION6"],
                board_context={
                    "board_height": 5,
                    "board_width": 5,
                    "board_active_cells": 6,
                    "board_source_attr": "frame",
                    "board_default_probe": {"x": 2, "y": 2},
                },
            )

            self.assertTrue(bundle["recent_records"])
            self.assertEqual(bundle["recent_records"][0]["action_name"], "ACTION6")
            self.assertEqual(
                bundle["recent_records"][0]["board_context_summary"]["board_height"],
                5,
            )
            self.assertEqual(bundle["board_context_summary"]["board_width"], 5)
            self.assertEqual(bundle["board_context_action_stats"]["ACTION6"]["progress"], 1)
            self.assertEqual(bundle["recommended_actions"][0], "ACTION6")

    def test_episode_summaries_capture_deeper_context_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            initialize_memory_schema(db_path)
            for step_index, outcome in enumerate(("stalled", "progress", "progress"), start=1):
                record_transition(
                    db_path,
                    EpisodeRecord(
                        game_id="ls20",
                        step_index=step_index,
                        action_name="ACTION6",
                        outcome=outcome,
                        progress_delta=1 if outcome == "progress" else 0,
                        game_state="ACTIVE",
                        prior_game_state="ACTIVE",
                        levels_completed=1 if outcome == "progress" else 0,
                        prior_frame_signature="sig-board",
                        next_game_state="ACTIVE",
                        next_frame_signature=f"sig-next-{step_index}",
                        prior_legal_actions=["ACTION1", "ACTION6"],
                        stagnation_before=step_index - 1,
                        stagnation_after=0 if outcome == "progress" else step_index,
                        decision_subgoal="inspect-active-board",
                        decision_source="symbolic",
                        prior_board_height=5,
                        prior_board_width=5,
                        prior_board_active_cells=6,
                        prior_board_source_attr="frame",
                        prior_board_default_probe={"x": 2, "y": 2},
                    ),
                )

            bundle = load_memory_bundle(
                db_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature="sig-board",
                subgoal="inspect-active-board",
                legal_actions=["ACTION1", "ACTION6"],
                board_context={
                    "board_height": 5,
                    "board_width": 5,
                    "board_active_cells": 6,
                    "board_source_attr": "frame",
                    "board_default_probe": {"x": 2, "y": 2},
                },
            )

            self.assertTrue(bundle["episode_summaries"])
            top_summary = bundle["episode_summaries"][0]
            self.assertEqual(top_summary["prior_game_state"], "ACTIVE")
            self.assertEqual(top_summary["prior_frame_signature"], "sig-board")
            self.assertEqual(top_summary["decision_subgoal"], "inspect-active-board")
            self.assertEqual(top_summary["first_step_index"], 1)
            self.assertEqual(top_summary["last_step_index"], 3)
            self.assertEqual(top_summary["total_events"], 3)
            self.assertEqual(top_summary["recent_action_sequence"], ["ACTION6", "ACTION6", "ACTION6"])
            self.assertEqual(top_summary["recent_outcome_sequence"], ["stalled", "progress", "progress"])
            self.assertEqual(top_summary["decision_sources"], {"symbolic": 3})
            self.assertEqual(top_summary["max_stagnation_after"], 1)
            self.assertEqual(top_summary["action_outcomes"]["ACTION6"]["progress"], 2)
            self.assertEqual(top_summary["action_outcomes"]["ACTION6"]["stalled"], 1)

    def test_board_context_action_stats_look_beyond_six_recent_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            initialize_memory_schema(db_path)
            for step_index in range(1, 11):
                record_transition(
                    db_path,
                    EpisodeRecord(
                        game_id="ls20",
                        step_index=step_index,
                        action_name="ACTION2",
                        outcome="stalled",
                        progress_delta=0,
                        game_state="ACTIVE",
                        prior_game_state="ACTIVE",
                        levels_completed=0,
                        prior_frame_signature=f"sig-window-{step_index}",
                        next_game_state="ACTIVE",
                        next_frame_signature=f"sig-window-{step_index}",
                        prior_legal_actions=["ACTION1", "ACTION2"],
                        stagnation_before=step_index - 1,
                        stagnation_after=step_index,
                        decision_subgoal="break-stall",
                        decision_source="fallback",
                        prior_board_height=5,
                        prior_board_width=5,
                        prior_board_active_cells=6,
                        prior_board_source_attr="frame",
                        prior_board_default_probe={"x": 2, "y": 2},
                    ),
                )

            bundle = load_memory_bundle(
                db_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature="sig-window-missing",
                subgoal="break-stall",
                legal_actions=["ACTION1", "ACTION2"],
                board_context={
                    "board_height": 5,
                    "board_width": 5,
                    "board_active_cells": 6,
                    "board_source_attr": "frame",
                    "board_default_probe": {"x": 2, "y": 2},
                },
            )

            self.assertEqual(bundle["board_context_action_stats"]["ACTION2"]["stalled"], 10)

    def test_hierarchical_memory_tables_capture_pattern_sequence_and_board_cluster_stats(self) -> None:
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
                    prior_frame_signature="sig-h1",
                    next_game_state="ACTIVE",
                    next_frame_signature="sig-h1",
                    prior_legal_actions=["ACTION1", "ACTION2", "ACTION6"],
                    stagnation_before=0,
                    stagnation_after=1,
                    decision_subgoal="break-stall",
                    decision_source="symbolic",
                    prior_board_height=5,
                    prior_board_width=5,
                    prior_board_active_cells=6,
                    prior_board_source_attr="frame",
                    prior_board_default_probe={"x": 2, "y": 2},
                ),
                EpisodeRecord(
                    game_id="ls20",
                    step_index=2,
                    action_name="ACTION2",
                    outcome="progress",
                    progress_delta=1,
                    game_state="ACTIVE",
                    prior_game_state="ACTIVE",
                    levels_completed=1,
                    prior_frame_signature="sig-h2",
                    next_game_state="ACTIVE",
                    next_frame_signature="sig-h2-next",
                    prior_legal_actions=["ACTION1", "ACTION2", "ACTION6"],
                    stagnation_before=1,
                    stagnation_after=0,
                    decision_subgoal="break-stall",
                    decision_source="symbolic",
                    prior_board_height=5,
                    prior_board_width=5,
                    prior_board_active_cells=6,
                    prior_board_source_attr="frame",
                    prior_board_default_probe={"x": 2, "y": 2},
                ),
            ]
            for record in records:
                record_transition(db_path, record)

            bundle = load_memory_bundle(
                db_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature="sig-missing",
                subgoal="break-stall",
                legal_actions=["ACTION1", "ACTION2", "ACTION6"],
                board_context={
                    "board_height": 5,
                    "board_width": 5,
                    "board_active_cells": 6,
                    "board_source_attr": "frame",
                    "board_default_probe": {"x": 2, "y": 2},
                },
                last_action="ACTION1",
            )

            self.assertEqual(bundle["pattern_action_stats"]["ACTION1"]["stalled"], 1)
            self.assertEqual(bundle["pattern_action_stats"]["ACTION2"]["progress"], 1)
            self.assertEqual(bundle["board_cluster_action_stats"]["ACTION2"]["progress"], 1)
            self.assertEqual(bundle["sequence_action_stats"]["ACTION2"]["progress"], 1)
            self.assertTrue(bundle["semantic_memories"])
            self.assertTrue(bundle["procedural_memories"])
            self.assertTrue(bundle["working_memory"])
            self.assertTrue(bundle["retrieval_summary"])
            self.assertEqual(bundle["recommended_actions"][0], "ACTION2")

    def test_recent_records_address_by_board_family_and_last_action_when_frame_signature_drifts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            initialize_memory_schema(db_path)
            record_transition(
                db_path,
                EpisodeRecord(
                    game_id="ls20",
                    step_index=1,
                    action_name="ACTION3",
                    outcome="stalled",
                    progress_delta=0,
                    game_state="ACTIVE",
                    prior_game_state="ACTIVE",
                    levels_completed=0,
                    prior_frame_signature="sig-target-family",
                    next_game_state="ACTIVE",
                    next_frame_signature="sig-target-family",
                    prior_legal_actions=["ACTION1", "ACTION3"],
                    stagnation_before=0,
                    stagnation_after=1,
                    decision_subgoal="break-stall",
                    decision_source="symbolic",
                    prior_board_height=6,
                    prior_board_width=6,
                    prior_board_active_cells=20,
                    prior_board_source_attr="frame",
                    prior_board_default_probe={"x": 1, "y": 1},
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
                    game_state="ACTIVE",
                    prior_game_state="ACTIVE",
                    levels_completed=1,
                    prior_frame_signature="sig-recent-other",
                    next_game_state="ACTIVE",
                    next_frame_signature="sig-recent-other-next",
                    prior_legal_actions=["ACTION1", "ACTION2", "ACTION3"],
                    stagnation_before=0,
                    stagnation_after=0,
                    decision_subgoal="break-stall",
                    decision_source="symbolic",
                    prior_board_height=6,
                    prior_board_width=6,
                    prior_board_active_cells=35,
                    prior_board_source_attr="frame",
                    prior_board_default_probe={"x": 5, "y": 5},
                ),
            )

            bundle = load_memory_bundle(
                db_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature="sig-live-new",
                subgoal="break-stall",
                legal_actions=["ACTION1", "ACTION2", "ACTION3"],
                board_context={
                    "board_height": 6,
                    "board_width": 6,
                    "board_active_cells": 18,
                    "board_source_attr": "frame",
                    "board_default_probe": {"x": 0, "y": 0},
                },
                last_action="ACTION3",
            )

            self.assertTrue(bundle["recent_records"])
            self.assertEqual(bundle["recent_records"][0]["action_name"], "ACTION3")
            self.assertEqual(bundle["recent_records"][0]["prior_frame_signature"], "sig-target-family")

    def test_episode_summaries_address_by_board_family_and_last_action_when_frame_signature_drifts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            initialize_memory_schema(db_path)
            for step_index in range(1, 3):
                record_transition(
                    db_path,
                    EpisodeRecord(
                        game_id="ls20",
                        step_index=step_index,
                        action_name="ACTION3",
                        outcome="stalled" if step_index == 1 else "progress",
                        progress_delta=0 if step_index == 1 else 1,
                        game_state="ACTIVE",
                        prior_game_state="ACTIVE",
                        levels_completed=0 if step_index == 1 else 1,
                        prior_frame_signature="sig-family-a",
                        next_game_state="ACTIVE",
                        next_frame_signature=f"sig-family-a-next-{step_index}",
                        prior_legal_actions=["ACTION1", "ACTION3"],
                        stagnation_before=step_index - 1,
                        stagnation_after=1 if step_index == 1 else 0,
                        decision_subgoal="break-stall",
                        decision_source="symbolic",
                        prior_board_height=6,
                        prior_board_width=6,
                        prior_board_active_cells=18,
                        prior_board_source_attr="frame",
                        prior_board_default_probe={"x": 1, "y": 1},
                    ),
                )
            for step_index in range(3, 5):
                record_transition(
                    db_path,
                    EpisodeRecord(
                        game_id="ls20",
                        step_index=step_index,
                        action_name="ACTION2",
                        outcome="progress",
                        progress_delta=1,
                        game_state="ACTIVE",
                        prior_game_state="ACTIVE",
                        levels_completed=1,
                        prior_frame_signature="sig-family-b",
                        next_game_state="ACTIVE",
                        next_frame_signature=f"sig-family-b-next-{step_index}",
                        prior_legal_actions=["ACTION1", "ACTION2"],
                        stagnation_before=0,
                        stagnation_after=0,
                        decision_subgoal="break-stall",
                        decision_source="symbolic",
                        prior_board_height=6,
                        prior_board_width=6,
                        prior_board_active_cells=34,
                        prior_board_source_attr="frame",
                        prior_board_default_probe={"x": 5, "y": 5},
                    ),
                )

            bundle = load_memory_bundle(
                db_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature="sig-live-drifted",
                subgoal="break-stall",
                legal_actions=["ACTION1", "ACTION2", "ACTION3"],
                board_context={
                    "board_height": 6,
                    "board_width": 6,
                    "board_active_cells": 20,
                    "board_source_attr": "frame",
                    "board_default_probe": {"x": 0, "y": 0},
                },
                last_action="ACTION3",
            )

            self.assertTrue(bundle["episode_summaries"])
            self.assertEqual(bundle["episode_summaries"][0]["prior_frame_signature"], "sig-family-a")
            self.assertEqual(bundle["episode_summaries"][0]["recent_action_sequence"][-1], "ACTION3")

    def test_recommended_actions_promote_untried_actions_during_board_context_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            initialize_memory_schema(db_path)
            for step_index, action_name in enumerate(("ACTION2", "ACTION3"), start=1):
                record_transition(
                    db_path,
                    EpisodeRecord(
                        game_id="ls20",
                        step_index=step_index,
                        action_name=action_name,
                        outcome="stalled",
                        progress_delta=0,
                        game_state="ACTIVE",
                        prior_game_state="ACTIVE",
                        levels_completed=0,
                        prior_frame_signature=f"sig-recovery-{step_index}",
                        next_game_state="ACTIVE",
                        next_frame_signature=f"sig-recovery-{step_index}",
                        prior_legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
                        stagnation_before=step_index - 1,
                        stagnation_after=step_index,
                        decision_subgoal="break-stall",
                        decision_source="fallback",
                        prior_board_height=5,
                        prior_board_width=5,
                        prior_board_active_cells=6,
                        prior_board_source_attr="frame",
                        prior_board_default_probe={"x": 2, "y": 2},
                    ),
                )

            bundle = load_memory_bundle(
                db_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature="sig-recovery-missing",
                subgoal="break-stall",
                legal_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
                board_context={
                    "board_height": 5,
                    "board_width": 5,
                    "board_active_cells": 6,
                    "board_source_attr": "frame",
                    "board_default_probe": {"x": 2, "y": 2},
                },
            )

            self.assertEqual(bundle["board_context_action_stats"]["ACTION2"]["stalled"], 1)
            self.assertEqual(bundle["board_context_action_stats"]["ACTION3"]["stalled"], 1)
            self.assertEqual(bundle["recommended_actions"][:2], ["ACTION1", "ACTION4"])

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

    def test_load_memory_bundle_from_stores_merges_hot_and_archive_with_hot_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            hot_db_path = Path(tmpdir) / "hot.sqlite3"
            archive_db_path = Path(tmpdir) / "archive.sqlite3"
            initialize_memory_schema(hot_db_path)
            initialize_memory_schema(archive_db_path)
            record_transition(
                hot_db_path,
                EpisodeRecord(
                    game_id="ls20",
                    step_index=10,
                    action_name="ACTION1",
                    outcome="stalled",
                    progress_delta=0,
                    game_state="ACTIVE",
                    prior_game_state="ACTIVE",
                    levels_completed=0,
                    prior_frame_signature="sig-hot",
                    next_game_state="ACTIVE",
                    next_frame_signature="sig-hot",
                    prior_legal_actions=["ACTION1", "ACTION2"],
                    stagnation_before=0,
                    stagnation_after=1,
                    decision_subgoal="break-stall",
                    decision_source="symbolic",
                ),
            )
            record_transition(
                archive_db_path,
                EpisodeRecord(
                    game_id="ls20",
                    step_index=1,
                    action_name="ACTION2",
                    outcome="progress",
                    progress_delta=1,
                    game_state="ACTIVE",
                    prior_game_state="ACTIVE",
                    levels_completed=1,
                    prior_frame_signature="sig-archive",
                    next_game_state="ACTIVE",
                    next_frame_signature="sig-archive-next",
                    prior_legal_actions=["ACTION1", "ACTION2"],
                    stagnation_before=0,
                    stagnation_after=0,
                    decision_subgoal="break-stall",
                    decision_source="symbolic",
                ),
            )

            bundle = load_memory_bundle_from_stores(
                hot_db_path=hot_db_path,
                archive_db_path=archive_db_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature="sig-live",
                subgoal="break-stall",
                legal_actions=["ACTION1", "ACTION2"],
                board_context=None,
                last_action="ACTION1",
            )

            self.assertEqual(bundle["action_stats"]["ACTION1"]["stalled"], 1)
            self.assertEqual(bundle["action_stats"]["ACTION2"]["progress"], 1)
            self.assertTrue(bundle["recent_records"])
            self.assertTrue(bundle["semantic_memories"])
            self.assertTrue(bundle["procedural_memories"])
            self.assertTrue(bundle["episodic_recalls"])
            self.assertTrue(bundle["retrieval_summary"])
            self.assertEqual(bundle["recent_records"][0]["prior_frame_signature"], "sig-hot")

    def test_load_memory_bundle_updates_semantic_and_procedural_access_counters(self) -> None:
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
                    game_state="ACTIVE",
                    prior_game_state="ACTIVE",
                    levels_completed=1,
                    prior_frame_signature="sig-readonly",
                    next_game_state="ACTIVE",
                    next_frame_signature="sig-readonly-next",
                    prior_legal_actions=["ACTION1", "ACTION2"],
                    stagnation_before=0,
                    stagnation_after=0,
                    decision_subgoal="advance-level",
                    decision_source="symbolic",
                ),
            )

            first_bundle = load_memory_bundle(
                db_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature="sig-readonly",
                subgoal="advance-level",
                legal_actions=["ACTION1", "ACTION2"],
            )
            second_bundle = load_memory_bundle(
                db_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature="sig-readonly",
                subgoal="advance-level",
                legal_actions=["ACTION1", "ACTION2"],
            )

            self.assertEqual(
                first_bundle["semantic_memories"][0]["action_name"],
                second_bundle["semantic_memories"][0]["action_name"],
            )
            self.assertEqual(
                first_bundle["procedural_memories"][0]["action_name"],
                second_bundle["procedural_memories"][0]["action_name"],
            )
            with closing(sqlite3.connect(db_path)) as connection:
                semantic_counts = connection.execute(
                    "SELECT access_count, last_access_step FROM live_semantic_memory"
                ).fetchall()
                procedural_counts = connection.execute(
                    "SELECT access_count, last_access_step FROM live_procedural_memory"
                ).fetchall()
            self.assertTrue(semantic_counts)
            self.assertTrue(procedural_counts)
            self.assertTrue(all(int(row[0] or 0) == 2 for row in semantic_counts))
            self.assertTrue(all(int(row[1] or 0) == 1 for row in semantic_counts))
            self.assertTrue(all(int(row[0] or 0) == 2 for row in procedural_counts))
            self.assertTrue(all(int(row[1] or 0) == 1 for row in procedural_counts))
            self.assertEqual(first_bundle["semantic_memories"][0]["access_count"], 0)
            self.assertEqual(second_bundle["semantic_memories"][0]["access_count"], 1)
            self.assertEqual(first_bundle["procedural_memories"][0]["access_count"], 0)
            self.assertEqual(second_bundle["procedural_memories"][0]["access_count"], 1)

    def test_load_memory_bundle_uses_explicit_retrieval_step_for_access_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            initialize_memory_schema(db_path)
            record_transition(
                db_path,
                EpisodeRecord(
                    game_id="ls20",
                    step_index=3,
                    action_name="ACTION2",
                    outcome="progress",
                    progress_delta=1,
                    game_state="ACTIVE",
                    prior_game_state="ACTIVE",
                    levels_completed=1,
                    prior_frame_signature="sig-explicit",
                    next_game_state="ACTIVE",
                    next_frame_signature="sig-explicit-next",
                    prior_legal_actions=["ACTION1", "ACTION2"],
                    stagnation_before=0,
                    stagnation_after=0,
                    decision_subgoal="advance-level",
                    decision_source="symbolic",
                ),
            )

            load_memory_bundle(
                db_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature="sig-explicit",
                subgoal="advance-level",
                legal_actions=["ACTION1", "ACTION2"],
                retrieval_step=27,
            )

            with closing(sqlite3.connect(db_path)) as connection:
                semantic_counts = connection.execute(
                    "SELECT access_count, last_access_step FROM live_semantic_memory"
                ).fetchall()
                procedural_counts = connection.execute(
                    "SELECT access_count, last_access_step FROM live_procedural_memory"
                ).fetchall()
            self.assertTrue(all(int(row[0] or 0) == 1 for row in semantic_counts))
            self.assertTrue(all(int(row[1] or 0) == 27 for row in semantic_counts))
            self.assertTrue(all(int(row[0] or 0) == 1 for row in procedural_counts))
            self.assertTrue(all(int(row[1] or 0) == 27 for row in procedural_counts))

    def test_merge_ranked_memory_items_does_not_let_large_step_index_dominate_relevance(self) -> None:
        merged = _merge_ranked_memory_items(
            [
                {
                    "memory_type": "semantic",
                    "state": "ACTIVE",
                    "subgoal": "break-stall",
                    "action_name": "ACTION1",
                    "retrieval_rank": [1.08, 1.0, 3.0, 1.0, 0.8, 2.5, 0.9, 0.02],
                    "score": 10.28,
                    "salience": 2.5,
                    "confidence": 0.9,
                    "last_step_index": 20,
                }
            ],
            [
                {
                    "memory_type": "semantic",
                    "state": "ACTIVE",
                    "subgoal": "break-stall",
                    "action_name": "ACTION2",
                    "retrieval_rank": [1.0, 1.0, 1.0, 0.0, 0.4, 1.2, 0.6, 0.99],
                    "score": 5.2,
                    "salience": 1.2,
                    "confidence": 0.6,
                    "last_step_index": 990,
                }
            ],
            limit=2,
        )

        self.assertEqual(merged[0]["action_name"], "ACTION1")

    def test_merge_ranked_memory_items_keeps_distinct_semantic_board_contexts(self) -> None:
        merged = _merge_ranked_memory_items(
            [
                {
                    "memory_type": "semantic",
                    "state": "ACTIVE",
                    "subgoal": "break-stall",
                    "action_name": "ACTION1",
                    "board_context_summary": {
                        "board_height": 3,
                        "board_width": 3,
                        "board_active_cells": 4,
                        "board_source_attr": "frame",
                    },
                    "retrieval_rank": [1.08, 1.0, 3.0, 1.0, 0.8, 2.5, 0.9, 0.02],
                    "score": 10.28,
                    "salience": 2.5,
                    "confidence": 0.9,
                    "last_step_index": 20,
                }
            ],
            [
                {
                    "memory_type": "semantic",
                    "state": "ACTIVE",
                    "subgoal": "break-stall",
                    "action_name": "ACTION1",
                    "board_context_summary": {
                        "board_height": 5,
                        "board_width": 5,
                        "board_active_cells": 8,
                        "board_source_attr": "grid",
                    },
                    "retrieval_rank": [1.0, 0.5, 1.5, 0.0, 0.4, 1.2, 0.6, 0.05],
                    "score": 5.2,
                    "salience": 1.2,
                    "confidence": 0.6,
                    "last_step_index": 50,
                }
            ],
            limit=4,
        )

        self.assertEqual(len(merged), 2)
        board_heights = {
            int(item["board_context_summary"]["board_height"])
            for item in merged
        }
        self.assertEqual(board_heights, {3, 5})

    def test_load_memory_bundle_from_stores_recomputes_merged_retrieval_summary_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            hot_db_path = Path(tmpdir) / "hot.sqlite3"
            archive_db_path = Path(tmpdir) / "archive.sqlite3"
            initialize_memory_schema(hot_db_path)
            initialize_memory_schema(archive_db_path)
            for db_path, step_index, action_name, frame_signature in (
                (hot_db_path, 1, "ACTION1", "sig-hot"),
                (archive_db_path, 2, "ACTION2", "sig-archive"),
            ):
                record_transition(
                    db_path,
                    EpisodeRecord(
                        game_id="ls20",
                        step_index=step_index,
                        action_name=action_name,
                        outcome="progress",
                        progress_delta=1,
                        game_state="ACTIVE",
                        prior_game_state="ACTIVE",
                        levels_completed=1,
                        prior_frame_signature=frame_signature,
                        next_game_state="ACTIVE",
                        next_frame_signature=f"{frame_signature}-next",
                        prior_legal_actions=["ACTION1", "ACTION2"],
                        stagnation_before=0,
                        stagnation_after=0,
                        decision_subgoal="advance-level",
                        decision_source="symbolic",
                    ),
                )

            bundle = load_memory_bundle_from_stores(
                hot_db_path=hot_db_path,
                archive_db_path=archive_db_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature="sig-hot",
                subgoal="advance-level",
                legal_actions=["ACTION1", "ACTION2"],
                board_context=None,
                last_action="ACTION1",
            )

            self.assertEqual(
                bundle["retrieval_summary"]["recent_record_count"],
                len(bundle["recent_records"]),
            )
            self.assertEqual(
                bundle["retrieval_summary"]["episode_summary_count"],
                len(bundle["episode_summaries"]),
            )

    def test_load_memory_bundle_from_stores_merges_trained_prior_buckets_across_hot_and_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            hot_db_path = Path(tmpdir) / "hot.sqlite3"
            archive_db_path = Path(tmpdir) / "archive.sqlite3"
            initialize_memory_schema(hot_db_path)
            initialize_memory_schema(archive_db_path)
            upsert_training_counts(
                hot_db_path,
                [
                    {
                        "scope": "subgoal",
                        "game_id": "ls20",
                        "subgoal": "inspect-active-board",
                        "action_name": "ACTION6",
                        "count": 11,
                    }
                ],
            )
            upsert_training_counts(
                archive_db_path,
                [
                    {
                        "scope": "state",
                        "game_id": "ls20",
                        "state": "ACTIVE",
                        "action_name": "ACTION1",
                        "count": 8,
                    }
                ],
            )

            bundle = load_memory_bundle_from_stores(
                hot_db_path=hot_db_path,
                archive_db_path=archive_db_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature="sig-missing",
                subgoal="inspect-active-board",
                legal_actions=["ACTION1", "ACTION6"],
                board_context=None,
                last_action=None,
            )

            self.assertEqual(bundle["trained_prior_bucket"], {"ACTION6": 11, "ACTION1": 8})
            self.assertEqual(bundle["recommended_actions"][0], "ACTION6")

    def test_load_memory_bundle_from_stores_recomputes_working_memory_from_merged_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            hot_db_path = Path(tmpdir) / "hot.sqlite3"
            archive_db_path = Path(tmpdir) / "archive.sqlite3"
            initialize_memory_schema(hot_db_path)
            initialize_memory_schema(archive_db_path)
            record_transition(
                hot_db_path,
                EpisodeRecord(
                    game_id="ls20",
                    step_index=10,
                    action_name="ACTION1",
                    outcome="stalled",
                    progress_delta=0,
                    game_state="ACTIVE",
                    prior_game_state="ACTIVE",
                    levels_completed=0,
                    prior_frame_signature="sig-hot",
                    next_game_state="ACTIVE",
                    next_frame_signature="sig-hot-next",
                    prior_legal_actions=["ACTION1", "ACTION2", "ACTION3"],
                    stagnation_before=1,
                    stagnation_after=2,
                    decision_subgoal="break-stall",
                    decision_source="symbolic",
                ),
            )
            record_transition(
                archive_db_path,
                EpisodeRecord(
                    game_id="ls20",
                    step_index=1,
                    action_name="ACTION3",
                    outcome="failed",
                    progress_delta=0,
                    game_state="ACTIVE",
                    prior_game_state="ACTIVE",
                    levels_completed=0,
                    prior_frame_signature="sig-archive",
                    next_game_state="GAME_OVER",
                    next_frame_signature="sig-archive-next",
                    prior_legal_actions=["ACTION1", "ACTION2", "ACTION3"],
                    stagnation_before=2,
                    stagnation_after=3,
                    decision_subgoal="advance-level",
                    decision_source="symbolic",
                ),
            )

            bundle = load_memory_bundle_from_stores(
                hot_db_path=hot_db_path,
                archive_db_path=archive_db_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature="sig-live",
                subgoal="break-stall",
                legal_actions=["ACTION1", "ACTION2", "ACTION3"],
                board_context=None,
                last_action="ACTION1",
            )

            working_memory = bundle["working_memory"]
            self.assertEqual(
                working_memory["recent_action_window"][:2],
                ["ACTION1", "ACTION3"],
            )
            self.assertEqual(
                working_memory["recent_actions"][:2],
                ["ACTION1", "ACTION3"],
            )
            self.assertEqual(
                [row["action_name"] for row in working_memory["open_failure_modes"]],
                ["ACTION1", "ACTION3"],
            )
            self.assertEqual(
                [row["action_name"] for row in working_memory["recent_failures"]],
                ["ACTION1", "ACTION3"],
            )
            self.assertEqual(
                working_memory["recent_failures"][0]["subgoal"],
                "break-stall",
            )
            self.assertEqual(
                working_memory["failure_streaks"][:2],
                [
                    {"action_name": "ACTION1", "count": 1},
                    {"action_name": "ACTION3", "count": 1},
                ],
            )
            self.assertEqual(
                working_memory["semantic_focus_actions"][:2],
                ["ACTION1", "ACTION3"],
            )
            self.assertEqual(
                working_memory["procedural_focus_actions"][:2],
                ["ACTION1", "ACTION3"],
            )
            self.assertIn("current_board_context", working_memory)

    def test_load_memory_bundle_from_stores_touches_only_surviving_cross_store_memories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            hot_db_path = Path(tmpdir) / "hot.sqlite3"
            archive_db_path = Path(tmpdir) / "archive.sqlite3"
            initialize_memory_schema(hot_db_path)
            initialize_memory_schema(archive_db_path)
            hot_record = EpisodeRecord(
                game_id="ls20",
                step_index=10,
                action_name="ACTION2",
                outcome="progress",
                progress_delta=1,
                game_state="ACTIVE",
                prior_game_state="ACTIVE",
                levels_completed=1,
                prior_frame_signature="sig-shared",
                next_game_state="ACTIVE",
                next_frame_signature="sig-shared-next",
                prior_legal_actions=["ACTION1", "ACTION2"],
                stagnation_before=0,
                stagnation_after=0,
                decision_subgoal="advance-level",
                decision_source="symbolic",
            )
            archive_record = EpisodeRecord(
                game_id="ls20",
                step_index=1,
                action_name="ACTION2",
                outcome="progress",
                progress_delta=1,
                game_state="ACTIVE",
                prior_game_state="ACTIVE",
                levels_completed=1,
                prior_frame_signature="sig-shared",
                next_game_state="ACTIVE",
                next_frame_signature="sig-shared-next",
                prior_legal_actions=["ACTION1", "ACTION2"],
                stagnation_before=0,
                stagnation_after=0,
                decision_subgoal="advance-level",
                decision_source="symbolic",
            )
            record_transition(hot_db_path, hot_record)
            record_transition(archive_db_path, archive_record)

            bundle = load_memory_bundle_from_stores(
                hot_db_path=hot_db_path,
                archive_db_path=archive_db_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature="sig-shared",
                subgoal="advance-level",
                legal_actions=["ACTION1", "ACTION2"],
                board_context=None,
                last_action="ACTION1",
                retrieval_step=27,
            )

            self.assertTrue(bundle["semantic_memories"])
            self.assertTrue(bundle["procedural_memories"])
            with closing(sqlite3.connect(hot_db_path)) as connection:
                hot_semantic = connection.execute(
                    "SELECT access_count, last_access_step FROM live_semantic_memory"
                ).fetchall()
                hot_procedural = connection.execute(
                    "SELECT access_count, last_access_step FROM live_procedural_memory"
                ).fetchall()
            with closing(sqlite3.connect(archive_db_path)) as connection:
                archive_semantic = connection.execute(
                    "SELECT access_count, last_access_step FROM live_semantic_memory"
                ).fetchall()
                archive_procedural = connection.execute(
                    "SELECT access_count, last_access_step FROM live_procedural_memory"
                ).fetchall()
            self.assertTrue(all(int(row[0] or 0) == 1 for row in hot_semantic))
            self.assertTrue(all(int(row[1] or 0) == 27 for row in hot_semantic))
            self.assertTrue(all(int(row[0] or 0) == 1 for row in hot_procedural))
            self.assertTrue(all(int(row[1] or 0) == 27 for row in hot_procedural))
            self.assertTrue(all(int(row[0] or 0) == 0 for row in archive_semantic))
            self.assertTrue(all(int(row[1] or 0) == 0 for row in archive_semantic))
            self.assertTrue(all(int(row[0] or 0) == 0 for row in archive_procedural))
            self.assertTrue(all(int(row[1] or 0) == 0 for row in archive_procedural))

    def test_load_memory_bundle_considers_older_exact_semantic_match_beyond_recent_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            initialize_memory_schema(db_path)
            for step_index in range(1, 81):
                if step_index == 1:
                    frame_signature = "sig-target"
                    subgoal = "advance-level"
                    board_height = 3
                    board_width = 3
                    board_active_cells = 4
                    board_source_attr = "frame"
                else:
                    frame_signature = f"sig-{step_index}"
                    subgoal = "other-subgoal"
                    board_height = 7
                    board_width = 7
                    board_active_cells = 1
                    board_source_attr = "grid"
                record_transition(
                    db_path,
                    EpisodeRecord(
                        game_id="ls20",
                        step_index=step_index,
                        action_name="ACTION2",
                        outcome="progress",
                        progress_delta=1,
                        game_state="ACTIVE",
                        prior_game_state="ACTIVE",
                        levels_completed=1,
                        prior_frame_signature=frame_signature,
                        next_game_state="ACTIVE",
                        next_frame_signature=f"{frame_signature}-next",
                        prior_legal_actions=["ACTION1", "ACTION2"],
                        stagnation_before=0,
                        stagnation_after=0,
                        decision_subgoal=subgoal,
                        decision_source="symbolic",
                        prior_board_height=board_height,
                        prior_board_width=board_width,
                        prior_board_active_cells=board_active_cells,
                        prior_board_source_attr=board_source_attr,
                    ),
                )

            bundle = load_memory_bundle(
                db_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature="sig-now",
                subgoal="advance-level",
                legal_actions=["ACTION1", "ACTION2"],
                board_context={
                    "board_height": 3,
                    "board_width": 3,
                    "board_active_cells": 4,
                    "board_source_attr": "frame",
                },
                last_action="ACTION1",
            )

            self.assertTrue(bundle["semantic_memories"])
            top_memory = bundle["semantic_memories"][0]
            self.assertEqual(top_memory["subgoal"], "advance-level")
            self.assertEqual(top_memory["board_context_summary"]["board_height"], 3)
            self.assertEqual(top_memory["board_context_summary"]["board_width"], 3)
            self.assertEqual(top_memory["board_context_summary"]["board_active_cells"], 4)

    def test_load_memory_bundle_considers_older_exact_procedural_match_beyond_recent_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            initialize_memory_schema(db_path)
            previous_record = None
            for step_index in range(1, 81):
                if step_index == 1:
                    action_name = "ACTION1"
                    subgoal = "advance-level"
                else:
                    action_name = "ACTION2"
                    subgoal = "other-subgoal"
                record = EpisodeRecord(
                    game_id="ls20",
                    step_index=step_index,
                    action_name=action_name,
                    outcome="progress",
                    progress_delta=1,
                    game_state="ACTIVE",
                    prior_game_state="ACTIVE",
                    levels_completed=1,
                    prior_frame_signature=f"sig-{step_index}",
                    next_game_state="ACTIVE",
                    next_frame_signature=f"sig-{step_index}-next",
                    prior_legal_actions=["ACTION1", "ACTION2"],
                    stagnation_before=0,
                    stagnation_after=0,
                    decision_subgoal=subgoal,
                    decision_source="symbolic",
                )
                record_transition(db_path, record)
                previous_record = record
            self.assertIsNotNone(previous_record)

            bundle = load_memory_bundle(
                db_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature="sig-now",
                subgoal="advance-level",
                legal_actions=["ACTION1", "ACTION2"],
                last_action="",
            )

            self.assertTrue(bundle["procedural_memories"])
            self.assertEqual(bundle["procedural_memories"][0]["subgoal"], "advance-level")

    def test_load_memory_bundle_from_stores_tolerates_archive_without_new_memory_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_db_path = Path(tmpdir) / "archive.sqlite3"
            initialize_schema(archive_db_path)

            bundle = load_memory_bundle_from_stores(
                hot_db_path=None,
                archive_db_path=archive_db_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature="sig-archive-only",
                subgoal="advance-level",
                legal_actions=["ACTION1", "ACTION2"],
                board_context=None,
                last_action="ACTION1",
            )

            self.assertEqual(bundle["semantic_memories"], [])
            self.assertEqual(bundle["procedural_memories"], [])

    def test_load_memory_bundle_returns_abstract_memories_for_unknown_game_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            initialize_memory_schema(db_path)
            upsert_abstract_memory_rows(
                db_path,
                [
                    {
                        "subgoal": "advance-level",
                        "state": "ACTIVE",
                        "board_height": 2,
                        "board_width": 2,
                        "board_active_cells": 4,
                        "board_source_attr": "frame",
                        "motif_tags": [
                            "axis_horizontal",
                            "axis_vertical",
                            "mirror_alignment_candidate",
                            "multi_piece_level",
                        ],
                        "action_name": "ACTION2",
                        "lead_in_actions": ["ACTION6"],
                        "lesson_text": "Transferred mirror alignment move",
                        "salience": 2.0,
                        "confidence": 0.8,
                        "progress": 3,
                        "failed": 0,
                        "stalled": 0,
                        "total": 3,
                        "last_outcome": "progress",
                        "last_step_index": 8,
                        "source_games": ["ar25", "bp35"],
                    }
                ],
            )

            bundle = load_memory_bundle(
                db_path,
                game_id="zz99",
                state="ACTIVE",
                frame_signature="sig-unknown",
                subgoal="advance-level",
                legal_actions=["ACTION1", "ACTION2"],
                board_context={
                    "board_height": 2,
                    "board_width": 2,
                    "board_active_cells": 4,
                    "board_source_attr": "frame",
                    "board_default_probe": {"x": 1, "y": 1},
                },
                board_motif_tags=[
                    "axis_horizontal",
                    "axis_vertical",
                    "mirror_alignment_candidate",
                    "multi_piece_level",
                ],
            )

            self.assertTrue(bundle["abstract_memories"])
            self.assertEqual(bundle["recommended_actions"][0], "ACTION2")
            self.assertIn("abstraction-transfer", bundle["retrieval_summary"]["top_recall_sources"])
            self.assertEqual(
                bundle["working_memory"]["abstract_focus_actions"][0]["action_name"],
                "ACTION2",
            )

    def test_known_game_priors_stay_ahead_of_supplementary_abstractions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.sqlite3"
            initialize_memory_schema(db_path)
            upsert_training_counts(
                db_path,
                [
                    {
                        "scope": "state",
                        "game_id": "ls20",
                        "state": "ACTIVE",
                        "action_name": "ACTION1",
                        "count": 12,
                    }
                ],
            )
            upsert_abstract_memory_rows(
                db_path,
                [
                    {
                        "subgoal": "advance-level",
                        "state": "ACTIVE",
                        "board_height": 2,
                        "board_width": 2,
                        "board_active_cells": 4,
                        "board_source_attr": "frame",
                        "motif_tags": ["mirror_alignment_candidate", "multi_piece_level"],
                        "action_name": "ACTION2",
                        "lead_in_actions": ["ACTION6"],
                        "lesson_text": "Transferred mirror alignment move",
                        "salience": 2.0,
                        "confidence": 0.8,
                        "progress": 3,
                        "failed": 0,
                        "stalled": 0,
                        "total": 3,
                        "last_outcome": "progress",
                        "last_step_index": 8,
                        "source_games": ["ar25", "bp35"],
                    }
                ],
            )

            bundle = load_memory_bundle(
                db_path,
                game_id="ls20",
                state="ACTIVE",
                frame_signature="sig-known",
                subgoal="advance-level",
                legal_actions=["ACTION1", "ACTION2"],
                board_context={
                    "board_height": 2,
                    "board_width": 2,
                    "board_active_cells": 4,
                    "board_source_attr": "frame",
                    "board_default_probe": {"x": 1, "y": 1},
                },
                board_motif_tags=["mirror_alignment_candidate", "multi_piece_level"],
            )

            self.assertEqual(bundle["trained_prior_bucket"]["ACTION1"], 12)
            self.assertEqual(bundle["recommended_actions"][0], "ACTION1")


if __name__ == "__main__":
    unittest.main()
