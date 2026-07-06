from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from agi_core.config import DEFAULT_RESET_STATE_NAMES, DEFAULT_WIN_STATE_NAME
from agi_core.image_input import (
    active_cell_count,
    board_context_match_score,
    board_context_summary,
    default_click_target,
    extract_board_motif_tags,
)
from agi_core.storage import (
    import_legacy_learning_json,
    initialize_memory_schema,
    load_best_prior_bucket,
    load_memory_bundle,
    load_memory_bundle_from_stores,
    record_transition as persist_transition,
)
from agi_core.storage.priors_sqlite import game_id_aliases
from agi_core.storage.memory_sqlite import (
    _compact_trained_prior_focus,
    _derive_recommended_actions,
    _derive_suppressed_actions,
)
from agi_core.types import ActionProposal, EpisodeRecord, GoalState, MemoryBundle, Observation


class HyperonBridge:
    """Defensive Hyperon wrapper with a deterministic shadow fact store."""

    def __init__(
        self,
        rules_path: str | Path | None = None,
        learning_path: str | Path | None = None,
        memory_db_path: str | Path | None = None,
        archive_memory_db_path: str | Path | None = None,
        trained_priors_db_path: str | Path | None = None,
        project_root: str | Path | None = None,
        reset_state_names: tuple[str, ...] = ("NOT_PLAYED", "GAME_OVER"),
        win_state_name: str = "WIN",
        enabled: bool = True,
        skip_probe: bool = False,
        python_executable: str | Path | None = None,
        probe_timeout_seconds: float = 5.0,
    ) -> None:
        self.rules_path = Path(rules_path) if rules_path is not None else None
        self.learning_path = Path(learning_path) if learning_path is not None else None
        self.memory_db_path = Path(memory_db_path) if memory_db_path is not None else None
        self.archive_memory_db_path = (
            Path(archive_memory_db_path) if archive_memory_db_path is not None else None
        )
        self.trained_priors_db_path = (
            Path(trained_priors_db_path) if trained_priors_db_path is not None else None
        )
        self.project_root = self._resolve_project_root(project_root, self.rules_path)
        self._reset_state_names = {str(name or "").upper() for name in reset_state_names}
        self._win_state_name = str(win_state_name or "").upper()
        self.enabled = enabled
        self.skip_probe = skip_probe
        self.python_executable = str(python_executable or sys.executable)
        self.probe_timeout_seconds = probe_timeout_seconds
        self.rules_text = self._load_rules_text()
        self.backend = self._try_import_hyperon()
        self.available = False
        self.facts: dict[str, Any] = {}
        self.last_error: str | None = None
        self.learned_knowledge: dict[str, Any] = self._load_learned_knowledge()
        self._ingested_record_keys: set[tuple[Any, ...]] = set()
        self._probe_attempted = False
        self._probe_ok = False
        self._runtime: Any = None
        self._trained_priors_error: str | None = None
        self._memory_error: str | None = None
        self._memory_store_available = False
        self._memory_write_available = False
        self._unpersisted_records: dict[tuple[Any, ...], EpisodeRecord] = {}
        self._last_observation_step_by_game: dict[str, int] = {}
        self._last_observation_state_by_game: dict[str, str] = {}
        self._current_memory_bundle = MemoryBundle()
        self._rules_loaded = False
        self._last_fact_program: str | None = None
        self._initialize_memory_store()

    def bind_game(self, game_id: str) -> None:
        game_key = str(game_id or "")
        stale_games = [
            record_game_id
            for record_game_id in self._last_observation_step_by_game
            if record_game_id != game_key
        ]
        for stale_game_id in stale_games:
            self._last_observation_step_by_game.pop(stale_game_id, None)
            self._last_observation_state_by_game.pop(stale_game_id, None)

    def sync(self, observation: Observation, memory: Any, goals: list[GoalState]) -> None:
        self._note_observation_boundary(observation)
        runtime_working = self._runtime_working_memory_seed(memory)
        active_subgoal = str(runtime_working.get("active_subgoal") or "").strip() or None
        bundle = self._load_memory_bundle(
            observation=observation,
            subgoal=active_subgoal,
            legal_actions=list(observation.legal_actions),
            memory=memory,
        )
        self._current_memory_bundle = MemoryBundle(**bundle)
        image_input = observation.image_input
        board_active_cells = int(
            observation.derived_features.get("board_active_cells", active_cell_count(image_input)) or 0
        )
        board_default_probe = dict(
            observation.derived_features.get(
                "board_default_probe",
                default_click_target(image_input) if image_input is not None else {},
            )
            or {}
        )
        board_source_attr = observation.derived_features.get(
            "board_source_attr",
            image_input.source_attr if image_input is not None else None,
        )
        self.facts = {
            "episode": {
                "game_id": observation.game_id,
                "step_index": observation.step_index,
                "game_state": observation.game_state,
                "last_action": observation.last_action,
                "levels_completed": observation.levels_completed,
                "legal_actions": list(observation.legal_actions),
                "frame_signature": observation.derived_features.get("frame_signature"),
                "has_image_input": image_input is not None,
                "board_height": image_input.height if image_input is not None else 0,
                "board_width": image_input.width if image_input is not None else 0,
                "board_active_cells": board_active_cells,
                "board_default_probe": board_default_probe,
                "board_source_attr": board_source_attr,
            },
            "goals": [
                {
                    "name": goal.name,
                    "target_state": goal.target_state,
                    "priority": goal.priority,
                    "metadata": dict(goal.metadata),
                }
                for goal in goals
            ],
            "legality": [
                {"action_name": action_name}
                for action_name in observation.legal_actions
            ],
            "memory": memory.summarize_for_hyperon(),
            "memory_bundle": bundle,
            "learned": self._memory_bundle_as_learned(
                bundle,
                subgoal_name=active_subgoal,
            ),
            "trained_prior_bucket": dict(bundle.get("trained_prior_bucket", {})),
            "proposals": [],
        }
        self._refresh_runtime()

    def current_memory_bundle(self) -> MemoryBundle:
        return self._current_memory_bundle

    def contextual_memory_bundle(
        self,
        observation: Observation,
        subgoal: str | None,
        legal_actions: list[str],
    ) -> MemoryBundle:
        if not subgoal:
            return self._current_memory_bundle
        bundle = self._load_memory_bundle(
            observation=observation,
            subgoal=subgoal,
            legal_actions=legal_actions,
            memory=None,
            touch_retrieved=False,
        )
        return MemoryBundle(**bundle)

    def record_transition(self, record: EpisodeRecord) -> None:
        record_key = self._record_key(record)
        self._unpersisted_records[record_key] = record
        if not self._ensure_memory_write_available():
            return
        try:
            self._flush_unpersisted_records()
            self._memory_error = None
        except Exception as exc:
            self._memory_store_available = True
            self._memory_write_available = False
            self._memory_error = repr(exc)

    def _flush_unpersisted_records(self) -> None:
        if self.memory_db_path is None:
            return
        for pending_key, pending_record in list(self._unpersisted_records.items()):
            persist_transition(self.memory_db_path, pending_record)
            self._unpersisted_records.pop(pending_key, None)

    def update_proposals(self, proposals: list[ActionProposal]) -> None:
        self.facts["proposals"] = [
            {
                "action_name": proposal.action_name,
                "source": proposal.source,
                "confidence": proposal.confidence,
                "expected_outcome": proposal.expected_outcome,
            }
            for proposal in proposals
        ]
        if not self.available or self._runtime is None:
            return
        try:
            self._run_program(self._build_proposal_fact_program())
            # Force the next sync to refresh base facts so stale proposal facts
            # cannot survive across turns in a reused runtime.
            self._last_fact_program = None
        except Exception as exc:
            self._disable_runtime(f"failed to sync proposal facts: {exc}")

    def run_rules(self, query: str) -> list[dict[str, Any]]:
        if self.available and self._runtime is not None:
            try:
                return self._run_rules_with_runtime(query)
            except Exception as exc:
                self._disable_runtime(f"hyperon query `{query}` failed: {exc}")
        return self._run_shadow_rules(query)

    def _run_rules_with_runtime(self, query: str) -> list[dict[str, Any]]:
        episode = self.facts.get("episode", {})
        memory = self.facts.get("memory", {})
        query_name, query_arg = self._parse_query(query)
        legal_actions = list(episode.get("legal_actions", []))
        if query_name in {"reset-needed", "goal-hints"} and self._uses_custom_terminal_states():
            return self._run_shadow_rules(query)
        if query_name == "reset-needed":
            state_name = self._symbol(episode.get("game_state"))
            if self._evaluate_scalar(f"!(reset-required {state_name})") is True:
                return [
                    {
                        "action_name": "RESET",
                        "confidence": 0.99,
                        "reason": "hyperon reset guard",
                    }
                ]
            return []
        if query_name == "stagnation":
            return self._run_shadow_rules(query)
        if query_name == "goal-hints":
            state_name = self._symbol(episode.get("game_state"))
            subgoal = None
            if int(memory.get("stagnation_steps", 0) or 0) > 0:
                subgoal = self._evaluate_scalar("!(stagnation-hint True)")
            working_memory_hint = self._working_memory_goal_hint(
                episode=episode,
                memory=memory,
                legal_actions=legal_actions,
            )
            if subgoal is None:
                subgoal = self._evaluate_scalar(f"!(goal-hint {state_name})")
            if (
                working_memory_hint is not None
                and (subgoal is None or str(subgoal) == "advance-level")
            ):
                subgoal = working_memory_hint["subgoal"]
            if (
                (subgoal is None or str(subgoal) == "advance-level")
                and int(episode.get("board_active_cells", 0) or 0) > 0
                and "ACTION6" in legal_actions
            ):
                subgoal = "inspect-active-board"
            if subgoal is None:
                return []
            confidence = 0.95 if episode.get("game_state") == self._win_state_name else 0.72
            if str(subgoal) == "inspect-active-board":
                confidence = 0.74
            elif (
                working_memory_hint is not None
                and str(subgoal) == str(working_memory_hint.get("subgoal") or "")
            ):
                confidence = float(working_memory_hint.get("confidence", confidence) or confidence)
            return [{"subgoal": str(subgoal), "confidence": confidence}]
        if query_name == "action-priors":
            return self._action_priors(
                legal_actions,
                memory,
                learned=self.facts.get("learned", {}),
                current_state=str(episode.get("game_state", "")),
                current_subgoal=query_arg,
                current_frame_signature=str(episode.get("frame_signature", "") or ""),
                base_prior_lookup=self._lookup_symbolic_prior,
            )
        if query_name == "memory-penalties":
            return self._run_shadow_rules(query)
        return self._run_shadow_rules(query)

    def _uses_custom_terminal_states(self) -> bool:
        return (
            self._win_state_name != DEFAULT_WIN_STATE_NAME
            or self._reset_state_names != set(DEFAULT_RESET_STATE_NAMES)
        )

    def _run_shadow_rules(self, query: str) -> list[dict[str, Any]]:
        episode = self.facts.get("episode", {})
        memory = self.facts.get("memory", {})
        query_name, query_arg = self._parse_query(query)
        legal_actions = list(episode.get("legal_actions", []))
        if query_name == "reset-needed":
            if episode.get("game_state") in self._reset_state_names:
                return [
                    {
                        "action_name": "RESET",
                        "confidence": 0.99,
                        "reason": "symbolic reset guard",
                    }
                ]
            return []
        if query_name == "stagnation":
            if memory.get("stagnation_steps", 0) >= 3:
                return [
                    {
                        "flag": "stagnation",
                        "confidence": 0.74,
                        "reason": "recent actions failed to make progress",
                    }
                ]
            return []
        if query_name == "goal-hints":
            if episode.get("game_state") == self._win_state_name:
                return [{"subgoal": "hold-win", "confidence": 0.95}]
            if memory.get("stagnation_steps", 0):
                return [{"subgoal": "break-stall", "confidence": 0.72}]
            working_memory_hint = self._working_memory_goal_hint(
                episode=episode,
                memory=memory,
                legal_actions=legal_actions,
            )
            if (
                working_memory_hint is not None
                and str(working_memory_hint.get("subgoal") or "") != "advance-level"
            ):
                return [working_memory_hint]
            if (
                int(episode.get("board_active_cells", 0) or 0) > 0
                and "ACTION6" in legal_actions
            ):
                return [{"subgoal": "inspect-active-board", "confidence": 0.74}]
            if working_memory_hint is not None:
                return [working_memory_hint]
            return [{"subgoal": "advance-level", "confidence": 0.70}]
        if query_name == "action-priors":
            return self._action_priors(
                legal_actions,
                memory,
                learned=self.facts.get("learned", {}),
                current_state=str(episode.get("game_state", "")),
                current_subgoal=query_arg,
                current_frame_signature=str(episode.get("frame_signature", "") or ""),
            )
        if query_name == "memory-penalties":
            failure_streaks = memory.get("failure_streaks", {})
            return [
                {"action_name": action_name, "penalty": value}
                for action_name, value in failure_streaks.items()
            ]
        if query_name == "game-action-priors":
            game_id = str(episode.get("game_id", "") or "")
            board_type = str(episode.get("board_type", "") or "")
            base = self._action_priors(
                legal_actions, memory,
                learned=self.facts.get("learned", {}),
                current_state=str(episode.get("game_state", "")),
                current_subgoal=query_arg,
                current_frame_signature=str(episode.get("frame_signature", "") or ""),
            )
            # Override with game-specific priors.
            _GAME_PRIORS: dict[str, dict[str, float]] = {
                "ar25": {"ACTION1": 0.55, "ACTION2": 0.78, "ACTION3": 0.60,
                         "ACTION4": 0.72, "ACTION5": 0.45, "ACTION6": 0.30},
            }
            if game_id in _GAME_PRIORS:
                for action, conf in _GAME_PRIORS[game_id].items():
                    base = [p for p in base if p.get("action_name") != action]
                    base.append({"action_name": action, "confidence": conf, "reason": f"game-prior:{game_id}"})
            # Override with board-type-specific priors.
            _BOARD_PRIORS: dict[str, dict[str, float]] = {
                "puzzle": {"ACTION1": 0.30, "ACTION2": 0.35, "ACTION3": 0.30,
                           "ACTION4": 0.30, "ACTION5": 0.80, "ACTION6": 0.70},
                "platformer": {"ACTION1": 0.60, "ACTION2": 0.65, "ACTION3": 0.60,
                               "ACTION4": 0.65, "ACTION5": 0.40, "ACTION6": 0.35},
            }
            if board_type in _BOARD_PRIORS:
                for action, conf in _BOARD_PRIORS[board_type].items():
                    base = [p for p in base if p.get("action_name") != action]
                    base.append({"action_name": action, "confidence": conf, "reason": f"board-prior:{board_type}"})
            return base
        if query_name == "game-goal-hints":
            board_type = str(episode.get("board_type", "") or "unknown")
            # Board-type-specific goal hints.
            _BOARD_GOAL_HINTS: dict[str, str] = {
                "puzzle": "explore-board",
                "platformer": "navigate-to-goal",
            }
            if board_type in _BOARD_GOAL_HINTS:
                return [{"subgoal": _BOARD_GOAL_HINTS[board_type], "confidence": 0.75,
                         "reason": f"board-goal-hint:{board_type}"}]
            # Fall back to the standard goal-hints logic.
            return self._run_shadow_rules("goal-hints")
        if query_name == "game-stagnation":
            board_type = str(episode.get("board_type", "") or "unknown")
            stagnation_steps = int(memory.get("stagnation_steps", 0) or 0)
            _BOARD_STAGNATION_THRESHOLD: dict[str, int] = {
                "puzzle": 2,
                "platformer": 5,
                "unknown": 3,
            }
            threshold = _BOARD_STAGNATION_THRESHOLD.get(board_type, 3)
            if stagnation_steps >= threshold:
                return [{"flag": "stagnation", "confidence": 0.74,
                         "reason": f"board-stagnation:{board_type}"}]
            return []
        if query_name == "game-reset-needed":
            game_id = str(episode.get("game_id", "") or "")
            state = str(episode.get("game_state", "") or "")
            game_id_aliases = {"ar25": "ar25"}
            # Override reset behavior for specific games.
            _GAME_RESET_NEEDED: dict[str, set[str]] = {
                "ar25": {"GAME_OVER", "NOT_PLAYED"},
            }
            canonical = game_id_aliases.get(game_id, game_id)
            if canonical in _GAME_RESET_NEEDED:
                if state in _GAME_RESET_NEEDED[canonical]:
                    return [{"action_name": "RESET", "confidence": 0.99,
                             "reason": f"game-reset:{game_id}:{state}"}]
                return []
            # Fall back to standard reset-needed logic.
            return self._run_shadow_rules("reset-needed")
        if query_name == "beam-strategy":
            game_id = str(episode.get("game_id", "") or "")
            level = int(episode.get("levels_completed", 0) or 0) + 1
            # AR25 per-level beam direction preferences.
            _BEAM_STRATEGY: dict[str, dict[int, list[tuple[str, float]]]] = {
                "ar25": {
                    1: [("down", 0.75), ("right", 0.65)],
                    2: [("right", 0.60), ("down", 0.55)],
                },
            }
            game_levels = _BEAM_STRATEGY.get(game_id, {})
            if level in game_levels:
                return [{"direction": d, "confidence": c} for d, c in game_levels[level]]
            return []
        if query_name == "board-reset-threshold":
            board_type = str(episode.get("board_type", "") or "unknown")
            _BOARD_RESET: dict[str, int] = {"puzzle": 8, "platformer": 12, "unknown": 6}
            return [{"threshold": _BOARD_RESET.get(board_type, 6), "board_type": board_type}]
        if query_name == "board-explore-bias":
            board_type = str(episode.get("board_type", "") or "unknown")
            _BOARD_BIAS: dict[str, str] = {"puzzle": "vertical", "platformer": "horizontal", "unknown": "horizontal"}
            return [{"axis": _BOARD_BIAS.get(board_type, "horizontal"), "board_type": board_type}]
        if query_name == "board-interact-range":
            board_type = str(episode.get("board_type", "") or "unknown")
            _BOARD_RANGE: dict[str, int] = {"puzzle": 4, "platformer": 2, "unknown": 3}
            return [{"range": _BOARD_RANGE.get(board_type, 3), "board_type": board_type}]
        if query_name == "board-move-bias":
            board_type = str(episode.get("board_type", "") or "unknown")
            _BOARD_MOVE: dict[str, str] = {"puzzle": "down", "platformer": "right", "unknown": "right"}
            return [{"direction": _BOARD_MOVE.get(board_type, "right"), "board_type": board_type}]
        if query_name == "board-beam-priority":
            board_type = str(episode.get("board_type", "") or "unknown")
            _BOARD_BEAM: dict[str, str] = {"puzzle": "high", "platformer": "low", "unknown": "medium"}
            return [{"priority": _BOARD_BEAM.get(board_type, "medium"), "board_type": board_type}]
        if query_name == "board-switch-urgency":
            board_type = str(episode.get("board_type", "") or "unknown")
            _BOARD_SWITCH: dict[str, str] = {"puzzle": "high", "platformer": "low", "unknown": "medium"}
            return [{"urgency": _BOARD_SWITCH.get(board_type, "medium"), "board_type": board_type}]
        if query_name == "board-path-preference":
            board_type = str(episode.get("board_type", "") or "unknown")
            _BOARD_PATH: dict[str, str] = {"puzzle": "direct", "platformer": "safe", "unknown": "balanced"}
            return [{"preference": _BOARD_PATH.get(board_type, "balanced"), "board_type": board_type}]
        if query_name == "board-retry-tolerance":
            board_type = str(episode.get("board_type", "") or "unknown")
            _BOARD_RETRY: dict[str, int] = {"puzzle": 2, "platformer": 5, "unknown": 3}
            return [{"tolerance": _BOARD_RETRY.get(board_type, 3), "board_type": board_type}]
        if query_name == "board-goal-proximity":
            board_type = str(episode.get("board_type", "") or "unknown")
            _BOARD_PROX: dict[str, int] = {"puzzle": 1, "platformer": 3, "unknown": 2}
            return [{"proximity": _BOARD_PROX.get(board_type, 2), "board_type": board_type}]
        if query_name == "board-entity-count":
            board_type = str(episode.get("board_type", "") or "unknown")
            _BOARD_ENT: dict[str, str] = {"puzzle": "multiple", "platformer": "single", "unknown": "single"}
            return [{"count": _BOARD_ENT.get(board_type, "single"), "board_type": board_type}]
        if query_name == "board-gravity-enabled":
            board_type = str(episode.get("board_type", "") or "unknown")
            _BOARD_GRV: dict[str, str] = {"puzzle": "false", "platformer": "true", "unknown": "false"}
            return [{"gravity": _BOARD_GRV.get(board_type, "false"), "board_type": board_type}]
        if query_name == "board-horizontal-scroll":
            board_type = str(episode.get("board_type", "") or "unknown")
            _BOARD_SCR: dict[str, str] = {"puzzle": "false", "platformer": "true", "unknown": "false"}
            return [{"scroll": _BOARD_SCR.get(board_type, "false"), "board_type": board_type}]
        if query_name == "board-collision-type":
            board_type = str(episode.get("board_type", "") or "unknown")
            _BOARD_COL: dict[str, str] = {"puzzle": "solid", "platformer": "solid", "unknown": "solid"}
            return [{"collision": _BOARD_COL.get(board_type, "solid"), "board_type": board_type}]
        if query_name == "board-exit-type":
            board_type = str(episode.get("board_type", "") or "unknown")
            _BOARD_EXT: dict[str, str] = {"puzzle": "action", "platformer": "position", "unknown": "position"}
            return [{"exit": _BOARD_EXT.get(board_type, "position"), "board_type": board_type}]
        return []

    def _working_memory_goal_hint(
        self,
        *,
        episode: dict[str, Any],
        memory: dict[str, Any],
        legal_actions: list[str],
    ) -> dict[str, Any] | None:
        working_memory = dict(memory.get("working_memory", {}) or {})
        if not working_memory:
            return None
        active_subgoal = str(working_memory.get("active_subgoal") or "")
        if active_subgoal == "inspect-active-board":
            if (
                int(episode.get("board_active_cells", 0) or 0) > 0
                and "ACTION6" in legal_actions
            ):
                return {"subgoal": "inspect-active-board", "confidence": 0.76}
        elif active_subgoal in {"break-stall", "advance-level"}:
            return {"subgoal": active_subgoal, "confidence": 0.73}
        if working_memory.get("recent_failures") or working_memory.get("open_failure_modes"):
            return {"subgoal": "break-stall", "confidence": 0.73}
        if working_memory.get("failure_streaks"):
            return {"subgoal": "break-stall", "confidence": 0.72}
        if (
            working_memory.get("semantic_focus_actions")
            or working_memory.get("procedural_focus_actions")
        ):
            return {"subgoal": "advance-level", "confidence": 0.71}
        return None

    def _action_priors(
        self,
        legal_actions: list[str],
        memory: dict[str, Any],
        learned: dict[str, Any] | None = None,
        current_state: str = "",
        current_subgoal: str | None = None,
        current_frame_signature: str = "",
        base_prior_lookup: Callable[[str], float | None] | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        success_counts = memory.get("success_counts", {})
        failure_streaks = memory.get("failure_streaks", {})
        bundle = self._load_contextual_bundle(
            current_state=current_state,
            current_frame_signature=current_frame_signature,
            current_subgoal=current_subgoal,
            legal_actions=legal_actions,
        )
        learned = learned or self._memory_bundle_as_learned(
            bundle,
            subgoal_name=current_subgoal,
        )
        learned_actions = dict(bundle.get("action_stats", {}))
        state_actions = dict(bundle.get("state_action_stats", {}))
        subgoal_actions = dict(bundle.get("subgoal_action_stats", {}))
        basin_actions = dict(bundle.get("basin_action_stats", {}))
        frame_actions = dict(bundle.get("frame_action_stats", {}))
        offline_actions = dict(bundle.get("trained_prior_bucket", {}))
        basin_status = dict(bundle.get("basin_status", {}))
        recommended_actions = list(bundle.get("recommended_actions", []))
        suppressed_actions = set(bundle.get("suppressed_actions", []))
        working_memory = dict(bundle.get("working_memory", {}) or {})
        recent_actions = memory.get("recent_actions", [])
        most_recent = recent_actions[-1] if recent_actions else None
        stagnation_steps = int(memory.get("stagnation_steps", 0) or 0)
        episode_context_attempts = self._episode_context_attempts(
            memory,
            state_name=current_state,
            frame_signature=current_frame_signature,
        )
        episode_distinct_attempts = len(episode_context_attempts)
        basin_hard_exhausted = self._is_episode_basin_hard_exhausted(
            legal_actions=legal_actions,
            stagnation_steps=stagnation_steps,
            basin_status=basin_status,
            episode_context_attempts=episode_context_attempts,
        )
        for index, action_name in enumerate(legal_actions):
            if action_name == "RESET":
                continue
            confidence = None
            if offline_actions:
                confidence = self._offline_prior_confidence(action_name, offline_actions)
            if base_prior_lookup is not None:
                symbolic_prior = base_prior_lookup(action_name)
                if confidence is None:
                    confidence = symbolic_prior
                elif symbolic_prior is not None:
                    confidence = max(confidence, symbolic_prior)
            if confidence is None:
                confidence = 0.72 - (index * 0.04)
            confidence += min(0.08, success_counts.get(action_name, 0) * 0.02)
            confidence -= min(0.25, failure_streaks.get(action_name, 0) * 0.06)
            learned_stats = learned_actions.get(action_name, {})
            learned_success = int(learned_stats.get("progress", 0) or 0)
            learned_failed = int(learned_stats.get("failed", 0) or 0)
            learned_stalled = int(learned_stats.get("stalled", 0) or 0)
            confidence += min(0.10, learned_success * 0.015)
            confidence -= min(0.12, learned_failed * 0.02)
            confidence -= min(0.06, learned_stalled * 0.01)
            confidence += self._contextual_learning_bias(
                basin_actions.get(action_name, {}),
                success_scale=0.04,
                failed_scale=0.02,
                stalled_scale=0.012,
                positive_cap=0.20,
                negative_cap=0.08,
            )
            confidence -= self._frame_context_loop_penalty(
                basin_actions.get(action_name, {}),
                action_name=action_name,
                most_recent_action=most_recent,
                stagnation_steps=stagnation_steps,
            )
            confidence += self._contextual_learning_bias(
                state_actions.get(action_name, {}),
                success_scale=0.03,
                failed_scale=0.02,
                stalled_scale=0.012,
                positive_cap=0.14,
                negative_cap=0.12,
            )
            confidence += self._contextual_learning_bias(
                subgoal_actions.get(action_name, {}),
                success_scale=0.035,
                failed_scale=0.02,
                stalled_scale=0.01,
                positive_cap=0.16,
                negative_cap=0.10,
            )
            confidence += self._contextual_learning_bias(
                frame_actions.get(action_name, {}),
                success_scale=0.012,
                failed_scale=0.008,
                stalled_scale=0.006,
                positive_cap=0.05,
                negative_cap=0.03,
            )
            confidence += self._stagnation_novelty_bonus(
                action_name=action_name,
                stagnation_steps=stagnation_steps,
                episode_context_attempts=episode_context_attempts,
            )
            confidence += self._recommended_action_bonus(action_name, recommended_actions)
            confidence -= self._suppressed_action_penalty(
                action_name=action_name,
                suppressed_actions=suppressed_actions,
                basin_status=basin_status,
            )
            confidence += self._working_memory_focus_bonus(
                action_name=action_name,
                working_memory=working_memory,
            )
            confidence -= self._working_memory_failure_penalty(
                action_name=action_name,
                working_memory=working_memory,
            )
            confidence += self._board_probe_prior_bonus(
                action_name=action_name,
                current_subgoal=current_subgoal,
            )
            confidence -= self._episode_basin_loop_penalty(
                action_name=action_name,
                legal_actions=legal_actions,
                stagnation_steps=stagnation_steps,
                episode_context_attempts=episode_context_attempts,
                episode_distinct_attempts=episode_distinct_attempts,
                basin_status=basin_status,
                basin_hard_exhausted=basin_hard_exhausted,
                most_recent_action=most_recent,
            )
            if stagnation_steps and action_name != most_recent:
                confidence += 0.03
            results.append(
                {
                    "action_name": action_name,
                    "confidence": round(max(0.05, min(confidence, 0.95)), 3),
                    "reason": "hyperon symbolic prior"
                    if base_prior_lookup is not None
                    else "symbolic prior from legal-action ranking",
                }
            )
        results.sort(key=lambda row: float(row.get("confidence", 0.0)), reverse=True)
        return results

    def _board_probe_prior_bonus(
        self,
        *,
        action_name: str,
        current_subgoal: str | None,
    ) -> float:
        if action_name != "ACTION6":
            return 0.0
        episode = self.facts.get("episode", {})
        if int(episode.get("board_active_cells", 0) or 0) <= 0:
            return 0.0
        bonus = 0.06
        subgoal_name = str(current_subgoal or "")
        if subgoal_name == "inspect-active-board":
            bonus += 0.04
        elif subgoal_name == "break-stall":
            bonus += 0.02
        return bonus

    def _offline_prior_confidence(
        self,
        action_name: str,
        bucket: dict[str, Any],
    ) -> float | None:
        values = [int(bucket.get(name, 0) or 0) for name in bucket]
        if not values:
            return None
        peak = max(values)
        if peak <= 0:
            return None
        count = int(bucket.get(action_name, 0) or 0)
        if count <= 0:
            return 0.12
        return round(0.45 + (0.35 * (count / peak)), 3)

    def _contextual_learning_bias(
        self,
        stats: dict[str, Any],
        success_scale: float,
        failed_scale: float,
        stalled_scale: float,
        positive_cap: float,
        negative_cap: float,
    ) -> float:
        if not stats:
            return 0.0
        progress = int(stats.get("progress", 0) or 0)
        failed = int(stats.get("failed", 0) or 0)
        stalled = int(stats.get("stalled", 0) or 0)
        positive = min(positive_cap, progress * success_scale)
        negative = min(negative_cap, (failed * failed_scale) + (stalled * stalled_scale))
        return positive - negative

    def _frame_context_loop_penalty(
        self,
        stats: dict[str, Any],
        *,
        action_name: str,
        most_recent_action: str | None,
        stagnation_steps: int,
    ) -> float:
        if not stats:
            return 0.0
        progress = int(stats.get("progress", 0) or 0)
        failed = int(stats.get("failed", 0) or 0)
        stalled = int(stats.get("stalled", 0) or 0)
        if progress > 0:
            return 0.0
        penalty = 0.0
        if failed > 0:
            penalty += min(0.20, 0.12 + (failed - 1) * 0.04)
        elif stalled >= 3:
            penalty += 0.16
        elif stalled >= 2:
            penalty += 0.10
        if stagnation_steps >= 2 and most_recent_action == action_name and stalled > 0:
            penalty += 0.04
        return min(0.24, penalty)

    def _episode_context_attempts(
        self,
        memory: dict[str, Any],
        *,
        state_name: str,
        frame_signature: str,
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        if not state_name:
            return counts
        for raw_record in memory.get("records", []):
            if not isinstance(raw_record, dict):
                continue
            if str(raw_record.get("prior_game_state") or "") != state_name:
                continue
            if frame_signature and str(raw_record.get("prior_frame_signature") or "") != frame_signature:
                continue
            action_name = str(raw_record.get("action_name") or "").upper()
            if not action_name:
                continue
            counts[action_name] = int(counts.get(action_name, 0) or 0) + 1
        return counts

    def _stagnation_novelty_bonus(
        self,
        *,
        action_name: str,
        stagnation_steps: int,
        episode_context_attempts: dict[str, int],
    ) -> float:
        if stagnation_steps < 3:
            return 0.0
        attempts = int(episode_context_attempts.get(action_name, 0) or 0)
        if attempts == 0:
            return 0.18
        if attempts == 1:
            return 0.04
        return -min(0.18, 0.06 * (attempts - 1))

    def _recommended_action_bonus(
        self,
        action_name: str,
        recommended_actions: list[str],
    ) -> float:
        if action_name not in recommended_actions:
            return 0.0
        index = recommended_actions.index(action_name)
        return max(0.02, 0.08 - (index * 0.02))

    def _suppressed_action_penalty(
        self,
        *,
        action_name: str,
        suppressed_actions: set[str],
        basin_status: dict[str, Any],
    ) -> float:
        if action_name not in suppressed_actions:
            return 0.0
        if basin_status.get("is_exhausted"):
            return 0.65
        if int(basin_status.get("progress_events", 0) or 0) <= 0:
            return 0.32
        return 0.18

    def _working_memory_focus_bonus(
        self,
        *,
        action_name: str,
        working_memory: dict[str, Any],
    ) -> float:
        semantic_focus = [
            str(name or "")
            for name in list(working_memory.get("semantic_focus_actions", []) or [])
            if str(name or "")
        ]
        procedural_focus = [
            str(name or "")
            for name in list(working_memory.get("procedural_focus_actions", []) or [])
            if str(name or "")
        ]
        abstract_focus = [
            str((row.get("action_name") if isinstance(row, dict) else row) or "")
            for row in list(working_memory.get("abstract_focus_actions", []) or [])
            if str((row.get("action_name") if isinstance(row, dict) else row) or "")
        ]
        bonus = 0.0
        if action_name in semantic_focus:
            bonus += max(0.01, 0.035 - (semantic_focus.index(action_name) * 0.01))
        if action_name in procedural_focus:
            bonus += max(0.01, 0.045 - (procedural_focus.index(action_name) * 0.015))
        if action_name in abstract_focus:
            bonus += max(0.015, 0.05 - (abstract_focus.index(action_name) * 0.015))
        return min(0.07, bonus)

    def _working_memory_failure_penalty(
        self,
        *,
        action_name: str,
        working_memory: dict[str, Any],
    ) -> float:
        penalty = 0.0
        recent_failures = list(working_memory.get("recent_failures", []) or [])
        if any(
            str(row.get("action_name", "") or "") == action_name
            for row in recent_failures
        ):
            penalty += 0.05
        open_failure_modes = list(working_memory.get("open_failure_modes", []) or [])
        if any(
            str(row.get("action_name", "") or "") == action_name
            for row in open_failure_modes
        ):
            penalty += 0.04
        for row in list(working_memory.get("failure_streaks", []) or []):
            if str(row.get("action_name", "") or "") != action_name:
                continue
            penalty += min(0.07, int(row.get("count", 0) or 0) * 0.02)
            break
        return min(0.14, penalty)

    def _episode_basin_loop_penalty(
        self,
        *,
        action_name: str,
        legal_actions: list[str],
        stagnation_steps: int,
        episode_context_attempts: dict[str, int],
        episode_distinct_attempts: int,
        basin_status: dict[str, Any],
        basin_hard_exhausted: bool,
        most_recent_action: str | None,
    ) -> float:
        attempts = int(episode_context_attempts.get(action_name, 0) or 0)
        if attempts <= 0:
            return 0.0
        candidate_actions = [action for action in legal_actions if action != "RESET"]
        if not candidate_actions:
            return 0.0
        no_progress_basin = int(basin_status.get("progress_events", 0) or 0) <= 0
        broad_exploration = episode_distinct_attempts >= min(3, len(candidate_actions))
        if stagnation_steps < 4 and not basin_hard_exhausted:
            return 0.0
        penalty = 0.0
        if no_progress_basin and broad_exploration:
            penalty += 0.18
        if attempts >= 2:
            penalty += min(0.20, 0.08 * (attempts - 1))
        if most_recent_action == action_name and stagnation_steps >= 6:
            penalty += 0.06
        if basin_hard_exhausted:
            penalty += 0.30
        return min(0.75, penalty)

    def _is_episode_basin_hard_exhausted(
        self,
        *,
        legal_actions: list[str],
        stagnation_steps: int,
        basin_status: dict[str, Any],
        episode_context_attempts: dict[str, int],
    ) -> bool:
        candidate_actions = [action for action in legal_actions if action != "RESET"]
        if not candidate_actions:
            return False
        if stagnation_steps < max(4, len(candidate_actions)):
            return False
        if int(basin_status.get("progress_events", 0) or 0) > 0:
            return False
        return all(int(episode_context_attempts.get(action, 0) or 0) > 0 for action in candidate_actions)

    def _initialize_memory_store(self) -> None:
        if self.memory_db_path is None and self.archive_memory_db_path is None:
            return
        if self.memory_db_path is None:
            try:
                if self.archive_memory_db_path is not None and self.archive_memory_db_path.exists():
                    initialize_memory_schema(self.archive_memory_db_path)
                self._memory_store_available = True
                self._memory_write_available = False
                self._memory_error = None
            except Exception:
                # Archive-only mode still stays readable; older or read-only archives
                # should fail soft in the storage loader rather than disabling memory.
                self._memory_store_available = True
                self._memory_write_available = False
            return
        try:
            initialize_memory_schema(self.memory_db_path)
            if self.learning_path is not None:
                import_legacy_learning_json(self.memory_db_path, self.learning_path)
            if self.archive_memory_db_path is not None and self.archive_memory_db_path.exists():
                try:
                    initialize_memory_schema(self.archive_memory_db_path)
                except Exception:
                    # Archive memory can be read-only; keep hot memory live and let
                    # archive-only feature loads fail soft at query time.
                    pass
            self._memory_store_available = True
            self._memory_write_available = True
            self._memory_error = None
        except Exception as exc:
            self._memory_store_available = False
            self._memory_write_available = False
            self._memory_error = repr(exc)

    def _ensure_memory_write_available(self) -> bool:
        if self.memory_db_path is None:
            return False
        if self._memory_write_available:
            return True
        self._initialize_memory_store()
        return self._memory_write_available

    def _note_observation_boundary(self, observation: Observation) -> None:
        game_id = str(observation.game_id or "")
        if not game_id:
            return
        step_index = int(observation.step_index or 0)
        state_name = str(observation.game_state or "")
        step_index_reliable = bool(
            observation.derived_features.get("step_index_reliable", True)
        )
        last_step_index = self._last_observation_step_by_game.get(game_id)
        last_state_name = self._last_observation_state_by_game.get(game_id, "")
        restart_detected = (
            step_index_reliable
            and last_step_index is not None
            and step_index < last_step_index
        )
        reset_exit_detected = (
            bool(last_state_name)
            and last_state_name in self._reset_state_names
            and state_name not in self._reset_state_names
        )
        if restart_detected or reset_exit_detected:
            self._clear_unpersisted_records_for_game(game_id)
        self._last_observation_step_by_game[game_id] = step_index
        self._last_observation_state_by_game[game_id] = state_name

    def _clear_unpersisted_records_for_game(self, game_id: str) -> None:
        pending_keys = [
            record_key
            for record_key, record in self._unpersisted_records.items()
            if str(self._record_field(record, "game_id", "") or "") == game_id
        ]
        for record_key in pending_keys:
            self._unpersisted_records.pop(record_key, None)

    def _load_memory_bundle(
        self,
        *,
        observation: Observation,
        subgoal: str | None,
        legal_actions: list[str],
        memory: Any | None,
        touch_retrieved: bool = True,
    ) -> dict[str, Any]:
        if not self._memory_store_available or (
            self.memory_db_path is None and self.archive_memory_db_path is None
        ):
            return self._augment_memory_layers(
                bundle=self._fallback_memory_bundle(
                    observation=observation,
                    subgoal=subgoal,
                    legal_actions=legal_actions,
                    memory=memory,
                ),
                observation=observation,
                subgoal=subgoal,
                legal_actions=legal_actions,
                memory=memory,
            )
        try:
            bundle = self._load_persisted_memory_bundle(
                observation=observation,
                subgoal=subgoal,
                legal_actions=legal_actions,
                touch_retrieved=touch_retrieved,
            )
            return self._augment_memory_layers(
                bundle=bundle,
                observation=observation,
                subgoal=subgoal,
                legal_actions=legal_actions,
                memory=memory,
            )
        except Exception as exc:
            try:
                bundle = self._load_persisted_memory_bundle(
                    observation=observation,
                    subgoal=subgoal,
                    legal_actions=legal_actions,
                    touch_retrieved=False,
                )
                self._memory_error = None
                return self._augment_memory_layers(
                    bundle=bundle,
                    observation=observation,
                    subgoal=subgoal,
                    legal_actions=legal_actions,
                    memory=memory,
                )
            except Exception:
                self._memory_store_available = False
                self._memory_error = repr(exc)
                return self._augment_memory_layers(
                    bundle=self._fallback_memory_bundle(
                        observation=observation,
                        subgoal=subgoal,
                        legal_actions=legal_actions,
                        memory=memory,
                    ),
                    observation=observation,
                    subgoal=subgoal,
                    legal_actions=legal_actions,
                    memory=memory,
                )

    def _load_persisted_memory_bundle(
        self,
        *,
        observation: Observation,
        subgoal: str | None,
        legal_actions: list[str],
        touch_retrieved: bool,
    ) -> dict[str, Any]:
        current_board_context = board_context_summary(
            observation.image_input,
            derived_features=observation.derived_features,
        )
        bundle = load_memory_bundle_from_stores(
            hot_db_path=self.memory_db_path,
            archive_db_path=self.archive_memory_db_path,
            game_id=observation.game_id,
            state=observation.game_state,
            frame_signature=str(
                observation.derived_features.get("frame_signature", "") or ""
            ),
            subgoal=subgoal,
            legal_actions=legal_actions,
            board_context=current_board_context,
            board_motif_tags=extract_board_motif_tags(
                observation.image_input,
                board_context=current_board_context,
                legal_actions=legal_actions,
                levels_completed=observation.levels_completed,
            ),
            last_action=observation.last_action,
            touch_retrieved=touch_retrieved,
            retrieval_step=int(observation.step_index),
        )
        pending_records = self._pending_records_for_game(observation.game_id)
        if pending_records:
            bundle = self._overlay_records_onto_bundle(
                bundle=bundle,
                records=pending_records,
                observation=observation,
                subgoal=subgoal,
                legal_actions=legal_actions,
            )
        return bundle

    def _load_contextual_bundle(
        self,
        *,
        current_state: str,
        current_frame_signature: str,
        current_subgoal: str | None,
        legal_actions: list[str],
    ) -> dict[str, Any]:
        episode = self.facts.get("episode", {})
        observation = Observation(
            game_id=str(episode.get("game_id", "")),
            step_index=int(episode.get("step_index", 0) or 0),
            game_state=current_state or str(episode.get("game_state", "")),
            levels_completed=int(episode.get("levels_completed", 0) or 0),
            last_action=episode.get("last_action"),
            legal_actions=list(legal_actions),
            derived_features={"frame_signature": current_frame_signature},
        )
        observation.derived_features.update(
            {
                "board_height": int(episode.get("board_height", 0) or 0),
                "board_width": int(episode.get("board_width", 0) or 0),
                "board_active_cells": int(episode.get("board_active_cells", 0) or 0),
                "board_default_probe": dict(episode.get("board_default_probe", {}) or {}),
                "board_source_attr": episode.get("board_source_attr"),
            }
        )
        bundle = self._load_memory_bundle(
            observation=observation,
            subgoal=current_subgoal,
            legal_actions=legal_actions,
            memory=None,
        )
        return bundle

    def _augment_memory_layers(
        self,
        *,
        bundle: dict[str, Any],
        observation: Observation,
        subgoal: str | None,
        legal_actions: list[str],
        memory: Any | None,
    ) -> dict[str, Any]:
        enriched = dict(bundle or {})
        working_memory = dict(enriched.get("working_memory", {}) or {})
        runtime_working = self._runtime_working_memory_seed(memory)
        if runtime_working:
            runtime_working.setdefault("active_state", str(observation.game_state or ""))
            runtime_working.setdefault(
                "active_frame_signature",
                str(observation.derived_features.get("frame_signature", "") or ""),
            )
            runtime_working["active_subgoal"] = str(
                subgoal
                or runtime_working.get("active_subgoal")
                or ""
            )
            runtime_working["legal_actions"] = list(legal_actions)
            runtime_working["board_context_summary"] = board_context_summary(
                observation.image_input,
                derived_features=observation.derived_features,
            )
            recent_actions = [
                str(action_name)
                for action_name in list(runtime_working.get("recent_actions", []) or [])
                if action_name
            ]
            if recent_actions:
                runtime_working["recent_action_window"] = recent_actions[-8:]
            recent_failures = self._runtime_recent_failures(
                runtime_working.get("recent_failures", []),
            )
            open_failure_modes = self._runtime_open_failure_modes(
                runtime_working.get("open_failure_modes", []),
            )
            if recent_failures:
                runtime_working["recent_failures"] = recent_failures
            if recent_failures or open_failure_modes:
                runtime_working["open_failure_modes"] = self._merge_open_failure_modes(
                    recent_failures + open_failure_modes,
                    working_memory.get("open_failure_modes", []),
                )
            failure_streaks = self._runtime_failure_streaks(
                runtime_working.get("failure_streaks", []),
            )
            if failure_streaks:
                runtime_working["failure_streaks"] = failure_streaks
            working_memory.update(runtime_working)
        enriched["working_memory"] = working_memory
        recommended_actions: list[str] = []
        for action_name in list(working_memory.get("recommended_actions", []) or []) + list(
            enriched.get("recommended_actions", []) or []
        ):
            normalized = str(action_name or "")
            if normalized and normalized not in recommended_actions:
                recommended_actions.append(normalized)
        if self._working_memory_has_action_guidance(working_memory):
            recommended_actions = self._refresh_runtime_recommended_actions(
                legal_actions=legal_actions,
                recommended_actions=recommended_actions,
                suppressed_actions=list(enriched.get("suppressed_actions", []) or []),
                working_memory=working_memory,
            )
        enriched["recommended_actions"] = recommended_actions
        trained_prior_focus = _compact_trained_prior_focus(
            dict(enriched.get("trained_prior_bucket", {}) or {}),
            legal_actions=legal_actions,
        )
        if working_memory:
            working_memory["recommended_actions"] = list(recommended_actions[:4])
            working_memory["trained_prior_focus"] = trained_prior_focus
            working_memory["suppressed_actions"] = list(
                enriched.get("suppressed_actions", []) or []
            )[:6]
            basin_status = dict(enriched.get("basin_status", {}) or {})
            working_memory["basin_exhausted"] = bool(
                basin_status.get("is_exhausted", False)
            )
            working_memory["basin_visits"] = int(basin_status.get("visits", 0) or 0)
        retrieval_summary = dict(enriched.get("retrieval_summary", {}) or {})
        retrieval_summary.setdefault("retrieval_version", "hybrid-multistore-v1")
        retrieval_summary["active_subgoal"] = str(
            subgoal
            or working_memory.get("active_subgoal")
            or retrieval_summary.get("active_subgoal")
            or ""
        )
        retrieval_summary["runtime_has_working_memory"] = bool(working_memory)
        retrieval_summary["suggested_actions"] = list(recommended_actions[:4])
        retrieval_summary["trained_prior_focus"] = trained_prior_focus
        enriched["retrieval_summary"] = retrieval_summary
        return enriched

    def _refresh_runtime_recommended_actions(
        self,
        *,
        legal_actions: list[str],
        recommended_actions: list[str],
        suppressed_actions: list[str],
        working_memory: dict[str, Any],
    ) -> list[str]:
        suppressed = {str(action_name or "") for action_name in list(suppressed_actions or []) if str(action_name or "")}
        candidates: list[str] = []
        for action_name in list(recommended_actions or []):
            normalized = str(action_name or "")
            if (
                normalized
                and normalized != "RESET"
                and normalized in legal_actions
                and normalized not in suppressed
                and normalized not in candidates
            ):
                candidates.append(normalized)
        for action_name in list(legal_actions or []):
            normalized = str(action_name or "")
            if (
                normalized
                and normalized != "RESET"
                and normalized not in suppressed
                and normalized not in candidates
            ):
                candidates.append(normalized)
        if not candidates:
            return []
        ranked = sorted(
            candidates,
            key=lambda action_name: (
                self._recommended_action_bonus(action_name, recommended_actions)
                + self._working_memory_focus_bonus(
                    action_name=action_name,
                    working_memory=working_memory,
                )
                - self._working_memory_failure_penalty(
                    action_name=action_name,
                    working_memory=working_memory,
                )
            ),
            reverse=True,
        )
        return ranked[:4]

    def _working_memory_has_action_guidance(
        self,
        working_memory: dict[str, Any],
    ) -> bool:
        return any(
            list(working_memory.get(key, []) or [])
            for key in (
                "recommended_actions",
                "semantic_focus_actions",
                "procedural_focus_actions",
                "abstract_focus_actions",
                "recent_failures",
                "open_failure_modes",
                "failure_streaks",
            )
        )

    def _runtime_working_memory_seed(self, memory: Any | None) -> dict[str, Any]:
        if memory is not None and hasattr(memory, "working_memory_snapshot"):
            return dict(memory.working_memory_snapshot() or {})
        current_bundle = self.current_memory_bundle()
        return dict(getattr(current_bundle, "working_memory", {}) or {})

    def _runtime_recent_failures(
        self,
        recent_failures: Any,
    ) -> list[dict[str, Any]]:
        normalized_rows: list[dict[str, Any]] = []
        for raw_row in list(recent_failures or [])[:4]:
            row = dict(raw_row or {})
            action_name = str(row.get("action_name", "") or "")
            if not action_name:
                continue
            normalized_rows.append(
                {
                    "action_name": action_name,
                    "subgoal": str(
                        row.get("subgoal", row.get("decision_subgoal", "")) or ""
                    ),
                    "outcome": str(row.get("outcome", "") or ""),
                    "stagnation_after": int(row.get("stagnation_after", 0) or 0),
                    "frame_signature": str(
                        row.get("frame_signature", row.get("prior_frame_signature", ""))
                        or ""
                    ),
                    "prior_game_state": str(row.get("prior_game_state", "") or ""),
                }
            )
        return normalized_rows

    def _runtime_open_failure_modes(
        self,
        open_failure_modes: Any,
    ) -> list[dict[str, Any]]:
        normalized_rows: list[dict[str, Any]] = []
        for raw_row in list(open_failure_modes or [])[:4]:
            row = dict(raw_row or {})
            action_name = str(row.get("action_name", "") or "")
            if not action_name:
                continue
            normalized_rows.append(
                {
                    "action_name": action_name,
                    "subgoal": str(
                        row.get("subgoal", row.get("decision_subgoal", "")) or ""
                    ),
                    "outcome": str(row.get("outcome", "") or ""),
                    "stagnation_after": int(row.get("stagnation_after", 0) or 0),
                    "frame_signature": str(
                        row.get("frame_signature", row.get("prior_frame_signature", "")) or ""
                    ),
                }
            )
        return normalized_rows

    def _merge_open_failure_modes(
        self,
        runtime_failures: list[dict[str, Any]],
        existing_failures: Any,
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str, str, str]] = set()
        for raw_row in [*runtime_failures, *list(existing_failures or [])]:
            row = dict(raw_row or {})
            action_name = str(row.get("action_name", "") or "")
            if not action_name:
                continue
            normalized_row = {
                "action_name": action_name,
                "subgoal": str(
                    row.get("subgoal", row.get("decision_subgoal", "")) or ""
                ),
                "outcome": str(row.get("outcome", "") or ""),
                "stagnation_after": int(row.get("stagnation_after", 0) or 0),
                "frame_signature": str(
                    row.get("frame_signature", row.get("prior_frame_signature", "")) or ""
                ),
            }
            row_key = (
                normalized_row["action_name"],
                normalized_row["subgoal"],
                normalized_row["outcome"],
                normalized_row["frame_signature"],
            )
            if row_key in seen_keys:
                continue
            seen_keys.add(row_key)
            merged.append(normalized_row)
            if len(merged) >= 4:
                break
        return merged

    def _runtime_failure_streaks(
        self,
        failure_streaks: Any,
    ) -> list[dict[str, Any]]:
        normalized_rows: list[dict[str, Any]] = []
        for raw_row in list(failure_streaks or [])[:6]:
            row = dict(raw_row or {})
            action_name = str(row.get("action_name", "") or "")
            if not action_name:
                continue
            normalized_rows.append(
                {
                    "action_name": action_name,
                    "count": int(row.get("count", 0) or 0),
                }
            )
        return normalized_rows

    def _fallback_memory_bundle(
        self,
        *,
        observation: Observation,
        subgoal: str | None,
        legal_actions: list[str],
        memory: Any | None,
    ) -> dict[str, Any]:
        records = self._fallback_memory_records(memory)
        trained_prior_bucket = self._load_trained_prior_bucket(
            observation,
            subgoal=subgoal,
        )
        learned = self._summarize_learned_knowledge(observation.game_id)
        if memory is not None:
            self._ingest_episode_records(records)
            learned = self._summarize_learned_knowledge(observation.game_id)
        frame_signature = str(
            observation.derived_features.get("frame_signature", "") or ""
        )
        state_name = str(observation.game_state or "")
        action_stats = self._clone_action_stats(learned.get("actions", {}))
        state_stats = self._clone_action_stats(
            (learned.get("states", {}).get(state_name, {}) or {}).get("actions", {})
        )
        frame_stats = self._clone_action_stats(
            (learned.get("frame_signatures", {}).get(frame_signature, {}) or {}).get(
                "actions", {}
            )
        )
        subgoal_stats = self._clone_action_stats(
            (learned.get("subgoals", {}).get(str(subgoal or ""), {}) or {}).get(
                "actions", {}
            )
        )
        if (
            not records
            and not action_stats
            and not state_stats
            and not frame_stats
            and not subgoal_stats
            and not trained_prior_bucket
        ):
            bundle = self._empty_memory_bundle()
            bundle["trained_prior_bucket"] = trained_prior_bucket
            return bundle
        pending_records = [
            record
            for record in records
            if self._record_key(record) not in self._ingested_record_keys
        ]
        if pending_records:
            self._merge_action_stats(
                action_stats,
                self._action_stats_from_records(records=pending_records),
            )
            self._merge_action_stats(
                state_stats,
                self._action_stats_from_records(
                    records=pending_records,
                    state_name=state_name,
                ),
            )
            self._merge_action_stats(
                frame_stats,
                self._action_stats_from_records(
                    records=pending_records,
                    frame_signature=frame_signature,
                ),
            )
            self._merge_action_stats(
                subgoal_stats,
                self._action_stats_from_records(
                    records=pending_records,
                    subgoal=str(subgoal or ""),
                ),
            )
        basin_action_stats = self._basin_action_stats_from_records(
            records=records,
            state_name=state_name,
            frame_signature=frame_signature,
        )
        recent_records = self._recent_records_from_records(
            records=records,
            state_name=state_name,
            frame_signature=frame_signature,
            subgoal=subgoal,
            board_context_summary=board_context_summary(
                observation.image_input,
                derived_features=observation.derived_features,
            ),
        )
        basin_status = self._fallback_basin_status(
            records=records,
            observation=observation,
            basin_action_stats=basin_action_stats,
        )
        suppressed_actions = _derive_suppressed_actions(
            legal_actions=legal_actions,
            basin_action_stats=basin_action_stats,
            state_stats=state_stats,
            subgoal_stats=subgoal_stats,
            basin_status=basin_status,
        )
        board_context_action_stats = self._action_stats_from_records(
            records=recent_records,
            board_context_summary=board_context_summary(
                observation.image_input,
                derived_features=observation.derived_features,
            ),
        )
        pattern_action_stats = self._action_stats_from_records(
            records=records,
            state_name=state_name,
            subgoal=str(subgoal or ""),
            board_context_summary=board_context_summary(
                observation.image_input,
                derived_features=observation.derived_features,
            ),
        )
        sequence_action_stats = self._sequence_action_stats_from_records(
            records=records,
            previous_action_name=observation.last_action,
            state_name=state_name,
            subgoal=str(subgoal or ""),
        )
        board_cluster_action_stats = self._action_stats_from_records(
            records=records,
            board_context_summary=board_context_summary(
                observation.image_input,
                derived_features=observation.derived_features,
            ),
        )
        recommended_actions = _derive_recommended_actions(
            legal_actions=legal_actions,
            action_stats=action_stats,
            state_stats=state_stats,
            basin_action_stats=basin_action_stats,
            frame_stats=frame_stats,
            subgoal_stats=subgoal_stats,
            pattern_stats=pattern_action_stats,
            sequence_stats=sequence_action_stats,
            board_cluster_stats=board_cluster_action_stats,
            board_context_stats=board_context_action_stats,
            trained_prior_bucket=trained_prior_bucket,
            abstract_memories=[],
            recent_records=recent_records,
            suppressed_actions=suppressed_actions,
            unknown_game_context=False,
        )
        semantic_memories = self._fallback_semantic_memories(
            state_name=state_name,
            subgoal=subgoal,
            pattern_action_stats=pattern_action_stats,
            board_context_summary=board_context_summary(
                observation.image_input,
                derived_features=observation.derived_features,
            ),
        )
        procedural_memories = self._fallback_procedural_memories(
            state_name=state_name,
            subgoal=subgoal,
            previous_action_name=observation.last_action,
            sequence_action_stats=sequence_action_stats,
        )
        episodic_recalls = self._fallback_episodic_recalls(
            recent_records=recent_records,
        )
        working_memory = self._fallback_working_memory(
            state_name=state_name,
            frame_signature=frame_signature,
            subgoal=subgoal,
            board_context_summary=board_context_summary(
                observation.image_input,
                derived_features=observation.derived_features,
            ),
            legal_actions=legal_actions,
            recent_records=recent_records,
            recommended_actions=recommended_actions,
            trained_prior_bucket=trained_prior_bucket,
            suppressed_actions=suppressed_actions,
            basin_status=basin_status,
        )
        return {
            "working_memory": working_memory,
            "episodic_recalls": episodic_recalls,
            "semantic_memories": semantic_memories,
            "procedural_memories": procedural_memories,
            "abstract_memories": [],
            "retrieval_summary": self._fallback_retrieval_summary(
                state_name=state_name,
                frame_signature=frame_signature,
                subgoal=subgoal,
                legal_actions=legal_actions,
                recent_records=recent_records,
                semantic_memories=semantic_memories,
                procedural_memories=procedural_memories,
                abstract_memories=[],
                recommended_actions=recommended_actions,
                trained_prior_bucket=trained_prior_bucket,
                suppressed_actions=suppressed_actions,
            ),
            "board_context_summary": board_context_summary(
                observation.image_input,
                derived_features=observation.derived_features,
            ),
            "board_context_action_stats": board_context_action_stats,
            "pattern_action_stats": pattern_action_stats,
            "sequence_action_stats": sequence_action_stats,
            "board_cluster_action_stats": board_cluster_action_stats,
            "recent_records": recent_records,
            "basin_status": basin_status,
            "basin_action_stats": basin_action_stats,
            "action_stats": action_stats,
            "state_action_stats": state_stats,
            "frame_action_stats": frame_stats,
            "subgoal_action_stats": subgoal_stats,
            "trained_prior_bucket": trained_prior_bucket,
            "recommended_actions": recommended_actions,
            "suppressed_actions": suppressed_actions,
        }

    def _overlay_records_onto_bundle(
        self,
        *,
        bundle: dict[str, Any],
        records: list[Any],
        observation: Observation,
        subgoal: str | None,
        legal_actions: list[str],
    ) -> dict[str, Any]:
        frame_signature = str(
            observation.derived_features.get("frame_signature", "") or ""
        )
        state_name = str(observation.game_state or "")
        action_stats = self._clone_action_stats(dict(bundle.get("action_stats", {})))
        state_stats = self._clone_action_stats(
            dict(bundle.get("state_action_stats", {}))
        )
        frame_stats = self._clone_action_stats(
            dict(bundle.get("frame_action_stats", {}))
        )
        subgoal_stats = self._clone_action_stats(
            dict(bundle.get("subgoal_action_stats", {}))
        )
        pattern_action_stats = self._clone_action_stats(
            dict(bundle.get("pattern_action_stats", {}))
        )
        sequence_action_stats = self._clone_action_stats(
            dict(bundle.get("sequence_action_stats", {}))
        )
        board_cluster_action_stats = self._clone_action_stats(
            dict(bundle.get("board_cluster_action_stats", {}))
        )
        basin_action_stats = self._clone_action_stats(
            dict(bundle.get("basin_action_stats", {}))
        )
        self._merge_action_stats(
            action_stats,
            self._action_stats_from_records(records=records),
        )
        self._merge_action_stats(
            state_stats,
            self._action_stats_from_records(
                records=records,
                state_name=state_name,
            ),
        )
        self._merge_action_stats(
            frame_stats,
            self._action_stats_from_records(
                records=records,
                frame_signature=frame_signature,
            ),
        )
        self._merge_action_stats(
            subgoal_stats,
            self._action_stats_from_records(
                records=records,
                subgoal=str(subgoal or ""),
            ),
        )
        self._merge_action_stats(
            basin_action_stats,
            self._basin_action_stats_from_records(
                records=records,
                state_name=state_name,
                frame_signature=frame_signature,
            ),
        )
        self._merge_action_stats(
            pattern_action_stats,
            self._action_stats_from_records(
                records=records,
                state_name=state_name,
                subgoal=str(subgoal or ""),
                board_context_summary=dict(bundle.get("board_context_summary", {}) or {}),
            ),
        )
        self._merge_action_stats(
            sequence_action_stats,
            self._sequence_action_stats_from_records(
                records=records,
                previous_action_name=observation.last_action,
                state_name=state_name,
                subgoal=str(subgoal or ""),
            ),
        )
        self._merge_action_stats(
            board_cluster_action_stats,
            self._action_stats_from_records(
                records=records,
                board_context_summary=dict(bundle.get("board_context_summary", {}) or {}),
            ),
        )
        recent_records = self._recent_records_from_records(
            records=[*records, *(bundle.get("recent_records", []) or [])],
            state_name=state_name,
            frame_signature=frame_signature,
            subgoal=subgoal,
            board_context_summary=dict(bundle.get("board_context_summary", {}) or {}),
        )
        basin_status = self._merge_basin_status(
            base=dict(bundle.get("basin_status", {})),
            overlay=self._fallback_basin_status(
                records=records,
                observation=observation,
                basin_action_stats=basin_action_stats,
            ),
            basin_action_stats=basin_action_stats,
            observation=observation,
        )
        trained_prior_bucket = dict(bundle.get("trained_prior_bucket", {}))
        suppressed_actions = _derive_suppressed_actions(
            legal_actions=legal_actions,
            basin_action_stats=basin_action_stats,
            state_stats=state_stats,
            subgoal_stats=subgoal_stats,
            basin_status=basin_status,
        )
        board_context_action_stats = self._action_stats_from_records(
            records=recent_records,
            board_context_summary=dict(bundle.get("board_context_summary", {}) or {}),
        )
        recommended_actions = _derive_recommended_actions(
            legal_actions=legal_actions,
            action_stats=action_stats,
            state_stats=state_stats,
            basin_action_stats=basin_action_stats,
            frame_stats=frame_stats,
            subgoal_stats=subgoal_stats,
            pattern_stats=pattern_action_stats,
            sequence_stats=sequence_action_stats,
            board_cluster_stats=board_cluster_action_stats,
            board_context_stats=board_context_action_stats,
            trained_prior_bucket=trained_prior_bucket,
            abstract_memories=[],
            recent_records=recent_records,
            suppressed_actions=suppressed_actions,
            unknown_game_context=False,
        )
        semantic_memories = self._fallback_semantic_memories(
            state_name=state_name,
            subgoal=subgoal,
            pattern_action_stats=pattern_action_stats,
            board_context_summary=board_context_summary(
                observation.image_input,
                derived_features=observation.derived_features,
            ),
        )
        procedural_memories = self._fallback_procedural_memories(
            state_name=state_name,
            subgoal=subgoal,
            previous_action_name=observation.last_action,
            sequence_action_stats=sequence_action_stats,
        )
        episodic_recalls = self._fallback_episodic_recalls(
            recent_records=recent_records,
            episode_summaries=list(bundle.get("episode_summaries", []) or []),
        )
        working_memory = self._fallback_working_memory(
            state_name=state_name,
            frame_signature=frame_signature,
            subgoal=subgoal,
            board_context_summary=board_context_summary(
                observation.image_input,
                derived_features=observation.derived_features,
            ),
            legal_actions=legal_actions,
            recent_records=recent_records,
            recommended_actions=recommended_actions,
            trained_prior_bucket=trained_prior_bucket,
            suppressed_actions=suppressed_actions,
            basin_status=basin_status,
        )
        return {
            "working_memory": working_memory,
            "episodic_recalls": episodic_recalls,
            "semantic_memories": semantic_memories,
            "procedural_memories": procedural_memories,
            "retrieval_summary": self._fallback_retrieval_summary(
                state_name=state_name,
                frame_signature=frame_signature,
                subgoal=subgoal,
                legal_actions=legal_actions,
                recent_records=recent_records,
                semantic_memories=semantic_memories,
                procedural_memories=procedural_memories,
                recommended_actions=recommended_actions,
                trained_prior_bucket=trained_prior_bucket,
                suppressed_actions=suppressed_actions,
            ),
            "board_context_summary": board_context_summary(
                observation.image_input,
                derived_features=observation.derived_features,
            ),
            "board_context_action_stats": board_context_action_stats,
            "pattern_action_stats": pattern_action_stats,
            "sequence_action_stats": sequence_action_stats,
            "board_cluster_action_stats": board_cluster_action_stats,
            "recent_records": recent_records,
            "basin_status": basin_status,
            "basin_action_stats": basin_action_stats,
            "action_stats": action_stats,
            "state_action_stats": state_stats,
            "frame_action_stats": frame_stats,
            "subgoal_action_stats": subgoal_stats,
            "trained_prior_bucket": trained_prior_bucket,
            "recommended_actions": recommended_actions,
            "suppressed_actions": suppressed_actions,
        }

    def _empty_memory_bundle(self) -> dict[str, Any]:
        return {
            "working_memory": {},
            "episodic_recalls": [],
            "semantic_memories": [],
            "procedural_memories": [],
            "abstract_memories": [],
            "retrieval_summary": {},
            "board_context_summary": {},
            "board_context_action_stats": {},
            "pattern_action_stats": {},
            "sequence_action_stats": {},
            "board_cluster_action_stats": {},
            "recent_records": [],
            "basin_status": {},
            "basin_action_stats": {},
            "action_stats": {},
            "state_action_stats": {},
            "frame_action_stats": {},
            "subgoal_action_stats": {},
            "trained_prior_bucket": {},
            "recommended_actions": [],
            "suppressed_actions": [],
        }

    def _fallback_working_memory(
        self,
        *,
        state_name: str,
        frame_signature: str,
        subgoal: str | None,
        board_context_summary: dict[str, Any],
        legal_actions: list[str],
        recent_records: list[dict[str, Any]],
        recommended_actions: list[str],
        trained_prior_bucket: dict[str, int],
        suppressed_actions: list[str],
        basin_status: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "active_state": state_name,
            "active_frame_signature": frame_signature,
            "active_subgoal": str(subgoal or ""),
            "board_context_summary": dict(board_context_summary or {}),
            "legal_actions": list(legal_actions),
            "recent_action_window": [
                str(record.get("action_name", "") or "")
                for record in recent_records[:8]
                if str(record.get("action_name", "") or "")
            ],
            "recommended_actions": list(recommended_actions[:4]),
            "trained_prior_focus": _compact_trained_prior_focus(
                trained_prior_bucket,
                legal_actions=legal_actions,
            ),
            "abstract_focus_actions": [],
            "suppressed_actions": list(suppressed_actions[:6]),
            "basin_exhausted": bool(basin_status.get("is_exhausted", False)),
        }

    def _fallback_episodic_recalls(
        self,
        *,
        recent_records: list[dict[str, Any]],
        episode_summaries: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        recalls: list[dict[str, Any]] = []
        for record in list(recent_records or [])[:4]:
            recalls.append(
                {
                    "recall_type": "record",
                    "action_name": str(record.get("action_name", "") or ""),
                    "outcome": str(record.get("outcome", "") or ""),
                    "prior_game_state": str(record.get("prior_game_state", "") or ""),
                    "prior_frame_signature": str(record.get("prior_frame_signature", "") or ""),
                    "decision_subgoal": str(record.get("decision_subgoal", "") or ""),
                    "score": 1.0,
                    "retrieval_provenance": "fallback-record",
                }
            )
        for summary in list(episode_summaries or [])[:4]:
            recalls.append(
                {
                    "recall_type": "summary",
                    "action_name": "",
                    "outcome": "",
                    "prior_game_state": str(summary.get("prior_game_state", "") or ""),
                    "prior_frame_signature": str(summary.get("prior_frame_signature", "") or ""),
                    "decision_subgoal": str(summary.get("decision_subgoal", "") or ""),
                    "score": 0.9,
                    "retrieval_provenance": "fallback-summary",
                }
            )
        return recalls[:8]

    def _fallback_semantic_memories(
        self,
        *,
        state_name: str,
        subgoal: str | None,
        pattern_action_stats: dict[str, dict[str, Any]],
        board_context_summary: dict[str, Any],
    ) -> list[dict[str, Any]]:
        memories: list[dict[str, Any]] = []
        for action_name, stats in sorted(
            pattern_action_stats.items(),
            key=lambda item: (
                int(item[1].get("progress", 0) or 0) - int(item[1].get("failed", 0) or 0),
                int(item[1].get("total", 0) or 0),
            ),
            reverse=True,
        )[:4]:
            memories.append(
                {
                    "memory_type": "semantic",
                    "state": state_name,
                    "subgoal": str(subgoal or ""),
                    "action_name": str(action_name or ""),
                    "lesson": f"Pattern memory for {action_name} in {state_name}",
                    "board_context_summary": dict(board_context_summary or {}),
                    "progress": int(stats.get("progress", 0) or 0),
                    "failed": int(stats.get("failed", 0) or 0),
                    "stalled": int(stats.get("stalled", 0) or 0),
                    "total": int(stats.get("total", 0) or 0),
                    "score": float(int(stats.get("progress", 0) or 0)),
                    "retrieval_provenance": "fallback-semantic",
                }
            )
        return memories

    def _fallback_procedural_memories(
        self,
        *,
        state_name: str,
        subgoal: str | None,
        previous_action_name: str | None,
        sequence_action_stats: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        memories: list[dict[str, Any]] = []
        for action_name, stats in sorted(
            sequence_action_stats.items(),
            key=lambda item: (
                int(item[1].get("progress", 0) or 0) - int(item[1].get("failed", 0) or 0),
                int(item[1].get("total", 0) or 0),
            ),
            reverse=True,
        )[:4]:
            memories.append(
                {
                    "memory_type": "procedural",
                    "state": state_name,
                    "subgoal": str(subgoal or ""),
                    "previous_action_name": str(previous_action_name or ""),
                    "action_name": str(action_name or ""),
                    "strategy_tag": "fallback-chain",
                    "progress": int(stats.get("progress", 0) or 0),
                    "failed": int(stats.get("failed", 0) or 0),
                    "stalled": int(stats.get("stalled", 0) or 0),
                    "total": int(stats.get("total", 0) or 0),
                    "score": float(int(stats.get("progress", 0) or 0)),
                    "retrieval_provenance": "fallback-procedural",
                }
            )
        return memories

    def _fallback_retrieval_summary(
        self,
        *,
        state_name: str,
        frame_signature: str,
        subgoal: str | None,
        legal_actions: list[str],
        recent_records: list[dict[str, Any]],
        semantic_memories: list[dict[str, Any]],
        procedural_memories: list[dict[str, Any]],
        abstract_memories: list[dict[str, Any]],
        recommended_actions: list[str],
        trained_prior_bucket: dict[str, int],
        suppressed_actions: list[str],
    ) -> dict[str, Any]:
        return {
            "retrieval_version": "hybrid-multistore-v1",
            "state": state_name,
            "frame_signature": frame_signature,
            "subgoal": str(subgoal or ""),
            "recent_record_count": len(recent_records),
            "semantic_memory_count": len(semantic_memories),
            "procedural_memory_count": len(procedural_memories),
            "abstract_memory_count": len(abstract_memories),
            "suggested_actions": list(recommended_actions[:4]),
            "trained_prior_focus": _compact_trained_prior_focus(
                trained_prior_bucket,
                legal_actions=legal_actions,
            ),
            "abstract_focus_actions": [],
            "suppressed_actions": list(suppressed_actions[:6]),
        }

    def _clone_action_stats(self, stats: dict[str, Any]) -> dict[str, dict[str, Any]]:
        cloned: dict[str, dict[str, Any]] = {}
        for action_name, value in stats.items():
            if not isinstance(value, dict):
                continue
            cloned[str(action_name)] = dict(value)
        return cloned

    def _merge_action_stats(
        self,
        target: dict[str, dict[str, Any]],
        overlay: dict[str, dict[str, Any]],
    ) -> None:
        for action_name, stats in overlay.items():
            entry = target.setdefault(
                action_name,
                {"progress": 0, "failed": 0, "stalled": 0, "total": 0, "last_outcome": None},
            )
            entry["progress"] = int(entry.get("progress", 0) or 0) + int(
                stats.get("progress", 0) or 0
            )
            entry["failed"] = int(entry.get("failed", 0) or 0) + int(
                stats.get("failed", 0) or 0
            )
            entry["stalled"] = int(entry.get("stalled", 0) or 0) + int(
                stats.get("stalled", 0) or 0
            )
            entry["total"] = int(entry.get("total", 0) or 0) + int(
                stats.get("total", 0) or 0
            )
            if stats.get("last_outcome") is not None:
                entry["last_outcome"] = stats.get("last_outcome")

    def _merge_basin_status(
        self,
        *,
        base: dict[str, Any],
        overlay: dict[str, Any],
        basin_action_stats: dict[str, dict[str, Any]],
        observation: Observation,
    ) -> dict[str, Any]:
        merged = dict(base)
        for key in ("visits", "actions_attempted", "progress_events", "failed_events", "stalled_events"):
            merged[key] = int(base.get(key, 0) or 0) + int(overlay.get(key, 0) or 0)
        merged["state"] = overlay.get("state") or base.get("state") or str(observation.game_state or "")
        merged["frame_signature"] = overlay.get("frame_signature") or base.get("frame_signature") or str(
            observation.derived_features.get("frame_signature", "") or ""
        )
        merged["distinct_actions_tried"] = len(basin_action_stats)
        merged["last_action_name"] = overlay.get("last_action_name") or base.get("last_action_name")
        merged["last_progress_step"] = overlay.get("last_progress_step")
        if merged["last_progress_step"] is None:
            merged["last_progress_step"] = base.get("last_progress_step")
        merged["action_outcomes"] = basin_action_stats
        merged["is_exhausted"] = self._fallback_is_basin_exhausted(
            observation=observation,
            basin_action_stats=basin_action_stats,
        )
        return merged

    def _pending_records_for_game(self, game_id: str) -> list[EpisodeRecord]:
        pending: list[EpisodeRecord] = []
        for record in self._unpersisted_records.values():
            if str(record.game_id or "") == game_id:
                pending.append(record)
        return pending

    def _fallback_memory_records(self, memory: Any | None) -> list[Any]:
        if memory is not None and hasattr(memory, "records"):
            return list(memory.records)
        raw_records = self.facts.get("memory", {}).get("records", [])
        if not isinstance(raw_records, list):
            return []
        return raw_records

    def _action_stats_from_records(
        self,
        *,
        records: list[Any],
        state_name: str | None = None,
        frame_signature: str | None = None,
        subgoal: str | None = None,
        board_context_summary: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        stats: dict[str, dict[str, Any]] = {}
        current_board_context = dict(board_context_summary or {})
        for record in records:
            row = self._record_to_recent_record(record)
            if state_name is not None and str(row.get("prior_game_state") or "") != state_name:
                continue
            if frame_signature is not None and str(
                row.get("prior_frame_signature") or ""
            ) != frame_signature:
                continue
            if subgoal is not None and str(row.get("decision_subgoal") or "") != subgoal:
                continue
            if board_context_summary is not None and board_context_match_score(
                current_board_context,
                dict(row.get("board_context_summary", {}) or {}),
            ) < 2:
                continue
            action_name = str(row.get("action_name") or "").upper()
            if not action_name:
                continue
            entry = stats.setdefault(
                action_name,
                {"progress": 0, "failed": 0, "stalled": 0, "total": 0, "last_outcome": None},
            )
            self._apply_context_learning(entry, str(row.get("outcome") or ""))
        return stats

    def _record_key(self, record: Any) -> tuple[Any, ...]:
        return (
            self._record_field(record, "game_id", ""),
            int(self._record_field(record, "step_index", 0) or 0),
            self._record_field(record, "action_name", ""),
            self._record_field(record, "outcome", ""),
            int(self._record_field(record, "progress_delta", 0) or 0),
            self._record_field(record, "game_state", ""),
            self._record_field(record, "prior_game_state", ""),
            self._record_field(record, "decision_subgoal", None),
            self._record_field(record, "prior_frame_signature", None),
        )

    def _record_field(self, record: Any, name: str, default: Any = None) -> Any:
        if isinstance(record, dict):
            return record.get(name, default)
        return getattr(record, name, default)

    def _basin_action_stats_from_records(
        self,
        *,
        records: list[Any],
        state_name: str,
        frame_signature: str,
    ) -> dict[str, dict[str, Any]]:
        basin_stats: dict[str, dict[str, Any]] = {}
        for record in records:
            row = self._record_to_recent_record(record)
            if str(row.get("prior_game_state") or "") != state_name:
                continue
            record_frame_signature = str(row.get("prior_frame_signature") or "")
            if frame_signature and record_frame_signature != frame_signature:
                continue
            action_name = str(row.get("action_name") or "").upper()
            if not action_name:
                continue
            entry = basin_stats.setdefault(
                action_name,
                {"progress": 0, "failed": 0, "stalled": 0, "total": 0, "last_outcome": None},
            )
            self._apply_context_learning(entry, str(row.get("outcome") or ""))
        return basin_stats

    def _recent_records_from_records(
        self,
        *,
        records: list[Any],
        state_name: str,
        frame_signature: str,
        subgoal: str | None,
        board_context_summary: dict[str, Any] | None = None,
        limit: int = 24,
    ) -> list[dict[str, Any]]:
        current_subgoal = str(subgoal or "")
        current_board_context = dict(board_context_summary or {})
        scored_rows: list[tuple[tuple[int, int, int, int, int, int], dict[str, Any]]] = []
        for record in records:
            row = self._record_to_recent_record(record)
            prior_state = str(row.get("prior_game_state") or "")
            prior_frame_signature = str(row.get("prior_frame_signature") or "")
            record_subgoal = str(row.get("decision_subgoal") or "")
            exact_basin = (
                1
                if state_name
                and frame_signature
                and prior_state == state_name
                and prior_frame_signature == frame_signature
                else 0
            )
            same_state = 1 if state_name and prior_state == state_name else 0
            same_subgoal = 1 if current_subgoal and record_subgoal == current_subgoal else 0
            same_frame = 1 if frame_signature and prior_frame_signature == frame_signature else 0
            board_match = board_context_match_score(
                current_board_context,
                dict(row.get("board_context_summary", {}) or {}),
            )
            scored_rows.append(
                (
                    (
                        exact_basin,
                        same_state,
                        same_subgoal,
                        board_match,
                        same_frame,
                        int(row.get("step_index", 0) or 0),
                    ),
                    row,
                )
            )
        scored_rows.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in scored_rows[:limit]]

    def _sequence_action_stats_from_records(
        self,
        *,
        records: list[Any],
        previous_action_name: str | None,
        state_name: str | None = None,
        subgoal: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        if not previous_action_name:
            return {}
        rows = [self._record_to_recent_record(record) for record in records]
        rows.sort(key=lambda row: int(row.get("step_index", 0) or 0))
        stats: dict[str, dict[str, Any]] = {}
        for index in range(1, len(rows)):
            prior = rows[index - 1]
            current = rows[index]
            if str(prior.get("action_name") or "") != str(previous_action_name):
                continue
            if state_name is not None and str(current.get("prior_game_state") or "") != state_name:
                continue
            if subgoal is not None and str(current.get("decision_subgoal") or "") != subgoal:
                continue
            action_name = str(current.get("action_name") or "").upper()
            if not action_name:
                continue
            entry = stats.setdefault(
                action_name,
                {"progress": 0, "failed": 0, "stalled": 0, "total": 0, "last_outcome": None},
            )
            self._apply_context_learning(entry, str(current.get("outcome") or ""))
        return stats

    def _record_to_recent_record(self, record: Any) -> dict[str, Any]:
        if isinstance(record, dict):
            return {
                "game_id": str(record.get("game_id", "")),
                "step_index": int(record.get("step_index", 0) or 0),
                "action_name": str(record.get("action_name", "")),
                "outcome": str(record.get("outcome", "")),
                "progress_delta": int(record.get("progress_delta", 0) or 0),
                "game_state": str(
                    record.get("game_state")
                    or record.get("next_game_state")
                    or ""
                ),
                "prior_game_state": str(record.get("prior_game_state", "")),
                "prior_frame_signature": str(record.get("prior_frame_signature", "") or ""),
                "next_frame_signature": str(record.get("next_frame_signature", "") or ""),
                "decision_subgoal": str(record.get("decision_subgoal", "") or ""),
                "decision_source": str(record.get("decision_source", "") or ""),
                "stagnation_before": int(record.get("stagnation_before", 0) or 0),
                "stagnation_after": int(record.get("stagnation_after", 0) or 0),
                "prior_legal_actions": list(record.get("prior_legal_actions", []) or []),
                "board_context_summary": {
                    "board_height": int(record.get("prior_board_height", 0) or 0),
                    "board_width": int(record.get("prior_board_width", 0) or 0),
                    "board_active_cells": int(record.get("prior_board_active_cells", 0) or 0),
                    "board_source_attr": str(record.get("prior_board_source_attr") or "")
                    or None,
                    "board_default_probe": dict(
                        record.get("prior_board_default_probe", {}) or {}
                    ),
                },
            }
        return {
            "game_id": str(getattr(record, "game_id", "") or ""),
            "step_index": int(getattr(record, "step_index", 0) or 0),
            "action_name": str(getattr(record, "action_name", "") or ""),
            "outcome": str(getattr(record, "outcome", "") or ""),
            "progress_delta": int(getattr(record, "progress_delta", 0) or 0),
            "game_state": str(
                getattr(record, "game_state", "")
                or getattr(record, "next_game_state", "")
                or ""
            ),
            "prior_game_state": str(getattr(record, "prior_game_state", "") or ""),
            "prior_frame_signature": str(
                getattr(record, "prior_frame_signature", "") or ""
            ),
            "next_frame_signature": str(
                getattr(record, "next_frame_signature", "") or ""
            ),
            "decision_subgoal": str(getattr(record, "decision_subgoal", "") or ""),
            "decision_source": str(getattr(record, "decision_source", "") or ""),
            "stagnation_before": int(getattr(record, "stagnation_before", 0) or 0),
            "stagnation_after": int(getattr(record, "stagnation_after", 0) or 0),
            "prior_legal_actions": list(
                getattr(record, "prior_legal_actions", []) or []
            ),
            "board_context_summary": {
                "board_height": int(getattr(record, "prior_board_height", 0) or 0),
                "board_width": int(getattr(record, "prior_board_width", 0) or 0),
                "board_active_cells": int(
                    getattr(record, "prior_board_active_cells", 0) or 0
                ),
                "board_source_attr": str(
                    getattr(record, "prior_board_source_attr", "") or ""
                )
                or None,
                "board_default_probe": dict(
                    getattr(record, "prior_board_default_probe", {}) or {}
                ),
            },
        }

    def _fallback_basin_status(
        self,
        *,
        records: list[Any],
        observation: Observation,
        basin_action_stats: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        state_name = str(observation.game_state or "")
        frame_signature = str(
            observation.derived_features.get("frame_signature", "") or ""
        )
        visits = 0
        actions_attempted = 0
        progress_events = 0
        failed_events = 0
        stalled_events = 0
        last_action_name = None
        last_progress_step = None
        for record in records:
            row = self._record_to_recent_record(record)
            if str(row.get("prior_game_state") or "") != state_name:
                continue
            record_frame_signature = str(row.get("prior_frame_signature") or "")
            if frame_signature and record_frame_signature != frame_signature:
                continue
            visits += 1
            actions_attempted += 1
            action_name = str(row.get("action_name") or "")
            if action_name:
                last_action_name = action_name
            outcome = str(row.get("outcome") or "")
            if outcome == "progress":
                progress_events += 1
                last_progress_step = int(row.get("step_index", 0) or 0)
            elif outcome == "failed":
                failed_events += 1
            elif outcome == "stalled":
                stalled_events += 1
        return {
            "state": state_name,
            "frame_signature": frame_signature,
            "visits": visits,
            "actions_attempted": actions_attempted,
            "distinct_actions_tried": len(basin_action_stats),
            "progress_events": progress_events,
            "failed_events": failed_events,
            "stalled_events": stalled_events,
            "last_action_name": last_action_name,
            "last_progress_step": last_progress_step,
            "is_exhausted": self._fallback_is_basin_exhausted(
                observation=observation,
                basin_action_stats=basin_action_stats,
            ),
            "action_outcomes": basin_action_stats,
        }

    def _fallback_is_basin_exhausted(
        self,
        *,
        observation: Observation,
        basin_action_stats: dict[str, dict[str, Any]],
    ) -> bool:
        state_name = str(observation.game_state or "")
        if not state_name:
            return False
        legal_actions = [action for action in observation.legal_actions if action != "RESET"]
        if not legal_actions:
            return False
        frame_signature = str(
            observation.derived_features.get("frame_signature", "") or ""
        )
        exhausted_actions = 0
        for action_name in legal_actions:
            counts = basin_action_stats.get(action_name, {})
            if int(counts.get("progress", 0) or 0) > 0:
                return False
            failed = int(counts.get("failed", 0) or 0)
            stalled = int(counts.get("stalled", 0) or 0)
            if not frame_signature and len(legal_actions) == 1:
                if failed > 0 or stalled >= 2:
                    exhausted_actions += 1
                continue
            if failed > 0 or stalled > 0:
                exhausted_actions += 1
        return exhausted_actions == len(legal_actions)

    def _memory_bundle_as_learned(
        self,
        bundle: dict[str, Any],
        *,
        subgoal_name: str | None = None,
    ) -> dict[str, Any]:
        episode = self.facts.get("episode", {})
        state_name = str(episode.get("game_state", "") or "")
        frame_signature = str(episode.get("frame_signature", "") or "")
        learned = {
            "actions": dict(bundle.get("action_stats", {})),
            "states": {},
            "subgoals": {},
            "frame_signatures": {},
            "records_seen": len(bundle.get("recent_records", [])),
        }
        if state_name:
            learned["states"][state_name] = {
                "actions": dict(bundle.get("state_action_stats", {}))
            }
        if frame_signature:
            learned["frame_signatures"][frame_signature] = {
                "actions": dict(bundle.get("frame_action_stats", {}))
            }
        subgoal_actions = dict(bundle.get("subgoal_action_stats", {}))
        resolved_subgoal_name = str(subgoal_name or "").strip()
        if not resolved_subgoal_name:
            resolved_subgoal_name = str(
                (
                    dict(bundle.get("working_memory", {}) or {}).get("active_subgoal")
                    or dict(bundle.get("retrieval_summary", {}) or {}).get("active_subgoal")
                    or ""
                )
            ).strip()
        if not resolved_subgoal_name:
            recent_subgoals = {
                str(record.get("decision_subgoal", "") or "").strip()
                for record in list(bundle.get("recent_records", []) or [])
                if str(record.get("decision_subgoal", "") or "").strip()
            }
            if len(recent_subgoals) == 1:
                resolved_subgoal_name = next(iter(recent_subgoals))
        if subgoal_actions and resolved_subgoal_name:
            learned["subgoals"][resolved_subgoal_name] = {"actions": subgoal_actions}
        return learned

    def _lookup_symbolic_prior(self, action_name: str) -> float | None:
        value = self._evaluate_scalar(f"!(symbolic-prior {self._symbol(action_name)})")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _refresh_runtime(self) -> None:
        if not self._ensure_runtime_capability():
            return
        fact_program = self._build_fact_program()
        if self.available and self._runtime is not None and self._last_fact_program == fact_program:
            return
        try:
            self._prepare_runtime_for_sync(fact_program)
            self.available = True
            self.last_error = None
        except Exception as exc:
            self._disable_runtime(f"failed to initialize Hyperon runtime: {exc}")

    def _prepare_runtime_for_sync(self, fact_program: str) -> None:
        if self._runtime is None:
            self._runtime = self._create_runtime()
            self._run_program(self.rules_text)
            self._rules_loaded = True
        elif self._last_fact_program != fact_program and not self._clear_runtime_facts():
            self._runtime = self._create_runtime()
            self._run_program(self.rules_text)
            self._rules_loaded = True
        self._run_program(fact_program)
        self._last_fact_program = fact_program

    def _clear_runtime_facts(self) -> bool:
        if self._runtime is None:
            return False
        for method_name in ("clear", "reset", "reset_space"):
            method = getattr(self._runtime, method_name, None)
            if callable(method):
                method()
                if self.rules_text:
                    self._run_program(self.rules_text)
                self._rules_loaded = True
                return True
        return False

    def _ensure_runtime_capability(self) -> bool:
        if not self.enabled:
            self.available = False
            self.last_error = "hyperon disabled by configuration"
            return False
        if self.backend is None:
            self.available = False
            self.last_error = "hyperon package is not importable"
            return False
        if self.skip_probe:
            self._probe_attempted = True
            self._probe_ok = True
            self.last_error = None
            return True
        if not self._probe_attempted:
            self._probe_attempted = True
            self._probe_ok, self.last_error = self._probe_runtime()
        if not self._probe_ok:
            self.available = False
            self._runtime = None
            return False
        return True

    def _probe_runtime(self) -> tuple[bool, str | None]:
        env = dict(os.environ)
        env["AGI_HYPERON_PROBE_ROOT"] = str(self.project_root)
        command = [
            self.python_executable,
            "-c",
            (
                "import os; "
                "from hyperon import Environment, MeTTa; "
                "root = os.environ['AGI_HYPERON_PROBE_ROOT']; "
                "MeTTa(env_builder=Environment.custom_env(working_dir=root, config_dir='')); "
                "print('ok')"
            ),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                cwd=self.project_root,
                env=env,
                text=True,
                timeout=self.probe_timeout_seconds,
            )
        except Exception as exc:
            return False, f"hyperon probe failed: {exc}"
        if completed.returncode == 0 and "ok" in completed.stdout:
            return True, None
        details = (completed.stderr or completed.stdout or "").strip()
        if details:
            details = details.splitlines()[-1]
        return False, details or "hyperon probe returned a non-zero exit code"

    def _create_runtime(self) -> Any:
        env_builder = self.backend.Environment.custom_env(
            working_dir=str(self.project_root),
            config_dir="",
        )
        return self.backend.MeTTa(env_builder=env_builder)

    def _run_program(self, program: str) -> list[list[Any]]:
        if self._runtime is None or not program.strip():
            return []
        return self._runtime.run(program)

    def _build_fact_program(self) -> str:
        lines: list[str] = []
        episode = self.facts.get("episode", {})
        memory = self.facts.get("memory", {})
        lines.append(f"(episode-game {self._symbol(episode.get('game_id'))})")
        board_type = episode.get("board_type", "unknown")
        if board_type:
            lines.append(f"(board-type {self._symbol(board_type)})")
        lines.append(f"(episode-step {int(episode.get('step_index', 0) or 0)})")
        lines.append(f"(episode-state {self._symbol(episode.get('game_state'))})")
        lines.append(
            f"(episode-levels-completed {int(episode.get('levels_completed', 0) or 0)})"
        )
        last_action = episode.get("last_action")
        if last_action:
            lines.append(f"(episode-last-action {self._symbol(last_action)})")
        for action_name in episode.get("legal_actions", []):
            lines.append(f"(legal-action {self._symbol(action_name)})")
        for goal in self.facts.get("goals", []):
            lines.append(
                "(goal-state "
                f"{self._symbol(goal.get('name'))} "
                f"{self._symbol(goal.get('target_state'))} "
                f"{float(goal.get('priority', 1.0) or 1.0)})"
            )
        lines.append(
            f"(memory-stagnation {int(memory.get('stagnation_steps', 0) or 0)})"
        )
        lines.append(f"(memory-llm-calls {int(memory.get('llm_calls', 0) or 0)})")
        for action_name in memory.get("recent_actions", []):
            lines.append(f"(recent-action {self._symbol(action_name)})")
        for action_name, count in memory.get("success_counts", {}).items():
            lines.append(f"(success-count {self._symbol(action_name)} {int(count)})")
        for action_name, count in memory.get("failure_streaks", {}).items():
            lines.append(f"(failure-streak {self._symbol(action_name)} {int(count)})")
        learned = self.facts.get("learned", {})
        for action_name, stats in learned.get("actions", {}).items():
            lines.append(
                f"(learned-progress-count {self._symbol(action_name)} {int(stats.get('progress', 0) or 0)})"
            )
            lines.append(
                f"(learned-failed-count {self._symbol(action_name)} {int(stats.get('failed', 0) or 0)})"
            )
            lines.append(
                f"(learned-stalled-count {self._symbol(action_name)} {int(stats.get('stalled', 0) or 0)})"
            )
        for state_name, state_entry in learned.get("states", {}).items():
            for action_name, stats in state_entry.get("actions", {}).items():
                lines.append(
                    f"(learned-state-progress-count {self._symbol(state_name)} {self._symbol(action_name)} {int(stats.get('progress', 0) or 0)})"
                )
                lines.append(
                    f"(learned-state-failed-count {self._symbol(state_name)} {self._symbol(action_name)} {int(stats.get('failed', 0) or 0)})"
                )
                lines.append(
                    f"(learned-state-stalled-count {self._symbol(state_name)} {self._symbol(action_name)} {int(stats.get('stalled', 0) or 0)})"
                )
        for subgoal_name, subgoal_entry in learned.get("subgoals", {}).items():
            for action_name, stats in subgoal_entry.get("actions", {}).items():
                lines.append(
                    f"(learned-subgoal-progress-count {self._symbol(subgoal_name)} {self._symbol(action_name)} {int(stats.get('progress', 0) or 0)})"
                )
                lines.append(
                    f"(learned-subgoal-failed-count {self._symbol(subgoal_name)} {self._symbol(action_name)} {int(stats.get('failed', 0) or 0)})"
                )
                lines.append(
                    f"(learned-subgoal-stalled-count {self._symbol(subgoal_name)} {self._symbol(action_name)} {int(stats.get('stalled', 0) or 0)})"
                )
        proposal_program = self._build_proposal_fact_program()
        if proposal_program:
            lines.append(proposal_program)
        return "\n".join(line for line in lines if line)

    def _build_proposal_fact_program(self) -> str:
        lines: list[str] = []
        for proposal in self.facts.get("proposals", []):
            lines.append(
                "(proposal "
                f"{self._symbol(proposal.get('action_name'))} "
                f"{self._symbol(proposal.get('source'))} "
                f"{float(proposal.get('confidence', 0.0) or 0.0)} "
                f"{self._symbol(proposal.get('expected_outcome') or 'unknown')})"
            )
        return "\n".join(lines)

    def _evaluate_scalar(self, program: str) -> Any | None:
        rows = self._run_program(program)
        if not rows or not rows[0]:
            return None
        return self._atom_to_python(rows[0][0])

    def _atom_to_python(self, atom: Any) -> Any:
        if self.backend is None:
            return repr(atom)
        if isinstance(atom, self.backend.SymbolAtom):
            return atom.get_name()
        if isinstance(atom, self.backend.VariableAtom):
            return atom.get_name()
        if isinstance(atom, self.backend.ExpressionAtom):
            return [self._atom_to_python(child) for child in atom.get_children()]
        if isinstance(atom, self.backend.GroundedAtom):
            try:
                obj = atom.get_object()
            except Exception:
                return repr(atom)
            return getattr(obj, "value", obj)
        return repr(atom)

    def _disable_runtime(self, message: str) -> None:
        self.available = False
        self._runtime = None
        self._rules_loaded = False
        self._last_fact_program = None
        self.last_error = message

    def _try_import_hyperon(self) -> Any:
        try:
            return importlib.import_module("hyperon")
        except Exception:
            return None

    def _load_rules_text(self) -> str:
        if self.rules_path is None or not self.rules_path.exists():
            return ""
        return self.rules_path.read_text(encoding="utf-8")

    def _load_learned_knowledge(self) -> dict[str, Any]:
        if self.learning_path is None or not self.learning_path.exists():
            return {"games": {}}
        try:
            payload = json.loads(self.learning_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"games": {}}
        if not isinstance(payload, dict):
            return {"games": {}}
        payload.setdefault("games", {})
        return payload

    def _persist_learned_knowledge(self) -> None:
        if self.learning_path is None:
            return
        self.learning_path.parent.mkdir(parents=True, exist_ok=True)
        self.learning_path.write_text(
            json.dumps(self.learned_knowledge, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _ingest_episode_records(self, records: list[Any]) -> None:
        changed = False
        for record in records:
            key = self._record_key(record)
            if key in self._ingested_record_keys:
                continue
            self._ingested_record_keys.add(key)
            self._apply_record_learning(record)
            changed = True
        if changed:
            self._persist_learned_knowledge()

    def _apply_record_learning(self, record: Any) -> None:
        game_id = str(self._record_field(record, "game_id", "") or "")
        action_name = str(self._record_field(record, "action_name", "") or "")
        outcome_name = str(self._record_field(record, "outcome", "") or "")
        prior_game_state = str(self._record_field(record, "prior_game_state", "") or "")
        prior_frame_signature = str(
            self._record_field(record, "prior_frame_signature", "") or ""
        )
        decision_subgoal = str(self._record_field(record, "decision_subgoal", "") or "")
        games = self.learned_knowledge.setdefault("games", {})
        game_entry = games.setdefault(
            game_id,
            {
                "actions": {},
                "states": {},
                "subgoals": {},
                "frame_signatures": {},
                "records_seen": 0,
            },
        )
        game_entry.setdefault("frame_signatures", {})
        action_entry = game_entry["actions"].setdefault(
            action_name,
            {"progress": 0, "failed": 0, "stalled": 0, "total": 0, "last_outcome": None},
        )
        action_entry["total"] = int(action_entry.get("total", 0) or 0) + 1
        if outcome_name == "progress":
            action_entry["progress"] = int(action_entry.get("progress", 0) or 0) + 1
        elif outcome_name == "failed":
            action_entry["failed"] = int(action_entry.get("failed", 0) or 0) + 1
        else:
            action_entry["stalled"] = int(action_entry.get("stalled", 0) or 0) + 1
        action_entry["last_outcome"] = outcome_name
        if prior_game_state:
            state_entry = game_entry["states"].setdefault(prior_game_state, {"actions": {}})
            self._apply_context_learning(
                state_entry["actions"].setdefault(
                    action_name,
                    {"progress": 0, "failed": 0, "stalled": 0, "total": 0, "last_outcome": None},
                ),
                outcome_name,
            )
        if prior_frame_signature:
            frame_entry = game_entry["frame_signatures"].setdefault(
                prior_frame_signature,
                {"actions": {}},
            )
            self._apply_context_learning(
                frame_entry["actions"].setdefault(
                    action_name,
                    {"progress": 0, "failed": 0, "stalled": 0, "total": 0, "last_outcome": None},
                ),
                outcome_name,
            )
        if decision_subgoal:
            subgoal_entry = game_entry["subgoals"].setdefault(decision_subgoal, {"actions": {}})
            self._apply_context_learning(
                subgoal_entry["actions"].setdefault(
                    action_name,
                    {"progress": 0, "failed": 0, "stalled": 0, "total": 0, "last_outcome": None},
                ),
                outcome_name,
            )
        game_entry["records_seen"] = int(game_entry.get("records_seen", 0) or 0) + 1

    def _summarize_learned_knowledge(self, game_id: str) -> dict[str, Any]:
        games = self.learned_knowledge.get("games", {})
        summary = None
        for alias in game_id_aliases(game_id):
            candidate = games.get(alias)
            if candidate is not None:
                summary = candidate
                break
        if summary is None:
            summary = {"actions": {}, "states": {}, "subgoals": {}, "records_seen": 0}
        summary.setdefault("actions", {})
        summary.setdefault("states", {})
        summary.setdefault("subgoals", {})
        summary.setdefault("records_seen", 0)
        summary.setdefault("frame_signatures", {})
        return summary

    def _load_trained_prior_bucket(
        self,
        observation: Observation,
        *,
        subgoal: str | None = None,
    ) -> dict[str, int]:
        if self.trained_priors_db_path is None:
            return {}
        try:
            bucket = load_best_prior_bucket(
                db_path=self.trained_priors_db_path,
                game_id=observation.game_id,
                state=observation.game_state,
                frame_signature=str(
                    observation.derived_features.get("frame_signature", "") or ""
                ),
                legal_actions=list(observation.legal_actions),
                subgoal=subgoal,
            )
            self._trained_priors_error = None
            return bucket
        except Exception as exc:
            self._trained_priors_error = repr(exc)
            return {}

    def _apply_context_learning(self, entry: dict[str, Any], outcome_name: str) -> None:
        entry["total"] = int(entry.get("total", 0) or 0) + 1
        if outcome_name == "progress":
            entry["progress"] = int(entry.get("progress", 0) or 0) + 1
        elif outcome_name == "failed":
            entry["failed"] = int(entry.get("failed", 0) or 0) + 1
        else:
            entry["stalled"] = int(entry.get("stalled", 0) or 0) + 1
        entry["last_outcome"] = outcome_name

    def _parse_query(self, query: str) -> tuple[str, str | None]:
        head, separator, tail = query.partition(":")
        if not separator:
            return query, None
        return head, tail or None

    def _resolve_project_root(
        self,
        project_root: str | Path | None,
        rules_path: Path | None,
    ) -> Path:
        if project_root is not None:
            return Path(project_root)
        if rules_path is not None:
            return rules_path.resolve().parents[3]
        return Path(__file__).resolve().parents[2]

    def _symbol(self, value: Any) -> str:
        text = str(value or "NONE").strip()
        cleaned = [
            char if char.isalnum() or char in {"-", "_", "."} else "_"
            for char in text
        ]
        symbol = "".join(cleaned) or "NONE"
        if symbol[0].isdigit():
            symbol = f"v_{symbol}"
        return symbol
