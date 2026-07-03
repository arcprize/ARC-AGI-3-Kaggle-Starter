from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from agi_core.storage import load_best_prior_bucket
from agi_core.types import ActionProposal, GoalState, Observation


class HyperonBridge:
    """Defensive Hyperon wrapper with a deterministic shadow fact store."""

    def __init__(
        self,
        rules_path: str | Path | None = None,
        learning_path: str | Path | None = None,
        trained_priors_db_path: str | Path | None = None,
        project_root: str | Path | None = None,
        enabled: bool = True,
        skip_probe: bool = False,
        python_executable: str | Path | None = None,
        probe_timeout_seconds: float = 5.0,
    ) -> None:
        self.rules_path = Path(rules_path) if rules_path is not None else None
        self.learning_path = Path(learning_path) if learning_path is not None else None
        self.trained_priors_db_path = (
            Path(trained_priors_db_path) if trained_priors_db_path is not None else None
        )
        self.project_root = self._resolve_project_root(project_root, self.rules_path)
        self.enabled = enabled
        self.skip_probe = skip_probe
        self.python_executable = str(python_executable or sys.executable)
        self.probe_timeout_seconds = probe_timeout_seconds
        self.rules_text = self._load_rules_text()
        self.backend = self._try_import_hyperon()
        self.available = False
        self.facts: dict[str, Any] = {}
        self.last_error: str | None = None
        self.learned_knowledge = self._load_learned_knowledge()
        self._probe_attempted = False
        self._probe_ok = False
        self._runtime: Any = None
        self._trained_priors_error: str | None = None
        self._ingested_record_keys: set[
            tuple[str, int, str, str, int, str, str | None, str | None, str | None]
        ] = set()

    def sync(self, observation: Observation, memory: Any, goals: list[GoalState]) -> None:
        self._ingest_episode_records(memory.records)
        self.facts = {
            "episode": {
                "game_id": observation.game_id,
                "step_index": observation.step_index,
                "game_state": observation.game_state,
                "last_action": observation.last_action,
                "levels_completed": observation.levels_completed,
                "legal_actions": list(observation.legal_actions),
                "frame_signature": observation.derived_features.get("frame_signature"),
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
            "learned": self._summarize_learned_knowledge(observation.game_id),
            "trained_prior_bucket": self._load_trained_prior_bucket(observation),
            "proposals": [],
        }
        self._refresh_runtime()

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
            if subgoal is None:
                subgoal = self._evaluate_scalar(f"!(goal-hint {state_name})")
            if subgoal is None:
                return []
            confidence = 0.95 if episode.get("game_state") == "WIN" else 0.72
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

    def _run_shadow_rules(self, query: str) -> list[dict[str, Any]]:
        episode = self.facts.get("episode", {})
        memory = self.facts.get("memory", {})
        query_name, query_arg = self._parse_query(query)
        legal_actions = list(episode.get("legal_actions", []))
        if query_name == "reset-needed":
            if episode.get("game_state") in {"NOT_PLAYED", "GAME_OVER"}:
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
            if episode.get("game_state") == "WIN":
                return [{"subgoal": "hold-win", "confidence": 0.95}]
            if memory.get("stagnation_steps", 0):
                return [{"subgoal": "break-stall", "confidence": 0.72}]
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
        return []

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
        learned = learned or {}
        learned_actions = learned.get("actions", {})
        learned_states = learned.get("states", {})
        learned_subgoals = learned.get("subgoals", {})
        learned_frame_signatures = learned.get("frame_signatures", {})
        state_actions = learned_states.get(current_state, {}).get("actions", {})
        subgoal_actions = learned_subgoals.get(current_subgoal or "", {}).get("actions", {})
        frame_actions = learned_frame_signatures.get(current_frame_signature, {}).get(
            "actions", {}
        )
        recent_actions = memory.get("recent_actions", [])
        most_recent = recent_actions[-1] if recent_actions else None
        stagnation_steps = int(memory.get("stagnation_steps", 0) or 0)
        offline_actions = dict(self.facts.get("trained_prior_bucket", {}))
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
                frame_actions.get(action_name, {}),
                success_scale=0.04,
                failed_scale=0.015,
                stalled_scale=0.01,
                positive_cap=0.18,
                negative_cap=0.06,
            )
            confidence += self._contextual_learning_bias(
                state_actions.get(action_name, {}),
                success_scale=0.025,
                failed_scale=0.025,
                stalled_scale=0.015,
                positive_cap=0.12,
                negative_cap=0.12,
            )
            confidence += self._contextual_learning_bias(
                subgoal_actions.get(action_name, {}),
                success_scale=0.03,
                failed_scale=0.02,
                stalled_scale=0.01,
                positive_cap=0.14,
                negative_cap=0.10,
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
        try:
            self._runtime = self._create_runtime()
            self._run_program(self.rules_text)
            self._run_program(self._build_fact_program())
            self.available = True
            self.last_error = None
        except Exception as exc:
            self._disable_runtime(f"failed to initialize Hyperon runtime: {exc}")

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
            key = (
                record.game_id,
                int(record.step_index),
                record.action_name,
                record.outcome,
                int(record.progress_delta),
                record.game_state,
                record.prior_game_state,
                record.decision_subgoal,
                record.prior_frame_signature,
            )
            if key in self._ingested_record_keys:
                continue
            self._ingested_record_keys.add(key)
            self._apply_record_learning(record)
            changed = True
        if changed:
            self._persist_learned_knowledge()

    def _apply_record_learning(self, record: Any) -> None:
        games = self.learned_knowledge.setdefault("games", {})
        game_entry = games.setdefault(
            record.game_id,
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
            record.action_name,
            {"progress": 0, "failed": 0, "stalled": 0, "total": 0, "last_outcome": None},
        )
        outcome_name = str(record.outcome)
        action_entry["total"] = int(action_entry.get("total", 0) or 0) + 1
        if outcome_name == "progress":
            action_entry["progress"] = int(action_entry.get("progress", 0) or 0) + 1
        elif outcome_name == "failed":
            action_entry["failed"] = int(action_entry.get("failed", 0) or 0) + 1
        else:
            action_entry["stalled"] = int(action_entry.get("stalled", 0) or 0) + 1
        action_entry["last_outcome"] = outcome_name
        state_name = str(record.prior_game_state or "")
        if state_name:
            state_entry = game_entry["states"].setdefault(state_name, {"actions": {}})
            self._apply_context_learning(
                state_entry["actions"].setdefault(
                    record.action_name,
                    {"progress": 0, "failed": 0, "stalled": 0, "total": 0, "last_outcome": None},
                ),
                outcome_name,
            )
        frame_signature = str(record.prior_frame_signature or "")
        if frame_signature:
            frame_entry = game_entry["frame_signatures"].setdefault(
                frame_signature,
                {"actions": {}},
            )
            self._apply_context_learning(
                frame_entry["actions"].setdefault(
                    record.action_name,
                    {"progress": 0, "failed": 0, "stalled": 0, "total": 0, "last_outcome": None},
                ),
                outcome_name,
            )
        subgoal_name = str(record.decision_subgoal or "")
        if subgoal_name:
            subgoal_entry = game_entry["subgoals"].setdefault(subgoal_name, {"actions": {}})
            self._apply_context_learning(
                subgoal_entry["actions"].setdefault(
                    record.action_name,
                    {"progress": 0, "failed": 0, "stalled": 0, "total": 0, "last_outcome": None},
                ),
                outcome_name,
            )
        game_entry["records_seen"] = int(game_entry.get("records_seen", 0) or 0) + 1

    def _summarize_learned_knowledge(self, game_id: str) -> dict[str, Any]:
        games = self.learned_knowledge.get("games", {})
        summary = games.get(
            game_id,
            {"actions": {}, "states": {}, "subgoals": {}, "records_seen": 0},
        )
        summary.setdefault("actions", {})
        summary.setdefault("states", {})
        summary.setdefault("subgoals", {})
        summary.setdefault("records_seen", 0)
        summary.setdefault("frame_signatures", {})
        return summary

    def _load_trained_prior_bucket(self, observation: Observation) -> dict[str, int]:
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
