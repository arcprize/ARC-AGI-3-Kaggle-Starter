from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict

from .image_input import board_context_summary
from .types import ActionDecision, EpisodeRecord, Observation


class EpisodicMemory:
    """Tracks recent transitions and lightweight action statistics."""

    def __init__(
        self,
        history_limit: int = 64,
        *,
        win_state_name: str = "WIN",
        reset_state_names: tuple[str, ...] = ("NOT_PLAYED", "GAME_OVER"),
    ) -> None:
        self.history_limit = history_limit
        self._win_state_name = str(win_state_name or "").upper()
        self._reset_state_names = {str(name or "").upper() for name in reset_state_names}
        self._records: deque[EpisodeRecord] = deque(maxlen=history_limit)
        self._new_records: deque[EpisodeRecord] = deque()
        self._recent_actions: deque[str] = deque(maxlen=16)
        self._success_counts: defaultdict[str, int] = defaultdict(int)
        self._failure_streaks: defaultdict[str, int] = defaultdict(int)
        self._llm_calls = 0
        self._stagnation_steps = 0
        self._observation_tick = 0
        self._game_id = ""
        self._pending_observation: Observation | None = None
        self._pending_decision: ActionDecision | None = None
        self._pending_score_floor_reopen: dict[str, object] | None = None
        self._recent_score_floor_fallback: dict[str, object] | None = None
        self._recent_probe_dead_end_escape: dict[str, object] | None = None

    def bind_game(self, game_id: str) -> None:
        if game_id != self._game_id:
            self._game_id = game_id
            self._reset_episode_state()

    def note_observation(self, observation: Observation) -> None:
        self.bind_game(observation.game_id)
        if self._is_same_game_restart_boundary(observation):
            self._reset_episode_state()
        self._observation_tick += 1
        if self._pending_observation is not None and self._pending_decision is not None:
            self._finalize_transition(observation)
        self._pending_observation = observation

    def note_decision(self, decision: ActionDecision) -> None:
        self._pending_decision = decision
        self._recent_actions.append(decision.action_name)
        if decision.action_name == "RESET":
            self.clear_pending_score_floor_reopen()
        if (
            decision.source == "fallback"
            and str(getattr(decision, "reason", "") or "").startswith("score-floor")
            and self._pending_observation is not None
        ):
            self._recent_score_floor_fallback = {
                "step_index": int(self._pending_observation.step_index),
                "step_index_reliable": bool(
                    self._pending_observation.derived_features.get(
                        "step_index_reliable",
                        True,
                    )
                ),
                "observation_tick": int(self._observation_tick),
                "state_name": str(self._pending_observation.game_state or ""),
            }
        if (
            decision.source == "fallback"
            and str(getattr(decision, "reason", "") or "")
            in {"probe-dead-end memory escape", "probe-dead-end executive escape"}
            and self._pending_observation is not None
        ):
            self._recent_probe_dead_end_escape = {
                "step_index": int(self._pending_observation.step_index),
                "step_index_reliable": bool(
                    self._pending_observation.derived_features.get(
                        "step_index_reliable",
                        True,
                    )
                ),
                "observation_tick": int(self._observation_tick),
                "state_name": str(self._pending_observation.game_state or ""),
            }

    def note_llm_call(self) -> None:
        self._llm_calls += 1

    def can_call_llm(self, limit: int) -> bool:
        return self._llm_calls < limit

    @property
    def stagnation_steps(self) -> int:
        return self._stagnation_steps

    @property
    def llm_calls(self) -> int:
        return self._llm_calls

    @property
    def records(self) -> list[EpisodeRecord]:
        return list(self._records)

    def recent_actions(self) -> list[str]:
        return list(self._recent_actions)

    def mark_pending_score_floor_reopen(
        self,
        observation: Observation,
        *,
        candidate_actions: list[str],
        debug: dict[str, object],
    ) -> None:
        state_name = str(observation.game_state or "")
        filtered_actions = [
            str(action_name)
            for action_name in candidate_actions
            if action_name and action_name != "RESET"
        ]
        if not state_name or not filtered_actions:
            self.clear_pending_score_floor_reopen()
            return
        summary = dict(debug or {})
        active_summary = dict(summary.get("active_reopen") or {})
        frame_signatures = [
            str(frame_signature)
            for frame_signature in (
                active_summary.get("frame_signatures")
                or summary.get("frame_signatures")
                or []
            )
            if frame_signature
        ]
        self._pending_score_floor_reopen = {
            "state_name": state_name,
            "step_index": int(observation.step_index),
            "step_index_reliable": bool(
                observation.derived_features.get("step_index_reliable", True)
            ),
            "observation_tick": int(self._observation_tick),
            "candidate_actions": filtered_actions,
            "frame_signatures": frame_signatures,
            "score_floor_debug": summary,
        }

    def pending_score_floor_reopen(
        self,
        observation: Observation,
    ) -> dict[str, object] | None:
        pending = self._pending_score_floor_reopen
        if pending is None:
            return None
        state_name = str(observation.game_state or "")
        if (
            not state_name
            or state_name != str(pending.get("state_name") or "")
            or state_name == self._win_state_name
            or state_name in self._reset_state_names
        ):
            self.clear_pending_score_floor_reopen()
            return None
        if self._step_delta_exceeds_limit(
            observation,
            pending,
            limit=4,
            clear_on_negative=False,
        ):
            self.clear_pending_score_floor_reopen()
            return None
        candidate_actions = [
            str(action_name)
            for action_name in list(pending.get("candidate_actions") or [])
            if action_name
        ]
        if not candidate_actions:
            self.clear_pending_score_floor_reopen()
            return None
        current_actions = {
            str(action_name)
            for action_name in observation.legal_actions
            if action_name and action_name != "RESET"
        }
        if "RESET" not in observation.legal_actions and current_actions:
            if not any(action_name in current_actions for action_name in candidate_actions):
                self.clear_pending_score_floor_reopen()
                return None
        return dict(pending)

    def clear_pending_score_floor_reopen(self) -> None:
        self._pending_score_floor_reopen = None

    def recent_score_floor_fallback(self, observation: Observation) -> bool:
        pending = self._recent_score_floor_fallback
        if pending is None:
            return False
        if str(observation.game_state or "") != str(pending.get("state_name") or ""):
            self._recent_score_floor_fallback = None
            return False
        if self._step_delta_exceeds_limit(
            observation,
            pending,
            limit=1,
            clear_on_negative=True,
        ):
            self._recent_score_floor_fallback = None
            return False
        return True

    def recent_probe_dead_end_escape(self, observation: Observation) -> bool:
        pending = self._recent_probe_dead_end_escape
        if pending is None:
            return False
        if str(observation.game_state or "") != str(pending.get("state_name") or ""):
            self._recent_probe_dead_end_escape = None
            return False
        if self._step_delta_exceeds_limit(
            observation,
            pending,
            limit=2,
            clear_on_negative=True,
        ):
            self._recent_probe_dead_end_escape = None
            return False
        return True

    def drain_new_records(self) -> list[EpisodeRecord]:
        drained = list(self._new_records)
        self._new_records.clear()
        return drained

    def most_recent_action(self) -> str | None:
        if not self._recent_actions:
            return None
        return self._recent_actions[-1]

    def repeated_failure_penalty(self, action_name: str) -> float:
        streak = self._failure_streaks.get(action_name, 0)
        recent_repeat = 0.0
        if self.most_recent_action() == action_name and self._stagnation_steps:
            recent_repeat = 0.05
        return min(0.60, (streak * 0.15) + recent_repeat)

    def loop_penalty(self, action_name: str) -> float:
        if self._stagnation_steps < 2:
            return 0.0
        recent = list(self._recent_actions)
        if len(recent) >= 4:
            tail = recent[-4:]
            if tail[0] == tail[2] and tail[1] == tail[3] and tail[0] != tail[1]:
                if action_name in {tail[0], tail[1]}:
                    return 0.18
        if len(recent) >= 3:
            tail = recent[-3:]
            if tail[0] == tail[1] == tail[2] == action_name:
                return 0.18
        return 0.0

    def success_bonus(self, action_name: str) -> float:
        return min(0.15, self._success_counts.get(action_name, 0) * 0.03)

    def context_stall_penalty(
        self,
        action_name: str,
        *,
        state_name: str,
        frame_signature: str | None,
    ) -> float:
        if not state_name:
            return 0.0
        stalled_hits = 0
        failed_hits = 0
        progress_hits = 0
        for record in reversed(self._records):
            if record.action_name != action_name:
                continue
            if record.prior_game_state != state_name:
                continue
            if frame_signature and record.prior_frame_signature != frame_signature:
                continue
            if record.outcome == "progress":
                progress_hits += 1
            elif record.outcome == "failed":
                failed_hits += 1
            elif record.outcome == "stalled":
                stalled_hits += 1
        if progress_hits > 0:
            return 0.0
        if failed_hits >= 1:
            return min(0.30, 0.18 + (failed_hits - 1) * 0.06)
        if stalled_hits >= 3:
            return 0.24
        if stalled_hits >= 2:
            return 0.16
        return 0.0

    def context_outcome_counts(
        self,
        action_name: str,
        *,
        state_name: str,
        frame_signature: str | None,
    ) -> dict[str, int]:
        counts = {"progress": 0, "failed": 0, "stalled": 0, "total": 0}
        if not state_name:
            return counts
        for record in reversed(self._records):
            if record.action_name != action_name:
                continue
            if record.prior_game_state != state_name:
                continue
            if frame_signature and record.prior_frame_signature != frame_signature:
                continue
            counts["total"] += 1
            if record.outcome in counts:
                counts[record.outcome] += 1
        return counts

    def current_basin_key(self, observation: Observation) -> tuple[str, str] | None:
        state_name = str(observation.game_state or "")
        frame_signature = str(observation.derived_features.get("frame_signature", "") or "")
        if not state_name or not frame_signature:
            return None
        return state_name, frame_signature

    def fallback_attempted_actions(
        self,
        observation: Observation,
        action_names: list[str] | None = None,
    ) -> set[str]:
        state_name = str(observation.game_state or "")
        frame_signature = str(observation.derived_features.get("frame_signature", "") or "")
        if not state_name or not frame_signature:
            return set()
        allowed = set(action_names or [])
        attempted: set[str] = set()
        for record in self._records:
            if allowed and record.action_name not in allowed:
                continue
            if record.prior_game_state != state_name:
                continue
            if record.prior_frame_signature != frame_signature:
                continue
            if record.decision_source != "fallback":
                continue
            attempted.add(record.action_name)
        return attempted

    def replan_frontier_attempted_actions(
        self,
        observation: Observation,
        action_names: list[str] | None = None,
    ) -> set[str]:
        state_name = str(observation.game_state or "")
        frame_signature = str(observation.derived_features.get("frame_signature", "") or "")
        if not state_name or not frame_signature:
            return set()
        allowed = set(action_names or [])
        attempted: set[str] = set()
        for record in self._records:
            if allowed and record.action_name not in allowed:
                continue
            if record.prior_game_state != state_name:
                continue
            if record.prior_frame_signature != frame_signature:
                continue
            if record.decision_source != "fallback":
                continue
            if record.decision_subgoal != "replan":
                continue
            attempted.add(record.action_name)
        return attempted

    def replan_frontier_exhausted_actions(
        self,
        observation: Observation,
        action_names: list[str] | None = None,
    ) -> set[str]:
        state_name = str(observation.game_state or "")
        frame_signature = str(observation.derived_features.get("frame_signature", "") or "")
        if not state_name or not frame_signature:
            return set()
        allowed = set(action_names or [])
        exhausted: set[str] = set()
        for action_name in self.replan_frontier_attempted_actions(
            observation,
            action_names=action_names,
        ):
            saw_escape = False
            for record in self._records:
                if record.action_name != action_name:
                    continue
                if allowed and record.action_name not in allowed:
                    continue
                if record.prior_game_state != state_name:
                    continue
                if record.prior_frame_signature != frame_signature:
                    continue
                if record.decision_source != "fallback":
                    continue
                if record.decision_subgoal != "replan":
                    continue
                if record.outcome == "progress":
                    saw_escape = True
                    break
                if record.next_game_state != record.prior_game_state:
                    saw_escape = True
                    break
                if (
                    record.next_frame_signature
                    and record.next_frame_signature != record.prior_frame_signature
                ):
                    saw_escape = True
                    break
            if not saw_escape:
                exhausted.add(action_name)
        return exhausted

    def recent_replan_frame_cycle(
        self,
        observation: Observation,
        *,
        window: int = 8,
    ) -> dict[str, object] | None:
        state_name = str(observation.game_state or "")
        current_frame_signature = str(
            observation.derived_features.get("frame_signature", "") or ""
        )
        if not state_name or not current_frame_signature:
            return None
        matching_records = [
            record
            for record in self._records
            if record.prior_game_state == state_name
            and record.decision_source == "fallback"
            and record.decision_subgoal == "replan"
            and record.outcome in {"stalled", "failed"}
            and record.prior_frame_signature
            and record.next_frame_signature
            and record.prior_frame_signature != record.next_frame_signature
        ]
        if len(matching_records) < 4:
            return None
        tail = matching_records[-min(len(matching_records), window) :]
        if len(tail) < 4:
            return None
        sequence = tail[-4:]
        prior_frames = [str(record.prior_frame_signature or "") for record in sequence]
        next_frames = [str(record.next_frame_signature or "") for record in sequence]
        if len(set(prior_frames)) != 2:
            return None
        if current_frame_signature not in set(prior_frames):
            return None
        if not (
            prior_frames[0] == prior_frames[2]
            and prior_frames[1] == prior_frames[3]
            and prior_frames[0] != prior_frames[1]
        ):
            return None
        if next_frames[0] != prior_frames[1] or next_frames[1] != prior_frames[2]:
            return None
        if next_frames[2] != prior_frames[3]:
            return None
        return {
            "frame_signatures": prior_frames,
            "actions": [record.action_name for record in sequence],
        }

    def is_replan_frontier_reopened(
        self,
        observation: Observation,
        action_names: list[str] | None = None,
        *,
        window: int = 24,
        min_attempts_per_action: int = 2,
    ) -> bool:
        summary = self.replan_frontier_reopen_summary(
            observation,
            action_names=action_names,
            window=window,
            min_attempts_per_action=min_attempts_per_action,
        )
        return bool(summary.get("is_reopened"))

    def replan_frontier_reopen_summary(
        self,
        observation: Observation,
        action_names: list[str] | None = None,
        *,
        window: int = 24,
        min_attempts_per_action: int = 2,
    ) -> dict[str, object]:
        state_name = str(observation.game_state or "")
        candidate_actions = [
            action_name
            for action_name in (action_names or observation.legal_actions)
            if action_name and action_name != "RESET"
        ]
        if not state_name or not candidate_actions:
            return {
                "is_reopened": False,
                "candidate_actions": list(candidate_actions),
                "recent_match_count": 0,
                "frame_signatures": [],
                "action_counts": {},
            }
        candidate_set = set(candidate_actions)
        recent_matches = [
            record
            for record in list(self._records)[-window:]
            if record.prior_game_state == state_name
            and record.decision_source == "fallback"
            and record.decision_subgoal == "replan"
            and record.outcome in {"stalled", "failed"}
            and record.action_name in candidate_set
        ]
        counts: defaultdict[str, int] = defaultdict(int)
        frame_signatures: set[str] = set()
        for record in recent_matches:
            counts[record.action_name] += 1
            if record.prior_frame_signature:
                frame_signatures.add(str(record.prior_frame_signature))
            if record.next_frame_signature:
                frame_signatures.add(str(record.next_frame_signature))
        reopened = True
        if len(recent_matches) < max(4, len(candidate_set) * min_attempts_per_action):
            reopened = False
        elif len(frame_signatures) < 2:
            reopened = False
        else:
            reopened = all(
                counts.get(action_name, 0) >= min_attempts_per_action
                for action_name in candidate_set
            )
        return {
            "is_reopened": reopened,
            "candidate_actions": list(candidate_actions),
            "recent_match_count": len(recent_matches),
            "frame_signatures": sorted(frame_signatures),
            "action_counts": {
                action_name: counts.get(action_name, 0)
                for action_name in candidate_actions
            },
        }

    def is_replan_frontier_exhausted(
        self,
        observation: Observation,
        action_names: list[str] | None = None,
    ) -> bool:
        active_actions = [
            action
            for action in (action_names or observation.legal_actions)
            if action != "RESET"
        ]
        if not active_actions:
            return False
        exhausted = self.replan_frontier_exhausted_actions(
            observation,
            action_names=active_actions,
        )
        return bool(exhausted) and all(action in exhausted for action in active_actions)

    def is_basin_exhausted(self, observation: Observation) -> bool:
        state_name = str(observation.game_state or "")
        frame_signature = str(observation.derived_features.get("frame_signature", "") or "")
        if not state_name:
            return False
        legal_actions = [action for action in observation.legal_actions if action != "RESET"]
        if not legal_actions:
            return False
        exhausted_actions = 0
        for action_name in legal_actions:
            counts = self.context_outcome_counts(
                action_name,
                state_name=state_name,
                frame_signature=frame_signature,
            )
            if counts["progress"] > 0:
                return False
            if not frame_signature and len(legal_actions) == 1:
                if counts["failed"] > 0 or counts["stalled"] >= 2:
                    exhausted_actions += 1
                continue
            if counts["failed"] > 0 or counts["stalled"] > 0:
                exhausted_actions += 1
        return exhausted_actions == len(legal_actions)

    def summarize_for_hyperon(self) -> dict[str, object]:
        return {
            "recent_actions": self.recent_actions(),
            "success_counts": dict(self._success_counts),
            "failure_streaks": dict(self._failure_streaks),
            "stagnation_steps": self._stagnation_steps,
            "llm_calls": self._llm_calls,
            "current_board_context": _observation_board_context(self._pending_observation),
            "working_memory": self.working_memory_snapshot(),
            "records": [asdict(record) for record in self._records],
        }

    def working_memory_snapshot(self) -> dict[str, object]:
        recent_actions = self.recent_actions()
        active_subgoal = None
        pending_action_name = (
            str(self._pending_decision.action_name or "")
            if self._pending_decision is not None
            else ""
        )
        recent_failures: list[dict[str, object]] = []
        open_failure_modes: list[dict[str, object]] = []
        seen_failure_modes: set[tuple[str, str]] = set()
        semantic_focus_actions: list[str] = []
        for record in reversed(self._records):
            if active_subgoal is None and record.decision_subgoal:
                active_subgoal = str(record.decision_subgoal)
            if record.outcome == "progress":
                action_name = str(record.action_name or "")
                if action_name and action_name not in semantic_focus_actions:
                    semantic_focus_actions.append(action_name)
            if record.outcome == "progress":
                continue
            failure_action_name = str(record.action_name or "")
            failure_subgoal = str(record.decision_subgoal or "")
            recent_failures.append(
                {
                    "action_name": failure_action_name,
                    "outcome": str(record.outcome or ""),
                    "prior_game_state": str(record.prior_game_state or ""),
                    "prior_frame_signature": str(record.prior_frame_signature or ""),
                    "decision_subgoal": failure_subgoal,
                    "stagnation_after": int(record.stagnation_after or 0),
                }
            )
            failure_mode_key = (failure_action_name, failure_subgoal)
            if failure_action_name and failure_mode_key not in seen_failure_modes:
                seen_failure_modes.add(failure_mode_key)
                open_failure_modes.append(
                    {
                        "action_name": failure_action_name,
                        "subgoal": failure_subgoal,
                        "outcome": str(record.outcome or ""),
                        "stagnation_after": int(record.stagnation_after or 0),
                        "frame_signature": str(record.prior_frame_signature or ""),
                    }
                )
            if len(recent_failures) >= 4:
                break
        if active_subgoal is None and self._pending_decision is not None:
            pending_subgoal = str(self._pending_decision.subgoal or "")
            if pending_subgoal:
                active_subgoal = pending_subgoal
        procedural_focus_actions = [
            action_name
            for action_name, _count in sorted(
                (
                    (str(action_name), int(count or 0))
                    for action_name, count in self._success_counts.items()
                    if int(count or 0) > 0
                ),
                key=lambda item: (-item[1], item[0]),
            )
            if action_name
        ][:3]
        failure_streaks = sorted(
            (
                {
                    "action_name": str(action_name),
                    "count": int(count or 0),
                }
                for action_name, count in self._failure_streaks.items()
                if int(count or 0) > 0
            ),
            key=lambda item: (-int(item["count"]), str(item["action_name"])),
        )
        current_board_context = _observation_board_context(self._pending_observation)
        active_state = str(self._pending_observation.game_state or "") if self._pending_observation is not None else ""
        active_frame_signature = (
            str(self._pending_observation.derived_features.get("frame_signature", "") or "")
            if self._pending_observation is not None
            else ""
        )
        legal_actions = (
            list(self._pending_observation.legal_actions)
            if self._pending_observation is not None
            else []
        )
        basin_exhausted = (
            self.is_basin_exhausted(self._pending_observation)
            if self._pending_observation is not None
            else False
        )
        suppressed_actions = list(legal_actions[:6]) if basin_exhausted else []
        recommended_actions: list[str] = []
        if (
            pending_action_name
            and pending_action_name not in suppressed_actions
            and pending_action_name != "RESET"
        ):
            recommended_actions.append(pending_action_name)
        for action_name in procedural_focus_actions:
            if action_name and action_name not in suppressed_actions and action_name not in recommended_actions:
                recommended_actions.append(action_name)
        for action_name in semantic_focus_actions[:3]:
            if action_name and action_name not in suppressed_actions and action_name not in recommended_actions:
                recommended_actions.append(action_name)
        basin_visits = 0
        if self._pending_observation is not None:
            basin_visits = sum(
                1
                for record in self._records
                if str(record.prior_game_state or "") == active_state
                and str(record.prior_frame_signature or "") == active_frame_signature
            )
        return {
            "active_state": active_state,
            "active_frame_signature": active_frame_signature,
            "active_subgoal": active_subgoal,
            "current_board_context": current_board_context,
            "board_context_summary": current_board_context,
            "legal_actions": legal_actions,
            "recent_actions": recent_actions[-8:],
            "recent_action_window": recent_actions[-8:],
            "recommended_actions": recommended_actions[:4],
            "suppressed_actions": suppressed_actions,
            "semantic_focus_actions": semantic_focus_actions[:3],
            "procedural_focus_actions": procedural_focus_actions,
            "recent_failures": recent_failures,
            "open_failure_modes": open_failure_modes[:4],
            "failure_streaks": failure_streaks[:6],
            "basin_exhausted": basin_exhausted,
            "basin_visits": basin_visits,
            "stagnation_steps": int(self._stagnation_steps),
            "llm_calls": int(self._llm_calls),
            "pending_score_floor_reopen": dict(self._pending_score_floor_reopen or {}),
            "recent_score_floor_fallback": dict(self._recent_score_floor_fallback or {}),
            "recent_probe_dead_end_escape": dict(self._recent_probe_dead_end_escape or {}),
        }

    def _finalize_transition(self, current_observation: Observation) -> None:
        previous = self._pending_observation
        decision = self._pending_decision
        if previous is None or decision is None:
            return
        progress_delta = current_observation.levels_completed - previous.levels_completed
        stagnation_before = self._stagnation_steps
        if current_observation.game_state == self._win_state_name or progress_delta > 0:
            outcome = "progress"
            self._success_counts[decision.action_name] += 1
            self._failure_streaks[decision.action_name] = 0
            self._stagnation_steps = 0
        elif current_observation.game_state in self._reset_state_names:
            outcome = "failed"
            self._failure_streaks[decision.action_name] += 1
            self._stagnation_steps += 1
        else:
            outcome = "stalled"
            self._failure_streaks[decision.action_name] += 1
            self._stagnation_steps += 1
        if outcome == "progress" or current_observation.game_state in self._reset_state_names:
            self.clear_pending_score_floor_reopen()
            self._recent_score_floor_fallback = None
            self._recent_probe_dead_end_escape = None
        previous_board_context = _observation_board_context(previous)
        record = EpisodeRecord(
            game_id=current_observation.game_id,
            step_index=current_observation.step_index,
            action_name=decision.action_name,
            outcome=outcome,
            progress_delta=progress_delta,
            game_state=current_observation.game_state,
            prior_game_state=previous.game_state,
            levels_completed=current_observation.levels_completed,
            prior_frame_signature=str(
                previous.derived_features.get("frame_signature", "") or ""
            )
            or None,
            next_game_state=current_observation.game_state,
            next_frame_signature=str(
                current_observation.derived_features.get("frame_signature", "") or ""
            )
            or None,
            prior_legal_actions=list(previous.legal_actions),
            stagnation_before=stagnation_before,
            stagnation_after=self._stagnation_steps,
            decision_subgoal=decision.subgoal,
            decision_source=decision.source,
            prior_board_height=int(previous_board_context["board_height"] or 0),
            prior_board_width=int(previous_board_context["board_width"] or 0),
            prior_board_active_cells=int(previous_board_context["board_active_cells"] or 0),
            prior_board_source_attr=str(previous_board_context["board_source_attr"] or "")
            or None,
            prior_board_default_probe=dict(previous_board_context["board_default_probe"] or {}),
        )
        self._records.append(record)
        self._new_records.append(record)
        self._pending_decision = None

    def _reset_episode_state(self) -> None:
        self._records.clear()
        self._new_records.clear()
        self._recent_actions.clear()
        self._success_counts.clear()
        self._failure_streaks.clear()
        self._llm_calls = 0
        self._stagnation_steps = 0
        self._observation_tick = 0
        self._pending_observation = None
        self._pending_decision = None
        self._pending_score_floor_reopen = None
        self._recent_score_floor_fallback = None
        self._recent_probe_dead_end_escape = None

    def _is_same_game_restart_boundary(self, observation: Observation) -> bool:
        previous = self._pending_observation
        if previous is None:
            return False
        current_step = int(observation.step_index or 0)
        previous_step = int(previous.step_index or 0)
        current_step_reliable = bool(
            observation.derived_features.get("step_index_reliable", True)
        )
        previous_step_reliable = bool(
            previous.derived_features.get("step_index_reliable", True)
        )
        restart_detected = (
            current_step_reliable
            and previous_step_reliable
            and current_step < previous_step
        )
        previous_state = str(previous.game_state or "")
        current_state = str(observation.game_state or "")
        reset_exit_detected = (
            bool(previous_state)
            and previous_state in self._reset_state_names
            and current_state not in self._reset_state_names
        )
        return restart_detected or reset_exit_detected

    def _step_delta_exceeds_limit(
        self,
        observation: Observation,
        pending: dict[str, object],
        *,
        limit: int,
        clear_on_negative: bool,
    ) -> bool:
        current_step_reliable = bool(
            observation.derived_features.get("step_index_reliable", True)
        )
        pending_step_reliable = bool(pending.get("step_index_reliable", True))
        if not current_step_reliable or not pending_step_reliable:
            observation_tick = int(pending.get("observation_tick") or 0)
            if observation_tick <= 0:
                return False
            return (self._observation_tick - observation_tick) > limit
        step_delta = int(observation.step_index) - int(pending.get("step_index") or 0)
        if clear_on_negative and step_delta < 0:
            return True
        return step_delta > limit


def _observation_board_context(observation: Observation | None) -> dict[str, object]:
    if observation is None:
        return board_context_summary(None)
    return board_context_summary(
        observation.image_input,
        derived_features=observation.derived_features,
    )
