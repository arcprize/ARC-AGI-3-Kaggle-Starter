from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict

from .types import ActionDecision, EpisodeRecord, Observation


class EpisodicMemory:
    """Tracks recent transitions and lightweight action statistics."""

    def __init__(self, history_limit: int = 64) -> None:
        self.history_limit = history_limit
        self._records: deque[EpisodeRecord] = deque(maxlen=history_limit)
        self._recent_actions: deque[str] = deque(maxlen=16)
        self._success_counts: defaultdict[str, int] = defaultdict(int)
        self._failure_streaks: defaultdict[str, int] = defaultdict(int)
        self._llm_calls = 0
        self._stagnation_steps = 0
        self._game_id = ""
        self._pending_observation: Observation | None = None
        self._pending_decision: ActionDecision | None = None

    def bind_game(self, game_id: str) -> None:
        if game_id != self._game_id:
            self._game_id = game_id
            self._records.clear()
            self._recent_actions.clear()
            self._success_counts.clear()
            self._failure_streaks.clear()
            self._llm_calls = 0
            self._stagnation_steps = 0
            self._pending_observation = None
            self._pending_decision = None

    def note_observation(self, observation: Observation) -> None:
        self.bind_game(observation.game_id)
        if self._pending_observation is not None and self._pending_decision is not None:
            self._finalize_transition(observation)
        self._pending_observation = observation

    def note_decision(self, decision: ActionDecision) -> None:
        self._pending_decision = decision
        self._recent_actions.append(decision.action_name)

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

    def summarize_for_hyperon(self) -> dict[str, object]:
        return {
            "recent_actions": self.recent_actions(),
            "success_counts": dict(self._success_counts),
            "failure_streaks": dict(self._failure_streaks),
            "stagnation_steps": self._stagnation_steps,
            "llm_calls": self._llm_calls,
            "records": [asdict(record) for record in self._records],
        }

    def _finalize_transition(self, current_observation: Observation) -> None:
        previous = self._pending_observation
        decision = self._pending_decision
        if previous is None or decision is None:
            return
        progress_delta = current_observation.levels_completed - previous.levels_completed
        if current_observation.game_state == "WIN" or progress_delta > 0:
            outcome = "progress"
            self._success_counts[decision.action_name] += 1
            self._failure_streaks[decision.action_name] = 0
            self._stagnation_steps = 0
        elif current_observation.game_state == "GAME_OVER":
            outcome = "failed"
            self._failure_streaks[decision.action_name] += 1
            self._stagnation_steps += 1
        else:
            outcome = "stalled"
            self._failure_streaks[decision.action_name] += 1
            self._stagnation_steps += 1
        self._records.append(
            EpisodeRecord(
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
                decision_subgoal=decision.subgoal,
                decision_source=decision.source,
            )
        )
        self._pending_decision = None
