"""Small empirical transition model learned online from compact states."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from .hypotheses import ActionKey
from .state import StateDiff, StructuredState


@dataclass(frozen=True)
class ModelPrediction:
    next_state: str | None
    expected_progress: float
    failure_risk: float
    change_probability: float
    confidence: float


class WorldModel:
    def __init__(self) -> None:
        self._state_counts: Counter[tuple[str, ActionKey, str]] = Counter()
        self._action_outcomes: dict[ActionKey, Counter[tuple[object, ...]]] = defaultdict(
            Counter
        )
        self._state_totals: Counter[tuple[str, ActionKey]] = Counter()
        self._action_totals: Counter[ActionKey] = Counter()
        self._progress: Counter[tuple[str, ActionKey]] = Counter()
        self._failures: Counter[tuple[str, ActionKey]] = Counter()
        self._changes: Counter[tuple[str, ActionKey]] = Counter()

    def update(
        self,
        before: StructuredState,
        action: ActionKey,
        after: StructuredState,
        diff: StateDiff,
    ) -> None:
        state_key = (before.fingerprint, action)
        self._state_counts[(before.fingerprint, action, after.fingerprint)] += 1
        self._state_totals[state_key] += 1
        self._action_totals[action] += 1
        self._action_outcomes[action][diff.outcome_signature()] += 1
        self._progress[state_key] += max(0, diff.progress_delta)
        self._failures[state_key] += int(diff.terminal == "GAME_OVER")
        self._changes[state_key] += int(diff.changed)

    def predict(self, state: StructuredState, action: ActionKey) -> ModelPrediction:
        key = (state.fingerprint, action)
        total = self._state_totals[key]
        if total:
            possible = [
                (count, target)
                for (source, candidate, target), count in self._state_counts.items()
                if source == state.fingerprint and candidate == action
            ]
            count, target = max(possible, default=(0, None))
            return ModelPrediction(
                target,
                self._progress[key] / total,
                self._failures[key] / total,
                self._changes[key] / total,
                count / total,
            )

        # Back off to effects of the same action in other states.
        action_total = self._action_totals[action]
        if not action_total:
            return ModelPrediction(None, 0.0, 0.0, 0.5, 0.0)
        outcomes = self._action_outcomes[action]
        most_common_count = outcomes.most_common(1)[0][1]
        progress = sum(count * max(0, int(outcome[1])) for outcome, count in outcomes.items())
        failures = sum(
            count for outcome, count in outcomes.items() if outcome[2] == "GAME_OVER"
        )
        changes = sum(count for outcome, count in outcomes.items() if bool(outcome[0]))
        return ModelPrediction(
            None,
            progress / action_total,
            failures / action_total,
            changes / action_total,
            0.5 * most_common_count / action_total,
        )

    def known_edges(self, state_fingerprint: str) -> list[tuple[ActionKey, str, float]]:
        edges = []
        for (source, action, target), count in self._state_counts.items():
            if source != state_fingerprint:
                continue
            total = self._state_totals[(source, action)]
            edges.append((action, target, count / total))
        return edges
