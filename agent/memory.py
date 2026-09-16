"""Bounded compact episodic and semantic memory."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from math import sqrt

from .hypotheses import ActionKey
from .state import StateDiff, StructuredState


@dataclass(frozen=True)
class TransitionRecord:
    step: int
    source: str
    action: ActionKey
    target: str
    outcome: tuple[object, ...]
    progress_delta: int
    terminal: str


class CompactMemory:
    def __init__(self, max_transitions: int = 512) -> None:
        self.transitions: deque[TransitionRecord] = deque(maxlen=max_transitions)
        self.visited_states: Counter[str] = Counter()
        self.raw_states: Counter[str] = Counter()
        self.state_aliases: dict[str, set[str]] = defaultdict(set)
        self.tested_actions: Counter[tuple[str, ActionKey]] = Counter()
        self.tested_contexts: Counter[ActionKey] = Counter()
        self.object_sightings: Counter[tuple[int, tuple[tuple[int, int], ...]]] = Counter()
        self.object_last_seen: dict[tuple[int, tuple[tuple[int, int], ...]], int] = {}
        self.transient_object_signatures: set[tuple[int, tuple[tuple[int, int], ...]]] = set()
        self.progress_events: list[int] = []
        self.deaths: list[int] = []
        self.wins: list[int] = []
        self.rejected_hypotheses: dict[str, int] = {}
        self.changed_locations: Counter[tuple[int, int]] = Counter()

    def observe_state(self, state: StructuredState, step: int) -> None:
        self.visited_states[state.fingerprint] += 1
        self.raw_states[state.raw_fingerprint] += 1
        self.state_aliases[state.fingerprint].add(state.raw_fingerprint)
        for obj in state.objects:
            signature = (obj.color, obj.shape_signature)
            self.object_sightings[signature] += 1
            self.object_last_seen[signature] = step

    def record_transition(
        self,
        step: int,
        before: StructuredState,
        action: ActionKey,
        after: StructuredState,
        diff: StateDiff,
    ) -> None:
        self.transitions.append(
            TransitionRecord(
                step,
                before.fingerprint,
                action,
                after.fingerprint,
                diff.outcome_signature(),
                diff.progress_delta,
                diff.terminal,
            )
        )
        self.tested_actions[(before.fingerprint, action)] += 1
        self.tested_contexts[action] += 1
        self.changed_locations.update(diff.changed_cells)
        for obj in diff.disappeared:
            signature = (obj.color, obj.shape_signature)
            if self.object_sightings[signature] < 3:
                self.transient_object_signatures.add(signature)
        if diff.progress_delta > 0:
            self.progress_events.append(step)
        if diff.terminal == "GAME_OVER":
            self.deaths.append(step)
        if diff.terminal == "WIN":
            self.wins.append(step)

    def action_novelty(self, state: StructuredState, action: ActionKey) -> float:
        local_tests = self.tested_actions[(state.fingerprint, action)]
        global_tests = self.tested_contexts[action]
        return 0.5 / (1.0 + local_tests) + 0.5 / (1.0 + global_tests)

    def state_novelty(self, fingerprint: str) -> float:
        return 1.0 / (1.0 + self.visited_states[fingerprint])

    def representation_confidence(self, fingerprint: str) -> float:
        """Discount conclusions drawn from an aggressively aliased state."""
        raw_variants = len(self.state_aliases.get(fingerprint, ()))
        return 1.0 / sqrt(max(1, raw_variants))

    def remember_rejected(self, hypothesis_ids: tuple[str, ...], step: int) -> None:
        for hypothesis_id in hypothesis_ids:
            self.rejected_hypotheses[hypothesis_id] = step

    def persistent_object_signatures(
        self, minimum: int = 3
    ) -> set[tuple[int, tuple[tuple[int, int], ...]]]:
        return {
            signature for signature, count in self.object_sightings.items() if count >= minimum
        }

    def transition_summary(self) -> dict[str, object]:
        outcomes: dict[str, int] = defaultdict(int)
        for transition in self.transitions:
            outcomes[str(transition.outcome)] += 1
        return {
            "states": len(self.visited_states),
            "raw_states": len(self.raw_states),
            "aliased_states": sum(
                len(raw_fingerprints) > 1 for raw_fingerprints in self.state_aliases.values()
            ),
            "max_aliases": max(map(len, self.state_aliases.values()), default=0),
            "transitions": len(self.transitions),
            "progress_events": len(self.progress_events),
            "deaths": len(self.deaths),
            "wins": len(self.wins),
            "outcomes": dict(sorted(outcomes.items(), key=lambda item: -item[1])[:8]),
        }
