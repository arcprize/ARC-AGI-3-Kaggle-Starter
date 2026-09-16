"""Explicit, competing transition hypotheses with falsification."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from math import log

from .state import StateDiff


@dataclass(frozen=True)
class ActionKey:
    action_id: int
    target_color: int | None = None
    region: tuple[int, int] | None = None

    def compact(self) -> str:
        suffix = ""
        if self.target_color is not None:
            suffix += f":c{self.target_color}"
        if self.region is not None:
            suffix += f":r{self.region[0]},{self.region[1]}"
        return f"A{self.action_id}{suffix}"


@dataclass(frozen=True)
class OutcomePattern:
    changed: bool | None = None
    progress: int | None = None
    terminal: str | None = None
    motion: tuple[int, int] | None = None
    appeared: bool | None = None
    disappeared: bool | None = None
    affected_color: int | None = None

    @classmethod
    def from_diff(cls, diff: StateDiff) -> OutcomePattern:
        deltas = {item.delta for item in diff.motions}
        motion = next(iter(deltas)) if len(deltas) == 1 else None
        return cls(
            changed=diff.changed,
            progress=diff.progress_delta,
            terminal=diff.terminal,
            motion=motion,
            appeared=bool(diff.appeared),
            disappeared=bool(diff.disappeared),
            affected_color=(
                diff.affected_colors[0] if len(diff.affected_colors) == 1 else None
            ),
        )

    def similarity(self, actual: OutcomePattern) -> float:
        pairs = [
            (self.changed, actual.changed, 2.0),
            (self.progress, actual.progress, 3.0),
            (self.terminal, actual.terminal, 2.5),
            (self.motion, actual.motion, 2.5),
            (self.appeared, actual.appeared, 1.0),
            (self.disappeared, actual.disappeared, 1.0),
            (self.affected_color, actual.affected_color, 1.0),
        ]
        considered = [(a, b, weight) for a, b, weight in pairs if a is not None]
        if not considered:
            return 0.5
        total = sum(weight for _, _, weight in considered)
        matched = sum(
            weight for expected, observed, weight in considered if expected == observed
        )
        return matched / total

    def label(self) -> str:
        fields = []
        for name in (
            "changed",
            "progress",
            "terminal",
            "motion",
            "appeared",
            "disappeared",
            "affected_color",
        ):
            value = getattr(self, name)
            if value is not None:
                fields.append(f"{name}={value}")
        return ",".join(fields) or "unspecified"


@dataclass
class Hypothesis:
    hypothesis_id: str
    action: ActionKey
    description: str
    prediction: OutcomePattern
    belief: float = 0.25
    support: list[int] = field(default_factory=list)
    contradictions: list[int] = field(default_factory=list)
    complexity_penalty: float = 0.0
    rejected: bool = False

    @property
    def effective_belief(self) -> float:
        return max(0.0, self.belief - self.complexity_penalty)


@dataclass(frozen=True)
class HypothesisUpdate:
    strengthened: tuple[str, ...]
    falsified: tuple[str, ...]
    generated: tuple[str, ...]


class HypothesisEngine:
    """Maintains a population of explanations instead of one winner."""

    def __init__(self, max_per_action: int = 8) -> None:
        self.max_per_action = max_per_action
        self.hypotheses: list[Hypothesis] = []
        self._next_id = 0
        self.rejected_signatures: dict[tuple[Hashable, ...], int] = {}

    def _new(
        self,
        action: ActionKey,
        pattern: OutcomePattern,
        description: str,
        *,
        belief: float = 0.2,
        complexity: float = 0.0,
    ) -> Hypothesis:
        hypothesis = Hypothesis(
            hypothesis_id=f"H{self._next_id}",
            action=action,
            description=description,
            prediction=pattern,
            belief=belief,
            complexity_penalty=complexity,
        )
        self._next_id += 1
        self.hypotheses.append(hypothesis)
        return hypothesis

    def seed_action(self, action: ActionKey) -> None:
        # Start with action-level priors. Context-specific rules are generated
        # after evidence, avoiding dozens of speculative click hypotheses.
        base_action = ActionKey(action.action_id)
        if any(h.action == base_action and not h.rejected for h in self.hypotheses):
            return
        label = base_action.compact()
        self._new(
            base_action,
            OutcomePattern(changed=False, progress=0),
            f"{label} has no visible effect",
        )
        self._new(
            base_action,
            OutcomePattern(changed=True, progress=0),
            f"{label} changes the scene",
        )
        self._new(
            base_action,
            OutcomePattern(changed=True, progress=1),
            f"{label} causes progress",
        )
        self._new(
            base_action,
            OutcomePattern(terminal="GAME_OVER"),
            f"{label} can be fatal",
            belief=0.12,
        )

    def applicable(self, action: ActionKey) -> list[Hypothesis]:
        self.seed_action(action)
        exact = [h for h in self.hypotheses if h.action == action and not h.rejected]
        general = [
            h
            for h in self.hypotheses
            if h.action.action_id == action.action_id
            and h.action.target_color is None
            and h.action.region is None
            and not h.rejected
        ]
        unique = {h.hypothesis_id: h for h in exact + general}
        return sorted(unique.values(), key=lambda h: h.effective_belief, reverse=True)

    def predict(self, action: ActionKey) -> list[tuple[OutcomePattern, float, str]]:
        hypotheses = self.applicable(action)
        total = sum(max(h.effective_belief, 0.01) for h in hypotheses) or 1.0
        return [
            (h.prediction, max(h.effective_belief, 0.01) / total, h.hypothesis_id)
            for h in hypotheses
        ]

    def disagreement(self, action: ActionKey) -> float:
        """Normalized entropy over distinct predictions."""
        grouped: dict[str, float] = {}
        for prediction, weight, _ in self.predict(action):
            grouped[prediction.label()] = grouped.get(prediction.label(), 0.0) + weight
        if len(grouped) <= 1:
            return 0.0
        entropy = -sum(p * log(max(p, 1e-12)) for p in grouped.values())
        return entropy / log(len(grouped))

    def update(self, action: ActionKey, diff: StateDiff, step: int) -> HypothesisUpdate:
        actual = OutcomePattern.from_diff(diff)
        hypotheses = self.applicable(action)
        strengthened: list[str] = []
        falsified: list[str] = []
        best_similarity = 0.0
        for hypothesis in hypotheses:
            similarity = hypothesis.prediction.similarity(actual)
            best_similarity = max(best_similarity, similarity)
            if similarity >= 0.72:
                hypothesis.support.append(step)
                hypothesis.belief = min(0.99, hypothesis.belief + 0.22 * similarity)
                strengthened.append(hypothesis.hypothesis_id)
            elif similarity <= 0.38:
                hypothesis.contradictions.append(step)
                hypothesis.belief *= 0.28
                if len(hypothesis.contradictions) >= 2 and hypothesis.belief < 0.08:
                    hypothesis.rejected = True
                    falsified.append(hypothesis.hypothesis_id)
                    signature = (hypothesis.action, hypothesis.prediction.label())
                    self.rejected_signatures[signature] = step
            else:
                hypothesis.belief *= 0.9

        generated: list[str] = []
        # A generic "scene changes" rule may fit while failing to explain a
        # directional motion, affected colour, or latent/terminal outcome.
        # Preserve the more specific observed alternative until future tests
        # support or falsify it.
        exact_match = any(h.prediction == actual for h in hypotheses)
        if not exact_match or best_similarity < 0.72:
            signature = (action, actual.label())
            last_rejected = self.rejected_signatures.get(signature, -10_000)
            if step - last_rejected >= 12:
                hypothesis = self._new(
                    action,
                    actual,
                    f"{action.compact()} predicts {actual.label()}",
                    belief=0.42,
                    complexity=0.02 * sum(v is not None for v in actual.__dict__.values()),
                )
                hypothesis.support.append(step)
                generated.append(hypothesis.hypothesis_id)

        base_action = ActionKey(action.action_id)
        active = [
            h for h in self.hypotheses if h.action in (action, base_action) and not h.rejected
        ]
        if len(active) > self.max_per_action:
            for hypothesis in sorted(active, key=lambda h: h.effective_belief)[
                : len(active) - self.max_per_action
            ]:
                hypothesis.rejected = True
                falsified.append(hypothesis.hypothesis_id)
        return HypothesisUpdate(tuple(strengthened), tuple(falsified), tuple(generated))

    def top(self, limit: int = 5) -> list[Hypothesis]:
        return sorted(
            (h for h in self.hypotheses if not h.rejected),
            key=lambda h: h.effective_belief,
            reverse=True,
        )[:limit]
