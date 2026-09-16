"""Salient action generation and experiment-value selection."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .config import Ablation, AgentConfig
from .hypotheses import ActionKey, HypothesisEngine
from .memory import CompactMemory
from .planner import ShortHorizonPlanner
from .state import StateDiff, StructuredState
from .world_model import WorldModel


@dataclass(frozen=True)
class ActionCandidate:
    key: ActionKey
    coordinates: tuple[int, int] | None = None

    def compact(self) -> str:
        return self.key.compact() + (f"@{self.coordinates}" if self.coordinates else "")


@dataclass(frozen=True)
class CandidateScore:
    candidate: ActionCandidate
    progress: float
    information: float
    novelty: float
    risk: float
    planning: float
    total: float


def _target_key(state: StructuredState, action_id: int, xy: tuple[int, int]) -> ActionKey:
    x, y = xy
    color = state.grid[y][x] if 0 <= y < state.height and 0 <= x < state.width else None
    region = (
        min(2, (3 * x) // max(1, state.width)),
        min(2, (3 * y) // max(1, state.height)),
    )
    return ActionKey(action_id, color, region)


def coordinate_candidates(
    state: StructuredState,
    memory: CompactMemory,
    previous_diff: StateDiff | None = None,
    limit: int = 24,
) -> list[tuple[int, int]]:
    """Generate deterministic clicks from entities, changes, edges, and coverage."""
    if state.width == 0 or state.height == 0:
        return [(32, 32)]
    ranked: list[tuple[int, tuple[int, int]]] = []
    for obj in state.objects:
        cx, cy = (round(obj.centroid[0]), round(obj.centroid[1]))
        ranked.append((0, (cx, cy)))
        x0, y0, x1, y1 = obj.bbox
        for xy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
            ranked.append((1, xy))
        # A representative boundary cell is useful when centroids fall in holes.
        boundary = min(
            obj.cells,
            key=lambda cell: (
                memory.changed_locations[cell],
                cell[1],
                cell[0],
            ),
        )
        ranked.append((2, boundary))
    if previous_diff:
        if previous_diff.changed_bbox:
            x0, y0, x1, y1 = previous_diff.changed_bbox
            ranked.append((0, ((x0 + x1) // 2, (y0 + y1) // 2)))
        for cell in sorted(previous_diff.changed_cells)[:8]:
            ranked.append((1, cell))

    # Coarse coverage points provide novel-cell probes without uniform random clicks.
    for fy in (1, 3, 5, 7):
        for fx in (1, 3, 5, 7):
            xy = (
                min(state.width - 1, fx * state.width // 8),
                min(state.height - 1, fy * state.height // 8),
            )
            ranked.append((3 + memory.changed_locations[xy], xy))

    output: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for _, (x, y) in sorted(ranked, key=lambda item: (item[0], item[1][1], item[1][0])):
        xy = (max(0, min(63, x)), max(0, min(63, y)))
        if xy not in seen:
            seen.add(xy)
            output.append(xy)
        if len(output) >= limit:
            break
    return output


def generate_candidates(
    legal_action_ids: list[int],
    state: StructuredState,
    memory: CompactMemory,
    previous_diff: StateDiff | None,
    coordinate_limit: int,
) -> list[ActionCandidate]:
    candidates: list[ActionCandidate] = []
    for action_id in legal_action_ids:
        if action_id == 0:
            continue
        if action_id != 6:
            candidates.append(ActionCandidate(ActionKey(action_id)))
            continue
        for xy in coordinate_candidates(state, memory, previous_diff, coordinate_limit):
            candidates.append(ActionCandidate(_target_key(state, action_id, xy), xy))
    return candidates


class ExperimentSelector:
    def __init__(
        self,
        config: AgentConfig,
        memory: CompactMemory,
        hypotheses: HypothesisEngine,
        world_model: WorldModel,
        planner: ShortHorizonPlanner,
        rng: random.Random,
    ) -> None:
        self.config = config
        self.memory = memory
        self.hypotheses = hypotheses
        self.world_model = world_model
        self.planner = planner
        self.rng = rng

    def rank(
        self, state: StructuredState, candidates: list[ActionCandidate]
    ) -> list[CandidateScore]:
        if not candidates:
            return []
        if self.config.ablation is Ablation.RANDOM:
            shuffled = list(candidates)
            self.rng.shuffle(shuffled)
            return [CandidateScore(c, 0, 0, 0, 0, 0, -i) for i, c in enumerate(shuffled)]

        planning_values = (
            self.planner.action_values(state, [candidate.key for candidate in candidates])
            if self.config.uses_planning
            else {}
        )
        scored: list[CandidateScore] = []
        for candidate in candidates:
            key = candidate.key
            model = self.world_model.predict(state, key)
            progress = model.expected_progress if self.config.uses_world_model else 0.0
            risk = model.failure_risk if self.config.uses_world_model else 0.0
            information = (
                self.hypotheses.disagreement(key) if self.config.uses_information_gain else 0.0
            )
            if self.config.uses_hypotheses and not self.config.uses_world_model:
                predictions = self.hypotheses.predict(key)
                progress = sum(
                    weight * max(0, prediction.progress or 0)
                    for prediction, weight, _ in predictions
                )
                risk = sum(
                    weight * int(prediction.terminal == "GAME_OVER")
                    for prediction, weight, _ in predictions
                )
            novelty = self.memory.action_novelty(state, key)
            # ACTION7 has a benchmark-wide documented meaning: undo. Retain it
            # for planning, but do not waste most of exploration on blind undo.
            if key.action_id == 7:
                information *= 0.25
                novelty *= 0.2
            planning = planning_values.get(key, 0.0)
            total = (
                self.config.lambda_progress * progress
                + self.config.lambda_information * information
                + self.config.lambda_novelty * novelty
                - self.config.lambda_risk * risk
                + planning
                + self.rng.random() * 1e-7
            )
            scored.append(
                CandidateScore(candidate, progress, information, novelty, risk, planning, total)
            )
        return sorted(scored, key=lambda score: score.total, reverse=True)
