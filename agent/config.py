"""Configuration and ablations for HypothesisWorld."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum


class Ablation(StrEnum):
    """The six research variants share one implementation."""

    RANDOM = "A"
    NOVELTY = "B"
    WORLD_MODEL = "C"
    HYPOTHESES = "D"
    ACTIVE_HYPOTHESES = "E"
    FULL = "F"


@dataclass(frozen=True)
class AgentConfig:
    ablation: Ablation = Ablation.FULL
    max_actions: int = 200
    lambda_information: float = 1.4
    lambda_novelty: float = 0.65
    lambda_progress: float = 4.0
    lambda_risk: float = 2.2
    planning_horizon: int = 2
    planning_confidence: float = 0.68
    max_coordinate_candidates: int = 24
    max_transitions: int = 512
    max_hypotheses_per_action: int = 8
    log_top_hypotheses: int = 5
    random_seed: int = 0

    @classmethod
    def from_env(cls) -> AgentConfig:
        raw = os.getenv("HYPOTHESISWORLD_ABLATION", "F").strip().upper()
        try:
            ablation = Ablation(raw)
        except ValueError:
            ablation = Ablation.FULL
        try:
            random_seed = int(os.getenv("HYPOTHESISWORLD_SEED", "0"))
        except ValueError:
            random_seed = 0
        return cls(ablation=ablation, random_seed=random_seed)

    @property
    def uses_world_model(self) -> bool:
        return self.ablation in {
            Ablation.WORLD_MODEL,
            Ablation.ACTIVE_HYPOTHESES,
            Ablation.FULL,
        }

    @property
    def uses_hypotheses(self) -> bool:
        return self.ablation in {
            Ablation.HYPOTHESES,
            Ablation.ACTIVE_HYPOTHESES,
            Ablation.FULL,
        }

    @property
    def uses_information_gain(self) -> bool:
        return self.ablation in {Ablation.ACTIVE_HYPOTHESES, Ablation.FULL}

    @property
    def uses_planning(self) -> bool:
        return self.ablation is Ablation.FULL
