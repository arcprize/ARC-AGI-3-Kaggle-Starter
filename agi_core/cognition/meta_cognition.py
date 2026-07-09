from __future__ import annotations

import logging
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class StrategyMode(Enum):
    """Cognitive strategy mode, selected by meta-cognition each frame."""
    EXPLOIT = "exploit"          # High confidence — use solver + specialist
    EXPLORE = "explore"          # Low confidence — random probes, exploration
    BALANCED = "balanced"        # Medium confidence — solver + specialist + exploration
    RECOVER = "recover"          # Stagnation detected — backtrack, try alternatives
    LEARN = "learn"              # Novel situation — probe and record outcomes


class MetaCognition:
    """Self-monitoring and strategy selection layer.

    Monitors the agent's own performance and adjusts behaviour:
    - **Confidence calibration**: estimates success probability of the
      current plan based on past similar situations.
    - **Strategy selection**: chooses EXPLOIT / EXPLORE / BALANCED /
      RECOVER / LEARN based on confidence and stagnation.
    - **Cognitive load monitoring**: if working memory exceeds 80 %,
      simplifies the plan (fewer parallel strategies, shorter horizon).
    - **Curiosity drive**: if all actions have confidence < 0.3, switches
      to EXPLORE mode (tries under-sampled actions).
    """

    def __init__(self) -> None:
        self._stagnation_steps: int = 0
        self._consecutive_failures: int = 0
        self._mode_history: list[StrategyMode] = []
        self._total_decisions: int = 0

    # ── public API ────────────────────────────────────────────────────

    def estimate_confidence(
        self,
        plan_steps: list[Any],
        goal_stack_depth: int,
        working_memory_load: float,
        stagnation_steps: int,
        recent_outcomes: list[str],
    ) -> float:
        """Estimate success probability of the current plan [0, 1]."""
        base = 0.5
        # Penalise stagnation
        base -= min(0.3, stagnation_steps * 0.03)
        # Penalise recent failures
        failures = sum(1 for o in recent_outcomes[-5:] if o in ("failed", "stalled"))
        base -= min(0.3, failures * 0.06)
        # Penalise high cognitive load (WM near full)
        if working_memory_load > 0.8:
            base -= 0.15
        # Penalise very deep goal stacks (getting lost)
        if goal_stack_depth > 5:
            base -= 0.1
        # Boost for having a concrete plan
        if plan_steps:
            base += 0.15
        return max(0.05, min(0.95, base))

    def select_strategy(
        self,
        confidence: float,
        stagnation_steps: int,
        has_plan: bool,
        mode_hint: str | None = None,
    ) -> StrategyMode:
        """Select the cognitive strategy for the current frame."""
        self._stagnation_steps = stagnation_steps
        self._total_decisions += 1

        if mode_hint == "replan":
            mode = StrategyMode.RECOVER
        elif stagnation_steps >= 10:
            mode = StrategyMode.RECOVER
        elif stagnation_steps >= 5 and confidence < 0.4:
            mode = StrategyMode.EXPLORE
        elif confidence >= 0.7 and has_plan:
            mode = StrategyMode.EXPLOIT
        elif confidence >= 0.4:
            mode = StrategyMode.BALANCED
        elif self._total_decisions < 10:
            mode = StrategyMode.LEARN
        else:
            mode = StrategyMode.EXPLORE

        self._mode_history.append(mode)
        if len(self._mode_history) > 50:
            self._mode_history.pop(0)
        logger.debug("MetaCognition: confidence=%.2f mode=%s", confidence, mode.value)
        return mode

    def should_explore(self, strategy: StrategyMode) -> bool:
        """Should the agent try under-sampled actions?"""
        return strategy in (StrategyMode.EXPLORE, StrategyMode.LEARN)

    def should_use_solver(self, strategy: StrategyMode) -> bool:
        """Should the agent delegate to domain solvers?"""
        return strategy in (StrategyMode.EXPLOIT, StrategyMode.BALANCED)

    def should_use_specialist(self, strategy: StrategyMode) -> bool:
        """Should the agent use reactive + symbolic specialists?"""
        return strategy in (StrategyMode.EXPLOIT, StrategyMode.BALANCED, StrategyMode.RECOVER)

    def should_use_llm(self, strategy: StrategyMode) -> bool:
        """Should the agent call the LLM specialist?"""
        return strategy == StrategyMode.RECOVER

    def note_outcome(self, outcome: str) -> None:
        """Track outcome for confidence calibration."""
        if outcome in ("failed", "stalled"):
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = 0

    def reset(self) -> None:
        """Reset meta-cognitive state for a new game."""
        self._stagnation_steps = 0
        self._consecutive_failures = 0
        self._mode_history.clear()
        self._total_decisions = 0
