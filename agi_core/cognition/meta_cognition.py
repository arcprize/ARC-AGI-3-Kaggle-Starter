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

    # ── public API ────────────────────────────────────────────────────

    def __init__(self) -> None:
        self._stagnation_steps: int = 0
        self._consecutive_failures: int = 0
        self._mode_history: list[StrategyMode] = []
        self._total_decisions: int = 0
        self._visit_counts: dict[str, int] = {}  # for curiosity tracking
        self._curiosity_actions: list[str] = []   # under-sampled actions

    def estimate_confidence(
        self,
        plan_steps: list[Any],
        goal_stack_depth: int,
        working_memory_load: float,
        stagnation_steps: int,
        recent_outcomes: list[str],
        consecutive_failures: int = 0,
    ) -> float:
        """Estimate success probability of the current plan [0, 1]."""
        base = 0.5
        base -= min(0.3, stagnation_steps * 0.03)
        failures = sum(1 for o in recent_outcomes[-5:] if o in ("failed", "stalled"))
        base -= min(0.3, failures * 0.06)
        base -= min(0.25, consecutive_failures * 0.08)
        if working_memory_load > 0.8:
            base -= 0.15
        if goal_stack_depth > 5:
            base -= 0.1
        if plan_steps:
            base += 0.15
        return max(0.05, min(0.95, base))

    def select_strategy(
        self,
        confidence: float,
        stagnation_steps: int,
        has_plan: bool,
        mode_hint: str | None = None,
        working_memory_load: float = 0.0,
        goal_progress: float = 0.0,
    ) -> StrategyMode:
        """Select the cognitive strategy for the current frame.

        Incorporates curiosity drive and cognitive load throttling:
        - **Curiosity**: when confidence is low and many actions are
          under-sampled, switch to EXPLORE to gather information.
        - **WM load throttling**: when WM > 80 %, force BALANCED/
          EXPLORE instead of EXPLOIT to reduce cognitive demand.
        - **Goal progress**: when >50 % of goals are completed, boost
          confidence by 0.1 to bias toward EXPLOIT.
        """
        self._stagnation_steps = stagnation_steps
        self._total_decisions += 1

        # Consecutive failures → force RECOVER (always, regardless of WM load)
        if self._consecutive_failures >= 3:
            mode = StrategyMode.RECOVER
            self._mode_history.append(mode)
            if len(self._mode_history) > 50:
                self._mode_history.pop(0)
            logger.debug("MetaCognition: confidence=%.2f mode=%s wm=%.2f consec=%d",
                         confidence, mode.value, working_memory_load, self._consecutive_failures)
            return mode

        # Replan hint always wins (even under WM load)
        if mode_hint == "replan":
            mode = StrategyMode.RECOVER
            self._mode_history.append(mode)
            if len(self._mode_history) > 50:
                self._mode_history.pop(0)
            logger.debug("MetaCognition: confidence=%.2f mode=%s replan",
                         confidence, mode.value)
            return mode

        # Goal progress boost
        if goal_progress > 0.5:
            confidence = min(0.95, confidence + 0.1)

        # When confidence is high, reset curiosity drive so EXPLOIT isn't
        # pre-empted by under-sampled actions from earlier exploration.
        if confidence >= 0.7:
            self._curiosity_actions = []

        # Priority-based strategy selection (first matching wins):
        #   1. WM overload → throttle
        #   2. Stagnation → RECOVER / EXPLORE
        #   3. Curiosity → EXPLORE
        #   4. Confidence-based default
        if working_memory_load > 0.8 and confidence < 0.7 and stagnation_steps >= 3:
            mode = StrategyMode.EXPLORE
        elif stagnation_steps >= 10:
            mode = StrategyMode.RECOVER
        elif stagnation_steps >= 5 and confidence < 0.4:
            mode = StrategyMode.EXPLORE
        elif self._curiosity_actions and (confidence < 0.4 or stagnation_steps >= 3):
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
        logger.debug("MetaCognition: confidence=%.2f mode=%s wm=%.2f",
                     confidence, mode.value, working_memory_load)
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

    def note_visit(self, action_name: str) -> None:
        """Track action visit counts for curiosity-driven exploration.
        Visit counts are capped at 50 to prevent unbounded accumulation."""
        current = self._visit_counts.get(action_name, 0)
        self._visit_counts[action_name] = min(50, current + 1)
        under_sampled = 30
        self._curiosity_actions = [
            a for a, c in self._visit_counts.items()
            if c < under_sampled
        ]

    def note_outcome(self, outcome: str) -> None:
        """Track outcome for confidence calibration."""
        if outcome in ("failed", "stalled"):
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = 0

    def decay_visits(self, factor: float = 0.8) -> None:
        """Periodically decay all visit counts to keep curiosity fresh.
        
        Halves counts so that previously over-sampled actions cycle back
        into the under-sampled pool, re-enabling curiosity-driven exploration.
        """
        for action in list(self._visit_counts):
            self._visit_counts[action] = max(1, int(self._visit_counts[action] * factor))
        self._curiosity_actions = [
            a for a, c in self._visit_counts.items()
            if c < 30
        ]

    def reset(self) -> None:
        """Reset meta-cognitive state for a new game."""
        self._stagnation_steps = 0
        self._consecutive_failures = 0
        self._mode_history.clear()
        self._total_decisions = 0
        self._visit_counts.clear()
        self._curiosity_actions = []
