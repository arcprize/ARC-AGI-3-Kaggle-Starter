from __future__ import annotations

import logging
from typing import Any

from agi_core.cognition import (
    GoalStack,
    MetaCognition,
    WorkingMemory,
)
from agi_core.types import AgentProposal, ExecutiveContext

logger = logging.getLogger(__name__)


class StrategyAgent:
    """Base class for a strategy agent with its own private cognition bundle.

    Each agent has its own WorkingMemory, GoalStack, and MetaCognition,
    allowing it to specialise in one aspect of gameplay (movement,
    exploration, interaction, or puzzle-solving).
    """

    def __init__(
        self,
        agent_id: str,
        wm_capacity: int = 7,
        wm_category_max: dict[str, int] | None = None,
        gs_max_failures: int = 3,
    ) -> None:
        self.agent_id = agent_id
        self.wm = WorkingMemory(capacity=wm_capacity)
        if wm_category_max:
            self.wm._category_max.update(wm_category_max)
        self.gs = GoalStack(max_failures_per_goal=gs_max_failures)
        self.mc = MetaCognition()
        self._success_count: int = 0
        self._total_count: int = 0

    # ── properties ────────────────────────────────────────────────────

    @property
    def success_rate(self) -> float:
        """Fraction of this agent's proposals that were accepted and led to progress."""
        if self._total_count == 0:
            return 0.5
        return self._success_count / self._total_count

    @property
    def goal_progress(self) -> float:
        return self.gs.progress_ratio

    @property
    def confidence(self) -> float:
        """Estimated reliability [0, 1] — inverse of consecutive failures."""
        return max(0.05, 1.0 - self.mc._consecutive_failures * 0.25)

    # ── public API ────────────────────────────────────────────────────

    def produce(self, context: ExecutiveContext) -> list[AgentProposal]:
        """Generate proposals for the current frame.
        
        Subclasses override this to implement their strategy logic.
        Returns a list of AgentProposal objects with agent_id set.
        """
        raise NotImplementedError

    def note_outcome(self, success: bool) -> None:
        """Record whether this agent's last proposal was successful."""
        self._total_count += 1
        if success:
            self._success_count += 1

    def reset(self) -> None:
        """Reset cognition state for a new game."""
        self.wm.clear()
        self.gs.reset()
        self.mc.reset()
        self._success_count = 0
        self._total_count = 0

    def _make_proposal(
        self,
        action_name: str,
        confidence: float,
        reason: str,
        subgoal: str | None = None,
        metadata: dict[str, Any] | None = None,
        expected_outcome: str | None = None,
    ) -> AgentProposal:
        """Build a proposal tagged with this agent's metadata."""
        return AgentProposal(
            action_name=action_name,
            source=f"agent-{self.agent_id}",
            confidence=confidence,
            reason=reason,
            agent_id=self.agent_id,
            agent_confidence=self.confidence,
            agent_goal_depth=self.gs.depth,
            subgoal=subgoal,
            metadata=metadata or {},
            expected_outcome=expected_outcome,
        )

    def apply_cognition(self, action_name: str, outcome: str, subgoal: str | None = None) -> None:
        """Update cognition state after a decision."""
        self.wm.record_action(action_name)
        self.wm.tick()
        self.mc.note_visit(action_name)
        self.mc.note_outcome(outcome)
        if subgoal:
            found = self.gs.find(subgoal)
            if found is None:
                self.gs.push(subgoal)
            self.gs.mark_active()
        elif self.gs.active is not None:
            self.gs.mark_active()
        logger.debug("agent %s: action=%s outcome=%s subgoal=%s wm_load=%.2f consec=%d",
                     self.agent_id, action_name, outcome, subgoal,
                     self.wm.load, self.mc._consecutive_failures)
