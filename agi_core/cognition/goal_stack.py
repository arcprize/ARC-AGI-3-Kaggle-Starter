from __future__ import annotations

import time
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class GoalStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Goal:
    """A node in the hierarchical goal tree.

    Attributes:
        name:        Human-readable label (e.g. ``"position-mirror-at-y9"``).
        description: Optional longer description.
        status:      Current lifecycle status.
        parent:      Parent goal (None for root).
        children:    Ordered list of subgoals.
        created_at:  When this goal was created.
        attempts:    How many times this goal was attempted.
        metadata:    Arbitrary key-value store (solver hints, targets, …).
    """
    name: str
    description: str = ""
    status: GoalStatus = GoalStatus.PENDING
    parent: Goal | None = None
    children: list[Goal] = field(default_factory=list)
    created_at: float = field(default_factory=time.monotonic)
    attempts: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"<Goal {self.name} [{self.status.value}]>"


class GoalStack:
    """Hierarchical goal manager.

    Mirrors how people decompose tasks:
    - A **root goal** (``"finish-level"``) is pushed first.
    - Subgoals are pushed as children.
    - Only the **active path** (root → leaf) is pursued.
    - When a subgoal fails, the stack **backtracks** to the parent.
    - Completed subgoals are marked so they are not retried.
    """

    def __init__(self, max_failures_per_goal: int = 3) -> None:
        self._root: Goal | None = None
        self._active_path: list[Goal] = []
        self._max_failures_per_goal = max_failures_per_goal
        self._progress_history: list[str] = []  # last N goal names

    # ── properties ────────────────────────────────────────────────────

    @property
    def root(self) -> Goal | None:
        return self._root

    @property
    def active(self) -> Goal | None:
        """The deepest PENDING or ACTIVE goal on the active path."""
        if not self._active_path:
            return None
        return self._active_path[-1]

    @property
    def active_path(self) -> list[Goal]:
        return list(self._active_path)

    @property
    def depth(self) -> int:
        return len(self._active_path)

    @property
    def progress_ratio(self) -> float:
        """Fraction of all goals in the tree that are COMPLETED."""
        if self._root is None:
            return 0.0
        total = 0
        completed = 0
        queue: deque[Goal] = deque([self._root])
        while queue:
            g = queue.popleft()
            total += 1
            if g.status == GoalStatus.COMPLETED:
                completed += 1
            queue.extend(g.children)
        return completed / total if total > 0 else 0.0

    # ── construction ───────────────────────────────────────────────────

    def set_root(self, name: str, **meta: Any) -> Goal:
        """Set or replace the root goal and reset the stack."""
        g = Goal(name=name, metadata=meta)
        self._root = g
        self._active_path = [g]
        self._progress_history.clear()
        return g

    def push(self, name: str, **meta: Any) -> Goal:
        """Push a subgoal under the current active goal.
        
        If the stack is empty, the goal is set as the root instead."""
        parent = self.active
        if parent is None:
            return self.set_root(name, **meta)
        g = Goal(name=name, parent=parent, metadata=meta)
        parent.children.append(g)
        self._active_path.append(g)
        return g

    def pop(self) -> Goal | None:
        """Mark the current active goal as COMPLETED and return to its parent."""
        if not self._active_path:
            return None
        g = self._active_path.pop()
        g.status = GoalStatus.COMPLETED
        return g

    def backtrack(self) -> Goal | None:
        """Mark the current active goal as FAILED and return to its parent.
        If a goal has failed more than ``_max_failures_per_goal`` times,
        skip it entirely (FAILED + SKIPPED).
        Also clears progress history to prevent cascading false backtracks."""
        if not self._active_path:
            return None
        self._progress_history.clear()
        g = self._active_path.pop()
        g.status = GoalStatus.FAILED
        g.attempts += 1
        if g.attempts >= self._max_failures_per_goal:
            g.status = GoalStatus.SKIPPED
            logger.info("GoalStack: skipping %s after %d failures", g.name, g.attempts)
            # Mark parent as failed once (increments its attempts counter)
            if self._active_path:
                self._active_path[-1].attempts += 1
            # Skip back to grandparent if parent is now also exhausted
            if self._active_path and self._active_path[-1].attempts >= self._max_failures_per_goal:
                gp = self._active_path.pop()
                gp.status = GoalStatus.FAILED
                gp.attempts += 1
                gp.status = GoalStatus.SKIPPED
                gp.children.clear()
                logger.info("GoalStack: skipping parent %s after %d failures", gp.name, gp.attempts)
        elif self._active_path:
            self._active_path[-1].attempts += 1
        return g

    def check_progress(self, current_goal_name: str | None = None) -> bool:
        """Return True if the agent is making progress on the current goal.
        
        Progress is defined as: the current goal name differs from the last
        3 entries in ``_progress_history`` (not stuck repeating the same goal).
        Safe to call multiple times — records only if name differs from last.
        """
        name = current_goal_name or (self.active.name if self.active else None)
        if name is None:
            return True
        # Avoid duplicate recording if the same goal name was just recorded
        if not self._progress_history or self._progress_history[-1] != name:
            self._progress_history.append(name)
        if len(self._progress_history) > 10:
            self._progress_history.pop(0)
        if len(self._progress_history) >= 4:
            last_4 = self._progress_history[-4:]
            if all(g == name for g in last_4):
                return False  # stuck on same goal for 4+ frames
        return True

    def mark_active(self) -> None:
        """Mark the current active goal as ACTIVE."""
        g = self.active
        if g is not None and g.status == GoalStatus.PENDING:
            g.status = GoalStatus.ACTIVE

    # ── queries ───────────────────────────────────────────────────────

    def find(self, name: str) -> Goal | None:
        """Find a goal by name anywhere in the tree (BFS)."""
        if self._root is None:
            return None
        queue: deque[Goal] = deque([self._root])
        while queue:
            g = queue.popleft()
            if g.name == name:
                return g
            queue.extend(g.children)
        return None

    def reset(self) -> None:
        self._root = None
        self._active_path = []
        self._progress_history.clear()

    def snapshot(self) -> dict[str, Any]:
        """Build a dict for logging / hyperon facts."""
        return {
            "active_path": [g.name for g in self._active_path],
            "active_status": self.active.status.value if self.active else None,
            "depth": self.depth,
            "progress_ratio": round(self.progress_ratio, 3),
        }
