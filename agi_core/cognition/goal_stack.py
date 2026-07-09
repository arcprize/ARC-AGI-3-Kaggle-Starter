from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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

    def __init__(self) -> None:
        self._root: Goal | None = None
        self._active_path: list[Goal] = []

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

    # ── construction ───────────────────────────────────────────────────

    def set_root(self, name: str, **meta: Any) -> Goal:
        """Set or replace the root goal and reset the stack."""
        g = Goal(name=name, metadata=meta)
        self._root = g
        self._active_path = [g]
        return g

    def push(self, name: str, **meta: Any) -> Goal:
        """Push a subgoal under the current active goal."""
        parent = self.active
        g = Goal(name=name, parent=parent, metadata=meta)
        if parent is not None:
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
        """Mark the current active goal as FAILED and return to its parent."""
        if not self._active_path:
            return None
        g = self._active_path.pop()
        g.status = GoalStatus.FAILED
        g.attempts += 1
        if self._active_path:
            self._active_path[-1].attempts += 1
        return g

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
        queue: list[Goal] = [self._root]
        while queue:
            g = queue.pop(0)
            if g.name == name:
                return g
            queue.extend(g.children)
        return None

    def reset(self) -> None:
        self._root = None
        self._active_path = []

    def snapshot(self) -> dict[str, Any]:
        """Build a dict for logging / hyperon facts."""
        return {
            "active_path": [g.name for g in self._active_path],
            "active_status": self.active.status.value if self.active else None,
            "depth": self.depth,
        }
