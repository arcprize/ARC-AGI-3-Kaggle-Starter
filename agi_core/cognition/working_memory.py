from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WMItem:
    """A single item in working memory.

    Attributes:
        key:         Unique identifier (e.g. ``"goal:position-mirror"``).
        content:     Arbitrary payload (dict, string, ActionDecision, …).
        category:    ``"percept"`` | ``"goal"`` | ``"plan"`` | ``"outcome"``.
        utility:     Current relevance [0, 1] — decays each frame.
        created_at:  Timestamp (monotonic) when the item was added.
        last_access: Timestamp of last retrieval.
        access_count: How many times this item has been retrieved.
    """
    key: str
    content: Any = None
    category: str = "percept"
    utility: float = 1.0
    created_at: float = field(default_factory=time.monotonic)
    last_access: float = field(default_factory=time.monotonic)
    access_count: int = 0

    def access(self) -> None:
        """Mark the item as accessed (boosts utility)."""
        self.last_access = time.monotonic()
        self.access_count += 1
        self.utility = min(1.0, self.utility + 0.15)

    def decay(self, rate: float = 0.92) -> None:
        """Apply temporal decay to utility."""
        self.utility *= rate


class WorkingMemory:
    """Capacity-limited scratchpad with temporal decay and interference.

    Models human working memory:
    - **Capacity**: hard limit (default 7 ± 2 items).
    - **Decay**: items fade each frame unless accessed.
    - **Interference**: when full, the lowest-utility item is evicted.
    - **Categories**: ``percept``, ``goal``, ``plan``, ``outcome`` each
      have a reserved slot so one category cannot crowd out all others.
    """

    MAX_ITEMS: int = 7
    _DECAY_RATE: float = 0.92

    def __init__(self, capacity: int = 7) -> None:
        self.capacity = capacity
        self._items: dict[str, WMItem] = {}
        self._category_max: dict[str, int] = {
            "goal": 2,
            "plan": 2,
            "outcome": 2,
            "percept": capacity - 2,
        }

    # ── public API ────────────────────────────────────────────────────

    @property
    def load(self) -> float:
        """Fraction of capacity used [0, 1]."""
        return len(self._items) / self.capacity

    @property
    def available_capacity(self) -> int:
        return self.capacity - len(self._items)

    def push(self, item: WMItem) -> None:
        """Add or update an item.  Evict lowest-utility item if full."""
        if item.key in self._items:
            self._items[item.key] = item
            return
        cat = item.category
        cat_items = {k: v for k, v in self._items.items() if v.category == cat}
        cat_max = self._category_max.get(cat, 3)
        if len(cat_items) >= cat_max:
            # Evict lowest-utility in this category
            victim = min(cat_items, key=lambda k: cat_items[k].utility)
            self._items.pop(victim, None)
        if len(self._items) >= self.capacity:
            victim = min(self._items, key=lambda k: self._items[k].utility)
            self._items.pop(victim, None)
        self._items[item.key] = item

    def get(self, key: str) -> WMItem | None:
        """Retrieve an item (boosts its utility)."""
        item = self._items.get(key)
        if item is not None:
            item.access()
        return item

    def remove(self, key: str) -> None:
        self._items.pop(key, None)

    def tick(self) -> None:
        """Called every frame — decay all items, prune dead ones."""
        dead_keys = []
        for key, item in self._items.items():
            item.decay(self._DECAY_RATE)
            if item.utility < 0.01:
                dead_keys.append(key)
        for k in dead_keys:
            self._items.pop(k, None)

    def snapshot(self) -> dict[str, Any]:
        """Return a snapshot for hyperon facts / logging."""
        items = []
        for item in sorted(self._items.values(), key=lambda i: -i.utility):
            items.append({
                "key": item.key,
                "category": item.category,
                "utility": round(item.utility, 3),
                "access_count": item.access_count,
            })
        return {
            "load": round(self.load, 3),
            "capacity": self.capacity,
            "items": items,
        }

    def get_by_category(self, category: str) -> list[WMItem]:
        return [v for v in self._items.values() if v.category == category]

    def clear(self) -> None:
        self._items.clear()
