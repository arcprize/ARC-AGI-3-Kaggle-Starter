from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


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

    _DECAY_RATE: float = 0.92

    def __init__(self, capacity: int = 7) -> None:
        self.capacity = capacity
        self._items: dict[str, WMItem] = {}
        self._category_max: dict[str, int] = {
            "goal": 2,
            "plan": 2,
            "outcome": 2,
            "percept": max(1, capacity - 2),
        }
        self._recent_actions: list[str] = []
        self._consolidation_counter: int = 0

    # ── public API ────────────────────────────────────────────────────

    @property
    def load(self) -> float:
        """Fraction of capacity used [0, 1]."""
        return len(self._items) / self.capacity

    @property
    def available_capacity(self) -> int:
        return self.capacity - len(self._items)

    @staticmethod
    def _key_prefix(key: str) -> str:
        """First underscore-delimited segment, or first 8 chars."""
        return key.split("_")[0] if "_" in key else key[:8]

    def _apply_interference(self, key: str) -> None:
        """Reduce utility of items sharing ``key``'s prefix (load-proportional)."""
        if self.available_capacity >= 3:
            return
        penalty = 1.0 - 0.15 * self.load
        prefix = self._key_prefix(key)
        for existing in list(self._items.values()):
            if existing.key == key:
                continue
            if self._key_prefix(existing.key) == prefix:
                existing.utility *= penalty

    def push(self, item: WMItem) -> None:
        """Add or update an item.  Evict lowest-utility item if full.
        Interference: if a similar key (same prefix) exists, the new item
        competes with it — but only AFTER the item has been successfully added."""
        # Update existing item in-place (preserves its category slot).
        if item.key in self._items:
            old = self._items[item.key]
            old.content = item.content
            old.utility = max(old.utility, item.utility)
            old.access()
            self._apply_interference(item.key)
            return
        cat = item.category
        cat_items = {k: v for k, v in self._items.items() if v.category == cat}
        cat_max = self._category_max.get(cat, 3)  # unknown categories default to 3
        if len(cat_items) >= cat_max:
            victim = min(cat_items, key=lambda k: cat_items[k].utility)
            self._items.pop(victim, None)
        if len(self._items) >= self.capacity:
            victim = min(self._items, key=lambda k: self._items[k].utility)
            self._items.pop(victim, None)
        # Insert the item before applying interference (so we don't penalise
        # items that were evicted making room).
        self._items[item.key] = item
        self._apply_interference(item.key)

    def get(self, key: str) -> WMItem | None:
        """Retrieve an item (boosts its utility)."""
        item = self._items.get(key)
        if item is not None:
            item.access()
        return item

    def remove(self, key: str) -> None:
        self._items.pop(key, None)

    def tick(self) -> None:
        """Called every frame — decay all items, prune dead ones.
        Also consolidates old items into a gist summary."""
        dead_keys = []
        oldest_time = float('inf')
        oldest_key = None
        for key, item in self._items.items():
            item.decay(self._DECAY_RATE)
            if item.utility < 0.01:
                dead_keys.append(key)
            if item.created_at < oldest_time:
                oldest_time = item.created_at
                oldest_key = key
        for k in dead_keys:
            self._items.pop(k, None)
        # Gist consolidation: every 50 ticks, compress oldest item into
        # an abstract summary (if it's an action sequence).
        self._consolidation_counter += 1
        if self._consolidation_counter >= 50 and oldest_key is not None:
            # If the oldest key was removed as dead, find the next oldest
            if oldest_key not in self._items:
                oldest_key = min(self._items, key=lambda k: self._items[k].created_at) if self._items else None
            oldest = self._items.get(oldest_key) if oldest_key is not None else None
            if oldest is not None and oldest.category == "outcome":
                gist = WMItem(
                    key=f"gist_{oldest_key}",
                    content=f"summarised: {str(oldest.content)[:60]}",
                    category="percept",
                    utility=0.3,
                )
                # Only remove the original if a percept slot is available
                # (otherwise the gist would be immediately evicted).
                cur_percepts = len([v for v in self._items.values() if v.category == 'percept'])
                max_percepts = self._category_max.get('percept', 3)
                if cur_percepts < max_percepts:
                    self.remove(oldest_key)
                    self.push(gist)
                else:
                    logger.debug("WM: skipping gist consolidation — percept slots full (%d/%d)",
                                 cur_percepts, max_percepts)
            self._consolidation_counter = 0  # reset regardless (prevents counter creep)

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
        self._recent_actions.clear()
        self._consolidation_counter = 0

    def record_action(self, action_name: str) -> None:
        """Append an action to the recent-actions buffer (last 20 kept)."""
        self._recent_actions.append(action_name)
        if len(self._recent_actions) > 20:
            self._recent_actions.pop(0)

    def recent_actions(self, n: int = 10) -> list[str]:
        """Return the last ``n`` recorded actions (empty list if ``n <= 0``)."""
        if n <= 0 or not self._recent_actions:
            return []
        return self._recent_actions[-n:]
