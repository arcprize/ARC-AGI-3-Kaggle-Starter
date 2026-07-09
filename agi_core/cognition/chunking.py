from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A compressed action sequence with pre/post conditions.

    Analogous to a learned skill — a chunk is retrieved when its
    precondition matches the current state, and its action sequence
    is expanded into plan steps directly.

    Attributes:
        name:          Human-readable label.
        preconditions: Dict of conditions that must be met for retrieval.
        actions:       The compressed action sequence (list of action names).
        postconditions: Expected state after executing the chunk.
        frequency:     How often this chunk has been successfully used.
        reliability:   Success rate [0, 1].
    """
    name: str
    preconditions: dict[str, Any] = field(default_factory=dict)
    actions: list[str] = field(default_factory=list)
    postconditions: dict[str, Any] = field(default_factory=dict)
    frequency: int = 1
    reliability: float = 1.0
    last_used: float = field(default_factory=time.monotonic)


class ChunkLibrary:
    """Library of compressed action sequences (chunks).

    Chunks are extracted from successful plan executions and re-used
    when similar situations arise, mimicking how people automatise
    practised skills.
    """

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}

    # ── public API ────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        return len(self._chunks)

    def add(self, chunk: Chunk) -> None:
        """Add or update a chunk."""
        existing = self._chunks.get(chunk.name)
        if existing is not None:
            existing.frequency += 1
            existing.last_used = time.monotonic()
            # If the new actions match the old ones, reliability goes up;
            # otherwise it drops (the chunk is less general than expected).
            if existing.actions == chunk.actions:
                existing.reliability = min(1.0, existing.reliability + 0.05)
            else:
                existing.reliability = max(0.0, existing.reliability - 0.1)
            existing.actions = chunk.actions  # update with latest
            return
        self._chunks[chunk.name] = chunk
        logger.debug("ChunkLibrary: added chunk '%s' (%d actions)",
                     chunk.name, len(chunk.actions))

    def match(
        self,
        dynamics: str,
        goal_count: int,
        action_window: list[str],
        top_k: int = 5,
    ) -> list[Chunk]:
        """Return matching chunks sorted by frequency × reliability.

        A chunk matches if its preconditions are satisfied by the
        current ``dynamics``, ``goal_count``, and recent ``action_window``.
        ``last_used`` is updated on matched chunks.
        """
        now = time.monotonic()
        scored: list[tuple[float, Chunk]] = []
        for chunk in self._chunks.values():
            pc = chunk.preconditions
            if pc.get("dynamics") and pc["dynamics"].lower() != dynamics.lower():
                continue
            if pc.get("goal_count") and pc["goal_count"] != goal_count:
                continue
            # Check action prefix (skip if no action history yet)
            prefix = pc.get("action_prefix", [])
            if prefix and action_window and action_window[-len(prefix):] != prefix:
                continue
            score = chunk.frequency * chunk.reliability
            scored.append((score, chunk))
        scored.sort(key=lambda x: -x[0])
        result = [c for _, c in scored[:top_k]]
        for c in result:
            c.last_used = now
        return result

    def extract(
        self,
        action_sequence: list[str],
        dynamics: str,
        goal_count: int,
        outcome: str,
    ) -> Chunk | None:
        """Extract a chunk from a successful action sequence.

        Returns a Chunk if the sequence is long enough and the outcome
        was successful (``"win"`` or ``"progress"``).
        """
        if outcome not in ("win", "progress"):
            return None
        if not dynamics or dynamics in ("unknown", "none", ""):
            return None
        if len(action_sequence) < 3:
            return None
        # Remove trivial repetition at the end (wait/click loops)
        while len(action_sequence) > 3 and action_sequence[-1] in ("ACTION6", "ACTION5"):
            action_sequence = action_sequence[:-1]
        if len(action_sequence) < 3:
            return None
        chunk = Chunk(
            name=f"auto_{dynamics}_{goal_count}g_{len(self._chunks)}",
            preconditions={
                "dynamics": dynamics.lower(),
                "goal_count": goal_count,
                "action_prefix": action_sequence[:3],
            },
            actions=action_sequence,
            postconditions={"outcome": outcome},
        )
        self.add(chunk)
        return chunk

    def extract_from_window(
        self,
        action_window: list[str],
        dynamics: str,
        goal_count: int,
    ) -> Chunk | None:
        """Extract a chunk from the last N actions (frame-level window).
        
        Called periodically during gameplay (not just on win).  If the
        window contains a non-trivial action sequence that looks reusable,
        save it as a chunk.
        """
        if len(action_window) < 6:
            return None
        # Only even-length windows can have a clean repeating pattern
        if len(action_window) % 2 != 0:
            return None
        half = len(action_window) // 2
        pattern = action_window[:half]
        second = action_window[half:]
        if pattern != second:
            return None
        # Reject trivial patterns (all the same action)
        if len(set(pattern)) == 1:
            return None
        return self.extract(pattern, dynamics, goal_count, "progress")

    def update_outcome(self, chunk_name: str, success: bool) -> None:
        """Update a chunk's reliability based on execution outcome."""
        chunk = self._chunks.get(chunk_name)
        if chunk is None:
            return
        if success:
            chunk.reliability = min(1.0, chunk.reliability + 0.1)
        else:
            chunk.reliability = max(0.0, chunk.reliability - 0.2)

    def decay_stale(self, max_idle_seconds: float = 300.0) -> None:
        """Gradually reduce reliability for chunks not used recently.
        
        Chunks that haven't been matched in ``max_idle_seconds`` seconds
        have their reliability halved — preventing stale schemas from
        blocking fresh exploration.
        """
        now = time.monotonic()
        for chunk in list(self._chunks.values()):
            elapsed = now - chunk.last_used
            if elapsed > max_idle_seconds:
                chunk.reliability *= 0.5
                if chunk.reliability < 0.05:
                    self._chunks.pop(chunk.name, None)
                    logger.debug("ChunkLibrary: pruned stale chunk '%s'", chunk.name)
