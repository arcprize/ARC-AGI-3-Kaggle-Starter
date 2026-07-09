from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryTrace:
    """A single consolidated memory trace.

    Attributes:
        trace_id:     Unique identifier.
        schema:       Compressed representation (dynamics, action pattern, outcome).
        episodes:     Source episode indices that formed this trace.
        strength:     Retrieval strength [0, 1] — decays with a power law.
        last_access:  When this trace was last retrieved (for spacing effect).
        access_count: How many times this trace was retrieved.
    """
    trace_id: str
    schema: dict[str, Any]
    episodes: list[int]
    strength: float = 1.0
    last_access: float = field(default_factory=time.monotonic)
    access_count: int = 0


@dataclass
class EpisodeSummary:
    """Compact summary of a single game episode."""
    episode_id: int
    dynamics: str
    action_sequence: list[str]
    outcome: str  # "win" | "timeout" | "reset"
    goal_count: int
    player_start: tuple[int, int]
    board_hash: int


class ConsolidatedMemory:
    """Episodic memory with consolidation, forgetting, and analogical retrieval.

    Key cognitive features:
    - **Consolidation**: after N episodes, structurally similar ones are
      compressed into schemas (run during idle or between levels).
    - **Forgetting curve**: retrieval strength follows a power-law decay
      (Ebbinghaus), restored by reconsolidation on access.
    - **Analogical retrieval**: when queried with a current situation,
      returns the most structurally similar past episodes.
    - **Reconsolidation**: retrieved traces are rewritten with updated
      information (strength boost + content update).
    """

    def __init__(self, max_episodes: int = 200) -> None:
        self._episodes: list[EpisodeSummary] = []
        self._traces: dict[str, MemoryTrace] = {}
        self._next_id: int = 0
        self._max_episodes = max_episodes
        self._consolidation_counter: int = 0

    # ── recording ─────────────────────────────────────────────────────

    def record(
        self,
        dynamics: str,
        action_sequence: list[str],
        outcome: str,
        goal_count: int,
        player_start: tuple[int, int],
        board_hash: int,
    ) -> None:
        """Store an episode summary (bounded by ``max_episodes``)."""
        ep = EpisodeSummary(
            episode_id=self._next_id,
            dynamics=dynamics,
            action_sequence=action_sequence,
            outcome=outcome,
            goal_count=goal_count,
            player_start=player_start,
            board_hash=board_hash,
        )
        self._episodes.append(ep)
        self._next_id += 1
        if len(self._episodes) > self._max_episodes:
            self._episodes.pop(0)
        self._consolidation_counter += 1

    # ── consolidation ─────────────────────────────────────────────────

    def consolidate(self) -> int:
        """Merge similar episodes into schemas.

        Called periodically (every N episodes or after a level completes).
        Returns the number of new traces created.
        """
        if len(self._episodes) < 2:
            return 0
        created = 0
        # Simple clustering: group by dynamics + outcome
        clusters: dict[tuple[str, str], list[EpisodeSummary]] = defaultdict(list)
        for ep in self._episodes:
            clusters[(ep.dynamics, ep.outcome)].append(ep)
        for (dynamics, outcome), eps in clusters.items():
            if len(eps) < 2:
                continue
            # Build a compressed schema
            common_prefix = self._longest_common_prefix(
                [e.action_sequence for e in eps]
            )
            schema = {
                "dynamics": dynamics,
                "outcome": outcome,
                "typical_actions": common_prefix,
                "avg_length": round(
                    sum(len(e.action_sequence) for e in eps) / len(eps), 1
                ),
            }
            trace_id = f"trace_{dynamics}_{outcome}_{len(self._traces)}"
            if trace_id not in self._traces:
                self._traces[trace_id] = MemoryTrace(
                    trace_id=trace_id,
                    schema=schema,
                    episodes=[e.episode_id for e in eps],
                )
                created += 1
        self._consolidation_counter = 0
        return created

    # ── forgetting curve ──────────────────────────────────────────────

    def _power_law_strength(self, age_hours: float) -> float:
        """Ebbinghaus power-law forgetting: S(t) = 1 / (1 + k * t^b)."""
        k = 0.5
        b = 0.6
        return 1.0 / (1.0 + k * (age_hours ** b))

    def apply_forgetting(self) -> None:
        """Decay all trace strengths by the power law since last access."""
        now = time.monotonic()
        for trace in self._traces.values():
            age_hours = (now - trace.last_access) / 3600.0
            trace.strength = self._power_law_strength(age_hours)
            # Boost for repeated access (spacing effect)
            trace.strength = min(1.0, trace.strength + 0.05 * math.log1p(trace.access_count))

    # ── analogical retrieval ──────────────────────────────────────────

    def analogical_retrieve(
        self,
        dynamics: str,
        action_sequence: list[str],
        top_k: int = 3,
    ) -> list[MemoryTrace]:
        """Return the ``top_k`` most structurally similar memory traces."""
        scored: list[tuple[float, MemoryTrace]] = []
        for trace in self._traces.values():
            if trace.schema.get("dynamics") != dynamics:
                continue
            sim = self._structural_similarity(
                action_sequence,
                trace.schema.get("typical_actions", []),
            )
            score = sim * trace.strength
            scored.append((score, trace))
        scored.sort(key=lambda x: -x[0])
        result = [t for _, t in scored[:top_k]]
        for t in result:
            t.access_count += 1
            t.last_access = time.monotonic()
        return result

    @staticmethod
    def _structural_similarity(
        seq_a: list[str],
        seq_b: list[str],
    ) -> float:
        """Jaccard-like similarity on action n-grams (structural, not sequential)."""
        if not seq_a or not seq_b:
            return 0.0
        set_a = set(seq_a)
        set_b = set(seq_b)
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union) if union else 0.0

    @staticmethod
    def _longest_common_prefix(sequences: list[list[str]]) -> list[str]:
        if not sequences:
            return []
        prefix = sequences[0]
        for seq in sequences[1:]:
            i = 0
            while i < len(prefix) and i < len(seq) and prefix[i] == seq[i]:
                i += 1
            prefix = prefix[:i]
        return prefix
