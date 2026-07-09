from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


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
        self._last_forgetting_time: float = time.monotonic()

    def last_episode(self) -> EpisodeSummary | None:
        """Return the most recent episode, or None."""
        return self._episodes[-1] if self._episodes else None

    @property
    def episode_count(self) -> int:
        return len(self._episodes)

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
        """Store an episode summary (bounded by ``max_episodes``).
        Silently ignores records with empty dynamics or action sequences."""
        if not dynamics or not action_sequence:
            return
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
        # Auto-forgetting: every 10 records OR every 60 seconds
        now = time.monotonic()
        elapsed = now - self._last_forgetting_time
        if self._consolidation_counter >= 10 or elapsed >= 60.0:
            self.apply_forgetting()
            self._consolidation_counter = 0
            self._last_forgetting_time = now

    # ── consolidation ─────────────────────────────────────────────────

    def consolidate(self) -> int:
        """Merge similar episodes into schemas.

        Called periodically (every N episodes or after a level completes).
        Returns the number of new traces created.
        """
        if len(self._episodes) < 2:
            return 0
        created = 0
        # Simple clustering: group by board_hash + dynamics + outcome
        # (prevents mixing episodes from different boards/games)
        clusters: dict[tuple[int, str, str], list[EpisodeSummary]] = defaultdict(list)
        for ep in self._episodes:
            clusters[(ep.board_hash, ep.dynamics.lower(), ep.outcome)].append(ep)
        for (board_hash, dynamics, outcome), eps in clusters.items():
            if len(eps) < 2:
                continue
            # Build a compressed schema
            common_prefix = self._longest_common_prefix(
                [e.action_sequence for e in eps]
            )
            # Skip clusters where episodes share no common prefix
            if not common_prefix:
                continue
            schema = {
                "dynamics": dynamics,
                "outcome": outcome,
                "typical_actions": common_prefix,
                "avg_length": round(
                    sum(len(e.action_sequence) for e in eps) / len(eps), 1
                ),
            }
            trace_id = f"trace_{board_hash}_{dynamics}_{outcome}"
            exists = self._traces.get(trace_id)
            if exists is not None:
                # Re-consolidation: merge new episode IDs and update schema
                old_ep_ids = set(exists.episodes)
                new_ids = [e.episode_id for e in eps if e.episode_id not in old_ep_ids]
                if new_ids:
                    exists.episodes.extend(new_ids)
                    exists.schema.update(schema)  # refresh typical_actions/avg_length
            else:
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
        """Decay all trace strengths by the power law since last access.
        Prunes traces below a minimum strength threshold.
        At least one trace is always kept (to avoid total wipeout)."""
        now = time.monotonic()
        dead = []
        for trace in self._traces.values():
            age_hours = (now - trace.last_access) / 3600.0
            trace.strength = self._power_law_strength(age_hours)
            trace.strength = min(1.0, trace.strength + 0.05 * math.log1p(trace.access_count))
            if trace.strength < 0.02:
                dead.append(trace.trace_id)
        # Keep at least one trace — reconsolidate the strongest dying trace
        survivors = len(self._traces) - len(dead)
        if survivors == 0 and dead:
            strongest = max(dead, key=lambda tid: self._traces[tid].strength)
            self._traces[strongest].strength = 0.5  # reconsolidate to half strength
            dead.remove(strongest)
        for tid in dead:
            self._traces.pop(tid, None)

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
            if trace.schema.get("dynamics") and trace.schema["dynamics"].lower() != dynamics.lower():
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
        """Hybrid similarity: Jaccard (set overlap) × sequential n-gram overlap.
        
        Returns 0.0 if either sequence is empty.  The final score is the
        geometric mean of the Jaccard score and the n-gram overlap score,
        so that both set composition AND action order matter.
        """
        if not seq_a or not seq_b:
            return 0.0
        # Jaccard (set-based)
        set_a = set(seq_a)
        set_b = set(seq_b)
        intersection = set_a & set_b
        union = set_a | set_b
        jaccard = len(intersection) / len(union) if union else 0.0
        # Bigram overlap (sequential)
        bigrams_a = set(zip(seq_a, seq_a[1:]))
        bigrams_b = set(zip(seq_b, seq_b[1:]))
        if not bigrams_a or not bigrams_b:
            bigram_score = jaccard  # fall back to jaccard for short sequences
        else:
            bg_intersection = bigrams_a & bigrams_b
            bg_union = bigrams_a | bigrams_b
            bigram_score = len(bg_intersection) / len(bg_union) if bg_union else 0.0
        # Geometric mean of the two scores
        return (jaccard * bigram_score) ** 0.5

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
