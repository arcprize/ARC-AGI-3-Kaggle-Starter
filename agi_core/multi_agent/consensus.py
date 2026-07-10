from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from agi_core.types import AgentProposal, ExecutiveContext

logger = logging.getLogger(__name__)

_PREFERRED_ACTIONS: dict[str, tuple[str, ...]] = {
    "movement": ("ACTION1", "ACTION2", "ACTION3", "ACTION4"),
    "exploration": ("ACTION5", "ACTION2"),
    "interaction": ("ACTION6",),
    "puzzle": ("ACTION5", "ACTION4"),
}

_RELEVANCE_MAP: dict[str, dict[str, float]] = {
    "movement": {
        "navigation": 0.9, "sliding": 0.8, "gravity": 0.7,
        "cellular-automata": 0.6, "static": 0.4,
    },
    "exploration": {
        "navigation": 0.8, "beam": 0.7, "signal": 0.7,
        "multi-agent": 0.7, "static": 0.5,
    },
    "interaction": {
        "push-pull": 0.9, "inventory": 0.8, "beam": 0.7,
        "pattern-transform": 0.6, "static": 0.3,
    },
    "puzzle": {
        "beam": 0.9, "signal": 0.8, "pattern-transform": 0.7,
        "multi-agent": 0.6, "static": 0.2,
    },
}


class ConsensusModule:
    """Aggregates proposals from multiple agents and selects the best action.
    
    Uses weighted voting, diversity bonuses, and conflict detection to
    produce a single combined proposal list for the critic to rescore.
    
    Parameters:
        diversity_bonus: Confidence boost for unique actions (default 0.05).
        diversity_penalty: Confidence reduction per duplicate (default 0.05).
        weight_success: Coefficient for historical success rate (default 0.35).
        weight_progress: Coefficient for goal progress (default 0.25).
        weight_confidence: Coefficient for agent confidence (default 0.25).
        weight_relevance: Coefficient for board-type relevance (default 0.15).
        weight_recency: Coefficient for recent-actions alignment (default 0.10).
        no_proposal_floor: Fallback weight for agents with no proposals (default 0.1).
    """

    def __init__(
        self,
        diversity_bonus: float = 0.05,
        diversity_penalty: float = 0.05,
        weight_success: float = 0.35,
        weight_progress: float = 0.25,
        weight_confidence: float = 0.25,
        weight_relevance: float = 0.15,
        weight_recency: float = 0.10,
        no_proposal_floor: float = 0.1,
    ) -> None:
        self._agent_stats: dict[str, dict[str, float]] = {}
        self._diversity_bonus = diversity_bonus
        self._diversity_penalty = diversity_penalty
        self._weight_success = weight_success
        self._weight_progress = weight_progress
        self._weight_confidence = weight_confidence
        self._weight_relevance = weight_relevance
        self._weight_recency = weight_recency
        self._no_proposal_floor = no_proposal_floor

    # ── public API ────────────────────────────────────────────────────

    def collect(
        self,
        agent_proposals: dict[str, list[AgentProposal]],
        context: ExecutiveContext,
    ) -> list[AgentProposal]:
        """Collect, weight, and diversify all agent proposals.
        
        Returns a single list of AgentProposal objects with adjusted
        confidence scores, ready for the critic rescore pass.
        """
        all_proposals: list[AgentProposal] = []
        action_counts: dict[str, int] = defaultdict(int)
        for proposals in agent_proposals.values():
            all_proposals.extend(proposals)
            for proposal in proposals:
                action_counts[proposal.action_name] += 1
        if not all_proposals:
            return []

        weights = self._compute_weights(agent_proposals, context)
        for p in all_proposals:
            count = action_counts[p.action_name]
            if count == 1:
                confidence = min(1.0, p.confidence + self._diversity_bonus)
            else:
                confidence = max(0.01, p.confidence - self._diversity_penalty * (count - 1))
            confidence *= weights.get(p.agent_id, 0.5)
            p.confidence = max(0.01, min(1.0, confidence))
        all_proposals.sort(key=lambda p: -p.confidence)
        return all_proposals

    def note_outcome(
        self,
        agent_id: str,
        decision_success: bool,
        proposal: AgentProposal | None = None,
    ) -> None:
        """Record an agent's outcome for future vote weighting."""
        stats = self._agent_stats.setdefault(agent_id, {"successes": 0.0, "total": 0.0})
        stats["total"] += 1.0
        if decision_success:
            stats["successes"] += 1.0

    # ── internal ──────────────────────────────────────────────────────

    def _compute_weights(
        self,
        agent_proposals: dict[str, list[AgentProposal]],
        context: ExecutiveContext,
    ) -> dict[str, float]:
        """Compute vote weight per agent based on historical + current signals."""
        weights: dict[str, float] = {}
        last_action, dynamics = _consensus_context_signals(context)
        for agent_id, proposals in agent_proposals.items():
            if not proposals:
                weights[agent_id] = self._no_proposal_floor
                continue
            sample = proposals[0]
            stats = self._agent_stats.get(agent_id, {"successes": 0.0, "total": 0.0})
            success_rate = (
                stats["successes"] / stats["total"] if stats["total"] > 0 else 0.5
            )
            goal_progress = 0.0
            if sample.agent_goal_depth > 0:
                goal_progress = 1.0 / (1.0 + sample.agent_goal_depth)
            weights[agent_id] = (
                self._weight_success * success_rate
                + self._weight_progress * goal_progress
                + self._weight_confidence * min(1.0, sample.agent_confidence / 0.8)
                + self._weight_relevance * _board_relevance(agent_id, dynamics)
                + self._weight_recency * _recency_score(agent_id, last_action)
            )
        # Normalise so max weight = 1.0
        if weights:
            max_w = max(weights.values())
            if max_w > 0:
                for aid in weights:
                    weights[aid] /= max_w
        return weights


def _consensus_context_signals(context: ExecutiveContext) -> tuple[str, str]:
    """Return cached context signals used for per-agent weighting."""
    cache = context.multi_agent_frame_cache
    last_action = cache.get("last_action_upper")
    dynamics = cache.get("primary_dynamics_lower")
    if last_action is None or dynamics is None:
        features = getattr(context.observation, "derived_features", {}) or {}
        if last_action is None:
            last_action = str(features.get("last_action", "") or "").upper()
            cache["last_action_upper"] = last_action
        if dynamics is None:
            dynamics = str(features.get("primary_dynamics", "") or "").lower()
            cache["primary_dynamics_lower"] = dynamics
    return str(last_action), str(dynamics)


def _recency_score(agent_id: str, last_action: str) -> float:
    """Score [0.1, 0.3] based on how well the agent's last action aligns with recent context."""
    return 0.3 if last_action in _PREFERRED_ACTIONS.get(agent_id, ()) else 0.1


def _board_relevance(agent_id: str, dynamics: str) -> float:
    """Heuristic relevance of each agent type to the current board."""
    return _RELEVANCE_MAP.get(agent_id, {}).get(dynamics, 0.5)
