from __future__ import annotations

from dataclasses import dataclass

from agi_core.image_input import board_probe_click_target, default_click_target
from agi_core.types import MemoryBundle


@dataclass(frozen=True, slots=True)
class ProposalEvidenceView:
    recommended_ranks: dict[str, int]
    trained_prior_bucket: dict[str, int]
    trained_prior_max: int
    suppressed_actions: set[str]
    semantic_memory_bonus: dict[str, float]
    procedural_memory_bonus: dict[str, float]
    episodic_recall_bonus: dict[str, float]
    working_memory_focus_bonus: dict[str, float]
    working_memory_failure_penalty: dict[str, float]
    recent_failure_actions: set[str]
    failure_streak_counts: dict[str, int]
    semantic_focus_ranks: dict[str, int]
    procedural_focus_ranks: dict[str, int]
    basin_status: dict[str, object]
    basin_action_stats: dict[str, dict[str, object]]
    state_action_stats: dict[str, dict[str, object]]
    subgoal_action_stats: dict[str, dict[str, object]]


def trained_prior_order(
    trained_prior_bucket: dict[str, object] | None,
    legal_actions,
) -> list[str]:
    legal = {str(action_name) for action_name in legal_actions if str(action_name)}
    ranked: list[tuple[int, str]] = []
    for action_name, count in dict(trained_prior_bucket or {}).items():
        normalized = str(action_name or "")
        value = int(count or 0)
        if not normalized or normalized not in legal or value <= 0:
            continue
        ranked.append((-value, normalized))
    ranked.sort()
    return [action_name for _, action_name in ranked]


def trained_prior_rank(
    trained_prior_bucket: dict[str, object] | None,
    legal_actions,
) -> dict[str, int]:
    return {
        action_name: index
        for index, action_name in enumerate(
            trained_prior_order(trained_prior_bucket, legal_actions)
        )
    }


def board_probe_metadata(context) -> dict[str, int]:
    image_input = context.observation.image_input
    if image_input is None:
        return default_click_target(image_input)
    frame_signature = str(
        context.observation.derived_features.get("frame_signature", "") or ""
    ) or None
    counts = context.memory.context_outcome_counts(
        "ACTION6",
        state_name=context.observation.game_state,
        frame_signature=frame_signature,
    )
    attempt_index = int(counts.get("failed", 0) or 0) + int(counts.get("stalled", 0) or 0)
    return board_probe_click_target(image_input, attempt_index=attempt_index)


def build_evidence_view(bundle: MemoryBundle | None) -> ProposalEvidenceView:
    if bundle is None:
        bundle = MemoryBundle()
    recommended_actions = [
        str(action_name)
        for action_name in list(bundle.recommended_actions or [])
        if str(action_name or "")
    ]
    trained_prior_bucket = {
        str(action_name): int(count)
        for action_name, count in dict(bundle.trained_prior_bucket or {}).items()
        if str(action_name) and int(count or 0) > 0
    }
    semantic_memories = list(bundle.semantic_memories or [])
    procedural_memories = list(bundle.procedural_memories or [])
    episodic_recalls = list(bundle.episodic_recalls or [])
    working_memory = dict(bundle.working_memory or {})
    semantic_focus_ranks = {
        action_name: index
        for index, action_name in enumerate(
            str(name or "")
            for name in list(working_memory.get("semantic_focus_actions", []) or [])
            if str(name or "")
        )
    }
    procedural_focus_ranks = {
        action_name: index
        for index, action_name in enumerate(
            str(name or "")
            for name in list(working_memory.get("procedural_focus_actions", []) or [])
            if str(name or "")
        )
    }
    recent_failure_actions = {
        str(row.get("action_name", "") or "")
        for row in list(working_memory.get("recent_failures", []) or [])
        if str(row.get("action_name", "") or "")
    }
    open_failure_actions = {
        str(row.get("action_name", "") or "")
        for row in list(working_memory.get("open_failure_modes", []) or [])
        if str(row.get("action_name", "") or "")
    }
    failure_streak_counts: dict[str, int] = {}
    for row in list(working_memory.get("failure_streaks", []) or []):
        action_name = str(row.get("action_name", "") or "")
        if action_name and action_name not in failure_streak_counts:
            failure_streak_counts[action_name] = int(row.get("count", 0) or 0)
    return ProposalEvidenceView(
        recommended_ranks={
            action_name: index for index, action_name in enumerate(recommended_actions)
        },
        trained_prior_bucket=trained_prior_bucket,
        trained_prior_max=max(trained_prior_bucket.values(), default=0),
        suppressed_actions=set(bundle.suppressed_actions or []),
        semantic_memory_bonus=_memory_row_bonus_map(semantic_memories, scale=0.10),
        procedural_memory_bonus=_memory_row_bonus_map(procedural_memories, scale=0.12),
        episodic_recall_bonus=_episodic_recall_bonus_map(episodic_recalls),
        working_memory_focus_bonus=_working_memory_focus_bonus_map(
            semantic_focus_ranks,
            procedural_focus_ranks,
        ),
        working_memory_failure_penalty=_working_memory_failure_penalty_map(
            recent_failure_actions,
            open_failure_actions,
            failure_streak_counts,
        ),
        recent_failure_actions=recent_failure_actions | open_failure_actions,
        failure_streak_counts=failure_streak_counts,
        semantic_focus_ranks=semantic_focus_ranks,
        procedural_focus_ranks=procedural_focus_ranks,
        basin_status=dict(bundle.basin_status or {}),
        basin_action_stats=dict(bundle.basin_action_stats or {}),
        state_action_stats=dict(bundle.state_action_stats or {}),
        subgoal_action_stats=dict(bundle.subgoal_action_stats or {}),
    )


def recommended_action_bonus(action_name: str, recommended_ranks: dict[str, int]) -> float:
    index = recommended_ranks.get(action_name)
    if index is None:
        return 0.0
    return max(0.03, 0.12 - (index * 0.03))


def suppressed_action_penalty(action_name: str, suppressed_actions: set[str]) -> float:
    if action_name not in suppressed_actions:
        return 0.0
    return 0.18


def trained_prior_bonus(
    action_name: str,
    trained_prior_bucket: dict[str, int],
    max_count: int,
) -> float:
    if action_name not in trained_prior_bucket or not trained_prior_bucket:
        return 0.0
    if max_count <= 0:
        return 0.0
    normalized = float(trained_prior_bucket.get(action_name, 0)) / float(max_count)
    return min(0.10, 0.10 * normalized)


def bundle_context_bias(
    stats: dict[str, object],
    *,
    success_scale: float,
    failed_scale: float,
    stalled_scale: float,
    positive_cap: float,
    negative_cap: float,
) -> float:
    if not stats:
        return 0.0
    progress = int(stats.get("progress", 0) or 0)
    failed = int(stats.get("failed", 0) or 0)
    stalled = int(stats.get("stalled", 0) or 0)
    positive = min(positive_cap, progress * success_scale)
    negative = min(negative_cap, (failed * failed_scale) + (stalled * stalled_scale))
    return positive - negative


def _memory_row_bonus_map(
    rows: list[dict[str, object]],
    *,
    scale: float,
) -> dict[str, float]:
    best_by_action: dict[str, float] = {}
    for row in rows:
        action_name = str(row.get("action_name", "") or "")
        if not action_name:
            continue
        confidence = float(row.get("confidence", 0.0) or 0.0)
        score = float(row.get("score", 0.0) or 0.0)
        best_by_action[action_name] = max(
            best_by_action.get(action_name, 0.0),
            min(scale, (confidence * 0.08) + (score * 0.01)),
        )
    return best_by_action


def _episodic_recall_bonus_map(
    recalls: list[dict[str, object]],
) -> dict[str, float]:
    bonus_by_action: dict[str, float] = {}
    for row in recalls:
        action_name = str(row.get("action_name", "") or "")
        if not action_name:
            continue
        outcome = str(row.get("outcome", "") or "")
        current = bonus_by_action.get(action_name, 0.0)
        if outcome == "progress":
            bonus_by_action[action_name] = max(current, 0.08)
        elif outcome in {"failed", "stalled"}:
            bonus_by_action[action_name] = min(current, -0.04)
    return bonus_by_action


def _working_memory_focus_bonus_map(
    semantic_focus_ranks: dict[str, int],
    procedural_focus_ranks: dict[str, int],
) -> dict[str, float]:
    bonus_by_action: dict[str, float] = {}
    for action_name, semantic_index in semantic_focus_ranks.items():
        bonus_by_action[action_name] = bonus_by_action.get(action_name, 0.0) + max(
            0.01,
            0.04 - (semantic_index * 0.01),
        )
    for action_name, procedural_index in procedural_focus_ranks.items():
        bonus_by_action[action_name] = bonus_by_action.get(action_name, 0.0) + max(
            0.01,
            0.05 - (procedural_index * 0.015),
        )
    return {
        action_name: min(0.08, bonus)
        for action_name, bonus in bonus_by_action.items()
    }


def _working_memory_failure_penalty_map(
    recent_failure_actions: set[str],
    open_failure_actions: set[str],
    failure_streak_counts: dict[str, int],
) -> dict[str, float]:
    penalty_by_action: dict[str, float] = {}
    for action_name in recent_failure_actions:
        if action_name:
            penalty_by_action[action_name] = penalty_by_action.get(action_name, 0.0) + 0.06
    for action_name in open_failure_actions:
        if action_name:
            penalty_by_action[action_name] = penalty_by_action.get(action_name, 0.0) + 0.05
    for action_name, count in failure_streak_counts.items():
        if action_name:
            penalty_by_action[action_name] = penalty_by_action.get(action_name, 0.0) + min(
                0.08,
                int(count or 0) * 0.02,
            )
    return {
        action_name: min(0.16, penalty)
        for action_name, penalty in penalty_by_action.items()
    }
