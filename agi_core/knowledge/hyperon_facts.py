from __future__ import annotations

import json
from typing import Any, Callable

from agi_core.image_input import (
    active_cell_count,
    board_context_summary,
    default_click_target,
    extract_board_motif_tags,
)
from agi_core.types import GoalState, MemoryBundle, Observation


def build_runtime_facts(
    observation: Observation,
    *,
    goals: list[GoalState],
    memory_summary: dict[str, Any],
    bundle: dict[str, Any],
    active_subgoal: str | None,
) -> dict[str, Any]:
    image_input = observation.image_input
    board_active_cells = int(
        observation.derived_features.get("board_active_cells", active_cell_count(image_input))
        or 0
    )
    board_default_probe = dict(
        observation.derived_features.get(
            "board_default_probe",
            default_click_target(image_input) if image_input is not None else {},
        )
        or {}
    )
    board_source_attr = observation.derived_features.get(
        "board_source_attr",
        image_input.source_attr if image_input is not None else None,
    )
    return {
        "episode": {
            "game_id": observation.game_id,
            "step_index": observation.step_index,
            "game_state": observation.game_state,
            "last_action": observation.last_action,
            "levels_completed": observation.levels_completed,
            "legal_actions": list(observation.legal_actions),
            "frame_signature": observation.derived_features.get("frame_signature"),
            "has_image_input": image_input is not None,
            "board_height": image_input.height if image_input is not None else 0,
            "board_width": image_input.width if image_input is not None else 0,
            "board_active_cells": board_active_cells,
            "board_default_probe": board_default_probe,
            "board_source_attr": board_source_attr,
            "has_beam": bool(observation.derived_features.get("has_beam", False)),
            "beam_visible": bool(observation.derived_features.get("beam_visible", False)),
            "beam_evidence_uncertain": bool(
                observation.derived_features.get("beam_evidence_uncertain", False)
            ),
            "analyzer_perception_failed": bool(
                observation.derived_features.get("analyzer_perception_failed", False)
            ),
        },
        "goals": [
            {
                "name": goal.name,
                "target_state": goal.target_state,
                "priority": goal.priority,
                "metadata": dict(goal.metadata),
            }
            for goal in goals
        ],
        "legality": [
            {"action_name": action_name}
            for action_name in observation.legal_actions
        ],
        "memory": memory_summary,
        "memory_bundle": bundle,
        "active_subgoal": active_subgoal,
        "trained_prior_bucket": dict(bundle.get("trained_prior_bucket", {})),
        "proposals": [],
    }


def build_contextual_bundle_cache_key(
    observation: Observation,
    subgoal: str,
    legal_actions: list[str],
    current_memory_bundle: MemoryBundle,
) -> tuple[Any, ...]:
    board_context = board_context_summary(
        observation.image_input,
        derived_features=observation.derived_features,
    )
    return (
        str(observation.game_id or ""),
        str(observation.game_state or ""),
        str(observation.derived_features.get("frame_signature", "") or ""),
        str(subgoal or ""),
        tuple(str(action_name or "") for action_name in legal_actions),
        json.dumps(board_context or {}, sort_keys=True, separators=(",", ":")),
        tuple(
            str(tag or "")
            for tag in extract_board_motif_tags(
                observation.image_input,
                board_context=board_context,
                legal_actions=legal_actions,
                levels_completed=observation.levels_completed,
            )
        ),
        repr(dict(current_memory_bundle.working_memory or {})),
        str(observation.last_action or ""),
        int(observation.step_index or 0),
    )


def build_fact_program(
    facts: dict[str, Any],
    *,
    symbol: Callable[[Any], str],
    proposal_program: str,
) -> str:
    lines: list[str] = []
    episode = facts.get("episode", {})
    memory = facts.get("memory", {})
    lines.append(f"(episode-game {symbol(episode.get('game_id'))})")
    board_type = episode.get("board_type", "unknown")
    if board_type:
        lines.append(f"(board-type {symbol(board_type)})")
    lines.append(f"(episode-step {int(episode.get('step_index', 0) or 0)})")
    lines.append(f"(episode-state {symbol(episode.get('game_state'))})")
    lines.append(
        f"(episode-levels-completed {int(episode.get('levels_completed', 0) or 0)})"
    )
    last_action = episode.get("last_action")
    if last_action:
        lines.append(f"(episode-last-action {symbol(last_action)})")
    for action_name in episode.get("legal_actions", []):
        lines.append(f"(legal-action {symbol(action_name)})")
    for goal in facts.get("goals", []):
        lines.append(
            "(goal-state "
            f"{symbol(goal.get('name'))} "
            f"{symbol(goal.get('target_state'))} "
            f"{float(goal.get('priority', 1.0) or 1.0)})"
        )
    lines.append(
        f"(memory-stagnation {int(memory.get('stagnation_steps', 0) or 0)})"
    )
    lines.append(f"(memory-llm-calls {int(memory.get('llm_calls', 0) or 0)})")
    for action_name in memory.get("recent_actions", []):
        lines.append(f"(recent-action {symbol(action_name)})")
    for action_name, count in memory.get("success_counts", {}).items():
        lines.append(f"(success-count {symbol(action_name)} {int(count)})")
    for action_name, count in memory.get("failure_streaks", {}).items():
        lines.append(f"(failure-streak {symbol(action_name)} {int(count)})")
    learned = facts.get("learned", {})
    for action_name, stats in learned.get("actions", {}).items():
        lines.append(
            f"(learned-progress-count {symbol(action_name)} {int(stats.get('progress', 0) or 0)})"
        )
        lines.append(
            f"(learned-failed-count {symbol(action_name)} {int(stats.get('failed', 0) or 0)})"
        )
        lines.append(
            f"(learned-stalled-count {symbol(action_name)} {int(stats.get('stalled', 0) or 0)})"
        )
    for state_name, state_entry in learned.get("states", {}).items():
        for action_name, stats in state_entry.get("actions", {}).items():
            lines.append(
                f"(learned-state-progress-count {symbol(state_name)} {symbol(action_name)} {int(stats.get('progress', 0) or 0)})"
            )
            lines.append(
                f"(learned-state-failed-count {symbol(state_name)} {symbol(action_name)} {int(stats.get('failed', 0) or 0)})"
            )
            lines.append(
                f"(learned-state-stalled-count {symbol(state_name)} {symbol(action_name)} {int(stats.get('stalled', 0) or 0)})"
            )
    for subgoal_name, subgoal_entry in learned.get("subgoals", {}).items():
        for action_name, stats in subgoal_entry.get("actions", {}).items():
            lines.append(
                f"(learned-subgoal-progress-count {symbol(subgoal_name)} {symbol(action_name)} {int(stats.get('progress', 0) or 0)})"
            )
            lines.append(
                f"(learned-subgoal-failed-count {symbol(subgoal_name)} {symbol(action_name)} {int(stats.get('failed', 0) or 0)})"
            )
            lines.append(
                f"(learned-subgoal-stalled-count {symbol(subgoal_name)} {symbol(action_name)} {int(stats.get('stalled', 0) or 0)})"
            )
    if proposal_program:
        lines.append(proposal_program)
    return "\n".join(line for line in lines if line)


def build_proposal_fact_program(
    proposals: list[dict[str, Any]],
    *,
    symbol: Callable[[Any], str],
) -> str:
    lines: list[str] = []
    for proposal in proposals:
        lines.append(
            "(proposal "
            f"{symbol(proposal.get('action_name'))} "
            f"{symbol(proposal.get('source'))} "
            f"{float(proposal.get('confidence', 0.0) or 0.0)} "
            f"{symbol(proposal.get('expected_outcome') or 'unknown')})"
        )
    return "\n".join(lines)
