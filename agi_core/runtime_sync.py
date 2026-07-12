from __future__ import annotations

from typing import Any


def _memory_sync_token(memory: Any) -> tuple[Any, ...]:
    recent_actions = getattr(memory, "_recent_actions", None)
    records = getattr(memory, "_records", None)
    new_records = getattr(memory, "_new_records", None)
    pending_decision = getattr(memory, "_pending_decision", None)
    pending_reopen = getattr(memory, "_pending_score_floor_reopen", None)
    return (
        id(memory),
        getattr(memory, "_observation_tick", None),
        len(records) if records is not None else None,
        len(new_records) if new_records is not None else None,
        len(recent_actions) if recent_actions is not None else None,
        getattr(memory, "_stagnation_steps", None),
        getattr(pending_decision, "action_name", None),
        getattr(pending_decision, "reason", None),
        repr(pending_reopen) if pending_reopen is not None else None,
    )


def _goals_sync_token(goals: Any) -> tuple[Any, ...]:
    if goals is None:
        return ()
    normalized: list[tuple[Any, ...]] = []
    for goal in list(goals):
        normalized.append(
            (
                getattr(goal, "name", None),
                getattr(goal, "target_state", None),
                getattr(goal, "priority", None),
                repr(getattr(goal, "metadata", None)),
            )
        )
    return tuple(normalized)


def ensure_hyperon_context_synced(context: Any) -> Any:
    """Sync Hyperon facts at most once per observed frame/context."""
    observation = getattr(context, "observation", None)
    memory = getattr(context, "memory", None)
    goals = getattr(context, "goals", None)
    frame_cache = getattr(context, "multi_agent_frame_cache", None)
    if not isinstance(frame_cache, dict):
        frame_cache = {}
        setattr(context, "multi_agent_frame_cache", frame_cache)
    token = (
        getattr(observation, "game_id", None),
        getattr(observation, "step_index", None),
        getattr(observation, "game_state", None),
        id(observation),
        _memory_sync_token(memory),
        _goals_sync_token(goals),
    )
    if frame_cache.get("__hyperon_sync_token__") == token:
        return getattr(context, "memory_bundle", None)
    context.hyperon.sync(context.observation, memory, goals)
    bundle = context.hyperon.current_memory_bundle()
    context.memory_bundle = bundle
    frame_cache["__hyperon_sync_token__"] = token
    return bundle
