from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AgentFrameSnapshot:
    game_id: str
    step_index: int
    levels_completed: int
    legal_actions: tuple[str, ...]
    derived_features: dict[str, Any] = field(default_factory=dict)
    frame_cache: dict[str, Any] = field(default_factory=dict)


def sanitize_payload(value: Any) -> Any:
    """Convert nested values into multiprocessing-friendly Python objects."""
    if isinstance(value, dict):
        return {str(key): sanitize_payload(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [sanitize_payload(inner) for inner in value]
    if isinstance(value, tuple):
        return tuple(sanitize_payload(inner) for inner in value)
    if isinstance(value, set):
        return {sanitize_payload(inner) for inner in value}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            return value
    return value
