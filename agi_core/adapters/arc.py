from __future__ import annotations

from typing import Any

from agi_core.image_input import (
    active_cell_count,
    default_click_target,
    extract_board_image_input,
    preferred_board_probe_action,
)
from agi_core.types import ActionDecision, Observation


class ArcAdapter:
    """Normalizes ARC frames and maps executive decisions to GameAction."""

    def __init__(
        self,
        game_action_cls: Any,
        default_game_id: str = "",
        deterministic_action: str = "ACTION1",
        reset_state_names: tuple[str, ...] = ("NOT_PLAYED", "GAME_OVER"),
    ) -> None:
        self._game_action_cls = game_action_cls
        self._game_id = default_game_id
        self._last_image_input = None
        self._reset_state_names = {str(name or "").upper() for name in reset_state_names}
        self._default_legal_actions = self._discover_default_actions()
        if deterministic_action in self._default_legal_actions:
            self._deterministic_action = deterministic_action
        elif self._default_legal_actions:
            self._deterministic_action = self._default_legal_actions[0]
        else:
            self._deterministic_action = "RESET"

    def bind_game(self, game_id: str) -> None:
        self._game_id = game_id

    def observe(self, frames: list[Any], latest_frame: Any) -> Observation:
        game_state = self._normalize_name(self._get_attr(latest_frame, "state"))
        legal_actions, legal_actions_known = self._extract_legal_actions(latest_frame, game_state)
        step_index, step_index_reliable = self._extract_step_index(frames, latest_frame)
        image_input = extract_board_image_input(latest_frame)
        self._last_image_input = image_input
        default_probe = default_click_target(image_input) if image_input is not None else {}
        return Observation(
            game_id=self._game_id or str(self._get_attr(latest_frame, "game_id", "")),
            step_index=step_index,
            game_state=game_state,
            levels_completed=int(
                self._get_attr(latest_frame, "levels_completed", "level", default=0) or 0
            ),
            last_action=self._normalize_name(
                self._get_attr(latest_frame, "last_action", "action")
            ),
            legal_actions=legal_actions,
            image_input=image_input,
            raw_frame_ref=latest_frame,
            derived_features={
                "frame_count": len(frames),
                "legal_actions": list(legal_actions),
                "legal_actions_known": legal_actions_known,
                "step_index_reliable": step_index_reliable,
                "frame_signature": image_input.frame_signature if image_input is not None else None,
                "board_height": image_input.height if image_input is not None else 0,
                "board_width": image_input.width if image_input is not None else 0,
                "board_active_cells": active_cell_count(image_input),
                "board_default_probe": default_probe,
                "board_source_attr": image_input.source_attr if image_input is not None else None,
                "has_image_input": image_input is not None,
            },
        )

    def legal_actions(self, observation: Observation) -> list[str]:
        return list(observation.legal_actions)

    def fallback_action_name(self, observation: Observation) -> str:
        if observation.game_state in self._reset_state_names:
            return "RESET"
        legal_actions = [name for name in observation.legal_actions if name != "RESET"]
        if legal_actions:
            board_probe_action = preferred_board_probe_action(
                legal_actions,
                observation.image_input,
            )
            if board_probe_action is not None:
                return board_probe_action
            if self._deterministic_action in legal_actions:
                return self._deterministic_action
            return legal_actions[0]
        return self._deterministic_action

    def fallback_decision(self, observation: Observation) -> ActionDecision:
        action_name = self.fallback_action_name(observation)
        return ActionDecision(
            action_name=action_name,
            source="fallback",
            confidence=0.05,
            reason="deterministic adapter fallback",
            subgoal=(
                "inspect-active-board"
                if action_name == "ACTION6" and observation.image_input is not None
                else None
            ),
            metadata=self.default_metadata_for_action(action_name),
            used_fallback=True,
        )

    def default_metadata_for_action(self, action_name: str) -> dict[str, Any]:
        return self._default_metadata_for(action_name)

    def to_game_action(self, decision: ActionDecision) -> Any:
        action_name = (decision.action_name or "RESET").upper()
        action = self._game_action_cls[action_name]
        if hasattr(action, "is_complex") and action.is_complex():
            metadata = decision.metadata or {}
            x = int(metadata.get("x", 32))
            y = int(metadata.get("y", 32))
            if hasattr(action, "set_data"):
                action.set_data({"x": x, "y": y})
            if hasattr(action, "reasoning"):
                action.reasoning = {
                    "source": decision.source,
                    "subgoal": decision.subgoal,
                    "reason": decision.reason,
                }
        elif hasattr(action, "reasoning"):
            action.reasoning = (
                f"{decision.source}: {decision.reason}"
                if decision.reason
                else decision.source
            )
        return action

    def _discover_default_actions(self) -> list[str]:
        action_names: list[str] = []
        try:
            for action in self._game_action_cls:
                name = self._normalize_name(action)
                if name:
                    action_names.append(name)
        except TypeError:
            pass
        if not action_names:
            action_names = [
                "RESET",
                "ACTION1",
                "ACTION2",
                "ACTION3",
                "ACTION4",
                "ACTION5",
                "ACTION6",
            ]
        return [name for name in action_names if name != "RESET"]

    def _default_metadata_for(self, action_name: str) -> dict[str, Any]:
        if action_name == "ACTION6":
            if self._last_image_input is not None and self._last_image_input.width > 0 and self._last_image_input.height > 0:
                return self._default_complex_click_metadata()
            return {"x": 32, "y": 32}
        return {}

    def _default_complex_click_metadata(self) -> dict[str, int]:
        return default_click_target(self._last_image_input)

    def _extract_legal_actions(self, latest_frame: Any, game_state: str) -> tuple[list[str], bool]:
        explicit = self._extract_explicit_actions(latest_frame)
        if explicit:
            return explicit, True
        if game_state in self._reset_state_names:
            return ["RESET"], True
        return [], False

    def _extract_step_index(self, frames: list[Any], latest_frame: Any) -> tuple[int, bool]:
        for name in ("step_index", "step", "turn_index", "turn", "frame_index"):
            value = getattr(latest_frame, name, None)
            try:
                if value is not None:
                    return int(value), True
            except (TypeError, ValueError):
                continue
        return max(len(frames) - 1, 0), False

    def _extract_explicit_actions(self, latest_frame: Any) -> list[str]:
        for name in ("legal_actions", "available_actions", "valid_actions", "actions"):
            value = getattr(latest_frame, name, None)
            if value is None:
                continue
            if isinstance(value, str):
                normalized = self._normalize_name(value)
                return [normalized] if normalized else []
            try:
                actions = [self._normalize_name(item) for item in value]
            except TypeError:
                continue
            actions = [item for item in actions if item]
            if actions:
                return actions
        return []

    def _get_attr(self, obj: Any, *names: str, default: Any = None) -> Any:
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
        return default

    def _normalize_name(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return str(value).upper()
        if isinstance(value, int):
            if value == 0:
                return "RESET"
            return f"ACTION{value}"
        if hasattr(value, "name"):
            return str(value.name).upper()
        text = str(value).strip()
        if not text:
            return None
        if "." in text:
            text = text.split(".")[-1]
        return text.upper()
