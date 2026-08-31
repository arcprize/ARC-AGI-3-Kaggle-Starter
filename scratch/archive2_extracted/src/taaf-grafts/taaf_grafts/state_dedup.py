"""State Deduplication Graft (state_dedup.py).

Prevents LLM agents from wasting actions in loops that return to previously
visited board states on the current level.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from inference.framework.solver import (
    HarnessSolver,
    _HarnessGameSession,
    _format_action_display,
    _grid_from_state,
    _is_engine_game_over,
)

from taaf_grafts.solver_base import SessionSeamMixin

_DEDUP_STOP_REASON = "state_dedup_visited"


class StateDedupSessionMixin:
    """Session mixin: tracks visited board grids per level. Voids the tail of an
    action batch if an action lands on a grid already visited on that level.
    """

    def _get_level_visited_grids(self) -> set[tuple[tuple[int, ...], ...]]:
        if not hasattr(self, "_level_visited_grids_store"):
            self._level_visited_grids_store: dict[int, set[tuple[tuple[int, ...], ...]]] = {}
        
        current_level = 0
        try:
            current_level = getattr(self.game.current_state, "level", 0)
        except Exception:
            pass

        if current_level not in self._level_visited_grids_store:
            self._level_visited_grids_store[current_level] = set()
            try:
                # Add initial grid of the level
                initial_grid = _grid_from_state(self.game.current_state)
                self._level_visited_grids_store[current_level].add(initial_grid)
            except Exception:
                pass
        return self._level_visited_grids_store[current_level]

    def step_env(self, arguments: dict[str, Any]) -> dict[str, Any]:
        actions, error = self._normalize_actions(arguments)
        if error is not None or actions is None or len(actions) == 0:
            return super().step_env(arguments)
        if self.should_stop() or _is_engine_game_over(self.game):
            return super().step_env(arguments)

        # Record pre-execution grid in the level memory
        visited_grids = self._get_level_visited_grids()

        executed_payloads: list[dict[str, Any]] = []
        total_reward = 0.0
        stop_reason: str | None = None
        batch_size = len(actions)
        requested_displays = [
            _format_action_display(action.id.name, dict(action.data))
            for action in actions
        ]

        for batch_index, action in enumerate(actions, start=1):
            if self.should_stop():
                stop_reason = "stopped"
                break
            if action.id.value not in self.game.current_state.available_actions:
                message = f"{_format_action_display(action.id.name, dict(action.data))} is not valid right now."
                if executed_payloads:
                    stop_reason = "invalid_action"
                    break
                return self._error_payload(message)

            try:
                payload = self._execute_action(
                    action,
                    batch_index=batch_index,
                    batch_size=batch_size,
                    flush_viewer_payload=False,
                )
            except Exception as exc:
                if executed_payloads:
                    stop_reason = "action_error"
                    break
                return self._error_payload(f"{type(exc).__name__}: {exc}")

            executed_payloads.append(payload)
            total_reward += float(payload.get("reward", 0.0) or 0.0)

            if payload.get("run_complete"):
                stop_reason = "run_complete"
                break
            if payload.get("game_over"):
                stop_reason = "game_over"
                break
            if payload.get("level_completed"):
                stop_reason = "level_completed"
                # Reset visited grids on level completion
                try:
                    next_level = getattr(self.game.current_state, "level", 0)
                    self._level_visited_grids_store[next_level] = set()
                except Exception:
                    pass
                break

            # Deduplication Check
            if payload.get("board_changed"):
                try:
                    current_grid = _grid_from_state(self.game.current_state)
                    if current_grid in visited_grids:
                        # Grid has been visited on this level before!
                        stop_reason = _DEDUP_STOP_REASON
                        print(f"[state_dedup] board state already visited on level {getattr(self.game.current_state, 'level', 0)}, trimming tail")
                        break
                    else:
                        visited_grids.add(current_grid)
                except Exception:
                    pass

        if not executed_payloads:
            return self._error_payload("No action was executed.")

        final_payload = dict(executed_payloads[-1])
        final_payload["reward"] = total_reward
        final_payload["last_reward"] = executed_payloads[-1].get("reward", 0.0)
        final_payload["batched"] = batch_size > 1
        final_payload["requested_count"] = batch_size
        final_payload["executed_count"] = len(executed_payloads)
        final_payload["requested_actions"] = requested_displays
        final_payload["executed_actions"] = [
            str(item.get("action_display") or item.get("action_name") or "")
            for item in executed_payloads
        ]
        final_payload["board_changed"] = any(
            bool(item.get("board_changed")) for item in executed_payloads
        )
        final_payload["stopped_early"] = len(executed_payloads) < batch_size
        if stop_reason is not None:
            final_payload["stop_reason"] = stop_reason
        self.write_viewer_payload()
        return final_payload


class _StateDedupGameSession(StateDedupSessionMixin, _HarnessGameSession):
    """Session class with state deduplication."""


class StateDedupHarnessSolver(SessionSeamMixin, HarnessSolver):
    """HarnessSolver with StateDedupSessionMixin."""

    session_class = _StateDedupGameSession
    label: str = "StateDedupHarnessSolver"

    @classmethod
    def from_solver(
        cls, base: HarnessSolver, **overrides: Any
    ) -> "StateDedupHarnessSolver":
        kwargs = {f.name: getattr(base, f.name) for f in fields(HarnessSolver) if f.init}
        kwargs.update(overrides)
        return cls(**kwargs)


_COMPOSED_DEDUP_SESSIONS: dict[str, type] = {}


def _composed_dedup_session_class(base_session: type) -> type:
    key = f"{base_session.__module__}.{base_session.__qualname__}"
    cached = _COMPOSED_DEDUP_SESSIONS.get(key)
    if cached is not None:
        return cached
    name = f"_StateDedup_{base_session.__name__.lstrip('_')}"
    composed = type(name, (StateDedupSessionMixin, base_session), {})
    composed.__qualname__ = name
    composed.__module__ = __name__
    globals()[name] = composed
    _COMPOSED_DEDUP_SESSIONS[key] = composed
    return composed


def apply_state_dedup(solver: Any) -> Any:
    """Compose state deduplication onto solver."""
    if isinstance(solver, SessionSeamMixin):
        base_session = getattr(solver, "session_class", _HarnessGameSession)
        if not issubclass(base_session, StateDedupSessionMixin):
            solver.session_class = _composed_dedup_session_class(base_session)
        return solver
    return StateDedupHarnessSolver.from_solver(solver)
