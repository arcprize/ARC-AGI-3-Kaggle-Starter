"""No-op overshoot short-circuit (design module: structural action trimmer).

WHY THIS EXISTS
---------------
The model routinely issues a *homogeneous batched sequence* — the same action
repeated, e.g. ``action(['RIGHT', 'RIGHT', ..., 'RIGHT'])`` to march a player
across the board (confirmed in v7 commit transcripts). The vendored batch
loop (``_HarnessGameSession.step_env``, solver.py:604-638) applies **all N
actions unconditionally**; it breaks only on level/win/game-over/invalid, never
on ``board_changed == False``. So once the mover hits a wall, the remaining
identical presses are pure no-ops that STILL execute and STILL increment the
scored action counter (``action_count == len(run.history)``; every
``_execute_action`` appends to history — solver.py:693, unconditional).

Because the per-level score is quadratic in the action ratio
(``min(115, (baseline / actions) ** 2 * 100)`` — game.py:403) and the harness
runs deep in the penalised regime, trimming that no-op overshoot both raises
the score on cleared levels and hands the freed budget back to the frontier.

THE INVARIANT (why this is provably non-negative)
-------------------------------------------------
In a grid-MDP where the observable state is the Markov state, an identical
action applied from an identical state is deterministic. If action ``a`` yields
no observable change — same grid (``board_changed == False``), same
``valid_actions``, zero reward, no level/terminal transition — TWICE in a row,
then every following identical ``a`` in the batch also produces no observable
change: they are provable no-ops. Skipping them leaves the model's next
observation byte-identical (same grid, same level, same valid actions) while
lowering the counted action total. Score is monotonically decreasing in action
count and the final state is unchanged, so the per-level score is
**monotonically non-decreasing**. Worst case (no overshoot ever occurs) the
graft does nothing and plays byte-identically to stock.

The one assumption is that the observable state (grid + valid_actions) is the
Markov state. ``board_changed`` is a rendered-frame diff (solver.py:703), NOT a
full-engine-state diff — the engine can carry off-grid latent state (a charge
counter, turn parity, seeded RNG). To harden against a latent effect that
manifests within a couple of presses, the trim requires a **two-strike**
confirmation (two consecutive no-ops with identical ``valid_actions``) before
skipping the remainder, and a strike resets the instant ``valid_actions`` or the
grid moves. A game whose hidden state advances on identical presses with ZERO
observable signal for 2+ presses could still violate the invariant — a
documented tail risk, which is why this ships as a default-OFF flag flipped on
only behind green gates and is a one-line rollback.

ARCHITECTURE (zero vendor edits)
--------------------------------
``ShortCircuitSessionMixin`` overrides ``step_env`` and touches ONLY the
eligible case — a homogeneous batch of >= 2 identical actions. Every other
shape (single action, mixed batch, parse error, terminal state) delegates to
``super().step_env(arguments)`` untouched, so it is byte-identical to stock
there. The eligible branch replicates the vendored loop + payload assembly
verbatim with ONE added break; ``STEP_ENV_SRC_HASH`` pins the vendored source
so :func:`verify_step_env_pin` (a companion gate) fails loudly the day the
vendored body drifts and the replica must be re-audited.

The mixin composes over any session via MRO — stock ``_HarnessGameSession`` for
the common path, or a banking/transfer session (their ``_execute_action``
override is still picked up because ``step_env`` calls ``self._execute_action``).
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import fields
from typing import Any

import arcengine

from inference.framework.solver import (
    HarnessSolver,
    _HarnessGameSession,
    _format_action_display,
    _is_engine_game_over,
)

from taaf_grafts.solver_base import SessionSeamMixin

# blake2b(inspect.getsource(_HarnessGameSession.step_env)) pinned at write time
# against vendored solver.py:588-661. The eligible-branch replica below mirrors
# that body's loop + assembly exactly (plus one no-op break); a drift-guard
# gate recomputes this and fails loudly if the vendored body ever changes.
STEP_ENV_SRC_HASH = (
    "681764e40ce8fefbeef27ac84b6142d3f4dfb7244ca6905851e418f65cf88a32"
    "cdceac0a2e73f25218917d5a09db53f98bd33808d69173cbef6c8fa32c5a4de0"
)


def verify_step_env_pin() -> tuple[bool, str]:
    """Recompute the vendored ``step_env`` source hash. Returns
    ``(ok, actual_hash)``; ``ok`` is False iff the vendored body drifted from
    the replica this module is pinned against (re-audit trigger)."""
    src = inspect.getsource(_HarnessGameSession.step_env)
    actual = hashlib.blake2b(src.encode("utf-8")).hexdigest()
    return actual == STEP_ENV_SRC_HASH, actual


def _action_key(action: "arcengine.ActionInput") -> tuple[Any, ...]:
    """Identity key for a normalized action: engine id + sorted data items.
    Two keys compare equal iff the actions are the same button/click."""
    data_items = tuple(sorted((str(k), v) for k, v in dict(action.data).items()))
    return (action.id.value, data_items)


def _is_homogeneous(actions: list["arcengine.ActionInput"]) -> bool:
    """True iff every action in the batch is identical (same id + data)."""
    first = _action_key(actions[0])
    return all(_action_key(other) == first for other in actions[1:])


class ShortCircuitSessionMixin:
    """Session mixin: trim the provable no-op tail of a homogeneous repeated
    batch. Cooperative-MRO — place FIRST in the bases so this ``step_env``
    wins, then ``super()`` chains into the stock (or banking/transfer) body
    for every non-eligible shape.
    """

    def step_env(self, arguments: dict[str, Any]) -> dict[str, Any]:
        # Eligibility: only a homogeneous batch of >= 2 identical actions is
        # touched. Parse errors, single actions, mixed batches, and terminal
        # states fall through to the stock body BYTE-IDENTICALLY.
        actions, error = self._normalize_actions(arguments)
        if error is not None or actions is None or len(actions) < 2:
            return super().step_env(arguments)
        if not _is_homogeneous(actions):
            return super().step_env(arguments)
        if self.should_stop() or _is_engine_game_over(self.game):
            return super().step_env(arguments)

        # Eligible branch: the vendored batch loop + assembly (solver.py:
        # 595-661) reproduced verbatim, specialised to identical actions, with
        # ONE added break — stop the instant an action is a confirmed no-op.
        executed_payloads: list[dict[str, Any]] = []
        total_reward = 0.0
        stop_reason: str | None = None
        batch_size = len(actions)
        requested_displays = [
            _format_action_display(action.id.name, dict(action.data))
            for action in actions
        ]
        # Two-strike no-op tracking: a strike is an execution that left the
        # grid AND valid_actions unchanged with zero reward. Two consecutive
        # strikes confirm the repeat is inert before the tail is trimmed.
        noop_run = 0
        prev_valid: tuple[str, ...] | None = None

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
                break

            # --- the added break: skip the provable no-op tail. A strike is a
            # zero-reward execution that changed neither the grid nor
            # valid_actions vs the previous execution; two consecutive strikes
            # confirm the repeat is inert (guards against a sub-grid effect that
            # surfaces on the next press). ``batch_index < batch_size`` keeps a
            # trailing no-op (nothing left to trim) byte-identical to stock.
            curr_valid = tuple(str(v) for v in (payload.get("valid_actions") or []))
            stalled = (
                not payload.get("board_changed")
                and float(payload.get("reward", 0.0) or 0.0) == 0.0
                and (prev_valid is None or curr_valid == prev_valid)
            )
            noop_run = noop_run + 1 if stalled else 0
            prev_valid = curr_valid
            if noop_run >= 2 and batch_index < batch_size:
                stop_reason = "noop_short_circuit"
                break

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


class _ShortCircuitGameSession(ShortCircuitSessionMixin, _HarnessGameSession):
    """Stock session + the no-op short-circuit. Used when no other session
    graft (banking/transfer) is active."""


class ShortCircuitHarnessSolver(SessionSeamMixin, HarnessSolver):
    """``HarnessSolver`` whose session trims no-op overshoot. Built via
    :meth:`from_solver` in cell 12 exactly like the banking/transfer solvers;
    the session is grafted purely through the ``session_class`` seam so there
    is no per-graft ``_play_one`` copy to drift against stock."""

    session_class = _ShortCircuitGameSession
    label: str = "ShortCircuitHarnessSolver"

    @classmethod
    def from_solver(
        cls, base: HarnessSolver, **overrides: Any
    ) -> "ShortCircuitHarnessSolver":
        """Build a short-circuit solver carrying every configured field of
        ``base`` (mirrors ``BankingHarnessSolver.from_solver``)."""
        kwargs = {f.name: getattr(base, f.name) for f in fields(HarnessSolver) if f.init}
        kwargs.update(overrides)
        return cls(**kwargs)


# Cache of composed session classes so a given base yields ONE stable class
# (deepcopy identity + no globals() churn across repeated apply calls).
_COMPOSED_SESSIONS: dict[str, type] = {}


def _composed_session_class(base_session: type) -> type:
    """A ``(ShortCircuitSessionMixin, base_session)`` class that is PICKLABLE.

    ``Benchmark.run`` deepcopies the solver twice and pickles it at teardown
    (``_save_solver``); the solver's instance ``session_class`` rides along in
    ``__dict__``. An anonymous ``type(...)`` class is unpicklable (its qualname
    resolves to nothing), which would crash the un-try/except'd teardown save.
    Registering the class as a module global under its own ``__qualname__``
    makes ``pickle`` resolve it by reference. Cached so repeated applies and the
    two deepcopies all share one class object."""
    key = f"{base_session.__module__}.{base_session.__qualname__}"
    cached = _COMPOSED_SESSIONS.get(key)
    if cached is not None:
        return cached
    name = f"_ShortCircuit_{base_session.__name__.lstrip('_')}"
    composed = type(name, (ShortCircuitSessionMixin, base_session), {})
    composed.__qualname__ = name
    composed.__module__ = __name__
    globals()[name] = composed  # make pickle-by-reference resolvable
    _COMPOSED_SESSIONS[key] = composed
    return composed


def apply_shortcircuit(solver: Any) -> Any:
    """Compose the no-op short-circuit onto ``solver`` and return it.

    - A stock ``HarnessSolver`` is replaced by a :class:`ShortCircuitHarnessSolver`
      carrying its fields (the common path — analyzer-chain grafts like
      efficiency/retry_guard leave the solver stock).
    - A solver already using the ``SessionSeamMixin`` seam (banking/transfer)
      keeps its identity; its ``session_class`` is wrapped so the short-circuit
      composes OVER the banking/transfer session via MRO. Idempotent.
    """
    if isinstance(solver, SessionSeamMixin):
        base_session = getattr(solver, "session_class", _HarnessGameSession)
        if not issubclass(base_session, ShortCircuitSessionMixin):
            solver.session_class = _composed_session_class(base_session)
        return solver
    return ShortCircuitHarnessSolver.from_solver(solver)
