"""Surprise-abort on mixed commit batches (design module: schema_void, WP1).

WHY THIS EXISTS
---------------
Schema's execute-stage discipline — "reality outranks the model; one surprise
voids the rest of the plan" — is the single mechanical piece of its harness
lift a 27B can use for free. Duck commits multi-action batches through
``step_env`` and the vendored loop (``_HarnessGameSession.step_env``,
solver.py:588-661) runs them to the end no matter what the world answers
mid-batch; it breaks only on level/win/game-over/invalid. The shortcircuit
graft already trims the one provable case — homogeneous no-op overshoot — but
a MIXED plan whose premise dies mid-batch ("press A to open, then walk the
corridor that never opened") still burns its whole tail, and every burned
action feeds the quadratic action-ratio penalty (game.py:403).

This mixin adds two batch-local "the plan is void" heuristics, checked after
each executed action of a mixed batch:

(D) VOID_ON_VALID_ACTIONS_COLLAPSE (``schema_void_collapse``): if ANY action
    remaining in the tail is no longer valid in the post-action state, the
    plan assumed an action family reality just removed — drop the tail now
    instead of marching valid intermediates into a guaranteed
    ``invalid_action`` break. Validity is judged by the EXACT predicate the
    vendored loop head applies (``action.id.value not in
    self.game.current_state.available_actions``, solver.py:608) — NOT by the
    payload's ``valid_actions`` field, which filters out RESET
    (``_engine_action_names``, solver.py:116-117) and would false-positive a
    legal RESET tail.
(E) VOID_ON_OSCILLATION (``schema_void_oscillation``): if an action CHANGES
    the board back to a grid already seen earlier in this same batch (the
    pre-batch grid counts), after >= 2 board changes in the batch, the plan
    is thrashing A->B->A — drop the tail. Only an ACTIVE return
    (``board_changed`` true) fires: a no-change action trivially "returns" to
    the previous grid, but trimming on a single no-op observation would be
    strictly more aggressive than shortcircuit's deliberately two-strike
    no-op posture, so no-ops are left to shortcircuit/stock. Batch-local
    only; no cross-batch history is consulted.

THE INVARIANTS (proved by unit tests + the offline gate)
--------------------------------------------------------
- PREFIX CONSISTENCY: an early stop only ever drops an un-executed tail, so
  the observation the model receives is identical to having committed only
  the kept prefix. No action is ever invented, reordered, or replayed;
  ``executed_count <= requested_count`` always.
- NON-ELIGIBLE SHAPES ARE UNTOUCHED: single actions, homogeneous batches
  (shortcircuit's territory), parse errors, and terminal states delegate to
  ``super().step_env(arguments)`` byte-identically — composing this mixin
  OVER shortcircuit leaves every shortcircuit/stock behaviour in place.
- STOCK BREAKS ARE VERBATIM: the eligible branch replicates the vendored
  loop + payload assembly (run_complete / game_over / level_completed /
  invalid-action / exception paths and the final aggregation) exactly; the
  replica's fidelity is pinned by ``shortcircuit_solver.STEP_ENV_SRC_HASH``
  (one shared pin, checked by ``verify_step_env_pin`` in the gate).
- GRACEFUL DEGRADATION: every surprise-heuristic computation runs inside
  ``try/except``. A heuristic error means "no trim" — the batch plays out
  exactly as stock — never a crashed step. The trim-evidence print is
  guarded the same way.

POST-ACTION GRID ACCESSOR (why it is correct)
---------------------------------------------
``_execute_action`` (solver.py:667-734) diffs
``_grid_from_state(previous_state) != _grid_from_state(new_state)`` to derive
``board_changed`` (solver.py:703), and ``taaf.game.Game.execute_action``
commits ``self._current_state = new_state`` before returning (game.py:570).
So immediately after ``self._execute_action(...)`` returns,
``_grid_from_state(self.game.current_state)`` IS the rendered post-action
grid the vendored code itself compared — and ``_grid_from_state``
(solver.py:93-98) builds a fresh immutable tuple-of-tuples, safe to retain
and hash across the batch.

TAIL RISKS (documented, accepted)
---------------------------------
- A dropped tail could have been load-bearing through LATENT state: an
  A->B->A toggle that charges a hidden counter, or a plan that deliberately
  revisits a grid. The oscillation trim assumes the observable grid is the
  Markov state — same assumption shortcircuit ships under, same mitigation:
  default-OFF flag, one-line rollback, judged on an isolation kernel.
- The collapse trim assumes tail id-validity is the right proxy for "the
  plan's action family survived". A game that removes and restores an
  action id within one batch would lose its tail conservatively (never
  incorrectly: the kept prefix is still consistent).
- ``stop_reason`` on a voided batch reads ``schema_void_*`` where stock
  would have later said ``invalid_action`` (or nothing); the executed prefix
  is identical, only the label the model sees differs.
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

from taaf_grafts.shortcircuit_solver import _is_homogeneous
from taaf_grafts.solver_base import SessionSeamMixin

# The two stop_reasons this mixin can introduce. Anything else in the replica
# below is the vendored loop's own vocabulary.
_VOID_STOP_REASONS = ("schema_void_collapse", "schema_void_oscillation")


class SchemaVoidSessionMixin:
    """Session mixin: void the un-executed tail of a MIXED batch the moment a
    surprise (valid-actions collapse / batch-local oscillation) proves the
    committed plan stale. Cooperative-MRO — place FIRST in the bases so this
    ``step_env`` wins; homogeneous batches and every other non-eligible shape
    chain into ``super()`` (shortcircuit and/or stock) untouched.
    """

    def step_env(self, arguments: dict[str, Any]) -> dict[str, Any]:
        # Eligibility: only a NON-homogeneous batch of >= 2 actions is
        # touched. Parse errors, single actions, homogeneous batches
        # (shortcircuit's territory), and terminal states fall through to
        # ``super()`` BYTE-IDENTICALLY.
        actions, error = self._normalize_actions(arguments)
        if error is not None or actions is None or len(actions) < 2:
            return super().step_env(arguments)
        if _is_homogeneous(actions):
            return super().step_env(arguments)
        if self.should_stop() or _is_engine_game_over(self.game):
            return super().step_env(arguments)

        # Eligible branch: the vendored batch loop + assembly (solver.py:
        # 595-661) reproduced verbatim, with the two surprise checks added
        # AFTER each executed action (and only while a tail remains).
        executed_payloads: list[dict[str, Any]] = []
        total_reward = 0.0
        stop_reason: str | None = None
        batch_size = len(actions)
        requested_displays = [
            _format_action_display(action.id.name, dict(action.data))
            for action in actions
        ]

        # Batch-local oscillation memory: the pre-batch grid plus the grid
        # after every board-changing execution. A capture failure disables
        # (E) for this batch only — degrade, never crash (invariant).
        seen_grids: set[tuple[tuple[int, ...], ...]] = set()
        osc_enabled = True
        board_changes = 0
        try:
            seen_grids.add(_grid_from_state(self.game.current_state))
        except Exception:  # noqa: BLE001 — heuristic error means "no trim"
            osc_enabled = False

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

            # --- the added surprise checks. Only while a tail remains
            # (``batch_index < batch_size``): a batch whose last action is the
            # "surprise" has nothing to trim and must stay byte-identical to
            # stock (no spurious stop_reason). The whole block is guarded —
            # a heuristic error degrades to "no trim", never a crashed batch.
            if batch_index < batch_size:
                void_reason: str | None = None
                try:
                    # (D) collapse: some remaining action is no longer valid.
                    # Same predicate as the vendored loop head (solver.py:608);
                    # see the module docstring for why payload["valid_actions"]
                    # (RESET-filtered) would be the WRONG authority here.
                    available = self.game.current_state.available_actions
                    if any(
                        tail_action.id.value not in available
                        for tail_action in actions[batch_index:]
                    ):
                        void_reason = "schema_void_collapse"
                    # (E) oscillation: an ACTIVE return to a batch-seen grid
                    # after >= 2 board changes. No-change executions are not
                    # "returns" (shortcircuit's two-strike territory).
                    elif osc_enabled and payload.get("board_changed"):
                        board_changes += 1
                        grid_now = _grid_from_state(self.game.current_state)
                        if board_changes >= 2 and grid_now in seen_grids:
                            void_reason = "schema_void_oscillation"
                        else:
                            seen_grids.add(grid_now)
                except Exception:  # noqa: BLE001 — degrade to no-trim
                    void_reason = None
                if void_reason is not None:
                    stop_reason = void_reason
                    break

        # Trim evidence for the commit log (plan §6: "if trim count ~= 0
        # everywhere, the graft is inert"). ONE line per void event, printed
        # after the trim decision; a failed print must never break the batch.
        if stop_reason in _VOID_STOP_REASONS:
            try:
                dropped = batch_size - len(executed_payloads)
                print(f"[schema_void] voided {dropped} of {batch_size} ({stop_reason})")
            except Exception:  # noqa: BLE001
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


class _SchemaVoidGameSession(SchemaVoidSessionMixin, _HarnessGameSession):
    """Stock session + the surprise-abort. Used when no other session graft
    (shortcircuit/banking/transfer) is active."""


class SchemaVoidHarnessSolver(SessionSeamMixin, HarnessSolver):
    """``HarnessSolver`` whose session voids stale mixed-batch tails. Built
    via :meth:`from_solver` exactly like the shortcircuit/banking solvers;
    the session is grafted purely through the ``session_class`` seam so there
    is no per-graft ``_play_one`` copy to drift against stock."""

    session_class = _SchemaVoidGameSession
    label: str = "SchemaVoidHarnessSolver"

    @classmethod
    def from_solver(
        cls, base: HarnessSolver, **overrides: Any
    ) -> "SchemaVoidHarnessSolver":
        """Build a schema-void solver carrying every configured field of
        ``base`` (mirrors ``ShortCircuitHarnessSolver.from_solver``)."""
        kwargs = {f.name: getattr(base, f.name) for f in fields(HarnessSolver) if f.init}
        kwargs.update(overrides)
        return cls(**kwargs)


# Cache of composed session classes so a given base yields ONE stable class
# (deepcopy identity + no globals() churn across repeated apply calls).
_COMPOSED_SESSIONS: dict[str, type] = {}


def _composed_session_class(base_session: type) -> type:
    """A ``(SchemaVoidSessionMixin, base_session)`` class that is PICKLABLE.

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
    name = f"_SchemaVoid_{base_session.__name__.lstrip('_')}"
    composed = type(name, (SchemaVoidSessionMixin, base_session), {})
    composed.__qualname__ = name
    composed.__module__ = __name__
    globals()[name] = composed  # make pickle-by-reference resolvable
    _COMPOSED_SESSIONS[key] = composed
    return composed


def apply_schema_void(solver: Any) -> Any:
    """Compose the surprise-abort onto ``solver`` and return it.

    - A stock ``HarnessSolver`` is replaced by a :class:`SchemaVoidHarnessSolver`
      carrying its fields.
    - A solver already using the ``SessionSeamMixin`` seam (shortcircuit/
      banking/transfer) keeps its identity; its ``session_class`` is wrapped so
      the void mixin composes OUTERMOST via MRO — applied after
      ``apply_shortcircuit``, void's ``step_env`` runs first and homogeneous
      batches fall through to shortcircuit via ``super()``. Idempotent.
    """
    if isinstance(solver, SessionSeamMixin):
        base_session = getattr(solver, "session_class", _HarnessGameSession)
        if not issubclass(base_session, SchemaVoidSessionMixin):
            solver.session_class = _composed_session_class(base_session)
        return solver
    return SchemaVoidHarnessSolver.from_solver(solver)
