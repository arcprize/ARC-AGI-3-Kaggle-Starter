"""Composite graft installer — the single notebook cell-12 entry point.

``install(bm, flags)`` is the ONLY thing the kernel calls. Its entire body
runs under one try/except that restores the original ``bm.solver`` object
on ANY error (and reverts any module-global knob patch it applied first),
so a broken graft can never take the run down: the worst case is stock
behaviour with a one-line ``[taaf_grafts] install failed -> stock`` note.

INVARIANTS (why this file is shaped the way it is):

- ALL flags default OFF; ``install(bm, {})`` is a proven no-op (the
  standing all-flags-off byte-identity gate — the 1.15-floor guarantee).
- Flag module imports are individually guarded, so the composite works on
  a dataset version where a flag's module does not exist yet
  (forward-compat): a missing module degrades that one flag to stock, a
  genuine error restores the whole solver to stock.
- ``transfer`` implies ``banking`` (the transfer session subclasses the
  banking session).
- The solver is swapped by ``from_solver`` (a field-copy, not a pickle):
  the subclass survives ``Benchmark.run``'s two deepcopies because
  ``HarnessSolver.__deepcopy__`` reconstructs ``type(self)`` and keeps
  ``analyzer_factory`` by reference.
- ``context_window`` is a MODULE-GLOBAL patch, never an env var: the
  ``LOCAL_ANALYZER_*`` knobs are frozen as module globals at first import
  of ``inference.framework.solver`` (during the cell-10 unpickle), but
  ``_LOCAL_ANALYZER_CONTEXT_WINDOW`` is re-read at every per-game
  ``ToolAgent`` construction, so reassigning the global in cell 12 lands;
  setting the env var then would be too late.
- The stock analyzer factory replicates ``HarnessSolver._make_analyzer``'s
  default branch EXACTLY except ``api_key=base_url=provider=None``: this
  sidesteps the deepcopy-closure ``_local_server_*`` landmine (a closure
  captured pre-deepcopy would read the wrong instance's server fields) by
  letting ``ToolAgent`` env-resolve its connection — which is byte-
  identical to the default branch in the Kaggle reality where
  ``start_local_server`` is off and those fields are empty.
"""

from __future__ import annotations

import importlib
import json
from typing import Any, Callable

GRAFTS_API_VERSION = 1

# Boolean feature flags that select a solver replacement.
_SOLVER_FLAGS = ("banking", "transfer", "shortcircuit")


# -- stock analyzer factory -------------------------------------------------


def make_stock_toolagent_factory(solver: Any) -> Callable[[Any, int], Any]:
    """Return an ``analyzer_factory`` that builds a stock ``ToolAgent``.

    Replicates ``HarnessSolver._make_analyzer``'s factory-less branch minus
    the deepcopy-closure landmine: ``api_key/base_url/provider`` are ``None``
    so ``ToolAgent`` env-resolves its connection at construction (the Kaggle
    reality), instead of reading ``solver._local_server_*`` off a closure
    captured before ``Benchmark.run``'s deepcopy.
    """

    def factory(game: Any, index: int) -> Any:
        from inference.agent.tool_agent import ToolAgent  # lazy: needs LLM env

        return ToolAgent(
            model=solver.model,
            timeout=solver.analyzer_timeout,
            save_request_logs=solver.save_request_logs,
            api_key=None,
            base_url=None,
            provider=None,
        )

    return factory


# -- analyzer chain layers --------------------------------------------------


def _load_retry_guard() -> type:
    from taaf_grafts.retry_guard import RetryGuard

    return RetryGuard


def _load_recovery() -> type:
    from taaf_grafts.recovery import RecoveryLayer

    return RecoveryLayer


# (flag, loader) pairs applied innermost-first, so the last entry is the
# OUTERMOST layer. RetryGuard is the outermost shipping layer; RecoveryLayer
# sits inside it (a RetryGuard backoff must govern the whole recovery turn,
# probes included). Loaders raise when their module is absent; the chain
# factory skips them (forward-compat).
_CHAIN_LAYERS: list[tuple[str, Callable[[], type]]] = [
    ("recovery", _load_recovery),
    ("retry_guard", _load_retry_guard),
]


def register_chain_layer(flag: str, loader: Callable[[], type]) -> None:
    """Register (or replace) an analyzer chain layer selected by ``flag``.

    ``loader`` is a zero-arg callable returning the layer CLASS (constructed
    as ``layer(inner)``); it may raise when its module is absent, in which
    case the chain factory skips the layer. Appended after the existing
    layers, i.e. becomes more outermost than earlier registrations.
    """
    name = str(flag)
    for i, (existing, _loader) in enumerate(_CHAIN_LAYERS):
        if existing == name:
            _CHAIN_LAYERS[i] = (name, loader)
            return
    _CHAIN_LAYERS.append((name, loader))


def _chain_flag_names() -> tuple[str, ...]:
    return tuple(flag for flag, _loader in _CHAIN_LAYERS)


def make_analyzer_chain(
    solver: Any,
    flags: dict[str, Any],
    *,
    inner_factory: Callable[[Any, int], Any] | None = None,
) -> Callable[[Any, int], Any]:
    """Build the analyzer factory: innermost stock ``ToolAgent`` (or the
    supplied ``inner_factory``), wrapped by every enabled chain layer with
    RetryGuard outermost. Each layer is responsible for proxying
    ``generated_tokens``/``total_tokens``/``_timeout`` inward and for
    degrading to its inner result on any error.

    ``inner_factory`` overrides the innermost analyzer (used by the later
    ToolAgent-subclass factory and by the gate to chain over a stub); it
    defaults to :func:`make_stock_toolagent_factory`.
    """
    inner = inner_factory or make_stock_toolagent_factory(solver)
    selected = [(flag, loader) for flag, loader in _CHAIN_LAYERS if flags.get(flag)]

    def factory(game: Any, index: int) -> Any:
        analyzer = inner(game, index)
        for _flag, loader in selected:
            try:
                layer_cls = loader()
            except Exception:  # noqa: BLE001 — absent layer module -> skip (forward-compat)
                continue
            analyzer = layer_cls(analyzer)
        return analyzer

    return factory


# -- solver replacement -----------------------------------------------------


def _import_optional(module: str, name: str) -> Any | None:
    """Import ``module.name`` returning ``None`` only on a MISSING module
    (per-flag forward-compat guard). Any other import error propagates to
    the caller's restore-to-stock handler."""
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError:
        return None
    return getattr(mod, name)


def _build_solver(original: Any, flags: dict[str, Any], active: dict[str, Any]) -> Any:
    want_transfer = bool(flags.get("transfer"))
    want_banking = bool(flags.get("banking")) or want_transfer
    want_shortcircuit = bool(flags.get("shortcircuit"))

    solver_obj = original
    if want_transfer:
        transfer_cls = _import_optional(
            "taaf_grafts.transfer_solver", "TransferHarnessSolver"
        )
        if transfer_cls is not None:
            solver_obj = transfer_cls.from_solver(original)
            active["transfer"] = True
            active["banking"] = True
    elif want_banking:
        banking_cls = _import_optional(
            "taaf_grafts.banking_solver", "BankingHarnessSolver"
        )
        if banking_cls is not None:
            solver_obj = banking_cls.from_solver(original)
            active["banking"] = True

    # The no-op short-circuit composes OVER whatever solver was selected
    # (stock/banking/transfer) via the session_class seam. A missing module
    # degrades this one flag to stock (forward-compat).
    if want_shortcircuit:
        apply_sc = _import_optional(
            "taaf_grafts.shortcircuit_solver", "apply_shortcircuit"
        )
        if apply_sc is not None:
            solver_obj = apply_sc(solver_obj)
            active["shortcircuit"] = True

    # Schema execute-stage discipline (surprise-abort on committed mixed
    # batches) composes over the SAME seam, applied AFTER shortcircuit so the
    # void mixin is outermost in the session MRO: heterogeneous batches are
    # void's, homogeneous batches fall through to SC via super(). A missing
    # module degrades this one flag to stock (forward-compat).
    if flags.get("schema_void"):
        apply_void = _import_optional("taaf_grafts.schema_void", "apply_schema_void")
        if apply_void is not None:
            solver_obj = apply_void(solver_obj)
            active["schema_void"] = True

    return solver_obj


# -- module-global knob patches ---------------------------------------------


def _patch_context_window(value: int) -> Callable[[], None]:
    """Reassign ``tool_agent._LOCAL_ANALYZER_CONTEXT_WINDOW`` and return a
    thunk that restores the previous value (used only on install failure)."""
    import inference.agent.tool_agent as ta

    previous = ta._LOCAL_ANALYZER_CONTEXT_WINDOW  # noqa: SLF001
    ta._LOCAL_ANALYZER_CONTEXT_WINDOW = value  # noqa: SLF001
    return lambda: setattr(ta, "_LOCAL_ANALYZER_CONTEXT_WINDOW", previous)


# -- banner -----------------------------------------------------------------


def _print_banner(active: dict[str, Any]) -> None:
    payload = json.dumps(
        {key: active[key] for key in sorted(active)}, separators=(",", ":")
    )
    print(f"TAAF_GRAFTS FEATURES={payload} API_VERSION={GRAFTS_API_VERSION}")


# -- the cell-12 entry point ------------------------------------------------


def install(
    bm: Any,
    flags: dict[str, Any] | None = None,
    *,
    expected_version: int | None = None,
) -> None:
    """Install the graft stack onto ``bm`` per ``flags`` (all default off).

    Blanket-guarded: on any error the original ``bm.solver`` is restored,
    every applied module-global patch is reverted, and a stock-fallback
    note is printed. Never raises.
    """
    flags = dict(flags or {})
    original = getattr(bm, "solver", None)
    reverts: list[Callable[[], None]] = []
    try:
        if (
            expected_version is not None
            and int(expected_version) != GRAFTS_API_VERSION
        ):
            raise RuntimeError(
                f"GRAFTS_API_VERSION mismatch: expected {expected_version}, "
                f"have {GRAFTS_API_VERSION}"
            )

        active: dict[str, Any] = {}

        context_window = flags.get("context_window")
        if context_window is not None:
            resolved = int(context_window)
            reverts.append(_patch_context_window(resolved))
            active["context_window"] = resolved

        solver_obj = _build_solver(original, flags, active)

        # The efficiency flag swaps the innermost analyzer from the stock
        # ToolAgent to an EfficiencyToolAgent (a report-only budget-note
        # subclass). A missing agent_ext module degrades this one flag to
        # stock (forward-compat); the factory itself falls back to a stock
        # ToolAgent on any per-game construction error.
        inner_factory: Callable[[Any, int], Any] | None = None
        efficiency_active = False
        if flags.get("efficiency"):
            make_eff = _import_optional(
                "taaf_grafts.agent_ext", "make_efficiency_toolagent_factory"
            )
            if make_eff is not None:
                inner_factory = make_eff(solver_obj)
                efficiency_active = True

        # The schema prompt/sandbox grafts are NESTED under efficiency: each is
        # an EfficiencyToolAgent subclass, so the flag only arms when
        # ``efficiency`` armed too (floor F always carries efficiency; a bare
        # schema_notes/schema_helpers flag stays inert rather than silently
        # dropping the efficiency note). Mutually exclusive by design — both on
        # arms only schema_notes, and the FEATURES banner then mismatches an
        # expect-both gate LOUDLY instead of pretending the stack is supported.
        if efficiency_active and flags.get("schema_notes"):
            make_notes = _import_optional(
                "taaf_grafts.schema_notes", "make_schema_notes_toolagent_factory"
            )
            if make_notes is not None:
                inner_factory = make_notes(solver_obj)
                active["schema_notes"] = True
        elif efficiency_active and flags.get("schema_helpers"):
            make_helpers = _import_optional(
                "taaf_grafts.schema_helpers", "make_schema_helpers_toolagent_factory"
            )
            if make_helpers is not None:
                inner_factory = make_helpers(solver_obj)
                active["schema_helpers"] = True

        chain_flags_on = any(flags.get(flag) for flag in _chain_flag_names())
        if efficiency_active or chain_flags_on:
            solver_obj.analyzer_factory = make_analyzer_chain(
                solver_obj, flags, inner_factory=inner_factory
            )
            if efficiency_active:
                active["efficiency"] = True
            for flag in _chain_flag_names():
                if flags.get(flag):
                    active[flag] = True

        bm.solver = solver_obj
        _print_banner(active)
        # Deterministic per-flag "armed" lines so the commit-log gate can
        # verify these grafts installed even on runs where they never fire
        # (banking fires only on wins; recovery only on stalls).
        if active.get("banking"):
            print("[banking] armed")
        if active.get("recovery"):
            print("[recovery] armed")
        if active.get("schema_void"):
            print("[schema_void] armed")
        if active.get("schema_notes"):
            print("[schema_notes] armed")
        if active.get("schema_helpers"):
            print("[schema_helpers] armed")
    except Exception as err:  # noqa: BLE001 — install must never take the run down
        for revert in reversed(reverts):
            try:
                revert()
            except Exception:  # noqa: BLE001
                pass
        try:
            bm.solver = original
        except Exception:  # noqa: BLE001
            pass
        print(f"[taaf_grafts] install failed -> stock: {err}")
