"""Schema-helpers graft (WP3): preloaded grid-analysis helpers in the
``python`` tool sandbox, plus a one-line discovery note in the user prompt.

WHY this exists: Schema (the top LB agents' pattern) grounds its reasoning in
``run_python`` experiments over the observed grids — diffing frames, finding
objects, summarizing action effects — instead of eyeballing ASCII. Our 27B
analyzer already has a ``python`` tool but rewrites (often buggily) the same
grid plumbing from scratch every game, burning tokens and tool turns. This
graft preloads small, correct, pure helpers into the sandbox so experimental
physics is one call away, without a world-model harness and without any new
tool schema (WP3 in docs/schema/V12_SCHEMA_PLAN.md).

THE INJECTION SEAM (verified against the vendored source, not the plan's
guess): the python tool does NOT exec in-process. ``ToolAgent._run_python_tool``
(vendor/taaf/src/ARC3-Inference/inference/agent/tool_agent.py:1453) hands the
model's ``code`` string to ``run_sandboxed_python`` (tool_agent.py:1547), which
spawns a fresh subprocess ``[sys.executable, "-I", "-S", "-c", _SANDBOX_BOOTSTRAP]``
(vendor/taaf/src/ARC3-Inference/inference/agent/python_tool_sandbox.py:458-468)
and ships the code as a JSON line over stdin (python_tool_sandbox.py:488-497).
Inside the child, the bootstrap compiles and execs the string under restricted
builtins (``SAFE_BUILTINS``) and a whitelisted ``_safe_import`` (bootstrap
source at python_tool_sandbox.py:58-112, 245-249, 372-375). Host-side globals
injection is therefore impossible — the ONLY host-controllable channel is the
``code`` string itself. So the plan's "option 2: sandbox-globals injection"
becomes, in reality: PREPEND a self-contained, import-free, SAFE_BUILTINS-only
source prelude to the model's code at the ``_run_python_tool`` seam. The
prelude is generated from the module-level functions below via
``inspect.getsource`` (the same trick the vendored sandbox uses for its
``__SEGMENTATION_SOURCE__``, python_tool_sandbox.py:398), so the unit-tested
functions and the injected functions are one and the same source.

INVARIANTS (why this file is shaped the way it is):

- DEGRADE TO STOCK ON ANY ERROR. Prelude preparation is wrapped in a blanket
  try/except; any failure passes the model's arguments through UNCHANGED.
  ``super()._run_python_tool`` is called exactly once — never retried — so
  ``action(...)`` side effects can never double-fire. If the prelude could not
  be built at import time, injection AND the discovery note both stay off
  (never advertise helpers that do not exist). The factory degrades
  SchemaHelpersToolAgent -> EfficiencyToolAgent -> stock ToolAgent.
- MODEL CODE PRESERVED VERBATIM. The model's code is appended after the
  prelude unmodified; the tool-result shape is untouched. Empty-code and
  syntax-error paths are byte-identical to stock: the model's code is
  compile-checked FIRST and any failure falls back to the stock path so the
  stock error text (tool_agent.py:1455-1461) is reproduced exactly. Code
  containing ``from __future__`` skips injection (future imports must lead the
  file). Known cosmetic cost: runtime tracebacks for model code shift line
  numbers by the prelude length.
- ONE KNOB ON TOP OF FLOOR F. Subclasses ``EfficiencyToolAgent`` (NOT stock
  ``ToolAgent``) so the "floor F + schema_helpers" isolation kernel keeps the
  efficiency note; this class only ADDS the sandbox prelude and one prompt
  line. Constructor is inherited unchanged (``game=``, ``baseline_actions=``,
  ``**tool_agent_kwargs``).
- PURE, SANDBOX-LEGAL HELPERS. The helpers use no imports and only names in
  the sandbox's ``SAFE_BUILTINS``; they do no I/O and call no LLM. The prelude
  defines exactly four public names (``grid_diff``, ``connected_components``,
  ``action_effect_summary``, ``recent_history``) plus three ``_sh_``-prefixed
  internals; it executes only ``def`` statements, so it cannot raise at
  injection time, and a model that redefines a helper wins (its code runs
  after the prelude).
- DISCOVERABILITY IS LOAD-BEARING. The WP3 gate is "the model actually calls
  a helper", so ``_build_user_prompt`` appends a short always-on discovery
  line (same degrade-to-super() pattern as ``EfficiencyToolAgent``).
"""

from __future__ import annotations

import inspect
import textwrap
from pathlib import Path
from typing import Any, Callable

from inference.agent.runtime_state import Frame, HistoryEntry
from inference.agent.tool_agent import ToolAgent

from taaf_grafts.agent_ext import EfficiencyToolAgent, _resolve_baselines


# -- pure sandbox helpers -----------------------------------------------------
#
# Everything between here and the prelude assembly is BOTH a normal, unit-
# testable module function AND (via inspect.getsource) the source injected
# into the python-tool subprocess. Rules for this section: no imports, no
# module-level constants, no type annotations, only SAFE_BUILTINS names
# (python_tool_sandbox.py:58-112) — the sandbox has no KeyError/IndexError/
# object/exec/globals, so use except Exception and plain data structures.


def _sh_get(obj, name, default=None):
    """Read ``name`` as a dict key or an object attribute (sandbox views are
    objects; tests and model code may hand in plain dicts)."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _sh_as_grid(x):
    """Coerce ``x`` into a list-of-lists grid, or None if it is not grid-like.

    Accepts the sandbox ``FrameView`` (its grid lives in ``_grid``,
    python_tool_sandbox.py:127-140), the host ``Frame`` (``grid`` of tuples),
    a frame payload dict ({"grid": [[...]]}), or a raw list/tuple of rows —
    so models can pass ``current_frame`` / ``previous_frame`` directly.
    """
    if x is None:
        return None
    grid = None
    for name in ("_grid", "grid"):
        value = getattr(x, name, None)
        if isinstance(value, (list, tuple)):
            grid = value
            break
    if grid is None and isinstance(x, dict):
        value = x.get("grid")
        if isinstance(value, (list, tuple)):
            grid = value
    if grid is None and isinstance(x, (list, tuple)):
        grid = x
    if grid is None:
        return None
    rows = []
    for row in grid:
        if not isinstance(row, (list, tuple)):
            return None
        rows.append(list(row))
    return rows


def _sh_sandbox_transitions():
    """Indirection so ``recent_history`` can reach the sandbox global
    ``transitions`` (installed by the vendored bootstrap,
    python_tool_sandbox.py:344) even though its own parameter shadows the
    name. Raises NameError where the global does not exist (host module)."""
    return transitions  # noqa: F821 — defined by the vendored sandbox bootstrap


def grid_diff(a, b):
    """Cell-level diff of two grids/frames.

    Returns {"n_cells": <changed count>, "bbox": [r0, c0, r1, c1] or None,
    "by_color": {"<old>-><new>": count, ...}} with an INCLUSIVE bbox. Shapes
    may differ: a cell present in only one grid diffs against None (keys like
    "None->3"). Accepts frames (current_frame, previous_frame) or raw
    list-of-lists grids. Pure: no I/O.
    """
    grid_a = _sh_as_grid(a)
    grid_b = _sh_as_grid(b)
    if grid_a is None or grid_b is None:
        raise ValueError(
            "grid_diff expects two grids or frames "
            "(e.g. grid_diff(previous_frame, current_frame))"
        )
    changed = []
    by_color = {}
    for r in range(max(len(grid_a), len(grid_b))):
        row_a = grid_a[r] if r < len(grid_a) else []
        row_b = grid_b[r] if r < len(grid_b) else []
        for c in range(max(len(row_a), len(row_b))):
            va = row_a[c] if c < len(row_a) else None
            vb = row_b[c] if c < len(row_b) else None
            if va != vb:
                changed.append((r, c))
                key = "{0}->{1}".format(va, vb)
                by_color[key] = by_color.get(key, 0) + 1
    if not changed:
        return {"n_cells": 0, "bbox": None, "by_color": {}}
    rows = [cell[0] for cell in changed]
    cols = [cell[1] for cell in changed]
    return {
        "n_cells": len(changed),
        "bbox": [min(rows), min(cols), max(rows), max(cols)],
        "by_color": by_color,
    }


def connected_components(grid, colors=None):
    """Connected components of same-colored cells, 4-connectivity.

    4-connectivity (up/down/left/right) is the deliberate choice: ARC objects
    read as orthogonally-glued blocks, and diagonal-only contact usually means
    SEPARATE objects — so diagonally touching same-color cells are distinct
    components. ``colors`` restricts which colors get components (single int
    or iterable); None means every cell. Returns a list of
    {"color", "size", "bbox": [r0, c0, r1, c1], "cells": [[r, c], ...]}
    sorted by size (largest first), then bbox position; cells are sorted
    row-major. Pure: no I/O.
    """
    cells_grid = _sh_as_grid(grid)
    if cells_grid is None:
        raise ValueError("connected_components expects a grid or frame")
    want = None
    if colors is not None:
        want = {colors} if isinstance(colors, int) else set(colors)
    seen = [[False] * len(row) for row in cells_grid]
    components = []
    for r in range(len(cells_grid)):
        for c in range(len(cells_grid[r])):
            if seen[r][c]:
                continue
            color = cells_grid[r][c]
            if want is not None and color not in want:
                seen[r][c] = True
                continue
            queue = [(r, c)]
            seen[r][c] = True
            cells = []
            i = 0
            while i < len(queue):
                rr, cc = queue[i]
                i += 1
                cells.append([rr, cc])
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if (
                        0 <= nr < len(cells_grid)
                        and 0 <= nc < len(cells_grid[nr])
                        and not seen[nr][nc]
                        and cells_grid[nr][nc] == color
                    ):
                        seen[nr][nc] = True
                        queue.append((nr, nc))
            cells.sort()
            row_ids = [cell[0] for cell in cells]
            col_ids = [cell[1] for cell in cells]
            components.append(
                {
                    "color": color,
                    "size": len(cells),
                    "bbox": [min(row_ids), min(col_ids), max(row_ids), max(col_ids)],
                    "cells": cells,
                }
            )
    components.sort(key=lambda comp: (-comp["size"], comp["bbox"][0], comp["bbox"][1]))
    return components


def action_effect_summary(before, after):
    """One-line human-readable summary of what changed between two
    grids/frames, built on grid_diff. Pure: no I/O."""
    diff = grid_diff(before, after)
    if not diff["n_cells"]:
        return "no change"
    pairs = sorted(diff["by_color"].items(), key=lambda item: (-item[1], item[0]))
    shown = ", ".join("{0} x{1}".format(key, count) for key, count in pairs[:4])
    more = ", +{0} more".format(len(pairs) - 4) if len(pairs) > 4 else ""
    bbox = diff["bbox"]
    return "{0} cells changed in rows {1}-{2}, cols {3}-{4}: {5}{6}".format(
        diff["n_cells"], bbox[0], bbox[2], bbox[1], bbox[3], shown, more
    )


def recent_history(n=10, transitions=None):
    """Read-only compact view of the last ``n`` transitions:
    [{"action": str, "board_changed": bool or None, "level": int}, ...].

    Reads the sandbox global ``transitions`` (the bootstrap's
    ActionTransitionView list) unless an explicit list is passed.
    ``board_changed`` prefers the transition's action-result flag and falls
    back to a before/after grid comparison; None when undecidable (e.g. the
    first transition has no before frame). Pure and read-only: never mutates
    history, never issues actions.
    """
    if transitions is None:
        try:
            transitions = _sh_sandbox_transitions()
        except Exception:
            return []
    if not isinstance(transitions, (list, tuple)):
        return []
    try:
        count = int(n)
    except Exception:
        return []
    if count <= 0:
        return []
    out = []
    for t in list(transitions)[-count:]:
        after = _sh_get(t, "after_frame")
        if after is None:
            after = _sh_get(t, "frame")
        result = _sh_get(t, "result")
        changed = None
        if isinstance(result, dict) and "board_changed" in result:
            changed = bool(result.get("board_changed"))
        else:
            before_grid = _sh_as_grid(_sh_get(t, "before_frame"))
            after_grid = _sh_as_grid(after)
            if before_grid is not None and after_grid is not None:
                changed = before_grid != after_grid
        out.append(
            {
                "action": str(_sh_get(t, "action", "") or ""),
                "board_changed": changed,
                "level": _sh_get(after, "level"),
            }
        )
    return out


# -- prelude assembly ---------------------------------------------------------

# Order is cosmetic (defs only), but keep internals before their callers.
_PRELUDE_FUNCTIONS = (
    _sh_get,
    _sh_as_grid,
    _sh_sandbox_transitions,
    grid_diff,
    connected_components,
    action_effect_summary,
    recent_history,
)


def _build_sandbox_prelude() -> str:
    """Assemble the injected source from the functions above (single source of
    truth — the same trick as the vendored ``__SEGMENTATION_SOURCE__``,
    python_tool_sandbox.py:398) and prove it compiles."""
    parts = [textwrap.dedent(inspect.getsource(fn)) for fn in _PRELUDE_FUNCTIONS]
    source = "\n".join(parts)
    compile(source, "<schema_helpers_prelude>", "exec")
    return source


try:
    SANDBOX_HELPERS_PRELUDE = _build_sandbox_prelude()
except Exception:  # noqa: BLE001 — no prelude => injection and note stay off
    SANDBOX_HELPERS_PRELUDE = ""


# Discovery note appended to every user prompt (<= ~80 tokens). The WP3 gate
# is "the model actually CALLS a helper", so this line is always on (when the
# prelude exists) rather than gated on game state.
HELPERS_PROMPT_NOTE = (
    "PYTHON HELPERS preloaded in the python tool: grid_diff(a,b), "
    "connected_components(grid, colors=None), action_effect_summary(before,after), "
    "recent_history(n). They accept frames (current_frame, previous_frame) or raw "
    "grids. Use them instead of rewriting grid code."
)


# -- the ToolAgent subclass ---------------------------------------------------


class SchemaHelpersToolAgent(EfficiencyToolAgent):
    """``EfficiencyToolAgent`` + sandbox-preloaded grid helpers (WP3).

    Subclasses ``EfficiencyToolAgent`` — NOT stock ``ToolAgent`` — so the
    "floor F + schema_helpers" isolation kernel is a true one-knob addition:
    the efficiency budget note is preserved and this class only adds (a) the
    helpers prelude in the python sandbox and (b) one discovery line in the
    user prompt. Constructor is inherited unchanged (``game=``,
    ``baseline_actions=``, ``**tool_agent_kwargs``). Every override degrades
    to the parent's behaviour on any error.
    """

    def _with_helpers_prelude(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Return ``arguments`` with the prelude prepended to ``code``, or the
        original ``arguments`` whenever injection must stay out of the way.

        Byte-identical stock behaviour is preserved for: empty code (stock's
        non-empty-code error), syntax errors (the model's code is compiled
        FIRST, so the stock error text reports the model's own line numbers),
        and ``from __future__`` code (future imports must lead the file).
        Any exception here is caught by ``_run_python_tool``.
        """
        code = str(arguments.get("code", ""))
        if not SANDBOX_HELPERS_PRELUDE or not code.strip():
            return arguments
        if "__future__" in code:
            return arguments
        compile(code, "<python_tool>", "exec")
        combined = SANDBOX_HELPERS_PRELUDE + "\n" + code
        compile(combined, "<python_tool>", "exec")
        prepared = dict(arguments)
        prepared["code"] = combined
        return prepared

    def _run_python_tool(self, state_path: Path, arguments: dict[str, Any]) -> Any:
        prepared = arguments
        try:
            prepared = self._with_helpers_prelude(arguments)
        except Exception:  # noqa: BLE001 — any prep failure => stock path unchanged
            prepared = arguments
        # Called exactly once, never retried: action() side effects can't double-fire.
        return super()._run_python_tool(state_path, prepared)

    def _helpers_discovery_note(self) -> str:
        # Only advertise what injection can actually deliver: with no prelude
        # (getsource failed), stay silent so the model is never told about
        # helpers that do not exist.
        if not SANDBOX_HELPERS_PRELUDE:
            return ""
        return HELPERS_PROMPT_NOTE

    def _build_user_prompt(
        self,
        action_num: int,
        *,
        valid_actions: list[str] | None,
        current_frame: Frame | None = None,
        history_entries: list[HistoryEntry] | None = None,
        previous_step_summary: dict[str, Any] | None = None,
    ) -> str:
        base = super()._build_user_prompt(
            action_num,
            valid_actions=valid_actions,
            current_frame=current_frame,
            history_entries=history_entries,
            previous_step_summary=previous_step_summary,
        )
        try:
            note = self._helpers_discovery_note()
        except Exception:  # noqa: BLE001 — any failure => parent's prompt
            return base
        if not note:
            return base
        return f"{base}\n{note}"


# -- factory (selected by composite when the ``schema_helpers`` flag is on) ---


def make_schema_helpers_toolagent_factory(solver: Any) -> Callable[[Any, int], Any]:
    """Return an ``analyzer_factory`` that builds a :class:`SchemaHelpersToolAgent`.

    Mirrors ``make_efficiency_toolagent_factory`` (``api_key/base_url/provider
    = None`` so ``ToolAgent`` env-resolves its connection; baselines via
    ``_resolve_baselines``). Degrade ladder, each rung in its own try/except:
    SchemaHelpersToolAgent -> EfficiencyToolAgent (floor F prompt survives) ->
    stock ToolAgent — a broken helpers layer can never crash a game.
    """

    def factory(game: Any, index: int) -> Any:
        try:
            return SchemaHelpersToolAgent(
                game=game,
                baseline_actions=_resolve_baselines(game),
                model=solver.model,
                timeout=solver.analyzer_timeout,
                save_request_logs=solver.save_request_logs,
                api_key=None,
                base_url=None,
                provider=None,
            )
        except Exception:  # noqa: BLE001 — fall to the efficiency rung
            pass
        try:
            return EfficiencyToolAgent(
                game=game,
                baseline_actions=_resolve_baselines(game),
                model=solver.model,
                timeout=solver.analyzer_timeout,
                save_request_logs=solver.save_request_logs,
                api_key=None,
                base_url=None,
                provider=None,
            )
        except Exception:  # noqa: BLE001 — degrade to stock ToolAgent
            return ToolAgent(
                model=solver.model,
                timeout=solver.analyzer_timeout,
                save_request_logs=solver.save_request_logs,
                api_key=None,
                base_url=None,
                provider=None,
            )

    return factory
