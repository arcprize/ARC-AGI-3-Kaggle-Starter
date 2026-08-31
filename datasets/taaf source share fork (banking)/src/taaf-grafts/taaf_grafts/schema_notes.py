"""Schema-loop deliberation graft: a static probe→plan→commit note on efficiency.

WHY this exists: Schema's harness lift came partly from a hard outer/inner
loop — probe an unknown control cheaply, certify a hypothesis, plan one short
sequence, commit, and void the plan the moment reality diverges. A 27B model
cannot run that loop as executable machinery (no reliable ``world_model.py``),
but it CAN be nudged toward the loop's *shape* by cheap, deterministic prompt
pressure. This graft appends a fixed ~100-token "SCHEMA LOOP" fragment to the
user prompt every turn, on top of the (dynamic) efficiency note, so the model
is reminded to probe-then-observe instead of raster-scanning or marching
blindly. See docs/schema/V12_SCHEMA_PLAN.md, WP2.

INVARIANTS (why this file is shaped the way it is):

- REPORT-ONLY. Exactly like the efficiency graft: the subclass NEVER curbs,
  aborts, or injects actions. It only appends text to the user prompt.
- STATIC PRESSURE, EVERY TURN. Unlike the efficiency note (which stays quiet
  when there is nothing to report), the schema loop note is appended on EVERY
  turn — it is a system-style standing reminder, cheap and deterministic, and
  its value is precisely that it is present *before* waste happens.
- DEGRADE LADDER, NEVER CRASH. ``_build_user_prompt`` wraps the note in a
  blanket try/except and returns the super() (efficiency) prompt unchanged on
  any failure. The factory degrades one rung at a time on construction error:
  ``SchemaNotesToolAgent`` → ``EfficiencyToolAgent`` → stock ``ToolAgent``.
  A broken schema layer can never cost more than the note itself.
- NESTED-UNDER-EFFICIENCY FLAG SEMANTICS. There is no standalone
  ``schema_notes``-without-``efficiency`` mode: this factory is only selected
  by the composite when BOTH the ``efficiency`` and ``schema_notes`` flags are
  on (the floor F always carries ``efficiency``). Flag off => this module is
  never imported => byte-identical stock/efficiency behaviour.
- ZERO VENDOR EDITS. Pure subclass over :class:`EfficiencyToolAgent`; the hot
  ``step_env`` action path is never wrapped or touched.
"""

from __future__ import annotations

from typing import Any, Callable

from inference.agent.runtime_state import Frame, HistoryEntry
from inference.agent.tool_agent import ToolAgent

from taaf_grafts.agent_ext import EfficiencyToolAgent, _resolve_baselines

# The standing deliberation fragment (~100 tokens). Wording is tuned for a 27B
# model: short imperative lines, one idea per line, no jargon beyond "no-op".
SCHEMA_LOOP_NOTE = (
    "SCHEMA LOOP (cheap):\n"
    "1) Unsure what a control does? Commit only 1-3 probe actions, then STOP "
    "and re-observe.\n"
    "2) Prefer one short planned sequence over raster scans or long repeats "
    "of one action.\n"
    "3) If the last batch mostly no-op'd or oscillated, change your "
    "hypothesis — do not resend the same batch.\n"
    "4) Before a long sequence, state the expected board change in one line; "
    "if wrong next turn, replan."
)


def build_schema_loop_note() -> str:
    """Return the per-turn schema loop fragment. Pure; static today.

    Kept as a function (not just the constant) so tests can snapshot it and a
    future version can make it context-dependent (e.g. drop probe advice once
    every control has been certified) without touching the agent override.
    """
    return SCHEMA_LOOP_NOTE


# -- the ToolAgent subclass -------------------------------------------------


class SchemaNotesToolAgent(EfficiencyToolAgent):
    """:class:`EfficiencyToolAgent` that also appends the schema loop note.

    Constructor is inherited verbatim from :class:`EfficiencyToolAgent`
    (``game`` + ``baseline_actions`` + all stock ``ToolAgent`` kwargs).
    ``super()._build_user_prompt`` already appends the dynamic efficiency
    note; this override appends the static schema fragment after it,
    newline-separated, every turn. Any failure in the note path returns the
    super() prompt unchanged.
    """

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
            note = build_schema_loop_note()
        except Exception:  # noqa: BLE001 — any failure => efficiency prompt
            return base
        if not note:
            return base
        return f"{base}\n{note}"


# -- factory (selected by composite when ``efficiency`` AND ``schema_notes``
#    flags are both on) ------------------------------------------------------


def make_schema_notes_toolagent_factory(solver: Any) -> Callable[[Any, int], Any]:
    """Return an ``analyzer_factory`` that builds a :class:`SchemaNotesToolAgent`.

    Mirrors ``agent_ext.make_efficiency_toolagent_factory`` (``api_key/
    base_url/provider = None`` so ``ToolAgent`` env-resolves its connection,
    baselines via :func:`_resolve_baselines`) but with a two-rung degrade
    ladder: if :class:`SchemaNotesToolAgent` construction raises for any
    reason, fall back to a plain :class:`EfficiencyToolAgent` (the flag
    semantics guarantee ``efficiency`` is on whenever we are selected); if
    THAT also raises, fall back to a stock ``ToolAgent``. Each rung has its
    own try/except so a game degrades rather than crashes.
    """

    def factory(game: Any, index: int) -> Any:
        try:
            baselines = _resolve_baselines(game)
            return SchemaNotesToolAgent(
                game=game,
                baseline_actions=baselines,
                model=solver.model,
                timeout=solver.analyzer_timeout,
                save_request_logs=solver.save_request_logs,
                api_key=None,
                base_url=None,
                provider=None,
            )
        except Exception:  # noqa: BLE001 — degrade one rung: efficiency
            pass
        try:
            baselines = _resolve_baselines(game)
            return EfficiencyToolAgent(
                game=game,
                baseline_actions=baselines,
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
