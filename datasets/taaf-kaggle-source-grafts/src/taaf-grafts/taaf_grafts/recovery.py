"""Recovery graft: control-flow layer that un-sticks a stalled game session.

WHY this exists: forensics of the v8 commit run showed the two dominant score
losses are CONTROL-FLOW failures the model cannot self-correct out of, and
that report-only prompt pressure (the efficiency note) demonstrably does not
change behaviour:

- GAME_OVER confusion loop (m0r0, scored 0.00): after the first death the
  agent spent ~half the whole per-game wall clock re-reading a chat history
  soaked in GAME_OVER turns ("Goal model: Unknown" x112, 137/172 turns stuck)
  even though the session auto-RESET gave it a fresh board.
- Hypothesis lock-in (SPACE x71 on a HUD bar): one wrong mechanic
  brute-forced for hundreds of actions with no discriminating probe.
- One-level-deeper wall (sk48): level 0 cleared, then level 1 stalled to the
  wall; the vendor level-transition wipe discards every mechanic learned.

``RecoveryLayer`` is an analyzer-chain layer (composed INSIDE RetryGuard,
outside the (Efficiency)ToolAgent) with three mechanisms:

- R1 REFRESH: on a detected death spiral / post-death stall / lock-in stall,
  clear the inner agent's chat history in place and write a synthesized
  fresh-start world model into the six knowledge keys the vendor wipe already
  owns — including a HYPOTHESIS GRAVEYARD of the world models held at each
  death (snapshotted at payload time, BEFORE the vendor wipe destroys them).
  Zero action cost.
- R2 PROBE: on a detected deathless lock-in, execute a bounded scripted probe
  burst (single actions through the normal ``step_env`` path) and feed the
  per-action evidence table back into the knowledge fields. At most
  ``PROBE_MAX_ACTIONS`` actions, at most once per level, only on a level that
  is already far past any plausible baseline.
- R3 HANDOFF: on every ``level_completed`` payload, distill the current
  world/goal/action models into one line appended to ``cross_level_notes`` —
  the only knowledge key the vendor level-transition wipe spares. Captured at
  payload-observation time, i.e. before that wipe runs.

INVARIANTS (matching the retry_guard/banking reference standard):

- The inner ``analyze`` call is never wrapped in try/except: an inner crash
  propagates exactly as stock.
- ALL recovery logic (tracking, detection, interventions, observation) is
  blanket-guarded; any internal error degrades the turn to stock behaviour.
- Every poke at inner-agent state (``_history_messages``,
  ``_summarized_knowledge``, ``_last_step_summary``) is hasattr/isinstance
  guarded; a non-ToolAgent inner degrades to observe-only.
- The ``step_env`` wrapper calls the wrapped callable BARE and only observes
  the returned payload inside its own guard; the action path is untouched.
- Flag off => this module is never imported (composite chain selection).
- Detection is pure and LLM-free: every detector/builder below is a pure
  function over frames/entries/payloads, unit-testable with no LLM, no GPU.

Score-safety: R1/R3 spend zero actions and only rewrite LLM-facing context in
states that, in the entire observed v8 run, yielded zero further levels. R2
spends <= 16 actions, once per level, only when the level already has >= 120
actions (>= 2-4x any plausible baseline, where the quadratic per-level factor
is already tiny); an unstuck level is worth an entire level weight plus access
to deeper (heavier) levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from inference.agent.runtime_state import Frame, HistoryEntry, load_runtime_state

# -- thresholds (see module docstring; conservative by design) ----------------

NOVELTY_WINDOW = 50
NOVELTY_MIN_NEW = 2  # <= this many new grids in a FULL window == flatline
DOMINANCE_WINDOW = 60
DOMINANCE_SHARE = 0.60

REFRESH_DEATHS_FAST = 3
REFRESH_MIN_ACTS = 30
REFRESH_POST_DEATH_ACTS = 50
REFRESH_LOCKIN_ACTS = 180
REFRESH_COOLDOWN_ACTS = 60
REFRESH_MAX_PER_LEVEL = 2

PROBE_MIN_ACTS = 120
PROBE_MAX_PER_LEVEL = 1
PROBE_REPEATS = 2
PROBE_MAX_ACTIONS = 16
PROBE_AFTER_REFRESH_ACTS = 30
MOUSE_PROBE_POINTS = ((16, 16), (16, 48), (48, 16), (48, 48))

GRAVEYARD_MAX_ENTRIES = 4
GRAVEYARD_ENTRY_MAX_CHARS = 200
HANDOFF_MAX_CHARS = 1200

_MAX_EVENTS = 4096

# The six knowledge keys the vendor level/death wipe owns (tool_agent.py
# _update_summarized_knowledge_from_step_summary). ``cross_level_notes`` is
# deliberately NOT here: recovery never overwrites it (R3 only appends).
WIPED_KNOWLEDGE_KEYS = (
    "world_model",
    "goal_model",
    "action_model",
    "recent_findings",
    "open_questions",
    "current_plan",
)


# -- pure detection ------------------------------------------------------------


def _entry_frames(entries: Sequence[HistoryEntry | None]) -> list[Frame | None]:
    return [e.frame if e is not None else None for e in entries]


def level_action_count(entries: Sequence[HistoryEntry | None], level: int) -> int:
    """How many history entries carry a frame on ``level``. Pure."""
    return sum(
        1
        for e in entries
        if e is not None and e.frame is not None and e.frame.level == level
    )


def count_new_frames(
    entries: Sequence[HistoryEntry | None],
    level: int,
    *,
    window: int = NOVELTY_WINDOW,
) -> tuple[int, bool]:
    """Return ``(n_new, window_full)`` for the last ``window`` same-level
    entries: how many of their grids never appeared earlier on that level.

    ``window_full`` is False until the level has more than ``window`` entries,
    so the flatline signal cannot fire on a young level. Pure.
    """
    level_grids = [
        e.frame.grid
        for e in entries
        if e is not None and e.frame is not None and e.frame.level == level
    ]
    if len(level_grids) <= window:
        return len(set(level_grids)), False
    earlier = set(level_grids[:-window])
    recent = level_grids[-window:]
    n_new = 0
    seen_in_recent: set[Any] = set()
    for grid in recent:
        if grid not in earlier and grid not in seen_in_recent:
            n_new += 1
        seen_in_recent.add(grid)
    return n_new, True


def _collapse_action_display(display: str) -> str:
    name = (display or "").strip()
    if name.upper().startswith("MOUSE"):
        return "MOUSE"
    return name


def dominant_action_share(
    entries: Sequence[HistoryEntry | None],
    level: int,
    *,
    window: int = DOMINANCE_WINDOW,
) -> tuple[str, float]:
    """Return ``(action_name, share)`` for the most common collapsed action
    display among the last ``window`` same-level entries. ``MOUSE(row=..,
    col=..)`` collapses to ``MOUSE``; ``RESET``/empty entries are excluded.
    Share is 0.0 when fewer than ``window`` same-level actions exist (young
    levels never trip dominance). Pure.
    """
    names = [
        _collapse_action_display(e.action)
        for e in entries
        if e is not None and e.frame is not None and e.frame.level == level
    ]
    names = [n for n in names if n and n.upper() != "RESET"]
    if len(names) < window:
        return "", 0.0
    recent = names[-window:]
    counts: dict[str, int] = {}
    for n in recent:
        counts[n] = counts.get(n, 0) + 1
    top = max(counts, key=lambda k: counts[k])
    return top, counts[top] / float(len(recent))


@dataclass
class LevelState:
    """Per-level tracking (engine ``frame.level`` is the opaque key)."""

    level: int
    deaths: int = 0
    death_mark_pending: bool = False
    acts_at_last_death: int | None = None
    acts_seen: int = 0
    refreshes: int = 0
    deaths_at_last_refresh: int = 0
    probes: int = 0
    acts_at_last_intervention: int | None = None
    graveyard: list[str] = field(default_factory=list)
    handoff_done: bool = False


def refresh_reason(
    state: LevelState,
    acts: int,
    novelty_new: int,
    window_full: bool,
    dom_share: float,
) -> str | None:
    """Which R1 trigger fires, or None. Pure over the given signals."""
    if state.refreshes >= REFRESH_MAX_PER_LEVEL:
        return None
    if (
        state.acts_at_last_intervention is not None
        and acts - state.acts_at_last_intervention < REFRESH_COOLDOWN_ACTS
    ):
        return None
    flatline = window_full and novelty_new <= NOVELTY_MIN_NEW
    if (
        state.deaths >= REFRESH_DEATHS_FAST
        and acts >= REFRESH_MIN_ACTS
        and state.deaths > state.deaths_at_last_refresh
    ):
        # The new-deaths requirement means a second spiral refresh needs new
        # evidence (another death), never just cooldown expiry.
        return "death_spiral"
    if (
        state.deaths >= 1
        and state.acts_at_last_death is not None
        and acts - state.acts_at_last_death >= REFRESH_POST_DEATH_ACTS
        and flatline
    ):
        return "post_death_stall"
    if (
        state.deaths == 0
        and acts >= REFRESH_LOCKIN_ACTS
        and state.probes >= 1
        and (flatline or dom_share >= DOMINANCE_SHARE)
    ):
        return "lockin_stall"
    return None


def probe_due(
    state: LevelState,
    acts: int,
    novelty_new: int,
    window_full: bool,
    dom_share: float,
) -> bool:
    """Whether the R2 probe burst fires now. Pure over the given signals."""
    if state.probes >= PROBE_MAX_PER_LEVEL:
        return False
    if acts < PROBE_MIN_ACTS:
        return False
    if (
        state.acts_at_last_intervention is not None
        and acts - state.acts_at_last_intervention < PROBE_AFTER_REFRESH_ACTS
    ):
        return False
    flatline = window_full and novelty_new <= NOVELTY_MIN_NEW
    return flatline or dom_share >= DOMINANCE_SHARE


def build_probe_plan(valid_actions: Sequence[str] | None) -> list[dict[str, Any]]:
    """Scripted probe burst: ``PROBE_REPEATS`` presses of each non-mouse valid
    action plus the four fixed quadrant clicks when the mouse is available,
    capped at ``PROBE_MAX_ACTIONS`` single-action ``step_env`` payloads. Pure.
    """
    plan: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in valid_actions or ():
        name = str(raw or "").strip()
        if not name or name.upper() == "RESET" or name in seen:
            continue
        seen.add(name)
        if name.upper() == "ACTION6" or name.upper() == "MOUSE":
            for row, col in MOUSE_PROBE_POINTS:
                plan.append({"action": name, "row": row, "col": col})
        else:
            for _ in range(PROBE_REPEATS):
                plan.append({"action": name})
    return plan[:PROBE_MAX_ACTIONS]


def diff_cells(before: Frame | None, after: Frame | None) -> tuple[int, str]:
    """(changed-cell count, compact region description) between two frames.

    Returns ``(0, "")`` when either frame is missing or shapes are unusable.
    Pure.
    """
    if before is None or after is None:
        return 0, ""
    bg, ag = before.grid, after.grid
    rows = max(len(bg), len(ag))
    changed = 0
    r0 = c0 = None
    r1 = c1 = -1
    for r in range(rows):
        brow = bg[r] if r < len(bg) else ()
        arow = ag[r] if r < len(ag) else ()
        cols = max(len(brow), len(arow))
        for c in range(cols):
            bv = brow[c] if c < len(brow) else None
            av = arow[c] if c < len(arow) else None
            if bv != av:
                changed += 1
                if r0 is None:
                    r0, c0 = r, c
                r1 = max(r1, r)
                c0 = min(c0, c)
                c1 = max(c1, c)
    if changed == 0 or r0 is None:
        return changed, ""
    return changed, f"rows {r0}-{r1} cols {c0}-{c1}"


def summarize_probe_observations(observations: Sequence[dict[str, Any]]) -> str:
    """One line per probe: action, board effect, events. Pure."""
    lines = []
    for obs in observations:
        action = str(obs.get("action", "?"))
        if obs.get("error"):
            lines.append(f"- {action}: rejected ({obs['error']})")
            continue
        effect = "no visible change"
        if obs.get("board_changed"):
            count = obs.get("changed_cells") or 0
            region = obs.get("changed_region") or ""
            effect = f"changed {count} cell(s)" + (f" in {region}" if region else "")
        events = [
            label
            for key, label in (
                ("game_over", "GAME_OVER"),
                ("level_completed", "LEVEL_COMPLETED"),
                ("run_complete", "RUN_COMPLETE"),
            )
            if obs.get(key)
        ]
        suffix = f" [{', '.join(events)}]" if events else ""
        lines.append(f"- {action}: {effect}{suffix}")
    return "\n".join(lines)


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_graveyard_entry(
    knowledge: dict[str, Any] | None, *, deaths: int, acts: int
) -> str:
    """Distill the world model held AT a death into one graveyard line. Pure."""
    knowledge = knowledge or {}
    parts = []
    for key, label in (
        ("goal_model", "goal"),
        ("action_model", "actions"),
        ("current_plan", "plan"),
        ("world_model", "world"),
    ):
        value = _clip(knowledge.get(key) or "", 60)
        if value:
            parts.append(f"{label}={value}")
    body = "; ".join(parts) if parts else "no explicit model recorded"
    return _clip(f"death #{deaths} at ~{acts} acts: {body}", GRAVEYARD_ENTRY_MAX_CHARS)


def build_fresh_start(
    *,
    level: int,
    deaths: int,
    acts: int,
    reason: str,
    graveyard: Sequence[str],
    valid_actions: Sequence[str] | None,
    probe_findings: str = "",
) -> dict[str, str]:
    """The six wiped-key values for an R1 fresh start. Never includes
    ``cross_level_notes``. Pure.
    """
    grave_lines = [f"  {i + 1}) {g}" for i, g in enumerate(graveyard)]
    grave_block = (
        "HYPOTHESIS GRAVEYARD — models held at each failure, do NOT re-run "
        "them blindly:\n" + "\n".join(grave_lines)
        if grave_lines
        else "No recorded prior hypotheses — treat everything as untested."
    )
    world = (
        f"FRESH RESTART on level {level} after {deaths} GAME_OVER(s) and "
        f"{acts} actions ({reason}). The previous conversation was cleared "
        f"because it was stuck in a failure loop; the board itself is live and "
        f"playable right now. {grave_block}"
    )
    actions_line = (
        "Valid actions: " + ", ".join(str(a) for a in (valid_actions or ()))
        if valid_actions
        else ""
    )
    open_q = (
        "Which mechanic class actually drives this level? Rule classes to "
        "test one at a time: movement/navigation, toggling/painting cells, "
        "selection then confirm, timing, click targets."
    )
    plan = (
        "Form ONE new hypothesis that is NOT in the graveyard, test it with "
        "the shortest discriminating sequence (<=5 actions), read the result, "
        "then commit or discard. Do not repeat any graveyard hypothesis and do "
        "not brute-force one action repeatedly."
    )
    findings = probe_findings or (
        "Prior findings were discarded with the stuck conversation; rebuild "
        "from the current board."
    )
    return {
        "world_model": _clip(world, 900),
        "goal_model": "Unknown — re-derive from the current board, not from memory.",
        "action_model": _clip(actions_line, 300),
        "recent_findings": _clip(findings, 700),
        "open_questions": _clip(open_q, 400),
        "current_plan": _clip(plan, 400),
    }


def distill_handoff(
    knowledge: dict[str, Any] | None, *, level: int, acts: int, deaths: int
) -> str:
    """One-line solved-mechanic summary for ``cross_level_notes``. Pure."""
    knowledge = knowledge or {}
    parts = []
    for key, label in (
        ("goal_model", "goal"),
        ("action_model", "actions"),
        ("world_model", "world"),
    ):
        value = _clip(knowledge.get(key) or "", 90)
        if value:
            parts.append(f"{label}={value}")
    body = "; ".join(parts) if parts else "mechanic not recorded"
    return f"L{level} SOLVED (acts={acts}, deaths={deaths}): {body}"


def merge_cross_level_notes(
    existing: str, line: str, *, max_chars: int = HANDOFF_MAX_CHARS
) -> str:
    """Append ``line`` to the notes, dropping oldest lines past the cap. Pure."""
    line = _clip(line, max_chars)
    lines = [l for l in str(existing or "").splitlines() if l.strip()]
    if line in lines:
        return "\n".join(lines)
    lines.append(line)
    while lines and len("\n".join(lines)) > max_chars:
        lines.pop(0)
    return "\n".join(lines)


# -- the chain layer -----------------------------------------------------------


class RecoveryLayer:
    """Analyzer chain layer: detection + R1/R2/R3 interventions.

    Construct via ``RecoveryLayer(inner_analyzer)``; drop-in for any object
    with the stock ``analyze(...)`` signature. Unknown attributes proxy to the
    inner agent (token counters, ``_timeout``), matching RetryGuard.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._levels: dict[int, LevelState] = {}
        self._current_level: int | None = None
        self.events: list[tuple[Any, ...]] = []

    # -- analyzer protocol ----------------------------------------------------

    def analyze(
        self,
        state_path: Any,
        action_num: int,
        *args: Any,
        valid_actions: Any = None,
        step_env: Any = None,
        transcript_path: Any = None,
        should_stop: Any = None,
        **kwargs: Any,
    ) -> Any:
        wrapped = step_env
        try:
            if step_env is not None:
                wrapped = self._wrap_step_env(step_env, transcript_path)
                self._pre_turn(
                    state_path, valid_actions, wrapped, transcript_path, should_stop
                )
        except Exception:  # noqa: BLE001 — recovery must never break a turn
            wrapped = step_env
        return self._inner.analyze(
            state_path,
            action_num,
            *args,
            valid_actions=valid_actions,
            step_env=wrapped,
            transcript_path=transcript_path,
            should_stop=should_stop,
            **kwargs,
        )

    # -- per-turn tracking + interventions -------------------------------------

    def _pre_turn(
        self,
        state_path: Any,
        valid_actions: Any,
        step_env: Callable[[dict[str, Any]], dict[str, Any]],
        transcript_path: Any,
        should_stop: Any,
    ) -> None:
        current_frame, entries = load_runtime_state(state_path)
        if current_frame is None:
            return
        level = int(current_frame.level)
        state = self._levels.get(level)
        if state is None:
            state = LevelState(level=level)
            self._levels[level] = state
        self._current_level = level

        acts = level_action_count(entries, level)
        state.acts_seen = acts
        if state.death_mark_pending:
            state.acts_at_last_death = acts
            state.death_mark_pending = False

        novelty_new, window_full = count_new_frames(entries, level)
        _dom_name, dom_share = dominant_action_share(entries, level)

        reason = refresh_reason(state, acts, novelty_new, window_full, dom_share)
        if reason is not None:
            self._do_refresh(
                state,
                acts=acts,
                reason=reason,
                valid_actions=valid_actions,
                transcript_path=transcript_path,
            )
            return

        if probe_due(state, acts, novelty_new, window_full, dom_share):
            self._do_probe(
                state,
                state_path=state_path,
                acts=acts,
                valid_actions=valid_actions,
                step_env=step_env,
                transcript_path=transcript_path,
                should_stop=should_stop,
            )

    def _do_refresh(
        self,
        state: LevelState,
        *,
        acts: int,
        reason: str,
        valid_actions: Any,
        transcript_path: Any,
    ) -> None:
        fresh = build_fresh_start(
            level=state.level,
            deaths=state.deaths,
            acts=acts,
            reason=reason,
            graveyard=state.graveyard,
            valid_actions=valid_actions if isinstance(valid_actions, list) else None,
        )
        history = getattr(self._inner, "_history_messages", None)
        if isinstance(history, list):
            history.clear()
        knowledge = getattr(self._inner, "_summarized_knowledge", None)
        if isinstance(knowledge, dict):
            for key in WIPED_KNOWLEDGE_KEYS:
                knowledge[key] = fresh[key]
        if hasattr(self._inner, "_last_step_summary"):
            self._inner._last_step_summary = None
        state.refreshes += 1
        state.deaths_at_last_refresh = state.deaths
        state.acts_at_last_intervention = acts
        self._record(("refresh", state.level, reason, state.deaths, acts))
        self._note(
            transcript_path,
            f"[recovery] refresh fired reason={reason} level={state.level} "
            f"deaths={state.deaths} acts={acts}",
        )

    def _do_probe(
        self,
        state: LevelState,
        *,
        state_path: Any,
        acts: int,
        valid_actions: Any,
        step_env: Callable[[dict[str, Any]], dict[str, Any]],
        transcript_path: Any,
        should_stop: Any,
    ) -> None:
        plan = build_probe_plan(valid_actions if isinstance(valid_actions, list) else None)
        if not plan:
            return
        state.probes += 1
        state.acts_at_last_intervention = acts
        observations: list[dict[str, Any]] = []
        before_frame, _ = load_runtime_state(state_path)
        for arguments in plan:
            if callable(should_stop):
                try:
                    if should_stop():
                        break
                except Exception:  # noqa: BLE001 — broken predicate == stop
                    break
            payload = step_env(dict(arguments))
            if not isinstance(payload, dict):
                break
            after_frame, _ = load_runtime_state(state_path)
            changed, region = diff_cells(before_frame, after_frame)
            action_label = arguments.get("action", "?")
            if "row" in arguments:
                action_label = (
                    f"{action_label}(row={arguments['row']}, col={arguments['col']})"
                )
            observations.append(
                {
                    "action": action_label,
                    "error": payload.get("error"),
                    "board_changed": bool(payload.get("board_changed")),
                    "changed_cells": changed,
                    "changed_region": region,
                    "game_over": bool(payload.get("game_over")),
                    "level_completed": bool(payload.get("level_completed")),
                    "run_complete": bool(payload.get("run_complete")),
                }
            )
            before_frame = after_frame
            if (
                payload.get("error")
                or payload.get("game_over")
                or payload.get("level_completed")
                or payload.get("run_complete")
                or payload.get("terminal")
            ):
                break
        if not observations:
            return
        if all(o.get("error") for o in observations):
            # Every probe was rejected (validity flipped mid-turn): refund the
            # level's single probe so a later, healthier turn can retry; the
            # acts_at_last_intervention cooldown still spaces the retry.
            state.probes -= 1
            return
        table = summarize_probe_observations(observations)
        knowledge = getattr(self._inner, "_summarized_knowledge", None)
        if isinstance(knowledge, dict):
            knowledge["recent_findings"] = _clip(
                "RECOVERY PROBE RESULTS (scripted single-action probes, just "
                "executed — this is fresh ground truth):\n" + table,
                900,
            )
            knowledge["open_questions"] = _clip(
                "Which of the board-changing probe actions above drives the "
                "level goal? Ignore actions probed as having no visible effect.",
                300,
            )
            knowledge["current_plan"] = _clip(
                "Build ONE hypothesis from the probe table (prefer actions "
                "that changed cells near interactive-looking objects) and test "
                "it with the shortest discriminating sequence.",
                300,
            )
        if hasattr(self._inner, "_last_step_summary"):
            executed = [str(o["action"]) for o in observations if not o.get("error")]
            self._inner._last_step_summary = {
                "start_action_num": None,
                "end_action_num": None,
                "executed_count": len(executed),
                "executed_actions": executed,
                "level": state.level,
                "level_transition": any(o["level_completed"] for o in observations),
                "run_complete": any(o["run_complete"] for o in observations),
                "game_over": any(o["game_over"] for o in observations),
                "board_changed": any(o["board_changed"] for o in observations),
                "stop_reason": "recovery_probe",
            }
        self._record(("probe", state.level, len(observations), acts))
        self._note(
            transcript_path,
            f"[recovery] probe fired level={state.level} acts={acts} "
            f"probes={len(observations)}",
        )

    # -- payload observation (graveyard + handoff) -----------------------------

    def _wrap_step_env(
        self,
        step_env: Callable[[dict[str, Any]], dict[str, Any]],
        transcript_path: Any,
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        def wrapped(arguments: dict[str, Any]) -> dict[str, Any]:
            payload = step_env(arguments)
            try:
                self._observe(payload, transcript_path)
            except Exception:  # noqa: BLE001 — observation must never break acting
                pass
            return payload

        return wrapped

    def _observe(self, payload: Any, transcript_path: Any) -> None:
        if not isinstance(payload, dict):
            return
        level_raw = payload.get("level")
        try:
            level = int(level_raw) if level_raw is not None else self._current_level
        except (TypeError, ValueError):
            level = self._current_level
        if level is None:
            return
        state = self._levels.get(level)
        if state is None:
            state = LevelState(level=level)
            self._levels[level] = state

        knowledge = getattr(self._inner, "_summarized_knowledge", None)

        # Only EXECUTED payloads are real events: the session's
        # _terminal_payload (returned for every call on an already-dead
        # board) carries game_over=True with executed falsy, and counting
        # those would inflate deaths into spurious refreshes.
        if payload.get("game_over") and payload.get("executed"):
            state.deaths += 1
            state.death_mark_pending = True
            if isinstance(knowledge, dict) and str(
                knowledge.get("world_model") or ""
            ).startswith("FRESH RESTART") and str(
                knowledge.get("goal_model") or ""
            ).startswith("Unknown"):
                # Nothing learned since the last refresh: a distilled entry
                # would just echo the injected fresh-start text. Skip it.
                pass
            else:
                entry = build_graveyard_entry(
                    knowledge if isinstance(knowledge, dict) else None,
                    deaths=state.deaths,
                    acts=state.acts_seen,
                )
                state.graveyard.append(entry)
                del state.graveyard[:-GRAVEYARD_MAX_ENTRIES]
            self._record(("death", level, state.deaths))

        if payload.get("level_completed") and payload.get("executed"):
            # The payload's "level" is the POST-transition number
            # (solver._level_number = completed+1); the mechanic that was just
            # solved belongs to the level this turn STARTED on, which
            # _pre_turn recorded. Attribute the handoff there.
            done_level = self._current_level if self._current_level is not None else level
            done_state = self._levels.get(done_level)
            if done_state is None:
                done_state = LevelState(level=done_level)
                self._levels[done_level] = done_state
            if not done_state.handoff_done:
                done_state.handoff_done = True
                if isinstance(knowledge, dict):
                    line = distill_handoff(
                        knowledge,
                        level=done_level,
                        acts=done_state.acts_seen,
                        deaths=done_state.deaths,
                    )
                    knowledge["cross_level_notes"] = merge_cross_level_notes(
                        str(knowledge.get("cross_level_notes") or ""), line
                    )
                self._record(("handoff", done_level))
                self._note(
                    transcript_path, f"[recovery] handoff level={done_level}"
                )

    # -- plumbing ---------------------------------------------------------------

    def _record(self, event: tuple[Any, ...]) -> None:
        self.events.append(event)
        if len(self.events) > _MAX_EVENTS:
            del self.events[: len(self.events) - _MAX_EVENTS]

    def _note(self, transcript_path: Any, message: str) -> None:
        print(message, flush=True)
        try:
            if transcript_path is not None:
                with open(transcript_path, "a", encoding="utf-8") as handle:
                    handle.write(message + "\n")
        except Exception:  # noqa: BLE001 — transcript is best-effort
            pass

    # -- transparent proxy to the inner agent -----------------------------------

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        try:
            inner = self.__dict__["_inner"]
        except KeyError as exc:  # during __init__, before _inner is bound
            raise AttributeError(name) from exc
        return getattr(inner, name)
