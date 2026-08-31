"""Action-efficiency graft: a ``ToolAgent`` subclass that surfaces waste.

WHY this exists: the per-level score is ``min(115, (baseline/actions)^2 * 100)``
— quadratic in efficiency. Forensics of the last real run showed the model
reasons correctly but overspends action baselines 13-26x through (1) net-zero
action bursts (30xLEFT then 30xRIGHT round-trips), (2) raster-scan / exhaustive
enumeration, and (3) death-driven auto-RESETs. The model is self-aware it is
wasting moves but realises it TOO LATE. This graft appends a terse,
per-turn budget note to the user prompt so the model sees the waste AT the
moment it happens and can correct in time.

INVARIANTS (why this file is shaped the way it is):

- REPORT-ONLY. The subclass NEVER curbs, aborts, or injects actions. It only
  appends text to the user prompt. The model's causal action history and its
  intended round-trips are left completely untouched (design non-goal:
  no mid-play mechanical action injection).
- STOCK-IDENTICAL WHEN OFF. The efficiency behaviour exists only inside this
  subclass; the composite builds it ONLY when the ``efficiency`` flag is on.
  Flag off => stock ``ToolAgent`` is constructed => byte-identical prompt.
- DEGRADE TO STOCK ON ANY ERROR. ``_build_user_prompt`` wraps the note
  computation in a blanket try/except and returns the stock prompt on any
  failure; the factory falls back to a stock ``ToolAgent`` if construction
  raises. A broken efficiency layer can never crash a game or corrupt a turn.
- PURE, LLM-FREE DETECTION. :func:`detect_net_zero_cycle`,
  :func:`detect_stagnation`, :func:`count_recent_revisits`,
  :func:`heuristic_action_target` and :func:`build_efficiency_note` are pure
  functions over scripted frame/action sequences, unit-testable with no LLM
  and no GPU.
- BASELINE-FREE PRESSURE. The load-bearing "commit / do not scan" reminder and
  the over-target budget line must fire even when NO per-level baseline is
  available. The real competition rerun strips baselines (the REST API omits
  ``baseline_actions`` and env ids are anonymised clones), so on the hidden set
  ``_resolve_baselines`` returns ``None`` for every game. To keep the pressure
  firing there, the note synthesizes a generous heuristic action target (a
  baseline PROXY) and adds pure, baseline-free stall detectors (stagnation,
  state revisits) that gate the reminder independently of any baseline. When a
  REAL baseline is available (offline / commit run) it is preferred; the
  heuristic is only the fallback.

The waste signal is derived entirely from the runtime-state history the model
already sees each turn (``load_runtime_state`` frames), so the hot ``step_env``
action path is never wrapped or touched.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from inference.agent.runtime_state import Frame, HistoryEntry
from inference.agent.tool_agent import ToolAgent

# A net-zero round-trip must span at least this many actions to be reported;
# below this a same-state match is a trivial no-op probe, not a waste burst.
_MIN_ROUNDTRIP_ACTIONS = 6
# Bound the backward scan so long games stay O(window) per turn.
_NET_ZERO_WINDOW = 240

# A run of at least this many consecutive no-board-change actions reads as a
# stall (the model is hammering an inert control / stuck on a screen).
_STAGNATION_MIN_RUN = 8
# The current state recurring at least this many times in the recent window
# reads as oscillation, even when no single clean net-zero round-trip exists.
_REVISIT_MIN = 4
# Bound the revisit scan (per-cell near-match is heavier than a tuple compare).
_REVISIT_WINDOW = 120

# Heuristic per-level action target, synthesized when the real baseline is
# hidden. The 25 public games' per-level baselines span ~8-300+ (median
# ~30-60); this returns an intentionally GENEROUS upper-middle figure so the
# escalation reminder never nags a genuinely short level, yet still fires once
# a level drags on. See :func:`heuristic_action_target`.
_HEURISTIC_BASE_TARGET = 50
_HEURISTIC_PER_ACTION = 5
_HEURISTIC_BOARD_CAP = 30
_HEURISTIC_MIN_TARGET = 40
_HEURISTIC_MAX_TARGET = 100


# -- pure waste detection ---------------------------------------------------


def detect_net_zero_cycle(
    current_frame: Frame | None,
    history_frames: Sequence[Frame | None],
    *,
    min_roundtrip_actions: int = _MIN_ROUNDTRIP_ACTIONS,
    window: int = _NET_ZERO_WINDOW,
) -> int | None:
    """Return the length (in actions) of the shortest recent net-zero
    round-trip that returned the board to ``current_frame``'s exact grid, or
    ``None``.

    A round-trip is reported only when (a) the current grid re-appears at a
    prior same-level frame, (b) at least one intervening frame DIVERGED from
    it (so a genuinely static/no-op board is not flagged), and (c) the span
    is ``>= min_roundtrip_actions``. Pure: no I/O, no LLM.

    ``history_frames`` is the full per-game frame history (oldest first);
    its last element is the current post-action frame. The scan walks
    backwards, stops at the first frame from a different level, and looks at
    most ``window`` frames back.
    """
    if current_frame is None or not history_frames:
        return None
    cur_grid = current_frame.grid
    cur_level = current_frame.level
    recent = list(history_frames[-max(1, window):])
    n = len(recent)
    saw_divergence = False
    for k in range(1, n + 1):
        frame = recent[n - k]
        if frame is None or frame.level != cur_level:
            break
        if frame.grid != cur_grid:
            saw_divergence = True
            continue
        cycle_actions = k - 1
        if saw_divergence and cycle_actions >= min_roundtrip_actions:
            return cycle_actions
    return None


def detect_stagnation(
    current_frame: Frame | None,
    history_frames: Sequence[Frame | None],
    *,
    min_run: int = _STAGNATION_MIN_RUN,
    window: int = _NET_ZERO_WINDOW,
) -> int | None:
    """Return how many consecutive most-recent same-level actions left the grid
    exactly equal to ``current_frame``'s grid, if that run is ``>= min_run``,
    else ``None``.

    Baseline-free stall signal: walks backward from the frame just before the
    current one over same-level frames, counting identical grids and stopping
    at the first frame that differs (or crosses a level boundary). A long
    no-change run means the model is issuing actions with zero board effect
    (stuck on a screen / hammering an inert control) — waste that needs no
    baseline to recognise. Pure: no I/O, no LLM.
    """
    if current_frame is None or not history_frames:
        return None
    cur_grid = current_frame.grid
    cur_level = current_frame.level
    recent = list(history_frames[-max(1, window):])
    n = len(recent)
    run = 0
    # k=1 is the current frame itself; start at its predecessor.
    for k in range(2, n + 1):
        frame = recent[n - k]
        if frame is None or frame.level != cur_level or frame.grid != cur_grid:
            break
        run += 1
    return run if run >= min_run else None


def count_recent_revisits(
    current_frame: Frame | None,
    history_frames: Sequence[Frame | None],
    *,
    window: int = _REVISIT_WINDOW,
) -> int:
    """Return how many recent same-level frames (excluding the current one)
    re-present ``current_frame``'s grid EXACTLY.

    Baseline-free oscillation signal that catches thrashing among a small set
    of states that a single net-zero round-trip check misses. EXACT-match only,
    deliberately: a near-match tolerance treats incremental single-object motion
    (clear one cell + set one cell = 2 changed cells) as a repeat, so it would
    flag a legitimate avatar marching across the board as "cycling" on every
    turn of genuine linear progress — contradictory, harmful advice on a core
    ARC-AGI-3 mechanic. Only a true return to a previously-occupied exact state
    is a revisit. Walks backward over same-level frames, stopping at the first
    level boundary. Pure: no I/O, no LLM.
    """
    if current_frame is None or not history_frames:
        return 0
    cur_grid = current_frame.grid
    cur_level = current_frame.level
    recent = list(history_frames[-max(1, window):])
    n = len(recent)
    count = 0
    for k in range(2, n + 1):
        frame = recent[n - k]
        if frame is None or frame.level != cur_level:
            break
        if frame.grid == cur_grid:
            count += 1
    return count


def heuristic_action_target(
    valid_action_count: int | None,
    board_cells: int | None,
) -> int:
    """Synthesize a generous per-level soft action target — a baseline PROXY —
    for when the real per-level baseline is hidden.

    Defensible without any baseline: the 25 public games' per-level baselines
    span ~8-300+ (median ~30-60), so this returns an intentionally generous
    upper-middle figure — a base allowance plus a few probe actions per
    available valid action and a bounded board-size allowance — clamped to
    ``[_HEURISTIC_MIN_TARGET, _HEURISTIC_MAX_TARGET]``. Being generous keeps the
    escalation reminder from nagging genuinely short levels while still firing
    once a level drags well past a normal solve length. Pure: no I/O, no LLM.
    """
    target = _HEURISTIC_BASE_TARGET
    if valid_action_count and valid_action_count > 0:
        target += _HEURISTIC_PER_ACTION * int(valid_action_count)
    if board_cells and board_cells > 0:
        target += min(_HEURISTIC_BOARD_CAP, int(int(board_cells) ** 0.5))
    return max(_HEURISTIC_MIN_TARGET, min(_HEURISTIC_MAX_TARGET, target))


def build_efficiency_note(
    *,
    level_number: int | None,
    actions_this_level: int | None,
    baseline_this_level: int | None,
    net_zero_actions: int | None,
    heuristic_target: int | None = None,
    stagnation_actions: int | None = None,
    revisit_count: int | None = None,
) -> str:
    """Assemble the per-turn budget note (may be empty). Pure.

    Emits: a header only when there is something to say; a budget line with an
    over-target ratio against either the REAL per-level baseline (preferred) or
    a synthesized generous heuristic PROXY target (``heuristic_target``) when no
    baseline is available; baseline-free stall lines (stagnation, net-zero
    round-trip, state revisits); and the load-bearing commit-and-stop reminder.

    The reminder — and the over-target pressure — fire WITHOUT any baseline:
    they are gated on ``over_budget`` (real OR proxy) OR any baseline-free stall
    signal, so the hidden set (baselines stripped) still gets the pressure that
    the offline commit run got via real baselines.
    """
    used = int(actions_this_level or 0)

    net_zero = (
        int(net_zero_actions)
        if net_zero_actions is not None and net_zero_actions > 0
        else None
    )
    stagnating = (
        int(stagnation_actions)
        if stagnation_actions is not None and stagnation_actions > 0
        else None
    )
    oscillating = (
        int(revisit_count)
        if revisit_count is not None and revisit_count >= _REVISIT_MIN
        else None
    )
    any_stall = net_zero is not None or stagnating is not None or oscillating is not None

    if used <= 0 and not any_stall:
        # Nothing spent yet this level and no waste to flag: stay quiet.
        return ""

    lines: list[str] = [
        "EFFICIENCY BUDGET — your score on each level is "
        "(baseline_actions / your_actions)^2, so every wasted action costs "
        "you quadratically."
    ]

    lvl_label = f"Level {level_number}" if level_number is not None else "This level"

    # Prefer the real baseline; fall back to the generous heuristic proxy.
    target: int | None = None
    target_is_proxy = False
    if baseline_this_level is not None and baseline_this_level > 0:
        target = int(baseline_this_level)
    elif heuristic_target is not None and heuristic_target > 0:
        target = int(heuristic_target)
        target_is_proxy = True

    over_budget = target is not None and used > target

    if target is not None:
        target_word = "typical target" if target_is_proxy else "target"
        if used > target:
            ratio = used / target
            lines.append(
                f"{lvl_label}: you have used {used} actions; a strong score "
                f"needs about {target} or fewer. You are {ratio:.1f}x over the "
                f"{target_word}."
            )
        elif used > 0:
            lines.append(
                f"{lvl_label}: you have used {used} of about {target} "
                f"{target_word} actions so far."
            )
    elif used > 0:
        lines.append(
            f"{lvl_label}: you have used {used} actions on this level so far."
        )

    if stagnating is not None:
        lines.append(
            f"STALL: the board has not changed for your last {stagnating} "
            f"actions on this level — those actions had no visible effect. Stop "
            f"repeating the same move; try a different action or target."
        )

    if net_zero is not None:
        lines.append(
            f"NET-ZERO WASTE: your last {net_zero} actions returned the board "
            f"to a state already seen {net_zero} actions ago — that round-trip "
            f"made no progress yet still cost {net_zero} actions. Do not "
            f"extend-and-retract or oscillate; a sequence that ends where it "
            f"began is pure waste."
        )

    if oscillating is not None:
        lines.append(
            f"REVISIT WASTE: this board state has recurred {oscillating} times "
            f"recently — you are cycling back through states you have already "
            f"visited. Break the loop and commit to a new line of play."
        )

    if over_budget or any_stall:
        lines.append(
            "If you are not steadily making progress toward the level goal, "
            "commit to your single best hypothesis and the shortest sequence "
            "that tests it — do not exhaustively scan rows/columns or enumerate "
            "every option. Test one idea, read the result, then decide."
        )

    if len(lines) == 1:
        # Only the header survived (nothing to report): stay quiet.
        return ""
    return "\n".join(lines)


# -- baseline resolution ----------------------------------------------------


def _resolve_baselines(game: Any) -> list[int] | None:
    """Return per-level baseline action counts for ``game`` or ``None``.

    Primary source: ``game.base_actions_per_level`` (populated by the offline
    arcade and the local rig). Fallback: a best-effort ``metadata.json``
    lookup by ``env_name``. In a real competition rerun both are unavailable
    (baselines are hidden and env ids are anonymised clones) — the note then
    degrades to baseline-free counters.
    """
    live = getattr(game, "base_actions_per_level", None)
    if isinstance(live, (list, tuple)) and live:
        try:
            return [int(x) for x in live]
        except (TypeError, ValueError):
            pass
    env_name = str(getattr(game, "env_name", "") or "").strip()
    if env_name:
        return _load_baselines_from_metadata(env_name)
    return None


def _metadata_roots() -> list[Path]:
    roots: list[Path] = []
    env_root = os.environ.get("ARC_ENVIRONMENT_FILES", "").strip()
    if env_root:
        roots.append(Path(env_root))
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "comp_data" / "extracted" / "environment_files"
        if candidate.is_dir():
            roots.append(candidate)
            break
    roots.append(Path("/kaggle/input"))
    return roots


def _load_baselines_from_metadata(env_name: str) -> list[int] | None:
    key = env_name.lower()
    for root in _metadata_roots():
        try:
            if not root.exists():
                continue
            direct = root / env_name
            search_dirs = [direct] if direct.is_dir() else [root]
            for base_dir in search_dirs:
                for meta in base_dir.rglob("metadata.json"):
                    try:
                        data = json.loads(meta.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        continue
                    gid = str(data.get("game_id", "")).lower()
                    title = str(data.get("title", "")).lower()
                    if gid.startswith(key) or title == key:
                        baselines = data.get("baseline_actions")
                        if isinstance(baselines, list) and baselines:
                            try:
                                return [int(x) for x in baselines]
                            except (TypeError, ValueError):
                                return None
        except Exception:  # noqa: BLE001 — best-effort; never break the run
            continue
    return None


# -- the ToolAgent subclass -------------------------------------------------


class EfficiencyToolAgent(ToolAgent):
    """``ToolAgent`` that appends a per-turn action-efficiency note.

    Constructor matches ``ToolAgent`` exactly (all its keyword-only args are
    forwarded verbatim via ``**tool_agent_kwargs``) plus two efficiency-only
    keywords: ``game`` (for the live per-level action counter) and
    ``baseline_actions`` (the per-level baseline for the budget line). Every
    override degrades to stock behaviour on any error.
    """

    def __init__(
        self,
        *,
        game: Any = None,
        baseline_actions: Sequence[int] | None = None,
        **tool_agent_kwargs: Any,
    ) -> None:
        super().__init__(**tool_agent_kwargs)
        self._eff_game = game
        self._eff_baselines: list[int] | None = (
            [int(x) for x in baseline_actions] if baseline_actions else None
        )

    def _level_and_actions(
        self,
        action_num: int,
        current_frame: Frame | None,
        history_entries: list[HistoryEntry] | None,
    ) -> tuple[int | None, int | None]:
        """Return ``(level_number_1based, actions_this_level)``.

        Primary source is the live ``game.game_run`` per-level counter;
        falls back to counting current-level frames in ``history_entries``.
        """
        gr = getattr(self._eff_game, "game_run", None)
        apl = getattr(gr, "actions_per_level", None)
        lc = getattr(gr, "levels_completed", None)
        if isinstance(apl, (list, tuple)) and apl and isinstance(lc, int):
            idx = min(max(0, lc), len(apl) - 1)
            try:
                return idx + 1, int(apl[idx])
            except (TypeError, ValueError):
                pass
        # Fallback: derive from the frame history the model itself sees.
        level = current_frame.level if current_frame is not None else 1
        if history_entries:
            count = sum(
                1
                for entry in history_entries
                if entry.frame is not None and entry.frame.level == level
            )
            return level, count
        return level, max(0, int(action_num))

    def _efficiency_note(
        self,
        action_num: int,
        current_frame: Frame | None,
        history_entries: list[HistoryEntry] | None,
        valid_actions: list[str] | None,
    ) -> str:
        level_number, actions_this_level = self._level_and_actions(
            action_num, current_frame, history_entries
        )
        baseline_this_level: int | None = None
        if self._eff_baselines and level_number is not None:
            idx = level_number - 1
            if 0 <= idx < len(self._eff_baselines):
                baseline_this_level = self._eff_baselines[idx]
        frames = [
            entry.frame for entry in (history_entries or []) if entry is not None
        ]
        net_zero = detect_net_zero_cycle(current_frame, frames)
        stagnation = detect_stagnation(current_frame, frames)
        revisits = count_recent_revisits(current_frame, frames)
        # Baseline PROXY (used only when no real baseline resolved): a generous
        # per-level target from the valid-action count and board size, so the
        # over-target pressure still fires on the baseline-stripped hidden set.
        board_cells = 0
        if current_frame is not None:
            rows, cols = current_frame.shape
            board_cells = rows * cols
        proxy_target = heuristic_action_target(
            len(valid_actions) if valid_actions else 0, board_cells
        )
        return build_efficiency_note(
            level_number=level_number,
            actions_this_level=actions_this_level,
            baseline_this_level=baseline_this_level,
            net_zero_actions=net_zero,
            heuristic_target=proxy_target,
            stagnation_actions=stagnation,
            revisit_count=revisits,
        )

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
            note = self._efficiency_note(
                action_num, current_frame, history_entries, valid_actions
            )
        except Exception:  # noqa: BLE001 — any failure => stock prompt
            return base
        if not note:
            return base
        return f"{base}\n{note}"


# -- factory (selected by composite when the ``efficiency`` flag is on) ------


def make_efficiency_toolagent_factory(solver: Any) -> Callable[[Any, int], Any]:
    """Return an ``analyzer_factory`` that builds an :class:`EfficiencyToolAgent`.

    Mirrors ``composite.make_stock_toolagent_factory`` exactly (``api_key/
    base_url/provider = None`` so ``ToolAgent`` env-resolves its connection),
    plus baseline resolution. If construction raises for any reason, the
    factory falls back to a stock ``ToolAgent`` so the game degrades to stock
    rather than crashing.
    """

    def factory(game: Any, index: int) -> Any:
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
