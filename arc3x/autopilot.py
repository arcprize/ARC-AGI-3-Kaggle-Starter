"""The framework seam: let the pilot take turns the language model would have taken.

WHY A WRAPPER AND NOT A PATCH
-----------------------------
``_HarnessGameSession.play`` calls exactly one thing per turn
(``solver.py:333-342``)::

    result = self.analyzer.analyze(
        self.state_path, self.action_count,
        valid_actions=_engine_action_names(self.game),
        step_env=self.step_env, ...
    )

Every input the pilot needs is in that call - the runtime-state file holds the
current frame and the whole action history, ``valid_actions`` holds the buttons,
and ``step_env`` executes presses. So the entire integration is a decorator around
one method: no solver subclass, no monkeypatched internals, and if the pilot is
removed the call site is byte-identical to stock. That property is the point.
Every submission since 2.14 has scored below it, and two of them scored below it
because a *graft* changed behaviour that nobody had isolated.

:func:`arm` is how it gets there - one patch of ``HarnessSolver._make_analyzer``,
which is called once per game and hands over the ``game`` object. Note that the
object owning ``analyze`` is the *session*, not the solver (``solver.py:195-198``);
aiming at ``HarnessSolver.analyzer`` finds nothing.

WHAT IT COSTS THE PROMPT: NOTHING
---------------------------------
The measured lesson from the v14/v15 regression is that prose is not free - the run
is clock-bound, so a 10 KB addendum re-sent every turn is itself paid for in
actions, and the one addendum section that asked the model to batch its own actions
("One reply can carry several actions") was **inert**: v15's probe shows
``turns == actions`` on all four games, zero batching, while a different section of
the same addendum cost tn36 10.71 -> 4.89. This wrapper adds **zero tokens to the
prompt**. It does not ask the model to batch; it batches.

THE HANDBACK IS THE SAFETY PROPERTY
-----------------------------------
``Pilot.decide`` returns ``None`` whenever the learned model cannot stand behind a
move, and this wrapper then calls the real analyzer, unchanged. So the worst case
is stock behaviour plus a few microseconds of numpy, and the best case is a turn
that spends 40 actions instead of 1. There is no configuration in which the model
is prevented from playing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from arc3x.pilot import Pilot

#: Per-game pilots, keyed by whatever the caller names the game. Module level
#: because the framework builds a solver per game and we want the learned model to
#: survive for the whole game rather than for one turn.
PILOTS: dict[str, Pilot] = {}


#: The only six names ``to_engine_action`` resolves, in both spellings
#: (``action_names.py:7-17``). Checked before the batch is sent because
#: ``_normalize_actions`` rejects the **whole** list on the first unknown name
#: (``solver.py:595-600``) - one bad spelling would throw away a 40-action turn.
_SPEAKABLE = {
    "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6",
    "UP", "DOWN", "LEFT", "RIGHT", "SPACE", "MOUSE",
}


def _sane(payloads: list[dict]) -> list[dict]:
    """Keep the leading run of payloads the framework will certainly accept.

    A *prefix*, not a filter: the presses are a route, so dropping one from the
    middle and closing the gap would send a different journey than the one the
    model verified. Truncating keeps the plan a valid, shorter version of itself.
    """
    out: list[dict] = []
    for p in payloads:
        name = str(p.get("action") or "").strip().upper()
        if name not in _SPEAKABLE or name == "RESET":
            break
        if name in {"ACTION6", "MOUSE"}:
            try:
                if not (0 <= int(p["row"]) <= 63 and 0 <= int(p["col"]) <= 63):
                    break
            except (KeyError, TypeError, ValueError):
                break
        out.append(p)
    return out


def _on_level(history: Sequence[Any], current_level: int) -> int:
    """Actions charged to the level we are standing on right now.

    Mirrors ``Flight._on_level``, and exists because ``Pilot._room`` has to
    measure against the *total* spend on this level rather than the pilot's own
    share - the language model takes turns too, and a laboratory allowance that
    ignored them would be an allowance in name only.

    Counted by walking back to the most recent level change and dropping one. The
    dropped entry is not an off-by-one guard: the earliest entry showing this
    level is either the seed frame the solver writes before any action
    (``solver.py:201-205``), or the frame produced by the action that *cleared*
    the previous level - and the scorer bills a clearing action to the level it
    cleared, not to the one it opened.
    """
    total = 0
    for i in range(len(history) - 1, -1, -1):
        frame = getattr(history[i], "frame", None)
        if int(getattr(frame, "level", current_level) or current_level) != current_level:
            break
        total += 1
    return max(0, total - 1)


def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _int_flag(name: str, default: int) -> int:
    """A bounded integer setting that cannot turn a malformed env var into policy."""
    try:
        return max(0, int(os.environ.get(name, str(default)).strip()))
    except (TypeError, ValueError):
        return default


def _pilot_mode() -> str:
    """Select autonomous, conservative, or mentally planned control.

    ``sidecar`` is the frozen v20 A/B control. ``mental`` permits the new
    self-verified imagination step while retaining the same observation window,
    one-action execution, and per-level action cap.
    """
    mode = os.environ.get("ARC3X_PILOT_MODE", "active").strip().lower()
    return mode if mode in {"active", "sidecar", "mental"} else "active"


@dataclass
class Autopilot:
    """Wraps one analyzer. ``__call__`` has ``analyze``'s exact signature.

    Everything is keyword-tolerant on purpose: the framework has grown arguments
    to ``analyze`` across versions, and a wrapper that enumerated them would break
    silently on the next one. Unknown keywords are passed straight through.
    """

    analyzer: Any
    game: str = "game"
    #: Constructed lazily from the analyzer's own module so this file does not
    #: import the inference package - it has to be importable in a bare notebook
    #: cell to be testable at all.
    result_factory: Callable[..., Any] | None = None
    pilot: Pilot = field(default_factory=Pilot)

    turns: int = 0
    pilot_turns: int = 0
    llm_turns: int = 0
    actions: int = 0
    failures: int = 0
    notes: list[str] = field(default_factory=list)

    def __call__(
        self,
        state_path: Any,
        action_count: int = 0,
        *,
        valid_actions: Sequence[str] | None = None,
        step_env: Callable[..., dict] | None = None,
        **kw: Any,
    ) -> Any:
        self.turns += 1
        plan = None
        if step_env is not None and _flag("ARC3X_PILOT"):
            try:
                mode = _pilot_mode()
                plan = self._think(state_path, valid_actions, sidecar=mode != "active", mental=mode == "mental")
            except Exception as exc:
                # A pilot crash must cost one stock turn, never the game. This is
                # the whole reason the wrapper is a wrapper.
                self.failures += 1
                self._note(f"pilot raised {type(exc).__name__}: {exc}")
                plan = None

        if plan is not None:
            payloads = _sane(plan.payloads(valid_actions))
            if not payloads:
                self._note(f"dropped {plan!r}: no resolvable action names")
                plan = None

        if plan is not None:
            try:
                payload = step_env({"actions": payloads})
            except Exception as exc:
                self.failures += 1
                self._note(f"step_env raised {type(exc).__name__}: {exc}")
                payload = None
            if isinstance(payload, dict) and not payload.get("error"):
                done = int(payload.get("executed_count") or len(payloads))
                self.actions += done
                self.pilot_turns += 1
                self._note(
                    f"{plan.phase} {done}/{len(payloads)}a "
                    f"stop={payload.get('stop_reason') or 'ran'}"
                )
                return self._result(step_executed=True, reasoning=f"pilot:{plan!r}")

        self.llm_turns += 1
        return self.analyzer.analyze(
            state_path,
            action_count,
            valid_actions=valid_actions,
            step_env=step_env,
            **kw,
        )

    # -- the pilot's two inputs -----------------------------------------------

    def _think(
        self,
        state_path: Any,
        valid_actions: Sequence[str] | None,
        *,
        sidecar: bool = False,
        mental: bool = False,
    ):
        """Read the runtime state the solver just wrote, and decide.

        ``load_runtime_state`` is imported here rather than at module scope so this
        file imports cleanly with no framework present; the fallback reader below
        is what makes the same code path testable from a plain JSON file.
        """
        frame, history = _read_state(Path(str(state_path)))
        if frame is None:
            return None
        import numpy as np

        grid = np.asarray(frame.grid, dtype=np.int16)
        if grid.ndim != 2 or not grid.size:
            return None
        level = int(getattr(frame, "level", 1) or 1)
        self.pilot.observe(history, observe_dream=not sidecar or mental)
        if sidecar:
            minimum = _int_flag("ARC3X_PILOT_MIN_HISTORY", 24)
            if len(history) < minimum:
                self._note(f"sidecar observing {len(history)}/{minimum} history entries")
                return None
            return self.pilot.assist(
                grid,
                valid_actions,
                level,
                spent_on_level=_on_level(history, level),
                max_actions=_int_flag("ARC3X_PILOT_SIDECAR_ACTIONS", 4),
                allow_imagination=mental,
            )
        return self.pilot.decide(
            grid, valid_actions, level, spent_on_level=_on_level(history, level)
        )

    def _result(self, **kw: Any) -> Any:
        """An ``AnalyzerTurnResult`` the solver will accept.

        Only ``step_executed`` is required (``tool_agent.py:518-523``), but the
        class is discovered rather than imported so a version that adds a field
        still works: whatever the wrapped analyzer's module calls a turn result is
        what gets built.
        """
        if self.result_factory is not None:
            return self.result_factory(**kw)
        try:
            module = type(self.analyzer).__module__
            cls = getattr(__import__(module, fromlist=["AnalyzerTurnResult"]), "AnalyzerTurnResult")
            self.result_factory = cls
            return cls(**kw)
        except Exception:
            # Duck-typed last resort. `play` reads `.retryable_failure`,
            # `.yielded_control` and `.step_executed` and nothing else.
            self.result_factory = _Result
            return _Result(**kw)

    def _note(self, text: str) -> None:
        if len(self.notes) < 400:
            self.notes.append(f"t{self.turns}: {text}")

    def summary(self) -> str:
        return (
            f"{self.game}: turns={self.turns} pilot={self.pilot_turns} llm={self.llm_turns} "
            f"actions_batched={self.actions} "
            f"a/pilot_turn={self.actions / self.pilot_turns if self.pilot_turns else 0:.1f} "
            f"fail={self.failures} | {self.pilot.summary()}"
        )


@dataclass
class _Result:
    step_executed: bool = False
    retryable_failure: bool = False
    reasoning: str = ""
    yielded_control: bool = False


def _read_state(path: Path):
    """``(frame, history)`` from the solver's runtime-state file.

    Prefers the framework's own loader, so the real data path is exercised when it
    is available. Falls back to reading the JSON directly with the same shapes,
    which is what lets this be tested with no inference package installed.
    """
    try:
        from inference.agent.runtime_state import load_runtime_state

        return load_runtime_state(path)
    except Exception:
        pass
    import json

    if not path.exists():
        return None, []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, []

    def frame_of(raw: Any):
        if not isinstance(raw, dict):
            return None
        grid = raw.get("grid") or []
        rows = tuple(tuple(int(c) for c in row) for row in grid if isinstance(row, (list, tuple)))
        if not rows:
            return None
        return _Frame(grid=rows, step=int(raw.get("step") or 0), level=max(1, int(raw.get("level") or 1)))

    history = []
    for raw in payload.get("history") or ():
        if not isinstance(raw, dict):
            continue
        frame = frame_of(raw.get("frame"))
        if frame is not None:
            history.append(_Entry(action=str(raw.get("action") or ""), frame=frame))
    return frame_of(payload.get("current_frame")), history


@dataclass(frozen=True)
class _Frame:
    grid: tuple[tuple[int, ...], ...]
    step: int
    level: int


@dataclass(frozen=True)
class _Entry:
    action: str
    frame: _Frame


def _game_id(game: Any) -> str:
    """The game's id, from where the framework actually keeps it.

    ``game.game_run.game_id`` is the real location (``solver.py:1376-1378``);
    ``game.game_id`` does not exist. Getting this wrong is not cosmetic - the
    pilots are keyed by it, so a constant fallback would hand game 2 the avatar,
    button deltas and wall map learned on game 1.
    """
    for path in (("game_run", "game_id"), ("game_id",)):
        node: Any = game
        for attr in path:
            node = getattr(node, attr, None)
            if node is None:
                break
        if isinstance(node, str) and node:
            return node
    return "game"


def install(session: Any, game: str | None = None) -> Autopilot | None:
    """Wrap one object's ``analyzer`` attribute in place. Returns the wrapper.

    The object is a ``_HarnessGameSession`` - the thing that owns ``analyze`` and
    the ``play`` loop (``solver.py:195-198``), not the ``HarnessSolver``. Prefer
    :func:`arm`, which reaches every game without anyone having to hold a session.

    Idempotent, and silent on failure: if anything about the object is not the
    shape this expects, it is left exactly as it was and the run proceeds stock.
    A graft that cannot install must cost nothing - the v14/v15 grafts failed to
    import and were *harmless*; the prose shipped alongside them was not.
    """
    analyzer = getattr(session, "analyzer", None)
    if analyzer is None or _is_shim(analyzer):
        return None
    name = game or _game_id(getattr(session, "game", None))
    wrapped, auto = _wrap(analyzer, name)
    session.analyzer = wrapped
    return auto


def arm(solver_cls: Any = None) -> bool:
    """Patch ``HarnessSolver._make_analyzer`` so every game gets a pilot.

    THIS IS THE SEAM, AND WHY IT IS THIS ONE
    ----------------------------------------
    ``_make_analyzer`` is called exactly once per game, is handed the ``game``
    object (so the pilot can be keyed correctly), and its return value becomes
    the session's ``analyzer`` (``solver.py:1383-1387``). One class-level patch
    therefore covers every game and every pass, with no notebook-side loop and
    no prompt tokens.

    ``HarnessSolver.analyzer_factory`` looks like the intended hook and is the
    wrong one: ``_make_analyzer`` returns ``analyzer_factory(game, index)``
    *instead of* building the ToolAgent, and the factory is never handed
    ``local_server`` - so using it would silently drop the per-server
    ``api_key``/``base_url``/``provider`` routing that multi-GPU runs depend on
    (``solver.py:1345-1366``). Wrapping the real method keeps all of it.
    """
    if solver_cls is None:
        try:
            from inference.framework.solver import HarnessSolver as solver_cls  # type: ignore
        except Exception:
            return False
    original = getattr(solver_cls, "_make_analyzer", None)
    if original is None or getattr(original, "_arc3x_armed", False):
        return False

    def _make_analyzer(self: Any, game: Any, index: int, local_server: Any = None) -> Any:
        analyzer = original(self, game, index, local_server)
        try:
            if _is_shim(analyzer):
                return analyzer
            wrapped, _auto = _wrap(analyzer, _game_id(game))
            return wrapped
        except Exception:
            # An unusable pilot must cost nothing. Stock analyzer, stock run.
            return analyzer

    _make_analyzer._arc3x_armed = True  # type: ignore[attr-defined]
    solver_cls._make_analyzer = _make_analyzer
    return True


def _is_shim(analyzer: Any) -> bool:
    return isinstance(analyzer, Autopilot) or getattr(analyzer, "_arc3x_shim", False)


def _wrap(analyzer: Any, name: str) -> tuple[Any, Autopilot]:
    """``(shim, wrapper)``. One fresh :class:`Pilot` per call.

    Fresh rather than reused across plays of the same game: ``Pilot._roll`` only
    clears the per-level sets on a level *change*, so a second pass starting back
    at level 1 would inherit "every frontier colour already tried, every cell
    already clicked" and do nothing at all. ``PILOTS`` keeps each one under a
    unique label so a finished run is still inspectable.
    """
    label = name if name not in PILOTS else f"{name}#{sum(k.startswith(name) for k in PILOTS)}"
    pilot = Pilot()
    PILOTS[label] = pilot
    auto = Autopilot(analyzer=analyzer, game=label, pilot=pilot)
    return _Shim(analyzer, auto), auto


class _Shim:
    """Presents ``analyze`` while delegating everything else to the real analyzer.

    ``play`` reads token counters straight off the analyzer object
    (``_analyzer_reported_tokens``, ``solver.py:86-92``, via ``hasattr``), so a
    bare function or a partial stand-in would zero the token accounting that the
    run's own ``solver_note`` reports.
    """

    _arc3x_shim = True

    def __init__(self, inner: Any, call: Autopilot) -> None:
        self._inner = inner
        self._call = call

    def analyze(self, *a: Any, **k: Any) -> Any:
        return self._call(*a, **k)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)
