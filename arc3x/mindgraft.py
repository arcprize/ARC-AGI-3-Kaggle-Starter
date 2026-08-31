"""The mind: a forward model learned from nothing but what the agent already saw.

WHAT THIS IS
------------
Sam's ask was "a mind where we think and move in our mind, and have the game inside
the mind itself, figuring out that this game is this game and playing it in the mind
and then doing it for real". This module is the mind proper: it learns a *predictive*
model of the game from observed play, checks that model against reality, and only
lets a plan out of the mind once the model has earned it.

The constraint that shapes everything here: **the 110 scored games are remote.** There
is no engine to deepcopy, no twin, no rollback. So the model cannot be searched into
existence - it has to be *induced* from the only thing the agent gets for free, which
is the transition history the framework already keeps:

    load_runtime_state(state_path) -> (current_frame, [HistoryEntry(action, frame), ...])

Consecutive entries are exactly ``(grid_before, action, grid_after)``. Reading them
costs zero actions, and the framework has written them since the first turn. That is
the corpus. ``inference/framework/solver.py`` builds those grids with
``_grid_from_state`` = ``state.frame.data`` - the raw engine grid, unrendered - so a
local twin generates byte-identical training data and this module can be measured
offline against the 25 dev games while running unchanged against the gateway.

WHAT IS ALREADY BUILT, AND WHY THIS IS A COMPOSITION
----------------------------------------------------
``arc3x/mind.py``'s ``Mechanics`` already does the *induction*: fold a transition in
with ``observe(action, before, after)``, take consensus with ``settle()``, and read
off ``moves`` (button -> learned delta), ``walk_mask`` (where the sprite may stand)
and ``plan`` (shortest button route by breadth-first search over the learned deltas -
imagination, with no actions spent). ``arc3x/percept.py`` is pure numpy over a 64x64
int array and, by its own docstring, "never touches a game object, so it works
identically against a local twin and against the gateway".

Both were written for a searching agent that scored 0.142 and is a dead end. Neither
of them ever needed the engine. What was missing is the piece below.

WHAT IS NEW HERE
----------------
1. **A codec** between the framework's action strings (``UP``, ``MOUSE(row=3,
   col=9)``) and ``Mechanics``' integer button ids, plus the ``step_env`` payload
   shape (``{"action": name, "row": r, "col": c}``) taken from the caller's own
   ``valid_actions`` spelling rather than a hardcoded name.
2. **``Mind.predict`` - an actual next-frame prediction.** ``Mechanics`` knew which
   way a button moves the sprite; it could not say what the board would *look* like
   afterwards. Prediction is what makes the model falsifiable, and falsifiability is
   the whole point: a plan is only worth spending real actions on if the model that
   produced it has been caught being right.
3. **``backtest`` - the honesty gate.** Fold the first part of a game's observed
   history into a fresh model, then predict the *held-out* tail it has never seen.
   That number, per game, decides whether the mind is allowed to act at all.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
No policy, no scoring, no LLM. Nothing here spends an action; ``predict`` and
``plan`` run in imagination at ~1 ms. The eleven-experiment ablation in
``docs/EXPERIMENT_LOG.md`` went 0-for-11 by shipping mechanisms that were plausible
and unmeasured, so the layer that lets this act on the real board is written only
after the numbers below say it should be.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from arc3x.mind import Mechanics

# -- 1. the action codec -----------------------------------------------------
# Authoritative, read from inference/agent/action_names.py: the engine speaks
# ACTION1..ACTION6 and the model-facing transcript speaks UP/DOWN/LEFT/RIGHT/
# SPACE/MOUSE. HistoryEntry.action carries the *display* form, and clicks arrive
# as "MOUSE(row=12, col=30)" (taaf_grafts/recovery.py:152-156 collapses exactly
# that shape). Mechanics keys everything by the engine's integer id.
#
# Note what is NOT claimed here: that ACTION1 means "up". The names are the
# vendor's convention, and the convention holds in 90-100% of the dev games but
# is not ground truth - Mechanics._convention seeds it as a prior that observed
# evidence overrides. The codec maps names to ids and nothing more.
DISPLAY_TO_AID: dict[str, int] = {
    "UP": 1,
    "DOWN": 2,
    "LEFT": 3,
    "RIGHT": 4,
    "SPACE": 5,
    "MOUSE": 6,
    "ACTION1": 1,
    "ACTION2": 2,
    "ACTION3": 3,
    "ACTION4": 4,
    "ACTION5": 5,
    "ACTION6": 6,
}

CLICK_AID = 6

# The inverse map, for the two directions the codec is used in: emitting a display
# string for a synthesized press, and labelling a button in a diagnostic. The
# framework's own display form is the model-facing one.
AID_LABEL: dict[int, str] = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "SPACE", 6: "MOUSE"}

# "MOUSE(row=12, col=30)" and the ACTION6(...) spelling recovery.py emits.
_CLICK_RE = re.compile(r"row\s*=\s*(-?\d+)\s*,\s*col\s*=\s*(-?\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class Press:
    """One button press, in the only two vocabularies that matter.

    ``aid`` is what ``Mechanics`` learns about; ``arguments`` is what ``step_env``
    executes. ``name`` is carried verbatim from the caller's ``valid_actions`` so a
    gateway that spells a button ``ACTION1`` is never handed ``UP``.
    """

    aid: int
    name: str = ""
    row: int = -1
    col: int = -1

    @property
    def is_click(self) -> bool:
        return self.aid == CLICK_AID

    def arguments(self, valid: Sequence[str] | None = None) -> dict[str, Any]:
        """The ``step_env`` payload, spelled the way this game spells it."""
        name = resolve_name(self.aid, valid) or self.name
        payload: dict[str, Any] = {"action": name}
        if self.is_click and self.row >= 0 and self.col >= 0:
            payload["row"] = int(self.row)
            payload["col"] = int(self.col)
        return payload

    def __repr__(self) -> str:
        if self.is_click:
            return f"A{self.aid}({self.row},{self.col})"
        return f"A{self.aid}"


def parse_press(display: str | None) -> Press | None:
    """Framework action display -> ``Press``, or ``None`` if it is not a button.

    ``None`` covers the three cases the model must not learn from as ordinary
    transitions: the empty seed entry the solver writes before the first action
    (``solver.py:201-205``), ``RESET`` (which teleports the sprite rather than
    moving it), and any name this codec does not recognise. Returning ``None``
    rather than guessing is what keeps one unknown label from poisoning the votes.
    """
    raw = str(display or "").strip()
    if not raw:
        return None
    head = raw.split("(", 1)[0].strip().upper()
    aid = DISPLAY_TO_AID.get(head)
    if aid is None:
        return None
    if aid != CLICK_AID:
        return Press(aid=aid, name=head)
    match = _CLICK_RE.search(raw)
    if match is None:
        # A click whose coordinates we cannot read is useless for learning: the
        # same button id at two different points is two different experiments.
        return None
    return Press(aid=aid, name=head, row=int(match.group(1)), col=int(match.group(2)))


def resolve_name(aid: int, valid: Sequence[str] | None) -> str | None:
    """The spelling *this game* uses for button ``aid``, taken from valid_actions.

    ``build_probe_plan`` accepts both ``ACTION6`` and ``MOUSE`` because the
    framework has been seen emitting either, so the safe move is to never invent a
    name: look it up in what the caller just handed us.
    """
    for raw in valid or ():
        name = str(raw or "").strip()
        if not name:
            continue
        if DISPLAY_TO_AID.get(name.upper()) == aid:
            return name
    return None


def is_reset(display: str | None) -> bool:
    return str(display or "").strip().upper() == "RESET"


# -- 2. transitions ----------------------------------------------------------


@dataclass(frozen=True)
class Transition:
    """One ``(before, press, after)`` triple, plus the two flags that change
    what it teaches: a level change means the board was rebuilt, and a reset
    means the sprite teleported."""

    press: Press
    before: np.ndarray
    after: np.ndarray
    level_before: int
    level_after: int

    @property
    def level_up(self) -> bool:
        return self.level_after > self.level_before

    @property
    def changed(self) -> bool:
        return bool((self.before != self.after).any())


def _grid(entry: Any) -> np.ndarray | None:
    """``HistoryEntry.frame.grid`` (nested int tuples) -> 2-D int array.

    Ragged rows are possible in principle - ``Frame.shape`` maxes over row
    lengths rather than assuming a rectangle - and numpy would build a 1-D object
    array from them, which every downstream comparison would silently mis-handle.
    Reject that instead.
    """
    frame = getattr(entry, "frame", None)
    grid = getattr(frame, "grid", None)
    if not grid:
        return None
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        return None
    arr = np.asarray(grid, dtype=np.int16)
    return arr if arr.ndim == 2 and arr.size else None


def transitions(entries: Sequence[Any]) -> list[Transition]:
    """Consecutive history entries -> the transitions worth learning from.

    ``HistoryEntry[i].action`` is the action that *produced* ``HistoryEntry[i]``,
    so the triple is ``(entries[i-1].frame, entries[i].action, entries[i].frame)``.
    Entries whose action is not a button, or whose grids differ in shape, are
    skipped rather than guessed at.
    """
    out: list[Transition] = []
    for i in range(1, len(entries)):
        prev, cur = entries[i - 1], entries[i]
        if is_reset(getattr(cur, "action", None)):
            continue
        press = parse_press(getattr(cur, "action", None))
        if press is None:
            continue
        before, after = _grid(prev), _grid(cur)
        if before is None or after is None or before.shape != after.shape:
            continue
        out.append(
            Transition(
                press=press,
                before=before,
                after=after,
                level_before=int(getattr(prev.frame, "level", 1)),
                level_after=int(getattr(cur.frame, "level", 1)),
            )
        )
    return out


# -- 3. the mind -------------------------------------------------------------


@dataclass
class Prediction:
    """What the mind thinks the next frame will be, and why.

    ``grid`` is ``None`` when the mind declines to guess - no learned delta for
    this button, or the sprite could not be located. Declining is a first-class
    answer: a model that only speaks when it has grounds is the one whose
    accuracy means something.
    """

    grid: np.ndarray | None
    reason: str
    moved: bool = False
    to: tuple[int, int] | None = None

    @property
    def spoke(self) -> bool:
        return self.grid is not None


@dataclass
class Mind:
    """A learned game, playable in imagination.

    ``absorb`` is incremental on purpose. The framework hands over the *whole*
    history every turn, and re-folding 3,000 transitions per turn would put a
    numpy pass over every past frame between the agent and its next action. The
    cursor makes each turn cost only what is new.
    """

    mech: Mechanics = field(default_factory=Mechanics)
    seen: int = 0
    folded: int = 0
    level: int = 1
    # Whether the geometry pass has been run since the avatar was established.
    # See `absorb`: knowing who you are is a precondition for learning what
    # stops you, so the second pass has to wait for the first to conclude.
    grounded: bool = False

    # -- learning ----------------------------------------------------------
    def absorb(self, entries: Sequence[Any], *, settle: bool = True) -> int:
        """Fold every transition not yet folded. Returns how many were new.

        Two passes, in the order they are learnable. ``observe`` votes on who the
        avatar is and how each button displaces it; ``settle`` decides. Only then
        can ``replay_geometry`` go back over the same history and ask what
        refused to let the sprite through - so the moment the avatar first becomes
        known, the transitions already folded get re-read for their geometry.
        That re-read happens **once** per game, not once per turn: afterwards
        ``observe`` tracks the sprite correctly as each new transition arrives.
        """
        trs = transitions(entries)
        fresh = trs[self.seen :]
        self.seen = len(trs)
        for tr in fresh:
            # A level transition is a scene cut: the after-frame is a new board,
            # not the result of translating the old avatar. Feeding that pair to
            # Mechanics would create false movement votes and false wall/ground
            # evidence. The level-up action is still consumed by the history
            # cursor, but only same-level transitions train the predictive model.
            if not tr.level_up:
                self.mech.observe(tr.press.aid, tr.before, tr.after)
                self.folded += 1
            self.level = tr.level_after
        if fresh and settle:
            self.mech.settle()
        if not self.grounded and self.mech.avatar >= 0 and trs:
            self.mech.replay_geometry(
                (t.press.aid, t.before, t.after) for t in trs if not t.level_up
            )
            self.grounded = True
        return len(fresh)

    # -- imagination -------------------------------------------------------
    def predict(self, grid: np.ndarray, aid: int) -> Prediction:
        """The next frame, according to the model. No actions, no engine.

        The model this composes is deliberately the simplest one that can be
        wrong in an informative way: *the sprite translates by the button's
        learned delta unless the destination is not standable, in which case
        nothing happens at all.* Everything it gets wrong is a finding -
        a board where the sprite leaves a trail behind it, or pushes a block,
        or where a counter ticks, all show up as a specific divergence rather
        than as a vague failure. That is the learning signal.
        """
        delta = self.mech.moves.get(aid)
        if delta is None:
            # Includes every use/select button on purpose: `moves` drops buttons
            # that change the board without translating the sprite, because a
            # route cannot be made of them and pretending otherwise would put a
            # confident wrong prediction where an honest silence belongs.
            return Prediction(None, "no learned delta")
        box = self.mech.locate(grid, hint=self.mech.pos)
        if box is None:
            return Prediction(None, "sprite not located")
        top, left, h, w = box
        dy, dx = delta
        nt, nl = top + dy, left + dx
        H, W = grid.shape
        walk = self.mech.walk_mask(grid)
        if not Mechanics._free(walk, nt, nl, h, w, H, W):
            # A refusal is a real prediction, and on a grid game it is the most
            # common one: most buttons, most of the time, are into a wall.
            return Prediction(grid.copy(), "blocked", moved=False, to=(top, left))
        foot = self.mech.footprint(grid, box)
        if not foot.any():
            return Prediction(None, "empty footprint")
        out = grid.copy()
        pixels = grid[top : top + h, left : left + w][foot]
        # Basic slicing gives a view, so both writes land in `out`. Order matters
        # when the boxes overlap: read the sprite, erase, then redraw.
        out[top : top + h, left : left + w][foot] = self.mech.background
        out[nt : nt + h, nl : nl + w][foot] = pixels
        return Prediction(out, "moved", moved=True, to=(nt, nl))

    def route(self, grid: np.ndarray, targets: list[tuple[int, int]]) -> list[int]:
        """Shortest learned-model route to any target cell. Free; spends nothing."""
        return self.mech.plan(grid, targets)

    def summary(self) -> str:
        return self.mech.summary()


# -- 4. the honesty gate -----------------------------------------------------


@dataclass
class Report:
    """How well a model induced from the *first* part of a game predicts the
    part it never saw. Every rate is over ``spoke``, not over ``n``, because a
    model that declines is not a model that is wrong."""

    game: str = ""
    n: int = 0            # held-out transitions
    spoke: int = 0        # ...on which the model was willing to predict
    exact: int = 0        # ...and got the whole grid right
    placed: int = 0       # ...and got the sprite's position right
    move_call: int = 0    # ...and got moved-vs-blocked right
    buttons: int = 0      # learned move deltas at the end of training
    assumed: int = 0      # ...of which came from the convention prior, not evidence

    def rate(self, field_name: str) -> float:
        got = int(getattr(self, field_name))
        return got / self.spoke if self.spoke else 0.0

    @property
    def coverage(self) -> float:
        return self.spoke / self.n if self.n else 0.0

    def line(self) -> str:
        return (
            f"{self.game:>6}  n={self.n:>4}  spoke={self.coverage:5.0%}  "
            f"exact={self.rate('exact'):5.0%}  place={self.rate('placed'):5.0%}  "
            f"movecall={self.rate('move_call'):5.0%}  "
            f"buttons={self.buttons}({self.assumed} assumed)"
        )


def backtest(entries: Sequence[Any], *, holdout: float = 0.3, game: str = "") -> Report:
    """Train on the first ``1-holdout`` of observed play, predict the rest.

    This is the number that decides whether the mind may act. It is measured the
    only honest way available: the model is built from transitions it has seen and
    scored on transitions it has not, with no engine consulted at any point, so
    the same procedure runs identically on a game nobody has ever played.

    Three rates, because they gate different things:

    * ``exact`` - the whole 64x64 grid is right. Strict, and a HUD clock alone is
      enough to hold it at zero forever while the model is otherwise perfect.
    * ``place`` - the sprite ends where the model said. This is what a route
      needs to be correct, so this is the planning gate.
    * ``movecall`` - moved-versus-blocked was called right. This is what a route
      needs to not walk into a wall, and it is the cheapest thing to be sure of.
    """
    trs = transitions(entries)
    report = Report(game=game, n=0)
    if len(trs) < 8:
        return report
    cut = max(4, int(len(trs) * (1.0 - holdout)))
    train, test = trs[:cut], trs[cut:]
    if not test:
        return report

    mind = Mind()
    for tr in train:
        if not tr.level_up:
            mind.mech.observe(tr.press.aid, tr.before, tr.after)
    mind.mech.settle()
    # Second pass: what stops the sprite, now that there is a sprite to stop.
    # Without this the model has no walls at all and every held-out miss is
    # "walked through one in imagination" - see Mechanics._track.
    mind.mech.replay_geometry(
        (t.press.aid, t.before, t.after) for t in train if not t.level_up
    )
    report.buttons = len(mind.mech.moves)
    report.assumed = len(mind.mech.assumed & set(mind.mech.moves))
    report.n = len(test)

    # `pos` is deliberately *not* reset here. The held-out tail is the frames
    # immediately after the training ones in a single run, so a live agent would
    # know where the sprite is, and `replay_geometry` leaves `pos` exactly there.
    # Clearing it would measure an agent that forgot, and would cost the first
    # `locate` its hint - with no hint `locate` ranks candidates by size rather
    # than proximity, which on a single-colour sprite can lock onto the largest
    # same-coloured clump on the board and mistrack every frame after it.
    #
    # Measured, and worth knowing: resetting scores 62% against 60%, but swings
    # individual games by 50 points either way (sc25 100 vs 33, ar25 1 vs 36).
    # Tracking, not delta learning, is now the fragile link in this model.
    for tr in test:
        # Keep the sprite tracked through the held-out tail exactly as a live run
        # would: `where` is how the real agent follows itself frame to frame, and
        # scoring without it would measure a blindfolded model.
        box_before = mind.mech.where(tr.before)
        pred = mind.predict(tr.before, tr.press.aid)
        if not pred.spoke:
            continue
        report.spoke += 1
        if np.array_equal(pred.grid, tr.after):
            report.exact += 1
        box = mind.mech.locate(tr.after, hint=mind.mech.pos)
        if box is not None and pred.to is not None and box[:2] == pred.to:
            report.placed += 1
        # moved-versus-blocked, judged against the *sprite*, not the frame.
        # Comparing to `(before != after).any()` scored ~84% and meant nothing:
        # nearly every board has a HUD that ticks on every action, so "the frame
        # changed" is true even when the press was refused.
        if box is not None and box_before is not None:
            if pred.moved == (box[:2] != box_before[:2]):
                report.move_call += 1
    return report


def aggregate(reports: Iterable[Report]) -> str:
    """One table plus a total line - the shape every arc3x measurement reports in,
    always with coverage next to the mean so that games the mind stayed silent on
    cannot hide inside a good-looking average."""
    rows = [r for r in reports if r.n]
    lines = [r.line() for r in rows]
    if not rows:
        return "no game produced enough transitions to backtest"
    spoke = sum(r.spoke for r in rows)
    total = sum(r.n for r in rows)
    lines.append("-" * 78)
    lines.append(
        f"{len(rows):>6}  n={total:>4}  spoke={spoke / total if total else 0:5.0%}  "
        f"exact={sum(r.exact for r in rows) / spoke if spoke else 0:5.0%}  "
        f"place={sum(r.placed for r in rows) / spoke if spoke else 0:5.0%}  "
        f"movecall={sum(r.move_call for r in rows) / spoke if spoke else 0:5.0%}  "
        f"games with a model={sum(1 for r in rows if r.buttons)}"
    )
    return "\n".join(lines)
