"""What does a click do? Decided by elimination rather than assumption.

WHY THIS EXISTS
---------------
``Mechanics`` learns ``button -> fixed (dy, dx) translation of one sprite``. That
model is structurally incapable of describing a click, because a click's effect
depends on *where* you clicked, and ``Mechanics.observe`` is handed only an action
id. The consequence is measured: the mind is silent on the 7 dev games with no
avatar, and ``r11l`` - where a click moves the sprite *to the clicked cell* -
backtests at 0% placement because there is no per-button delta to learn.

The harvested run of 2026-08-23 made this the priority rather than a curiosity.
Of the four games played, the best scorer was ``tn36`` at 2.67, and ``tn36``
offers ``MOUSE`` and nothing else on all 118 of its turns. It cleared level 0 in
37 actions against a baseline of 32, then spent 358 actions failing level 1 -
another click-only level. So the single richest game in the sample is one this
model could not say one word about.

WHAT IS GENERAL HERE
--------------------
Six predicates, none naming a game. A click either

    TELEPORT  moved something to where you clicked
    PAINT     set the clicked cell to a colour that is constant across clicks,
              i.e. there is a currently-selected colour and you are drawing
    TOGGLE    changed the clicked cell to something that depends on what was
              already there
    WIDGET    changed the board somewhere *else*, so the clicked region is a
              button rather than a place
    SELECT    did something that depends on a previous click (two-stage
              select-then-act)
    INERT     did nothing

and the model's job is to find out which, from evidence, at run time. That is the
same discipline ``Mechanics.settle`` already applies to movement: accumulate
votes, require a minimum, take the consensus, and stay silent when there is not
enough to say.

THE HUD PROBLEM, WHICH BITES HARDER HERE
----------------------------------------
``INERT`` cannot be decided with ``(before != after).any()``. Nearly every board
carries a counter that ticks on every action - ``s5i5`` changes 1 pixel on 175 of
175 actions, ``ls20`` 2 pixels in rows 61-62 - so by that test no click is ever
inert and no click's effect is ever localisable. Worse, a ticking counter changes
a *different* pixel each tick, so looking for pixels that change every time does
not find it.

So ``learn_volatile`` runs first over **every** transition, click or not, and
marks any cell that changes on at least ``VOLATILE_FRAC`` of them. Every judgement
below is made on ``diff & ~volatile``. This is the same defect that made the
movement model's ``movecall`` meaningless at 84% until it was judged against
located displacement instead of "did the frame change".
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from arc3x.percept import background, mask_component, moved_objects

# A cell that changes on at least this fraction of all transitions is chrome, not
# content. Deliberately low: a counter that ticks every action sits near 1.0, and
# a genuine game object that a random policy happens to disturb half the time
# would need the board to be almost entirely mobile to reach 0.5.
VOLATILE_FRAC = 0.5

# "At the click" - clicks land on a cell but a sprite is several cells wide, and a
# lattice game snaps to a cell boundary, so an exact-pixel test is too strict.
NEAR = 2

# Beyond this, a change is somewhere else entirely and the click acted remotely.
FAR = 5

# Nothing is claimed from fewer observations than this, matching the spirit of
# ``Mechanics.settle(min_votes=2)``: one coincidence is not a mechanic.
MIN_CLICKS = 4

# Below this correlation between where you clicked and where the board changed,
# the effect does not follow the click and the click position is decoration. The
# first probe over 25 games found 10 of 17 click games verdicted WIDGET at 89-100%
# consistency, which does not distinguish "this is a button over there" from "this
# game ignores the coordinate entirely" - and those two need opposite searches.
FOLLOW_CORR = 0.25

# A colour is only the paint colour if it is written on this share of the clicks
# that did anything. Added after the first probe verdicted PAINT on bp35 and su15
# with 6% and 7% support, where ``predict`` then scored 0% and 1% exact: the
# dominant-colour test passed on a handful of incidental cells while 93% of clicks
# never touched the clicked cell at all.
PAINT_SHARE = 0.5

# How exactly a teleport has to repeat before it is allowed to predict a frame.
TELE_SHARE = 0.7

HYPOTHESES = ("teleport", "paint", "toggle", "widget", "select", "step", "inert")


def _extent(size: int) -> int:
    """Half-width of an object's footprint, from the only extent a ``Move`` has.

    ``percept.Move`` records ``size`` (pixel count) and the anchor ``(top, left)``
    but not height/width, so the footprint is approximated as the square with that
    area. This over-estimates a long thin sprite along one axis and under-estimates
    it along the other; both errors are bounded by ``NEAR`` in practice and the
    alternative - re-segmenting the frame here - would duplicate ``moved_objects``.
    """
    return max(1, int(np.ceil(np.sqrt(max(1, size))))) // 2


def _near_anchor(top: int, left: int, size: int, r: int, c: int, slack: int) -> bool:
    """Is (r, c) inside an object anchored at (top, left), give or take ``slack``?"""
    ext = _extent(size) + slack
    return (top - slack) <= r <= (top + 2 * ext) and (left - slack) <= c <= (left + 2 * ext)


@dataclass
class ClickModel:
    """Click semantics, induced from ``(before, after, row, col)`` triples."""

    # -- pass 1: what is chrome -------------------------------------------
    seen: int = 0
    churn: np.ndarray | None = None  # per-cell count of changes over all transitions

    # -- pass 2: what a click does ----------------------------------------
    n: int = 0
    support: Counter = field(default_factory=Counter)
    # The colour the clicked cell BECAME. If one colour dominates across clicks at
    # many different cells, that is a selected colour and the game is a paint game.
    fills: Counter = field(default_factory=Counter)
    # Distinct cells that were painted, so a single cell clicked 40 times cannot
    # masquerade as a constant fill colour.
    fill_cells: set = field(default_factory=set)
    # Colour that was under the click -> whether the click did anything. This is
    # the click equivalent of ``passable``/``blocking``: it says which cells are
    # worth clicking at all.
    live: Counter = field(default_factory=Counter)
    dead: Counter = field(default_factory=Counter)
    # (row, col) -> effect signatures observed there. More than one signature at
    # one cell means the effect is not a function of position alone, which is what
    # SELECT looks like from the outside.
    at_cell: dict = field(default_factory=dict)
    # Where the sprite went, when a click moved something to the click.
    teleports: int = 0
    # (click_row, click_col, change_centre_row, change_centre_col) for every click
    # that did something. Feeds the only test that separates a remote button from a
    # coordinate the game throws away.
    follow: list = field(default_factory=list)
    # A teleport that always lands the same shape at the same offset from the click
    # is fully predictable, so the shape and the offset are kept rather than just
    # counted: (colour, mask bytes, mask height, mask width) -> hits, and
    # (anchor - click) -> hits.
    tele_shape: Counter = field(default_factory=Counter)
    tele_offset: Counter = field(default_factory=Counter)
    # -- honesty ledger -------------------------------------------------
    # A semantic label is not yet a model.  These count whether a prediction
    # formed *before* a later click actually described the pixels it claimed
    # would change.  The pilot uses this as the click analogue of Dream's held-
    # out movement gate.
    prediction_hits: int = 0
    prediction_misses: int = 0
    prediction_abstains: int = 0

    # -- pass 1 -----------------------------------------------------------

    def learn_volatile(self, steps: Iterable[tuple[np.ndarray, np.ndarray]]) -> int:
        """Accumulate per-cell change frequency over ALL transitions.

        Runs on every transition, not only clicks: a HUD ticks on every action
        regardless of which button caused it, so restricting this to clicks would
        both undercount the chrome and waste the evidence.
        """
        for before, after in steps:
            if before.shape != after.shape:
                continue
            if self.churn is None:
                self.churn = np.zeros(before.shape, dtype=np.int32)
            if self.churn.shape != before.shape:
                continue
            self.churn += (before != after)
            self.seen += 1
        return self.seen

    @property
    def volatile(self) -> np.ndarray | None:
        """Cells that are chrome. ``None`` until pass 1 has run."""
        if self.churn is None or self.seen <= 0:
            return None
        return self.churn >= max(2, int(VOLATILE_FRAC * self.seen))

    def content(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """The changes that are not chrome."""
        diff = before != after
        vol = self.volatile
        if vol is not None and vol.shape == diff.shape:
            diff = diff & ~vol
        return diff

    # -- pass 2 -----------------------------------------------------------

    def observe(self, before: np.ndarray, after: np.ndarray, r: int, c: int) -> str:
        """Judge one click against all six predicates. Returns the tags it earned.

        A click may support more than one hypothesis - moving the sprite onto the
        clicked cell is both TELEPORT and a change at the click - so support is
        counted per hypothesis and the verdict is taken at the end. Forcing a
        single label per observation is what would turn an ambiguity into a wrong
        confident answer.
        """
        if before.shape != after.shape:
            return ""
        H, W = before.shape
        if not (0 <= r < H and 0 <= c < W):
            return ""
        self.n += 1
        diff = self.content(before, after)
        n_changed = int(diff.sum())
        under = int(before[r, c])
        tags: list[str] = []

        if n_changed == 0:
            self.support["inert"] += 1
            self.dead[under] += 1
            self._record(r, c, ("inert",))
            return "inert"

        self.live[under] += 1

        # Did the clicked cell itself change? Distinguishing PAINT from TOGGLE is
        # an aggregate question - whether the new colour is constant - so both are
        # credited here and separated in ``verdict``.
        if bool(diff[r, c]):
            became = int(after[r, c])
            self.fills[became] += 1
            self.fill_cells.add((r, c))
            self.support["paint"] += 1
            self.support["toggle"] += 1
            tags += ["paint", "toggle"]

        # Did something MOVE to the click? Checked against the located object
        # rather than the raw pixels, because a sprite is several cells wide and
        # its footprint is what has to cover the clicked cell.
        mv, van, app = moved_objects(before, after)
        landed = False
        for m in mv:
            if not m.moved:
                continue
            if _near_anchor(m.top + m.dy, m.left + m.dx, m.size, r, c, NEAR):
                self._record_teleport(after, m.top + m.dy, m.left + m.dx, m.color, r, c)
                landed = True
                break
        if not landed:
            for m in app:
                if _near_anchor(m.top, m.left, m.size, r, c, NEAR):
                    self._record_teleport(after, m.top, m.left, m.color, r, c)
                    landed = True
                    break
        if landed:
            self.support["teleport"] += 1
            self.teleports += 1
            tags.append("teleport")

        # Did it act somewhere else entirely? Measured as the nearest non-chrome
        # change: if even the closest one is far away, the click was a button.
        ys, xs = np.nonzero(diff)
        if ys.size:
            near = int(np.min(np.abs(ys - r) + np.abs(xs - c)))
            if near > FAR:
                self.support["widget"] += 1
                tags.append("widget")
            # Where the board actually reacted, kept for the position-dependence
            # test. The centre of the changed region rather than the nearest cell:
            # a click that is ignored still produces a change centred wherever the
            # game's activity happens to be, and it is the *correlation* with the
            # click, not the distance, that tells the two apart.
            self.follow.append((r, c, float(ys.mean()), float(xs.mean())))

        self._record(r, c, self._signature(before, after, diff))
        return ",".join(tags)

    def _record_teleport(
        self, after: np.ndarray, top: int, left: int, color: int, r: int, c: int
    ) -> None:
        """Keep the landed sprite's exact footprint and its offset from the click.

        Counting teleports says only that the genre is right. To *place* a sprite -
        the thing a route needs - the model has to know which pixels move and where
        they end up relative to the click, so the mask is lifted straight out of the
        frame it landed in rather than approximated from ``size``.
        """
        H, W = after.shape
        if not (0 <= top < H and 0 <= left < W):
            return
        if int(after[top, left]) != int(color):
            # The bbox corner is not always a member pixel of the sprite. Without a
            # member pixel there is nothing to flood from, so decline rather than
            # record a mask of the wrong thing.
            return
        sub, sy, sx, _ = mask_component(after == color, top, left)
        self.tele_shape[(int(color), sub.tobytes(), int(sub.shape[0]), int(sub.shape[1]))] += 1
        self.tele_offset[(int(sy - r), int(sx - c))] += 1

    def _signature(self, before: np.ndarray, after: np.ndarray, diff: np.ndarray) -> tuple:
        """A compact fingerprint of what a click did, for the SELECT test."""
        if not diff.any():
            return ("inert",)
        return (int(diff.sum()), tuple(sorted({int(v) for v in after[diff]})))

    def _record(self, r: int, c: int, sig: tuple) -> None:
        self.at_cell.setdefault((r, c), Counter())[sig] += 1

    # -- what was learned -------------------------------------------------

    @property
    def follows(self) -> tuple[float, float, int]:
        """(|corr(row)|, |corr(col)|, n) between where you clicked and what moved.

        This is the statistic the first probe was missing. Ten of seventeen click
        games came back WIDGET at 89-100% consistency, which reads as a confident
        answer but is really two answers wearing one label: either there is a button
        elsewhere on the board, or the game does not read the coordinate at all and
        the change is simply wherever the action already was. A button-per-place
        game needs a search over 4096 positions; a coordinate-ignoring game needs a
        search over exactly one. Correlation separates them and distance cannot,
        because both put the change far from a randomly chosen cell.
        """
        n = len(self.follow)
        if n < MIN_CLICKS:
            return (0.0, 0.0, n)
        arr = np.asarray(self.follow, dtype=np.float64)
        out = []
        for click_col, change_col in ((0, 2), (1, 3)):
            a, b = arr[:, click_col], arr[:, change_col]
            if a.std() < 1e-9 or b.std() < 1e-9:
                out.append(0.0)
                continue
            out.append(abs(float(np.corrcoef(a, b)[0, 1])))
        return (out[0], out[1], n)

    @property
    def tele_rule(self) -> tuple[int, np.ndarray, tuple[int, int]] | None:
        """(colour, footprint mask, (dy, dx) from click to mask corner), or ``None``.

        Only returned when one shape and one offset both dominate, because a
        prediction assembled from a modal shape and an unrelated modal offset would
        be confidently wrong rather than usefully silent.
        """
        if not self.tele_shape or not self.tele_offset:
            return None
        (color, raw, h, w), shits = self.tele_shape.most_common(1)[0]
        offset, ohits = self.tele_offset.most_common(1)[0]
        total = sum(self.tele_shape.values())
        if total < MIN_CLICKS:
            return None
        if shits < TELE_SHARE * total or ohits < TELE_SHARE * sum(self.tele_offset.values()):
            return None
        mask = np.frombuffer(raw, dtype=bool).reshape(h, w)
        return (int(color), mask, (int(offset[0]), int(offset[1])))

    @property
    def repeats(self) -> tuple[int, int]:
        """(cells clicked more than once, of those, cells with >1 distinct effect).

        The SELECT signal. If the same cell reliably does the same thing, the
        effect is a function of position and a one-stage model is enough. If it
        does different things on different visits, either state matters - a
        selection was made earlier - or the board simply moved on underneath,
        which is why this is reported rather than trusted.
        """
        multi = [s for s in self.at_cell.values() if sum(s.values()) > 1]
        return len(multi), sum(1 for s in multi if len(s) > 1)

    @property
    def fill_color(self) -> int:
        """The dominant colour clicks write, or -1 if there is no dominant one."""
        if len(self.fill_cells) < MIN_CLICKS or not self.fills:
            return -1
        color, hits = self.fills.most_common(1)[0]
        return int(color) if hits >= 0.7 * sum(self.fills.values()) else -1

    def verdict(self) -> tuple[str, float]:
        """The winning hypothesis and the fraction of clicks supporting it.

        Order matters, and it is ordered by how *specific* each claim is rather
        than by how much support it has: TELEPORT and PAINT say where the effect
        lands and are therefore predictive, WIDGET and TOGGLE only say that
        something happened, and INERT is the null. A weakly-supported specific
        claim is worth more than a strongly-supported vague one, so the specific
        ones are tested first against a threshold.
        """
        if self.n < MIN_CLICKS:
            return ("unknown", 0.0)
        inert = self.support["inert"] / self.n
        if inert >= 0.9:
            return ("inert", inert)
        # Judge the active hypotheses against the clicks that DID something: a
        # game where 90% of the board is scenery would otherwise dilute a perfect
        # teleport rule down below every threshold.
        active = self.n - self.support["inert"]
        if active <= 0:
            return ("inert", inert)
        tele = self.support["teleport"] / active
        if tele >= 0.6:
            return ("teleport", tele)
        paint = self.support["paint"] / active
        # Both tests, not just the colour one. The first probe showed why: a game
        # can have a perfectly dominant fill colour across four cells while 94% of
        # its clicks never touch the clicked cell, and calling that PAINT licenses
        # ``predict`` to return a wrong grid on every single click.
        if self.fill_color >= 0 and paint >= PAINT_SHARE:
            return ("paint", paint)
        fr, fc, fn = self.follows
        widget = self.support["widget"] / active
        toggle = self.support["toggle"] / active
        # Does the coordinate matter at all? Asked before TOGGLE and WIDGET because
        # it is the more useful answer when it is true: it collapses the click search
        # space from every cell to one, and no amount of "something changed over
        # there" support can establish that the something was caused by *where* the
        # click was.
        if fn >= MIN_CLICKS and max(fr, fc) < FOLLOW_CORR and widget >= 0.6:
            return ("step", 1.0 - max(fr, fc))
        if toggle >= 0.6:
            return ("toggle", toggle)
        if widget >= 0.6:
            return ("widget", widget)
        multi, varying = self.repeats
        if multi >= MIN_CLICKS and varying >= 0.5 * multi:
            return ("select", varying / max(1, multi))
        return ("mixed", max(tele, toggle, widget))

    # -- using it ---------------------------------------------------------

    def predict(self, grid: np.ndarray, r: int, c: int) -> np.ndarray | None:
        """The frame a click would produce, or ``None`` if the model cannot say.

        Only the two hypotheses that locate their own effect can predict at all.
        WIDGET knows a button exists but not what it does; TOGGLE knows the cell
        changes but not to what; SELECT needs state this model does not carry.
        Returning ``None`` there is the point - the same contract
        ``Mind.predict`` uses, where declining is not an error.
        """
        kind, _ = self.verdict()
        if kind == "inert":
            return grid.copy()
        if kind == "paint":
            color = self.fill_color
            if color < 0:
                return None
            H, W = grid.shape
            if not (0 <= r < H and 0 <= c < W):
                return None
            out = grid.copy()
            out[r, c] = color
            return out
        if kind == "teleport":
            return self._predict_teleport(grid, r, c)
        return None

    def grade(self, before: np.ndarray, predicted: np.ndarray | None, actual: np.ndarray) -> str:
        """Score a pre-click prediction against the effect it specifically claims.

        The comparison intentionally focuses on pixels the prediction says will
        change. A forward model of a click need not reproduce an independent
        animation or a countdown to be useful; it does need to be right about
        the paint stroke or teleported sprite that licensed an action. Volatile
        HUD pixels are excluded using evidence accumulated from earlier frames.
        """
        if (
            predicted is None
            or before.shape != actual.shape
            or predicted.shape != before.shape
        ):
            self.prediction_abstains += 1
            return "abstain"
        focus = predicted != before
        volatile = self.volatile
        if volatile is not None and volatile.shape == focus.shape:
            focus &= ~volatile
        if not focus.any():
            self.prediction_abstains += 1
            return "abstain"
        if bool(np.array_equal(predicted[focus], actual[focus])):
            self.prediction_hits += 1
            return "hit"
        self.prediction_misses += 1
        return "miss"

    @property
    def predictive(self) -> bool:
        """Whether click predictions have earned the right to guide v21 plans."""
        judged = self.prediction_hits + self.prediction_misses
        return judged >= MIN_CLICKS and self.prediction_hits >= 0.8 * judged

    @property
    def prediction_accuracy(self) -> float:
        judged = self.prediction_hits + self.prediction_misses
        return self.prediction_hits / judged if judged else 0.0

    def _predict_teleport(self, grid: np.ndarray, r: int, c: int) -> np.ndarray | None:
        """Lift the sprite out of where it is and stamp it at the click.

        Declines on every ambiguity: no settled rule, the sprite not found exactly
        once, a footprint that would fall off the board. A wrong frame here is worse
        than no frame, because ``place`` is the gate a route is allowed through.
        """
        rule = self.tele_rule
        if rule is None:
            return None
        color, mask, (dy, dx) = rule
        H, W = grid.shape
        top, left = r + dy, c + dx
        mh, mw = mask.shape
        if not (0 <= top and top + mh <= H and 0 <= left and left + mw <= W):
            return None

        here = grid == color
        found = []
        seen = np.zeros_like(here)
        for y in range(H):
            for x in range(W):
                if not here[y, x] or seen[y, x]:
                    continue
                sub, sy, sx, size = mask_component(here, y, x)
                seen[sy : sy + sub.shape[0], sx : sx + sub.shape[1]] |= sub
                if sub.shape == mask.shape and size == int(mask.sum()):
                    found.append((sub, sy, sx))
        if len(found) != 1:
            return None

        sub, sy, sx = found[0]
        out = grid.copy()
        bg = background(grid)
        region = out[sy : sy + sub.shape[0], sx : sx + sub.shape[1]]
        region[sub] = bg
        region = out[top : top + mh, left : left + mw]
        region[mask] = color
        return out

    @property
    def position_matters(self) -> bool:
        """Should a planner search click positions at all?

        ``False`` says the coordinate is decoration and one click is every click -
        which is the difference between a 4096-wide search and a 1-wide one, on the
        games that offer no other button.
        """
        kind, _ = self.verdict()
        if kind in ("step", "inert"):
            return False
        return True

    def clickable(self, grid: np.ndarray) -> list[int]:
        """Colours a click has been seen to do something to, best first.

        This is the click analogue of ``Mechanics.frontier_colors``: it converts
        "clicks matter here" into a shortlist of *where* to click, without ever
        naming a game. A colour that has been clicked repeatedly to no effect is
        excluded, which is the only thing that makes a click budget affordable -
        a 64x64 board is 4096 candidate clicks, and this cuts it to the colours
        that have ever responded.
        """
        present = {int(v) for v in np.unique(grid)}
        scored = []
        for color in present:
            hit, miss = self.live.get(color, 0), self.dead.get(color, 0)
            if hit <= 0:
                continue
            scored.append((hit / (hit + miss), hit, color))
        scored.sort(reverse=True)
        return [c for _, _, c in scored]

    def summary(self) -> str:
        kind, conf = self.verdict()
        multi, varying = self.repeats
        vol = self.volatile
        chrome = 0 if vol is None else int(vol.sum())
        fr, fc, fn = self.follows
        bits = [
            f"click={kind}({conf:.0%})",
            f"n={self.n}",
            f"inert={self.support['inert']}",
            f"tele={self.support['teleport']}",
            f"paint={self.support['paint']}",
            f"widget={self.support['widget']}",
            f"follow={max(fr, fc):.2f}",
            f"fill={self.fill_color}",
            f"cells={len(self.fill_cells)}",
            f"revisited={multi}/varying={varying}",
            f"chrome={chrome}px",
        ]
        return "  ".join(bits)
