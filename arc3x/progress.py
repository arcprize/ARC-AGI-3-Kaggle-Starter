"""What "getting closer to done" means, discovered instead of being told.

THE HOLE THIS FILLS
-------------------
The imagination in ``dream.py`` could only search toward one kind of goal: a
colour that vanishes when the avatar walks onto it. Measured consequence, on the
four dev games where the forward model is most accurate:

    dc22   move=100%(153)  collect=[]   thought=  0   levels=0
    ka59   move=100%(207)  collect=[]   thought=  0   levels=0
    m0r0   move= 92%(126)  collect=[]   thought=  0   levels=0
    sp80   move= 80%(195)  collect=[8]  thought= 61   levels=1

A *perfect* copy of the game completed nothing, because a route needs a
destination and there was none. The one game that identified a target is the one
game that planned, and the only one that finished a level. Prediction was never
the bottleneck; the objective was.

The obvious fix - learn the goal from what changed when ``levels_completed`` went
up - cannot bootstrap. It needs a completed level to learn what completes a
level, and levels are exactly what is not happening.

THE RULE
--------
So the objective has to be readable from ordinary play, before any win. The one
that generalises is:

    **progress is whatever ratchets.**

A quantity that moves one way and does not come back is the game being solved. A
quantity that oscillates is just the avatar wandering. That single test separates
them with no knowledge of the genre:

  * gems collected      - the gem colour's pixel count only ever falls
  * markers cleared     - same
  * a region painted    - the paint colour's count only ever rises
  * sokoban             - a crate parked on a target *hides* the target pixel,
                          so the target colour's count falls and stays fallen.
                          Sokoban therefore needs no sokoban-specific code.
  * walking around      - the floor colour falls and rises as the sprite covers
                          and uncovers it, in equal measure, so it is rejected

That last line is why the test is a *ratchet* and not merely "went down once".

WHAT WOULD FOOL IT, AND WHY IT DOES NOT
---------------------------------------
A draining step-counter HUD is a perfect monotone decrease, and treating it as
progress would make the agent burn its own remaining steps on purpose - the worst
possible failure. ``ls20`` has literally that widget, and this repo has been
bitten by HUD pixels before: they made every frame unique and silently killed
Go-Explore. So counting happens only over pixels outside the volatility HUD mask,
tracking starts only once that mask has enough observations to be meaningful, and
the history is discarded whenever the mask changes shape.

**The mask alone is not enough, and we found that out the hard way.** Volatility
asks "does this pixel change often". A bar that loses one pixel per action changes
each individual pixel on about 1/154 of frames, so it never crosses the threshold:
cd82's mask came out completely empty, its 154-pixel bar counted as playfield, and
the agent reported progress on 240 consecutive clicks while clicking at random and
watching its own budget drain. Exactly the failure this paragraph was written to
rule out, arriving through the one door the mask does not cover.

So there is a second, count-based guard - see ``CLOCK_RATE`` - which asks how
*often* a colour's total moves and by how much. A counter ticks by one on nearly
every action; a set of objects goes away in object-sized chunks on the few actions
that touch one.

The history is also cut on a level change or a reset, because those restore the
board: comparing a fresh level's counts against the previous one's would read the
restoration as a giant increase and reject every real collectible.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

# The volatility mask is meaningless until it has seen a few transitions, and a
# wrong mask poisons every count taken under it, so tracking waits.
MIN_OBSERVED = 16
# Generous definition of "HUD": a pixel changing more than half the time is not
# telling us about the puzzle. Better to ignore a real object than to chase a clock.
HUD_THRESH = 0.5
# A ratchet has to click more than once, and may tolerate the occasional undo -
# pushing a crate back off a target is a legitimate move, not evidence against.
MIN_CLICKS = 2
UNDO_RATIO = 3
# An objective is a set of *things*, not a flood. A colour covering a quarter of
# the board is floor, wall or sky; its count drifting is not the puzzle being
# solved, and no amount of walking will drive it to zero. dc22 offered exactly
# this - a board-sized colour 0 with a clean downward ratchet - and chasing it
# would have replaced "no destination" with "an unreachable one".
MAX_SHARE = 0.25
# A colour that moves on nearly every action, by exactly one pixel, is a counter:
# a step budget, a health bar, a fuel gauge. It is the one thing that must never
# be mistaken for an objective, because "make this number go down" is satisfied by
# doing literally anything and the agent will happily spend its whole budget
# watching it drain. Measured on cd82: colour 4 falls 154 -> 126 by exactly one
# pixel per click, and the agent reported progress on 240 consecutive clicks.
#
# The HUD mask cannot catch this. Volatility asks "does this pixel change often",
# and a bar that loses one pixel per action changes each individual pixel on about
# 1/154 of frames - far under HUD_THRESH. cd82's mask is empty. So the test has to
# be on the *count* rather than on the pixels: how often does the total move, and
# by how much.
CLOCK_RATE = 0.55   # moved on this fraction of observed frames
CLOCK_UNIT = 0.75   # and that fraction of the moves were by exactly one pixel
CLOCK_MIN = 12      # not before there is enough history to mean anything


@dataclass
class Progress:
    """Per-colour pixel counts over time, and which of them ratchet."""

    fell: Counter = field(default_factory=Counter)
    rose: Counter = field(default_factory=Counter)
    last: dict[int, int] = field(default_factory=dict)
    hud: np.ndarray | None = None
    ignore: set[int] = field(default_factory=set)
    # largest count ever seen per colour, to recognise a flood rather than a thing
    peak: Counter = field(default_factory=Counter)
    field_size: int = 0
    # How many frame-to-frame comparisons have been made, how often each colour's
    # count moved at all, and how often it moved by exactly one pixel. Together
    # these separate a counter from a collectible - see CLOCK_RATE above.
    steps: int = 0
    moved: Counter = field(default_factory=Counter)
    unit: Counter = field(default_factory=Counter)

    def cut(self) -> None:
        """Forget the previous frame without forgetting what was learned.

        Called on a level change or a reset. The board has been restored, so the
        next comparison is not a continuation of the last one - but the ratchet
        evidence gathered so far is still true.
        """
        self.last.clear()

    def add(self, frame: np.ndarray, hud: np.ndarray | None, *, observed: int) -> None:
        """Fold one frame into the count history."""
        if observed < MIN_OBSERVED or hud is None:
            return
        if self.hud is None:
            # Freeze the mask on first use and never revise it. Recomputing it
            # every frame looks more responsive and is in fact fatal: the mask
            # wiggles as ``observed`` grows, and clearing the history on each
            # wiggle means no colour ever accumulates enough clicks to qualify.
            # ka59 ended a 240-step run with a perfectly empty ledger that way.
            self.hud = hud.copy()
            self.field_size = int((~hud).sum())
        vals, counts = np.unique(frame[~self.hud], return_counts=True)
        now = {int(v): int(n) for v, n in zip(vals, counts)}
        had_prev = bool(self.last)
        if had_prev:
            self.steps += 1
        for c, n in now.items():
            if n > self.peak[c]:
                self.peak[c] = n
            if c in self.ignore:
                continue
            prev = self.last.get(c)
            if prev is not None:
                if n != prev:
                    self.moved[c] += 1
                    if abs(n - prev) == 1:
                        self.unit[c] += 1
                if n < prev:
                    self.fell[c] += 1
                elif n > prev:
                    self.rose[c] += 1
        # A colour that disappeared entirely fell to zero; that is the strongest
        # possible ratchet click and would otherwise go unrecorded.
        for c, prev in self.last.items():
            if c not in now and c not in self.ignore and prev > 0:
                self.fell[c] += 1
                self.moved[c] += 1
                if prev == 1:
                    self.unit[c] += 1
        self.last = now

    # -- what ratchets -----------------------------------------------------

    def _flood(self, c: int) -> bool:
        """Is this colour scenery rather than a thing worth chasing?"""
        return bool(
            self.field_size and self.peak[c] > MAX_SHARE * self.field_size
        )

    def _clock(self, c: int) -> bool:
        """Is this colour a counter - a step budget, a bar - rather than a thing?

        Two conditions, and both are needed. *Moves nearly every frame* on its own
        would reject a collectible in a game where every move picks something up.
        *Moves by exactly one* on its own would reject single-pixel pickups. A
        quantity that does both is a number being displayed, not a set of objects
        being cleared: objects go away in object-sized chunks, and only on the few
        actions that actually touch one.

        This is the guard that ``MAX_SHARE`` is to floods. Both exist because the
        ratchet rule is powerful enough to find a monotone quantity in almost any
        game, and not every monotone quantity is the point of the game.
        """
        n = self.moved[c]
        return bool(
            self.steps >= CLOCK_MIN
            and n >= CLOCK_RATE * self.steps
            and self.unit[c] >= CLOCK_UNIT * n
        )

    def _ratchet(self, down: bool) -> set[int]:
        a, b = (self.fell, self.rose) if down else (self.rose, self.fell)
        return {
            c
            for c, n in a.items()
            if n >= MIN_CLICKS
            and n >= UNDO_RATIO * b.get(c, 0)
            and not self._flood(c)
            and not self._clock(c)
        }

    @property
    def consumed(self) -> set[int]:
        """Colours being used up. Fewer of these is closer to done."""
        return self._ratchet(down=True)

    @property
    def built(self) -> set[int]:
        """Colours being accumulated. More of these is closer to done."""
        return self._ratchet(down=False) - self._ratchet(down=True)

    def count(self, frame: np.ndarray, colors: set[int]) -> int:
        if not colors:
            return 0
        m = np.isin(frame, list(colors))
        if self.hud is not None:
            m &= ~self.hud
        return int(m.sum())

    # There is deliberately no ``score`` here. Turning these counts into a single
    # distance-from-done needs ``collectible`` as well, which lives on ``Dream``,
    # and this class used to carry a second copy of that arithmetic that nothing
    # called - a duplicate of the exact subtraction that produced 960 phantom
    # successes across the suite. One implementation, in ``Dream.objective``.

    def summary(self) -> str:
        return f"consumed={sorted(self.consumed)} built={sorted(self.built)}"
