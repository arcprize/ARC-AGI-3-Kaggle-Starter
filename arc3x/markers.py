"""Where the goal is, read off the first frame, before anything has been won.

WHY THIS EXISTS
---------------
``Dream.target_colors`` is ``(collectible | prog.consumed) - retired``, and both
of those sources are **reactive**:

  * ``collectible`` needs a colour to have *already* vanished under the avatar's
    own feet (``dream.py`` ``collect[c] >= 1``).
  * ``prog.consumed`` needs a count to have *already* ratcheted down twice
    (``progress.py`` ``MIN_CLICKS = 2``).

So the destination set is empty until the agent has accidentally succeeded twice,
and ``progress.py``'s own opening table is what that costs:

    dc22   move=100%(153)  collect=[]   thought=  0   levels=0
    ka59   move=100%(207)  collect=[]   thought=  0   levels=0
    m0r0   move= 92%(126)  collect=[]   thought=  0   levels=0
    sp80   move= 80%(195)  collect=[8]  thought= 61   levels=1

A *perfect* copy of the game completed nothing, because a route needs a
destination. ``progress.py`` fixed half of this - the ratchet needs no completed
level - but it still needs the first success. On a game where the first success
is a multi-step placement rather than an accidental bump, that never arrives.

THE EVIDENCE THAT SAYS WHAT TO LOOK FOR
---------------------------------------
All 25 dev games ship as source, so their win conditions can simply be read
(``datasets/arc-prize-2026-arc-agi-3/environment_files/*/*/*.py``, the guard on
each ``self.next_level()``). Read on 2026-08-23:

    ka59  every tag-A overlaps some tag-B, and every tag-C some tag-D
    s5i5  every tag-A is at the position of some tag-B
    lp85  every tag-A has a "goal" at (x+1, y+1); every tag-B a "goal-o"
    tu93  the avatar is at the position of some exit
    wa30  all tag-A sit on a predicate-true cell
    dc22  one sprite's (x, y) equals another's
    m0r0  the count of uncollected items reaches 0
    ls20  the avatar reaches each of N targets, in order
    cd82  a 10x10 canvas equals a 10x10 target, both diagonals masked out
    re86  the composited sprites equal a reference canvas
    cn04  no pixel of colour 8 or 13 remains
    r11l  no entry is still flagged
    ft09  each tile matches-or-differs from its neighbour, per a flag

**Ten of the thirteen readable conditions are one predicate: every object of kind
A must be co-located with an object of kind B.** cd82 and re86 are that same idea
at pixel granularity. Which means the destination is not hidden - it is *drawn on
the board*, and it was drawn before the first action.

WHAT THAT MAKES DETECTABLE
--------------------------
A target marker, in a game that draws one, is:

  * **repeated** - a game draws its N targets identically, so the same
    (colour, shape) signature occurs more than once. This is the load-bearing
    signal, and it is why a lone blob is never proposed.
  * **static** - it does not move under its own steam, and it is not a colour the
    avatar has been seen to move.
  * **small** - it marks a place. Scenery covering a quarter of the board is
    floor or wall, which is the same ``MAX_SHARE`` judgement ``progress.py``
    already makes for the same reason.

None of those three name a game, a genre, or a colour. What they do *not* do is
prove the guess right - and they do not have to, for two reasons that come from
measurement rather than from hope:

  1. ``Dream.retire`` already exists and already handles a wrong destination,
     because "a wrong destination costs more than no destination" was learned the
     expensive way. A speculative source can be plugged into a system that knows
     how to abandon it.
  2. Guessing wrong on level 0 is nearly free. Level 0 carries weight 1 of 21-55,
     at most 4.8% of a game, and an action spent on a level that is never cleared
     costs *no score at all* - only budget. See ``ARCHITECTURE.md`` section 2(b).

So this module is deliberately a **proposer**, ranked, with the ranking exposed;
it is not an oracle. It answers "what would a person try first", which on the
evidence above is "walk to one of those repeated little things".

WHAT IS NOT HERE
----------------
The region-match variant - cd82 and re86, where the objective is the mismatch
between a static reference picture and a changing workspace - is a different
detector over ``Volatility.changes`` and is specified in ``PLAN.md`` rather than
written here. Two games, and it needs a change history rather than one frame.

STATUS: never executed. Not wired into ``Dream.target_colors``. The measurement
that decides whether it should be is ``arc3x/why_markers.py``, which reports how
many actions the reactive sources need to reach the same answer this returns from
frame 0 - and that number is the whole claim.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from arc3x.percept import Blob, background, blobs

# A marker marks a place, so it is small. 64 px is an 8x8 tile - larger than any
# target sprite read in the census above (tu93's exits are 3x3, cd82's swatches
# 5x5) and small enough to exclude a wall segment or a panel.
MAX_MARKER = 64
# Repetition is the signal. One blob of a colour is a thing; several identical
# blobs are a *set of places*, which is what every cover condition in the census
# quantifies over. Two is the least that can be called repeated.
MIN_REPEAT = 2
# The same judgement, and the same number, as ``progress.MAX_SHARE``: a colour
# covering a quarter of the playfield is floor, wall or sky. Kept as a separate
# name because it is applied to a colour's *whole* share here, not to a ratchet.
MAX_SHARE = 0.25


@dataclass(frozen=True)
class MarkerSet:
    """A set of identical static blobs - a candidate answer to "go where?".

    ``cells`` are the integer centres, which is both what a planner routes to and
    what a click would aim at, so the same proposal serves a movement game and a
    click game with no branch.
    """

    color: int
    height: int
    width: int
    size: int
    cells: tuple[tuple[int, int], ...]
    share: float

    @property
    def count(self) -> int:
        return len(self.cells)

    @property
    def signature(self) -> tuple[int, int, int, int]:
        return (self.color, self.height, self.width, self.size)

    def __str__(self) -> str:
        return (
            f"c{self.color} {self.height}x{self.width}(n{self.size}) "
            f"x{self.count} @{list(self.cells[:4])}"
        )


def markers(
    frame: np.ndarray,
    *,
    moved: set[int] | None = None,
    movers: int = 0,
    bg: int | None = None,
    min_repeat: int = MIN_REPEAT,
    max_marker: int = MAX_MARKER,
) -> list[MarkerSet]:
    """Candidate target sets on this frame, best guess first.

    ``moved`` is the set of colours the agent has seen move - those are pieces,
    not places, and proposing one as a destination would send the avatar chasing
    itself. Passing an empty set is legitimate and is what happens on frame 0:
    the repetition and size tests still carry the proposal.

    ``movers`` is how many movable pieces are believed to exist, and it is used
    only for ranking, never for filtering. In a cover game the target count is
    usually at least the piece count - N crates need N sockets - so a set with
    enough members to receive every piece is preferred. It is a tie-break rather
    than a rule because the census has both shapes: lp85 pairs its tags off
    one-for-one, while tu93 offers several exits to a single avatar.

    Groups by (colour, bbox height, bbox width, pixel count) rather than by exact
    mask. That signature is cheap, needs nothing beyond ``blobs``, and for sprites
    this small it separates the cases that matter; two different 3x3 shapes with
    the same pixel count are possible but were not observed in the census.
    """
    moved = moved or set()
    if bg is None:
        bg = background(frame)
    field = int(frame.size)
    # A colour's whole share of the frame, not the group's: a wall drawn as forty
    # identical 3x3 tiles is still a wall, and only the total says so.
    shares = {
        int(c): int(n) / field
        for c, n in zip(*np.unique(frame, return_counts=True))
    }

    groups: dict[tuple[int, int, int, int], list[Blob]] = defaultdict(list)
    for b in blobs(frame, ignore={bg}):
        if b.color in moved:
            continue          # a piece, not a place
        if b.size > max_marker:
            continue          # scenery, panel, wall run
        if shares.get(b.color, 0.0) > MAX_SHARE:
            continue          # floor, wall or sky
        groups[(b.color, b.height, b.width, b.size)].append(b)

    out: list[MarkerSet] = []
    for (color, h, w, size), members in groups.items():
        if len(members) < min_repeat:
            continue
        members.sort(key=lambda b: (b.top, b.left))
        out.append(
            MarkerSet(
                color=color,
                height=h,
                width=w,
                size=size,
                cells=tuple(b.center for b in members),
                share=shares.get(color, 0.0),
            )
        )

    out.sort(key=lambda m: _rank(m, movers), reverse=True)
    return out


def _rank(m: MarkerSet, movers: int) -> tuple[int, int, int]:
    """Which proposal a person would try first.

    Three keys, in order, each one a claim that can be argued with:

      1. **Enough of them to go round.** A set that can receive every piece is a
         plausible socket set; one with fewer members than there are pieces
         cannot satisfy a cover condition on its own. Ranking, not filtering -
         see the docstring above.
      2. **More of them.** Repetition is the evidence for "these are places",
         so more repetitions is more evidence.
      3. **Smaller.** A marker marks a spot. Between two repeated static shapes
         the smaller one is the more likely to be a socket rather than a slab.
    """
    return (1 if movers and m.count >= movers else 0, m.count, -m.size)


def marker_colors(sets: list[MarkerSet], top: int = 2) -> set[int]:
    """The colours of the best few proposals - the shape ``Dream`` consumes.

    ``target_colors`` is a set of colours, so a colour-level view is what plugs
    in. Two by default rather than one because several census games have *two*
    tag pairs to satisfy at once (ka59, lp85), and because ``retire`` strikes the
    whole believed set when it fails, which makes a slightly wider first guess
    cheap to walk back.
    """
    return {m.color for m in sets[:top]}


def summary(sets: list[MarkerSet], top: int = 4) -> str:
    if not sets:
        return "markers: none"
    return "markers: " + " | ".join(str(m) for m in sets[:top])
