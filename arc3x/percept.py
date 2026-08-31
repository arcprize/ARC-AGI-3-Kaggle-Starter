"""What a person notices in the first ten seconds of an unseen game.

A human handed an ARC-AGI-3 game does not reason about 4,096 pixels. They see a
handful of *things*, they press a button, and they watch which thing moved. This
module is that faculty, and nothing more: frame in, structured facts out. No
policy, no search, no scoring.

The five primitives, in the order a person uses them:

  1. ``blobs``       - "what objects are on screen?" Connected same-colour runs.
  2. ``rigid_shift`` - "what moved, and how far?" A colour whose pixel set is the
                       same shape translated. This is how the avatar is found.
  3. ``Volatility``  - "what is scenery, what is alive, what is just a counter?"
                       Pixels that change on every single action are a HUD clock,
                       not game state; the cell abstraction in ``cell.py`` was
                       already burned once by treating them as state, which made
                       every frame unique and silently killed Go-Explore.
  4. ``tile_size``   - "what is the grid?" Movement deltas share a divisor: an
                       8px step means the game is a grid of 8px cells, and
                       planning should happen on that grid, not per pixel.
  5. ``touching``    - "what am I about to bump into?" The colours immediately
                       ahead of a sprite in a direction of travel, which is how
                       walls and hazards get labelled without reading source.

Everything here is pure numpy over a 64x64 int array. It never touches a game
object, so it works identically against a local twin and against the gateway.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd

import numpy as np

GRID = 64


# -- 1. objects --------------------------------------------------------------


@dataclass(frozen=True)
class Blob:
    """One connected run of a single colour - a "thing" on screen."""

    color: int
    size: int
    top: int
    left: int
    height: int
    width: int
    cy: float
    cx: float

    @property
    def center(self) -> tuple[int, int]:
        """Integer centre, clamped into the grid. What a person would click."""
        return (
            int(min(GRID - 1, max(0, round(self.cy)))),
            int(min(GRID - 1, max(0, round(self.cx)))),
        )

    @property
    def is_rect(self) -> bool:
        return self.size == self.height * self.width


def blobs(frame: np.ndarray, ignore: set[int] | None = None) -> list[Blob]:
    """Connected same-colour components, 4-connectivity, largest first.

    Implemented as an explicit stack flood fill rather than scipy.label because
    the competition image is not guaranteed to have scipy, and 64x64 is small
    enough that it does not matter.
    """
    ignore = ignore or set()
    seen = np.zeros(frame.shape, dtype=bool)
    out: list[Blob] = []
    h, w = frame.shape
    for y0 in range(h):
        for x0 in range(w):
            if seen[y0, x0]:
                continue
            c = int(frame[y0, x0])
            if c in ignore:
                seen[y0, x0] = True
                continue
            stack = [(y0, x0)]
            seen[y0, x0] = True
            pix: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                pix.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx]:
                        if int(frame[ny, nx]) == c:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            ys = [p[0] for p in pix]
            xs = [p[1] for p in pix]
            out.append(
                Blob(
                    color=c,
                    size=len(pix),
                    top=min(ys),
                    left=min(xs),
                    height=max(ys) - min(ys) + 1,
                    width=max(xs) - min(xs) + 1,
                    cy=sum(ys) / len(pix),
                    cx=sum(xs) / len(pix),
                )
            )
    out.sort(key=lambda b: -b.size)
    return out


# -- 2. what moved ------------------------------------------------------------


def rigid_shift(
    before: np.ndarray, after: np.ndarray, color: int
) -> tuple[int, int] | None:
    """Did ``color``'s pixel set move as a rigid body? Returns (dy, dx) or None.

    The bounding box gives the candidate offset in O(1) instead of searching a
    -16..16 window, and the full mask comparison then either confirms it or
    rejects it outright. A returned (0, 0) means the colour is present and did
    not move, which is different from None (shape changed, or colour vanished).
    """
    mb = before == color
    ma = after == color
    nb = int(mb.sum())
    if nb == 0 or nb != int(ma.sum()):
        return None
    yb, xb = np.nonzero(mb)
    ya, xa = np.nonzero(ma)
    dy = int(ya.min() - yb.min())
    dx = int(xa.min() - xb.min())
    if dy == 0 and dx == 0:
        return (0, 0) if np.array_equal(mb, ma) else None
    # Shift mb by (dy, dx) and require an exact match. np.roll would wrap, which
    # would silently accept a sprite that left one edge and reappeared on the
    # other, so the slice is done explicitly.
    h, w = before.shape
    sy0, sy1 = max(0, dy), min(h, h + dy)
    sx0, sx1 = max(0, dx), min(w, w + dx)
    if sy0 >= sy1 or sx0 >= sx1:
        return None
    shifted = np.zeros_like(mb)
    shifted[sy0:sy1, sx0:sx1] = mb[sy0 - dy : sy1 - dy, sx0 - dx : sx1 - dx]
    return (dy, dx) if np.array_equal(shifted, ma) else None


def all_shifts(before: np.ndarray, after: np.ndarray) -> dict[int, tuple[int, int]]:
    """Every colour that moved rigidly, and by how much. Excludes stationary."""
    out: dict[int, tuple[int, int]] = {}
    for c in np.unique(before):
        s = rigid_shift(before, after, int(c))
        if s is not None and s != (0, 0):
            out[int(c)] = s
    return out


def changed(before: np.ndarray, after: np.ndarray) -> int:
    return int((before != after).sum())


# -- 2b. what moved, tracked as an OBJECT rather than as a colour -------------
#
# Comparing whole colour masks finds an avatar in only 9 of the 25 dev games,
# because an avatar that shares its colour with any scenery fails the test: the
# mask contains stationary pixels, so it is not a rigid translation of itself.
# A person does not track colours, they track the little thing that moved. That
# is what this does, and it is also cheaper: a move changes few pixels, so the
# flood fills happen inside the bounding box of the difference instead of over
# the whole frame.


@dataclass(frozen=True)
class Move:
    """One object that changed between two frames."""

    color: int
    size: int
    dy: int
    dx: int
    top: int
    left: int

    @property
    def delta(self) -> tuple[int, int]:
        return (self.dy, self.dx)

    @property
    def moved(self) -> bool:
        return (self.dy, self.dx) != (0, 0)


def _component(
    frame: np.ndarray, y: int, x: int, limit: int = 4096
) -> tuple[np.ndarray, int, int, int]:
    """Connected same-colour component containing (y, x): (mask, top, left, size)."""
    c = int(frame[y, x])
    h, w = frame.shape
    mask = np.zeros((h, w), dtype=bool)
    stack = [(y, x)]
    mask[y, x] = True
    n = 1
    top, left, bot, right = y, x, y, x
    while stack and n <= limit:
        cy, cx = stack.pop()
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < h and 0 <= nx < w and not mask[ny, nx]:
                if int(frame[ny, nx]) == c:
                    mask[ny, nx] = True
                    n += 1
                    stack.append((ny, nx))
                    top = min(top, ny)
                    left = min(left, nx)
                    bot = max(bot, ny)
                    right = max(right, nx)
    return mask[top : bot + 1, left : right + 1], top, left, n


def mask_component(
    mask: np.ndarray, y: int, x: int, limit: int = 4096
) -> tuple[np.ndarray, int, int, int]:
    """Connected True-region of ``mask`` containing (y, x): (sub, top, left, size).

    The same flood fill as ``_component`` but over a boolean mask rather than a
    colour, which is what finds a *multi-colour* sprite as one object: union the
    body colours into a mask, and the avatar is a single connected region of it.
    """
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    stack = [(y, x)]
    seen[y, x] = True
    n = 1
    top, left, bot, right = y, x, y, x
    while stack and n <= limit:
        cy, cx = stack.pop()
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and mask[ny, nx]:
                seen[ny, nx] = True
                n += 1
                stack.append((ny, nx))
                top = min(top, ny)
                left = min(left, nx)
                bot = max(bot, ny)
                right = max(right, nx)
    return seen[top : bot + 1, left : right + 1], top, left, n


def moved_objects(
    before: np.ndarray,
    after: np.ndarray,
    *,
    max_size: int = 256,
    max_moves: int = 6,
) -> tuple[list[Move], list[Move], list[Move]]:
    """(moved, vanished, appeared) between two frames, as whole objects.

    For each colour present in the changed region, the object that left is the
    connected component in ``before`` under a changed pixel, and the object that
    arrived is the component in ``after``. If the two have the same pixel shape,
    the object translated and the offset is exact; if only one side exists, the
    object was collected or spawned - which is how the agent learns what a goal
    and a hazard look like without being told.
    """
    diff = before != after
    if not diff.any():
        return [], [], []
    ys, xs = np.nonzero(diff)
    moved: list[Move] = []
    vanished: list[Move] = []
    appeared: list[Move] = []
    # Colours that lost pixels here are candidates for "the thing that moved".
    done_b = np.zeros(before.shape, dtype=bool)
    done_a = np.zeros(before.shape, dtype=bool)
    lefts: list[tuple[np.ndarray, int, int, int, int]] = []
    arrivals: list[tuple[np.ndarray, int, int, int, int]] = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        if not done_b[y, x]:
            m, t, l, n = _component(before, y, x, limit=max_size)
            done_b[t : t + m.shape[0], l : l + m.shape[1]] |= m
            if n <= max_size:
                lefts.append((m, t, l, n, int(before[y, x])))
        if not done_a[y, x]:
            m, t, l, n = _component(after, y, x, limit=max_size)
            done_a[t : t + m.shape[0], l : l + m.shape[1]] |= m
            if n <= max_size:
                arrivals.append((m, t, l, n, int(after[y, x])))
        if len(lefts) > 4 * max_moves and len(arrivals) > 4 * max_moves:
            break

    used = set()
    for mb, tb, lb, nb, cb in lefts:
        best = None
        for j, (ma, ta, la, na, ca) in enumerate(arrivals):
            if j in used or ca != cb or na != nb or ma.shape != mb.shape:
                continue
            if not np.array_equal(ma, mb):
                continue
            d = abs(ta - tb) + abs(la - lb)
            if best is None or d < best[0]:
                best = (d, j, ta, la)
        if best is None:
            vanished.append(Move(cb, nb, 0, 0, tb, lb))
            continue
        _d, j, ta, la = best
        used.add(j)
        moved.append(Move(cb, nb, ta - tb, la - lb, tb, lb))
    for j, (ma, ta, la, na, ca) in enumerate(arrivals):
        if j not in used:
            appeared.append(Move(ca, na, 0, 0, ta, la))
    moved.sort(key=lambda m: m.size)
    return moved[:max_moves], vanished[:max_moves], appeared[:max_moves]



# -- 3. scenery vs state vs counter ------------------------------------------


@dataclass
class Volatility:
    """Which pixels are game state, and which are just a clock ticking.

    ``changes`` counts, per pixel, how many observed transitions altered it.
    ``observed`` is how many transitions were seen. A pixel that changes on
    essentially every transition is a HUD counter: it carries no positional
    information and must be excluded from any state fingerprint, or every state
    looks novel forever.
    """

    changes: np.ndarray = field(
        default_factory=lambda: np.zeros((GRID, GRID), dtype=np.int32)
    )
    observed: int = 0

    def add(self, before: np.ndarray, after: np.ndarray) -> None:
        """Record one same-shaped transition, restarting on a new board shape.

        Kaggle games currently render at ``GRID`` square pixels, but the online
        pilot and its offline harness intentionally accept smaller boards.  A
        transition from a differently shaped level has no pixelwise meaning, so
        it is safer to begin a fresh volatility ledger than to broadcast a stale
        64x64 HUD mask (or silently count unrelated pixels as a clock).
        """
        if before.shape != after.shape:
            return
        if self.changes.shape != before.shape:
            self.changes = np.zeros(before.shape, dtype=np.int32)
            self.observed = 0
        self.changes += (before != after).astype(np.int32)
        self.observed += 1

    def hud_mask(self, thresh: float = 0.9) -> np.ndarray:
        """Pixels that change almost every action: a counter, not the world."""
        if self.observed < 8:
            return np.zeros(self.changes.shape, dtype=bool)
        return self.changes >= max(1, int(thresh * self.observed))

    def static_mask(self) -> np.ndarray:
        """Pixels that never changed once: walls, borders, decoration."""
        return self.changes == 0

    @property
    def live_mask(self) -> np.ndarray:
        """The pixels worth fingerprinting: they move, but not every tick."""
        return (~self.static_mask()) & (~self.hud_mask())


# -- 4. the grid --------------------------------------------------------------


def tile_size(deltas: dict[int, tuple[int, int]]) -> int:
    """The step size the game really works in, from the observed move deltas.

    A game whose avatar moves 8px per press is an 8px grid game; planning it per
    pixel multiplies the search space by 64 for no benefit. The gcd of every
    non-zero component recovers that step without any assumption about the game.
    """
    g = 0
    for dy, dx in deltas.values():
        for v in (abs(dy), abs(dx)):
            if v:
                g = gcd(g, v)
    return g or 1


# -- 5. what is in the way ----------------------------------------------------


def touching(
    frame: np.ndarray, mask: np.ndarray, dy: int, dx: int, ignore: set[int] | None = None
) -> set[int]:
    """Colours in the band ``mask`` would sweep into if shifted by (dy, dx).

    Used to name the thing that blocked a move, or the thing that killed us,
    without ever knowing what the game calls it.
    """
    ignore = ignore or set()
    h, w = frame.shape
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return set()
    out: set[int] = set()
    steps = max(abs(dy), abs(dx)) or 1
    uy = dy / steps
    ux = dx / steps
    for k in range(1, steps + 1):
        ty = np.round(ys + uy * k).astype(int)
        tx = np.round(xs + ux * k).astype(int)
        ok = (ty >= 0) & (ty < h) & (tx >= 0) & (tx < w)
        if not ok.any():
            continue
        vals = frame[ty[ok], tx[ok]]
        inside = mask[ty[ok], tx[ok]]
        for v in np.unique(vals[~inside]):
            if int(v) not in ignore:
                out.add(int(v))
    return out


def background(frame: np.ndarray) -> int:
    vals, counts = np.unique(frame, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def fingerprint(frame: np.ndarray, live: np.ndarray | None = None) -> bytes:
    """A hashable state key. ``live`` masks out HUD counters and dead scenery."""
    if live is None:
        return frame.astype(np.int8).tobytes()
    f = frame.astype(np.int8).copy()
    f[~live] = -1
    return f.tobytes()
