"""Palette-invariant ways to describe a frame, so a policy can transfer between games.

THE MEASUREMENT THAT FORCED THIS FILE
-------------------------------------
``arc3x/transfer.py`` trained one policy on 17 games and tested it on 8 it had
never seen. It reached 3.75x random on the games it trained on and **1.10x** on
the others - nothing. ``arc3x/why_no_transfer.py`` then measured both ends of the
function it was trying to learn, ``frame -> action id``, and found the two ends
fail for opposite reasons:

* **The output end is fine.** Button ids share meaning across games far more than
  expected: ACTION1 moves north in 90% of the games that offer it, ACTION2 south
  in 92%, ACTION3 west in 100%, ACTION4 east in 92%. Only ACTION5, the use
  button, is genuinely per-game. So emitting a wire id is not the problem.
* **The input end is the problem.** The avatar is 9 different colours across the
  14 games where it could be identified; the most common one covers 29%. The
  background spreads over 9 colours too. The old encoding was a one-hot over
  *absolute colour index*, so a hidden unit that learned "colour 4 here means me"
  fires on a wall in the next game. The model was not failing to generalise - it
  was generalising correctly over features that do not mean the same thing twice.

So the fix is not more teacher data. It is to describe a frame by what things
*do* rather than by what colour they are.

THREE ENCODINGS, DELIBERATELY COMPARABLE
----------------------------------------
Each returns a flat float32 vector over the same 16x16 coarse grid, so the same
training harness measures all three and the only thing that changes is meaning.

``color``  16 planes, one-hot over absolute colour index. The old behaviour,
           kept as the control. Without it the other two have nothing to beat.

``rank``   16 planes, one-hot over colour *rank by area in this frame* - the
           most common colour is plane 0, the next plane 1, and so on. Free: no
           actions, no learning, no per-game state. It is palette-invariant by
           construction, and it captures the one regularity that holds almost
           everywhere in these games - the floor is the biggest region and the
           thing you control is a small rare blob. Cells pool by *rarest* rank
           rather than by maximum colour value, because a 4x4 cell containing
           one avatar pixel and fifteen floor pixels should read as the avatar;
           that is the whole reason the avatar survives pooling at all.

``role``   8 planes of learned meaning, multi-hot: avatar body, background,
           blocking, passable, fatal, goal, vanished, unattributed. Taken from
           ``Mechanics`` (``arc3x/mind.py``), which learns every one of them by
           pressing buttons and watching - no engine source, nothing keyed on
           game id. ``why_no_transfer.py`` measured the price at **27 billed
           actions per game**, against a human baseline of 26-230 actions for
           level 0 alone. Multi-hot rather than one-hot because a colour can be
           both passable and a goal, and collapsing that loses the distinction
           the planner needs.

Roles are the honest target: they are what a person works out in the first few
seconds ("that one is me, those kill me, that is the door"). Ranks are the cheap
approximation that needs no actions at all. Measuring both says how much of the
transfer is available for free and how much has to be paid for.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# The coarse grid is shared with the student's click head, so a click slot and a
# feature cell are the same square. Changing one here would silently decouple
# where the model looks from where it clicks.
GRID = 64
POOL = 4
CELLS = GRID // POOL
N_COLOR = 16
N_ROLE = 8

N_IN_COLOR = N_COLOR * CELLS * CELLS   # 4096
N_IN_RANK = N_COLOR * CELLS * CELLS    # 4096
N_IN_ROLE = N_ROLE * CELLS * CELLS     # 2048

ROLE_NAMES = (
    "avatar", "background", "blocking", "passable",
    "fatal", "goal", "vanished", "other",
)


def _fit_grid(frame: np.ndarray) -> np.ndarray:
    """Pad or crop to 64x64 so a short frame cannot change the vector's length."""
    f = np.asarray(frame)
    if f.shape == (GRID, GRID):
        return f
    out = np.zeros((GRID, GRID), dtype=np.int16)
    h, w = min(GRID, f.shape[0]), min(GRID, f.shape[1])
    out[:h, :w] = f[:h, :w]
    return out


def _cells_any(mask: np.ndarray) -> np.ndarray:
    """(64,64) bool -> (16,16) bool: does any pixel of this cell have the property?

    ``any`` and not ``all``: these games draw single-cell sprites, and a cell that
    only counts a property when every pixel has it cannot see them.
    """
    return mask.reshape(CELLS, POOL, CELLS, POOL).any(axis=(1, 3))


# -- color: the control -----------------------------------------------------


def encode_color(frame: np.ndarray, ctx: Any = None) -> np.ndarray:
    """One-hot over absolute colour index. Identical to ``student.featurise``."""
    f = _fit_grid(frame)
    blocks = f.reshape(CELLS, POOL, CELLS, POOL).max(axis=(1, 3))
    blocks = np.clip(blocks, 0, N_COLOR - 1).astype(np.intp)
    x = np.zeros(N_IN_COLOR, dtype=np.float32)
    x[blocks.ravel() * (CELLS * CELLS) + _OFFSET] = 1.0
    return x


_OFFSET = np.arange(CELLS * CELLS, dtype=np.intp)


# -- rank: palette-invariant and free ---------------------------------------


def rank_map(frame: np.ndarray) -> np.ndarray:
    """Per pixel, the rank of its colour by area in this frame. 0 = most common.

    Ties are broken by colour index only to keep the function deterministic; a
    tie between two equally large regions carries no information either way.
    """
    f = _fit_grid(frame)
    vals, counts = np.unique(f, return_counts=True)
    # Stable sort on (-count, value) so the ordering does not wobble frame to
    # frame when two regions happen to be the same size.
    order = np.lexsort((vals, -counts))
    lut = np.full(N_COLOR, N_COLOR - 1, dtype=np.intp)
    for r, i in enumerate(order):
        c = int(vals[i])
        if 0 <= c < N_COLOR:
            lut[c] = min(r, N_COLOR - 1)
    return lut[np.clip(f, 0, N_COLOR - 1)]


def encode_rank(frame: np.ndarray, ctx: Any = None) -> np.ndarray:
    """One-hot over colour rank, pooled so the rarest colour in a cell wins."""
    r = rank_map(frame)
    blocks = r.reshape(CELLS, POOL, CELLS, POOL).min(axis=(1, 3)).astype(np.intp)
    x = np.zeros(N_IN_RANK, dtype=np.float32)
    x[blocks.ravel() * (CELLS * CELLS) + _OFFSET] = 1.0
    return x


# -- role: learned meaning --------------------------------------------------


def role_lut(m: Any) -> np.ndarray:
    """colour -> multi-hot role vector, from what ``Mechanics`` believes.

    A colour with no evidence at all lands in ``other``, which keeps the vector
    non-zero everywhere and lets the model learn "I do not know what this is"
    as its own feature rather than as silence.
    """
    lut = np.zeros((N_COLOR, N_ROLE), dtype=np.float32)
    body = set(getattr(m, "body", set()) or ())
    av = int(getattr(m, "avatar", -1))
    if av >= 0:
        body.add(av)
    # ``blocked_set`` is a property on Mechanics, but tolerate a callable so a
    # stand-in object in a test does not have to match that detail.
    blocked = getattr(m, "blocked_set", set())
    blocked = set(blocked() if callable(blocked) else blocked)
    bg = int(getattr(m, "background", -1))

    def counter(name: str) -> dict:
        return dict(getattr(m, name, {}) or {})

    passable, fatal, goal, vanished = (
        counter("passable"), counter("fatal"), counter("goal_colors"), counter("vanished")
    )
    for c in range(N_COLOR):
        v = lut[c]
        if c in body:
            v[0] = 1.0
        if c == bg:
            v[1] = 1.0
        if c in blocked:
            v[2] = 1.0
        if passable.get(c, 0) > 0:
            v[3] = 1.0
        if fatal.get(c, 0) > 0:
            v[4] = 1.0
        if goal.get(c, 0) > 0:
            v[5] = 1.0
        if vanished.get(c, 0) > 0:
            v[6] = 1.0
        if not v.any():
            v[7] = 1.0
    return lut


def encode_role(frame: np.ndarray, ctx: Any = None) -> np.ndarray:
    """Multi-hot role planes. ``ctx`` is a ``Mechanics`` (or a role LUT).

    With no ``ctx`` every cell reads ``other``, which is a legitimate state - it
    is what an agent sees before it has pressed anything - and it makes the
    encoder safe to call on the first frame of a game.
    """
    lut = ctx if isinstance(ctx, np.ndarray) else (
        role_lut(ctx) if ctx is not None else None
    )
    f = _fit_grid(frame)
    x = np.zeros((N_ROLE, CELLS, CELLS), dtype=np.float32)
    if lut is None:
        x[ROLE_NAMES.index("other")] = 1.0
        return x.ravel()
    cf = np.clip(f, 0, N_COLOR - 1)
    for r in range(N_ROLE):
        colors = np.nonzero(lut[:, r])[0]
        if colors.size:
            x[r] = _cells_any(np.isin(cf, colors))
    return x.ravel()


ENCODERS = {
    "color": (encode_color, N_IN_COLOR),
    "rank": (encode_rank, N_IN_RANK),
    "role": (encode_role, N_IN_ROLE),
}
