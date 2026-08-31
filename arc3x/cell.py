"""Cell abstraction for Go-Explore: deciding when two situations are "the same".

WHY THIS EXISTS
---------------
Go-Explore's power comes entirely from the *cell* being a **coarse** summary of
a state. Many different action sequences must land in the same cell, so that
"have I been here before?" carries information and "keep the shortest route to
a known cell" actually fires.

The first version of this search hashed the raw 64x64 frame. Measured on the
real games, that key is effectively **bijective**:

    tn36:  60 steps -> 60 distinct keys      (every single step "novel")
    sk48: 150 steps -> 124 distinct keys

With a 1:1 key the archive degenerates into a log of every state ever visited,
novelty stops being a signal, and the whole thing collapses to a random walk.
That is why the search explored 1,713 cells on tn36 and completed zero levels.

WHAT MAKES EVERY FRAME UNIQUE
-----------------------------
Only 2-3% of the 4096 pixels ever change during a walk. Inspecting those on
tn36 found the culprit immediately - row 1 is a 49-pixel HUD bar draining by
exactly 6 every action:

    row 1 colour-sum over time: 441, 435, 429, 423, 417, 411, 405, 399, ...

That is a *clock*, not state. It alone makes every frame globally unique.
Meanwhile the actual game state (rows 42-46) repeats constantly.

THE GENERAL RULE
----------------
A pixel whose value moves **monotonically with the action count** is a clock:
a timer, an energy bar, a step counter, a score readout. State revisits values;
a clock never does. So:

    informative = varies during probing  AND  not monotone-in-time

This is measured per game from random probe walks - purely frame-derived, no
per-game knowledge, no hand-written list of HUD locations. Verified collapse:

    game   raw-frame keys   informative-only keys
    tn36        60/60             43/60
    sk48        91/120            28/120     (3.3x collapse)
    m0r0        77/120            29/120     (2.7x collapse)

Monotonicity is required in *every* walk where the pixel varies (not just one),
so a pixel that happens to drift one way in a single short walk is not
mistakenly discarded as a clock.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from arc3x.twin import Act, Twin


@dataclass
class CellKey:
    """A calibrated frame -> cell-id function.

    ``mask`` selects the pixels that carry state. ``pool`` optionally coarsens
    further by max-pooling over pool x pool blocks before hashing, which merges
    near-identical states (useful when a game has a large continuously-moving
    body, e.g. a long trail).
    """

    mask: np.ndarray  # bool (64, 64); which pixels are informative
    pool: int = 1
    n_varying: int = 0
    n_clock: int = 0

    @property
    def n_informative(self) -> int:
        return int(self.mask.sum())

    def __call__(self, frame: np.ndarray, level: int) -> bytes:
        """Hash the informative part of the frame plus the level index.

        ``frame[mask]`` is numpy boolean indexing, so this is C-speed - about
        2 microseconds, versus the ~1.2 ms it costs to step the engine. The key
        function is never the bottleneck.
        """
        if self.pool > 1:
            h, w = frame.shape
            p = self.pool
            hh, ww = h // p * p, w // p * p
            blocks = frame[:hh, :ww].reshape(hh // p, p, ww // p, p).max(axis=(1, 3))
            payload = blocks.tobytes()
        else:
            payload = frame[self.mask].tobytes()
        return hashlib.blake2b(
            payload + bytes((level & 0xFF,)), digest_size=16
        ).digest()


def calibrate(
    root: Any,
    *,
    walks: int = 4,
    steps: int = 150,
    seed: int = 0,
    pool: int = 1,
) -> CellKey:
    """Learn which pixels carry state, by random probing of a cloned game.

    Costs ``walks * steps`` simulated actions (~600 steps, under a second) and
    spends zero graded actions because everything runs on deepcopies.
    """
    stacks: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    for w in range(walks):
        g = copy.deepcopy(root)
        cur = Twin.valid_actions(g)
        frames: list[np.ndarray] = []
        for _ in range(steps):
            if not cur:
                break
            obs = Twin.step_game(g, cur[int(rng.integers(len(cur)))])
            if obs.terminal:
                break
            frames.append(obs.frame)
            cur = obs.valid or cur
        if len(frames) >= 3:
            stacks.append(np.stack(frames))

    varies = np.zeros((64, 64), dtype=bool)
    # A pixel is a clock only if it is monotone in EVERY walk where it moves.
    clock = np.ones((64, 64), dtype=bool)
    if not stacks:
        # Nothing observable (game died instantly): fall back to the full frame.
        return CellKey(mask=np.ones((64, 64), dtype=bool), pool=pool)

    for st in stacks:
        v = st.max(axis=0) != st.min(axis=0)
        varies |= v
        d = np.diff(st.astype(np.int16), axis=0)
        mono = (d >= 0).all(axis=0) | (d <= 0).all(axis=0)
        # Pixels that did not vary in this walk say nothing about clock-ness,
        # so they must not veto the verdict from other walks.
        clock &= mono | ~v

    clock &= varies
    mask = varies & ~clock

    # Degenerate fallbacks, in order of preference.
    if not mask.any():
        mask = varies.copy()
    if not mask.any():
        mask = np.ones((64, 64), dtype=bool)

    return CellKey(
        mask=mask,
        pool=pool,
        n_varying=int(varies.sum()),
        n_clock=int(clock.sum()),
    )
