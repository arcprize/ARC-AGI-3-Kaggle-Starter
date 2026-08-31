"""Go-Explore against the real, billed gateway - with RESET as the only rewind.

WHY THIS EXISTS
---------------
The stall report says the agent's problem in one line. On every game that scores
zero, ``imagine:noplan`` fires once per round, in lockstep with ``steer``, and
``cover:covered`` fires the same number of times:

    ka59  0/7 levels  steerx1673, imagine:noplanx1673, cover:coveredx1673
    cn04  0/6 levels  steerx2648, imagine:noplanx2648, cover:coveredx2647
    su15  0/9 levels  nosteerx2864, noobjectivex2842

Read it as English: *the whole reachable board has been walked, and there is
nothing to aim at.* ``Dream.objective`` returns ``None`` because no colour
ratchets during wandering, so the imagination has no gradient to descend, so the
round falls through to ``flail`` - a random walk - for the rest of the budget.

Meanwhile free search in the local twin clears level 0 on **17 of 24** games in
**4 to 35 actions** (``sweep.py``, 1200 s/game). So level 0 is not deep. The
agent is not failing for lack of a reachable answer; it is failing for lack of a
method that can *return to a promising state*. A random walk cannot: if the
interesting thing is twenty steps down a corridor, drifting away from it is the
overwhelmingly likely next event, and there is no way back.

THE REWIND IS REAL, AND IT IS NOT DEEPCOPY
------------------------------------------
``deepcopy`` is a local privilege - the 110 scored games ship no engine file, so
there is no object to clone. But two facts from ``graded.py`` combine into a
rewind that works through the gateway:

  * RESET restores the *current level* from its pristine clone and costs exactly
    one action (``handle_reset`` routes to ``level_reset`` once play has begun).
  * 23 of 25 games use no randomness at all and no game constructor takes a
    seed, so the same action prefix reproduces the same frames.

Therefore any state reachable by a plan ``p`` from the level's start can be
re-reached for ``1 + len(p)`` billed actions. That is Go-Explore's "return to
cell" step, priced. It is also what a person does: die, start the level again,
walk straight back to the bit you had not tried yet.

WHAT CHANGES BECAUSE THE REWIND IS BILLED
-----------------------------------------
Twin Go-Explore picks the most promising cell and teleports to it for free, so
selection ignores depth. Here every jump costs actions, which changes the rule to
*promise per action*:

    score = promise(node) / (1 + restart_cost(node))

and ``restart_cost`` is **zero for the node we are already standing on**. That
one term is what makes the search stick: it expands where it is until that place
is exhausted or unproductive, and only then pays to jump. The behaviour that
falls out is depth-first with cost-aware backtracking, which is the right shape
when a rewind is not free.

WHY THE CELL KEY IS NOT ``fingerprint(frame, live_mask)``
--------------------------------------------------------
The agent already keeps a novelty set, keyed on
``percept.fingerprint(frame, Volatility.live_mask)``, and it does nothing,
because ``hud_mask`` is a *frequency* test at 90%: a pixel has to change on
almost every action to be discarded. tn36's row-1 bar drains by six pixels per
action, but *which* six moves along the bar, so each individual pixel changes on
about 1/49 of frames and survives the filter. cd82's 154-pixel bar loses one
pixel per action - about 1/154. Both bars therefore enter the key, and with a
clock in the key every state is novel for ever. Measured in ``cell.py``: raw
frames gave 60 distinct keys in 60 steps on tn36.

``cell.py`` already has the rule that fixes it - a pixel that moves
**monotonically with the action count** is a clock, because state revisits values
and a counter never does. But ``cell.calibrate`` learns it from deepcopied probe
walks, which do not exist here. ``Clockless`` below learns the same rule from the
frames of ordinary billed play, at zero extra cost, by treating the stretch
between two RESETs as one walk.

The mask is frozen once, deliberately. ``progress.py`` records what happens
otherwise: a mask that is recomputed as evidence accumulates wiggles, every
wiggle invalidates the history, and no cell ever accumulates enough evidence to
matter - "ka59 ended a 240-step run with a perfectly empty ledger that way".

NOT A REPLACEMENT FOR THE MODEL
-------------------------------
This is the fallback, not the plan. ``Dream`` plus ``Mechanics`` is strictly
better when it has an objective, because it walks to a place it can prove is
better and does it in the fewest actions - and the score is quadratic in actions.
Search is what the agent does when the model has nothing to say, which today is
most of the time and on most games. Every action taken here still flows through
``Agent.act``, so ``Mechanics``, ``Dream`` and ``Progress`` keep learning from it;
a level cleared by blind search is a level whose model is then carried forward by
``on_new_level``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from arc3x.agent import Agent

GRID = 64
# A segment has to be long enough for "monotone" to mean anything. Two frames
# are monotone by definition, which would brand every moving pixel a clock.
MIN_SEG = 5
# Total frames of evidence before the mask is computed and frozen. Roughly the
# length of ``wiggle``, so the key is ready by the time the first round starts.
MIN_EVIDENCE = 28
# How many click coordinates to offer per cell. Each one is a billed action, so
# this is a real cost - but on the six click-only games it is the entire action
# space, which is why it is larger when there is nothing else to press.
CLICKS_WITH_MOVES = 8
CLICKS_ONLY = 20
# Above this many nodes the archive is pruned of exhausted leaves. Purely a
# memory bound; the search does not depend on it.
MAX_NODES = 20000
# Coarsening stops here. At pool 8 a 64x64 board is an 8x8 summary, which is
# about the granularity of a tile-based level - past that the key stops being able
# to tell one room from another, and merging cells that are genuinely different
# loses the routes between them.
MAX_POOL = 8
# Actions of evidence before the create-rate is trusted, and the rate above which
# the key is judged too fine. 0.7 rather than 1.0 because a walk that revisits
# nothing is already useless as novelty - it does not have to be perfectly
# bijective to be worthless.
COARSEN_AFTER = 96
COARSEN_RATE = 0.7


@dataclass
class Clockless:
    """Which pixels carry state, learned from billed play instead of probe walks.

    Same rule as ``cell.calibrate``: informative = *varied somewhere* and *not
    monotone in time within any stretch where it varied*. The difference is the
    evidence, which here is the frames the agent was going to see anyway.

    Held incrementally rather than as a stack of frames, so the whole thing is
    six 64x64 arrays regardless of how long the game runs.

    KNOWN ASYMMETRY, AND WHY IT IS THE RIGHT WAY ROUND
    A pixel that changes exactly *once* in a stretch is monotone by construction,
    so a collectible that is picked up and never comes back is filed as a clock and
    dropped from the key - which means "collected it" and "walked past it" become
    the same cell. That is a real loss of signal and it is not a bug to be argued
    away.

    It is nonetheless the safe direction. Over-rejecting costs novelty on one kind
    of event; under-rejecting puts a counter in the key, which is **measured** to
    make the key bijective and the search worthless (60 distinct keys in 60 steps).
    And the residual over-fineness has its own remedy in ``Relive.coarsen``, which
    watches the create-rate, whereas a counter in the key has none.

    The test that would separate the two properly is a group one: the pixels of a
    draining bar each change once, but *collectively* the bar changes on nearly
    every action, where ten collectibles change on a few percent of them. That is
    ``progress.py``'s CLOCK_RATE lifted from colour counts to pixels. Not
    implemented, because it has not been measured - ``why_cells.py`` is the
    instrument for that, and guessing at abstractions is how the last two silent
    zeros happened.
    """

    # Per-segment accumulators, reset by ``cut``.
    lo: np.ndarray | None = None
    hi: np.ndarray | None = None
    up: np.ndarray | None = None      # every diff so far was >= 0
    down: np.ndarray | None = None    # every diff so far was <= 0
    prev: np.ndarray | None = None
    seg_len: int = 0

    # Verdicts accumulated over closed segments.
    varies: np.ndarray = field(
        default_factory=lambda: np.zeros((GRID, GRID), dtype=bool)
    )
    clock: np.ndarray = field(
        default_factory=lambda: np.ones((GRID, GRID), dtype=bool)
    )
    n_frames: int = 0
    n_segments: int = 0
    mask: np.ndarray | None = None
    # Block size the informative pixels are max-pooled into before hashing. One
    # means "no pooling". Raised by ``Relive.coarsen`` when the archive shows the
    # key is too fine to be useful - see the note there. Not a tuned constant: the
    # game's own behaviour decides it.
    pool: int = 1

    def feed(self, frame: np.ndarray) -> None:
        f = frame.astype(np.int16)
        self.n_frames += 1
        if self.prev is None:
            self.lo = f.copy()
            self.hi = f.copy()
            self.up = np.ones((GRID, GRID), dtype=bool)
            self.down = np.ones((GRID, GRID), dtype=bool)
            self.prev = f
            self.seg_len = 1
            return
        d = f - self.prev
        self.up &= d >= 0
        self.down &= d <= 0
        np.minimum(self.lo, f, out=self.lo)
        np.maximum(self.hi, f, out=self.hi)
        self.prev = f
        self.seg_len += 1

    def cut(self) -> None:
        """Close the current stretch. Called on a RESET, a death or a new level.

        The cut matters: a RESET restores the counter too, so a clock read across
        a reset is *not* monotone and would be mistaken for state. Every stretch
        between two resets is one walk, exactly as ``cell.calibrate`` treats one
        probe.
        """
        if self.prev is not None and self.seg_len >= MIN_SEG:
            assert self.lo is not None and self.hi is not None
            assert self.up is not None and self.down is not None
            v = self.hi != self.lo
            mono = self.up | self.down
            self.varies |= v
            # A pixel that did not move in this stretch says nothing about
            # clock-ness here, so it must not veto the verdict from other
            # stretches.
            self.clock &= mono | ~v
            self.n_segments += 1
        self.lo = self.hi = self.up = self.down = self.prev = None
        self.seg_len = 0

    @property
    def ready(self) -> bool:
        return self.mask is not None or self.n_frames >= MIN_EVIDENCE

    def freeze(self) -> bool:
        """Settle the mask now. True if this call is the one that settled it.

        The caller needs to know, because a key computed under a different mask
        is a different key: nodes archived before the freeze can never be matched
        again, and every attempt to return to one bills a RESET. Measured on ka59
        before this existed - 481 of 500 actions in one search were restarts to
        cells that had been keyed on the raw frame and no longer existed.
        """
        first = self.mask is None
        self._freeze()
        return first

    def _freeze(self) -> np.ndarray:
        if self.mask is not None:
            return self.mask
        # Fold in whatever the open stretch already knows, without closing it -
        # otherwise a game that never resets never gets a mask.
        varies = self.varies.copy()
        clock = self.clock.copy()
        if self.prev is not None and self.seg_len >= MIN_SEG:
            assert self.lo is not None and self.hi is not None
            assert self.up is not None and self.down is not None
            v = self.hi != self.lo
            varies |= v
            clock &= (self.up | self.down) | ~v
        clock &= varies
        m = varies & ~clock
        if not m.any():
            m = varies.copy()
        if not m.any():
            m = np.ones((GRID, GRID), dtype=bool)
        self.mask = m
        return m

    def key(self, frame: np.ndarray, level: int) -> bytes:
        """Hash the informative pixels plus the level index.

        Before there is enough evidence the whole frame is used, which is the
        conservative choice: an over-fine key wastes a few early actions, an
        over-coarse one merges two states that are not the same and loses the
        route between them.

        Non-informative pixels are blanked to -1 *before* pooling, so pooling
        coarsens the informative mask rather than the raw frame. Pooling the raw
        frame would let a clock inside a block dominate the max and quietly
        reintroduce exactly what the mask was built to remove.
        """
        f = np.asarray(frame, dtype=np.int8)
        if self.ready:
            f = f.copy()
            f[~self._freeze()] = -1
        p = self.pool
        if p > 1:
            h, w = f.shape
            hh, ww = h // p * p, w // p * p
            f = f[:hh, :ww].reshape(hh // p, p, ww // p, p).max(axis=(1, 3))
        return hashlib.blake2b(
            f.tobytes() + bytes((level & 0xFF, p & 0xFF)), digest_size=16
        ).digest()

    def summary(self) -> str:
        m = self.mask
        n = int(m.sum()) if m is not None else -1
        return (
            f"clockless informative={n} varies={int(self.varies.sum())} "
            f"pool={self.pool} segments={self.n_segments} frames={self.n_frames}"
        )


Step = tuple[int, int, int]  # (aid, x, y)


@dataclass
class Node:
    """One archived situation, and the cheapest known way back to it."""

    plan: tuple[Step, ...]
    frame: np.ndarray
    level: int = 0
    # ``None`` means "not enumerated yet". Deliberately lazy: most cells are added
    # and never chosen, and enumerating one costs a blob segmentation of the whole
    # frame. Measured eagerly on cn04 - 800 billed actions took 165 seconds, almost
    # all of it segmenting 222 frames that were never expanded.
    untried: list[Step] | None = None
    # Every step ever offered here, so ``deepen`` can widen the click list without
    # re-offering what has already been paid for.
    done: set[Step] = field(default_factory=set)
    visits: int = 0
    # Expansions from here that landed in a cell never seen before. The signal
    # that says "this place is a frontier" as opposed to "this place is a dead
    # end I keep paying to revisit".
    novel: int = 0
    tries: int = 0
    # Times a RESET-and-replay of ``plan`` failed to arrive here. A cell we cannot
    # get back to is worthless however promising it looks, and it is *expensive*:
    # each failed attempt bills 1 + depth. Measured on ka59 before this counter
    # existed - 481 of 500 actions in one search were failed restarts to the same
    # unreachable cell.
    misses: int = 0

    @property
    def depth(self) -> int:
        return len(self.plan)

    @property
    def promise(self) -> float:
        if self.misses >= 2:
            return 0.0
        if self.untried is not None and not self.untried:
            return 0.0
        # Laplace-smoothed novelty rate, so an unexpanded node is optimistic and
        # a node whose children were all stale decays without ever hitting zero.
        rate = (self.novel + 1.0) / (self.tries + 2.0)
        n = 8 if self.untried is None else len(self.untried)
        return rate * (1.0 + n) ** 0.5 / (1.0 + self.visits) ** 0.5


class Relive:
    """The restart-replay archive search, driven through one ``Agent``."""

    def __init__(self, agent: "Agent") -> None:
        self.a = agent
        self.ck = Clockless()
        self.arch: dict[bytes, Node] = {}
        self.plan: tuple[Step, ...] = ()
        self.at: Node | None = None
        # (plan-so-far, step) pairs that killed us. Recorded by plan prefix and
        # not by cell, because a death wipes the cell we were standing in.
        self.fatal: set[tuple[tuple[Step, ...], Step]] = set()
        self.restarts = 0
        self.replayed = 0
        self.drift = 0
        self.cells = 0
        # How many click coordinates a cell offers. Starts small because each one
        # is a billed action, and doubles only when the narrow archive has been
        # fully expanded - so breadth is paid for out of evidence that the cheap
        # version is exhausted, never up front.
        self.click_k = 0
        # Cells created and actions spent since the last coarsening. Their ratio is
        # the whole self-calibration: see ``coarsen``.
        self.made = 0
        self.took = 0
        self.coarsenings = 0

    # -- bookkeeping the Agent calls ---------------------------------------

    def observe(self, frame: np.ndarray, step: Step | None = None) -> None:
        """One billed action that did not restore the board, from *anywhere*.

        Called by ``Agent.act``, which is the only place that sees every action,
        so ``self.plan`` is a true recording of the prefix since the last
        restoration no matter which strategy spent the action. That is what makes
        a route trustworthy, and it is worth more than the bookkeeping saved:
        the walking, imagining and clicking branches now archive cells the search
        can return to, for free, and the search never has to trust a plan it did
        not build. Before this, a ``run_level`` that followed a round of ordinary
        play recorded the current frame under whatever stale plan was left over,
        and every attempt to return to it billed a RESET and arrived elsewhere.

        Archiving waits for the mask, because a key computed under the
        provisional whole-frame key is a different key and could never be
        matched again - the same reason ``run_level`` clears the archive the
        moment the mask settles.
        """
        self.ck.feed(frame)
        if step is not None:
            self.plan = self.plan + (step,)
        if self.ck.mask is not None:
            self.at, _ = self._add(
                self.ck.key(frame, self.a.level), self.plan, frame
            )

    def cut(self) -> None:
        """A RESET, a death or a new level: the stretch ends and we are at the top."""
        self.ck.cut()
        self.plan = ()
        self.at = None

    def new_level(self) -> None:
        """Fresh board, fresh archive. The keys carry the level index anyway, but
        the *plans* do not survive - a plan is a route from this level's start."""
        self.cut()
        self.arch.clear()
        self.fatal.clear()

    # -- the action space of a cell ---------------------------------------

    def steps_at(self, frame: np.ndarray) -> list[Step]:
        """What is worth trying from here, best first.

        Movement and use buttons come first because they cost one action each and
        there are at most six of them. Clicks are ranked by
        ``Agent.click_candidates``, which is blob-derived and already knows which
        spots have been shown to do nothing - so the truncation below drops the
        least promising coordinates, not arbitrary ones.
        """
        declared = list(self.a.run._declared)
        simple = [(a, 0, 0) for a in declared if a != 6]
        if 6 not in declared:
            return simple
        if not self.click_k:
            self.click_k = CLICKS_ONLY if not simple else CLICKS_WITH_MOVES
        cands = self.a.click_candidates(frame)[: self.click_k]
        return simple + [(6, int(x), int(y)) for (y, x) in cands]

    def deepen(self) -> int:
        """The archive dried up. Widen every cell's click list and try again.

        The alternative when nothing is left untried is a random walk, and a
        random walk cannot return to a cell. Doubling the offered coordinates
        re-opens every cell in the archive at once, which is strictly more
        informative per action - the frames are already known, so the *ranking* of
        the new coordinates is as good as the old ones were.

        Returns how many cells came back to life; zero means clicking really is
        exhausted and only randomness is left.
        """
        if 6 not in self.a.run._declared:
            return 0
        self.click_k = min(256, max(1, self.click_k) * 2)
        woken = 0
        for n in self.arch.values():
            if n.untried is None:
                continue
            fresh = [s for s in self.steps_at(n.frame) if s not in n.done]
            if fresh:
                n.untried.extend(fresh)
                n.done.update(fresh)
                woken += 1
        return woken

    # -- archive ----------------------------------------------------------

    def _add(self, key: bytes, plan: tuple[Step, ...], frame: np.ndarray) -> tuple[Node, bool]:
        node = self.arch.get(key)
        if node is None:
            # int8 because colours are 0..15, and the archive holds thousands of
            # these: at the frame's native width that would be hundreds of
            # megabytes for no extra information. Kept for every cell because
            # ``coarsen`` has to re-key the whole archive, and re-keying without
            # the frame would mean throwing the routes away.
            node = Node(
                plan=plan,
                frame=np.asarray(frame, dtype=np.int8).copy(),
                level=self.a.level,
            )
            self.arch[key] = node
            self.cells += 1
            if len(self.arch) > MAX_NODES:
                self._prune()
            return node, True
        if len(plan) < node.depth:
            # A cheaper route to a place we already know. Worth keeping: every
            # future visit to this cell is billed by the length of this plan, and
            # the shorter route may also be reachable where the old one is not.
            node.plan = plan
            node.misses = 0
        return node, False

    def _ensure(self, node: Node, frame: np.ndarray) -> list[Step]:
        """Enumerate this cell's action list, the first time it is actually used."""
        if node.untried is None:
            steps = self.steps_at(frame)
            node.untried = steps
            node.done = set(steps)
        return node.untried

    def coarsen(self) -> bool:
        """The key is too fine to carry information. Merge cells, keep the routes.

        Novelty is only a signal if many action sequences land in the same cell.
        The rate at which cells are *created per action spent* measures that
        directly, and it needs no per-game knowledge: a rate near 1.0 means every
        state looks new, which is the failure ``cell.py`` documents and which was
        measured again here after wiring the search in -

            ka59  informative=514   128 cells in ~430 actions
            bp35  informative=1542  227 cells
            cn04  informative=1890  295 cells

        so the pool size is not a constant to be tuned, it is a thing the game
        tells us. Doubling it merges each 2x2 block of informative pixels, which
        removes sub-tile rendering detail - a sprite's facing direction, an
        animation phase - before it removes anything positional.

        The archive is **re-keyed rather than cleared**, keeping the shortest plan
        for each merged cell. Clearing would be much simpler and much worse: the
        plans are the only thing that makes the rewind affordable, and they are
        still valid, because a coarser key does not change what an action does.
        Progress counters (``novel``, ``tries``, ``visits``) are summed into the
        survivor, and ``untried`` is intersected - a step already paid for at one
        of the merged cells should not be paid for again.
        """
        if self.ck.pool >= MAX_POOL:
            return False
        self.ck.pool *= 2
        self.coarsenings += 1
        merged: dict[bytes, Node] = {}
        for n in self.arch.values():
            k = self.ck.key(n.frame, n.level)
            keep = merged.get(k)
            if keep is None:
                merged[k] = n
                continue
            # Deeper of the two loses its plan but not its evidence.
            if n.depth < keep.depth:
                keep, n = n, keep
                merged[k] = keep
            keep.novel += n.novel
            keep.tries += n.tries
            keep.visits += n.visits
            keep.misses = min(keep.misses, n.misses)
            if keep.untried is None:
                keep.untried = n.untried
                keep.done = n.done
            elif n.untried is not None:
                already = keep.done
                keep.untried.extend(s for s in n.untried if s not in already)
                keep.done = already | n.done
        self.arch = merged
        self.made = 0
        self.took = 0
        self.at = None
        return True

    def _prune(self) -> None:
        """Memory bound only. Exhausted leaves first, then the most expensive.

        Dropping exhausted leaves usually suffices, but it can free nothing at
        all: a cell that has never been expanded is optimistic by construction,
        so a long run of ordinary play archives twenty thousand cells that are
        every one of them 'promising'. The fallback drops the *deepest* of those,
        because depth is exactly the price of ever using one - a cell 400 actions
        down costs 401 to reach and will never win ``_choose``.
        """
        for k in [k for k, n in self.arch.items() if n.promise <= 0.0]:
            self.arch.pop(k, None)
        if len(self.arch) <= MAX_NODES:
            return
        order = sorted(self.arch.items(), key=lambda kv: -kv[1].depth)
        for k, n in order[: len(self.arch) - MAX_NODES]:
            if n is not self.at:
                self.arch.pop(k, None)

    def _choose(self) -> Node | None:
        """Promise per billed action. Standing still is free, so it wins ties."""
        best: Node | None = None
        best_score = 0.0
        for n in self.arch.values():
            p = n.promise
            if p <= 0.0:
                continue
            cost = 0 if n is self.at else 1 + n.depth
            s = p / (1.0 + cost)
            if s > best_score:
                best, best_score = n, s
        return best

    # -- the rewind -------------------------------------------------------

    def _goto(self, node: Node) -> bool:
        """RESET and replay ``node.plan``. Returns False if we ended up elsewhere.

        Ending up elsewhere is not an error - two of the 25 games touch ``random``,
        and a plan can also stop working because the level advanced under it - it is
        just information, so the archive is told where we actually are and the
        caller expands that instead of insisting on the plan.
        """
        a = self.a
        if any((node.plan[:i], st) in self.fatal for i, st in enumerate(node.plan)):
            # This plan is known to die partway through, so replaying it buys a
            # RESET, some actions and the same death. ``misses`` is the counter for
            # "we cannot get back here", and that is exactly what this is - set it
            # to the threshold rather than invent a second flag for the same fact.
            node.misses = 2
            return False
        self.restarts += 1
        a.obs = a.run.reset()
        if a.obs.full_reset:
            # Back at the start of the whole *game*, not this level. `handle_reset`
            # only does that at action 0 or after a WIN, so it should be
            # unreachable from here - but if it ever is, every plan in the archive
            # is a route from a level that is no longer loaded, and `Agent.level`
            # would silently disagree with the engine because it only ever moves
            # up. Resync and throw the archive away rather than search a fiction.
            a.level = a.obs.levels_completed
            a.on_new_level()
            return False
        # ``on_restored`` cuts the evidence stretch, clears ``plan`` and archives
        # the level-start cell, because a RESET restores the board and the counter
        # with it. From here every ``act`` extends ``plan`` and moves ``at``.
        a.on_restored()
        lv = a.level
        for i, st in enumerate(node.plan):
            if a.spent:
                return False
            obs = a.act(*st)
            self.replayed += 1
            if obs.won or a.level != lv:
                # The replay itself finished the level. The archive has already
                # been rebuilt by ``on_new_level``; ``node`` no longer exists.
                return False
            if obs.game_over:
                # The replay died where it did not die before, so the plan is not
                # a plan any more. ``Agent.act`` has already paid for the reset and
                # put us back at the level start.
                self.fatal.add((node.plan[:i], st))
                node.misses += 1
                return False
        landed = self.at
        if landed is not node:
            self.drift += 1
            node.misses += 1
            return False
        node.misses = 0
        return True

    # -- the loop ---------------------------------------------------------

    def run_level(self, max_actions: int = 400) -> str:
        """Search this level until it breaks, the archive dries up, or budget ends.

        Returns 'level', 'win', 'dry' (nothing left untried anywhere) or 'budget'.
        """
        a = self.a
        assert a.obs is not None
        if not self.ck.ready:
            # Not enough frames yet for a key that is anything but the raw frame,
            # and a raw-frame key makes every state novel for ever.
            return "dry"
        if self.ck.freeze():
            # First real mask. Anything archived under the provisional key is
            # unreachable now, so it is worse than nothing: it would be chosen,
            # paid for with a RESET, and never arrived at.
            self.arch.clear()
            self.fatal.clear()
            self.at = None
        used0 = a.run.actions
        lv0 = a.level
        self.at, _ = self._add(self.ck.key(a.obs.frame, a.level), self.plan, a.obs.frame)

        while not a.spent and a.run.actions - used0 < max_actions:
            # Self-calibration, checked before anything is spent on the next step:
            # if almost every action has been minting a brand new cell, the key is
            # carrying no information and coarsening it is worth more than any
            # amount of further searching under it.
            if self.took >= COARSEN_AFTER and self.made >= COARSEN_RATE * self.took:
                if self.coarsen():
                    self.at, _ = self._add(
                        self.ck.key(a.obs.frame, a.level), self.plan, a.obs.frame
                    )
                else:
                    # Already as coarse as it goes; stop re-testing every step.
                    self.took = self.made = 0
            node = self._choose()
            if node is None:
                # Nothing left untried anywhere. Widening the click list re-opens
                # the archive; if even that finds nothing new, clicking really is
                # exhausted and the caller should try something else entirely.
                if not self.deepen():
                    return "dry"
                continue
            if node is not self.at:
                if not self._goto(node):
                    if a.obs is not None and a.obs.won:
                        return "win"
                    if a.level != lv0:
                        return "level"
                    continue
                node = self.at
                assert node is not None

            assert a.obs is not None
            untried = self._ensure(node, a.obs.frame)
            if not untried:
                continue
            st = untried.pop(0)
            # The true action prefix, which after a drift or a shorter route may
            # be longer than ``node.plan``. Deaths are recorded against what we
            # actually did, because that is what would have to be repeated.
            here = self.plan
            if (here, st) in self.fatal:
                continue
            node.visits += 1
            node.tries += 1
            self.took += 1
            before = self.cells
            # ``Agent.act`` extends ``plan``, archives where we landed and moves
            # ``at`` - see ``observe``. Nothing to do here but read the outcome.
            obs = a.act(*st)
            if obs.won:
                return "win"
            if a.level != lv0:
                return "level"
            if obs.game_over:
                # ``Agent.act`` already paid for the reset, and ``on_restored``
                # put us back at the level start with an empty plan.
                self.fatal.add((here, st))
                continue
            if self.cells > before:
                node.novel += 1
                self.made += 1
        return "budget"

    def summary(self) -> str:
        live = sum(1 for n in self.arch.values() if n.promise > 0.0)
        deep = max((n.depth for n in self.arch.values()), default=0)
        return (
            f"relive cells={self.cells} held={len(self.arch)} live={live} "
            f"deepest={deep} restarts={self.restarts} replay={self.replayed} "
            f"drift={self.drift} coarsen={self.coarsenings} | {self.ck.summary()}"
        )
