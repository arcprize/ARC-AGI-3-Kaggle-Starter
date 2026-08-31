"""The game inside the head: a learned copy you can play without spending actions.

This is the piece the rest of the agent was missing. ``mind.Mechanics`` learns
"which button moves me and what blocks me", which is enough to walk but not to
*think*. Thinking requires a runnable copy of the game - something you can hand a
frame and a button and get back the frame that would follow - because that is what
lets you try a hundred ideas before committing to one.

Why this is the whole economic argument. Every action in the world is billed and
the score goes as (baseline/actions)^2, so a wrong move is paid for twice: once in
actions and once in the ratio. Every action in the *dream* is free. So the agent
should spend its real actions on two things only - learning the dream, and
executing a plan the dream says will work.

The three things it has to know to be a useful copy, and all three are learned by
watching, never hardcoded:

  ``walk``     where the sprite may stand. From ``Mechanics.passable``.
  ``push``     what the sprite shoves. Learned from transitions where some other
               object translated by exactly the sprite's own delta on the same
               action - that is what being pushed looks like from outside.
  ``collect``  what disappears when the sprite touches it. Learned from
               transitions where an object vanished from the cells the sprite
               moved into.

``predict`` is deliberately allowed to answer **None** meaning "I do not know what
happens here". An imagination that confidently makes things up is worse than no
imagination at all, because the agent will spend real actions executing fiction.
``accuracy`` measures exactly this and is the number that says whether planning
inside the dream is trustworthy yet.

The loop, which is the human one:

    dream.observe(...)                  # watch, and correct the copy when wrong
    plan = dream.think(frame, goal)     # play it out in your head, free
    for a in plan: run.step(a)          # only now spend actions
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from arc3x.mind import Mechanics
from arc3x.percept import Volatility, mask_component, moved_objects
from arc3x.progress import Progress

MAX_IMAGINE = 4096
# One unit of "something got collected" is worth more than any possible amount of
# "something got assembled". 4096 is the whole board, so no assembly term can ever
# reach across a single collection step. Keeping them in one integer lets the
# breadth-first search compare boards without a second criterion.
BUILT_SCALE = 4096


@dataclass
class Dream:
    """A runnable, self-correcting copy of the game being played."""

    m: Mechanics
    # colour -> times an object of that colour translated exactly as we did
    push: Counter = field(default_factory=Counter)
    # colour -> times an object of that colour vanished as we stepped onto it
    collect: Counter = field(default_factory=Counter)
    # colour -> times it was there, we stepped on it, and it stayed put
    inert: Counter = field(default_factory=Counter)
    # Which pixels are a ticking counter rather than the world. Every game here
    # has some HUD, ls20 literally has a step-counter widget, and those pixels
    # change on every single action. Grading a prediction against them means
    # being wrong forever - which is exactly what the first measurement showed:
    # 0% accuracy at 0% abstention. This repo has been bitten by that before,
    # when HUD pixels made every frame unique and silently killed Go-Explore.
    vol: Volatility = field(default_factory=Volatility)
    # What counts as getting closer to done, learned from what ratchets rather
    # than from a completed level. Without this the imagination has no
    # destination on any game that is not "eat the gems", and a route with no
    # destination is no route: dc22 and ka59 predict perfectly and plan nothing.
    prog: Progress = field(default_factory=Progress)
    # Colours the agent chased to exhaustion and got nothing for. The ratchet can
    # only say "this is being used up", never "this is what the game wants", and
    # some games use something up as a side effect of play. Measured: one game
    # drains a colour from 63 pixels to *zero* and then dies, another converts one
    # colour into another two pixels per click for hundreds of actions. Both are
    # real monotone progress on a quantity that is not the win condition.
    #
    # So there has to be a way to be wrong and recover. A colour whose count hit
    # the floor with no level to show for it is definitively not the objective, and
    # continuing to push it is the most expensive mistake available - the agent has
    # a destination, believes in it, and walks there for the rest of the budget.
    retired: set[int] = field(default_factory=set)
    # honesty ledger: was the copy right? Split by whether anything actually
    # happened, because a game where the avatar is usually blocked hands out a
    # free 100% to any model that predicts "nothing changed" - ka59 scored
    # exactly that. Only ``acc_move`` says whether the copy can be planned in.
    hits_move: int = 0
    misses_move: int = 0
    hits_still: int = 0
    misses_still: int = 0
    abstains: int = 0
    # transitions where something moved that we neither moved nor shoved
    lively: int = 0

    # -- learning the copy -------------------------------------------------

    @property
    def known(self) -> set[int]:
        """Colours the model has an actual opinion about.

        Everything the agent has classified: ground it has stood on, things that
        refused it, its own body, things it can shove or pick up, and the
        background. A colour outside this set is something the agent has never
        interacted with and has no business predicting.
        """
        out = set(self.m.passable) | set(self.m.blocking) | set(self.m.fatal)
        out |= self.m.body or ({self.m.avatar} if self.m.avatar >= 0 else set())
        out |= self.pushable | self.collectible
        if self.m.background >= 0:
            out.add(self.m.background)
        return out

    def project(self, frame: np.ndarray) -> tuple[tuple[int, int] | None, int]:
        """The only part of a frame a plan actually depends on.

        Reading the dev games' source settles what a general forward model can
        and cannot be held to. Every game decorates its frame with things no
        general model will ever reproduce:

          * ``wa30`` calls ``set_rotation`` on every move, so the avatar is
            redrawn *facing* the direction of travel, and then repaints the
            borders of nearby sprites to advertise what is interactable.
          * ``ka59`` marks which of several identical blocks is under control by
            recolouring its centre pixel, and runs an NPC that takes six
            pathfinding steps of its own after each of ours.
          * both drain an on-screen step counter every single action.

        Demanding an exact frame match makes the copy wrong forever on all of
        that - which is what the 5% measurement was really reporting. The copy
        had the *movement* right and the decoration wrong. Worse, an exact-frame
        contract makes multi-step imagination impossible in principle, because
        step two reads back a frame whose decoration is already stale.

        So the copy commits to what a plan is made of and nothing else: where
        the avatar is, and how much of the target remains. Rotation, selection
        markers, affordance highlights and counters become irrelevant by
        construction rather than by masking - the same reason a person can plan
        a route through a room without imagining the wallpaper.
        """
        box = self.m.locate(frame, hint=self.m.pos)
        pos = (box[0], box[1]) if box is not None else None
        want = self.collectible
        left = int(np.isin(frame, list(want)).sum()) if want else 0
        return pos, left

    def _grade(
        self,
        pred: np.ndarray | None,
        actual: np.ndarray,
        pos0: tuple[int, int] | None,
    ) -> str:
        """``hit``/``miss`` on a transition where we moved, or where we did not.

        Abstaining when the *truth* is unreadable matters as much as abstaining
        when the prediction is: if the avatar cannot be found in the real frame
        either, scoring the two ``None``s as a match would report confidence
        the copy has not earned.
        """
        if pred is None:
            return "abstain"
        truth = self.project(actual)
        if truth[0] is None:
            return "abstain"
        ok = self.project(pred) == truth
        moved = truth[0] != pos0
        if moved:
            if ok:
                self.hits_move += 1
            else:
                self.misses_move += 1
        elif ok:
            self.hits_still += 1
        else:
            self.misses_still += 1
        return "hit" if ok else "miss"

    def agree(self, before: np.ndarray, pred: np.ndarray, actual: np.ndarray) -> bool:
        """Did the copy get the plannable part of the world right?"""
        return self._grade(pred, actual, self.project(before)[0]) == "hit"

    def cut(self) -> None:
        """A level changed or was reset, so the board has been restored.

        The ratchet evidence survives; only the frame-to-frame comparison is
        broken, because otherwise the restoration reads as a huge increase and
        every genuine collectible gets rejected.
        """
        self.prog.cut()

    def observe(self, action: int, before: np.ndarray, after: np.ndarray) -> None:
        """Watch one real transition and correct the copy where it was wrong.

        Called *before* ``Mechanics.observe``, so the prediction being graded is
        the one the agent would actually have acted on.
        """
        pred = self.predict(before, action)
        verdict = self._grade(pred, after, self.project(before)[0])
        if verdict == "abstain":
            self.abstains += 1
        self.vol.add(before, after)
        hud = self.vol.hud_mask(thresh=0.75)

        # What ratchets is the objective, and it has to be read from every frame,
        # including the ones where we cannot even find ourselves - a game whose
        # avatar we have not identified is exactly the game that most needs a goal.
        self.prog.ignore = (self.m.body or set()) | (
            {self.m.background} if self.m.background >= 0 else set()
        )
        self.prog.add(
            after, self.vol.hud_mask(thresh=0.5), observed=self.vol.observed
        )

        d = self.m.deltas.get(action)
        if d is None or self.m.avatar < 0 or d == (0, 0):
            return
        box = self.m.locate(before, hint=self.m.pos)
        if box is None:
            return
        mv, van, _app = moved_objects(before, after)
        body = self.m.body or {self.m.avatar}
        # Anything inside the HUD is a digit changing, not an object doing
        # something, so it must not teach the copy about pushing or collecting.
        van = [x for x in van if not hud[x.top, x.left]]
        mv = [x for x in mv if not hud[x.top, x.left]]
        # Did we actually move? Only then can anything have been pushed by us.
        we_moved = any(x.color in body and x.delta == d for x in mv)
        # Something moved that we neither moved nor shoved, so the world has
        # actors of its own - ka59 runs an NPC six pathfinding steps per turn.
        # A long imagined route is stale before it can be walked, so this is
        # counted and later caps how deep the imagination is allowed to go.
        if any(x.moved and x.color not in body and x.delta != d for x in mv):
            self.lively += 1
        if we_moved:
            for x in mv:
                if x.color in body or not x.moved:
                    continue
                if x.delta == d:
                    # It went exactly where we went, on the button that moves us.
                    # From the outside, that is what "I pushed it" looks like.
                    self.push[x.color] += 1
            cells, swept = self._swept(before, box, d)
            # A thing was collected only if it vanished *from a cell we walked
            # into*. Matching on colour alone made ls20 conclude that the wall
            # was collectible, because a HUD digit of the same colour happened to
            # disappear on the same action.
            gone = {x.color for x in van if (x.top, x.left) in cells}
            for c in swept:
                if c in gone:
                    self.collect[c] += 1
                elif c not in self.push:
                    self.inert[c] += 1

    def _swept(self, frame: np.ndarray, box, d) -> tuple[set, set[int]]:
        """The cells our footprint moves into, and the colours sitting in them."""
        t, l, h, w = box
        dy, dx = d
        H, W = frame.shape
        foot = self.m.footprint(frame, box)
        body = self.m.body or {self.m.avatar}
        cells: set[tuple[int, int]] = set()
        cols: set[int] = set()
        ys, xs = np.nonzero(foot)
        for y, x in zip(ys.tolist(), xs.tolist()):
            ny, nx = t + y + dy, l + x + dx
            if not (0 <= ny < H and 0 <= nx < W):
                continue
            fy, fx = ny - t, nx - l
            if 0 <= fy < h and 0 <= fx < w and foot[fy, fx]:
                continue
            c = int(frame[ny, nx])
            if c not in body:
                cells.add((ny, nx))
                cols.add(c)
        return cells, cols

    # -- what the copy believes -------------------------------------------

    @property
    def pushable(self) -> set[int]:
        return {c for c, n in self.push.items() if n >= 1}

    @property
    def collectible(self) -> set[int]:
        """Colours that vanish on contact - and are therefore worth touching."""
        return {
            c
            for c, n in self.collect.items()
            if n >= 1 and n > self.inert.get(c, 0)
        }

    @property
    def hits(self) -> int:
        return self.hits_move + self.hits_still

    @property
    def misses(self) -> int:
        return self.misses_move + self.misses_still

    @property
    def acc_move(self) -> float:
        """Accuracy on transitions where the avatar actually went somewhere.

        This is the number that decides whether a route can be trusted. Overall
        accuracy is dominated by whichever case is more common, and on a game
        where most moves are refused that case is "nothing happened" - so the
        copy can look perfect while knowing nothing about getting anywhere.
        """
        g = self.hits_move + self.misses_move
        return self.hits_move / g if g else 0.0

    @property
    def acc_still(self) -> float:
        """Accuracy on refusals: did it know which moves were impossible?"""
        g = self.hits_still + self.misses_still
        return self.hits_still / g if g else 0.0

    @property
    def calm(self) -> bool:
        """Does the world only move when we move it?

        A game with its own actors can still be played, but not by planning a
        long route and walking it blind - by the time step five arrives the NPC
        has moved thirty times. Detecting this is what lets the agent choose
        between "commit to a route" and "take one step and look again", which is
        the same judgement a person makes on seeing something else move.
        """
        seen = self.hits + self.misses + self.abstains
        return seen < 8 or self.lively <= 0.15 * seen

    @property
    def confident(self) -> bool:
        """Is the copy good enough to plan inside?

        Gated on ``acc_move``, not on overall accuracy. A route is a claim about
        where the avatar ends up, so being right about refusals is not evidence
        that a route will work - and on a game where most moves are refused,
        overall accuracy is almost entirely refusals.
        """
        moved = self.hits_move + self.misses_move
        return moved >= 8 and self.hits_move >= 0.8 * moved

    @property
    def accuracy(self) -> float:
        graded = self.hits + self.misses
        return self.hits / graded if graded else 0.0

    # -- the imagination ---------------------------------------------------

    def predict(self, frame: np.ndarray, action: int) -> np.ndarray | None:
        """The frame that would follow. ``None`` means "I genuinely don't know".

        Abstaining is a feature. The agent uses the dream to decide where to
        spend billed actions, so a copy that invents an answer costs real score.
        Anything outside the learned repertoire - a click, an unknown button, a
        frame where the sprite cannot be found - returns None rather than a guess.
        """
        d = self.m.deltas.get(action)
        if d is None or self.m.avatar < 0:
            return None
        box = self.m.locate(frame, hint=self.m.pos)
        if box is None:
            return None
        if d == (0, 0):
            return frame.copy()

        t, l, h, w = box
        dy, dx = d
        H, W = frame.shape
        foot = self.m.footprint(frame, box)
        if not foot.any():
            return None
        walk = self.m.walk_mask(frame)
        pushable = self.pushable
        collectible = self.collectible

        # What our own pixels move into, cell by cell. Doing this per pixel rather
        # than per bounding box is what makes a non-rectangular sprite work.
        pushed: dict[int, list[tuple[int, int]]] = {}
        ys, xs = np.nonzero(foot)
        src = [(t + int(y), l + int(x)) for y, x in zip(ys, xs)]
        for y, x in src:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < H and 0 <= nx < W):
                return frame.copy()  # walked into the edge: nothing happens
            if (ny - t, nx - l) == (y - t, x - l):
                continue
            fy, fx = ny - t, nx - l
            if 0 <= fy < h and 0 <= fx < w and foot[fy, fx]:
                continue  # our own tail vacates it
            c = int(frame[ny, nx])
            if c in collectible:
                continue  # it will be picked up
            if c in pushable:
                pushed.setdefault(c, []).append((ny, nx))
                continue
            if not walk[ny, nx]:
                return frame.copy()  # blocked: the world does not change

        out = frame.copy()
        bg = self.m.background if self.m.background >= 0 else 0

        # Shove each pushed object one step, if there is room behind it. A crate
        # against a wall stops us, which is why this can still return "no change".
        for c, cells in pushed.items():
            for cy, cx in cells:
                sub, ot, ol, _n = mask_component(frame == c, cy, cx)
                oys, oxs = np.nonzero(sub)
                dest = [(ot + int(y) + dy, ol + int(x) + dx) for y, x in zip(oys, oxs)]
                own = {(ot + int(y), ol + int(x)) for y, x in zip(oys, oxs)}
                for py, px in dest:
                    if not (0 <= py < H and 0 <= px < W):
                        return frame.copy()
                    if (py, px) in own:
                        continue
                    if not walk[py, px] and int(frame[py, px]) not in collectible:
                        return frame.copy()
                for py, px in own:
                    out[py, px] = bg
                for py, px in dest:
                    out[py, px] = c

        # Collect whatever we are stepping onto: the whole object goes, not just
        # the pixel we touched, which is what "picking it up" means.
        for y, x in src:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < H and 0 <= nx < W):
                continue
            c = int(frame[ny, nx])
            if c in collectible:
                sub, ot, ol, _n = mask_component(frame == c, ny, nx)
                region = out[ot : ot + sub.shape[0], ol : ol + sub.shape[1]]
                region[sub] = bg

        # Finally move ourselves.
        cols = frame[foot.nonzero()[0] + t, foot.nonzero()[1] + l]
        for (y, x) in src:
            out[y, x] = bg
        for (y, x), c in zip(src, cols.tolist()):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                out[ny, nx] = int(c)
        return out

    def rollout(self, frame: np.ndarray, actions: list[int]) -> np.ndarray | None:
        """Play a whole sequence out in the head. None if the copy loses track."""
        cur = frame
        for a in actions:
            nxt = self.predict(cur, a)
            if nxt is None:
                return None
            cur = nxt
        return cur

    # -- thinking, i.e. searching inside the copy --------------------------

    def objective(self, frame: np.ndarray) -> int | None:
        """Distance from done. Lower is better; ``None`` means "no idea yet".

        Two sources, both learned by watching and neither needing a completed
        level. The narrow one is ``collectible`` - things seen to vanish under
        the avatar's own feet. The general one is ``Progress`` - any colour whose
        count ratchets. The union matters because they fail on opposite games:
        contact-collection is silent on a game where nothing is picked up, and
        the ratchet is silent until a few frames have been watched.

        Distinguishing None from 0 is the point. "Everything is equally good" and
        "I do not know what good means" call for opposite behaviour, and only the
        second should stop the agent from planning at all.

        THE BUILT TERM HAS TO BE BOUNDED, AND WAS NOT
        ---------------------------------------------
        This used to be ``count(consumed) - count(built)``, and that subtraction
        is a reward with no floor: on a game where every click paints a few more
        pixels of some colour, the objective falls by a few *every click, for
        ever*. Measured - one game clicked 237 times, every single click scored as
        an improvement, and it completed nothing. 960 phantom successes across the
        suite, the largest single waste anywhere in the run.

        The fix is to measure assembly as **distance below its own record** rather
        than as a raw count. That has a floor at zero, and it cannot be farmed:
        painting more raises the record by the same amount, so the reward for
        exceeding it is zero. It is the ratchet's own logic applied to itself.

        Consumption keeps a plain count because it already has a real floor - the
        collectibles run out - and it is scaled so that one collectible outranks
        any amount of assembly. Assembly is a tie-break, not a gradient: it says
        *which* of two equally-collected boards is further along, never that
        painting is worth more than picking something up.
        """
        few = self.target_colors
        many = self.prog.built - few - self.retired
        if not few and not many:
            return None
        gap = 0
        if many:
            target = sum(self.prog.peak[c] for c in many)
            gap = max(0, target - self.prog.count(frame, many))
        if not few:
            return gap
        return self.prog.count(frame, few) * BUILT_SCALE + min(gap, BUILT_SCALE - 1)

    @property
    def target_colors(self) -> set[int]:
        """The colours currently believed to be the objective, minus the retired.

        One source of truth. Three places used to recompute
        ``collectible | consumed`` independently - the objective, the cell list the
        planner walks to, and the click ranking - and a retirement that reached only
        some of them would leave the agent avoiding a colour in one branch while
        still chasing it in another.
        """
        return (self.collectible | self.prog.consumed) - self.retired

    def retire(self, why: str = "") -> set[int]:
        """Give up on the current objective and let the next one form.

        Called when the agent has driven the objective as far as it goes and the
        game has not ended. Everything currently believed to be the target is
        struck, which is deliberately blunt: the evidence says the *set* was wrong,
        not which member of it, and the ratchet will re-form a new set from
        whatever is still moving. Returns what was retired, for the record.

        Without this the most confident failure mode in the system is unrecoverable
        - a wrong destination costs more than no destination, because the agent
        walks to it for the rest of the budget.
        """
        gone = self.target_colors
        self.retired |= gone
        return gone

    def think(
        self,
        frame: np.ndarray,
        *,
        max_nodes: int = MAX_IMAGINE,
        max_depth: int = 40,
    ) -> list[int]:
        """Search the dream for the shortest sequence that makes visible progress.

        "Progress" without being told the win condition is the crux, and the
        answer that generalises is not a genre but a *shape of evidence*: some
        measurable quantity is ratcheting, and the search should push it further.
        ``objective`` supplies that quantity; this function just finds the
        cheapest imagined route that improves it.

        Breadth-first over *imagined* frames, so what comes back is the shortest
        known route to progress - which is what a quadratic score wants. Nothing
        here spends an action.
        """
        if self.m.avatar < 0:
            return []
        start = self.objective(frame)
        if start is None:
            return []
        if not self.calm:
            # Something else is moving. Look one step ahead only, and re-plan
            # after every real action rather than betting on a stale route.
            max_depth = min(max_depth, 1)
        mv = list(self.m.moves)
        seen = {self._key(frame)}
        q: deque[tuple[np.ndarray, list[int]]] = deque([(frame, [])])
        n = 0
        while q and n < max_nodes:
            cur, path = q.popleft()
            n += 1
            if len(path) >= max_depth:
                continue
            for a in mv:
                nxt = self.predict(cur, a)
                if nxt is None:
                    continue
                k = self._key(nxt)
                if k in seen:
                    continue
                seen.add(k)
                got = self.objective(nxt)
                if got is not None and got < start:
                    return path + [a]
                q.append((nxt, path + [a]))
        return []

    def wants(self, frame: np.ndarray) -> list[tuple[int, int]]:
        """Cells worth reaching, for the walker to use when the search comes back
        empty. The imagination only returns a route it can *prove* helps; this is
        the weaker claim that the ratcheting colours are where to go looking."""
        few = self.target_colors
        if not few:
            return []
        ys, xs = np.nonzero(np.isin(frame, list(few)))
        return list(zip(ys.tolist(), xs.tolist()))[:512]

    def route(
        self,
        frame: np.ndarray,
        *,
        max_nodes: int = MAX_IMAGINE,
        max_depth: int = 40,
    ) -> list[int]:
        """The cheapest route to progress: walk if you can, imagine if you must.

        Frame-space search is strictly more powerful - it is the only thing that
        can reason about a crate being shoved or a gem being swallowed - and
        strictly more expensive, because every node is a whole predicted 64x64
        frame. At 600 nodes over four buttons that is about five moves of
        lookahead, which is why ``ls20`` knew exactly which colours it was
        consuming and still returned no plan: the nearest one was further away
        than the search could see.

        Position-space search over the walk mask handles the ordinary case - the
        thing I want is somewhere I can stand - at thousands of nodes for the
        price of a few frame predictions. So try that first, and keep the
        expensive imagination for when it comes back empty, which is exactly the
        case where the board itself has to change for progress to happen.
        """
        want = self.wants(frame)
        if want:
            walked = self.m.plan(frame, want)
            if walked:
                return walked
        return self.think(frame, max_nodes=max_nodes, max_depth=max_depth)

    @staticmethod
    def _key(frame: np.ndarray) -> bytes:
        return frame.astype(np.int8).tobytes()

    def summary(self) -> str:
        return (
            f"dream move={self.acc_move:.0%}({self.hits_move + self.misses_move}) "
            f"still={self.acc_still:.0%}({self.hits_still + self.misses_still}) "
            f"{self.abstains} abstain{'' if self.calm else ' LIVELY'} "
            f"push={sorted(self.pushable)} collect={sorted(self.collectible)} "
            f"{self.prog.summary()}"
        )
