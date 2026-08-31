"""A general agent for a game it has never seen. One shot, billed per action.

The policy is the one a person uses, in the same order, for the same reasons.

  1. WIGGLE. Press every button a few times and watch what moves. ~25 actions,
     charged to level 0, where the weight is 1 out of 21..55 - so this costs
     almost nothing and buys the entire model.
  2. LOOK. Decide what is worth walking to. With no reward signal but
     ``levels_completed``, the ranking a person uses is: something that looked
     like a goal last time > something rare and small > somewhere I have not
     been. Rarity first is not arbitrary - a single blob of a colour that
     appears once is the thing the level was built around.
  3. THINK. Plan the route inside the learned model. This is free, so it is
     where all the searching happens, and because it is breadth-first the route
     is the shortest the model knows - which matters when score goes as
     (baseline/actions)^2.
  4. WALK. Execute, checking after every action that the avatar went where the
     model said. A mismatch is information: the thing in the way gets added to
     the blocking set and the route is re-planned.
  5. CARRY IT OVER. On a new level, keep the model. This is where the score
     comes from: level 0 pays for the learning, levels 1..n are worth 2..n times
     as much and reuse it for free.

Death is cheap and therefore worth risking: ``level_reset`` costs one action and
clears GAME_OVER, so the agent finds out what is lethal by touching it once.

For games with no steerable avatar - eight of the 25 dev games declare action 6
and little else - the same loop runs over clicks instead of moves, ranking
candidate targets by the same rarity heuristic.
"""

from __future__ import annotations

import time
from collections import Counter

import numpy as np

from arc3x.dream import BUILT_SCALE, Dream
from arc3x.graded import GObs, GradedRun
from arc3x.mind import Mechanics
from arc3x.percept import Volatility, background, blobs, fingerprint
from arc3x.relive import Relive

WIGGLE_REPS = 4
# Actions the objective may go without improving before the agent concludes it was
# chasing the wrong thing. Generous, because a real objective can be temporarily
# unreachable - behind a door, on the far side of a hazard - and giving up on it
# then would be worse than the disease. But not unbounded, and the reason is *not*
# the one first written here ("a level worth 1/45th of the score should not absorb
# all of it"). That reasoning was wrong: an action spent on a level that is never
# cleared costs zero score, so a dead objective does not damage the current level at
# all. What it costs is the *later* levels those actions could have bought, which is
# where nearly all the points are. So the limit exists to force a re-aim, not to
# economise.
STALE_LIMIT = 120
# How long one bout of archive search runs before the ranked repertoire gets
# another look. Long enough that a restart (1 + depth actions) is amortised over
# several expansions; short enough that a model which has just learned something
# from the search gets to use it.
RELIVE_ACTIONS = 500
# Rounds of plain ranked clicking allowed before the archive search takes over on
# a game with no objective. Enough for the ratchet to see a few hundred frames and
# form one, not enough to spend the budget discovering that it will not.
CLICK_GRACE = 8


class Agent:
    """One play of one game. Holds the model; the model outlives each level."""

    def __init__(
        self,
        run: GradedRun,
        *,
        budget: int = 3000,
        seconds: float = 240.0,
        seed: int = 0,
        verbose: bool = False,
    ) -> None:
        self.run = run
        self.budget = budget
        self.deadline = time.perf_counter() + seconds
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.m = Mechanics()
        # The imagination, and the objective it searches toward. Without this the
        # agent could only walk to colours that had *already* coincided with a
        # completed level - a signal that needs a win to learn what a win is.
        self.dream = Dream(self.m)
        # The fallback for when the imagination has no objective to descend, which
        # the stall report says is the usual case: archive search that rewinds by
        # RESET-and-replay instead of by deepcopy. Constructed last because it
        # holds a reference back to this agent and uses ``click_candidates``.
        self.relive = Relive(self)
        self.vol = Volatility()
        self.seen: set[bytes] = set()
        self.visited: set[tuple[int, int]] = set()
        self.tried_targets: set[tuple[int, int]] = set()
        # (blocked colour, use button) pairs already attempted while standing next
        # to that colour. "Walk to the door, press the button" is the shape of every
        # key/switch/lever game, but there is no way to tell from the outside which
        # button it is, so all of them get one try each and the result is remembered.
        # Without the memory the frontier walk re-presses the same useless button on
        # every round for the rest of the budget.
        self.tried_use: set[tuple[int, int]] = set()
        self.dead_clicks: Counter = Counter()
        # Clicks that actually moved the objective, so they can be tried first
        # next round. Without this the ranking is memoryless and rediscovers the
        # same useful spot from scratch every time.
        self.good_clicks: Counter = Counter()
        # The best objective value seen on this level. Progress is measured
        # against this and never against the previous frame: a click that toggles
        # a colour down, and the next one back up, beats a per-step comparison
        # half the time and loops forever. Measured - it produced
        # 'click:progress x237' on a game that completed nothing.
        self.best_obj: int | None = None
        # Set by every ``act``: did that one action beat the record above.
        self.last_gain: bool = False
        # Actions since the objective last improved. When the agent has driven the
        # objective as far as it will go and no level has come, the objective was
        # not the win condition - and that is the most expensive belief available,
        # because the agent has a destination it trusts and walks there for the rest
        # of the budget. All 25 dev games currently spend their entire cap.
        self.stale: int = 0
        # Set the instant the objective hits zero without a level to show for it.
        self.floored: bool = False
        self.level = 0
        self.obs: GObs | None = None

    # -- plumbing ---------------------------------------------------------

    @property
    def spent(self) -> bool:
        return self.run.actions >= self.budget or time.perf_counter() > self.deadline

    @property
    def timed_out(self) -> bool:
        """True when the clock stopped us rather than the action budget.

        Kaggle bills actions, not seconds, so a run cut short by the wall clock
        is not a measurement - it is a different, machine-dependent experiment,
        and two such runs are not comparable. It has to be visible in the report
        rather than silently averaged in.
        """
        return self.run.actions < self.budget and time.perf_counter() > self.deadline

    def act(self, aid: int, x: int = 0, y: int = 0) -> GObs:
        """One billed action, folded into the model and the novelty memory."""
        before = self.obs.frame if self.obs is not None else None
        obs = self.run.step(aid, x, y)
        level_up = obs.levels_completed > self.level
        if before is not None:
            self.vol.add(before, obs.frame)
            # The dream grades itself *before* the mechanics update, so the
            # prediction being scored is the one the agent actually acted on.
            self.dream.observe(aid, before, obs.frame)
            self.m.observe(
                aid, before, obs.frame, level_up=level_up, died=obs.game_over
            )
        self.obs = obs
        self.seen.add(fingerprint(obs.frame, self.vol.live_mask))
        if not level_up and not obs.game_over:
            # Evidence for the cell key, and the action itself, so the recorded
            # route to this state is the one actually walked - whichever strategy
            # walked it. Skipped on a level change or a death because both restore
            # the board, and a jump across a restoration is not a frame-to-frame
            # transition - see ``Clockless.cut``.
            self.relive.observe(obs.frame, (aid, x, y))
        # Did that action leave the board better than we have ever had it on this
        # level? Kept here rather than in the branch that cares, because every
        # branch moves the board and a record that only counts clicks lets the
        # click branch re-claim a gain that walking earned.
        self.last_gain = False
        s = self.dream.objective(obs.frame)
        # Caught here and not at the top of the round, because the moment the
        # objective bottoms out is often the moment the agent dies: one measured
        # game drains a colour from 63 pixels to zero, hits GAME_OVER, and the
        # automatic reset below restores the board - so a check at the top of the
        # round sees 63 again and never learns anything. The flag survives the
        # reset; the frame does not.
        #
        # ``s < BUILT_SCALE`` is exactly "no pixels of the target colours remain":
        # the objective is count*BUILT_SCALE plus a smaller tie-break term.
        if s is not None and s < BUILT_SCALE and not level_up and self.dream.target_colors:
            self.floored = True
        if s is not None and (self.best_obj is None or s < self.best_obj):
            self.best_obj = s
            self.last_gain = True
            self.stale = 0
        elif s is not None:
            self.stale += 1
        if level_up:
            if self.verbose:
                print(
                    f"    level {obs.levels_completed} at action {self.run.actions}"
                    f"  [{self.m.summary()}]"
                )
            self.level = obs.levels_completed
            self.on_new_level()
        elif obs.game_over:
            # One action to undo death, and the level is pristine again.
            self.obs = self.run.reset()
            self.on_restored()
        return obs

    def on_new_level(self) -> None:
        self.visited.clear()
        self.tried_targets.clear()
        # A new board has new doors, and a button that did nothing to the last
        # level's blue wall may be exactly what opens this one's. Cleared here and
        # deliberately *not* in ``on_restored``: a death does not un-press a button.
        self.tried_use.clear()
        self.dead_clicks.clear()
        # Coordinates do not survive a new board, so which spot was useful does
        # not either - but what the ratchet learned about *colours* does, and that
        # is what re-finds the spot cheaply.
        self.good_clicks.clear()
        # A fresh board restores everything, so the old best is unbeatable and
        # would make every real gain on the new level look like a regression.
        self.best_obj = None
        self.stale = 0
        self.floored = False
        # A new level puts us somewhere else entirely, so the tracked position is
        # stale and would drag ``locate`` toward the wrong lookalike.
        self.m.pos = None
        # The board has been restored, so frame-to-frame counts no longer
        # continue; what was learned about which colours ratchet still holds.
        self.dream.cut()
        # A plan is a route from *this* level's start, so none of them survive.
        # The learned cell mask does.
        self.relive.new_level()
        if self.obs is not None:
            self.relive.observe(self.obs.frame)

    def on_restored(self) -> None:
        """A death restored the board, but the level did not change.

        This used to call ``on_new_level``, and those are different events. A death
        rewinds the *board*; it does not rewind what the agent has achieved on this
        *level*, so the objective record has to survive it.

        Clearing the record here is what let two measured games re-earn the same
        progress for ever: drain the target colour to zero, die, get a fresh record,
        drain the restored board again. 375 and 346 'progress' claims respectively -
        all of them the same forty pixels, none of them a level.

        Keeping it turns that loop into the signal it should always have been. The
        record is already at the floor, nothing can beat it, so the objective goes
        stale, gets retired, and the agent looks for a different one.
        """
        self.visited.clear()
        self.m.pos = None
        self.dream.cut()
        # A RESET restores the level and its counters, so the stretch of frames
        # ends here and the current plan is no longer where we are standing.
        self.relive.cut()
        if self.obs is not None:
            self.relive.observe(self.obs.frame)

    # -- 1. wiggle ---------------------------------------------------------

    def wiggle(self) -> None:
        """Press everything a few times. The cheapest information in the game.

        Each button several times **in a row**, not round-robin, and that detail
        is worth two of wa30's four movement buttons.

        Some sprites turn to face the way they are walking, so the first press of a
        new direction both turns and steps and the ink inside the cell shifts to
        the other end of it - a step of 4 reads as 7. Pressing 1,2,3,4,1,2,3,4
        makes *every* press a direction change, so every press is a turn and the
        artifact is the only thing the model ever sees. Measured: 2 votes for the
        artifact, 1 for the truth, and ``moves`` came out ``{2:(7,0), 4:(0,7)}``.

        Pressing 1,1,1,1 then 2,2,2,2 turns once and then walks three times, so the
        truth outvotes the artifact 3 to 1. It is also what a person does - press it
        again and see whether the same thing happens - and it costs nothing, because
        level 0 carries weight 0 in the score and its actions are free.
        """
        acts = [a for a in self.run._declared if a != 6]
        for rep in range(2):
            for a in acts:
                miss = 0
                for _ in range(WIGGLE_REPS if rep == 0 else 2):
                    if self.spent:
                        return
                    before = self.obs.frame if self.obs is not None else None
                    self.act(a)
                    # A move that is refused twice tells us about walls, so keep
                    # going even when nothing happens - but not four times. Two
                    # refusals have already made the point and the rest is waste.
                    if (
                        before is not None
                        and self.obs is not None
                        and not (before != self.obs.frame).any()
                    ):
                        miss += 1
                        if miss >= 2:
                            break
                    else:
                        miss = 0
            self.m.settle()
            if self.m.avatar >= 0 and len(self.m.moves) >= 2:
                # Enough to steer with. Reversibility already confirmed it.
                break
        self.m.settle()

    # -- 2. look -----------------------------------------------------------

    def targets(self, frame: np.ndarray) -> list[tuple[int, int]]:
        """Cells worth walking to, best first.

        The ranking is the human one: a colour that has ever coincided with a
        level completion, then a rare small object, then anywhere unvisited.
        """
        bg = self.m.background if self.m.background >= 0 else background(frame)
        bl = blobs(frame, ignore={bg, self.m.avatar})
        if not bl:
            return []
        colcount = Counter(b.color for b in bl)
        goal = {c for c, _ in self.m.goal_colors.most_common(4)}
        bad = self.m.blocked_set
        scored: list[tuple[tuple, tuple[int, int]]] = []
        box = self.m.where(frame)
        ay, ax = (box[0], box[1]) if box else (32, 32)
        for b in bl:
            if b.color in bad and b.color not in goal:
                continue
            cy, cx = b.center
            if (cy, cx) in self.tried_targets:
                continue
            key = (
                0 if b.color in goal else 1,     # a known goal beats everything
                colcount[b.color],               # rare things are special
                b.size,                          # small things are pickups
                abs(cy - ay) + abs(cx - ax),      # then nearest, to stay cheap
            )
            scored.append((key, (cy, cx)))
        scored.sort()
        return [p for _k, p in scored]

    # -- 3+4. think and walk ----------------------------------------------

    def navigate(self, cells: list[tuple[int, int]], max_actions: int = 120) -> str:
        """Walk toward any of ``cells``, re-planning after every single step.

        Re-planning per step rather than per route is the whole trick. The model
        starts out believing nothing blocks it, so a route planned once goes
        straight through the first wall and the remaining actions are wasted.
        Executing one action and re-planning costs nothing extra - planning is
        free - and each refused move teaches the model a wall, so the route
        repairs itself as it is walked. This is why the agent maps a level in
        O(path) billed actions instead of O(area).

        Returns 'level', 'win', 'stuck' or 'spent'.
        """
        assert self.obs is not None
        stuck_at: Counter = Counter()
        for _ in range(max_actions):
            if self.spent:
                return "spent"
            box = self.m.where(self.obs.frame)
            if box is None:
                return "stuck"
            here = (box[0], box[1])
            if here in cells:
                # Standing on a target with nothing having happened means this
                # target is not one. Reporting success here spent zero actions and
                # sent ``play`` round the loop with the same target list, for 190
                # empty rounds on one measured game. Strike it and re-aim.
                self.tried_targets.add(here)
                cells = [c for c in cells if c != here]
                if not cells:
                    return "stuck"
            path = self.m.plan(self.obs.frame, cells)
            if not path:
                return "stuck"
            lv = self.level
            before = self.obs.frame
            self.act(path[0])
            if self.obs.won:
                return "win"
            if self.level != lv:
                return "level"
            if not (before != self.obs.frame).any():
                # Refused. Mechanics.observe has just recorded the blocking
                # colour, so the next plan routes around it - unless we have now
                # been refused from this same square repeatedly, in which case
                # the model is not learning and we should try something else.
                stuck_at[here] += 1
                if stuck_at[here] >= 3:
                    return "stuck"
                continue
            nb = self.m.where(self.obs.frame)
            if nb:
                self.visited.add((nb[0], nb[1]))
        return "stuck"

    def explore_step(self) -> str:
        """Nothing looks interesting: walk to the farthest place we have not been.

        The frontier comes from the model's own reachability, which is free to
        compute, so exploration is directed rather than random. A random walk
        needs O(n^2) actions to cover n cells; walking the frontier needs O(n),
        and at a quadratic score that difference is the whole game.
        """
        assert self.obs is not None
        reach = self.m.reachable(self.obs.frame)
        fresh = [(len(p), c) for c, p in reach.items() if p and c not in self.visited]
        if not fresh:
            self.visited.clear()
            fresh = [(len(p), c) for c, p in reach.items() if p]
        if not fresh:
            return "stuck"
        fresh.sort(reverse=True)
        return self.navigate([c for _n, c in fresh[:8]], max_actions=60)

    def imagine(self, max_actions: int = 200) -> str:
        """Act on the imagination: think for free, spend one action, think again.

        This is the step the agent was missing entirely. ``navigate`` walks toward
        cells that some heuristic ranked; this walks toward whatever the learned
        copy of the game says will *reduce the distance to done*, which is the
        only one of the two that generalises to a game whose goal nobody named.

        Re-planning after each single action rather than committing to the whole
        route is the same discipline as ``navigate``, for the same reason: the
        copy is sometimes wrong, and one billed action is the cheapest possible
        way to find that out.

        Returns 'level', 'win', 'noplan', 'stuck' or 'budget'.
        """
        assert self.obs is not None
        used0 = self.run.actions
        stall = 0
        while self.run.actions - used0 < max_actions and not self.spent:
            plan = self.dream.route(self.obs.frame)
            if not plan:
                return "noplan"
            lv = self.level
            before = self.obs.frame
            self.act(plan[0])
            if self.obs.won:
                return "win"
            if self.level != lv:
                return "level"
            if not (before != self.obs.frame).any():
                # The copy promised a change and nothing happened. Mechanics has
                # just learned the obstacle, so one more try is worth it - but a
                # copy that keeps being wrong here is not worth spending on.
                stall += 1
                if stall >= 4:
                    return "stuck"
            else:
                stall = 0
        return "budget"

    # -- the click branch --------------------------------------------------

    def click_candidates(self, frame: np.ndarray) -> list[tuple[int, int]]:
        """Where a person would click, best first.

        Blob centres, plus the four corners of blobs big enough that the centre
        might miss the interactive part. Ranked by the same rarity heuristic as
        walking targets, and de-prioritised once a click there has been shown to
        do nothing.

        Above rarity sits the objective: a colour the ratchet says is being
        *consumed* is the thing the level is about, so click that first. This is
        the click-side twin of ``Dream.wants`` - walking to the disappearing
        colours and clicking the disappearing colours are the same instinct - and
        it is what gives a game with no avatar a destination at all.
        """
        bg = self.m.background if self.m.background >= 0 else background(frame)
        bl = blobs(frame, ignore={bg})
        colcount = Counter(b.color for b in bl)
        goal = {c for c, _ in self.m.goal_colors.most_common(4)}
        want = self.dream.target_colors
        made = self.dream.prog.built - want - self.dream.retired
        out: list[tuple[tuple, tuple[int, int]]] = []
        for b in bl:
            pts = [b.center]
            if b.size >= 9:
                pts += [
                    (b.top, b.left),
                    (b.top, b.left + b.width - 1),
                    (b.top + b.height - 1, b.left),
                    (b.top + b.height - 1, b.left + b.width - 1),
                ]
            for y, x in pts:
                y = int(min(63, max(0, y)))
                x = int(min(63, max(0, x)))
                out.append(
                    (
                        (
                            self.dead_clicks[(y, x)],
                            # Clicks that have actually moved the objective come
                            # first, and are tried before anything untested.
                            -self.good_clicks[(y, x)],
                            0 if b.color in want else 1 if b.color in goal else 2,
                            1 if b.color in made else 0,
                            colcount[b.color],
                            b.size,
                        ),
                        (y, x),
                    )
                )
        out.sort()
        seen: set[tuple[int, int]] = set()
        uniq = []
        for _k, p in out:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return uniq

    def click_round(self, max_clicks: int = 48) -> str:
        """Click the ranked candidates, judged by the objective and not by whether
        the picture moved.

        The old version returned success on *any* pixel change, and ``play`` took
        that as a reason to start the round again - so on a game whose buttons all
        repaint part of the board, the agent clicked 259 times, something changed
        every single time, and it never once asked whether the change was
        progress. Measured over the suite, that single confusion accounted for the
        largest block of wasted actions anywhere in the run.

        So there are two different successes, and they must not share a name:

          progress  the board is better than we have ever had it on this level -
                    this is worth repeating
          change    the picture moved and the objective did not - this is
                    information while we still have no objective, and mere
                    animation once we do

        "Better than ever" and not "better than the previous frame": a click that
        toggles something off, and the next one back on, beats a per-frame
        comparison every other time and loops for ever. Measured - it reported
        progress 237 times on a game that completed nothing. A record can only be
        beaten as often as there is genuine room to beat it.

        Returns 'level', 'win', 'progress', 'change' or 'nothing'.
        """
        assert self.obs is not None
        cands = self.click_candidates(self.obs.frame)
        if not cands:
            return "nothing"
        changed = False
        for y, x in cands[:max_clicks]:
            if self.spent:
                break
            lv = self.level
            before = self.obs.frame
            had_obj = self.dream.objective(before) is not None
            self.act(6, x=x, y=y)
            if self.obs.won:
                return "win"
            if self.level != lv:
                self.good_clicks[(y, x)] += 2
                return "level"
            if not (before != self.obs.frame).any():
                self.dead_clicks[(y, x)] += 1
                continue
            changed = True
            if self.last_gain:
                self.good_clicks[(y, x)] += 1
                return "progress"
            if not had_obj:
                # No notion of progress yet, so a change is the only teacher we
                # have. Take it and let the ratchet form.
                return "change"
            # It moved the board and did not improve on the record. Not fatal -
            # some games make you build before you clear - but it goes to the back
            # of the queue rather than being tried again next round.
            self.dead_clicks[(y, x)] += 1
        return "change" if changed else "nothing"

    # -- the act branch ----------------------------------------------------

    def act_round(self) -> str:
        """Press each button that changes the board without moving us, once.

        ``Mechanics.moves`` filters for a non-zero displacement, which silently
        discards every button that does something *other* than walk - use, grab,
        drop, select, rotate. Twelve of the 25 dev games have one; two have nothing
        else, five working buttons and four, all invisible to the planner. A route
        genuinely cannot be made of them, which is why they were excluded, but a
        person presses them constantly.

        Cheap by construction: at most one press per button, so the whole branch
        costs about five actions. Ordered by how often the button has been seen to
        do anything, because a button that has worked more often is more likely to
        be the interaction and less likely to be a menu key.

        Returns 'level', 'win', 'progress', 'change' or 'nothing'.
        """
        assert self.obs is not None
        acts = [a for a in self.m.acts if a != 6]
        if not acts:
            return "nothing"
        changed = False
        for a in acts:
            if self.spent:
                break
            lv = self.level
            before = self.obs.frame
            self.act(a)
            if self.obs.won:
                return "win"
            if self.level != lv:
                return "level"
            if not (before != self.obs.frame).any():
                continue
            changed = True
            if self.last_gain:
                return "progress"
        return "change" if changed else "nothing"

    # -- fallback ----------------------------------------------------------

    def flail(self, n: int = 40) -> None:
        """No model, nothing ranked worked. Random, but never repeat a dead action."""
        acts = list(self.run._declared) or [1, 2, 3, 4, 5, 6]
        dead = {a for a in acts if self.m.tries[a] >= 4 and self.m.noop[a] == self.m.tries[a]}
        live = [a for a in acts if a not in dead] or acts
        for _ in range(n):
            if self.spent:
                return
            a = int(self.rng.choice(live))
            if a == 6:
                cands = self.click_candidates(self.obs.frame) if self.obs else []
                if cands:
                    y, x = cands[int(self.rng.integers(len(cands)))]
                else:
                    y, x = int(self.rng.integers(64)), int(self.rng.integers(64))
                self.act(6, x=x, y=y)
            else:
                self.act(a)
            if self.obs is not None and self.obs.won:
                return

    # -- the whole play ----------------------------------------------------

    def cover(self, max_actions: int = 400) -> str:
        """Stand on every square the model thinks we can reach.

        This is the workhorse, and it is deliberately dumber than the targeting
        heuristic. Most levels complete when the avatar touches the right thing,
        and enumerating everything walkable is the only method that does not
        depend on guessing which thing that is. Directed frontier walking makes
        it affordable: covering n squares costs O(n) actions, where a random walk
        would cost O(n^2) - and O(n) for a 5px grid on a 64x64 board is a few
        hundred actions, which is exactly the budget the scoring formula allows.
        """
        used0 = self.run.actions
        while self.run.actions - used0 < max_actions and not self.spent:
            out = self.explore_step()
            if out in ("level", "win"):
                return out
            if out == "stuck":
                return "covered"
        return "budget"

    def push_frontier(self, max_actions: int = 200) -> str:
        """Walk into each kind of thing we have never been allowed to enter, once,
        and then try every use button while standing next to it.

        A blocked colour might be a wall, a door, a hazard or the goal, and from
        the outside those look identical. A person tries each kind once and reads
        the result. One attempt per *colour*, not per tile, keeps it cheap.

        The second half is the general shape of a very large family of games -
        locked door, switch, lever, terminal, crate you have to grab before you can
        shove. None of them open by being walked into; you stand next to the thing
        and press the button. Pressing use *in place* in the middle of the room, as
        the standalone use branch does, can never discover any of them, because the
        button only does something when there is something to do it to.

        So the pairing is what matters: position plus press. Bounded by construction
        - one press per (frontier colour, use button) pair, remembered across rounds,
        so a game with four frontier colours and five use buttons costs at most 20
        actions ever, not per round.
        """
        assert self.obs is not None
        used0 = self.run.actions
        for color, spots in sorted(
            self.m.frontier_colors(self.obs.frame).items(),
            key=lambda kv: (
                0 if kv[0] in {c for c, _ in self.m.goal_colors.most_common(4)} else 1,
                -len(kv[1]),
            ),
        ):
            if self.spent or self.run.actions - used0 >= max_actions:
                return "budget"
            if color in self.m.fatal:
                continue
            out = self.navigate(spots[:24], max_actions=40)
            if out in ("level", "win"):
                return out
            box = self.m.where(self.obs.frame)
            if box is None:
                continue
            # We are standing next to it; now shove into it from here.
            for a, (dy, dx) in self.m.moves.items():
                ny, nx = box[0] + dy, box[1] + dx
                if not (0 <= ny < 64 and 0 <= nx < 64):
                    continue
                patch = self.obs.frame[ny : ny + box[2], nx : nx + box[3]]
                if patch.size and (patch == color).any():
                    lv = self.level
                    self.act(a)
                    if self.obs.won:
                        return "win"
                    if self.level != lv:
                        return "level"
                    break
            # Shoving did not open it. Try using it.
            for a in self.m.acts:
                if a == 6 or (color, a) in self.tried_use:
                    continue
                if self.spent or self.run.actions - used0 >= max_actions:
                    return "budget"
                self.tried_use.add((color, a))
                lv = self.level
                self.act(a)
                if self.obs.won:
                    return "win"
                if self.level != lv:
                    return "level"
                if self.last_gain:
                    return "used"
        return "pushed"

    def play(self) -> None:
        self.obs = self.run.reset()
        self.seen.add(fingerprint(self.obs.frame))
        self.wiggle()
        if self.verbose:
            print(f"    after wiggle ({self.run.actions} actions): {self.m.summary()}")

        rounds = 0
        idle = 0
        while not self.spent and not (self.obs and self.obs.won):
            rounds += 1
            spent0 = self.run.actions
            level_at_start = self.level
            L = level_at_start
            steerable = self.m.avatar >= 0 and bool(self.m.moves)
            # Recorded once per round, because "did this agent ever have a
            # destination on this level" is the first question to ask of a zero
            # and the score cannot answer it.
            self.run.note("steer" if steerable else "nosteer", L)
            if self.dream.objective(self.obs.frame) is None:
                self.run.note("noobjective", L)

            # Before anything else: is the thing we are chasing still worth
            # chasing? Two exhaustion signals, and both mean the same thing - the
            # objective has been driven as far as it goes and the game did not end,
            # so it was never the win condition.
            #
            #   floor    every pixel of the target colour is gone. Definitive: one
            #            measured game drains a colour from 63 to zero and dies.
            #   stale    STALE_LIMIT actions since the record last moved, having
            #            moved at least once. Covers the slower version - another
            #            measured game converts one colour into another two pixels
            #            at a time for hundreds of actions and never finishes.
            #
            # Retiring is blunt on purpose: the evidence says the *set* was wrong,
            # not which member, and the ratchet re-forms a new one from whatever is
            # still moving.
            if self.dream.target_colors:
                floor = self.floored
                if floor or self.stale >= STALE_LIMIT:
                    gone = self.dream.retire()
                    self.run.note(f"retire:{'floor' if floor else 'stale'}", L)
                    if self.verbose:
                        print(f"    retired {sorted(gone)} at action {self.run.actions}")
                    self.best_obj = None
                    self.stale = 0
                    self.floored = False
                    self.tried_targets.clear()

            if steerable:
                # First ask the imagination. It is the only part of the agent that
                # can say "this specific sequence makes the board closer to done"
                # rather than "this cell looks interesting", so when it has an
                # opinion it outranks every heuristic below.
                out = self.imagine()
                self.run.note(f"imagine:{out}", L)
                if out in ("level", "win"):
                    continue
                # A known goal colour means we can go straight there. This is
                # the transfer that pays for the level-0 exploration: levels
                # 1..n are worth 2..n times as much and cost a fraction.
                goal = {c for c, _ in self.m.goal_colors.most_common(3)}
                if goal:
                    cells = self.goal_cells(self.obs.frame, goal)
                    if cells:
                        out = self.navigate(cells, max_actions=150)
                        self.run.note(f"goal:{out}", L)
                        if out in ("level", "win"):
                            continue
                out = self.cover()
                self.run.note(f"cover:{out}", L)
                if out in ("level", "win"):
                    continue
                out = self.push_frontier()
                self.run.note(f"frontier:{out}", L)
                # 'used' means standing next to a blocked colour and pressing a
                # button moved the objective - a door opened. Re-plan immediately:
                # the board has changed in a way the rest of this round's cached
                # target lists do not know about.
                if out in ("level", "win", "used"):
                    continue

            # The use buttons, before the clicks. Cheaper - one press each rather
            # than up to 48 - and on the two games with no movement buttons at all
            # they are the only thing that does anything.
            out = self.act_round()
            if out != "nothing":
                self.run.note(f"use:{out}", L)
            if out in ("level", "win", "progress"):
                continue

            if 6 in self.run._declared:
                out = self.click_round()
                self.run.note(f"click:{out}", L)
                # Only a genuine gain earns another round. 'change' used to earn
                # one, and that is the loop that burned 259 actions on a single
                # level: click, something moved, start over, click again.
                if out in ("level", "win", "progress"):
                    continue
                if out == "change" and self.dream.objective(self.obs.frame) is None:
                    # Still no objective, so changes are the only thing feeding
                    # the ratchet. Worth a few rounds - and only a few. Left
                    # unbounded this is the single largest waste in the suite:
                    # su15 spent 709 rounds here and tn36 2,940, clicking, seeing
                    # the picture move, and starting over, because a change is not
                    # a gain and nothing was ever accumulating. After the grace
                    # period the round falls through to the archive search below,
                    # which clicks too but can go back to a spot that mattered.
                    if rounds <= CLICK_GRACE:
                        continue

            if self.level != level_at_start:
                continue

            # Nothing worked this round. Reconsider the model with weaker
            # evidence, wipe the memory of where we have been, and if that
            # still yields nothing, reset the level - one action - so the next
            # round starts from a pristine board rather than a stuck corner.
            self.run.note("roundfail", L)
            # A round that ran the whole repertoire and spent nothing is not
            # cautious, it is stuck: every branch declined immediately and the
            # only action billed was the reset at the bottom. Measured at 190 such
            # rounds on one game - roughly half a level's budget spent deciding
            # not to act. Three in a row means the deterministic branches have
            # nothing left to say here, so go and get new evidence instead.
            if self.run.actions - spent0 <= 1:
                idle += 1
                self.run.note("idleround", L)
            else:
                idle = 0
            self.m.settle(min_votes=1)
            self.visited.clear()
            self.tried_targets.clear()
            self.dead_clicks.clear()

            # Search, and only then flail. The whole ranked repertoire has just
            # declined, which is exactly the state the stall report describes -
            # nothing to aim at, board already walked - and the honest answer to
            # "I have no idea what to do" is to try things systematically and be
            # able to come back to the interesting ones. A random walk does the
            # first half and cannot do the second: RESET plus replay is the only
            # rewind that works through the gateway, and it costs 1 + depth.
            #
            # Level 0 gets the lion's share of this, deliberately. Its weight is 1
            # out of 21..55, so a level-0 score driven to nearly zero by searching
            # costs under 5 points of the 100 available, while *clearing* it is
            # what unlocks every later level and the completion cap with them.
            #
            # KNOWN TO KEY ON THE WRONG VARIABLE - not yet changed, see below.
            # Level i's score is (b/n)^2 * 100, so adding m actions to a level that
            # has already cost n multiplies its score by (n/(n+m))^2. That is
            # *baseline-free*: adding 500 to a level 50 actions old keeps 0.8% of
            # its points, while adding 500 to one already 2000 old keeps 64%. The
            # marginal cost of a search action falls as 1/n^3. So the quantity that
            # decides how much search a level can afford is how much it has already
            # spent, and this line ignores it - it throttles a hopeless level that
            # is 2000 actions deep exactly as hard as a fresh one, buying nothing,
            # because a level that is never cleared scores 0 whatever is spent on
            # it. (Measured: the best notebook run on tn36 put 483 of its 583
            # actions into a level it failed and still took the full completion cap;
            # m0r0 put 477 into a level it *cleared* and kept 0.02 of 4.76.)
            #
            # The shape that follows is m = clamp(ALPHA/(L+1) * spent_on_level,
            # floor/(L+1), RELIVE_ACTIONS), which holds the fraction of level score
            # surrendered roughly constant instead of unknown, and still protects
            # later levels more because their absolute points are worth more. Left
            # unimplemented on purpose: ALPHA and the floor are guesses, relive.py
            # has not yet executed once, and stacking a tuned budget policy on an
            # unmeasured search is how the last eleven grafts got their reputation.
            # First A/B after the suite runs - see PLAN.md Phase 3.
            out = self.relive.run_level(RELIVE_ACTIONS if L == 0 else RELIVE_ACTIONS // 4)
            self.run.note(f"relive:{out}", L)
            if self.obs is not None and self.obs.won:
                break
            if out in ("level", "win"):
                continue
            if out == "budget":
                # It still had somewhere to go; give it the next round too rather
                # than interrupting a search that is making progress.
                continue

            # 'dry' - the archive is fully expanded and clicking is exhausted.
            # Randomness is the only thing left that can seed a new cell.
            if idle >= 3:
                self.run.note("exhausted", L)
                idle = 0
                self.flail(60)
            elif rounds % 2 == 0:
                self.flail(50)
            else:
                self.obs = self.run.reset()
                # Same event as a death: the board is back, the level is not new.
                self.on_restored()
        if self.timed_out:
            self.run.note("TIMEOUT", self.level)

    def goal_cells(self, frame: np.ndarray, goal: set[int]) -> list[tuple[int, int]]:
        """Every pixel showing a colour that has previously ended a level."""
        mask = np.isin(frame, list(goal))
        ys, xs = np.nonzero(mask)
        return list(zip(ys.tolist(), xs.tolist()))[:512]


def play_agent(run: GradedRun, **kw) -> None:
    """Entry point matching ``graded.run_suite``'s ``agent_fn`` contract."""
    Agent(run, **kw).play()
