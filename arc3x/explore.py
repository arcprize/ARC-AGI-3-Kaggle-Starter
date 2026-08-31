"""Go-Explore search core for ARC-AGI-3 - general, no per-game knowledge.

WHY THIS EXISTS
---------------
The incumbent agent spent ~17.6 seconds and ~1,700 LLM tokens per *single*
engine action. Scoring is ``min(115, (baseline/actions)^2 * 100)`` per
*completed* level and 0 for an uncompleted one, so an agent that cannot afford
enough actions to finish a level scores nothing at all - which is exactly what
happened (781 actions on sk48 level 0, baseline 61, never completed).

The engine, however, is a pure-Python in-process object we can ``deepcopy`` and
step at ~700 actions/sec for free, sustained, including the deepcopy cost of
restarting dead clones. That is ~12,000x more actions per second than the LLM
agent managed. So we decouple "actions taken" from "LLM calls": search
hundreds of thousands of actions locally, then replay one short winning line to
the graded environment.

THE ALGORITHM (four general ideas, no game-specific logic anywhere)
------------------------------------------------------------------
1. ARCHIVE / GO-EXPLORE. Keep a map from "situation" to the *shortest known
   action plan* that reaches it. Repeatedly: pick a promising archived
   situation, restore it, explore from there, and file away every new situation
   found. This is what beats sparse reward - no reward shaping, no domain
   knowledge, just "have I ever seen this situation before?".

   What counts as "the same situation" is the whole ballgame, and it is not the
   raw frame - a HUD timer draining one pixel per action makes every raw frame
   unique, which silently reduces this to a random walk. ``cell.py`` calibrates
   a coarse key that discards clock-like pixels. Read its docstring; it is the
   single most important design decision in here.

2. ENGINE-EXACT ACTION SETS. ``_get_valid_actions()`` hands us the legal moves
   *including* concrete ACTION6 click coordinates, collapsing a 4096-wide
   coordinate space to a branching factor of 2-13 on most games. For the two
   games with hundreds of legal clicks we group clicks by connected same-colour
   region and sample one representative per region - still purely frame-derived,
   still general.

3. STICKY ROLLOUTS + NO-OP PRUNING. Grid games need the same action repeated to
   cross a room, so the rollout policy repeats its previous action with high
   probability. Any action that leaves the frame *and* the legal action set
   unchanged is recorded as a no-op for that situation and never retried there.

4. PLAN COMPRESSION (this is where the score comes from). A random walk that
   finishes a level takes hundreds of actions; scoring is quadratic in that
   number. So after finding *any* solution we shrink it: splice out loops
   (revisited frames) and greedily drop action windows, re-verifying by replay
   after every edit. Verification makes it sound even when the frame does not
   capture the full hidden state. 400 actions -> near-baseline is routine, and
   (61/400)^2*100 = 2.3 versus (61/70)^2*100 = 76.

Run:
    .venv/Scripts/python.exe arc3x/explore.py --game sk48 --budget 60
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from arc3x.cell import CellKey, calibrate
from arc3x.twin import Act, Obs, Twin, default_env_dir

# ---------------------------------------------------------------------------
# scoring (mirrors taaf/game.py:_compute_final_score)
# ---------------------------------------------------------------------------


def level_score(baseline: int, actions: int) -> float:
    """RHAE for one completed level. Capped at 115 = 1.15x human."""
    if actions <= 0:
        return 0.0
    return min(115.0, (baseline / actions) ** 2 * 100.0)


def game_score(baselines: Sequence[int], actions_per_level: Sequence[int]) -> float:
    """Weighted average with 1-indexed level weights, capped by depth reached.

    ``actions_per_level[i] <= 0`` means level i was not completed (scores 0).
    """
    n = len(baselines)
    weights = [i + 1 for i in range(n)]
    total_w = sum(weights)
    num = 0.0
    max_w = 0.0
    for i in range(n):
        used = actions_per_level[i] if i < len(actions_per_level) else 0
        s = level_score(baselines[i], used) if used > 0 else 0.0
        if s > 0:
            max_w += weights[i]
        num += weights[i] * s
    if total_w == 0:
        return 0.0
    return min(num / total_w, max_w / total_w * 100.0)


# ---------------------------------------------------------------------------
# frame-derived click reduction (general: uses only the pixels)
# ---------------------------------------------------------------------------


def components(frame: np.ndarray) -> np.ndarray:
    """Label 4-connected same-colour regions of the frame.

    Used only to shrink very large click sets. Nothing about any specific game
    is assumed - two pixels of the same colour that touch are one object.
    """
    h, w = frame.shape
    lab = np.full((h, w), -1, dtype=np.int32)
    nxt = 0
    for sy in range(h):
        for sx in range(w):
            if lab[sy, sx] != -1:
                continue
            col = frame[sy, sx]
            q = deque([(sy, sx)])
            lab[sy, sx] = nxt
            while q:
                y, x = q.popleft()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and lab[ny, nx] == -1 and frame[ny, nx] == col:
                        lab[ny, nx] = nxt
                        q.append((ny, nx))
            nxt += 1
    return lab


CLICK_GROUP_THRESHOLD = 32


def reduce_clicks(frame: np.ndarray, acts: tuple[Act, ...]) -> tuple[Act, ...]:
    """Keep one representative click per connected region; pass others through.

    Only kicks in when the engine offers a lot of clicks (r11l: 256,
    su15: 224). Below the threshold the exact engine set is already small
    enough to enumerate.
    """
    clicks = [a for a in acts if a.is_click]
    if len(clicks) <= CLICK_GROUP_THRESHOLD:
        return acts
    others = [a for a in acts if not a.is_click]
    try:
        lab = components(frame)
    except Exception:
        return acts
    h, w = frame.shape
    best: dict[int, Act] = {}
    for a in clicks:
        if not (0 <= a.y < h and 0 <= a.x < w):
            best[-1 - len(best)] = a
            continue
        cid = int(lab[a.y, a.x])
        best.setdefault(cid, a)
    return tuple(others + list(best.values()))


# ---------------------------------------------------------------------------
# archive
# ---------------------------------------------------------------------------


@dataclass
class Node:
    """One archived situation: how to get there, and how to explore from it."""

    key: bytes
    plan: tuple[Act, ...]
    level: int
    valid: tuple[Act, ...]
    visits: int = 0
    snap: Any = None  # cached engine snapshot; may be dropped to save memory
    noop: set[Act] = field(default_factory=set)
    tried: set[Act] = field(default_factory=set)
    dead: bool = False

    @property
    def depth(self) -> int:
        return len(self.plan)

    @property
    def untried(self) -> int:
        """How many legal actions have never been taken from this cell."""
        if not self.valid:
            return 0
        return sum(1 for a in self.valid if a not in self.tried and a not in self.noop)

    @property
    def weight(self) -> float:
        """Selection weight: prefer cells with unexplored options, then shallow.

        The original weight was ``1/(sqrt(visits+1) * (depth+1)^0.25)`` - pure
        novelty. Combined with a random rollout policy that never recorded which
        actions it had already taken from a cell, the search re-tried the same
        few actions from the same cells indefinitely and never systematically
        covered the reachable set.

        Tracking ``tried`` turns this into something much closer to a
        breadth-first sweep of the abstract cell graph: a cell with unexplored
        actions outranks one that is fully expanded, and because the archive
        always keeps the *shortest* plan to each cell, the first solution found
        is near-minimal in actions - which is exactly what the quadratic score
        rewards. A fully expanded cell keeps a small residual weight rather than
        zero, since its successors' plans may later shorten.
        """
        frontier = 4.0 if self.untried > 0 else 0.25
        return frontier / ((self.visits + 1) ** 0.5 * (self.depth + 1) ** 0.25)


@dataclass
class LevelResult:
    level: int
    plan: tuple[Act, ...] | None
    raw_len: int = 0
    cells: int = 0
    steps: int = 0
    seconds: float = 0.0
    won_game: bool = False


class Explorer:
    """Solve one level at a time from a given starting engine state."""

    def __init__(
        self,
        twin: Twin,
        *,
        cell: CellKey,
        seed: int = 0,
        sticky: float = 0.7,
        rollout: int = 48,
        max_depth: int = 1500,
        snap_cap: int = 250,
        snap_min_depth: int = 16,
        verbose: bool = True,
        recorder: Any | None = None,
        prior: Any | None = None,
        prior_mix: float = 0.6,
    ):
        self.twin = twin
        self.cell = cell
        self.rng = np.random.default_rng(seed)
        self.sticky = sticky
        self.rollout = rollout
        self.max_depth = max_depth
        self.snap_cap = snap_cap
        self.snap_min_depth = snap_min_depth
        self.verbose = verbose
        # Optional self-imitation recorder (see arc3x/selfplay_data.py) and
        # optional learned action prior (arc3x/student.py). Both default off so
        # the search's measured behaviour is unchanged unless asked for.
        self.recorder = recorder
        self.prior = prior
        self.prior_mix = prior_mix
        self.steps = 0
        self.snaps = 0
        # Plans that looked like a win inside their own rollout but failed to
        # replay from a fresh engine. Should stay 0; non-zero means the rollout
        # bookkeeping has desynchronised from the engine again.
        self.false_plans = 0

    # -- plumbing ---------------------------------------------------------

    def _restore(self, root: Any, node: Node) -> Any:
        """Get a fresh engine object positioned at ``node``.

        SNAPSHOT POLICY (this is the performance fix).

        ``copy.deepcopy(game)`` costs ~20 ms, while stepping the engine costs
        ~1.2 ms. The original code snapshotted **every newly archived cell** -
        and with a near-bijective key a new cell was created on almost every
        step, so the search paid a full 20 ms deepcopy per step. Measured
        result: 19 steps/sec against a ~700 steps/sec ceiling.

        Snapshots are now taken only when a node is actually *restored*, which
        happens once per rollout rather than once per step - roughly 48x less
        often. We also skip snapshotting shallow nodes, because replaying a
        16-action prefix (~19 ms) is already as cheap as the deepcopy itself.
        Replay is exact: the games are verified deterministic on 25/25.
        """
        if node.snap is not None:
            return copy.deepcopy(node.snap)
        g = copy.deepcopy(root)
        for a in node.plan:
            Twin.step_game(g, a)
            self.steps += 1
        if node.depth >= self.snap_min_depth and self.snaps < self.snap_cap:
            node.snap = copy.deepcopy(g)
            self.snaps += 1
        return g

    def _reclaim(self, order: list[Node]) -> None:
        """Drop snapshots of fully-expanded cells to bound memory.

        A snapshot is a whole deepcopied game object. Holding thousands of them
        per process is what silently destroyed the parallel sweep: measured 24
        steps/sec/worker with 10 workers against 546 single-process, which is
        memory thrash, not CPU starvation (12 cores were available). Kaggle's
        RAM is tighter still, so the cap has to be small and enforced.

        Cells with no untried actions left are the right ones to evict: we have
        already enumerated their successors, so restoring them is low value.
        """
        freed = 0
        for nd in order:
            if nd.snap is not None and nd.untried == 0 and nd.visits > 0:
                nd.snap = None
                freed += 1
        self.snaps = max(0, self.snaps - freed)

    def _select(self, nodes: list[Node], k: int = 24) -> Node:
        """Tournament selection - O(k), not O(len(archive)) per rollout."""
        best: Node | None = None
        bw = -1.0
        n = len(nodes)
        for _ in range(min(k, n)):
            c = nodes[int(self.rng.integers(n))]
            if c.dead:
                continue
            w = c.weight
            if w > bw:
                bw, best = w, c
        return best or nodes[0]

    # -- the search -------------------------------------------------------

    def solve_level(
        self, root: Any, start_level: int, baseline: int, budget_s: float
    ) -> LevelResult:
        """Find *some* action sequence from ``root`` that completes one level."""
        t0 = time.perf_counter()
        probe = copy.deepcopy(root)
        valid0 = Twin.valid_actions(probe)
        frame0 = self.twin.current().frame if start_level == 0 else None

        # Seed the archive with the root situation.
        seed_obs = Obs(
            frame=frame0 if frame0 is not None else np.zeros((64, 64), dtype=np.int8),
            level=start_level,
            score=start_level,
            state=None,
            valid=valid0,
        )
        root_node = Node(
            key=b"ROOT", plan=(), level=start_level, valid=valid0, snap=copy.deepcopy(root)
        )
        archive: dict[bytes, Node] = {root_node.key: root_node}
        order: list[Node] = [root_node]

        best_plan: tuple[Act, ...] | None = None
        won_game = False
        rollout_len = self.rollout
        barren = 0

        while time.perf_counter() - t0 < budget_s and best_plan is None:
            node = self._select(order)
            node.visits += 1
            g = self._restore(root, node)
            plan = list(node.plan)
            cur_node: Node = node
            cur_valid = node.valid or valid0
            cur_frame = None
            prev: Act | None = None
            new_cells = 0
            grouped = cur_valid
            refresh = 0

            for _ in range(rollout_len):
                if len(plan) >= self.max_depth:
                    break
                if not cur_valid:
                    break
                if refresh <= 0:
                    grouped = (
                        reduce_clicks(cur_frame, cur_valid)
                        if cur_frame is not None
                        else cur_valid
                    )
                    refresh = 8
                refresh -= 1

                # Systematic before random: an action never taken from this cell
                # beats one already tried. Without this the rollout kept picking
                # the same few actions from the same cells forever and never
                # covered the reachable set.
                fresh = [
                    a for a in grouped if a not in cur_node.tried and a not in cur_node.noop
                ]
                pool = (
                    fresh
                    or [a for a in grouped if a not in cur_node.noop]
                    or list(grouped)
                )
                # Stickiness lets grid games cross a room with one repeated move,
                # but repeating a *click* is nearly always wasted, so only stick
                # on non-click actions. Purely action-type derived, not per-game.
                if (
                    prev is not None
                    and not prev.is_click
                    and prev in pool
                    and self.rng.random() < self.sticky
                ):
                    a = prev
                elif self.prior is not None and cur_frame is not None and len(pool) > 1:
                    # Learned prior, mixed with uniform so coverage is preserved:
                    # every legal action keeps at least (1-mix)/n probability, so
                    # a wrong prior slows the search but cannot make a state
                    # unreachable. Go-Explore's guarantee is coverage; the prior
                    # only reorders it.
                    p = self.prior.prior(cur_frame, pool)
                    p = self.prior_mix * p + (1.0 - self.prior_mix) / len(pool)
                    p = p / p.sum()
                    a = pool[int(self.rng.choice(len(pool), p=p))]
                else:
                    a = pool[int(self.rng.integers(len(pool)))]
                cur_node.tried.add(a)

                frame_before = cur_frame
                legal_before = tuple(pool)
                obs = Twin.step_game(g, a)
                self.steps += 1
                plan.append(a)
                prev = a

                if obs.won:
                    if self.recorder is not None:
                        self.recorder.add(frame_before, a, legal_before, "level")
                    best_plan = tuple(plan)
                    won_game = True
                    break
                if obs.level > start_level:
                    if self.recorder is not None:
                        self.recorder.add(frame_before, a, legal_before, "level")
                    best_plan = tuple(plan)
                    break
                if obs.game_over:
                    # Dead branch: file it as dead so we never restore into it.
                    if self.recorder is not None:
                        self.recorder.add(frame_before, a, legal_before, "dead")
                    dk = self.cell(obs.frame, obs.level)
                    if dk not in archive:
                        archive[dk] = Node(dk, tuple(plan), obs.level, (), dead=True)
                    break

                k = self.cell(obs.frame, obs.level)
                if k == cur_node.key:
                    # Landed in the same cell. Record it as unproductive so this
                    # cell stops re-trying it, but DO NOT remove it from `plan`.
                    #
                    # The action was really applied to `g`. The cell key is a
                    # deliberately coarse abstraction, so "same cell" does not
                    # mean "same engine state" - a counter may have moved, an
                    # object may have shifted inside a masked-out region. An
                    # earlier version popped the action here, which desynchronised
                    # `plan` from `g`: every later action in that rollout was
                    # recorded against a state it was not taken from, so a plan
                    # that completed a level in the search failed on replay.
                    # Measured cost of that bug: tu93 claimed 1 level and
                    # replayed 0, sp80 claimed 2 and replayed 1, lp85 claimed 5
                    # and replayed 4 - always the level found by the rollout that
                    # had dropped actions. compress() could not catch it because
                    # it verifies its own edits but never its starting plan.
                    cur_node.noop.add(a)
                    prev = None
                    cur_valid = obs.valid or cur_valid
                    cur_frame = obs.frame
                    continue
                cur_valid = obs.valid
                cur_frame = obs.frame

                old = archive.get(k)
                if old is None:
                    # This action reached a state the search had never seen. By
                    # the cell abstraction's own definition that is progress, so
                    # it is a correct imitation target - and there are ~10,000x
                    # more of these than there are actions in the final plan.
                    if self.recorder is not None:
                        self.recorder.add(frame_before, a, legal_before, "new")
                    # No snapshot here on purpose - see _restore's docstring.
                    nd = Node(k, tuple(plan), obs.level, obs.valid)
                    archive[k] = nd
                    order.append(nd)
                    new_cells += 1
                    cur_node = nd
                else:
                    if len(plan) < old.depth and not old.dead:
                        # Cheaper route to a known situation - keep the short one.
                        old.plan = tuple(plan)
                        old.valid = obs.valid
                        old.snap = None  # stale: taken for the longer plan
                    cur_node = old

            # VERIFY BEFORE BELIEVING.
            #
            # A rollout reports success from *inside* its own trajectory: `plan`
            # is what the rollout thinks it did, and `g` is what the engine
            # actually did. Those two can drift apart - they did, via a dropped
            # no-op action - and when they drift the search happily returns a
            # plan that never worked, which `compress` then passes through
            # untouched because it only verifies its own edits.
            #
            # A plan is worth exactly as much as a fresh engine's willingness to
            # replay it. One replay of <=max_depth actions costs ~0.5 s of
            # simulation against a 300 s budget, and it converts "claimed
            # levels" into "levels that will actually score on the gateway".
            # If it fails we throw the plan away and keep searching.
            if best_plan is not None and not self._reaches(
                root, best_plan, start_level + 1
            ):
                self.false_plans += 1
                if self.verbose:
                    print(
                        f"    L{start_level}: rejected a {len(best_plan)}-action "
                        f"plan that did not replay (#{self.false_plans})"
                    )
                best_plan = None
                won_game = False

            # Adaptive rollout length: if nothing new turns up, look further.
            if new_cells == 0:
                barren += 1
                if barren >= 12:
                    rollout_len = min(int(rollout_len * 1.5) + 8, 400)
                    barren = 0
            else:
                barren = 0
            if self.snaps >= self.snap_cap:
                self._reclaim(order)

        return LevelResult(
            level=start_level,
            plan=best_plan,
            raw_len=len(best_plan) if best_plan else 0,
            cells=len(archive),
            steps=self.steps,
            seconds=time.perf_counter() - t0,
            won_game=won_game,
        )

    # -- compression ------------------------------------------------------

    def _reaches(self, root: Any, plan: Sequence[Act], target_level: int) -> bool:
        """Does ``plan`` still complete the level? Verified by real replay."""
        g = copy.deepcopy(root)
        for a in plan:
            obs = Twin.step_game(g, a)
            self.steps += 1
            if obs.game_over:
                return False
            if obs.level >= target_level or obs.won:
                return True
        return False

    def _keys_along(self, root: Any, plan: Sequence[Act]) -> list[bytes]:
        """Cell id after each action, for loop detection during compression.

        This must use the calibrated cell key, not the raw frame. With the raw
        frame a HUD timer made every state unique, so "the same situation twice"
        never happened and the loop-splice pass below could never fire at all.
        """
        g = copy.deepcopy(root)
        keys: list[bytes] = [b"START"]
        for a in plan:
            obs = Twin.step_game(g, a)
            self.steps += 1
            keys.append(self.cell(obs.frame, obs.level))
            if obs.terminal:
                break
        return keys

    def compress(
        self, root: Any, plan: Sequence[Act], target_level: int, budget_s: float = 20.0
    ) -> tuple[Act, ...]:
        """Shrink a working plan. Every edit is verified, so it stays correct.

        Two general passes, both purely mechanical:
          * loop splice - if the same frame appears twice, cut what is between.
          * window drop - try deleting runs of 16/8/4/2/1 actions.

        Both passes verify each *edit* by replay, which is not the same as
        verifying the *input*: an invalid plan with no accepted edits used to be
        returned unchanged and counted as a solved level. So the input is
        checked first, and an unreplayable plan comes back empty - callers treat
        an empty plan as "level not solved", which is the honest answer.
        """
        t0 = time.perf_counter()
        cur = list(plan)
        if not cur or not self._reaches(root, cur, target_level):
            return ()

        # pass 1: loop removal, biggest loops first
        improved = True
        while improved and time.perf_counter() - t0 < budget_s:
            improved = False
            keys = self._keys_along(root, cur)
            first: dict[bytes, int] = {}
            cuts: list[tuple[int, int]] = []
            for i, k in enumerate(keys):
                if k in first:
                    cuts.append((first[k], i))
                else:
                    first[k] = i
            cuts.sort(key=lambda c: c[1] - c[0], reverse=True)
            for i, j in cuts:
                if j - i <= 0 or j > len(cur):
                    continue
                cand = cur[:i] + cur[j:]
                if len(cand) >= len(cur):
                    continue
                if self._reaches(root, cand, target_level):
                    cur = cand
                    improved = True
                    break

        # pass 2: window drop
        for win in (16, 8, 4, 2, 1):
            i = 0
            while i + win <= len(cur) and time.perf_counter() - t0 < budget_s:
                cand = cur[:i] + cur[i + win :]
                if self._reaches(root, cand, target_level):
                    cur = cand
                else:
                    i += 1

        return tuple(cur)


# ---------------------------------------------------------------------------
# whole-game driver
# ---------------------------------------------------------------------------


@dataclass
class GameSolution:
    game_id: str
    plan: list[Act]
    actions_per_level: list[int]
    baselines: list[int]
    levels_solved: int
    est_score: float
    steps: int
    seconds: float


def solve_game(
    game_id: str,
    *,
    env_dir: Path | None = None,
    budget_s: float = 120.0,
    per_level_cap: float | None = None,
    seed: int = 0,
    verbose: bool = True,
    max_levels: int | None = None,
    restarts: int = 3,
    recorder: Any | None = None,
    prior: Any | None = None,
    prior_mix: float = 0.6,
) -> GameSolution:
    """Search a whole game level by level; return one concatenated plan."""
    t0 = time.perf_counter()
    twin = Twin(game_id, env_dir)
    baselines = twin.baselines or [100] * twin.n_levels
    n_levels = max_levels or twin.n_levels

    root = twin.snapshot()
    # Every graded run starts with RESET; do the same here so the plan we hand
    # back is replayable verbatim from a fresh game.
    Twin.step_game(root, Act(0))

    # Learn which pixels carry state before searching. ~600 simulated actions,
    # under a second, zero graded actions. Without this the archive key is
    # bijective and Go-Explore degenerates into a random walk.
    cell = calibrate(root, seed=seed)
    if verbose:
        print(
            f"  cell key: {cell.n_informative} informative px "
            f"({cell.n_varying} varying, {cell.n_clock} clock/HUD discarded)"
        )
    ex = Explorer(
        twin,
        cell=cell,
        seed=seed,
        verbose=verbose,
        recorder=recorder,
        prior=prior,
        prior_mix=prior_mix,
    )

    full: list[Act] = []
    per_level: list[int] = []
    level = 0
    while level < n_levels and time.perf_counter() - t0 < budget_s:
        left = budget_s - (time.perf_counter() - t0)
        share = min(left, per_level_cap or left)
        base = baselines[level] if level < len(baselines) else 100

        # Restarts. solve_level is a randomised search, and randomised search on
        # this kind of problem has a heavy-tailed runtime: an unlucky opening can
        # trap a rollout distribution in a dead region for the whole budget while
        # a different seed escapes in seconds. Re-seeding and starting the
        # archive fresh is strictly better than spending the tail of the budget
        # in a search that has already stopped finding cells. The old code also
        # simply discarded the 25% reserved for compression whenever a level
        # failed, which is pure waste.
        res = None
        spent = 0.0
        for attempt in range(restarts):
            budget_here = share * 0.75 - spent
            if budget_here <= 1.0:
                break
            slice_s = budget_here if attempt == restarts - 1 else budget_here * 0.55
            ex.rng = np.random.default_rng(seed + 1013 * (level + 1) + attempt)
            a0 = time.perf_counter()
            res = ex.solve_level(root, level, base, slice_s)
            spent += time.perf_counter() - a0
            if res.plan is not None:
                break
            if verbose:
                print(
                    f"  L{level}: attempt {attempt + 1} failed  cells={res.cells:,} "
                    f"{res.seconds:.1f}s"
                )
        if res is None or res.plan is None:
            if verbose:
                cells = res.cells if res else 0
                print(
                    f"  L{level}: NOT SOLVED  cells={cells:,} "
                    f"steps={ex.steps:,} {spent:.1f}s"
                )
            break
        tight = ex.compress(root, res.plan, level + 1, budget_s=min(share * 0.25, 30.0))
        if not tight:
            # compress() returns empty only when the plan does not replay from a
            # fresh engine. Reporting the level anyway would inflate the local
            # score and score zero on the gateway, so stop here instead.
            if verbose:
                print(f"  L{level}: plan failed verification - not counted")
            break
        sc = level_score(base, len(tight))
        if verbose:
            print(
                f"  L{level}: solved raw={res.raw_len:4d} -> {len(tight):4d} "
                f"(baseline {base:3d})  score {sc:6.1f}  cells={res.cells:,} "
                f"{res.seconds:.1f}s"
            )
        full.extend(tight)
        per_level.append(len(tight))
        # Advance the root to the state right after this level completes.
        g = copy.deepcopy(root)
        for a in tight:
            Twin.step_game(g, a)
        root = g
        ex.steps += len(tight)
        level += 1
        if res.won_game:
            break

        # RE-CALIBRATE. The mask is `varies & ~clock`, learned by probing one
        # root, so it describes *that level's* screen. Measured on the next
        # level it is wrong in both directions: cd82 level 2 has 139 informative
        # pixels the level-0 mask excludes (the search is blind to 14% of the
        # state), and vc33 level 1 has 1,779 clock-like pixels where level 0 had
        # 494, so ~1,285 timer pixels get hashed into the key and it turns
        # bijective again - the exact failure the mask was introduced to fix,
        # reappearing at every level past the first. Probing costs ~600
        # simulated actions, well under a second, and zero graded actions.
        cell = calibrate(root, seed=seed + level)
        ex.cell = cell
        if verbose:
            print(
                f"  recalibrated for L{level}: {cell.n_informative} informative px "
                f"({cell.n_varying} varying, {cell.n_clock} clock/HUD discarded)"
            )

    est = game_score(baselines, per_level)
    return GameSolution(
        game_id=game_id,
        plan=full,
        actions_per_level=per_level,
        baselines=list(baselines),
        levels_solved=len(per_level),
        est_score=est,
        steps=ex.steps,
        seconds=time.perf_counter() - t0,
    )


def discover_games(env_dir: Path) -> list[str]:
    out: list[str] = []
    for meta in sorted(env_dir.rglob("*/*/metadata.json")):
        try:
            out.append(json.loads(meta.read_text(encoding="utf-8"))["game_id"])
        except Exception:
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="", help="game id prefix, or blank for all")
    ap.add_argument("--budget", type=float, default=120.0, help="seconds per game")
    ap.add_argument("--levels", type=int, default=0, help="stop after N levels (0=all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="write plans as json")
    args = ap.parse_args()

    env_dir = default_env_dir()
    ids = discover_games(env_dir)
    if args.game:
        ids = [g for g in ids if g.startswith(args.game)]
    if not ids:
        print("no matching games")
        return 1

    print(f"env_dir: {env_dir}\ngames:   {len(ids)}  budget {args.budget:.0f}s each\n")
    results: list[GameSolution] = []
    for gid in ids:
        print(f"{gid}")
        sol = solve_game(
            gid,
            env_dir=env_dir,
            budget_s=args.budget,
            seed=args.seed,
            max_levels=args.levels or None,
        )
        results.append(sol)
        print(
            f"  => {sol.levels_solved}/{len(sol.baselines)} levels, "
            f"est game score {sol.est_score:.2f}, {sol.steps:,} sim steps, "
            f"{sol.seconds:.1f}s\n"
        )

    total = sum(r.est_score for r in results) / len(results)
    print(f"MEAN ESTIMATED SCORE over {len(results)} games: {total:.3f}")
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    r.game_id: {
                        "plan": [[a.aid, a.x, a.y] for a in r.plan],
                        "actions_per_level": r.actions_per_level,
                        "baselines": r.baselines,
                        "est_score": r.est_score,
                    }
                    for r in results
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
