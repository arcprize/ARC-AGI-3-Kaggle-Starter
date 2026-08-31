"""An honest offline replica of the graded run. This is the measuring stick.

WHY THIS EXISTS
---------------
The evaluation set is 110 private games the agent has never seen; the 25 shipped
games are a development set. So the only number worth optimising is "what would
this agent score on a game it cannot look inside", and every previous local
measurement in this repo answered a different, easier question.

Three specific advantages the local search has been taking, all of which the
graded gateway withholds:

1. ``copy.deepcopy(game)``. Free state snapshots, free rewind, free replay.
   Behind the gateway there is no game object at all - only HTTP.
2. ``game._get_valid_actions()``. Its own docstring says "this method is for
   internal use only, the data here is never exposed via the API or to
   Users/Agents" (arcengine/base_game.py). It returns the *state-dependent*
   legal set, and for ACTION6 the *concrete legal click coordinates*. The API
   exposes ``available_actions``, which is the constant list handed to the
   constructor - typically ``[1,2,3,4,5,6]``. An agent that reads valid actions
   is being told which of 4,096 clicks are live; a graded agent has to work that
   out from the picture, like a person does.
3. ``baseline_actions``. Present locally, stripped from the competition API.
   The scoring server keeps them, so they are used here to *score* the run but
   are never shown to the agent.

WHAT IT COSTS TO ACT, EXACTLY
-----------------------------
Copied from arc_agi/scorecard.py so the arithmetic cannot drift:

* ``Card.inc_action_count`` - any action in 1..7 costs 1. Illegal or no-op
  actions cost the same as useful ones.
* ``Card.inc_reset_count`` - a RESET costs 1 as well (it increments both
  ``resets`` and ``actions``). RESET is not free.
* ``Card.set_levels_completed`` - when the level counter changes, the pair
  ``(levels_completed, actions_so_far)`` is appended to ``actions_by_level``.
* ``EnvironmentScorecard._calculate_score`` - level *i* is charged
  ``actions_by_level[i] - actions_by_level[i-1]``: **every action spent while
  stuck on a level is billed to that level, and to no other.**
* ``EnvironmentScoreCalculator`` - per level ``min(115, (baseline/actions)^2 *
  100)`` if completed, else 0; game score is the weight-1..n mean, capped by
  ``sum(weights of scoring levels) / sum(all weights) * 100``.

The two consequences that should drive any agent design:

* **Level 0 is nearly free.** On a 9-level game the weights sum to 45, so level
  0 can absorb 2.2% of the achievable score. Thousands of actions can be spent
  there learning the game for almost nothing.
* **Efficiency beats depth, quadratically.** Four levels at 10x baseline scores
  0.22; two levels at baseline scores 6.67. Completing a level sloppily is
  worth roughly nothing, so exploration has to be paid for out of level 0's
  budget and the later levels have to be played nearly clean.

RESET SEMANTICS, SETTLED
------------------------
``ONLY_RESET_LEVELS`` turns out to be almost a no-op. ``handle_reset()`` routes
to ``full_reset()`` only when ``_action_count == 0`` or the state is WIN, and
``level_reset()`` otherwise - and ``level_reset()`` never zeroes
``_action_count``. So with or without the flag the behaviour an agent sees is:
the first RESET starts the play, later RESETs restore the *current level* from
its pristine clone, and a RESET after a win starts a fresh play. Since the
scorecard keeps ``max(run.score for run in runs)``, a fully-won game can be
replayed for a better score - the only way to get a second chance at one.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from arc3x.twin import Twin, default_env_dir


@dataclass
class GObs:
    """Exactly what the competition API returns, and nothing more.

    ``FrameDataRaw`` carries ``frame`` (a *list* of frames, one per engine tick
    of the action), ``state``, ``levels_completed`` and ``available_actions``.
    The list matters: an action can animate over several ticks, and the
    intermediate frames are real information about the dynamics.
    """

    frame: np.ndarray
    frames: list[np.ndarray]
    available_actions: tuple[int, ...]
    state: str
    levels_completed: int
    full_reset: bool = False

    @property
    def game_over(self) -> bool:
        return self.state == "GAME_OVER"

    @property
    def won(self) -> bool:
        return self.state == "WIN"

    @property
    def terminal(self) -> bool:
        return self.game_over or self.won


def level_score(baseline: int, actions: int) -> float:
    """``EnvironmentScoreCalculator.add_level``, verbatim."""
    if actions <= 0:
        return 0.0
    return min(115.0, (baseline / actions) ** 2 * 100.0)


def score_from_card(
    baselines: Sequence[int],
    actions_by_level: Sequence[tuple[int, int]],
    total_actions: int,
) -> tuple[float, list[float], list[int]]:
    """``EnvironmentScorecard._calculate_score`` + ``to_score``, verbatim.

    Returns (game score, per-level scores, per-level actions charged).
    """
    scores: list[float] = []
    charged: list[int] = []
    indices: list[int] = []
    prev = 0
    for i in range(len(baselines)):
        if i < len(actions_by_level):
            _lvl, at = actions_by_level[i]
            used = at - prev
            prev = at
            done = True
        else:
            used = total_actions - prev
            prev = total_actions
            done = False
        s = level_score(baselines[i], used) if done else 0.0
        scores.append(s)
        charged.append(used)
        indices.append(i + 1)

    total = 0.0
    tw = 0
    mw = 0
    for i, s in enumerate(scores):
        w = indices[i]
        total += s * w
        tw += w
        if s > 0:
            mw += w
    if tw == 0:
        return 0.0, scores, charged
    return min(total / tw, mw / tw * 100.0), scores, charged


@dataclass
class GradedRun:
    """One graded play of one game, billed the way Kaggle bills it.

    The engine is a local twin, but the object is never handed out and no
    snapshot is ever taken, so an agent holding a ``GradedRun`` has exactly the
    powers it would have against ``http://gateway:8001``.
    """

    game_id: str
    env_dir: Path | None = None
    verbose: bool = False

    actions: int = 0
    resets: int = 0
    levels_completed: int = 0
    actions_by_level: list[tuple[int, int]] = field(default_factory=list)
    state: str = "NOT_PLAYED"
    # level -> what the agent tried there and how each attempt ended. The mean
    # score says a game failed; this says *how*, which is the difference between
    # "the planner had no destination" and "it had one and could not reach it".
    # Those two need opposite fixes and the mean cannot tell them apart.
    notes: dict[int, Counter] = field(default_factory=dict)
    _twin: Any = None
    _game: Any = None
    _baselines: list[int] = field(default_factory=list)
    _n_levels: int = 0
    _declared: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        self._twin = Twin(self.game_id, self.env_dir or default_env_dir())
        # The live engine object. Deliberately not snapshotted anywhere.
        self._game = self._twin.game
        self._baselines = list(self._twin.baselines or [])
        self._n_levels = int(self._twin.n_levels)
        declared = list(getattr(self._game, "_available_actions", []) or [])
        self._declared = tuple(int(a) for a in declared if int(a) != 0)

    # -- the only two things an agent may do ------------------------------

    def reset(self) -> GObs:
        """RESET. Costs one action, same as any other (Card.inc_reset_count)."""
        obs = self._raw(0, 0, 0)
        self.actions += 1
        self.resets += 1
        self._bookkeep(obs)
        return obs

    def step(self, aid: int, x: int = 0, y: int = 0) -> GObs:
        """One action. Illegal and no-op actions are billed like any other."""
        if aid == 0:
            return self.reset()
        obs = self._raw(aid, x, y)
        self.actions += 1
        self._bookkeep(obs)
        return obs

    # -- internals ---------------------------------------------------------

    def _raw(self, aid: int, x: int, y: int) -> GObs:
        from arcengine import ActionInput, GameAction

        ga = GameAction.from_id(aid)
        data = {"x": int(x), "y": int(y)} if aid == 6 else {}
        fd = self._game.perform_action(ActionInput(id=ga, data=data), raw=True)
        frames = [np.asarray(f, dtype=np.int8) for f in (getattr(fd, "frame", None) or [])]
        last = frames[-1] if frames else np.zeros((64, 64), dtype=np.int8)
        st = getattr(fd, "state", None)
        return GObs(
            frame=last,
            frames=frames,
            available_actions=self._declared,
            state=getattr(st, "name", str(st)),
            levels_completed=int(getattr(fd, "levels_completed", 0) or 0),
            full_reset=bool(getattr(fd, "full_reset", False)),
        )

    def _bookkeep(self, obs: GObs) -> None:
        self.state = obs.state
        if obs.levels_completed != self.levels_completed:
            # Card.set_levels_completed: record (level, cumulative actions).
            self.actions_by_level.append((obs.levels_completed, self.actions))
            if self.verbose:
                print(f"    level {obs.levels_completed} at action {self.actions}")
        self.levels_completed = obs.levels_completed

    # -- diagnostics -------------------------------------------------------

    def note(self, reason: str, level: int | None = None) -> None:
        """Record how one attempt ended, against the level it was spent on.

        ``level`` is the level the attempt *started* on, which is not always the
        current one: the attempt that clears a level advances the counter before
        the note is written, and filing "cover completed a level" under the level
        it moved us *to* points the blame at exactly the wrong place.

        This is the only channel out of an agent, because ``agent_fn`` returns
        nothing and the score alone cannot distinguish a game the agent never
        understood from one it understood and could not execute.
        """
        lvl = self.levels_completed if level is None else int(level)
        self.notes.setdefault(lvl, Counter())[reason] += 1

    # -- results -----------------------------------------------------------

    @property
    def n_levels(self) -> int:
        return self._n_levels

    def score(self) -> float:
        return self.report()["score"]

    def report(self) -> dict:
        base = self._baselines or [100] * self._n_levels
        sc, per, charged = score_from_card(base, self.actions_by_level, self.actions)
        return {
            "game_id": self.game_id,
            "score": sc,
            "levels_completed": self.levels_completed,
            "n_levels": self._n_levels,
            "actions": self.actions,
            "resets": self.resets,
            "state": self.state,
            "level_scores": [round(s, 2) for s in per],
            "level_actions": charged,
            "baselines": base,
            # Plain dicts so the whole report survives a process boundary.
            "notes": {lvl: dict(c) for lvl, c in sorted(self.notes.items())},
            # Where the run died, and what it cost. The level that ate the budget
            # is the one worth fixing; every other level is noise.
            "stalled_on": self.levels_completed,
            "stall_cost": charged[self.levels_completed]
            if self.levels_completed < len(charged)
            else 0,
        }


# -- suite ------------------------------------------------------------------


def _play_one(job: tuple) -> dict:
    """One game, start to finish, in whatever process picks it up.

    The ``GradedRun`` is built *here* rather than passed in: it holds a live
    engine object that has no business crossing a process boundary. Only the
    game id, the env path and the finished report ever travel.
    """
    gid, env_dir, agent_fn, agent_kw = job
    run = GradedRun(gid, env_dir)
    try:
        agent_fn(run, **agent_kw)
    except Exception as exc:  # one broken game must not sink the suite
        run.note(f"crash:{type(exc).__name__}")
    rep = run.report()
    return rep


def _line(rep: dict) -> str:
    """One game's result, with the reason it stopped where it did."""
    notes = rep.get("notes", {})
    stall = notes.get(rep.get("stalled_on", 0), {})
    why = ",".join(f"{k}x{v}" for k, v in sorted(stall.items(), key=lambda kv: -kv[1])[:4])
    return (
        f"  {rep['game_id'].split('-')[0]:6s} {rep['levels_completed']}/{rep['n_levels']} "
        f"levels  score {rep['score']:6.2f}  actions {rep['actions']:6d}  "
        f"charged {rep['level_actions'][:4]} vs base {rep['baselines'][:4]}"
        + (f"\n         stuck on L{rep['stalled_on']} for {rep['stall_cost']} actions: {why}" if why else "")
    )


def run_suite(
    agent_fn: Any,
    game_ids: Sequence[str] | None = None,
    *,
    env_dir: Path | None = None,
    verbose: bool = True,
    workers: int = 1,
    **agent_kw: Any,
) -> dict:
    """Play every game once with ``agent_fn(run, **kw)`` and average the scores.

    The mean over games is what the leaderboard reports, so this returns the
    same statistic - computed on the development set, which is the closest
    honest proxy available for a private set of unseen games.

    ``workers`` runs the games concurrently. Every game is fully independent -
    its own ``Twin``, its own engine, no shared state and no writes - so this is
    exact, not approximate. It matters because a serial pass over 25 games at a
    3000-action budget takes the better part of an hour, and a measurement that
    slow gets run once instead of after every change, which is how a regression
    survives. Keep ``workers=1`` when a traceback matters more than the clock.
    """
    from arc3x.explore import discover_games

    env_dir = Path(env_dir) if env_dir else default_env_dir()
    ids = list(game_ids) if game_ids else discover_games(env_dir)
    jobs = [(gid, env_dir, agent_fn, dict(agent_kw)) for gid in ids]

    reports: list[dict] = []
    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        import os

        # Set *before* the pool exists. With 'spawn' the child re-imports this
        # module - and numpy with it - so the BLAS thread pool is sized at import
        # time and an env var set inside the worker arrives too late. Without
        # this, 6 workers each claim all 12 cores, 72 threads thrash over 12, and
        # games slow by more than an order of magnitude: measured ls20 at 32
        # actions in 60s where serial did 401. On 64x64 boards single-threaded
        # numpy is the right choice anyway; the threading only ever cost us.
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ.setdefault(var, "1")

        # 'spawn' on Windows re-imports this module in the child, so agent_fn has
        # to be importable by name - a module-level function, not a lambda or a
        # bound method. play_agent in arc3x.agent satisfies that.
        with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            for rep in pool.imap_unordered(_play_one, jobs):
                reports.append(rep)
                if verbose:
                    print(_line(rep), flush=True)
        # imap_unordered returns in completion order; restore the caller's order
        # so two runs are diffable line by line.
        order = {gid: i for i, gid in enumerate(ids)}
        reports.sort(key=lambda r: order.get(r["game_id"], 0))
    else:
        for job in jobs:
            rep = _play_one(job)
            reports.append(rep)
            if verbose:
                print(_line(rep), flush=True)

    mean = sum(r["score"] for r in reports) / max(1, len(reports))
    if verbose:
        print(stall_summary(reports), flush=True)
    return {"mean_score": mean, "reports": reports}


def stall_summary(reports: Sequence[dict]) -> str:
    """Where the whole suite is losing, ranked.

    Reports coverage out of the full set next to the mean, always. A mean of
    0.142 over 25 games with 23 zeros in it reads like a small positive result
    and is in fact one game working, so the count has to travel with the number.
    """
    n = len(reports)
    cleared = Counter()
    for r in reports:
        cleared[r["levels_completed"]] += 1
    reasons: Counter = Counter()
    for r in reports:
        for reason, k in r.get("notes", {}).get(r.get("stalled_on", 0), {}).items():
            reasons[reason] += k
    timed = [
        r["game_id"].split("-")[0]
        for r in reports
        if any("TIMEOUT" in c for c in r.get("notes", {}).values())
    ]
    lines = [
        f"MEAN {sum(r['score'] for r in reports) / max(1, n):.3f}   over {n} games",
        "levels cleared:  " + "  ".join(f"{k} lvl x{v}" for k, v in sorted(cleared.items())),
        f"  reached L1+: {sum(v for k, v in cleared.items() if k >= 1)}/{n}"
        f"   reached L2+: {sum(v for k, v in cleared.items() if k >= 2)}/{n}",
        "why the run stopped, summed over games: "
        + ("  ".join(f"{k}={v}" for k, v in reasons.most_common(8)) or "(no notes)"),
    ]
    if timed:
        # Loud, because this invalidates the comparison rather than degrading it.
        lines.append(
            f"!! {len(timed)} game(s) cut off by the WALL CLOCK, not the action "
            f"budget: {timed}\n   this number is not comparable to another run - "
            f"raise --seconds or lower --workers"
        )
    return "\n".join(lines)
