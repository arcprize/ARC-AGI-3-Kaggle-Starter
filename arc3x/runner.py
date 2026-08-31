"""Play the graded gateway using plans searched in the local twin.

THE TWO ENGINES
---------------
Graded engine: behind ``http://gateway:8001/``, ``OperationMode.COMPETITION``,
``environments_dir=""``. Every action is counted and scored. It cannot be
cloned, rewound, or inspected ahead of time.

Twin: the same game as a local Python object built from the competition dataset
in ``OperationMode.OFFLINE``. ``copy.deepcopy`` is a full state snapshot and
stepping a clone leaves the real game's action count at zero, so search here is
free and runs ~550-1000 actions/sec instead of one per 17.6 s through an LLM.

This module is the bridge: search the twin for free, replay only the winning
line against the gateway.

IDENTIFYING WHICH GAME WE ARE PLAYING
-------------------------------------
This is the part that is easy to get wrong. ``clone_game_ids`` in
``taaf/competition_arcade.py`` mints clone IDs as ``f"{prefix}{i:03d}"`` -
``c000``, ``c001``, ... - because an ``arc_agi`` competition scorecard can only
create one run per game ID, so a 25-game set has to be re-exposed under fresh
IDs to fill ~110 runs. Those IDs carry **no family prefix**, so a graded
``c047`` cannot be mapped to its local twin by name.

So we identify by observation instead: RESET, read the opening frame, and match
it against the opening frame of each local family. A game's first frame after
RESET is a deterministic fingerprint (23 of the 25 games use no randomness at
all, and no constructor accepts a seed). Exact match first, nearest-Hamming
second, with a similarity floor so a genuinely unknown game is reported as
unknown rather than silently mis-identified.

Name matching is still tried first, because it is free and correct whenever the
gateway does expose real IDs.

REPLAY IS VERIFIED, NOT ASSUMED
-------------------------------
**Read this before relying on rung 1.** The scored set is 110 private games the
agent has never seen (competition Data page, confirmed 2026-08-23), *not* clones
of the 25 that ship with the dataset. An earlier version of this docstring
claimed the opposite; that claim came from ``taaf/competition_arcade.py``, which
is explicitly a **local simulator** whose ``clone_game_ids`` mints ``k000,
k001…`` to fake 110 runs out of 25 games for harness testing. It was never
evidence about the real competition.

So ``identify`` will almost always return "unknown" on a scored game, rung 1
will not fire, and the run will be decided by rungs 2 and 3. Rung 1 remains
correct and worth keeping - it costs nothing when it does not match, and it is
how this module is tested offline against the local twins - but it is not the
scoring path.

Even when a match does occur it is not trusted: after each graded action, the
gateway's frame is compared to the frame the twin predicted, and on divergence
we stop immediately rather than burn the remaining actions on a plan that is
already wrong.

FALLBACK LADDER
---------------
1. Verified plan replay (free search, exact).
2. Student policy (``arc3x/student.py``) acting greedily on live gateway frames.
   Needed because once replay diverges we cannot clone the gateway state, so we
   cannot search from there - but a feed-forward policy needs no clone. Numpy
   only: it cannot time out or rate-limit, which is how experiment 11 turned
   2.68 local into 0.60 on Kaggle.
3. Uniform random over the legal action set. Never worse than doing nothing,
   and level 0 completions do happen by chance.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from arc3x.twin import Act, Twin, default_env_dir

# A frame this similar to a family's opening frame is taken to be that family.
# Below it we decline to guess: a wrong family means every replayed action is
# wrong, which is strictly worse than falling back to the policy.
MATCH_FLOOR = 0.97


@dataclass
class Family:
    """One of the 25 local game families, with its opening-frame fingerprint."""

    prefix: str
    game_id: str
    frame0: np.ndarray
    baselines: list[int]
    n_levels: int
    plan: list[Act] = field(default_factory=list)


def _similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of pixels that agree. 1.0 == identical frames."""
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    return float((a == b).mean())


def build_families(
    env_dir: Path | None = None,
    plans_path: str | Path | None = None,
    game_ids: Sequence[str] | None = None,
) -> list[Family]:
    """Load every local family, its opening frame, and any searched plan.

    Costs one RESET per family in the twin. All of it is free simulation.
    """
    env_dir = Path(env_dir) if env_dir else default_env_dir()
    from arc3x.explore import discover_games

    ids = list(game_ids) if game_ids else discover_games(env_dir)

    plans: dict[str, list[Act]] = {}
    if plans_path and Path(plans_path).exists():
        blob = json.load(open(plans_path))
        results = blob["results"] if isinstance(blob, dict) else blob
        for r in results:
            if r.get("plan"):
                pre = r["game_id"].split("-")[0]
                plans[pre] = [Act(int(a), int(x), int(y)) for a, x, y in r["plan"]]

    out: list[Family] = []
    for gid in ids:
        try:
            twin = Twin(gid, env_dir)
            root = twin.snapshot()
            frame0 = Twin.step_game(root, Act(0)).frame
        except Exception:
            continue
        pre = gid.split("-")[0]
        out.append(
            Family(
                prefix=pre,
                game_id=gid,
                frame0=np.asarray(frame0),
                baselines=list(twin.baselines or []),
                n_levels=int(twin.n_levels),
                plan=plans.get(pre, []),
            )
        )
    return out


def identify(
    families: Sequence[Family], graded_game_id: str, frame0: np.ndarray
) -> tuple[Family | None, float, str]:
    """Which family is this graded game? Returns (family, similarity, how)."""
    # 1. Real ID exposed -> free and exact.
    pre = str(graded_game_id).split("-")[0].lower()
    for f in families:
        if f.prefix.lower() == pre:
            return f, 1.0, "name"

    # 2. Opaque clone ID (c000, ...) -> match the opening frame.
    best: Family | None = None
    best_sim = -1.0
    for f in families:
        s = _similarity(np.asarray(frame0), f.frame0)
        if s > best_sim:
            best, best_sim = f, s
    if best is not None and best_sim >= MATCH_FLOOR:
        return best, best_sim, "frame"
    return None, max(best_sim, 0.0), "unknown"


# -- the graded side --------------------------------------------------------


@dataclass
class GradedGame:
    """Minimal interface this runner needs from the graded environment.

    Kept as a protocol-ish adapter so the same runner drives the real gateway,
    TAAF's local ``CompetitionArcadeServer``, or a plain local twin in tests.
    """

    step: Callable[[Act], Any]      # -> object with .frame/.level/.valid/.terminal
    valid: Callable[[], Sequence[Act]]
    reset: Callable[[], Any]


@dataclass
class PlayResult:
    graded_game_id: str
    family: str | None
    how: str
    similarity: float
    actions_used: int
    levels_reached: int
    diverged_at: int | None
    source: str          # which rung of the ladder produced the actions
    seconds: float


def play_game(
    graded: GradedGame,
    families: Sequence[Family],
    *,
    graded_game_id: str,
    action_cap: int = 800,
    student: Any | None = None,
    rng: np.random.Generator | None = None,
    verbose: bool = True,
) -> PlayResult:
    """Play one graded game: identify it, replay its plan, fall back if needed."""
    t0 = time.perf_counter()
    rng = rng or np.random.default_rng(0)

    obs = graded.reset()
    frame = np.asarray(getattr(obs, "frame", np.zeros((64, 64), dtype=np.int8)))
    level = int(getattr(obs, "level", 0))
    used = 1  # the RESET itself is a graded action

    fam, sim, how = identify(families, graded_game_id, frame)
    if verbose:
        name = fam.prefix if fam else "UNKNOWN"
        print(f"  {graded_game_id}: {name} via {how} (similarity {sim:.4f})")

    diverged_at: int | None = None
    source = "none"

    # -- rung 1: verified plan replay -------------------------------------
    if fam is not None and fam.plan:
        source = "plan"
        # Mirror the plan in the twin so we know what each frame *should* be.
        twin = Twin(fam.game_id, default_env_dir())
        mirror = twin.snapshot()
        Twin.step_game(mirror, Act(0))
        for i, a in enumerate(fam.plan):
            if used >= action_cap:
                break
            want = Twin.step_game(mirror, a)
            got = graded.step(a)
            used += 1
            gframe = np.asarray(getattr(got, "frame", None))
            level = int(getattr(got, "level", level))
            if _similarity(gframe, want.frame) < MATCH_FLOOR:
                # The clone is not the family we searched. Everything after this
                # point in the plan is meaningless; stop before wasting it.
                diverged_at = i
                if verbose:
                    print(
                        f"    diverged at action {i}/{len(fam.plan)} "
                        f"(sim {_similarity(gframe, want.frame):.3f}) -> fallback"
                    )
                break
            frame = gframe
            if getattr(got, "terminal", False):
                break

    # -- rungs 2 and 3: act on live frames, no cloning required ------------
    if (fam is None or not fam.plan or diverged_at is not None) and used < action_cap:
        source = "student" if student is not None else "random"
        while used < action_cap:
            legal = list(graded.valid())
            if not legal:
                break
            if student is not None:
                p = student.prior(frame, legal)
                a = legal[int(rng.choice(len(legal), p=p))]
            else:
                a = legal[int(rng.integers(len(legal)))]
            got = graded.step(a)
            used += 1
            frame = np.asarray(getattr(got, "frame", frame))
            level = int(getattr(got, "level", level))
            if getattr(got, "terminal", False):
                break

    return PlayResult(
        graded_game_id=graded_game_id,
        family=fam.prefix if fam else None,
        how=how,
        similarity=sim,
        actions_used=used,
        levels_reached=level,
        diverged_at=diverged_at,
        source=source,
        seconds=time.perf_counter() - t0,
    )


# -- the real gateway -------------------------------------------------------


def gateway_as_graded(env: Any) -> GradedGame:
    """Wrap an ``arc_agi`` COMPETITION-mode environment as a GradedGame.

    ``RemoteEnvironmentWrapper`` exposes ``reset()`` and
    ``step(action, data={"x": .., "y": ..})``, both returning a ``FrameDataRaw``
    with ``.frame`` (a list of frames), ``.available_actions``, ``.state`` and
    ``.levels_completed``. Everything here is one HTTP round trip per action, so
    the plan we replay has to be short - which is exactly what the free search
    and the compression pass are for.

    One asymmetry to know about: the gateway reports ACTION6 as a single legal
    action with no coordinates, because it will not enumerate 4,096 clicks. The
    twin *does* return concrete click coordinates. That is fine for replay, where
    we supply our own coordinates, but it means the fallback policy has to choose
    the coordinate itself - which is why the student has a 256-cell click head.
    """
    from arc3x.twin import ACTION_BY_ID

    def _obs(fd: Any) -> Any:
        frames = getattr(fd, "frame", None) or []
        frame = (
            np.asarray(frames[-1], dtype=np.int8)
            if frames
            else np.zeros((64, 64), dtype=np.int8)
        )
        state = str(getattr(fd, "state", "") or "")
        return _GatedObs(
            frame=frame,
            level=int(getattr(fd, "levels_completed", 0) or 0),
            terminal=state.upper() in {"GAME_OVER", "WIN"},
            raw=fd,
        )

    last: dict[str, Any] = {"fd": None}

    def _reset() -> Any:
        fd = env.reset()
        last["fd"] = fd
        return _obs(fd)

    def _step(a: Act) -> Any:
        ga = ACTION_BY_ID[a.aid]
        data = {"x": int(a.x), "y": int(a.y)} if a.is_click else None
        fd = env.step(ga, data=data)
        last["fd"] = fd
        return _obs(fd)

    def _valid() -> Sequence[Act]:
        fd = last["fd"]
        raw = list(getattr(fd, "available_actions", None) or [])
        out: list[Act] = []
        for item in raw:
            aid = getattr(item, "value", item)
            try:
                aid = int(aid)
            except (TypeError, ValueError):
                # Some builds report names ("ACTION3"); map back through the table.
                name = str(aid).upper().replace("GAMEACTION.", "")
                aid = next(
                    (k for k, v in ACTION_BY_ID.items() if str(v).upper().endswith(name)),
                    None,
                )
                if aid is None:
                    continue
            if aid == 6:
                # No coordinates from the gateway; the caller's policy picks one.
                out.append(Act(6, 0, 0))
            elif aid in ACTION_BY_ID:
                out.append(Act(int(aid)))
        return tuple(out)

    return GradedGame(step=_step, valid=_valid, reset=_reset)


@dataclass
class _GatedObs:
    frame: np.ndarray
    level: int
    terminal: bool
    raw: Any = None


# -- offline self-test ------------------------------------------------------


def twin_as_graded(game_id: str, env_dir: Path | None = None) -> GradedGame:
    """Wrap a local twin behind the GradedGame interface.

    Lets the whole runner - identification, replay, verification, fallback - be
    tested without a gateway. Not a substitute for a real gateway check of
    ``ONLY_RESET_LEVELS`` semantics, which this cannot observe.
    """
    twin = Twin(game_id, Path(env_dir) if env_dir else default_env_dir())
    state: dict[str, Any] = {"g": twin.snapshot()}

    def _reset() -> Any:
        state["g"] = twin.snapshot()
        return Twin.step_game(state["g"], Act(0))

    return GradedGame(
        step=lambda a: Twin.step_game(state["g"], a),
        valid=lambda: Twin.valid_actions(state["g"]),
        reset=_reset,
    )
