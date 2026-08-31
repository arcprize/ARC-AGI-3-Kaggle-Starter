"""Can the mind actually predict the game? Measure it before trusting it.

The whole plan rests on one claim: the agent can learn a copy of an unseen game
good enough to plan inside. That claim is cheap to make and easy to get wrong, so
this grades it the only honest way - by asking the copy to predict the next frame
*before* the real one arrives, on every single transition, and counting.

Three numbers matter, and they mean different things:

  ``acc``      of the predictions it committed to, how many were exactly right.
               Exactly, pixel for pixel - a near miss is a wrong plan.
  ``abst``     how often it admitted it did not know. High abstention is honest
               but useless; high accuracy at low abstention is what we want.
  ``lvl``      levels actually completed, because a perfect predictor that never
               finishes a level scores zero.

    .venv/Scripts/python.exe arc3x/measure_dream.py            # all 25
    .venv/Scripts/python.exe arc3x/measure_dream.py ls20 m0r0  # a few
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from arc3x.dream import Dream
from arc3x.explore import discover_games
from arc3x.graded import GradedRun
from arc3x.mind import Mechanics
from arc3x.twin import default_env_dir

WIGGLE = 4    # presses per action while learning the controls
WALK = 240    # transitions to grade the copy over


def measure(game_id: str, env_dir: Path, seed: int = 0) -> dict:
    run = GradedRun(game_id, env_dir)
    obs = run.reset()
    m = Mechanics()
    dream = Dream(m)
    rng = np.random.default_rng(seed)
    acts = [a for a in obs.available_actions if a != 6]
    pre = game_id.split("-")[0]
    if not acts:
        return {"game": pre, "note": "click-only", "acc": 0.0, "abst": 1.0, "lvl": 0}

    # 1. learn the controls, the cheapest information in the game
    for _ in range(WIGGLE):
        for a in acts:
            before = obs.frame
            obs = run.step(a)
            m.observe(a, before, obs.frame)
            if obs.terminal:
                obs = run.reset()
        m.settle()
    m.settle()

    # 2. grade the copy on everything that happens next, letting it learn as it goes
    thought = 0
    level = obs.levels_completed
    for _ in range(WALK):
        before = obs.frame
        # Prefer a move the dream proposes: that is the case the agent would
        # actually bet actions on, so it is the case worth grading.
        plan = dream.route(before, max_nodes=600, max_depth=14) if dream.confident else []
        a = plan[0] if plan else int(rng.choice(acts))
        if plan:
            thought += 1
        obs = run.step(a)
        # Grade first, using the model as it stood when the choice was made.
        dream.observe(a, before, obs.frame)
        m.observe(
            a,
            before,
            obs.frame,
            level_up=obs.levels_completed > level,
            died=obs.game_over,
        )
        if obs.levels_completed != level:
            dream.cut()
        level = obs.levels_completed
        if obs.terminal:
            obs = run.reset()
            dream.cut()
            m.pos = None

    graded = dream.hits + dream.misses
    total = graded + dream.abstains
    return {
        "game": pre,
        "acc": dream.acc_move,
        "still": dream.acc_still,
        "nmove": dream.hits_move + dream.misses_move,
        "abst": dream.abstains / total if total else 1.0,
        "graded": graded,
        "lvl": obs.levels_completed,
        "actions": run.actions,
        "push": sorted(dream.pushable),
        "collect": sorted(dream.collectible),
        "thought": thought,
        "calm": dream.calm,
        "lively": dream.lively,
        "avatar": m.avatar,
        "body": sorted(m.body),
        "consumed": sorted(dream.prog.consumed),
        "built": sorted(dream.prog.built),
    }


def main() -> int:
    env_dir = default_env_dir()
    games = discover_games(env_dir)
    want = sys.argv[1:]
    if want:
        games = [g for g in games if g.split("-")[0] in want]

    t0 = time.perf_counter()
    rows = []
    for gid in games:
        try:
            r = measure(gid, env_dir)
        except Exception as exc:
            print(f"{gid.split('-')[0]:6s} {type(exc).__name__}: {exc}", flush=True)
            continue
        rows.append(r)
        print(
            f"{r['game']:6s} move={r['acc']:5.0%}({r.get('nmove', 0):3d}) "
            f"still={r.get('still', 0):5.0%} abst={r['abst']:5.0%} "
            f"lvl={r['lvl']} thought={r.get('thought', 0):3d} "
            f"{'calm  ' if r.get('calm', True) else 'LIVELY'} "
            f"collect={r.get('collect', [])} "
            f"ratchet-={r.get('consumed', [])} ratchet+={r.get('built', [])}",
            flush=True,
        )

    live = [r for r in rows if r.get("nmove")]
    if live:
        print(
            f"\nmean move-accuracy {sum(r['acc'] for r in live) / len(live):.1%} over "
            f"{len(live)} games where the avatar ever moved   "
            f"({time.perf_counter() - t0:.0f}s)"
        )
        good = [r for r in live if r["acc"] >= 0.8]
        print(
            f"copy is >=80% right about where it ends up: {len(good)}/{len(rows)} "
            f"games {[r['game'] for r in good]}"
        )
        blind = [r for r in rows if not r.get("nmove")]
        print(f"no steerable avatar found at all: {len(blind)} {[r['game'] for r in blind]}")
        print(f"levels completed: {sum(r['lvl'] for r in rows)} across {len(rows)} games")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
