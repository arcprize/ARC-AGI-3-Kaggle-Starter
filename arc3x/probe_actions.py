"""Does every button that matters get used? Classify actions by what they do.

WHY ASK
-------
``Mechanics.moves`` keeps only the buttons that *translate the avatar*, because
those are the ones a route is made of. Everything else is counted as a no-op and
never appears in a plan again.

That quietly throws away a whole class of button. ``wa30``'s ACTION5 grabs or
interacts with whatever the avatar is facing; a select-then-act game needs a
press that changes which object is under control; other games have a use, drop or
rotate. None of those move you, all of them change the board, and a person uses
them constantly - you walk up to the thing and press the use button.

So there are three kinds of action, not two:

  MOVE   the avatar translates                     -> routes are made of these
  ACT    the frame changes, the avatar does not    -> currently thrown away
  DEAD   nothing ever happens                      -> correctly ignored

This measures how many of the 25 games have an ACT button, and how much of the
board it moves, which decides whether "walk there, then press use" is worth
building as a general capability.

    .venv/Scripts/python.exe arc3x/probe_actions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from arc3x.explore import discover_games
from arc3x.graded import GradedRun
from arc3x.mind import Mechanics
from arc3x.twin import default_env_dir

REPS = 6


def probe(game_id: str, env_dir: Path) -> dict:
    run = GradedRun(game_id, env_dir)
    obs = run.reset()
    m = Mechanics()
    acts = [a for a in obs.available_actions if a != 6]
    # per action: how often it changed the frame, and how often the avatar moved
    changed = {a: 0 for a in acts}
    shifted = {a: 0 for a in acts}
    tried = {a: 0 for a in acts}
    px = {a: 0 for a in acts}

    for rep in range(REPS):
        for a in acts:
            before = obs.frame
            box0 = m.where(before) if m.avatar >= 0 else None
            obs = run.step(a)
            m.observe(a, before, obs.frame, died=obs.game_over)
            tried[a] += 1
            n = int((before != obs.frame).sum())
            if n:
                changed[a] += 1
                px[a] += n
                box1 = m.where(obs.frame) if m.avatar >= 0 else None
                if box0 and box1 and box0[:2] != box1[:2]:
                    shifted[a] += 1
            if obs.terminal:
                obs = run.reset()
                m.pos = None
        m.settle()
    m.settle()

    kinds: dict[int, str] = {}
    for a in acts:
        if not changed[a]:
            kinds[a] = "DEAD"
        elif shifted[a]:
            kinds[a] = "MOVE"
        else:
            kinds[a] = "ACT"
    return {
        "game": game_id.split("-")[0],
        "kinds": kinds,
        "px": {a: (px[a] // max(1, changed[a])) for a in acts},
        "in_moves": sorted(m.moves),
        "avatar": m.avatar,
    }


def main() -> int:
    env_dir = default_env_dir()
    games = discover_games(env_dir)
    want = sys.argv[1:]
    if want:
        games = [g for g in games if g.split("-")[0] in want]

    rows = []
    for gid in games:
        try:
            r = probe(gid, env_dir)
        except Exception as exc:
            print(f"{gid.split('-')[0]:6s} {type(exc).__name__}: {exc}", flush=True)
            continue
        rows.append(r)
        acted = [a for a, k in r["kinds"].items() if k == "ACT"]
        # An ACT button that the planner cannot see is wasted capability, and
        # that is the whole point of the measurement.
        lost = [a for a in acted if a not in r["in_moves"]]
        print(
            f"{r['game']:6s} "
            + " ".join(f"{a}:{r['kinds'][a]}({r['px'][a]}px)" for a in sorted(r["kinds"]))
            + f"  planner-sees={r['in_moves']}"
            + (f"  IGNORES-ACT={lost}" if lost else ""),
            flush=True,
        )

    has_act = [r for r in rows if any(k == "ACT" for k in r["kinds"].values())]
    print(
        f"\n{len(has_act)}/{len(rows)} games have a button that changes the board "
        f"without moving the avatar: {[r['game'] for r in has_act]}"
    )
    print(
        "read: every one of those is a 'use' button the planner currently cannot\n"
        "include in a route. If the count is high, 'walk there then press use' is\n"
        "a general capability worth building, not a per-game special case."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
