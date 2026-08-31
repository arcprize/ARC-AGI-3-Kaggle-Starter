"""Does "wiggle a button and see what moved" actually work? Measure it.

The whole agent rests on one hypothesis: press an action, and some colour's
pixel set will have moved as a rigid body. If that holds, the agent knows what
it is and how far it steps, which is everything needed to plan. If it does not
hold, the agent is blind and needs a different mechanism.

This probe answers it through the graded interface only - the same frames, the
same constant action list, no game source, no snapshots - so whatever it reports
is something the agent can also discover on a game nobody has seen.

    .venv/Scripts/python.exe arc3x/probe.py            # all 25
    .venv/Scripts/python.exe arc3x/probe.py ls20 vc33   # a few
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from arc3x.explore import discover_games
from arc3x.graded import GradedRun
from arc3x.percept import (
    Volatility,
    background,
    blobs,
    changed,
    moved_objects,
    tile_size,
)
from arc3x.twin import default_env_dir

REPS = 6  # presses per action; the first may be blocked by a wall


def probe(game_id: str, env_dir: Path) -> dict:
    run = GradedRun(game_id, env_dir)
    obs = run.reset()
    vol = Volatility()
    acts = [a for a in obs.available_actions if a != 6]
    clicks = 6 in obs.available_actions

    # -- wiggle every non-click action, several times, recording what moved
    shift_votes: dict[int, Counter] = {a: Counter() for a in acts}
    diff_sizes: dict[int, list[int]] = {a: [] for a in acts}
    gone = Counter()
    born = Counter()
    ticks: list[int] = []
    for a in acts:
        for _ in range(REPS):
            before = obs.frame
            obs = run.step(a)
            ticks.append(len(obs.frames))
            vol.add(before, obs.frame)
            diff_sizes[a].append(changed(before, obs.frame))
            mv, van, app = moved_objects(before, obs.frame)
            for m in mv:
                if m.moved:
                    shift_votes[a][(m.color, m.delta)] += 1
            for m in van:
                gone[m.color] += 1
            for m in app:
                born[m.color] += 1
            if obs.terminal:
                obs = run.reset()

    # An avatar colour is one that moved under at least two different actions
    # (so it is not a one-off animation) and moved consistently.
    per_color: dict[int, dict[int, tuple[int, int]]] = {}
    for a, votes in shift_votes.items():
        for (c, d), n in votes.items():
            if n >= 2:
                per_color.setdefault(c, {})[a] = d
    avatar = -1
    if per_color:
        # prefer the colour that responds to the most actions, then the smallest
        # sprite (a person picks the little thing they are steering, not the
        # whole screen scrolling)
        sizes = {int(c): int((obs.frame == c).sum()) for c in per_color}
        avatar = sorted(
            per_color, key=lambda c: (-len(per_color[c]), sizes.get(int(c), 1 << 30))
        )[0]
    deltas = per_color.get(avatar, {})

    # -- clicks: do blob centres do anything?
    click_hits = click_tries = 0
    if clicks:
        cand = [b for b in blobs(obs.frame, ignore={background(obs.frame)})][:24]
        for b in cand:
            y, x = b.center
            before = obs.frame
            obs = run.step(6, x=x, y=y)
            vol.add(before, obs.frame)
            click_tries += 1
            if changed(before, obs.frame):
                click_hits += 1
            if obs.terminal:
                obs = run.reset()

    # -- how much of the screen is a ticking counter?
    hud = int(vol.hud_mask().sum())
    static = int(vol.static_mask().sum())

    return {
        "game": game_id.split("-")[0],
        "declared": list(obs.available_actions),
        "ticks": f"{min(ticks)}-{max(ticks)}" if ticks else "-",
        "avatar": int(avatar),
        "deltas": {a: d for a, d in sorted(deltas.items())},
        "tile": tile_size(deltas) if deltas else 0,
        "n_moving_colors": len(per_color),
        "median_diff": {a: int(np.median(v)) for a, v in diff_sizes.items() if v},
        "bg": background(obs.frame),
        "n_blobs": len(blobs(obs.frame, ignore={background(obs.frame)})),
        "clicks": f"{click_hits}/{click_tries}" if click_tries else "-",
        "hud_px": hud,
        "static_px": static,
        "vanish": dict(gone.most_common(3)),
        "appear": dict(born.most_common(3)),
        "levels": obs.levels_completed,
        "actions": run.actions,
        "state": obs.state,
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
            rows.append(probe(gid, env_dir))
        except Exception as exc:
            print(f"{gid}: {type(exc).__name__}: {exc}")
            continue
        r = rows[-1]
        d = " ".join(f"{a}:{dy},{dx}" for a, (dy, dx) in r["deltas"].items()) or "-"
        print(
            f"{r['game']:6s} act={str(r['declared']):22s} tick={r['ticks']:5s} "
            f"avatar={r['avatar']:3d} tile={r['tile']:2d} blobs={r['n_blobs']:4d} "
            f"hud={r['hud_px']:4d} click={r['clicks']:6s} lvl={r['levels']} "
            f"[{d}]"
        )

    found = sum(1 for r in rows if r["avatar"] >= 0)
    print(f"\navatar found in {found}/{len(rows)} games  "
          f"({time.perf_counter() - t0:.1f}s)")
    tiles = Counter(r["tile"] for r in rows if r["avatar"] >= 0)
    print(f"tile sizes: {dict(tiles)}")
    blind = [r["game"] for r in rows if r["avatar"] < 0]
    if blind:
        print(f"no rigid mover: {blind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
