"""How many actions does the reactive path need to reach what ``markers`` says at frame 0?

WHY THIS EXISTS
---------------
``markers.py`` claims the destination is drawn on the board before the first
action. That claim is worth exactly nothing until it is measured, and the trap is
measuring the wrong thing: it is easy to write a detector that fires on every
game and call the coverage a success.

So this file measures **latency and agreement**, side by side, on the same
evidence stream:

  * **frame 0** - what ``markers`` proposes before a single action is spent.
  * **react@N** - the action index at which ``Dream.target_colors`` first becomes
    non-empty, which is the earliest moment the current agent has any destination
    at all.
  * **agree** - whether the colour the reactive path eventually settles on is one
    ``markers`` already named at frame 0.

Agreement plus latency is the whole claim. If the reactive path takes 137 actions
to name colour 14 and the detector named colour 14 at action 0, that is 137
actions of planning bought for nothing - and per the measured rule in
``memory/depth-and-efficiency-multiply.md`` those are the *cheapest* actions in
the game to save on level 0 and the most expensive to waste on level 1.

WHAT THIS DOES NOT MEASURE
--------------------------
**It does not measure correctness.** ``Dream.target_colors`` is itself only a
proxy for the truth - it is the thing being replaced, and ``dream.py`` documents
it firing on 960 phantom successes before the bounded-built fix. So:

  * agreement is **corroboration by an independent route**, not proof;
  * disagreement is a **flag to inspect**, not a refutation.

One row is a genuine hard check, because its ground truth was read from the
game's own source rather than inferred: **tu93's exits are tag
``0015msvpvzxhqf`` = sprite ``0014mzhhvzrazi``, a plain 3x3 of colour 14**
(``tu93.py`` lines 396-445). If the detector does not put colour 14 in a 3x3
group on tu93's first frame, it is wrong and no amount of coverage elsewhere
redeems it. ``EXPECT`` below holds that and only that, because that is all that
has actually been traced to pixels.

Two games are expected to **fail honestly**: cd82 and re86 win by matching a
reference picture, not by covering markers, and ``markers.py`` deliberately does
not implement the region-match variant. A detector that scores well on those two
would be matching something it does not understand.

WHY A RANDOM WALK
-----------------
Both sides must see identical evidence, and the reactive sources specifically
need *accidental* successes - ``collectible`` needs a colour to have vanished
underfoot, ``prog.consumed`` needs two ratchet steps. Driving the real ``Agent``
would let its policy decide the latency, which is the quantity under test. A
seeded uniform walk over the declared valid actions is the neutral, reproducible
choice. It is also pessimistic for the detector in the only direction that
matters: it hands the reactive path the easiest possible route to a bump.

Run:
    PYTHONPATH=. .venv/Scripts/python.exe arc3x/why_markers.py --steps 400
    PYTHONPATH=. .venv/Scripts/python.exe arc3x/why_markers.py --games tu93,cd82 -v
"""

from __future__ import annotations

import argparse
import random
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc3x.dream import Dream
from arc3x.explore import discover_games
from arc3x.markers import markers, summary
from arc3x.mind import Mechanics
from arc3x.percept import background
from arc3x.twin import Act, Twin, default_env_dir

# The only ground truth in this file that was read down to pixels, from
# ``tu93.py`` 396-445: the exit sprite is a 3x3 of colour 14. Everything else is
# reported for inspection rather than graded, because a table of guesses dressed
# up as expectations is worse than no table.
EXPECT: dict[str, int] = {"tu93": 14}
# cd82 and re86 win by matching a reference picture. ``markers`` does not
# implement that and should not appear to succeed on them.
REGION_MATCH = {"cd82", "re86"}


def _moved_colors(mech: Mechanics) -> set[int]:
    """Colours seen to move - pieces, not places.

    ``votes`` is keyed ``(action, colour, delta)`` and is only credited for a real
    non-background displacement (``mind.py`` line 133), so its colours are exactly
    the things that have been observed to travel. The believed avatar is added
    because it may have been identified by the located-box route (``shifts``)
    without ever contributing a parsed delta - which is the rotating-sprite case.
    """
    seen = {int(c) for (_a, c, _d) in mech.votes}
    if mech.avatar >= 0:
        seen.add(int(mech.avatar))
    return seen | {int(c) for c in mech.body}


def one(gid: str, env, *, steps: int, seed: int, verbose: bool) -> dict:
    """Play one game at random, watching both paths form their answer."""
    twin = Twin(gid, env)
    twin.current()
    # One RESET first, exactly as ``why_moves.py`` does: without it the frame is
    # the title card and every percept reads the wrong board.
    obs = Twin.step_game(twin.game, Act(5))

    mech = Mechanics()
    dream = Dream(mech)
    rng = random.Random(seed)

    bg = background(obs.frame)
    # The detector's answer, before anything has been learned or spent. ``moved``
    # is empty here by construction - that is the point of the row.
    first = markers(obs.frame, moved=set(), movers=0, bg=bg)

    react_at: int | None = None
    react_set: set[int] = set()
    level = obs.level
    spent = 0

    for i in range(steps):
        valid = [x.aid for x in obs.valid] or [1, 2, 3, 4, 5]
        aid = rng.choice(valid)
        if aid == 6:
            act = Act(aid=6, x=rng.randrange(64), y=rng.randrange(64))
        else:
            act = Act(aid=aid)
        before = obs.frame
        obs = Twin.step_game(twin.game, act)
        spent += 1
        # Order matters and is not arbitrary: ``Dream.observe`` grades its own
        # prediction against the transition and must run *before* ``Mechanics``
        # folds the same transition in, or it would be graded on a model that has
        # already seen the answer (``dream.py`` line 203).
        dream.observe(aid, before, obs.frame)
        mech.observe(aid, before, obs.frame, level_up=obs.level != level)
        if obs.level != level:
            level = obs.level
            dream.cut()
        if react_at is None and dream.target_colors:
            react_at = spent
            react_set = set(dream.target_colors)
        if obs.terminal:
            obs = Twin.step_game(twin.game, Act(5))
            spent += 1
            dream.cut()

    late = set(dream.target_colors)
    proposed = {m.color for m in first}
    top2 = {m.color for m in first[:2]}
    answer = react_set or late

    row = {
        "game": gid.split("-")[0],
        "first": first,
        "proposed": proposed,
        "top2": top2,
        "react_at": react_at,
        "react": answer,
        "late": late,
        "levels": level,
        "steps": spent,
    }
    if verbose:
        print(f"\n=== {row['game']}  bg={bg}  level reached {level}")
        print(f"    frame0 {summary(first, top=6)}")
        print(f"    moved by end: {sorted(_moved_colors(mech))}")
        print(f"    reactive: first@{react_at} {sorted(answer)}  end {sorted(late)}")
        print(f"    dream: {dream.summary() if hasattr(dream, 'summary') else ''}")
    return row


def _verdict(row: dict) -> str:
    """One word per game, and the two words that are not congratulations.

    ``EXPECT`` rows are graded against source-read pixels, so they can genuinely
    fail. Everything else can only agree, disagree, or have nothing to compare
    against - and saying so is the honest reading, not a hedge.
    """
    g = row["game"]
    if g in EXPECT:
        want = EXPECT[g]
        if want in row["top2"]:
            return "PASS(top2)"
        if want in row["proposed"]:
            return "PASS(ranked-low)"
        return f"FAIL wanted c{want}"
    if g in REGION_MATCH:
        return "n/a region-match"
    if not row["proposed"]:
        return "no proposal"
    if not row["react"]:
        return "no comparator"
    if row["react"] & row["top2"]:
        return "agrees(top2)"
    if row["react"] & row["proposed"]:
        return "agrees(ranked-low)"
    return "disagrees"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--games", default="")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    env = default_env_dir()
    allg = discover_games(env)
    if a.games:
        want = [w.strip() for w in a.games.split(",") if w.strip()]
        gids = [g for g in allg if any(g.startswith(w) for w in want)]
    else:
        gids = allg

    rows = []
    for gid in gids:
        try:
            rows.append(one(gid, env, steps=a.steps, seed=a.seed, verbose=a.verbose))
        except Exception as exc:
            # Same reason as ``smoke_relive.py``: 25 games' worth of evidence per
            # run matters more than a clean traceback, and a first game that dies
            # must not hide the 24 behind it.
            print(f"{gid.split('-')[0]:6s} !! {type(exc).__name__}: {exc}")
            traceback.print_exc()

    if not rows:
        print("no game produced a row")
        return 1

    print("\n" + "=" * 78)
    print(f"{'game':7s}{'n':>3s} {'frame0 proposal':34s}{'react@':>7s} {'reactive':14s}verdict")
    for r in sorted(rows, key=lambda r: r["game"]):
        first = r["first"]
        head = str(first[0]) if first else "-"
        at = str(r["react_at"]) if r["react_at"] else f">{r['steps']}"
        print(
            f"{r['game']:7s}{len(first):>3d} {head[:33]:34s}{at:>7s} "
            f"{str(sorted(r['react']))[:13]:14s}{_verdict(r)}"
        )

    n = len(rows)
    with_prop = [r for r in rows if r["proposed"]]
    graded = [r for r in rows if r["game"] in EXPECT]
    comparable = [
        r
        for r in rows
        if r["proposed"] and r["react"] and r["game"] not in REGION_MATCH
    ]
    agree = [r for r in comparable if r["react"] & r["proposed"]]
    lat = [r["react_at"] for r in rows if r["react_at"]]

    print("\n" + "-" * 78)
    print(f"proposes something at frame 0:  {len(with_prop)}/{n} games")
    print(f"source-graded rows passing:     "
          f"{sum(1 for r in graded if EXPECT[r['game']] in r['proposed'])}/{len(graded)}"
          f"   (the only rows that can truly fail)")
    print(f"agrees with the reactive path:  {len(agree)}/{len(comparable)} comparable")
    if lat:
        lat.sort()
        print(f"reactive latency, when it fires: median {lat[len(lat) // 2]} actions, "
              f"min {lat[0]}, max {lat[-1]}  over {len(lat)}/{n} games")
    print(f"reactive never fired at all in:  {n - len(lat)}/{n} games"
          f"   <- these are where a frame-0 proposal is the whole difference")
    print(
        "\nCoverage is not success. The number that matters is the last one: on a game\n"
        "where the reactive path never fires, the planner currently has no destination\n"
        "for the entire budget, and a frame-0 proposal is the only thing that can\n"
        "change that. Read the FAIL and disagrees rows before believing any of it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
