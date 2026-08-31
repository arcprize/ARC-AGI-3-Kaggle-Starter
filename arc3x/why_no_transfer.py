"""Why the distilled policy scored zero on unseen games. A diagnosis, measured.

THE RESULT BEING EXPLAINED
--------------------------
``arc3x/transfer.py`` trained one student on the 17 TUNE games and tested it on
the 8 HOLD games, which no tuning has touched:

    tune (trained on)   0.732 vs 0.195 random   3.75x
    HOLD (never seen)   0.193 vs 0.176 random   1.10x   <- nothing
    HOLD policy play    0.000 vs 0.000 random   +0.000  <- nothing

So the model learns perfectly well and transfers not at all. Since the scored set
is 110 games the agent has never seen, "not at all" is the number that counts.

THE HYPOTHESIS
--------------
The student maps ``frame -> action id``. That function can only transfer if both
of its ends mean the same thing in every game. Neither obviously does:

* **The output end.** ACTION1..ACTION5 are just wire ids. Nothing in the
  competition spec says ACTION1 is "up". If it is up in one game and left in the
  next, then a model that learned "this frame -> ACTION1" is emitting a *wire
  id* when the transferable fact was a *direction*, and it will be wrong roughly
  as often as the ids disagree.
* **The input end.** Colour indices are per-game too. The student's input is a
  one-hot over absolute colour index, so "the avatar is colour 4" is learned as a
  fact about pixel values. If the avatar is colour 9 in the next game, the
  feature that fired for "me" now fires for a wall.

If both hold, the student was fitting a function that does not exist across
games, and no amount of extra teacher data fixes it - which matters, because the
cheap response to a 1.10x would otherwise be "run a longer sweep".

WHAT IS MEASURED
----------------
Both ends, per game, learned **online from observation only** - the same way an
agent must learn them on a hidden game. ``Mechanics.observe``/``settle``
(``arc3x/mind.py``) votes on which colour is the avatar and what displacement
each button produces; this presses every declared button a few times and reads
off what it concluded. No engine source is read and nothing is keyed on game id,
so the procedure is exactly what would run on one of the 110.

Then the two agreement numbers: for each button id, the largest fraction of
games that agree on its direction, and the same for the avatar colour. Those
fractions are the ceiling on transfer for a policy indexed by wire id over
absolute colours.

    .venv/Scripts/python.exe arc3x/why_no_transfer.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc3x.explore import discover_games
from arc3x.graded import GradedRun
from arc3x.mind import Mechanics
from arc3x.suite import HOLD
from arc3x.twin import default_env_dir

REPS = 8  # presses per button; Mechanics.settle wants >= 2 agreeing votes


def compass(d: tuple[int, int] | None) -> str:
    """A displacement as a direction name, ignoring magnitude.

    Magnitude is dropped on purpose: a game with a 4-pixel tile and one with an
    8-pixel tile can still agree that ACTION1 goes up, and it is the agreement
    that is in question here, not the tile size.
    """
    if d is None:
        return "-"
    dy, dx = d
    if dy == 0 and dx == 0:
        return "0"
    return ("N" if dy < 0 else "S" if dy > 0 else "") + ("W" if dx < 0 else "E" if dx > 0 else "")


def learn(game_id: str, env_dir: Path, reps: int = REPS) -> dict:
    """Press every declared button ``reps`` times; report what Mechanics believes.

    Deliberately uses ``GradedRun`` rather than a ``Twin``: it bills every action
    the way Kaggle does, so the printed cost is the real price of this probe on a
    hidden game.
    """
    run = GradedRun(game_id, env_dir, verbose=False)
    obs = run.reset()
    m = Mechanics()
    acts = [a for a in obs.available_actions if a != 6]

    for _ in range(reps):
        for a in acts:
            before = obs.frame
            obs = run.step(a)
            m.observe(a, before, obs.frame, died=obs.game_over)
            if obs.terminal:
                obs = run.reset()
                m.pos = None
        m.settle()
    m.settle()

    return {
        "game": game_id.split("-")[0],
        "declared": acts,
        "avatar": int(m.avatar),
        "background": int(m.background),
        "deltas": {int(a): (int(d[0]), int(d[1])) for a, d in m.deltas.items()},
        "assumed": {int(a) for a in m.assumed},
        "clicks": 6 in obs.available_actions,
        "cost": run.actions,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="*", help="short ids; default all 25")
    ap.add_argument("--reps", type=int, default=REPS)
    a = ap.parse_args(argv)

    env_dir = default_env_dir()
    ids = discover_games(env_dir)
    if a.games:
        ids = [g for g in ids if g.split("-")[0] in a.games]

    rows: list[dict] = []
    print(f"{'game':6s} {'avatar':>6s} {'bg':>3s} {'click':>5s}  "
          + "  ".join(f"A{i}" for i in range(1, 6)) + "   cost")
    for gid in ids:
        try:
            r = learn(gid, env_dir, a.reps)
        except Exception as exc:
            print(f"{gid.split('-')[0]:6s} {type(exc).__name__}: {exc}", flush=True)
            continue
        rows.append(r)
        cells = []
        for i in range(1, 6):
            if i not in r["declared"]:
                cells.append(" . ")          # not offered by this game
            else:
                # lowercase = filled in from the convention rather than observed,
                # so an assumption can never be mistaken for a measurement.
                d = compass(r["deltas"].get(i))
                cells.append(f"{d.lower() if i in r['assumed'] else d:>3s}")
        star = "*" if r["game"] in HOLD else " "
        print(
            f"{r['game']:5s}{star} {r['avatar']:6d} {r['background']:3d} "
            f"{'yes' if r['clicks'] else '  -':>5s}  " + " ".join(cells)
            + f"  {r['cost']:5d}",
            flush=True,
        )

    if not rows:
        return 1
    n = len(rows)
    print(f"\n(* = holdout game.  '.' = button not offered, '-' = no delta learned,"
          f"\n '0' = pressed but the avatar never moved, i.e. an ACT or dead button)")

    # -- the output end: does a button id mean one thing? --------------------
    print("\n--- does a button id mean the same thing in every game? ---------")
    print(f"{'button':8s} {'games offering':>14s} {'directions seen':>16s} "
          f"{'modal':>18s} {'agreement':>10s}")
    agree: list[float] = []
    for i in range(1, 6):
        seen = [compass(r["deltas"].get(i)) for r in rows if i in r["declared"]]
        moving = [s for s in seen if s not in ("-", "0")]
        if not moving:
            continue
        c = Counter(moving)
        top, k = c.most_common(1)[0]
        frac = k / len(moving)
        agree.append(frac)
        print(f"A{i:<7d} {len(seen):14d} {len(c):16d} {top + ' (' + str(k) + ')':>18s} "
              f"{frac:9.0%}")
    mean_agree = sum(agree) / max(1, len(agree))
    print(
        f"\nmean agreement over movement buttons: {mean_agree:.0%}\n"
        "This is the ceiling on transfer for a policy whose output is a wire id.\n"
        "A model that learned 'this looks like a corridor -> ACTION1' is right\n"
        "about the direction and still presses the wrong button the rest of the time."
    )

    # -- the input end: is a colour index a stable feature? ------------------
    print("\n--- is the avatar the same colour in every game? ----------------")
    av = Counter(r["avatar"] for r in rows if r["avatar"] >= 0)
    bg = Counter(r["background"] for r in rows)
    found = sum(av.values())
    print(f"avatar found in {found}/{n} games, over {len(av)} distinct colours: "
          + " ".join(f"c{c}x{k}" for c, k in av.most_common()))
    print(f"background over {len(bg)} distinct colours: "
          + " ".join(f"c{c}x{k}" for c, k in bg.most_common()))
    if found:
        top_c, top_k = av.most_common(1)[0]
        print(
            f"\nmodal avatar colour c{top_c} covers {top_k / found:.0%} of games.\n"
            "The student's input is a one-hot over absolute colour index, so the\n"
            "unit that learned 'me' in training fires on something else here."
        )

    print(
        f"\nprobe cost: {sum(r['cost'] for r in rows) / n:.0f} billed actions per game "
        f"to learn both ends from scratch.\n"
        "That is the price of not needing them to agree - and it is small against a\n"
        "human baseline of 26-230 actions for level 0 alone."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
