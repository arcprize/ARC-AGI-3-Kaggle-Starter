"""What would it actually take to score 10+? Solve the scoring formula backwards.

Every design argument in this repo has been made with a made-up 9-level example.
This uses the real level counts and the real human baselines of the 25 dev games,
so the target is a measured number rather than an illustration.

The policy being priced is the one the gateway actually permits:

  * ``level_reset()`` restores the current level from a pristine clone, costs one
    action, and clears GAME_OVER - so search *is* available online, per level.
  * level 0 carries weight 1 of ``sum(1..n)``, so burning thousands of actions
    there costs almost nothing. That is where the game gets learned.
  * every action spent on level *i* is billed to level *i* alone, so levels 1..k
    have to be played at close to human action counts on the first attempt.
  * exactly one play per game (``competition_mode`` blocks a second
    environment), so there is no retry to fall back on.

Output is a table of mean leaderboard score against (deepest level cleared,
efficiency multiplier vs the human baseline).

    .venv/Scripts/python.exe arc3x/target.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc3x.explore import discover_games
from arc3x.graded import score_from_card
from arc3x.twin import Twin, default_env_dir

EXPLORE_ACTIONS = 5000  # what we spend on level 0 learning the game


def price(
    baselines: list[int], deepest: int, eff: float, explore: int = EXPLORE_ACTIONS
) -> float:
    """Score for: burn ``explore`` actions on level 0, then clear 1..deepest at ``eff``.

    ``eff`` is the multiplier on the human baseline, so eff=1.5 means we take 50%
    more actions than a person did. Levels past ``deepest`` are never completed.
    """
    n = len(baselines)
    if deepest >= n:
        deepest = n - 1
    abl: list[tuple[int, int]] = []
    total = explore
    abl.append((1, total))
    for i in range(1, deepest + 1):
        total += max(1, round(baselines[i] * eff))
        abl.append((i + 1, total))
    sc, _per, _ch = score_from_card(baselines, abl, total)
    return sc


def main() -> int:
    env_dir = default_env_dir()
    games = discover_games(env_dir)
    info: list[tuple[str, list[int]]] = []
    for gid in games:
        t = Twin(gid, env_dir)
        info.append((gid.split("-")[0], [int(b) for b in (t.baselines or [])]))

    print(f"{'game':7s} {'lvls':>4s}  baselines")
    print("-" * 72)
    for pre, base in info:
        print(f"{pre:7s} {len(base):4d}  {base[:12]}")
    lv = [len(b) for _p, b in info]
    print(
        f"\nlevels per game: min {min(lv)} median {sorted(lv)[len(lv)//2]} max {max(lv)}"
    )
    tot = [sum(range(1, len(b) + 1)) for _p, b in info]
    print(f"weight totals:   min {min(tot)} max {max(tot)}")

    effs = [1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
    print(
        "\nmean leaderboard score - burn level 0 on exploration, then clear 1..k\n"
        f"(exploration = {EXPLORE_ACTIONS} actions charged to level 0)\n"
    )
    print("clear thru  " + "".join(f"{e:>8.2f}x" for e in effs))
    print("-" * (12 + 9 * len(effs)))
    for k in range(1, 8):
        row = []
        for e in effs:
            vals = [price(b, k, e) for _p, b in info if len(b) > k]
            n_ok = len(vals)
            # games with fewer levels than k still contribute: clear all of them
            vals += [price(b, len(b) - 1, e) for _p, b in info if len(b) <= k]
            row.append(sum(vals) / len(vals))
        print(f"level {k:<6d}" + "".join(f"{v:9.2f}" for v in row) + f"   ({n_ok} games deep enough)")

    print(
        "\nsame, but level 0 played cleanly at the same efficiency instead of burned:\n"
    )
    print("clear thru  " + "".join(f"{e:>8.2f}x" for e in effs))
    print("-" * (12 + 9 * len(effs)))
    for k in range(0, 8):
        row = []
        for e in effs:
            vals = []
            for _p, b in info:
                kk = min(k, len(b) - 1)
                vals.append(price(b, kk, e, explore=max(1, round(b[0] * e))))
            row.append(sum(vals) / len(vals))
        print(f"level {k:<6d}" + "".join(f"{v:9.2f}" for v in row))

    print(
        "\nread: to clear 10.0 we need either\n"
        "  - level 0 burned + levels 1..4 at <=2x human, or\n"
        "  - level 0 played clean + levels 1..3 at <=2x human.\n"
        "Depth raises the cap (sum of cleared weights / sum of all weights * 100),\n"
        "so each extra level is worth more than any efficiency gain above ~1.5x."
    )

    # -- the model we can actually build ----------------------------------
    #
    # An online searcher does not spend a multiple of the human baseline; it
    # spends however many actions its search needs, which is roughly
    # independent of how quickly a person could have done it. That asymmetry is
    # in our favour twice over: level weights grow with depth AND human
    # baselines grow with depth, so a fixed search cost buys a better and
    # better ratio the deeper it goes.
    print("\n" + "=" * 72)
    print("CONSTANT-COST SEARCH: c actions per level, clears levels 0..k")
    print("=" * 72)
    costs = [100, 200, 400, 800, 1600, 3200]
    print("clear thru  " + "".join(f"{c:>8d}a" for c in costs))
    print("-" * (12 + 9 * len(costs)))
    for k in range(0, 10):
        row = []
        for c in costs:
            vals = []
            for _p, b in info:
                kk = min(k, len(b) - 1)
                abl = [(i + 1, c * (i + 1)) for i in range(kk + 1)]
                sc, _p2, _c2 = score_from_card(b, abl, c * (kk + 1))
                vals.append(sc)
            row.append(sum(vals) / len(vals))
        print(f"level {k:<6d}" + "".join(f"{v:9.2f}" for v in row))
    print(
        "\nread: full clear at 100 billed actions/level scores 62.1; at 200, 33.9;\n"
        "at 400, 13.7; at 800 it is 3.8 and hopeless. Partial depth reaches 10+ at\n"
        "100a/level by level 3, 200a by level 4, 400a by level 6.\n"
        "So the engineering problem is precisely: clear a level of a game we have\n"
        "never seen in ~100-300 BILLED actions, using level_reset as the restart.\n"
        "Offline Go-Explore needs ~50k sim steps for that, which is 100x too many -\n"
        "hence the plan: spend level 0's near-free budget learning a forward model,\n"
        "then plan inside that model for free and spend actions only on execution."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
