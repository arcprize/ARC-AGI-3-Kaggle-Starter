"""How much wider is the action space through the gateway than in the twin?

WHY ASK
-------
``arc3x/transfer.py`` measurement B played the 8 holdout games with a uniform
random policy and a distilled policy, 3000 billed actions each, and both scored
**0.00 on every game**. Yet Go-Explore solves several of those same games in a
handful of actions - cd82 two levels in 5 and 6, ft09 in 20 and 7. A 5-action
solution that random search cannot find in 3000 tries means the two are not
drawing from the same set.

They are not. ``Twin.valid_actions`` calls ``game._get_valid_actions()``, a
method on the local engine object, and for ACTION6 it returns the **concrete
coordinates the engine will accept**. The gateway cannot do that: it reports
ACTION6 as one entry with no coordinates, because enumerating 4,096 clicks per
turn over HTTP is not a thing it offers. So a policy behind the gateway has to
invent its own coordinate, and ``arc3x/transfer.py`` offers the 256 coarse cells
that match the student's click head.

That makes ``_get_valid_actions`` a second privilege of local search, alongside
``deepcopy``. Free rewind was already known to be unavailable on hidden games;
this one was not recorded anywhere, and for click games it is the larger of the
two - it is the difference between choosing among a handful of options and
choosing among a few hundred.

This measures the gap per game so it is a number rather than an argument: the
mean branching factor the searcher sees, the branching factor a gateway policy
sees, and what each implies for finding a short solution by chance.

    .venv/Scripts/python.exe arc3x/action_space.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from arc3x.explore import discover_games
from arc3x.suite import HOLD
from arc3x.transfer import candidates
from arc3x.twin import Act, Twin, default_env_dir

WALK = 60  # random steps per game; enough to leave the title screen and move about


def survey(game_id: str, env_dir: Path, seed: int = 0) -> dict:
    """Walk randomly in the twin, recording both views of the legal set each turn.

    The walk is driven by the engine's own legal set, because a walk driven by
    guessed clicks would mostly no-op and never reach the states where the two
    views differ most.
    """
    rng = np.random.default_rng(seed)
    twin = Twin(game_id, env_dir)
    g = twin.snapshot()
    obs = Twin.step_game(g, Act(0))
    declared = tuple(sorted({a.aid for a in (Twin.valid_actions(g) or ())}))

    twin_n: list[int] = []
    gate_n: list[int] = []
    clicky = 0
    for _ in range(WALK):
        legal = Twin.valid_actions(g)
        if not legal:
            break
        twin_n.append(len(legal))
        # What the same state looks like from behind the gateway: declared ids
        # only, with every coarse cell on offer wherever ACTION6 is declared.
        gate_n.append(len(candidates(tuple({a.aid for a in legal}))))
        clicky += sum(1 for a in legal if a.is_click)
        obs = Twin.step_game(g, legal[int(rng.integers(len(legal)))])
        if obs.terminal:
            g = twin.snapshot()
            Twin.step_game(g, Act(0))

    return {
        "game": game_id.split("-")[0],
        "declared": declared,
        "twin": float(np.mean(twin_n)) if twin_n else 0.0,
        "gate": float(np.mean(gate_n)) if gate_n else 0.0,
        "click_frac": clicky / max(1, sum(twin_n)),
        "steps": len(twin_n),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="*")
    a = ap.parse_args(argv)

    env_dir = default_env_dir()
    ids = discover_games(env_dir)
    if a.games:
        ids = [g for g in ids if g.split("-")[0] in a.games]

    rows: list[dict] = []
    print(f"{'game':6s} {'declared':>16s} {'twin b':>7s} {'gate b':>7s} {'ratio':>6s} "
          f"{'clicks':>7s}")
    for gid in ids:
        try:
            r = survey(gid, env_dir)
        except Exception as exc:
            print(f"{gid.split('-')[0]:6s} {type(exc).__name__}: {exc}", flush=True)
            continue
        rows.append(r)
        star = "*" if r["game"] in HOLD else " "
        print(
            f"{r['game']:5s}{star} {str(list(r['declared'])):>16s} "
            f"{r['twin']:7.1f} {r['gate']:7.1f} "
            f"{r['gate'] / max(1e-9, r['twin']):5.0f}x {r['click_frac']:6.0%}",
            flush=True,
        )

    if not rows:
        return 1
    movers = [r for r in rows if 6 not in r["declared"]]
    clickers = [r for r in rows if 6 in r["declared"]]
    print(f"\n(* = holdout.  b = mean branching factor over {WALK} steps.)")
    for name, grp in (("no ACTION6", movers), ("has ACTION6", clickers)):
        if not grp:
            continue
        tw = float(np.mean([r["twin"] for r in grp]))
        ga = float(np.mean([r["gate"] for r in grp]))
        print(f"\n{name:12s} n={len(grp):2d}   twin b={tw:6.1f}   gateway b={ga:6.1f}   "
              f"{ga / max(1e-9, tw):.0f}x wider")
        # Chance of hitting one specific 5-action line, the length Go-Explore
        # needed on cd82. Stated as odds because the exponent is the point.
        for label, b in (("twin", tw), ("gateway", ga)):
            print(f"    a specific 5-action line by chance, {label:8s}: 1 in {b ** 5:,.0f}")
    print(
        "\nread: where the two columns are close, a gateway policy is choosing from\n"
        "the same set the searcher chose from, and imitation is a fair transfer of\n"
        "what search found. Where the gateway column is hundreds of times wider,\n"
        "search was solving an easier problem than the one that gets scored, and\n"
        "its plan length is not evidence that the line is findable online."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
