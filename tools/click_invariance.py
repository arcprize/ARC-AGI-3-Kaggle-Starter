"""Does the click coordinate matter? Asked of the engine, not of a correlation.

WHY THIS EXISTS
---------------
``tools/click_probe.py`` verdicted ``step`` - "the coordinate is decoration" - on 8
of the 17 dev games that offer clicks, including ``tn36``, which was the best game
in the harvested Kaggle run and offers ``MOUSE`` and nothing else. That verdict
rests on one statistic: the correlation between where the click landed and where the
board changed, near zero on all eight.

Near-zero correlation is necessary evidence and not sufficient evidence, for three
reasons that have nothing to do with the games:

1. **A concentrated board fakes it.** If the change is always in the same small
   region - one sprite, one counter - the change centre barely varies, and a
   correlation computed against an almost-constant variable is noise whether the
   coordinate matters or not.
2. **Correlation is linear.** "The object nearest the click moves" is total
   coordinate dependence with no linear signature at all.
3. **It measures the wrong thing.** The planner's question is not "does the change
   track the click" but "if I click elsewhere, do I get a different board".

So this asks that question literally: freeze a state, click many well-separated
cells from that same frozen state, and count how many distinct boards come back.

WHAT THIS IS AND IS NOT
-----------------------
This is an **offline oracle**, and it uses a power the real run does not have:
``Twin.snapshot`` deep-copies the game so the same state can be replayed. The 110
graded games are behind a gateway with no snapshot and no rollback, so nothing here
can ship.

That is the point. The shippable statistic is ``ClickModel.follows``, which needs
only the history any agent already has. This measures whether that cheap statistic
tells the truth, using ground truth the live agent will never see. Confirming a
proxy against an oracle offline, then shipping the proxy, is the only way to know
the proxy is worth shipping.

READING THE OUTPUT
------------------
``outcomes``  - distinct boards produced by N clicks at N different cells, averaged
                over the sampled states. **This is the true size of the click action
                space.** 1.0 means every click does the same thing and a planner
                searching 4096 positions is searching a space of one.
``modal``     - share of clicks landing on the single most common outcome. High with
                low ``outcomes`` is a game with one dominant effect and a few special
                cells, which is a different thing from a game that ignores position.
``verdict``   - what ``ClickModel`` said from history alone, for comparison.

Run:
    .venv/Scripts/python.exe tools/click_invariance.py
    .venv/Scripts/python.exe tools/click_invariance.py --games tn36 vc33 r11l
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratch" / "archive2_extracted" / "src" / "ARC3-Inference"))

import numpy as np  # noqa: E402

from arc3x.clicks import ClickModel  # noqa: E402
from arc3x.twin import Act, Twin, default_env_dir  # noqa: E402

RESET = Act(0)


def _spread_cells(h: int, w: int, k: int) -> list[tuple[int, int]]:
    """``k`` cells spread over the board on a lattice.

    Deliberately not random: two random cells can land next to each other, and
    neighbouring cells are exactly where a coordinate-sensitive game is most likely
    to agree by accident. A lattice guarantees the samples are far apart, which is
    the condition under which "same outcome" is informative.
    """
    side = max(2, int(np.ceil(np.sqrt(k))))
    rows = np.linspace(0, h - 1, side).round().astype(int)
    cols = np.linspace(0, w - 1, side).round().astype(int)
    return [(int(r), int(c)) for r in rows for c in cols][:k]


def measure(game_id: str, *, states: int, cells: int, gap: int, seed: int, env_dir: Path):
    """Return (mean distinct outcomes, mean modal share, states probed, verdict).

    The model is fitted on the same random walk that produces the probe states, so
    the verdict it reports is the one an agent would have held at that moment.
    """
    rng = random.Random(f"{game_id}:{seed}")
    twin = Twin(game_id, env_dir=env_dir)
    obs = twin.current()
    H, W = obs.frame.shape
    grid_cells = _spread_cells(H, W, cells)

    model = ClickModel()
    prev = obs.frame
    distinct: list[float] = []
    modal: list[float] = []

    for state_i in range(states):
        # Walk ``gap`` actions to reach a genuinely different state. The first probe
        # happens at frame 0, which is the state the agent actually has to plan from.
        for _ in range(gap if state_i else 0):
            aids = sorted({a.aid for a in obs.valid if a.aid != 0})
            if not aids:
                break
            aid = rng.choice(aids)
            act = Act(6, rng.randrange(W), rng.randrange(H)) if aid == 6 else Act(aid)
            obs = Twin.step_game(twin.game, act)
            model.learn_volatile([(prev, obs.frame)])
            if act.aid == 6:
                model.observe(prev, obs.frame, act.y, act.x)
            prev = obs.frame
            if obs.terminal:
                obs = Twin.step_game(twin.game, RESET)
                prev = obs.frame

        if 6 not in {a.aid for a in obs.valid}:
            continue  # this state does not offer a click at all

        snap = twin.snapshot()
        results: dict[bytes, int] = {}
        for (r, c) in grid_cells:
            branch = copy.deepcopy(snap)
            out = Twin.step_game(branch, Act(6, c, r))
            key = np.ascontiguousarray(out.frame).tobytes()
            results[key] = results.get(key, 0) + 1
        total = sum(results.values())
        if not total:
            continue
        distinct.append(len(results))
        modal.append(max(results.values()) / total)

    kind, conf = model.verdict()
    fr, fc, _ = model.follows
    return (
        float(np.mean(distinct)) if distinct else 0.0,
        float(np.mean(modal)) if modal else 0.0,
        len(distinct),
        kind,
        conf,
        max(fr, fc),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--states", type=int, default=4, help="states probed per game")
    ap.add_argument("--cells", type=int, default=25, help="click cells tried per state")
    ap.add_argument("--gap", type=int, default=25, help="random actions between probed states")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    env_dir = default_env_dir()
    if args.games:
        ids = list(args.games)
    else:
        from arc3x.explore import discover_games

        ids = discover_games(env_dir)

    print(
        f"click invariance: {len(ids)} games, {args.states} states each, "
        f"{args.cells} lattice cells per state, counterfactual clicks from a frozen state\n"
    )
    print(f"{'game':>6}  {'outcomes':>8}/{args.cells:<3} {'modal':>6}  {'states':>6}  {'history verdict':>16}  follow")
    rows = []
    for gid in ids:
        short = gid.split("-")[0]
        try:
            n_out, modal, n_states, kind, conf, follow = measure(
                gid, states=args.states, cells=args.cells, gap=args.gap, seed=args.seed, env_dir=env_dir
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{short:>6}  FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        if not n_states:
            print(f"{short:>6}  no click offered in any probed state", flush=True)
            continue
        rows.append((short, n_out, modal, kind, follow))
        print(
            f"{short:>6}  {n_out:8.1f}/{args.cells:<3} {modal:6.0%}  {n_states:6d}  "
            f"{kind + f'({conf:.0%})':>16}  {follow:5.2f}",
            flush=True,
        )

    if not rows:
        return 0
    print()
    # The comparison this whole file exists for: does the shippable statistic agree
    # with the oracle? A `step` verdict is only trustworthy where outcomes ~ 1.
    agree = [(g, n, k) for g, n, m, k, f in rows if k == "step"]
    if agree:
        print("games the history-only model called `step` (coordinate is decoration):")
        for g, n, k in agree:
            ok = "CONFIRMED" if n <= 1.5 else ("PARTIAL" if n <= 3 else "REFUTED")
            print(f"  {g:>6}  {n:5.1f} distinct outcomes  -> {ok}")
    others = [(g, n, k) for g, n, m, k, f in rows if k != "step"]
    if others:
        print("\ngames it did not call `step`, for contrast:")
        for g, n, k in others:
            print(f"  {g:>6}  {n:5.1f} distinct outcomes  ({k})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
