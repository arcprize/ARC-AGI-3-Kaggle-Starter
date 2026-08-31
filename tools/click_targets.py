"""Where are the cells that matter? Asked of every cell, not a sample.

WHY THIS EXISTS
---------------
``tools/click_invariance.py`` at 25 sampled cells said tn36 had 1.0 distinct click
outcomes - "the coordinate is decoration". At all 4096 cells it has 11, with 96% of
cells producing one identical board. The lattice had sampled only the boring 96% and
inverted the conclusion.

That correction is the useful finding. Across the games measured exhaustively at frame
0, the click space is not 4096 and not 1: it is a handful of outcomes dominated by one
"nothing special" result, with a small minority of cells doing everything interesting.

    m0r0 1   sp80 1   ka59 2   vc33 3   lf52 3   dc22 3   cn04 3
    s5i5 5   tn36 11   su15 144   r11l 3096

So the planner's question is not "does position matter" but "which cells are the
special ones, and can they be recognised from the frame without trying them". This
measures the answer, because the prior that goes in the agent's system prompt has to
be a recognition rule it can apply, not a fact it cannot use.

WHAT IS MEASURED
----------------
For each game, from a frozen state, every one of the 4096 cells is clicked in its own
deep copy of the game. Cells are grouped by the exact frame they produce. The largest
group is the *modal* outcome - "nothing special" - and every other cell is *special*.

Then the only question that matters for a search policy: are the special cells
recognisable? Three candidate rules, each a general property of the frame and none of
them naming a game:

``on_object``   share of special cells sitting on a non-background colour.
``on_minority`` share sitting on a colour that covers less than 5% of the board.
``clusters``    connected components of the special mask, and their sizes. A handful
                of compact clusters means "there are N buttons and they are regions";
                thousands of scattered cells means the effect is positional
                everywhere and there is nothing to shortlist.

The comparison that decides whether a rule is worth anything is against the base rate:
if 70% of the whole board is non-background, then "special cells are 70% non-background"
tells the agent nothing. So the base rate is printed next to every share.

THIS IS AN OFFLINE ORACLE
-------------------------
It deep-copies the game 4096 times per state. The 110 graded games are behind a
gateway with no snapshot, so nothing here can ship - it exists to decide what the
shippable prior should say.

Run:
    .venv/Scripts/python.exe tools/click_targets.py
    .venv/Scripts/python.exe tools/click_targets.py --games tn36 s5i5 --gap 40
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratch" / "archive2_extracted" / "src" / "ARC3-Inference"))

import numpy as np  # noqa: E402

from arc3x.percept import background, blobs, mask_component  # noqa: E402
from arc3x.twin import Act, Twin, default_env_dir  # noqa: E402

RESET = Act(0)
MINORITY_FRAC = 0.05


def _components(mask: np.ndarray) -> list[int]:
    """Sizes of the connected components of a boolean mask, largest first."""
    seen = np.zeros_like(mask)
    sizes: list[int] = []
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            sub, sy, sx, size = mask_component(mask, y, x)
            seen[sy : sy + sub.shape[0], sx : sx + sub.shape[1]] |= sub
            sizes.append(int(size))
    return sorted(sizes, reverse=True)


def sweep(game_id: str, *, gap: int, seed: int, env_dir: Path) -> dict | None:
    """Click every cell from one frozen state; describe the cells that were special."""
    rng = random.Random(f"{game_id}:{seed}")
    twin = Twin(game_id, env_dir=env_dir)
    obs = twin.current()

    for _ in range(gap):
        aids = sorted({a.aid for a in obs.valid if a.aid != 0})
        if not aids:
            break
        aid = rng.choice(aids)
        H0, W0 = obs.frame.shape
        act = Act(6, rng.randrange(W0), rng.randrange(H0)) if aid == 6 else Act(aid)
        obs = Twin.step_game(twin.game, act)
        if obs.terminal:
            obs = Twin.step_game(twin.game, RESET)

    if 6 not in {a.aid for a in obs.valid}:
        return None

    before = obs.frame
    H, W = before.shape
    snap = twin.snapshot()

    outcome: dict[bytes, list[tuple[int, int]]] = {}
    for r in range(H):
        for c in range(W):
            branch = copy.deepcopy(snap)
            out = Twin.step_game(branch, Act(6, c, r))
            key = np.ascontiguousarray(out.frame).tobytes()
            outcome.setdefault(key, []).append((r, c))

    groups = sorted(outcome.values(), key=len, reverse=True)
    modal = groups[0]
    special = [cell for g in groups[1:] for cell in g]

    mask = np.zeros((H, W), dtype=bool)
    for r, c in special:
        mask[r, c] = True

    bg = background(before)
    total = H * W
    counts = Counter(int(v) for v in before.ravel())
    minority = {col for col, n in counts.items() if n < MINORITY_FRAC * total}

    def share(cells, pred) -> float:
        return (sum(1 for r, c in cells if pred(int(before[r, c]))) / len(cells)) if cells else 0.0

    # Base rates over the whole board, so a share can be compared to chance.
    board_cells = [(r, c) for r in range(H) for c in range(W)]

    # Which blobs do the special cells fall on? A blob touched by a special cell is a
    # thing the agent could have clicked deliberately.
    all_blobs = blobs(before, ignore={bg})
    touched = 0
    for b in all_blobs:
        sub = mask[b.top : b.top + b.height, b.left : b.left + b.width]
        if sub.any():
            touched += 1

    return {
        "game": game_id.split("-")[0],
        "outcomes": len(groups),
        "modal": len(modal) / total,
        "n_special": len(special),
        "on_object": share(special, lambda v: v != bg),
        "on_object_base": share(board_cells, lambda v: v != bg),
        "on_minority": share(special, lambda v: v in minority),
        "on_minority_base": share(board_cells, lambda v: v in minority),
        "clusters": _components(mask),
        "blobs_total": len(all_blobs),
        "blobs_touched": touched,
        "colors": Counter(int(before[r, c]) for r, c in special).most_common(4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--gap", type=int, default=0, help="random actions before the sweep (0 = frame 0)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    env_dir = default_env_dir()
    if args.games:
        ids = list(args.games)
    else:
        from arc3x.explore import discover_games

        ids = discover_games(env_dir)

    print(f"click targets: every cell clicked from a frozen state, gap={args.gap}\n")
    rows = []
    for gid in ids:
        short = gid.split("-")[0]
        try:
            res = sweep(gid, gap=args.gap, seed=args.seed, env_dir=env_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"{short:>6}  FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        if res is None:
            print(f"{short:>6}  no click offered", flush=True)
            continue
        rows.append(res)
        clusters = res["clusters"]
        shape = f"{len(clusters)} clusters, biggest {clusters[0] if clusters else 0}"
        print(
            f"{short:>6}  outcomes={res['outcomes']:>4}  modal={res['modal']:5.1%}  "
            f"special={res['n_special']:>4}  "
            f"on_object={res['on_object']:5.0%} (base {res['on_object_base']:4.0%})  "
            f"on_minority={res['on_minority']:5.0%} (base {res['on_minority_base']:4.0%})  "
            f"{shape}  blobs {res['blobs_touched']}/{res['blobs_total']}  "
            f"colors={res['colors']}",
            flush=True,
        )

    if not rows:
        return 0

    # The verdict on each candidate recognition rule: does it beat the base rate, and
    # by how much, averaged over the games where there is anything to recognise?
    live = [r for r in rows if 1 < r["outcomes"] and r["n_special"] < 0.5 * 4096]
    print()
    print(f"{len(live)} of {len(rows)} games have a small special set worth recognising")
    if live:
        for key in ("on_object", "on_minority"):
            got = float(np.mean([r[key] for r in live]))
            base = float(np.mean([r[f"{key}_base"] for r in live]))
            lift = got / base if base > 0 else float("inf")
            print(f"  {key:>12}: {got:.0%} vs base {base:.0%}  = {lift:.2f}x chance")
        cl = [len(r["clusters"]) for r in live]
        print(f"  clusters per game: median {int(np.median(cl))}, max {max(cl)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
