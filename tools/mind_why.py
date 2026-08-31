"""Why the mind's placement predictions miss: learned delta vs observed displacement.

``tools/mind_backtest.py`` reports that the model calls moved-versus-blocked right
84% of the time but puts the sprite in the right cell only 47% of the time. Knowing
*that* a button moves you while not knowing *where* it lands is a specific failure
with a small number of possible causes, and guessing between them is exactly what
this project cannot afford. So this measures it.

For each button, on the held-out tail of observed play, it prints:

  * the delta the model learned, and whether that came from evidence or the prior;
  * every displacement the sprite was actually observed to make, with counts.

Then the diagnosis reads itself off the two columns:

  * learned ``(-1,0)`` against observed ``(-8,0)`` - the game moves on a lattice and
    the model learned one pixel of it. ``Mechanics.tile`` exists for this.
  * learned ``(-1,0)`` against observed a spread of magnitudes - one action produces
    several engine frames and the sprite travels while they play out. ``Obs.n_frames``
    counts them.
  * learned ``(-1,0)`` against observed nothing at all - the thing being tracked is
    not the thing that moves, so ``locate`` is on the wrong clump.

Run:
    .venv/Scripts/python.exe tools/mind_why.py s5i5 r11l tu93 lp85 ls20
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratch" / "archive2_extracted" / "src" / "ARC3-Inference"))

import numpy as np  # noqa: E402

from arc3x.mind import Mechanics  # noqa: E402
from arc3x.mindgraft import AID_LABEL, Mind, transitions  # noqa: E402
from arc3x.twin import default_env_dir  # noqa: E402
from tools.mind_backtest import collect  # noqa: E402


def explain(game_id: str, *, steps: int, seed: int, holdout: float) -> None:
    env_dir = default_env_dir()
    entries = collect(game_id, steps=steps, seed=seed, env_dir=env_dir)
    trs = transitions(entries)
    if len(trs) < 8:
        print(f"{game_id}: only {len(trs)} transitions")
        return
    cut = max(4, int(len(trs) * (1.0 - holdout)))
    train, test = trs[:cut], trs[cut:]

    mind = Mind()
    for tr in train:
        mind.mech.observe(tr.press.aid, tr.before, tr.after, level_up=tr.level_up)
    mind.mech.settle()
    # The same second pass `backtest` runs: `observe` cannot learn what stops the
    # sprite until `settle` has said which object the sprite is.
    mind.mech.replay_geometry((t.press.aid, t.before, t.after) for t in train)
    # Where training left the sprite. The probes below each walk the tail and
    # mutate `pos` as they go, so each one has to start from here rather than
    # from `None` - an unhinted `locate` ranks by size, not proximity.
    at_cut = mind.mech.pos

    # What actually happened in the tail, per button, measured the same way the
    # scorer measures truth: locate the sprite before and after.
    observed: dict[int, Counter] = {}
    frames_per_action: Counter = Counter()
    mind.mech.pos = at_cut
    for tr in test:
        box_b = mind.mech.locate(tr.before, hint=mind.mech.pos)
        box_a = mind.mech.locate(tr.after, hint=box_b[:2] if box_b else None)
        if box_b is None or box_a is None:
            observed.setdefault(tr.press.aid, Counter())["unlocatable"] += 1
            continue
        d = (box_a[0] - box_b[0], box_a[1] - box_b[1])
        observed.setdefault(tr.press.aid, Counter())[d] += 1
        frames_per_action[tr.press.aid] += 1
        mind.mech.pos = box_a[:2]

    tile = mind.mech.tile
    print(f"\n=== {game_id.split('-')[0]}  avatar={mind.mech.avatar} "
          f"body={sorted(mind.mech.body)} background={mind.mech.background} "
          f"tile={tile} ===")

    # The clock test. `Mechanics.observe` decides "nothing happened" with
    # `(before != after).any()` over the whole grid, so anything that changes on
    # every action makes every action look eventful: `noop` never increments,
    # `_blame_block` never fires, and `walk_mask` stays at its permissive all-ones
    # fallback - which is why a model can predict movement into a wall forever
    # without learning better.
    #
    # Looking for pixels that change on >=90% of actions is the wrong test: a
    # ticking counter changes a *different* pixel each tick, so it hides from it.
    # The right test is to isolate the actions where the sprite provably did not
    # move and ask what changed anyway - and where.
    if train:
        quiet = sum(1 for tr in train if not tr.changed)
        still_counts: list[int] = []
        still_box = [64, 64, -1, -1]
        for tr in train:
            b = mind.mech.locate(tr.before, hint=None)
            a = mind.mech.locate(tr.after, hint=b[:2] if b else None)
            if b is None or a is None or b[:2] != a[:2]:
                continue
            diff = tr.before != tr.after
            n = int(diff.sum())
            if not n:
                continue
            still_counts.append(n)
            ys, xs = np.nonzero(diff)
            still_box = [
                min(still_box[0], int(ys.min())),
                min(still_box[1], int(xs.min())),
                max(still_box[2], int(ys.max())),
                max(still_box[3], int(xs.max())),
            ]
        med = int(np.median(still_counts)) if still_counts else 0
        print(
            f"  clock test: {quiet}/{len(train)} actions changed nothing at all "
            f"(noop total={sum(mind.mech.noop.values())}). "
            f"On {len(still_counts)} actions the sprite did not move yet "
            f"{med} pixels changed (median), all inside rows "
            f"{still_box[0]}-{still_box[2]} cols {still_box[1]}-{still_box[3]}"
        )
    if mind.mech.avatar < 0:
        print("  no avatar was ever identified - a translation model cannot apply here")

    # Which way do the misses go? "Predicted a move, got refused" and "predicted a
    # refusal, got a move" have opposite fixes, and averaging them into one
    # `place=47%` hides which one is happening. For the first kind, print what was
    # actually sitting in the destination cells and whether the model had any
    # evidence about those colours - because `walk_mask` is only permissive when
    # `passable` is empty, and `_under` folds the background in, so a game whose
    # sprite ever moved should already have a restrictive mask.
    mind.mech.pos = at_cut
    ghost, phantom, right = 0, 0, 0
    dest_colors: Counter = Counter()
    for tr in test:
        mind.mech.where(tr.before)
        pred = mind.predict(tr.before, tr.press.aid)
        if not pred.spoke:
            continue
        box_b = mind.mech.locate(tr.before, hint=mind.mech.pos)
        box_a = mind.mech.locate(tr.after, hint=box_b[:2] if box_b else None)
        if box_b is None or box_a is None:
            continue
        really_moved = box_a[:2] != box_b[:2]
        if pred.moved == really_moved:
            right += 1
            continue
        if pred.moved and not really_moved:
            ghost += 1  # walked through a wall in imagination
            top, left, h, w = box_b
            dy, dx = mind.mech.moves[tr.press.aid]
            foot = mind.mech.footprint(tr.before, box_b)
            H, W = tr.before.shape
            ys, xs = np.nonzero(foot)
            for y, x in zip(ys.tolist(), xs.tolist()):
                ny, nx = top + y + dy, left + x + dx
                if not (0 <= ny < H and 0 <= nx < W):
                    dest_colors["<edge>"] += 1
                    continue
                fy, fx = ny - top, nx - left
                if 0 <= fy < h and 0 <= fx < w and foot[fy, fx]:
                    continue
                dest_colors[int(tr.before[ny, nx])] += 1
        else:
            phantom += 1  # predicted a wall that was not there
    tot = ghost + phantom + right
    if tot:
        print(
            f"  move calls: {right}/{tot} right, {ghost} ghost (moved in mind, "
            f"refused for real), {phantom} phantom (blocked in mind, moved for real)"
        )
        if dest_colors:
            desc = ", ".join(
                f"{c}(passable={mind.mech.passable.get(c, 0) if c != '<edge>' else '-'},"
                f"blocking={mind.mech.blocking.get(c, 0) if c != '<edge>' else '-'})x{n}"
                for c, n in dest_colors.most_common(6)
            )
            print(f"  ghost destinations held: {desc}")
        print(f"  walk_mask open on {mind.mech.walk_mask(test[0].before).mean():.0%} of pixels")
    aids = sorted(set(observed) | set(mind.mech.moves) | set(mind.mech.deltas))
    for aid in aids:
        learned = mind.mech.moves.get(aid)
        raw = mind.mech.deltas.get(aid)
        tag = " (assumed)" if aid in mind.mech.assumed else ""
        seen = observed.get(aid, Counter())
        top = ", ".join(f"{k}x{v}" for k, v in seen.most_common(5)) or "-"
        print(
            f"  {AID_LABEL.get(aid, aid):<6} learned={learned}{tag}"
            f"  delta={raw}  tries={mind.mech.tries[aid]}"
            f"  noop={mind.mech.noop[aid]}  shifts={mind.mech.shifts[aid]}"
        )
        print(f"         observed in tail: {top}")


def main() -> int:
    games = sys.argv[1:] or ["s5i5", "r11l", "tu93", "lp85", "ls20"]
    for g in games:
        try:
            explain(g, steps=250, seed=7, holdout=0.3)
        except Exception as exc:  # noqa: BLE001
            print(f"{g}: FAILED {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
