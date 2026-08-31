"""Does anything the searcher learned on these games transfer to games it has never seen?

WHY THIS IS THE ONLY NUMBER THAT MATTERS
----------------------------------------
The scored set is **110 private games the agent has never seen**. The 25 games
that ship with the competition dataset are a development instrument, not
families the scored set is drawn from. Two consequences kill the obvious plan:

* ``arc3x/explore.py`` needs a local engine file to ``deepcopy``. A hidden game
  has none, and the gateway runs ``environments_dir=""`` and cannot be cloned or
  rewound. **Free search cannot run on a scored game at all.**
* ``arc3x/runner.py`` rung 1 replays a searched plan only after an opening-frame
  match at 0.97. Against unseen games that will essentially never fire.

So Go-Explore's 4.735 over the 25 dev games is unreachable on the leaderboard.
Search's only route there is as a **teacher**: it plays the dev games for free
and its plans train a policy that reads pixels rather than game ids.

Whether that works is the whole question, and nothing in this repo had measured
it. ``train_student.py`` reports ``hist["val_acc"]``, a random split over
*examples pooled from every game* - it answers "can the model predict the
searcher's move in a game it trained on". That is not transfer. A model can score
well there by memorising 25 board layouts and be worthless on the 110.

WHAT IS MEASURED
----------------
Leave-*games*-out, using the fixed split in ``arc3x/suite.py``: train on the 17
TUNE games, test on the 8 HOLD games, which no tuning has touched. The split is
alphabetical and fixed, never re-drawn, because a holdout re-rolled after a
disappointing result is a lottery rather than a holdout.

**A. Action prediction** (default). Of the searcher's moves on HOLD games, how
many does the model rank first, against picking uniformly from the same legal set
computed on those same HOLD states? Without that baseline the number means
nothing, since a state offering two moves gives 0.50 for free.

A is run once per encoding from ``arc3x/features.py`` - ``color``, ``rank``,
``role`` - over **identical states and identical labels**, so the only variable
is what a frame is taken to mean. That comparison is the point of the file:
the first run of this test scored 3.75x on trained games and 1.10x on unseen
ones, and ``arc3x/why_no_transfer.py`` traced the failure to the input encoding
rather than to the amount of data. This is where that diagnosis gets tested.

**B. Policy play** (``--play``). Play the HOLD games through ``GradedRun``,
billed exactly as Kaggle bills, against uniform random on the same candidates,
budget and seeds.

One caveat on B that is not cosmetic. ``harvest`` builds its legal-action masks
from ``Twin.valid_actions``, which calls the engine's ``_get_valid_actions`` and
returns **concrete click coordinates**. The gateway offers ACTION6 with no
coordinates, so at play time the policy proposes its own and this module offers
all 256 coarse cells. ``arc3x/action_space.py`` measured that gap: for games with
clicks the gateway's branching factor is 20-83x the twin's. So on click games the
policy faces a decision it was never trained on, and B's numbers there say
nothing about imitation - only about the 9 movement-only games, where both sides
see the same 4 to 5 actions, is B a fair test.

    .venv/Scripts/python.exe arc3x/transfer.py --plans scratch/plans_1200.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc3x.features import CELLS, ENCODERS, POOL, role_lut
from arc3x.graded import run_suite
from arc3x.mind import Mechanics
from arc3x.student import Student, slot_of
from arc3x.suite import HOLD, TUNE
from arc3x.twin import Act, Twin, default_env_dir

SCRATCH = Path(__file__).resolve().parent.parent / "scratch"


# -- candidate actions ------------------------------------------------------


def candidates(declared: tuple[int, ...]) -> list[Act]:
    """The actions a policy may choose from, given only what the gateway says.

    This deliberately does **not** consult the engine. ``Twin.valid_actions``
    would hand back exact click coordinates, and using it here would measure a
    policy holding information the real gateway never provides - which is how a
    local number turns into a leaderboard disappointment.

    For ACTION6 the gateway offers no coordinate, so all 256 coarse cells are
    proposed, one per output slot of the student's click head. The pixel offered
    is the cell's interior so ``slot_of`` maps it back to the cell it came from
    rather than to a neighbour.
    """
    acts = [Act(a) for a in sorted(declared) if a != 6]
    if 6 in declared:
        half = POOL // 2
        acts += [
            Act(6, POOL * c + half, POOL * r + half)
            for r in range(CELLS)
            for c in range(CELLS)
        ]
    return acts


# -- A: harvest, keeping raw frames so every encoding sees the same states ---


def harvest_frames(
    root, plan, frame0
) -> tuple[list[tuple[np.ndarray, int, np.ndarray]], Mechanics]:
    """Replay one plan in the twin: (frame, label, legal-mask) plus what was learned.

    Two differences from ``student.harvest_plan``, which this otherwise mirrors
    exactly (state recorded *before* each action, single-choice states dropped):

    * the raw frame is kept instead of a feature vector, so the three encodings
      can be compared on identical states rather than on separate harvests;
    * ``Mechanics`` is fed the same transitions as they go past. That is free -
      the replay is happening anyway - and it is also the honest way round: an
      agent on a hidden game learns what things do *while* playing, from exactly
      this stream of before/after pairs.

    The returned ``Mechanics`` is settled over the whole plan, so using it to
    encode early frames uses knowledge from later in the game. That makes the
    role encoding an **upper bound** on what an online agent could do, which is
    the right first question: if even the upper bound does not transfer, the
    online version cannot.
    """
    g = copy.deepcopy(root)
    valid = Twin.valid_actions(g)
    frame = frame0
    m = Mechanics()
    out: list[tuple[np.ndarray, int, np.ndarray]] = []
    level = 0
    for i, a in enumerate(plan):
        if frame is not None and valid and len(valid) > 1:
            slots = sorted({slot_of(v) for v in valid})
            lab = slot_of(a)
            if lab in slots:
                out.append((frame, lab, np.array(slots, dtype=np.intp)))
        obs = Twin.step_game(g, a)
        if frame is not None:
            m.observe(
                a.aid, frame, obs.frame,
                level_up=int(obs.level) > level, died=obs.game_over,
            )
        level = int(obs.level)
        if i % 10 == 9:
            m.settle()
        if obs.terminal:
            break
        frame = obs.frame
        valid = obs.valid or valid
    m.settle()
    return out, m


def harvest(plans_path: Path, env_dir: Path) -> dict[str, tuple[list, Mechanics]]:
    """Replay every solved plan in its own twin, keyed by short game id."""
    from arc3x.explore import discover_games

    blob = json.loads(Path(plans_path).read_text())
    results = blob["results"] if isinstance(blob, dict) else blob
    solved = [r for r in results if r.get("plan") and r.get("levels_solved", 0) > 0]
    by_prefix = {g.split("-")[0]: g for g in discover_games(env_dir)}

    out: dict[str, tuple[list, Mechanics]] = {}
    for r in solved:
        pre = r["game_id"].split("-")[0]
        gid = by_prefix.get(pre)
        if gid is None:
            continue
        root = Twin(gid, env_dir).snapshot()
        frame0 = Twin.step_game(root, Act(0)).frame
        plan = [Act(int(a), int(x), int(y)) for a, x, y in r["plan"]]
        pairs, mech = harvest_frames(root, plan, frame0)
        if pairs:
            out[pre] = (pairs, mech)
    return out


# -- B: policy play, billed like Kaggle -------------------------------------


def play_policy(run, *, weights: str | None = None, budget: int = 3000, seed: int = 0) -> None:
    """Play one game to the action cap. ``weights=None`` means uniform random.

    Module-level and dependency-light on purpose: ``run_suite`` ships this to
    worker processes, so it must survive being re-imported rather than closed
    over. Uses the ``color`` encoding, the only one ``Student.prior`` builds
    itself; a role-encoded policy needs ``Mechanics`` learned online, which is an
    agent rather than a measurement.
    """
    rng = np.random.default_rng(seed)
    student = Student.load(weights) if weights else None
    run.note("policy:student" if student else "policy:random")

    obs = run.reset()
    deaths = 0
    while run.actions < budget:
        acts = candidates(tuple(obs.available_actions))
        if not acts:
            run.note("noactions")
            return
        if student is not None:
            p = student.prior(obs.frame, acts)
            i = int(rng.choice(len(acts), p=p))
        else:
            i = int(rng.integers(len(acts)))
        a = acts[i]
        obs = run.step(a.aid, a.x, a.y)
        if obs.won:
            run.note("win")
            return
        if obs.game_over:
            # One action to clear it, and worth spending: a policy that stops at
            # the first death scores zero on every game that can kill you.
            deaths += 1
            obs = run.reset()
    run.note(f"budget:deaths={deaths}")


def _curve(x, y, ms, tune_games, span, te, a, name: str) -> None:
    """Is HOLD transfer rising with the number of training games, or flat?

    A single 1.10x cannot distinguish "the idea is wrong" from "209 examples is
    too few", and those want opposite responses - abandon distillation, or run a
    longer sweep. The trend separates them. Flat at 1.0 across a 3x range of
    training data means more teacher data is not the missing ingredient.

    Games are added in a fixed alphabetical order, not sampled, so the numbers are
    reproducible and no favourable subset can be picked after the fact.
    """
    rh = None
    print(f"  --- learning curve, encoding '{name}' ---")
    print(f"  {'games':>5s} {'examples':>9s} {'tune x':>8s} {'HOLD x':>8s}")
    for k in (3, 5, 8, len(tune_games)):
        if k > len(tune_games):
            continue
        sub = tune_games[:k]
        idx = np.array([i for g in sub for i in range(*span[g])], dtype=np.intp)
        if len(idx) < 20:
            continue
        m = Student.new(hidden=a.hidden, seed=a.seed, n_in=x.shape[1])
        m.fit(x[idx], y[idx], [ms[i] for i in idx],
              epochs=a.epochs, lr=a.lr, seed=a.seed, val_frac=0.0, log=False)
        rt = m.baseline_acc(ms, idx)
        rh = rh if rh is not None else m.baseline_acc(ms, te)
        xt = m._acc(x, y, ms, idx) / max(rt, 1e-9)
        xh = m._acc(x, y, ms, te) / max(rh, 1e-9)
        print(f"  {k:5d} {len(idx):9d} {xt:7.2f}x {xh:7.2f}x")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    ap.add_argument("--plans", required=True, help="a sweep.py output json")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--encodings", default="color,rank,role")
    ap.add_argument("--curve", action="store_true",
                    help="also report HOLD transfer vs number of training games")
    ap.add_argument("--play", action="store_true", help="also run measurement B")
    ap.add_argument("--budget", type=int, default=3000, help="action cap per game in B")
    ap.add_argument("-w", "--workers", type=int, default=8)
    a = ap.parse_args(argv)

    env_dir = default_env_dir()
    t0 = time.perf_counter()
    per_game = harvest(Path(a.plans), env_dir)
    tune_games = [g for g in TUNE if g in per_game]
    hold_games = [g for g in HOLD if g in per_game]
    print(f"plans        : {a.plans}")
    print(f"solved games : {len(per_game)}  ({len(tune_games)} tune, {len(hold_games)} hold)")
    print(f"  train on   : {' '.join(tune_games) or '-'}")
    print(f"  test on    : {' '.join(hold_games) or '-'}")
    if not tune_games or not hold_games:
        print("\nnot enough games on one side of the split to measure transfer.")
        return 1

    # One pooled store of raw states, with an index range per game, so every
    # encoding and every per-game readout works off the same examples.
    frames: list[np.ndarray] = []
    ys: list[int] = []
    ms: list[np.ndarray] = []
    span: dict[str, tuple[int, int]] = {}
    luts: dict[str, np.ndarray] = {}
    for g in tune_games + hold_games:
        pairs, mech = per_game[g]
        lo = len(ys)
        for fr, lab, mk in pairs:
            frames.append(fr)
            ys.append(lab)
            ms.append(mk)
        span[g] = (lo, len(ys))
        luts[g] = role_lut(mech)
    y = np.array(ys, dtype=np.intp)
    tr = np.array([i for g in tune_games for i in range(*span[g])], dtype=np.intp)
    te = np.array([i for g in hold_games for i in range(*span[g])], dtype=np.intp)
    game_of = {i: g for g in span for i in range(*span[g])}
    print(
        f"examples     : {len(tr):,} train / {len(te):,} test "
        f"(harvested in {time.perf_counter() - t0:.1f}s)"
    )

    summary: dict[str, tuple[float, float, float, float]] = {}
    best: tuple[float, str, Student] | None = None
    for name in [e.strip() for e in a.encodings.split(",") if e.strip()]:
        enc, n_in = ENCODERS[name]
        x = np.stack([enc(frames[i], luts[game_of[i]]) for i in range(len(frames))])
        print(f"\n=== encoding '{name}'  ({n_in} inputs) " + "=" * 34)
        model = Student.new(hidden=a.hidden, seed=a.seed, n_in=n_in)
        # val_frac=0 keeps all but one TUNE example for training; the honest
        # validation set is the HOLD games, not a random slice of the same boards.
        model.fit(
            x[tr], y[tr], [ms[i] for i in tr],
            epochs=a.epochs, lr=a.lr, seed=a.seed, val_frac=0.0, log=True,
        )
        model.games = tuple(tune_games)

        rt, at = model.baseline_acc(ms, tr), model._acc(x, y, ms, tr)
        rh, ah = model.baseline_acc(ms, te), model._acc(x, y, ms, te)
        summary[name] = (rt, at, rh, ah)
        print(f"  {'tune (trained on)':22s} random {rt:.3f}  model {at:.3f}  "
              f"{at / max(rt, 1e-9):.2f}x")
        print(f"  {'HOLD (never seen)':22s} random {rh:.3f}  model {ah:.3f}  "
              f"{ah / max(rh, 1e-9):.2f}x")
        for g in hold_games:
            w = np.arange(*span[g], dtype=np.intp)
            r0, a0 = model.baseline_acc(ms, w), model._acc(x, y, ms, w)
            print(f"    {g:8s} n={len(w):4d}  random {r0:.3f}  model {a0:.3f}  "
                  f"{a0 / max(r0, 1e-9):.2f}x")
        if best is None or ah / max(rh, 1e-9) > best[0]:
            best = (ah / max(rh, 1e-9), name, model)

        if a.curve:
            _curve(x, y, ms, tune_games, span, te, a, name)

    print("\n--- A. transfer by encoding -------------------------------------")
    print(f"{'encoding':10s} {'tune x':>8s} {'HOLD x':>8s}   {'reading':s}")
    for name, (rt, at, rh, ah) in summary.items():
        xt, xh = at / max(rt, 1e-9), ah / max(rh, 1e-9)
        read = (
            "learns, does not transfer" if xt > 1.5 and xh < 1.2
            else "transfers" if xh >= 1.2
            else "learns nothing"
        )
        print(f"{name:10s} {xt:7.2f}x {xh:7.2f}x   {read}")
    print(
        "\nHOLD x is the number that matters: 1.00x means the policy is worth\n"
        "exactly nothing on a game it has not seen, which is every scored game."
    )

    SCRATCH.mkdir(parents=True, exist_ok=True)
    assert best is not None
    out = SCRATCH / f"student_{best[1]}_tune_only.npz"
    best[2].save(out)
    print(f"\nbest encoding on HOLD: '{best[1]}' at {best[0]:.2f}x -> {out}")
    if not a.play:
        return 0

    print("\n--- B. policy play on HOLD, billed like Kaggle ------------------")
    print("(colour encoding only; see the module docstring on why click games\n"
          " are not a fair test of imitation here)")
    from arc3x.explore import discover_games

    color_w = SCRATCH / "student_color_tune_only.npz"
    if not color_w.exists():
        print("  need the 'color' encoding trained for B; add it to --encodings")
        return 1
    ids = [g for g in discover_games(env_dir) if g.split("-")[0] in HOLD]
    means: dict[str, float] = {}
    reports: dict[str, list[dict]] = {}
    for label, w in (("random", None), ("student", str(color_w))):
        res = run_suite(
            play_policy, ids, env_dir=env_dir, workers=a.workers, verbose=False,
            weights=w, budget=a.budget, seed=a.seed,
        )
        means[label] = res["mean_score"]
        reports[label] = res["reports"]
        print(f"  {label:8s} mean {res['mean_score']:6.3f}")

    print(f"\n{'game':8s} {'lvls rnd':>9s} {'lvls stu':>9s} {'score rnd':>10s} {'score stu':>10s}")
    by = {lab: {r["game_id"].split("-")[0]: r for r in rs} for lab, rs in reports.items()}
    for g in sorted(by["random"]):
        r, s = by["random"][g], by["student"].get(g, {})
        print(
            f"{g:8s} {r['levels_completed']:>4d}/{r['n_levels']:<4d} "
            f"{s.get('levels_completed', 0):>4d}/{s.get('n_levels', 0):<4d} "
            f"{r['score']:10.2f} {s.get('score', 0.0):10.2f}"
        )
    print(f"\nHOLD mean: student {means['student']:.3f} vs random {means['random']:.3f} "
          f"({means['student'] - means['random']:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
