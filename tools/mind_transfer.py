"""Does what the mind learned on level 0 remain true on level 1?

WHY THIS IS THE MEASUREMENT THAT MATTERS
----------------------------------------
The competition scorer (``arc3x/graded.py`` ``score_from_card``, copied verbatim from
``EnvironmentScorecard._calculate_score``) weights level *i* by ``i+1`` and returns
``min(weighted mean, completion cap)``. Over the 25 dev games' real level counts that
puts a hard ceiling of **3.52** on clearing level 0 alone and **10.57** on clearing
levels 0 and 1. The submitted notebook scores **2.14** - 61% of the level-0-only
ceiling - so it is already respectable on level 0 and the missing points are all depth.

The documented reason depth fails: the vendor's framework **wipes the agent's
accumulated knowledge at a level transition**, sparing only ``cross_level_notes``. The
recorded ``sk48`` failure is exactly "level 0 cleared, then level 1 stalled to the wall
because every learned mechanic had been discarded".

The mind is immune to that wipe *if and only if* what it learned on level 0 is still
true on level 1, because it re-induces its model from the ``(action, frame)`` history,
which the framework keeps. Re-deriving costs **zero actions**. So the whole value of the
mind as a depth mechanism rests on one empirical question, and this measures it:

    train the model on level 0's transitions ONLY, then predict level 1+'s.

WHAT WOULD FALSIFY IT
---------------------
If a game re-binds its buttons, changes its step size, or swaps the avatar's colour
between levels, then level-0 knowledge is *worse* than nothing - it would be injected
into the prompt as confident, wrong fact. So this reports the two models side by side:
what level 0 taught, and what level 1 teaches on its own. Identical deltas mean the
carry-over is sound. Different deltas mean the mind must re-learn, and must be told to
distrust itself across a boundary.

Random play rarely clears a level, so histories that span a boundary are scarce. Both
policies here are legitimate: the engine is only ever a **data generator**, stepped one
action at a time with no snapshot, search or rollback, and the transfer question is a
property of the *history*, not of the policy that produced it.

Run:
    .venv/Scripts/python.exe tools/mind_transfer.py                 # all 25
    .venv/Scripts/python.exe tools/mind_transfer.py --games ls20 sk48 --steps 3000
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratch" / "archive2_extracted" / "src" / "ARC3-Inference"))

import numpy as np  # noqa: E402

from arc3x.mind import Mechanics  # noqa: E402
from arc3x.mindgraft import AID_LABEL, Mind, Transition, transitions  # noqa: E402
from arc3x.twin import default_env_dir  # noqa: E402
from tools.mind_backtest import collect  # noqa: E402

from inference.agent.runtime_state import Frame, HistoryEntry  # noqa: E402


class RecordingRun:
    """A ``GradedRun`` that also keeps the framework-shaped history.

    ``GradedRun`` bills actions the way Kaggle bills them but throws every frame
    away, and random play crosses a level boundary on only 3 of the 25 games - too
    few to answer the transfer question. Wrapping the run rather than editing it
    keeps ``arc3x/graded.py`` honest: this adds a *reader*, not a new power. The
    engine is still stepped one action at a time with no snapshot or rollback, so
    the recorded history is exactly what a live agent would have seen.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.entries: list[HistoryEntry] = []

    def __getattr__(self, name):  # everything not overridden goes to the run
        return getattr(self._inner, name)

    def _seed(self) -> None:
        if not self.entries:
            obs = self._inner.current()
            self.entries.append(HistoryEntry(action="", frame=self._frame(obs)))

    def _frame(self, obs) -> Frame:
        n = getattr(self._inner, "n_levels", 1) or 1
        return Frame(
            grid=tuple(tuple(int(c) for c in row) for row in obs.frame),
            step=int(self._inner.actions),
            level=max(1, min(int(n), int(obs.levels_completed) + 1)),
        )

    def _log(self, display: str, obs) -> None:
        self.entries.append(HistoryEntry(action=display, frame=self._frame(obs)))

    def step(self, aid: int, x: int = 0, y: int = 0):
        self._seed()
        obs = self._inner.step(aid, x, y)
        name = AID_LABEL.get(aid, f"ACTION{aid}")
        self._log(f"{name}(row={y}, col={x})" if aid == 6 else name, obs)
        return obs

    def reset(self):
        self._seed()
        obs = self._inner.reset()
        self._log("RESET", obs)
        return obs


def _fit(trs: list[Transition]) -> Mechanics:
    """Both passes, in the only order they are learnable: who am I, then what stops me."""
    mech = Mechanics()
    for tr in trs:
        mech.observe(tr.press.aid, tr.before, tr.after, level_up=tr.level_up)
    mech.settle()
    mech.replay_geometry((t.press.aid, t.before, t.after) for t in trs)
    return mech


def _score(mech: Mechanics, trs: list[Transition]) -> tuple[int, int, int, int]:
    """(spoke, exact, placed, move_call) over transitions this model never saw."""
    mind = Mind(mech=mech)
    spoke = exact = placed = call = 0
    for tr in trs:
        box_before = mind.mech.where(tr.before)
        pred = mind.predict(tr.before, tr.press.aid)
        if not pred.spoke:
            continue
        spoke += 1
        if np.array_equal(pred.grid, tr.after):
            exact += 1
        box = mind.mech.locate(tr.after, hint=mind.mech.pos)
        if box is not None and pred.to is not None and box[:2] == pred.to:
            placed += 1
        if box is not None and box_before is not None:
            if pred.moved == (box[:2] != box_before[:2]):
                call += 1
    return spoke, exact, placed, call


def _agree(a: Mechanics, b: Mechanics) -> tuple[int, int, list[str]]:
    """Do the two models believe the same things? (agreed, compared, disagreements)"""
    notes: list[str] = []
    if a.avatar != b.avatar:
        notes.append(f"avatar {a.avatar}->{b.avatar}")
    if a.background != b.background:
        notes.append(f"background {a.background}->{b.background}")
    if a.tile != b.tile:
        notes.append(f"tile {a.tile}->{b.tile}")
    agreed = compared = 0
    for aid in sorted(set(a.moves) | set(b.moves)):
        da, db = a.moves.get(aid), b.moves.get(aid)
        if da is None or db is None:
            notes.append(f"{AID_LABEL.get(aid, aid)} {da}/{db}")
            continue
        compared += 1
        if da == db:
            agreed += 1
        else:
            notes.append(f"{AID_LABEL.get(aid, aid)} {da}->{db}")
    return agreed, compared, notes


def transfer(game_id: str, *, steps: int, seed: int, env_dir: Path) -> dict | None:
    entries = collect(game_id, steps=steps, seed=seed, env_dir=env_dir)
    return _judge(game_id, entries)


def transfer_agent(game_id: str, *, env_dir: Path) -> dict | None:
    """Same measurement, on history generated by the local search agent.

    Needed because random play clears a level on almost nothing. The policy is
    irrelevant to the question - transfer is a property of the history - so the
    only thing that matters is that every frame here was produced by stepping the
    engine forward one billed action at a time, which it was.
    """
    from arc3x.agent import play_agent
    from arc3x.graded import GradedRun

    rec = RecordingRun(GradedRun(game_id, env_dir=env_dir, verbose=False))
    try:
        play_agent(rec)
    except Exception as exc:  # noqa: BLE001 — a crashed policy still leaves usable history
        print(f"{game_id.split('-')[0]:>6}  policy raised {type(exc).__name__}: {exc}")
    return _judge(game_id, rec.entries)


def _judge(game_id: str, entries) -> dict | None:
    trs = transitions(entries)
    if len(trs) < 16:
        return None
    # Split at the first level change. `level_after` comes from the framework's own
    # Frame.level, so this is the boundary the vendor wipes knowledge at.
    cut = next((i for i, t in enumerate(trs) if t.level_up), -1)
    if cut < 0:
        return {"game": game_id.split("-")[0], "reached": False, "n0": len(trs)}
    early, late = trs[: cut + 1], trs[cut + 1 :]
    if len(early) < 8 or len(late) < 8:
        return {"game": game_id.split("-")[0], "reached": False, "n0": len(trs)}

    from_l0 = _fit(early)
    from_l1 = _fit(late)
    spoke, exact, placed, call = _score(_fit(early), late)
    agreed, compared, notes = _agree(from_l0, from_l1)
    return {
        "game": game_id.split("-")[0],
        "reached": True,
        "n0": len(early),
        "n1": len(late),
        "spoke": spoke,
        "exact": exact,
        "placed": placed,
        "call": call,
        "buttons0": len(from_l0.moves),
        "buttons1": len(from_l1.moves),
        "agreed": agreed,
        "compared": compared,
        "notes": notes,
    }


def _work(job: tuple) -> dict | None:
    gid, mode, steps, seed, env_dir = job
    try:
        if mode == "agent":
            return transfer_agent(gid, env_dir=env_dir)
        return transfer(gid, steps=steps, seed=seed, env_dir=env_dir)
    except Exception as exc:  # noqa: BLE001
        return {"game": gid.split("-")[0], "failed": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--mode",
        choices=["random", "agent"],
        default="random",
        help="'agent' drives the local search policy, which actually clears levels",
    )
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    env_dir = default_env_dir()
    if args.games:
        ids = list(args.games)
    else:
        from arc3x.explore import discover_games

        ids = discover_games(env_dir)

    print(
        f"cross-level transfer: {len(ids)} games, policy={args.mode}"
        + (f", up to {args.steps} actions" if args.mode == "random" else "")
        + ",\ntrain on level 0's transitions only, predict level 1+'s\n"
    )
    jobs = [(g, args.mode, args.steps, args.seed, env_dir) for g in ids]
    rows: list[dict] = []
    if args.workers > 1:
        from multiprocessing import Pool

        with Pool(args.workers) as pool:
            results = pool.map(_work, jobs)
    else:
        results = [_work(j) for j in jobs]

    for r in results:
        if r is None:
            continue
        if r.get("failed"):
            print(f"{r['game']:>6}  FAILED {r['failed']}", flush=True)
            continue
        rows.append(r)
        if not r["reached"]:
            print(f"{r['game']:>6}  never left level 0 ({r['n0']} transitions)", flush=True)
            continue
        sp = r["spoke"] or 1
        print(
            f"{r['game']:>6}  L0 taught {r['buttons0']} buttons from {r['n0']:>4} steps, "
            f"L1+ has {r['n1']:>4}  spoke={r['spoke'] / r['n1']:4.0%}  "
            f"place={r['placed'] / sp:4.0%}  exact={r['exact'] / sp:4.0%}  "
            f"movecall={r['call'] / sp:4.0%}  agree={r['agreed']}/{r['compared']}"
            + (f"  [{'; '.join(r['notes'])}]" if r["notes"] else ""),
            flush=True,
        )

    got = [r for r in rows if r.get("reached")]
    print()
    if not got:
        print(
            "No game crossed a level boundary under random play, so cross-level transfer\n"
            "is NOT MEASURED here. That is itself the finding: a stronger policy is needed\n"
            "to generate boundary-spanning history before this question can be answered."
        )
        print(f"({len(rows)} games played, all stayed on level 0)")
        return 0
    sp = sum(r["spoke"] for r in got) or 1
    n1 = sum(r["n1"] for r in got) or 1
    print(
        f"{len(got):>6} games crossed a boundary  spoke={sp / n1:4.0%}  "
        f"place={sum(r['placed'] for r in got) / sp:4.0%}  "
        f"exact={sum(r['exact'] for r in got) / sp:4.0%}  "
        f"movecall={sum(r['call'] for r in got) / sp:4.0%}  "
        f"deltas agreeing across the boundary="
        f"{sum(r['agreed'] for r in got)}/{sum(r['compared'] for r in got)}"
    )
    changed = Counter()
    for r in got:
        for note in r["notes"]:
            changed[note.split()[0]] += 1
    if changed:
        print("  what changed across the boundary, by kind: " + ", ".join(
            f"{k}x{v}" for k, v in changed.most_common()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
