"""Measure the click model before it spends a real action.

WHY THIS EXISTS
---------------
``arc3x/clicks.py`` is a hypothesis, and this project's record on unmeasured
hypotheses is eleven leaderboard experiments and eleven results at or below the
baseline. So the click model gets a number first, on the same terms
``tools/mind_backtest.py`` imposes on the movement model.

The harvested Kaggle run of 2026-08-23 is why this is urgent rather than tidy.
Its best game was ``tn36`` at 2.67, and ``tn36`` offers ``MOUSE`` and nothing
else on all 118 of its turns: it cleared level 0 in 37 actions against a baseline
of 32, then burned 358 actions failing level 1. The mind cannot describe that
game at all, because ``Mechanics.observe`` is handed an action id with no
coordinates.

WHAT MAKES THIS HONEST
----------------------
* **Random click coordinates.** ``collect`` is imported from ``mind_backtest``
  rather than reimplemented, so clicks land on ``rng.randrange`` cells. The
  engine will happily hand over its legal click list; the gateway will not.
* **Train on the past, score on the future.** Both passes - the chrome mask and
  the click semantics - see only the first ``1 - holdout`` of transitions. The
  chrome mask being applied to held-out frames is generalisation, not leakage: it
  is a statistic over training frames, exactly as a live run would have it.
* **The same parser the graft uses.** Click coordinates come from
  ``mindgraft.transitions`` -> ``parse_press``, so a formatting mismatch shows up
  here instead of on Kaggle.

READING THE OUTPUT
------------------
``kind``   - the verdict, and what fraction of active clicks supports it.
``spoke``  - held-out clicks the model would predict at all. Only INERT and PAINT
             locate their own effect, so a low number is a statement about the
             game's genre, not a failure.
``exact``  - the whole grid was right. One ticking HUD pixel pins this at zero
             forever, which is the finding that forced the two-pass design.
``content``- the grid was right outside chrome. **This is the usable number.**
``live``   - of held-out clicks whose under-colour was seen in training, how
             often the model called "this click does something" correctly. This
             is the one that pays for itself even when nothing is predictable: a
             64x64 board is 4096 candidate clicks and an agent has ~100 actions.

Run:
    .venv/Scripts/python.exe tools/click_probe.py
    .venv/Scripts/python.exe tools/click_probe.py --games tn36 cd82 --steps 400
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratch" / "archive2_extracted" / "src" / "ARC3-Inference"))

import numpy as np  # noqa: E402

from arc3x.clicks import MIN_CLICKS, ClickModel  # noqa: E402
from arc3x.mindgraft import Transition, transitions  # noqa: E402
from arc3x.twin import default_env_dir  # noqa: E402

from mind_backtest import collect  # noqa: E402


@dataclass
class ClickReport:
    """One game's worth of held-out click accuracy."""

    game: str
    n_all: int = 0  # transitions of any kind
    n_train: int = 0  # click transitions trained on
    n_test: int = 0  # click transitions held out
    kind: str = "unknown"
    conf: float = 0.0
    spoke: int = 0
    exact: int = 0
    content: int = 0
    live_called: int = 0  # held-out clicks the live/dead counters had an opinion on
    live_right: int = 0
    chrome: int = 0
    active_frac: float = 0.0  # fraction of TRAINING clicks that did anything
    follow: float = 0.0  # max |corr| between click position and where the board moved
    notes: list[str] = field(default_factory=list)

    def line(self) -> str:
        def pct(num: int, den: int) -> str:
            return f"{num / den:5.0%}" if den else "    -"

        return (
            f"{self.game:>6}  {self.kind:>8}({self.conf:3.0%})  "
            f"clicks={self.n_train:>3}/{self.n_test:<3} of {self.n_all:>3}  "
            f"active={self.active_frac:4.0%}  "
            f"follow={self.follow:4.2f}  "
            f"spoke={pct(self.spoke, self.n_test)}  "
            f"exact={pct(self.exact, self.spoke)}  "
            f"content={pct(self.content, self.spoke)}  "
            f"live={pct(self.live_right, self.live_called)}({self.live_called})  "
            f"chrome={self.chrome:>4}px"
            + (("  " + "; ".join(self.notes)) if self.notes else "")
        )


def _live_call(model: ClickModel, under: int) -> bool | None:
    """Would the model say a click on this colour does something? ``None`` = no idea.

    This is deliberately cruder than ``clickable``: it asks about one colour so it
    can be scored per held-out click, and abstains on a colour never clicked in
    training rather than guessing, so the reported rate is not inflated by
    coin-flips on unseen colours.
    """
    hit, miss = model.live.get(under, 0), model.dead.get(under, 0)
    if hit + miss < 2:
        return None
    return hit > miss


def probe(game: str, entries, *, holdout: float) -> ClickReport:
    trs: list[Transition] = transitions(entries)
    rep = ClickReport(game=game, n_all=len(trs))
    if not trs:
        rep.notes.append("no transitions")
        return rep

    cut = max(1, int(round(len(trs) * (1.0 - holdout))))
    train, test = trs[:cut], trs[cut:]

    clicks_train = [t for t in train if t.press.is_click and t.press.row is not None]
    clicks_test = [t for t in test if t.press.is_click and t.press.row is not None]
    rep.n_train, rep.n_test = len(clicks_train), len(clicks_test)
    if not clicks_train:
        rep.notes.append("no clicks offered" if not any(t.press.is_click for t in trs) else "clicks but no coords")
        return rep

    model = ClickModel()
    # Pass 1 over EVERY training transition, click or not: a HUD ticks on every
    # action regardless of which button caused it.
    model.learn_volatile([(t.before, t.after) for t in train])
    for t in clicks_train:
        model.observe(t.before, t.after, int(t.press.row), int(t.press.col))

    rep.kind, rep.conf = model.verdict()
    vol = model.volatile
    rep.chrome = 0 if vol is None else int(vol.sum())
    rep.active_frac = (model.n - model.support["inert"]) / model.n if model.n else 0.0
    fr, fc, _ = model.follows
    rep.follow = max(fr, fc)

    for t in clicks_test:
        r, c = int(t.press.row), int(t.press.col)
        truth_diff = model.content(t.before, t.after)
        did_something = bool(truth_diff.any())

        call = _live_call(model, int(t.before[r, c]))
        if call is not None:
            rep.live_called += 1
            rep.live_right += int(call == did_something)

        guess = model.predict(t.before, r, c)
        if guess is None:
            continue
        rep.spoke += 1
        if np.array_equal(guess, t.after):
            rep.exact += 1
        # Right where it matters: agree with the real frame on every non-chrome
        # cell. A model that gets the chrome wrong has still told the planner the
        # truth about the board.
        keep = np.ones_like(truth_diff) if vol is None or vol.shape != truth_diff.shape else ~vol
        if np.array_equal(guess[keep], t.after[keep]):
            rep.content += 1

    if rep.n_test < MIN_CLICKS:
        rep.notes.append("thin holdout")
    return rep


def aggregate(reports: list[ClickReport]) -> str:
    scored = [r for r in reports if r.n_train >= MIN_CLICKS]
    if not scored:
        return "no game had enough clicks to score"
    kinds: dict[str, int] = {}
    for r in scored:
        kinds[r.kind] = kinds.get(r.kind, 0) + 1
    spoke = sum(r.spoke for r in scored)
    tested = sum(r.n_test for r in scored)
    content = sum(r.content for r in scored)
    exact = sum(r.exact for r in scored)
    lc = sum(r.live_called for r in scored)
    lr = sum(r.live_right for r in scored)
    taxonomy = "  ".join(f"{k}={v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]))
    lines = [
        f"{len(scored)} of {len(reports)} games had >= {MIN_CLICKS} training clicks",
        f"taxonomy: {taxonomy}",
        f"held-out clicks: {tested}   spoke: {spoke}"
        + (f" ({spoke / tested:.0%})" if tested else "")
        + (f"   exact: {exact / spoke:.0%}   content: {content / spoke:.0%}" if spoke else ""),
        f"live/dead call: {lr}/{lc}" + (f" = {lr / lc:.0%}" if lc else ""),
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", nargs="*", default=None, help="game ids (default: all)")
    ap.add_argument("--steps", type=int, default=250, help="actions of random play per game")
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    env_dir = default_env_dir()
    if args.games:
        ids = list(args.games)
    else:
        from arc3x.explore import discover_games

        ids = discover_games(env_dir)

    print(
        f"click probe: {len(ids)} games, {args.steps} random actions each, "
        f"clicks at uniformly random cells, predicting the last {args.holdout:.0%}\n"
    )
    reports: list[ClickReport] = []
    for gid in ids:
        short = gid.split("-")[0]
        try:
            entries = collect(gid, steps=args.steps, seed=args.seed, env_dir=env_dir)
            rep = probe(short, entries, holdout=args.holdout)
        except Exception as exc:  # noqa: BLE001 — one broken game must not hide the rest
            print(f"{short:>6}  FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        reports.append(rep)
        print(rep.line(), flush=True)

    print()
    print(aggregate(reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
