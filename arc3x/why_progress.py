"""Print the objective's own arithmetic, click by click. A debugging aid.

Written because two successive fixes to the click branch both failed the same
way: the run reported ``click:progress`` hundreds of times on a game that
completed nothing. Reasoning about the formula produced two wrong diagnoses, so
this prints the terms instead - which colours the ratchet believes are being
consumed and built, their pixel counts, the peak the built term is measured
against, and the resulting objective - for every action of a real graded play.

    .venv/Scripts/python.exe arc3x/why_progress.py cd82 --actions 80
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc3x.agent import Agent
from arc3x.explore import discover_games
from arc3x.graded import GradedRun
from arc3x.twin import default_env_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("--actions", type=int, default=80)
    a = ap.parse_args(argv)

    env = default_env_dir()
    gid = next(g for g in discover_games(env) if g.startswith(a.game))
    run = GradedRun(gid, env)
    agent = Agent(run, budget=a.actions, seconds=600.0)

    # Wrap ``act`` so every billed action prints the objective's components.
    real_act = agent.act
    prev = {"s": None}

    def traced(aid: int, x: int = 0, y: int = 0):
        obs = real_act(aid, x=x, y=y)
        p = agent.dream.prog
        few = agent.dream.collectible | p.consumed
        many = p.built - few
        cf = p.count(obs.frame, few)
        cm = p.count(obs.frame, many)
        peak = sum(p.peak[c] for c in many)
        s = agent.dream.objective(obs.frame)
        d = "" if prev["s"] is None or s is None else f"  d={s - prev['s']:+d}"
        prev["s"] = s
        print(
            f"  a{run.actions:4d} act{aid}"
            f"{f'({x},{y})' if aid == 6 else '      '}"
            f"  few={sorted(few)}:{cf:4d}"
            f"  many={sorted(many)}:{cm:4d}/peak{peak:4d}"
            f"  obj={s}{d}"
            f"  gain={agent.last_gain}  best={agent.best_obj}",
            flush=True,
        )
        return obs

    agent.act = traced  # type: ignore[method-assign]
    agent.play()
    print(f"\n{run.report()['score']:.2f}  {agent.m.summary()}")
    print(f"  {agent.dream.prog.summary()}")
    print(f"  fell={dict(agent.dream.prog.fell)}")
    print(f"  rose={dict(agent.dream.prog.rose)}")
    print(f"  peak={dict(agent.dream.prog.peak)}  field={agent.dream.prog.field_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
