"""Why is the search 16 steps/sec on some games and 400 on others?

Aggregate throughput across a 25-game parallel sweep measured 412 steps/sec -
*below* the 546 measured in a single process. Parallelism buys nothing, so the
bottleneck is shared, not per-core. The suspects are the two things done per
step and per restore:

  * ``game.perform_action``  - pure-Python engine logic
  * ``copy.deepcopy(game)``  - allocation-heavy state snapshot

deepcopy builds a large graph of small Python objects, which is exactly the
workload CPython's generational GC handles worst: every new container is
tracked, and each collection rescans the survivors. Disabling the GC around an
allocation-heavy phase is a standard fix, and pickle round-tripping is
sometimes faster than deepcopy because it walks the graph in C.

Run: python arc3x/bench_step.py g50t sk48 wa30
"""

from __future__ import annotations

import copy
import gc
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arc3x.explore import discover_games
from arc3x.twin import Act, Twin, default_env_dir

N_STEP = 300


def time_steps(root, n: int = N_STEP) -> tuple[float, int]:
    """Steps/sec on a live game, restarting from ``root`` whenever it dies."""
    g = copy.deepcopy(root)
    valid = Twin.valid_actions(g)
    done = 0
    t0 = time.perf_counter()
    while done < n:
        if not valid:
            g = copy.deepcopy(root)
            valid = Twin.valid_actions(g)
            continue
        obs = Twin.step_game(g, valid[done % len(valid)])
        done += 1
        if obs.terminal or not obs.valid:
            g = copy.deepcopy(root)
            valid = Twin.valid_actions(g)
        else:
            valid = obs.valid
    return time.perf_counter() - t0, done


def main() -> None:
    wanted = sys.argv[1:] or ["g50t", "sk48", "wa30"]
    env_dir = default_env_dir()
    by_prefix = {g.split("-")[0]: g for g in discover_games(env_dir)}

    print(f"{'game':6s} {'deepcopy':>9s} {'pickle':>8s} {'step':>7s} "
          f"{'gc on':>8s} {'gc off':>8s} {'speedup':>8s}")
    print(f"{'':6s} {'ms':>9s} {'ms':>8s} {'ms':>7s} {'st/s':>8s} {'st/s':>8s}")
    print("-" * 60)
    for pre in wanted:
        gid = by_prefix.get(pre)
        if gid is None:
            print(f"{pre}: not found")
            continue
        twin = Twin(gid, env_dir)
        root = twin.snapshot()
        Twin.step_game(root, Act(0))

        gc.enable()
        t = time.perf_counter()
        for _ in range(20):
            copy.deepcopy(root)
        dc_ms = (time.perf_counter() - t) / 20 * 1e3

        t = time.perf_counter()
        try:
            for _ in range(20):
                pickle.loads(pickle.dumps(root, -1))
            pk_ms = (time.perf_counter() - t) / 20 * 1e3
        except Exception as exc:
            pk_ms = float("nan")
            print(f"  ({pre}: pickle failed: {type(exc).__name__})")

        sec_on, n = time_steps(root)
        on = n / sec_on

        gc.disable()
        try:
            sec_off, n2 = time_steps(root)
            off = n2 / sec_off
        finally:
            gc.enable()

        print(f"{pre:6s} {dc_ms:9.1f} {pk_ms:8.1f} {sec_on / n * 1e3:7.2f} "
              f"{on:8.0f} {off:8.0f} {off / max(on, 1e-9):7.2f}x")


if __name__ == "__main__":
    main()
