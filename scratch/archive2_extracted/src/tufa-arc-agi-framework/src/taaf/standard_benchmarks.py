"""Team-agreed standard benchmarks: shared game lists + weights so
results are directly comparable across runs. Each benchmark is frozen
once published; new ones land as new functions.
"""

from __future__ import annotations

import collections

import taaf.benchmark
import taaf.competition_arcade
import taaf.game
import taaf.game_api
import taaf.solver


def make_benchmark_all_games(solver: taaf.solver.Solver | None = None) -> taaf.benchmark.Benchmark:
    """All-game benchmark: ``re_arc``'s ``train`` + ``eval`` datasets.
    Pinning to those JSON files keeps the benchmark stable as games are
    added.

    Weights: ``20`` for ``official``-tagged games regardless of level
    count; ``min(number_of_levels, 8)`` otherwise. Callers are free to
    bump ``n_passes`` — the games and weights are the benchmark identity.
    """
    import re_arc

    train_ids = re_arc.list_game_ids(datasets="train")
    eval_ids = re_arc.list_game_ids(datasets="eval")
    game_ids = list(train_ids) + list(eval_ids)
    official_ids = set(re_arc.list_game_ids(datasets=["train", "eval"], include_tags="official"))

    # number_of_levels requires starting each game.
    session = taaf.game.RunSession()
    games: list[taaf.game.Game] = []
    weights: list[float] = []
    for gid in game_ids:
        probe = taaf.game_api.GameAPI(env_name=gid)
        probe.start_game(session=session)
        n_levels = probe.number_of_levels
        if gid in official_ids:
            weights.append(20.0)
        else:
            weights.append(float(min(n_levels, 8)))
        games.append(taaf.game_api.GameAPI(env_name=gid))

    return taaf.benchmark.Benchmark(
        label="all_games",
        games=games,
        solver=solver,
        n_passes=1,
        game_weights=weights,
    )


def make_benchmark_kaggle_official_110(
    solver: taaf.solver.Solver | None = None,
    *,
    arcade_spec: taaf.game_api.ArcadeSpec | None = None,
    competition_clone_ids: bool = False,
    competition_sim: bool = False,
) -> taaf.benchmark.Benchmark:
    """Kaggle submission-shaped public benchmark (R2.57).

    The 25 official games are repeated round-robin to 110 independent
    ``GameAPI`` entries, with ``n_passes=1``. When
    ``competition_clone_ids=True``, the repeated entries use the same
    deterministic clone IDs exposed by
    ``CompetitionArcadeServer.official_110()``, making the benchmark
    compatible with the competition-style Arcade that does not support
    multiple runs for the same game ID.

    ``competition_sim=True`` targets a local competition-arcade simulator
    (R11.13) started lazily on whichever process runs the benchmark: it
    sets a ``competition_sim`` ``ArcadeSpec`` and forces
    ``competition_clone_ids`` so the game IDs line up with the server's
    official-110 clone set. This lets a non-submission run exercise the
    submission-shaped arcade without an actual leaderboard submission.
    """
    import re_arc

    if competition_sim:
        if arcade_spec is not None:
            raise ValueError("Pass either competition_sim=True or an explicit arcade_spec, not both.")
        arcade_spec = taaf.game_api.ArcadeSpec(competition_sim=True)
        competition_clone_ids = True
    official_ids = list(re_arc.list_game_ids(datasets=["train", "eval"], include_tags="official"))
    if len(official_ids) != 25:
        raise RuntimeError(f"Expected 25 official games for Kaggle benchmark, got {len(official_ids)}.")
    if competition_clone_ids:
        # Competition arcade registers each clone as a distinct EnvironmentInfo,
        # so the natural ``info.game_id`` (``p005`` etc.) is already unique — no
        # external override needed.
        game_ids = taaf.competition_arcade.clone_game_ids(official_ids, total_runs=110)
        external_ids: list[str | None] = [None] * len(game_ids)
    else:
        # No competition arcade to synthesize unique env IDs — re_arc only knows
        # the 25 base IDs. Override each instance's public ``game_id`` with
        # ``{base}_{i}`` where ``i`` is the per-base occurrence index, so
        # ``Benchmark.game_runs`` has unique IDs (Benchmark.run asserts this)
        # while each instance still plays its base re_arc env underneath.
        counts: collections.Counter[str] = collections.Counter()
        game_ids = []
        external_ids = []
        for i in range(110):
            base = official_ids[i % len(official_ids)]
            game_ids.append(base)
            external_ids.append(f"{base}_{counts[base]}")
            counts[base] += 1
    games: list[taaf.game.Game]
    if arcade_spec is None:
        games = [taaf.game_api.GameAPI(env_name=g, external_game_id=ext) for g, ext in zip(game_ids, external_ids)]
    else:
        games = [
            taaf.game_api.GameAPI(env_name=g, external_game_id=ext, arcade_spec=arcade_spec)
            for g, ext in zip(game_ids, external_ids)
        ]
    return taaf.benchmark.Benchmark(
        label="kaggle_official_110",
        games=games,
        solver=solver,
        n_passes=1,
    )
