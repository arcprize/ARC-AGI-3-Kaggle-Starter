from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agi_core.knowledge.beam_solver import solve_beam_puzzle
from agi_core.orch_or.game_model import PlanStep
from agi_core.orch_or.search import astar

if TYPE_CHECKING:
    from agi_core.orch_or.game_model import GameModel


def _is_beam_controller(game_model: GameModel) -> bool:
    features = game_model.features
    if not features.has_beam or not features.beam_positions or game_model.player_pos is None:
        return False
    px, py = game_model.player_pos
    return any(abs(bx - px) <= 6 and abs(by - py) <= 6
               for bx, by in features.beam_positions)


def _get_beam_cells(game_model: GameModel) -> list[tuple[int, int]]:
    comps = [c for c in game_model.components if c.dominant_color == game_model.features.beam_color and not c.touches_border]
    if comps:
        return comps[0].cells
    return game_model.features.beam_positions


def goal_seeker_strategy(game_model: GameModel, context: dict | None = None) -> list[PlanStep]:
    steps: list[PlanStep] = []
    player = game_model.player_pos
    goals = game_model.goal_positions
    ctx = context or {}
    if player is None or not goals:
        return steps
    px, py = player

    non_beam_set = ctx.get("non_beam_controllers", set())
    is_beam_controller = (game_model.features.has_beam
                          and game_model.features.player_color not in non_beam_set
                          and _is_beam_controller(game_model))

    if is_beam_controller and game_model.features.mirror_axis >= 0:
        beam_cells = _get_beam_cells(game_model)
        solver_result = solve_beam_puzzle(
            goal_positions=goals,
            beam_cells=beam_cells,
            mirror_axis=game_model.features.mirror_axis,
            player_pos=player,
            current_mirror_pos=player,
            width=game_model.width,
            height=game_model.height,
            memory_db_path=ctx.get("memory_db_path"),
            game_id=ctx.get("game_step", ""),
        )
        mirror_dir = solver_result.get("mirror_direction")
        if mirror_dir and mirror_dir not in ctx.get("failed_directions", set()):
            return [_move_step(mirror_dir)]
        switch_count = int(ctx.get("switch_attempted", 0))
        all_coverable = solver_result.get("all_coverable", False)
        uncovered = solver_result.get("uncovered_goals", [])
        if not mirror_dir and switch_count < 2:
            return [_switch_step()]
        if uncovered:
            px2, py2 = player
            nearest = min(uncovered, key=lambda g: abs(g[0] - px2) + abs(g[1] - py2))
            if abs(nearest[0] - px2) + abs(nearest[1] - py2) <= 3:
                if switch_count < 2:
                    return [_switch_step()]
                return [_move_step("right")]
            if all_coverable:
                if switch_count < 2:
                    return [_switch_step()]
        return steps

    goal_history: list[list[tuple[int, int]]] = ctx.get("goal_history", [])
    stable_goal: tuple[int, int] | None = ctx.get("stable_goal")

    cluster = _compute_goal_cluster(goal_history, goals, stable_goal)

    min_dist = min(abs(gx - px) + abs(gy - py) for gx, gy in goals)
    if min_dist > 8:
        return _approach_phase(player, cluster, game_model, ctx)
    return _proximity_phase(player, goals, goal_history, game_model, ctx)


def _compute_goal_cluster(
    goal_history: list[list[tuple[int, int]]],
    current_goals: list[tuple[int, int]],
    stable_goal: tuple[int, int] | None = None,
) -> tuple[int, int]:
    all_positions: list[tuple[int, int]] = list(current_goals)
    for frame_goals in goal_history:
        all_positions.extend(frame_goals)
    if stable_goal is not None:
        all_positions.append(stable_goal)
    if not all_positions:
        return current_goals[0]
    xs = [p[0] for p in all_positions]
    ys = [p[1] for p in all_positions]
    xs.sort()
    ys.sort()
    cx = xs[len(xs) // 2]
    cy = ys[len(ys) // 2]
    return (cx, cy)


def _approach_phase(
    player: tuple[int, int],
    cluster: tuple[int, int],
    game_model: GameModel,
    ctx: dict,
) -> list[PlanStep]:
    walkable = _expanded_walkable(game_model, ctx)
    path = astar(player, cluster, walkable, game_model.width, game_model.height)
    if path and len(path) > 1:
        steps: list[PlanStep] = []
        for i in range(1, len(path), 3):
            prev = path[i - 1]
            curr = path[i]
            steps.append(_move_step(_direction(prev, curr)))
        return steps
    return _fallback_move(player, cluster, game_model, ctx)


def _proximity_phase(
    player: tuple[int, int],
    goals: list[tuple[int, int]],
    goal_history: list[list[tuple[int, int]]],
    game_model: GameModel,
    ctx: dict,
) -> list[PlanStep]:
    px, py = player
    nearby = [g for g in goals if abs(g[0] - px) + abs(g[1] - py) <= 8]
    if not nearby:
        return []
    _goal_stability: dict[tuple[int, int], int] = {}
    for frame_goals in goal_history:
        for g in frame_goals:
            for ng in nearby:
                if abs(g[0] - ng[0]) + abs(g[1] - ng[1]) <= 3:
                    _goal_stability[ng] = _goal_stability.get(ng, 0) + 1
    for g in goals:
        _goal_stability.setdefault(g, 0)

    walkable = _expanded_walkable(game_model, ctx)
    obs_set = set((ox, oy) for ox, oy, otype in game_model.obstacles if otype == "interactive")
    scored = []
    for gx, gy in nearby:
        dist = abs(gx - px) + abs(gy - py)
        stability = _goal_stability.get((gx, gy), 0)
        blocked = (gx, gy) in obs_set
        adj_walkable = any((gx + dx, gy + dy) in walkable for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)))
        score = stability * 10 - dist * 2 - (50 if blocked else 0) + (20 if adj_walkable else 0)
        scored.append((score, gx, gy))
    scored.sort(key=lambda x: -x[0])
    _, tx, ty = scored[0]

    path = astar(player, (tx, ty), walkable, game_model.width, game_model.height)
    if path and len(path) > 1:
        for i in range(1, len(path), 3):
            prev = path[i - 1]
            curr = path[i]
            return [_move_step(_direction(prev, curr))]
    for gx, gy in nearby:
        path = astar(player, (gx, gy), walkable, game_model.width, game_model.height)
        if path and len(path) > 1:
            return [_move_step(_direction(player, path[1]))]
    return []


def _expanded_walkable(game_model: GameModel, ctx: dict) -> set[tuple[int, int]]:
    walkable = set(game_model.walkable)
    for c in game_model.components:
        if not c.touches_border:
            for cx, cy in c.cells:
                walkable.add((cx, cy))
    return walkable


def _fallback_move(
    player: tuple[int, int],
    target: tuple[int, int],
    game_model: GameModel,
    ctx: dict,
) -> list[PlanStep]:
    px, py = player
    tx, ty = target
    dx, dy = tx - px, ty - py
    candidates = []
    if dx > 0: candidates.append("right")
    elif dx < 0: candidates.append("left")
    if dy > 0: candidates.append("down")
    elif dy < 0: candidates.append("up")
    failed = ctx.get("failed_directions", set())
    for d in candidates:
        if d in failed:
            continue
        dv = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
        dvx, dvy = dv.get(d, (0, 0))
        nx, ny = px + dvx, py + dvy
        if (nx, ny) in game_model.walkable:
            return [_move_step(d)]
    return []


def _move_step(direction: str) -> PlanStep:
    return PlanStep(
        action="move",
        params={"direction": direction},
        source="goal_seeker",
        preconditions=frozenset(),
        expected_effect=f"move_{direction}",
    )


def _switch_step() -> PlanStep:
    return PlanStep(
        action="switch",
        params={},
        source="goal_seeker",
        preconditions=frozenset(),
        expected_effect="switch_entity",
    )


def _direction(from_pos: tuple[int, int], to_pos: tuple[int, int]) -> str:
    fx, fy = from_pos
    tx, ty = to_pos
    if tx > fx: return "right"
    if tx < fx: return "left"
    if ty > fy: return "down"
    if ty < fy: return "up"
    return "right"
