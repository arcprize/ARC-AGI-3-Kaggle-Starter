from __future__ import annotations

"""Wrapper functions for ProcessPoolManager that receive the board via shared
memory instead of pickle.  These run in worker subprocesses.

Each function:
  1. Attaches to the shared memory board by name
  2. Reconstructs the board at memory speed (no pickle)
  3. Runs the original computation
  4. Detaches
  5. Returns the result (small — components list ~1KB)
"""

from multiprocessing import shared_memory
from typing import Any

from agi_core.orch_or.analyzer import _find_components, _detect_sprites

_HEADER_BYTES = 8


def _read_board(name: str, width: int, height: int) -> list[list[int]]:
    shm = shared_memory.SharedMemory(name=name)
    try:
        buf = shm.buf
        data_start = _HEADER_BYTES
        board = [[0] * width for _ in range(height)]
        for y in range(height):
            row = board[y]
            offset = data_start + y * width
            for x in range(width):
                row[x] = buf[offset + x]
        return board
    finally:
        try:
            shm.close()
        except Exception:
            pass


def shared_find_components(
    shm_name: str,
    shm_width: int,
    shm_height: int,
    bg_color: int,
) -> Any:
    board = _read_board(shm_name, shm_width, shm_height)
    return _find_components(board, shm_width, shm_height, bg_color)


def shared_detect_sprites(
    shm_name: str,
    grid_scale: int,
    bg_color: int,
    player_pos: tuple[int, int] | None,
    goal_positions: list[tuple[int, int]],
    components: list,
    width: int,
    height: int,
) -> Any:
    board = _read_board(shm_name, width, height)
    return _detect_sprites(board, grid_scale, bg_color, player_pos, goal_positions, components, width, height)
