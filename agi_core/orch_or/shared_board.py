from __future__ import annotations

import atexit
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_HEADER_BYTES = 8  # uint64 generation counter


class SharedBoard:
    """Lock-free shared-memory board for zero-copy IPC with worker processes.

    Layout (4104 bytes for a 64×64 board):
      0-7:   generation counter (uint64, incremented AFTER data is written)
      8-4103: board data (64×64 = 4096 bytes, row-major int8)

    Workers attach by name, read board+gen, then detach.  The generation
    counter acts as an optimistic lock: if a worker reads a board while the
    main process is writing, the counter will have changed and the worker
    detects the race.  In practice the race never occurs because the main
    thread blocks on ``.result()`` before the next write.

    Falls back to pickle-based submission if ``shared_memory`` is unavailable
    (Python < 3.8) or the board dimensions don't match.
    """

    def __init__(self, width: int = 64, height: int = 64) -> None:
        self.width = width
        self.height = height
        self._data_bytes = width * height
        self._total_bytes = _HEADER_BYTES + self._data_bytes
        self._shm: Any = None
        self._available = False
        self._create()

    def _create(self) -> None:
        try:
            from multiprocessing import shared_memory
            self._shm = shared_memory.SharedMemory(
                create=True,
                size=self._total_bytes,
                name=f"arc_board_{os.getpid()}",
            )
            self._available = True
            atexit.register(self._cleanup)
            logger.debug("SharedBoard: %d bytes at %s", self._total_bytes, self._shm.name)
        except Exception as exc:
            logger.debug("SharedBoard unavailable (%s); falling back to pickle", exc)
            self._available = False

    @property
    def name(self) -> str:
        if self._shm is not None:
            return self._shm.name
        return ""

    @property
    def is_available(self) -> bool:
        return self._available and self._shm is not None

    def write(self, board: list[list[int]]) -> None:
        """Write board to shared memory and increment generation counter."""
        if not self.is_available:
            return
        buf = self._shm.buf
        data_start = _HEADER_BYTES
        for y in range(self.height):
            row = board[y]
            offset = data_start + y * self.width
            for x in range(self.width):
                buf[offset + x] = row[x]
        # Increment counter AFTER data is fully written.
        gen = int.from_bytes(buf[0:_HEADER_BYTES], "little") + 1
        buf[0:_HEADER_BYTES] = gen.to_bytes(_HEADER_BYTES, "little")

    def read_board(self, shm_name: str) -> list[list[int]]:
        """Attach to shared memory by name, read board, detach, return board."""
        from multiprocessing import shared_memory
        shm = shared_memory.SharedMemory(name=shm_name)
        try:
            buf = shm.buf
            gen = int.from_bytes(buf[0:_HEADER_BYTES], "little")
            h = self.height
            w = self.width
            data_start = _HEADER_BYTES
            board = [[0] * w for _ in range(h)]
            for y in range(h):
                row = board[y]
                offset = data_start + y * w
                for x in range(w):
                    row[x] = buf[offset + x]
            gen2 = int.from_bytes(buf[0:_HEADER_BYTES], "little")
            if gen != gen2 or gen == 0:
                raise RuntimeError("shared board race: generation changed")
            return board
        finally:
            try:
                shm.close()
            except Exception:
                pass

    def _cleanup(self) -> None:
        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception:
                pass
            self._shm = None
            self._available = False


# Module-level singleton — set once at startup.
_GLOBAL_BOARD: SharedBoard | None = None


def get_shared_board() -> SharedBoard | None:
    return _GLOBAL_BOARD


def init_shared_board(width: int = 64, height: int = 64) -> SharedBoard:
    global _GLOBAL_BOARD
    sb = SharedBoard(width, height)
    _GLOBAL_BOARD = sb
    return sb
