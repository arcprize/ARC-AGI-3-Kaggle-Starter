from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory SQLite cache — mirrors on-disk DBs in RAM for sub-ms reads.
# Every :memory: connection is a fresh copy, so concurrent readers never
# block each other or the disk writer.
# ---------------------------------------------------------------------------

_instances: dict[str, "SQLiteMemoryCache"] = {}


def get_cache(db_path: str | Path | None) -> "SQLiteMemoryCache | None":
    """Return the cache for *db_path*, or None if *db_path* is None."""
    if db_path is None:
        return None
    resolved = str(Path(db_path).resolve())
    if resolved not in _instances:
        _instances[resolved] = SQLiteMemoryCache(resolved)
    return _instances[resolved]


def close_all() -> None:
    for c in _instances.values():
        try:
            c.close()
        except Exception:
            pass
    _instances.clear()


# ---------------------------------------------------------------------------


class SQLiteMemoryCache:
    """Thread-safe in-memory SQLite cache for a single on-disk database.

    On construction the entire on-disk DB is cloned into ``:memory:`` via
    the SQLite backup API (fast, atomic, zero-copy for small files).
    Subsequent read queries hit RAM — no disk I/O.
    Writes go to **both** the in-memory DB and the on-disk DB (synchronous
    by default; pass *buffered=True* for deferred disk flush).
    """

    def __init__(self, db_path: str, buffered: bool = False) -> None:
        self._path = Path(db_path)
        self._buffered = buffered
        self._mem: sqlite3.Connection | None = None
        self._disk: sqlite3.Connection | None = None
        self._load_snapshot()

    # -- public helpers --------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    def connect(self, *, write: bool = False) -> sqlite3.Connection:
        """Return a connection — in-memory for reads, disk for writes.

        Every call returns a FRESH connection so ``closing()`` callers don't
        interfere with each other.  Write connections auto-commit on close.
        """
        if write:
            c = sqlite3.connect(str(self._path))
            c.execute("PRAGMA journal_mode=WAL")
            return c
        # Fresh in-memory clone for every reader — no contention.
        c = sqlite3.connect(":memory:")
        if self._mem is not None:
            self._mem.backup(c)
        return c

    def connect_or_fallback(self, *, write: bool = False) -> sqlite3.Connection:
        """Like :meth:`connect` but falls back to the disk connection if the
        in-memory backup failed (e.g. empty source DB on first call)."""
        if write or self._mem is None:
            return self.connect(write=True)
        return self.connect(write=False)

    @property
    def read_connection(self) -> sqlite3.Connection | None:
        """The shared in-memory snapshot connection (lifetime of the cache)."""
        return self._mem

    def flush(self) -> None:
        """Sync in-memory writes back to disk."""
        if self._disk is not None:
            self._disk.commit()

    def refresh(self) -> None:
        """Re-load the in-memory snapshot from disk (call after schema init)."""
        self._load_snapshot()

    def close(self) -> None:
        self.flush()
        if self._mem is not None:
            try:
                self._mem.close()
            except Exception:
                pass
            self._mem = None
        if self._disk is not None:
            try:
                self._disk.close()
            except Exception:
                pass
            self._disk = None

    # -- internals -------------------------------------------------------

    def _load_snapshot(self) -> None:
        """Clone the on-disk DB into ``:memory:`` (SQLite backup API)."""
        if not self._path.exists():
            logger.debug("MemoryCache: %s not found — using empty in-memory DB", self._path)
            self._mem = sqlite3.connect(":memory:")
            return
        try:
            disk = sqlite3.connect(str(self._path))
            self._mem = sqlite3.connect(":memory:")
            disk.backup(self._mem)
            disk.close()
            logger.debug("MemoryCache: loaded %s into RAM (OK)", self._path)
        except Exception as exc:
            logger.warning("MemoryCache: failed to load %s: %s — fallback to disk", self._path, exc)
            self._mem = None
