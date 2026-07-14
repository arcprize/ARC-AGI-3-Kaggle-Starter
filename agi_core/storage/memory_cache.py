from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory SQLite cache — mirrors on-disk DBs in RAM for sub-ms reads.
# Every :memory: connection is a fresh copy, so concurrent readers never
# block each other or the disk writer.
# ---------------------------------------------------------------------------

_instances: dict[str, "SQLiteMemoryCache"] = {}
_instances_lock = threading.RLock()


def get_cache(db_path: str | Path | None) -> "SQLiteMemoryCache | None":
    """Return the cache for *db_path*, or None if *db_path* is None."""
    if db_path is None:
        return None
    resolved = str(Path(db_path).resolve())
    with _instances_lock:
        cache = _instances.get(resolved)
        if cache is None:
            cache = SQLiteMemoryCache(resolved)
            _instances[resolved] = cache
        return cache


def drop_cache(db_path: str | Path | None) -> None:
    if db_path is None:
        return
    resolved = str(Path(db_path).resolve())
    with _instances_lock:
        cache = _instances.pop(resolved, None)
    if cache is not None:
        try:
            cache.close()
        except Exception:
            pass


def close_all() -> None:
    with _instances_lock:
        caches = list(_instances.values())
        _instances.clear()
    for c in caches:
        try:
            c.close()
        except Exception:
            pass


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
        self._lock = threading.RLock()
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
            try:
                c.execute("PRAGMA journal_mode=WAL")
                return c
            except Exception:
                try:
                    c.close()
                except Exception:
                    pass
                raise
        # Fresh in-memory clone for every reader — no contention after the
        # short, serialised backup from the shared snapshot completes.
        c = self._clone_snapshot()
        return c if c is not None else sqlite3.connect(":memory:")

    def connect_or_fallback(self, *, write: bool = False) -> sqlite3.Connection:
        """Like :meth:`connect` but falls back to the disk connection if the
        in-memory backup failed (e.g. empty source DB on first call)."""
        if write:
            return self.connect(write=True)
        c = self._clone_snapshot()
        return c if c is not None else self.connect(write=True)

    @property
    def read_connection(self) -> sqlite3.Connection | None:
        """The shared in-memory snapshot connection (lifetime of the cache)."""
        return self._mem

    def flush(self) -> None:
        """Sync in-memory writes back to disk."""
        with self._lock:
            if self._disk is not None:
                self._disk.commit()

    def refresh(self) -> None:
        """Re-load the in-memory snapshot from disk (call after schema init)."""
        self._load_snapshot()

    def close(self) -> None:
        with self._lock:
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
        with self._lock:
            if self._mem is not None:
                try:
                    self._mem.close()
                except Exception:
                    pass
                self._mem = None
            if not self._path.exists():
                logger.debug("MemoryCache: %s not found — using empty in-memory DB", self._path)
                self._mem = sqlite3.connect(":memory:", check_same_thread=False)
                return
            disk: sqlite3.Connection | None = None
            snapshot: sqlite3.Connection | None = None
            try:
                disk = sqlite3.connect(str(self._path))
                snapshot = sqlite3.connect(":memory:", check_same_thread=False)
                disk.backup(snapshot)
                self._mem = snapshot
                logger.debug("MemoryCache: loaded %s into RAM (OK)", self._path)
            except Exception as exc:
                if snapshot is not None:
                    try:
                        snapshot.close()
                    except Exception:
                        pass
                logger.warning("MemoryCache: failed to load %s: %s — fallback to disk", self._path, exc)
                self._mem = None
            finally:
                if disk is not None:
                    try:
                        disk.close()
                    except Exception:
                        pass

    def _clone_snapshot(self) -> sqlite3.Connection | None:
        """Return a thread-local clone of the current shared snapshot."""
        connection = sqlite3.connect(":memory:")
        try:
            with self._lock:
                if self._mem is None:
                    connection.close()
                    return None
                self._mem.backup(connection)
            return connection
        except Exception:
            try:
                connection.close()
            except Exception:
                pass
            raise
