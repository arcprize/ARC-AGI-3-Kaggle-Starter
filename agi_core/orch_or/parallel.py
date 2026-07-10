from __future__ import annotations

import atexit
import logging
import os
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
    as_completed,
)
from multiprocessing import cpu_count
from threading import Lock
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread pool — for I/O-bound work (SQLite queries, etc.).
# Python's GIL prevents threads from parallelizing CPU-bound code; threads
# only help when they release the GIL (file/network I/O, C extensions).
# Default workers = min(4, cpu_count) to match actual I/O parallelism.
# ---------------------------------------------------------------------------


class ThreadPoolManager:
    _instance: ThreadPoolExecutor | None = None
    _lock: Lock = Lock()
    _max_workers: int = 0  # 0 = auto (resolved to min(4, cpu_count))
    _pending_futures: set[Future] = set()
    _pending_names: dict[Future, str] = {}

    _atexit_registered = False

    @classmethod
    def _resolve_workers(cls) -> int:
        if cls._max_workers <= 0:
            try:
                cpus = cpu_count() or 1
                return max(2, min(8, cpus))  # scale with CPU count up to 8
            except Exception:
                return 2
        return cls._max_workers

    @classmethod
    def set_max_workers(cls, n: int) -> None:
        cls._max_workers = n
        cls._shutdown()
        cls._instance = None

    @classmethod
    def executor(cls) -> ThreadPoolExecutor:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    n = cls._resolve_workers()
                    logger.debug("ThreadPoolManager pool: %d workers", n)
                    cls._instance = ThreadPoolExecutor(
                        max_workers=n,
                        thread_name_prefix="orch_or",
                    )
                    if not cls._atexit_registered:
                        atexit.register(cls._shutdown)
                        cls._atexit_registered = True
        return cls._instance

    @classmethod
    def _shutdown(cls) -> None:
        inst = cls._instance
        if inst is not None:
            inst.shutdown(wait=False)
            cls._instance = None
        cls._pending_futures.clear()
        cls._pending_names.clear()

    @classmethod
    def active_worker_count(cls) -> int:
        cls._collect_done_pending()
        return len(cls._pending_futures)

    @classmethod
    def _collect_done_pending(cls) -> None:
        done = {f for f in cls._pending_futures if f.done()}
        cls._pending_futures -= done
        for f in done:
            cls._pending_names.pop(f, None)

    @classmethod
    def _cancel_pending_future(cls, future: Future) -> bool:
        """Attempt to cancel a pending future.

        Returns True when the future is no longer active and has been removed
        from tracking. Returns False when the work is still running.
        """
        if future.done():
            cls._pending_futures.discard(future)
            cls._pending_names.pop(future, None)
            return True
        if future.cancel():
            cls._pending_futures.discard(future)
            cls._pending_names.pop(future, None)
            return True
        return False

    @classmethod
    def run_all(
        cls,
        tasks: list[tuple[Callable[[], Any], str]],
        timeout: float | None = None,
    ) -> list[tuple[Any, str]]:
        if not tasks:
            cls._collect_done_pending()
            return []
        exec = cls.executor()
        futures: dict[Future, str] = {}
        for fn, name in tasks:
            f = exec.submit(fn)
            futures[f] = name
        cls._pending_futures.update(futures.keys())
        cls._pending_names.update(futures)

        done_set = set()
        if timeout is not None and timeout > 0:
            try:
                for f in as_completed(futures.keys(), timeout=timeout):
                    done_set.add(f)
            except TimeoutError:
                pass
        else:
            done_set = set(futures.keys())
            wait(done_set)

        results: list[tuple[Any, str]] = []
        for f in done_set:
            name = futures[f]
            cls._pending_futures.discard(f)
            cls._pending_names.pop(f, None)
            try:
                results.append((f.result(), name))
            except Exception as exc:
                logger.debug("task %s failed: %s", name, exc)
                results.append((None, name))
        if logger.isEnabledFor(logging.DEBUG) and len(futures) > 1:
            finished = len(results)
            logger.debug("run_all: %d/%d tasks done (%s)", finished, len(futures),
                         ", ".join(r[1] if r[0] is not None else f"{r[1]}:FAIL" for r in results))

        # Cancel remaining futures when possible; keep running work tracked so
        # health checks still see stale tasks that survived the timeout.
        remaining = set(futures.keys()) - done_set
        cancelled = 0
        for f in remaining:
            if cls._cancel_pending_future(f):
                cancelled += 1
        if cancelled:
            logger.debug("run_all: %d tasks cancelled after timeout", cancelled)
        running = len(remaining) - cancelled
        if running:
            logger.debug("run_all: %d tasks still running after timeout", running)
        return results

    @classmethod
    def run_in_parallel(
        cls,
        tasks: list[tuple[Callable[[], Any], str]],
        timeout: float | None = None,
    ) -> list[tuple[Any, str]]:
        if not tasks:
            return []
        exec = cls.executor()
        futures: dict[Future, str] = {exec.submit(fn): name for fn, name in tasks}
        cls._pending_futures.update(futures.keys())
        cls._pending_names.update(futures)

        done, _ = wait(futures.keys(), timeout=timeout, return_when=FIRST_COMPLETED)
        results: list[tuple[Any, str]] = []
        for f in done:
            name = futures[f]
            cls._pending_futures.discard(f)
            cls._pending_names.pop(f, None)
            try:
                results.append((f.result(), name))
            except Exception as exc:
                logger.debug("task %s failed: %s", name, exc)
                results.append((None, name))

        for f in set(futures.keys()) - done:
            cls._cancel_pending_future(f)
        return results

    @classmethod
    def submit(cls, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        """Submit a task and track it in pending futures."""
        f = cls.executor().submit(fn, *args, **kwargs)
        cls._pending_futures.add(f)
        return f

    @classmethod
    def map(cls, fn, *iterables, timeout=None):
        return cls.executor().map(fn, *iterables, timeout=timeout)


# ---------------------------------------------------------------------------
# Process pool — for CPU-bound work that can bypass the GIL.
# Data must be picklable.  Enables TRUE parallelism for heavy operations
# like A* pathfinding, component detection, and grid analysis.
# Default workers = min(8, cpu_count).
# Lazily created; env AGI_PLANNER_WORKERS applies to both pools.
# ---------------------------------------------------------------------------


class ProcessPoolManager:
    _instance: ProcessPoolExecutor | None = None
    _lock: Lock = Lock()
    _max_workers: int = 0
    _pending_futures: set[Future] = set()
    _pending_names: dict[Future, str] = {}

    _atexit_registered = False

    @classmethod
    def _resolve_workers(cls) -> int:
        if cls._max_workers <= 0:
            try:
                return min(8, cpu_count() or 1)
            except Exception:
                return 2
        return cls._max_workers

    @classmethod
    def set_max_workers(cls, n: int) -> None:
        cls._max_workers = n
        cls._shutdown()
        cls._instance = None

    @classmethod
    def executor(cls) -> ProcessPoolExecutor:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    n = cls._resolve_workers()
                    logger.debug("ProcessPoolManager pool: %d workers", n)
                    cls._instance = ProcessPoolExecutor(
                        max_workers=n,
                    )
                    if not cls._atexit_registered:
                        atexit.register(cls._shutdown)
                        cls._atexit_registered = True
        return cls._instance

    @classmethod
    def _shutdown(cls) -> None:
        inst = cls._instance
        if inst is not None:
            inst.shutdown(wait=False)
            cls._instance = None
        cls._pending_futures.clear()
        cls._pending_names.clear()

    @classmethod
    def _cancel_pending_future(cls, future: Future) -> bool:
        if future.done():
            cls._pending_futures.discard(future)
            cls._pending_names.pop(future, None)
            return True
        if future.cancel():
            cls._pending_futures.discard(future)
            cls._pending_names.pop(future, None)
            return True
        return False

    @classmethod
    def submit(cls, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        f = cls.executor().submit(fn, *args, **kwargs)
        cls._pending_futures.add(f)
        return f

    @classmethod
    def run_all(
        cls,
        tasks: list[tuple[Callable[[], Any], str]],
        timeout: float | None = None,
    ) -> list[tuple[Any, str]]:
        if not tasks:
            return []
        exec = cls.executor()
        futures: dict[Future, str] = {}
        for fn, name in tasks:
            f = exec.submit(fn)
            futures[f] = name
        cls._pending_futures.update(futures.keys())
        cls._pending_names.update(futures)
        done_set = set()
        if timeout is not None and timeout > 0:
            try:
                for f in as_completed(futures.keys(), timeout=timeout):
                    done_set.add(f)
            except TimeoutError:
                pass
        else:
            done_set = set(futures.keys())
            wait(done_set)
        results: list[tuple[Any, str]] = []
        for f in done_set:
            name = futures[f]
            cls._pending_futures.discard(f)
            cls._pending_names.pop(f, None)
            try:
                results.append((f.result(), name))
            except Exception as exc:
                logger.debug("process task %s failed: %s", name, exc)
                results.append((None, name))
        # Cancel remaining futures
        remaining = set(futures.keys()) - done_set
        for f in remaining:
            cls._cancel_pending_future(f)
        return results

    @classmethod
    def map(cls, fn, *iterables, timeout=None):
        return cls.executor().map(fn, *iterables, timeout=timeout)


# ---------------------------------------------------------------------------
# Sync both pool worker counts from the same env var.
# ---------------------------------------------------------------------------

def sync_pool_worker_counts(n: int, thread_workers: int = -1) -> None:
    """Set worker counts for both thread and process pools.
    
    Args:
        n: Worker count for the process pool (CPU-bound). 0 = auto-resolve.
        thread_workers: Worker count for the thread pool (I/O-bound).
            -1 (default) = auto-resolve (max(2, min(8, cpu_count))).
            0 = same as ``n``.
    """
    ProcessPoolManager.set_max_workers(n)
    if thread_workers == -1:
        ThreadPoolManager.set_max_workers(0)  # auto-resolve
    elif thread_workers == 0:
        ThreadPoolManager.set_max_workers(n)
    else:
        ThreadPoolManager.set_max_workers(thread_workers)
