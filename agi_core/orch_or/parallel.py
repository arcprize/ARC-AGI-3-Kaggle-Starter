from __future__ import annotations

import atexit
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from threading import Lock
from typing import Any, Callable


class ThreadPoolManager:
    _instance: ThreadPoolExecutor | None = None
    _lock: Lock = Lock()
    _max_workers: int = 4

    @classmethod
    def set_max_workers(cls, n: int) -> None:
        cls._max_workers = max(1, n)
        cls._shutdown()
        cls._instance = None

    _atexit_registered = False

    @classmethod
    def executor(cls) -> ThreadPoolExecutor:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = ThreadPoolExecutor(
                        max_workers=cls._max_workers,
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

    @classmethod
    def run_in_parallel(
        cls,
        tasks: list[tuple[Callable[[], Any], str]],
        timeout: float | None = None,
    ) -> list[tuple[Any, str]]:
        if not tasks:
            return []
        exec = cls.executor()
        futures = {exec.submit(fn): name for fn, name in tasks}
        done, _ = wait(futures.keys(), timeout=timeout, return_when=FIRST_COMPLETED)
        results: list[tuple[Any, str]] = []
        for future in done:
            name = futures[future]
            try:
                results.append((future.result(), name))
            except Exception:
                results.append((None, name))
        for future in set(futures.keys()) - done:
            future.cancel()
        return results

    @classmethod
    def run_all(
        cls,
        tasks: list[tuple[Callable[[], Any], str]],
        timeout: float | None = None,
    ) -> list[tuple[Any, str]]:
        if not tasks:
            return []
        exec = cls.executor()
        futures = {exec.submit(fn): name for fn, name in tasks}
        done, _ = wait(futures.keys(), timeout=timeout)
        results: list[tuple[Any, str]] = []
        for future in done:
            name = futures[future]
            try:
                results.append((future.result(), name))
            except Exception:
                results.append((None, name))
        for future in set(futures.keys()) - done:
            future.cancel()
        return results
