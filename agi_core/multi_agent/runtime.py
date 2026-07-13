from __future__ import annotations

import importlib
import logging
import multiprocessing
import queue
import time
from concurrent.futures import Future
from dataclasses import dataclass
from multiprocessing import Process
from multiprocessing.queues import Queue
from threading import RLock
from typing import Any

from agi_core.multi_agent.payload import AgentFrameSnapshot
from agi_core.orch_or.parallel import ThreadPoolManager

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AgentOutcomeUpdate:
    action_name: str
    outcome: str
    subgoal: str | None
    success: bool


@dataclass(slots=True)
class WorkerCommand:
    kind: str
    request_id: int = 0
    payload: Any = None


@dataclass(slots=True)
class WorkerResult:
    kind: str
    request_id: int
    ok: bool
    payload: Any = None
    error: str = ""


def _resolve_agent_class(class_path: str) -> type[Any]:
    module_name, qualname = class_path.split(":", 1)
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _encode_command(command: WorkerCommand) -> tuple[str, int, Any]:
    return (command.kind, int(command.request_id), command.payload)


def _decode_command(message: Any) -> WorkerCommand | None:
    if not isinstance(message, tuple) or len(message) != 3:
        return None
    return WorkerCommand(kind=str(message[0]), request_id=int(message[1]), payload=message[2])


def _encode_result(result: WorkerResult) -> tuple[str, int, bool, Any, str]:
    return (
        result.kind,
        int(result.request_id),
        bool(result.ok),
        result.payload,
        str(result.error or ""),
    )


def _decode_result(message: Any) -> WorkerResult | None:
    if not isinstance(message, tuple) or len(message) != 5:
        return None
    return WorkerResult(
        kind=str(message[0]),
        request_id=int(message[1]),
        ok=bool(message[2]),
        payload=message[3],
        error=str(message[4] or ""),
    )


def _worker_main(
    agent_id: str,
    agent_class_path: str,
    command_queue: Queue,
    result_queue: Queue,
) -> None:
    agent_cls = _resolve_agent_class(agent_class_path)
    agent = agent_cls()
    if hasattr(agent, "agent_id"):
        try:
            agent.agent_id = agent_id
        except Exception:
            pass
    result_queue.put(_encode_result(WorkerResult(kind="ready", request_id=0, ok=True)))
    while True:
        command = _decode_command(command_queue.get())
        if command is None:
            continue
        if command.kind == "shutdown":
            break
        try:
            if command.kind == "produce":
                proposals = agent.produce(command.payload)
                result_queue.put(
                    _encode_result(WorkerResult(
                        kind="produce",
                        request_id=command.request_id,
                        ok=True,
                        payload=proposals,
                    ))
                )
            elif command.kind == "apply_outcome":
                update = command.payload
                agent.apply_cognition(
                    str(update.action_name),
                    str(update.outcome),
                    update.subgoal,
                )
                agent.note_outcome(bool(update.success))
            elif command.kind == "reset":
                agent.reset()
        except Exception as exc:
            if command.kind == "produce":
                result_queue.put(
                    _encode_result(WorkerResult(
                        kind="produce",
                        request_id=command.request_id,
                        ok=False,
                        error=f"{type(exc).__name__}: {exc}",
                    ))
                )


class _ProcessTaskHandle:
    def __init__(self, worker: "_ProcessAgentWorker", request_id: int) -> None:
        self._worker = worker
        self._request_id = request_id
        self._done = False
        self._result: Any = None
        self._exception: Exception | None = None

    def done(self) -> bool:
        self._worker.drain_results()
        if not self._done and not self._worker.is_alive():
            self._worker.fail_pending(RuntimeError("process worker exited"))
        return self._done

    def result(self, timeout: float | None = None) -> Any:
        if not self._done:
            self._worker.wait_for_request(self._request_id, timeout=timeout)
        if self._exception is not None:
            raise self._exception
        return self._result

    def set_result(self, value: Any) -> None:
        self._result = value
        self._done = True

    def set_exception(self, exc: Exception) -> None:
        self._exception = exc
        self._done = True

    def cancel(self) -> bool:
        return False


class _ProcessAgentWorker:
    def __init__(self, agent_id: str, agent_class_path: str) -> None:
        self.agent_id = agent_id
        self.agent_class_path = agent_class_path
        self.command_queue: Queue = multiprocessing.Queue()
        self.result_queue: Queue = multiprocessing.Queue()
        self.process: Process | None = None
        self._next_request_id = 1
        self._pending: dict[int, _ProcessTaskHandle] = {}

    def ensure_started(self) -> None:
        if self.process is not None and self.process.is_alive():
            return
        self.process = multiprocessing.Process(
            target=_worker_main,
            args=(self.agent_id, self.agent_class_path, self.command_queue, self.result_queue),
            daemon=True,
        )
        self.process.start()
        self._wait_until_ready()

    def is_alive(self) -> bool:
        return bool(self.process is not None and self.process.is_alive())

    def submit_produce(self, snapshot: AgentFrameSnapshot) -> _ProcessTaskHandle | None:
        self.drain_results()
        if any(not handle._done for handle in self._pending.values()):
            return None
        self.ensure_started()
        request_id = self._next_request_id
        self._next_request_id += 1
        handle = _ProcessTaskHandle(self, request_id)
        self._pending[request_id] = handle
        self.command_queue.put(_encode_command(WorkerCommand(kind="produce", request_id=request_id, payload=snapshot)))
        return handle

    def apply_outcome(self, update: AgentOutcomeUpdate) -> None:
        self.ensure_started()
        self.command_queue.put(_encode_command(WorkerCommand(kind="apply_outcome", payload=update)))

    def reset(self) -> None:
        self.drain_results()
        self._pending.clear()
        if self.process is not None and self.process.is_alive():
            self.command_queue.put(_encode_command(WorkerCommand(kind="reset")))

    def shutdown(self) -> None:
        self.drain_results()
        self._pending.clear()
        if self.process is not None:
            if self.process.is_alive():
                try:
                    self.command_queue.put(_encode_command(WorkerCommand(kind="shutdown")))
                except Exception:
                    pass
                self.process.join(timeout=0.25)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=0.25)
            self.process = None
        try:
            self.command_queue.close()
        except Exception:
            pass
        try:
            self.result_queue.close()
        except Exception:
            pass

    def fail_pending(self, exc: Exception) -> None:
        for request_id, handle in list(self._pending.items()):
            if not handle._done:
                handle.set_exception(exc)
            self._pending.pop(request_id, None)

    def drain_results(self) -> None:
        while True:
            try:
                result = self.result_queue.get_nowait()
            except queue.Empty:
                break
            self._apply_result(result)

    def wait_for_request(self, request_id: int, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while True:
            handle = self._pending.get(request_id)
            if handle is None or handle._done:
                return
            if not self.is_alive():
                self.fail_pending(RuntimeError("process worker exited"))
                return
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if remaining is not None and remaining <= 0.0:
                raise TimeoutError(f"worker result timed out for {self.agent_id}")
            try:
                result = self.result_queue.get(timeout=remaining if remaining is not None else 0.05)
            except queue.Empty:
                continue
            self._apply_result(result)

    def _apply_result(self, result: Any) -> None:
        if isinstance(result, WorkerResult):
            decoded = result
        else:
            decoded = _decode_result(result)
        if decoded is None:
            return
        if decoded.kind == "ready":
            return
        handle = self._pending.pop(decoded.request_id, None)
        if handle is None:
            return
        if decoded.ok:
            handle.set_result(decoded.payload)
        else:
            handle.set_exception(RuntimeError(decoded.error or "worker produce failed"))

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + 5.0
        while True:
            if self.process is None or not self.process.is_alive():
                raise RuntimeError(f"worker {self.agent_id} exited during startup")
            remaining = max(0.0, deadline - time.monotonic())
            if remaining <= 0.0:
                raise RuntimeError(f"worker {self.agent_id} did not become ready")
            try:
                result = self.result_queue.get(timeout=remaining)
            except queue.Empty:
                continue
            decoded = _decode_result(result)
            if decoded is not None and decoded.kind == "ready":
                return
            self._apply_result(decoded)


class MultiAgentExecutorBackend:
    def active_worker_count(self) -> int:
        raise NotImplementedError

    def submit(
        self,
        agent_id: str,
        agent: Any,
        snapshot: AgentFrameSnapshot,
    ) -> Future | _ProcessTaskHandle | None:
        raise NotImplementedError

    def apply_outcome(self, agent_id: str, update: AgentOutcomeUpdate) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError


class ThreadMultiAgentExecutor(MultiAgentExecutorBackend):
    def active_worker_count(self) -> int:
        return ThreadPoolManager.active_worker_count()

    def submit(
        self,
        agent_id: str,
        agent: Any,
        snapshot: AgentFrameSnapshot,
    ) -> Future | None:
        return ThreadPoolManager.submit_unique(
            f"agent:{agent_id}",
            agent.produce,
            snapshot,
        )

    def apply_outcome(self, agent_id: str, update: AgentOutcomeUpdate) -> None:
        return None

    def reset(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


class ProcessMultiAgentExecutor(MultiAgentExecutorBackend):
    def __init__(self, agents: list[Any]) -> None:
        self._lock = RLock()
        self._workers: dict[str, _ProcessAgentWorker] = {}
        for agent in agents:
            class_path = _agent_class_path(agent)
            if class_path is None:
                raise ValueError(f"agent {agent!r} is not importable for process execution")
            worker = _ProcessAgentWorker(str(agent.agent_id), class_path)
            worker.ensure_started()
            self._workers[str(agent.agent_id)] = worker

    def active_worker_count(self) -> int:
        with self._lock:
            total = 0
            for worker in self._workers.values():
                worker.drain_results()
                total += sum(1 for handle in worker._pending.values() if not handle._done)
            return total

    def submit(
        self,
        agent_id: str,
        agent: Any,
        snapshot: AgentFrameSnapshot,
    ) -> _ProcessTaskHandle | None:
        worker = self._workers[str(agent_id)]
        return worker.submit_produce(snapshot)

    def apply_outcome(self, agent_id: str, update: AgentOutcomeUpdate) -> None:
        worker = self._workers.get(str(agent_id))
        if worker is not None:
            worker.apply_outcome(update)

    def reset(self) -> None:
        for worker in self._workers.values():
            worker.reset()

    def shutdown(self) -> None:
        for worker in self._workers.values():
            worker.shutdown()


def _agent_class_path(agent: Any) -> str | None:
    module_name = getattr(agent.__class__, "__module__", "")
    qualname = getattr(agent.__class__, "__qualname__", "")
    if not module_name or not qualname or "<locals>" in qualname:
        return None
    try:
        resolved = _resolve_agent_class(f"{module_name}:{qualname}")
    except Exception:
        return None
    if resolved is not agent.__class__:
        return None
    return f"{module_name}:{qualname}"


def build_multi_agent_executor(config: Any, agents: list[Any]) -> MultiAgentExecutorBackend:
    requested = str(getattr(config, "multi_agent_executor", "process") or "process").strip().lower()
    if requested == "process":
        try:
            return ProcessMultiAgentExecutor(agents)
        except Exception as exc:
            logger.warning("Multi-agent process executor unavailable, falling back to threads: %s", exc)
    return ThreadMultiAgentExecutor()
