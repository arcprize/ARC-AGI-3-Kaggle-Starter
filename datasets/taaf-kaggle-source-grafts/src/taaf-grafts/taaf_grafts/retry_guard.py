"""Bounded-retry + vLLM health-probe analyzer layer (design module 6).

The failure this closes: ``_HarnessGameSession.play`` retries a
``retryable_failure`` turn every 1s forever (``ANALYZER_RETRY_BACKOFF_SECONDS``,
solver.py:312-317) with no cap, so a dead local vLLM server burns every game's
full ``max_runtime_s_per_game`` at one useless request per second. A single
hung request can also eat the whole ``_timeout`` before it even becomes a
retryable turn.

``RetryGuard`` wraps the inner analyzer and is a *transparent pass-through on
every healthy turn* — it returns the inner ``AnalyzerTurnResult`` object
untouched — so it can ride alongside another shipping flag without confounding
that flag's score attribution. It acts only after ``failure_threshold``
consecutive retryable turns, and only once a health probe of
``{LOCAL_ANALYZER_BASE_URL}/models`` confirms the server is actually dead. When
dead it absorbs an exponential backoff (doubling to ``backoff_cap_s``) *inside*
``analyze`` before returning the same retryable result, so the play loop's 1s
cadence becomes a growing-but-bounded cadence instead.

Invariants (all failure-toward-stock, matching banking's reference standard):

- The inner ``analyze`` call is never wrapped in try/except: an inner crash
  must propagate exactly as stock (the game crashes), never be swallowed.
- All guard logic (streak/probe/backoff) is wrapped in a blanket try/except
  that returns the already-computed inner result on any error.
- Both the health probe's socket timeout and the backoff sleep are clamped to
  the time left in ``request_timeout_seconds`` (measured from ``analyze``
  entry, minus a small margin), so total ``analyze`` wall time per call stays
  <= the solver's per-request budget and the game ends
  ``gave_up``/``cancelled``, never ``crashed``. When no budget remains the
  probe is skipped entirely.
- ``should_stop`` is polled between backoff slices of <= 5s, so a stop request
  is honoured within one slice.
- ``generated_tokens`` / ``total_tokens`` / ``_timeout`` (and anything else the
  session duck-reads off the analyzer) proxy through to the inner agent, so
  token accounting and request-timeout mixing are byte-identical to unguarded.
"""

from __future__ import annotations

import os
import time
import urllib.request
from typing import Any

DEFAULT_FAILURE_THRESHOLD = 30
DEFAULT_PROBE_TIMEOUT_S = 5.0
DEFAULT_BACKOFF_CAP_S = 60.0

_BACKOFF_BASE_S = 1.0
_SHOULD_STOP_SLICE_S = 5.0
_TIMEOUT_MARGIN_S = 0.25
_MAX_EVENTS = 4096


class RetryGuard:
    """Analyzer chain layer: bounded retry + health-gated backoff.

    Construct via ``RetryGuard(inner_analyzer)``; drop-in for any object with
    the stock ``analyze(...)`` signature. Every unknown attribute proxies to
    ``inner`` so the session's duck-typed reads (token counters, ``_timeout``)
    pass straight through.
    """

    def __init__(
        self,
        inner: Any,
        *,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        probe_timeout_s: float = DEFAULT_PROBE_TIMEOUT_S,
        backoff_cap_s: float = DEFAULT_BACKOFF_CAP_S,
    ) -> None:
        self._inner = inner
        self._failure_threshold = max(1, int(failure_threshold))
        self._probe_timeout_s = max(0.1, float(probe_timeout_s))
        self._backoff_cap_s = max(_BACKOFF_BASE_S, float(backoff_cap_s))
        self._streak = 0
        self._backoff_s = 0.0
        self.events: list[tuple[Any, ...]] = []

    # -- observability (real attrs so they win over the inner proxy) ---------

    @property
    def consecutive_failures(self) -> int:
        return self._streak

    @property
    def backoff_seconds(self) -> float:
        return self._backoff_s

    # -- analyzer protocol ---------------------------------------------------

    def analyze(
        self,
        state_path: Any,
        action_num: int,
        *args: Any,
        request_timeout_seconds: float | None = None,
        should_stop: Any = None,
        **kwargs: Any,
    ) -> Any:
        started = time.monotonic()
        result = self._inner.analyze(
            state_path,
            action_num,
            *args,
            request_timeout_seconds=request_timeout_seconds,
            should_stop=should_stop,
            **kwargs,
        )
        try:
            self._govern(result, started, request_timeout_seconds, should_stop)
        except Exception:  # noqa: BLE001 — the guard must never break the turn
            pass
        return result

    # -- guard logic ---------------------------------------------------------

    def _govern(
        self,
        result: Any,
        started: float,
        request_timeout_seconds: float | None,
        should_stop: Any,
    ) -> None:
        if result is None or not getattr(result, "retryable_failure", False):
            self._streak = 0
            self._backoff_s = 0.0
            return
        self._streak += 1
        if self._streak < self._failure_threshold:
            return
        # The probe's socket timeout is spent inside analyze(), so it must fit
        # within the same per-request budget as the backoff — otherwise a slow
        # probe alone pushes total analyze wall past request_timeout_seconds.
        # With no budget left, skip the probe (assume dead) and let the
        # self-clamping backoff record the growing cadence at ~0 actual sleep,
        # never overshooting the deadline.
        budget = self._remaining_budget(started, request_timeout_seconds)
        if budget is None or budget > 0.0:
            alive = self._probe(budget)
            self._record(("probe", alive, self._streak))
            if alive:
                # Server is up: the stock 1s retry cadence is the correct
                # response (transient hiccup); pass the result through cleanly.
                return
        self._backoff(started, request_timeout_seconds, should_stop)

    @staticmethod
    def _remaining_budget(
        started: float, request_timeout_seconds: float | None
    ) -> float | None:
        """Seconds left in the per-request budget (minus the safety margin), or
        None when the session imposes no request timeout."""
        if request_timeout_seconds is None:
            return None
        try:
            return (
                float(request_timeout_seconds)
                - _TIMEOUT_MARGIN_S
                - (time.monotonic() - started)
            )
        except (TypeError, ValueError):
            return None

    def _probe(self, budget: float | None) -> bool:
        base = os.environ.get("LOCAL_ANALYZER_BASE_URL", "").strip().rstrip("/")
        if not base:
            return False
        url = f"{base}/models"
        timeout = self._probe_timeout_s
        if budget is not None:
            timeout = min(timeout, max(0.1, budget))
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                status = int(getattr(resp, "status", None) or resp.getcode())
                return 200 <= status < 300
        except Exception:  # noqa: BLE001 — any probe failure means "dead"
            return False

    def _backoff(
        self,
        started: float,
        request_timeout_seconds: float | None,
        should_stop: Any,
    ) -> None:
        self._backoff_s = (
            _BACKOFF_BASE_S
            if self._backoff_s <= 0.0
            else min(self._backoff_cap_s, self._backoff_s * 2.0)
        )
        target = self._backoff_s
        if request_timeout_seconds is not None:
            try:
                budget = (
                    float(request_timeout_seconds)
                    - _TIMEOUT_MARGIN_S
                    - (time.monotonic() - started)
                )
            except (TypeError, ValueError):
                budget = target
            target = min(target, max(0.0, budget))
        self._record(("backoff", self._backoff_s, target))

        end = time.monotonic() + target
        while True:
            if should_stop is not None:
                try:
                    if should_stop():
                        return
                except Exception:  # noqa: BLE001 — treat a broken predicate as stop
                    return
            remaining = end - time.monotonic()
            if remaining <= 0.0:
                return
            time.sleep(min(_SHOULD_STOP_SLICE_S, remaining))

    def _record(self, event: tuple[Any, ...]) -> None:
        self.events.append(event)
        if len(self.events) > _MAX_EVENTS:
            del self.events[: len(self.events) - _MAX_EVENTS]

    # -- transparent proxy to the inner agent --------------------------------

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal lookup misses (i.e. not a RetryGuard attr).
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        try:
            inner = self.__dict__["_inner"]
        except KeyError as exc:  # during __init__, before _inner is bound
            raise AttributeError(name) from exc
        return getattr(inner, name)
