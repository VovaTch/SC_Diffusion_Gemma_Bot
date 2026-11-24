"""
Priority-ordered serialization gate for local vLLM requests.

vLLM throughput for DiffusionGemma plateaus at a single stream (~130 tok/s
aggregate across 4-16 concurrent requests). Serializing per HTTP call with
priority lets S1 preempt pending S2 steps (e.g. between S2's tool-calling
step and its structured directive step) without starving either loop.
"""

from __future__ import annotations

import heapq
import itertools
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from starcraft_diffusion_gemma.llm.client import JsonDict, Transport


class PriorityGate:
    """Thread-safe single-slot lock that wakes waiters in priority order."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._busy = False
        self._waiters: list[tuple[int, int]] = []
        self._seq = itertools.count()

    @contextmanager
    def acquire(self, priority: int) -> Iterator[None]:
        """
        Block until this caller has exclusive access (lower ``priority`` first).

        Args:
            priority (int): Scheduling priority (0 = highest, e.g. S1; 1 = S2).
        """
        ticket = (priority, next(self._seq))
        with self._cond:
            heapq.heappush(self._waiters, ticket)
            while self._busy or self._waiters[0] != ticket:
                self._cond.wait()
            heapq.heappop(self._waiters)
            self._busy = True
        try:
            yield
        finally:
            with self._cond:
                self._busy = False
                self._cond.notify_all()


class GatedTransport:
    """``Transport`` wrapper that serializes ``post()`` calls through a ``PriorityGate``."""

    def __init__(self, inner: Transport, gate: PriorityGate, priority: int) -> None:
        self._inner = inner
        self._gate = gate
        self._priority = priority

    def post(self, path: str, body: JsonDict) -> JsonDict:
        """POST via the wrapped transport once the priority gate grants the slot."""
        with self._gate.acquire(self._priority):
            return self._inner.post(path, body)
