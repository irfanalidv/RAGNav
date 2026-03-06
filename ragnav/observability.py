from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Trace:
    """
    Lightweight trace collector for retrieval/index builds.
    Stores timings + arbitrary counters/metadata.
    """

    timings_ms: dict[str, float] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def incr(self, key: str, n: int = 1) -> None:
        self.counters[key] = int(self.counters.get(key, 0)) + int(n)

    def set_meta(self, key: str, value: Any) -> None:
        self.meta[key] = value

    def span(self, name: str):
        return _Span(self, name)


class _Span:
    def __init__(self, trace: Trace, name: str):
        self._trace = trace
        self._name = name
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt_ms = (time.perf_counter() - self._t0) * 1000.0
        self._trace.timings_ms[self._name] = float(self._trace.timings_ms.get(self._name, 0.0) + dt_ms)
        return False

