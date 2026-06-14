from __future__ import annotations

import time

from ragnav.observability import Trace


def test_trace_span_accumulates_timing():
    tr = Trace()
    with tr.span("step"):
        time.sleep(0.001)
    assert tr.timings_ms["step"] > 0.0


def test_trace_incr_and_meta():
    tr = Trace()
    tr.incr("hits", 2)
    tr.set_meta("mode", "test")
    assert tr.counters["hits"] == 2
    assert tr.meta["mode"] == "test"
