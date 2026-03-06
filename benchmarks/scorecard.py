from __future__ import annotations

import io
import json
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class SuiteResult:
    name: str
    ok: bool
    elapsed_ms: int
    output: str
    json: dict[str, Any] | None = None
    error: str | None = None


def _run(name: str, fn: Callable[[], None]) -> SuiteResult:
    buf = io.StringIO()
    t0 = time.perf_counter()
    try:
        with redirect_stdout(buf):
            fn()
        ok = True
        err = None
    except Exception as e:
        ok = False
        err = f"{type(e).__name__}: {e}"
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    out = buf.getvalue().strip()

    # Try parse last JSON object printed by the suite.
    parsed: dict[str, Any] | None = None
    if out:
        # quick path: whole output is JSON
        try:
            v = json.loads(out)
            if isinstance(v, dict):
                parsed = v
        except Exception:
            # fallback: find last {...} region
            last_l = out.rfind("{")
            last_r = out.rfind("}")
            if last_l != -1 and last_r != -1 and last_r > last_l:
                snippet = out[last_l : last_r + 1]
                try:
                    v = json.loads(snippet)
                    if isinstance(v, dict):
                        parsed = v
                except Exception:
                    parsed = None

    return SuiteResult(name=name, ok=ok, elapsed_ms=elapsed_ms, output=out, json=parsed, error=err)


def main() -> None:
    """
    One-command offline health + capability summary.

    Run:
      python3 -m benchmarks.scorecard
    """
    from .offline_smoke import main as offline_smoke
    from .regression import main as regression
    from .paper_eval import main as paper_eval
    from .entity_eval import main as entity_eval
    from .entity_eval_excerpt import main as entity_eval_excerpt
    from .security_eval import main as security_eval
    from .citation_eval import main as citation_eval
    from .paper_pdf_suite import main as paper_pdf_suite

    suites = [
        ("offline_smoke", offline_smoke),
        ("regression", regression),
        ("paper_eval", paper_eval),
        ("entity_eval", entity_eval),
        ("entity_eval_excerpt", entity_eval_excerpt),
        ("security_eval", security_eval),
        ("citation_eval", citation_eval),
        ("paper_pdf_suite", paper_pdf_suite),
    ]

    results: list[SuiteResult] = []
    for name, fn in suites:
        results.append(_run(name, fn))

    overall_ok = all(r.ok for r in results)

    report = {
        "ok": overall_ok,
        "suites": [
            {
                "name": r.name,
                "ok": r.ok,
                "elapsed_ms": r.elapsed_ms,
                "error": r.error,
                "json": r.json,
                "output_tail": (r.output[-800:] if r.output else ""),
            }
            for r in results
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

