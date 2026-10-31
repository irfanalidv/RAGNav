from __future__ import annotations

from pathlib import Path

from .exceptions import RAGNavIngestError


def download_bytes(url: str, *, out_path: str | Path | None = None, timeout_s: int = 60) -> bytes:
    """
    Download bytes from a URL (HTTP timeout ``timeout_s``) and optionally save a copy
    to ``out_path``. The file at ``out_path`` is always overwritten, never read back.

    This function lives in `ragnav` (not `examples/` or `pipelines/`) so the simplest
    end-user flows can share one implementation.
    """
    try:
        import requests
    except ImportError as e:
        raise RAGNavIngestError(
            'Missing optional dependency `requests`. Install with: pip install -e ".[pdf]"'
        ) from e

    resp = requests.get(url, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.content
    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return data


def download_pdf(url: str, *, out_path: str | Path | None = None, timeout_s: int = 60) -> bytes:
    """
    Convenience wrapper for PDF URLs (arXiv, etc.).
    """
    return download_bytes(url, out_path=out_path, timeout_s=timeout_s)
