from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ragnav.exceptions import RAGNavIngestError
from ragnav.net import download_bytes, download_pdf


def test_download_bytes_writes_file(tmp_path):
    mock_resp = MagicMock()
    mock_resp.content = b"pdf-bytes"
    mock_resp.raise_for_status = MagicMock()
    with patch("requests.get", return_value=mock_resp) as get:
        out = tmp_path / "doc.pdf"
        data = download_bytes("https://example.com/doc.pdf", out_path=out)
    assert data == b"pdf-bytes"
    assert out.read_bytes() == b"pdf-bytes"
    get.assert_called_once()


def test_download_bytes_raises_ingest_error_without_requests(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "requests":
            raise ImportError("no requests")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RAGNavIngestError, match="requests"):
        download_pdf("https://example.com/x.pdf")
