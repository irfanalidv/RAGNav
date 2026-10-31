"""MistralClient wiring with the SDK mocked (no network)."""

import sys
import types
from types import SimpleNamespace

import pytest

from ragnav.cost import CostTracker
from ragnav.exceptions import RAGNavLLMError
from ragnav.llm.mistral import MistralClient


class _FakeMistral:
    def __init__(self, api_key):
        self.api_key = api_key
        self.chat = SimpleNamespace(complete=self._complete)
        self.embeddings = SimpleNamespace(create=self._embed)

    def _complete(self, *, model, messages, temperature):
        msg = SimpleNamespace(content="  hello there  ")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    def _embed(self, *, model, inputs):
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2]) for _ in inputs])


@pytest.fixture
def fake_sdk(monkeypatch):
    """Install a fake `mistralai.client` module (mistralai >= 2 layout)."""
    pkg = types.ModuleType("mistralai")
    pkg.__path__ = []
    client_mod = types.ModuleType("mistralai.client")
    client_mod.Mistral = _FakeMistral
    monkeypatch.setitem(sys.modules, "mistralai", pkg)
    monkeypatch.setitem(sys.modules, "mistralai.client", client_mod)
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")


def test_chat_and_embed_with_mocked_sdk(fake_sdk):
    tracker = CostTracker()
    client = MistralClient(cost_tracker=tracker)
    assert client.chat(messages=[{"role": "user", "content": "hi"}]) == "hello there"
    assert client.embed(inputs=["a", "b"]) == [[0.1, 0.2], [0.1, 0.2]]
    assert tracker.report().calls == 1


def test_legacy_top_level_import(monkeypatch):
    pkg = types.ModuleType("mistralai")
    pkg.Mistral = _FakeMistral
    monkeypatch.setitem(sys.modules, "mistralai", pkg)
    monkeypatch.setitem(sys.modules, "mistralai.client", None)
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    assert isinstance(MistralClient()._client, _FakeMistral)


def test_missing_api_key_raises(fake_sdk, monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "dotenv", None)
    with pytest.raises(RAGNavLLMError):
        MistralClient()
