from __future__ import annotations

import json

from ragnav.llm.fake import FakeLLMClient
from ragnav.pipelines.agentic import AgenticConfig, agentic_retrieve_then_answer


def test_agentic_retrieve_then_answer_loops_until_answer():
    hits = [{"block_id": "b1", "content": "Paris is the capital of France."}]
    calls = {"n": 0}

    def retrieve_raw(query: str, max_blocks: int):
        calls["n"] += 1
        return hits[:max_blocks]

    class AgentLLM(FakeLLMClient):
        def chat(self, *, messages, model=None, temperature=0):
            if calls["n"] == 0:
                return json.dumps({"action": "retrieve", "query": "capital of France"})
            return json.dumps({"action": "answer", "answer": "Paris is the capital of France."})

    answer = agentic_retrieve_then_answer(
        query="What is the capital of France?",
        llm=AgentLLM(),
        retrieve_raw=retrieve_raw,
        cfg=AgenticConfig(max_steps=3),
    )
    assert "Paris" in answer
    assert calls["n"] == 1


def test_agentic_falls_back_when_llm_returns_unparseable_json():
    hits = [{"content": "Berlin is the capital of Germany."}]

    class BrokenThenAnswer(FakeLLMClient):
        def __init__(self):
            super().__init__()
            self._step = 0

        def chat(self, *, messages, model=None, temperature=0):
            self._step += 1
            if self._step == 1:
                return "not json"
            return json.dumps({"action": "answer", "answer": "Berlin"})

    answer = agentic_retrieve_then_answer(
        query="capital of Germany?",
        llm=BrokenThenAnswer(),
        retrieve_raw=lambda q, k: hits,
        cfg=AgenticConfig(max_steps=2),
    )
    assert "Berlin" in answer
