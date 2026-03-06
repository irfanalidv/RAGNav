from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from ..json_utils import extract_json
from ..llm.base import LLMClient


@dataclass(frozen=True)
class AgenticConfig:
    max_steps: int = 3
    max_tool_blocks: int = 8
    max_tool_chars: int = 14000


RetrieveRawFn = Callable[[str, int], list[dict]]


def agentic_retrieve_then_answer(
    *,
    query: str,
    llm: LLMClient,
    retrieve_raw: RetrieveRawFn,
    cfg: AgenticConfig = AgenticConfig(),
) -> str:
    """
    Minimal agentic loop:
    - model chooses retrieve vs answer (strict JSON)
    - retrieve returns JSON-serializable evidence blocks
    - model answers with evidence
    """
    memory: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are an agent that can call a retrieval tool.\n"
                "Always output STRICT JSON with one of:\n"
                '  {"action":"retrieve","query":"..."}\n'
                '  {"action":"answer","answer":"..."}\n'
                "Prefer retrieving evidence before answering."
            ),
        }
    ]

    for _ in range(cfg.max_steps):
        memory.append(
            {
                "role": "user",
                "content": (
                    f"User query: {query}\n\n"
                    "Tool: ragnav_retrieve_raw(query) -> list of evidence blocks.\n"
                    "Decide whether to retrieve or answer."
                ),
            }
        )
        raw = llm.chat(messages=memory, temperature=0)

        try:
            action = extract_json(raw)
        except Exception:
            action = {"action": "retrieve", "query": query}

        if isinstance(action, dict) and action.get("action") == "retrieve":
            q = str(action.get("query") or query)
            hits = retrieve_raw(q, cfg.max_tool_blocks)
            memory.append(
                {
                    "role": "user",
                    "content": "Tool result (ragnav_retrieve_raw):\n"
                    + json.dumps(hits, indent=2)[: cfg.max_tool_chars],
                }
            )
            continue

        if isinstance(action, dict) and action.get("action") == "answer":
            return str(action.get("answer") or "").strip()

    # fallback: force final answer
    memory.append(
        {"role": "user", "content": 'Return final answer JSON now: {"action":"answer","answer":"..."}'},
    )
    final_raw = llm.chat(messages=memory, temperature=0)
    final = extract_json(final_raw)
    return str(final.get("answer") if isinstance(final, dict) else final_raw).strip()

