from __future__ import annotations

import json
from typing import Any

from ragnav.ingest.markdown import ingest_markdown_string
from ragnav.env import load_env
from ragnav.json_utils import extract_json
from ragnav.llm.mistral import MistralClient
from ragnav.retrieval import RAGNavIndex, RAGNavRetriever
from ragnav.display import print_wrapped


DOC_SURVEY = """
# Context Engineering Survey (Excerpt)

## 6. Evaluation
### 6.1 Evaluation Frameworks and Methodologies
Component-level assessment includes prompt evaluation, long-context evaluation (needle-in-a-haystack),
self-refinement, and structured/relational data integration.

### 6.2 Benchmarks
Benchmarks include LongMemEval for memory, BFCL/T-Eval for tool use, and WebArena for web agents.
"""

DOC_FINANCE = """
# ACME Corp Q1 FY25 Earnings (Excerpt)

## EBITDA Adjustments
We adjusted EBITDA for stock-based compensation and restructuring costs.
"""


def agentic_retrieve_answer(query: str, retriever: RAGNavRetriever, llm: MistralClient) -> str:
    """
    Minimal agent loop:
    - Ask LLM whether to retrieve, or answer.
    - If retrieve: call retriever.retrieve_raw and feed results back.
    """
    memory: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are an agent that can call a retrieval tool.\n"
                "When you need more information, request retrieval.\n"
                "Always output STRICT JSON with one of:\n"
                '  {"action":"retrieve","query":"..."}\n'
                '  {"action":"answer","answer":"..."}\n'
            ),
        }
    ]

    tool_schema = {
        "name": "ragnav_retrieve_raw",
        "input": {"query": "string"},
        "output": [
            {
                "doc_id": "string",
                "block_id": "string",
                "title": "string|null",
                "anchors": "object",
                "content": "string",
            }
        ],
    }

    for _ in range(3):
        prompt = f"""
User query: {query}

Tool available:
{json.dumps(tool_schema, indent=2)}

Decide next step. If you can answer confidently, answer.
If you need supporting evidence, retrieve.
""".strip()

        memory.append({"role": "user", "content": prompt})
        raw = llm.chat(messages=memory, temperature=0)

        try:
            action = extract_json(raw)
        except Exception:
            # If model didn't comply, force a single retrieval then answer.
            action = {"action": "retrieve", "query": query}

        if isinstance(action, dict) and action.get("action") == "retrieve":
            q = str(action.get("query") or query)
            hits = retriever.retrieve_raw(q, max_blocks=6)
            memory.append(
                {
                    "role": "user",
                    "content": (
                        "Tool result (ragnav_retrieve_raw):\n"
                        f"{json.dumps(hits, indent=2)[:12000]}"
                    ),
                }
            )
            continue

        if isinstance(action, dict) and action.get("action") == "answer":
            return str(action.get("answer") or "").strip()

        # Unexpected output → answer with a final pass
        memory.append(
            {
                "role": "user",
                "content": "Return the final answer now as JSON: {\"action\":\"answer\",\"answer\":\"...\"}",
            }
        )
        final_raw = llm.chat(messages=memory, temperature=0)
        final = extract_json(final_raw)
        return str(final.get("answer") if isinstance(final, dict) else final_raw).strip()

    # If we hit max steps, ask for an answer from accumulated context
    memory.append(
        {
            "role": "user",
            "content": "Return the final answer now as JSON: {\"action\":\"answer\",\"answer\":\"...\"}",
        }
    )
    final_raw = llm.chat(messages=memory, temperature=0)
    try:
        final = extract_json(final_raw)
        return str(final.get("answer") if isinstance(final, dict) else final_raw).strip()
    except Exception:
        return final_raw.strip()


def main() -> None:
    load_env()

    llm = MistralClient()

    docs = []
    blocks = []
    for name, content, meta in [
        ("survey.md", DOC_SURVEY, {"type": "paper", "topic": "eval"}),
        ("finance.md", DOC_FINANCE, {"type": "finance", "topic": "earnings"}),
    ]:
        d, b = ingest_markdown_string(content, name=name, metadata=meta)
        docs.append(d)
        blocks.extend(b)

    index = RAGNavIndex.build(documents=docs, blocks=blocks, llm=llm)
    retriever = RAGNavRetriever(index=index, llm=llm)

    query = "What evaluation methods are described in the survey?"
    answer = agentic_retrieve_answer(query, retriever, llm)

    print("## Answer\n")
    print_wrapped(answer)


if __name__ == "__main__":
    main()

