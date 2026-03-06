from __future__ import annotations

from pathlib import Path

import requests

from ragnav.env import load_env
from ragnav.ingest.pdf import ingest_pdf_bytes
from ragnav.ingest.pdf import PdfIngestOptions
from ragnav.llm.mistral import MistralClient
from ragnav.retrieval import RAGNavIndex, RAGNavRetriever
from ragnav.utils import print_wrapped


PDF_URL = "https://arxiv.org/pdf/2507.13334.pdf"


def download_pdf(url: str, out_path: Path) -> bytes:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    return resp.content


def agentic_answer(query: str, retriever: RAGNavRetriever, llm: MistralClient) -> str:
    # Delegate to the existing minimal agent loop behavior:
    # retrieve_raw() supplies citeable raw text, then the model answers.
    import json

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

    for _ in range(3):
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

        from ragnav.json_utils import extract_json

        try:
            action = extract_json(raw)
        except Exception:
            action = {"action": "retrieve", "query": query}

        if isinstance(action, dict) and action.get("action") == "retrieve":
            q = str(action.get("query") or query)
            hits = retriever.retrieve_raw(q, max_blocks=8)
            memory.append(
                {
                    "role": "user",
                    "content": "Tool result (ragnav_retrieve_raw):\n"
                    + json.dumps(hits, indent=2)[:14000],
                }
            )
            continue

        if isinstance(action, dict) and action.get("action") == "answer":
            return str(action.get("answer") or "").strip()

    # fallback: answer with last pass
    memory.append(
        {
            "role": "user",
            "content": 'Return final answer JSON now: {"action":"answer","answer":"..."}',
        },
    )
    final_raw = llm.chat(messages=memory, temperature=0)
    from ragnav.json_utils import extract_json

    final = extract_json(final_raw)
    return str(final.get("answer") if isinstance(final, dict) else final_raw).strip()


def main() -> None:
    load_env()

    llm = MistralClient()

    data_dir = Path("data")
    pdf_bytes = download_pdf(PDF_URL, data_dir / "2507.13334.pdf")

    # Cap pages for cost/latency; increase if you want full-doc indexing.
    doc, blocks = ingest_pdf_bytes(
        pdf_bytes,
        name="2507.13334.pdf",
        metadata={"source": "arxiv", "url": PDF_URL},
        opts=PdfIngestOptions(max_pages=20),
    )
    index = RAGNavIndex.build(documents=[doc], blocks=blocks, llm=llm)
    retriever = RAGNavRetriever(index=index, llm=llm)

    query = "What are the evaluation methods used in this paper?"
    answer = agentic_answer(query, retriever, llm)

    print("\n## Answer\n")
    print_wrapped(answer)


if __name__ == "__main__":
    main()
