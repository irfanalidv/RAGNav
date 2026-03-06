from __future__ import annotations

from ragnav.ingest.markdown import ingest_markdown_string
from ragnav.env import load_env
from ragnav.llm.mistral import MistralClient
from ragnav.retrieval import RAGNavIndex, RAGNavRetriever


MARKDOWN = """
# DeepSeek-R1: Incentivizing Reasoning Capabilities

## Abstract
We introduce two reasoning models and describe their training.

## 1. Introduction
This section motivates the work.

### 1.1 Contributions
- A pure RL approach without cold-start data.
- A stronger model using cold-start + iterative RL.

## 5. Conclusion, Limitations, and Future Work
In this work, we share our journey in enhancing reasoning abilities through reinforcement learning.
DeepSeek-R1-Zero represents a pure RL approach without relying on cold-start data.
DeepSeek-R1 leverages cold-start data alongside iterative RL fine-tuning.
"""


def main() -> None:
    load_env()

    doc, blocks = ingest_markdown_string(MARKDOWN, name="deepseek.md")
    mistral = MistralClient()
    index = RAGNavIndex.build(documents=[doc], blocks=blocks, llm=mistral)
    retriever = RAGNavRetriever(index=index, llm=mistral)

    query = "What are the conclusions?"
    result = retriever.retrieve(query, k_final=5)

    print("== RAGNav hybrid retrieval (expanded context) ==")
    for b in result.blocks[:8]:
        title = " > ".join(b.heading_path) if b.heading_path else b.block_id
        print(f"- {title} ({b.anchors})")

    print("\n== PageIndex-style tree prompt baseline (raw) ==")
    baseline = retriever.tree_prompt_baseline(query)
    print(baseline["raw"])

    print("\n== Tree search (LLM navigates structure, then expands) ==")
    nav = retriever.tree_search_llm(query, max_nodes=120, max_steps=2, nodes_per_step=6)
    for b in nav.blocks[:10]:
        title = " > ".join(b.heading_path) if b.heading_path else b.block_id
        print(f"- {title} ({b.anchors})")

    print("\n== Hybrid tree search (candidates -> LLM pick) ==")
    hnav = retriever.hybrid_tree_search_llm(query, k_candidates=30, pick_k=6)
    for b in hnav.blocks[:10]:
        title = " > ".join(b.heading_path) if b.heading_path else b.block_id
        print(f"- {title} ({b.anchors})")


if __name__ == "__main__":
    main()

