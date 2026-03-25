from __future__ import annotations

from ragnav.ingest.markdown import ingest_markdown_string
from ragnav.env import load_env
from ragnav.llm.mistral import MistralClient
from ragnav.retrieval import RAGNavIndex, RAGNavRetriever
from ragnav.display import print_tree, print_wrapped


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
    # Step 0: preparation (mirrors PageIndex cookbooks)
    load_env()

    mistral = MistralClient()

    # Step 1: “structure” generation (RAGNav ingests into a block graph)
    doc, blocks = ingest_markdown_string(MARKDOWN, name="deepseek.md")
    print("## Step 1: Ingested structure\n")
    print_tree(blocks)

    # Step 2: Retrieval
    index = RAGNavIndex.build(documents=[doc], blocks=blocks, llm=mistral)
    retriever = RAGNavRetriever(index=index, llm=mistral)

    query = "What are the conclusions in this document?"
    print("\n## Step 2: Retrieval\n")

    hybrid = retriever.retrieve(query, k_final=5)
    print("### 2.1 RAGNav hybrid retrieval (BM25 + embeddings + structure expansion)")
    for b in hybrid.blocks[:8]:
        title = " > ".join(b.heading_path) if b.heading_path else b.block_id
        print(f"- {title}  {b.anchors}")

    print("\n### 2.2 PageIndex-style baseline (LLM navigates a tree prompt)")
    baseline = retriever.tree_prompt_baseline(query)
    print_wrapped(baseline["raw"])

    # Step 3: Answer generation (minimal)
    # This is intentionally simple for parity with PageIndex "minimal example".
    context = "\n\n".join(b.text for b in hybrid.blocks[:6])
    answer_prompt = f"""Answer the question using ONLY the provided context.
If you cite something, mention the section heading when possible.

Question: {query}

Context:
{context}
"""
    ans = mistral.chat(messages=[{"role": "user", "content": answer_prompt}], temperature=0)
    print("\n## Step 3: Answer generation\n")
    print_wrapped(ans)


if __name__ == "__main__":
    main()

