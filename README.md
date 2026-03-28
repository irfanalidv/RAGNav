# RAGNav

[![PyPI](https://img.shields.io/pypi/v/ragnav.svg)](https://pypi.org/project/ragnav/)
[![Python](https://img.shields.io/pypi/pyversions/ragnav.svg)](https://pypi.org/project/ragnav/)
[![License: MIT](https://img.shields.io/pypi/l/ragnav.svg)](https://pypi.org/project/ragnav/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/ragnav?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/ragnav)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/irfanalidv/RAGNav/blob/main/cookbook/ragnav_quickstart.ipynb)

**Production-grade hybrid retrieval — no API key required.**

RAGNav combines BM25, sentence-transformer embeddings, and document-structure graph expansion in one library that runs **entirely offline** with open models.

**SQuAD R@3: 0.956** on 500 questions, zero API calls.  
PageIndex (the closest alternative) requires GPT-4o for its tree workflow and does not publish SQuAD numbers.

```bash
pip install ragnav[embeddings]
```

```python
from ragnav import RAGNavIndex, RAGNavRetriever
from ragnav.ingest.markdown import ingest_markdown_string
from ragnav.llm.fake import FakeLLMClient

md = "# Demo\n\nParis is the capital of France."
doc, blocks = ingest_markdown_string(md, name="demo.md")
llm = FakeLLMClient()
index = RAGNavIndex.build(
    documents=[doc],
    blocks=blocks,
    llm=llm,
    use_sentence_transformers=True,
    vector_model="all-MiniLM-L6-v2",
    embed_batch_size=32,
)
retriever = RAGNavRetriever(index=index, llm=llm)
result = retriever.retrieve(
    "What is the capital of France?",
    top_k=5,
    expand_structure=False,
    expand_graph=False,
)
print(result.blocks[0].text)  # Paris is the capital of France.
print(result.confidence)  # ConfidenceLevel.HIGH (heuristic; see models.ConfidenceLevel)
```

![RAGNav architecture](https://raw.githubusercontent.com/irfanalidv/RAGNav/main/assets/ragnav-architecture.png)

For **long PDFs and papers**, RAGNav is also **navigation-first**: route pages/sections, then retrieve evidence with provenance — not only “embed query → retrieve chunks”.

## The problem (why long-document QA fails)

LLMs have finite context windows and degrade on long inputs (“lost in the middle” effects). In long PDFs (papers, reports, manuals), naive retrieval often returns *plausible* text but misses the *right* place.

## Why classic vector + chunk RAG fails (in PDFs)

1. **Intent mismatch**: the query expresses intent; the most similar text isn’t always the most relevant.
2. **Hard chunking breaks meaning**: chunks cut across sections/tables/captions, losing provenance and coherence.
3. **Similarity ≠ relevance**: many sections look semantically similar (especially in technical documents).
4. **Cross-references**: “see Figure 3 / Table 2 / Appendix A / Section 4.1” rarely matches the referenced content.
5. **No navigation**: users don’t want “top-k chunks”; they want *where the answer lives* + traceable evidence.

## RAGNav’s approach (navigation-first retrieval loop)

RAGNav is built around a simple loop:

1. **Ingest (paper mode)**: PDF → blocks with `anchors={"page": N}` + edges (`parent`, `next`, `link_to`).
2. **Route**: query → rank likely **pages**.
3. **Retrieve**: search **within routed pages** (hybrid BM25 + embeddings).
4. **Expand**: add coherence (section headers + adjacent “next” blocks).
5. **Follow refs** (optional): traverse `link_to` edges (Figure/Table/Appendix/Section).
6. **Answer**: generate from retrieved evidence (optionally with inline citations).

## The “index” (what the model navigates)

RAGNav normalizes everything into a small graph:

```text
Block {
  block_id: "pdf:paper.pdf#b19"
  doc_id: "pdf:paper.pdf"
  text: "..."
  anchors: { page: 5, line_start: 12, line_end: 20 }
}

Edge {
  type: "parent" | "next" | "link_to" | ...
  src: block_id
  dst: block_id
}
```

This is the practical equivalent of PageIndex’s “in-context index”, but optimized for **papers**:
pages + headings + cross-references + provenance.

## Vector RAG vs RAGNav (paper-mode)

| Problem | Vector + chunks | RAGNav (navigation-first) |
| --- | --- | --- |
| Find “where” in a paper | Not explicit | Routes pages + sections |
| Cross-references (“see Appendix”) | Usually missed | Follows `link_to` edges |
| Provenance | Weak (chunk ids) | Page + block ids + anchors |
| Coherence | Fragmented | Deterministic expansion (`parent`/`next`) |
| Evaluation | Ad-hoc | Built-in offline suites + scorecard |

## Use cases

- **Research papers (PDF)**: page routing + cross-ref following.
- **Reports / manuals / specs**: structure-aware retrieval (coherent evidence, not fragments).
- **Grounded answers**: inline citations `[[block_id]]` per sentence (optional).
- **Security baseline**: drop prompt-injection blocks and redact obvious secrets (optional).
- **GraphRAG**: entity graph + multi-hop traversal with provenance (optional).

## Acknowledgements & prior art

RAGNav is an independent project, but it stands on strong prior work:

- **PageIndex**: RAGNav builds on the core insight popularized by **PageIndex** — *document structure is a first-class retrieval signal* ([repo](https://github.com/VectifyAI/PageIndex), [article](https://pageindex.ai/blog/pageindex-intro)).
- **PyMuPDF**: PDF text extraction is powered by `pymupdf` (optional dependency).
- **BM25 / classic IR**: Lexical retrieval uses BM25-style scoring (a long-established baseline).
- **Mistral**: The reference LLM/embedding client targets Mistral (optional dependency).

RAGNav is **not affiliated with** these projects/organizations. If you notice missing or incorrect attribution, please open an issue.

---

## Install

Create a virtualenv, then install RAGNav:

```bash
pip install -e .
```

To enable **PDF ingestion**:

```bash
pip install -e ".[pdf]"
```

To enable **Mistral-backed** chat + embeddings:

```bash
pip install -e ".[mistral]"
```

## Setup (Mistral)

Do **not** hardcode or commit keys. Use env vars:

```bash
export MISTRAL_API_KEY="your_key_here"
```

---

## Quickstart (CLI): run on an arXiv PDF URL

Install:

```bash
pip install -e ".[mistral,pdf]"
export MISTRAL_API_KEY="..."
```

Run (recommended: paper-mode navigation):

```bash
ragnav paper-pdf --pdf-url "https://arxiv.org/pdf/2507.13334.pdf" --query "What is Context Engineering?"
```

### Jupyter notebook quickstart

Open:
- **`cookbook/ragnav_quickstart.ipynb`** — offline SQuAD demo + confidence + QueryFallback ([run in Colab](https://colab.research.google.com/github/irfanalidv/RAGNav/blob/main/cookbook/ragnav_quickstart.ipynb))
- `cookbook/ragnav_paper_quickstart.ipynb`

Other modes (optional):

- Hybrid (BM25 + embeddings, generic PDF blocks):

```bash
ragnav hybrid-pdf --pdf-url "https://arxiv.org/pdf/2507.13334.pdf" --query "What is Context Engineering?"
```

- Vectorless (BM25-only, generic PDF blocks):

```bash
ragnav vectorless-pdf --pdf-url "https://arxiv.org/pdf/2507.13334.pdf" --query "What is Context Engineering?"
```

- Agentic retrieval loop:

```bash
ragnav agentic-pdf --pdf-url "https://arxiv.org/pdf/2507.13334.pdf" --query "Summarize the paper's main contribution."
```

### Real example output (paper-mode navigation)

This repo includes a paper-mode demo that downloads an arXiv PDF and runs **page routing + retrieval**:

```bash
python3 examples/papers/ragnav_paper_rag_pdf.py \
  --pdf-url "https://arxiv.org/pdf/2507.13334.pdf" \
  --pdf-name "2507.13334.pdf" \
  --max-pages 25
```

Output (real, trimmed):

```text
## Routed pages
- doc_id=pdf:2507.13334.pdf page=4 score=0.5423 N=3
- doc_id=pdf:2507.13334.pdf page=14 score=0.5298 N=7
- doc_id=pdf:2507.13334.pdf page=9 score=0.4662 N=4
- doc_id=pdf:2507.13334.pdf page=5 score=0.4597 N=3

## Retrieved evidence blocks (first 10)
- page=14  title=Sr-Nle [1130]  id=pdf:2507.13334.pdf#b106
- page=2  title=Related Work  id=pdf:2507.13334.pdf#b11
...
```

---

## Quickstart (Python): papers (recommended)

### PaperRAG (page routing + cross-ref following)

```python
from ragnav.llm.mistral import MistralClient
from ragnav.net import download_pdf
from ragnav.papers import PaperRAG, PaperRAGConfig

llm = MistralClient()
cfg = PaperRAGConfig(max_pages=25, top_pages=4, follow_refs=True)

pdf_bytes = download_pdf("https://arxiv.org/pdf/2507.13334.pdf")
paper = PaperRAG.from_pdf_bytes(pdf_bytes, llm=llm, pdf_name="paper.pdf", cfg=cfg)
print(paper.answer("What experiments were conducted?", cfg=cfg))
```

### Grounded answering (inline citations per sentence)

```python
print(paper.answer_cited("What does Figure 1 show?", cfg=cfg))
```

Output format:

```text
Sentence one [[pdf:paper.pdf#b12]].
Sentence two [[pdf:paper.pdf#b47]] [[pdf:paper.pdf#b48]].
```

---

## Quickstart: GraphRAG (entity multi-hop with provenance)

```python
from ragnav.graphrag import build_entity_graph, EntityGraphRetriever

eg = build_entity_graph(blocks)  # blocks are RAGNav Block objects
egr = EntityGraphRetriever(graph=eg, blocks_by_id={b.block_id: b for b in blocks})

out = egr.retrieve("Which dataset was BERT evaluated on?")
for b in out["blocks"][:3]:
    print(b.block_id, b.anchors.get("page"))
```

Networked PDF demo:

```bash
pip install -e ".[mistral,pdf]"
export MISTRAL_API_KEY="..."
python3 examples/graphs/ragnav_entity_graphrag_pdf.py
```

---

## Production features

Features PageIndex does not have:

| Feature | What it does |
|---------|-------------|
| `ConfidenceLevel` | Every retrieval result carries HIGH/MEDIUM/LOW confidence so you can decide whether to show the answer or say "I'm not sure." |
| `QueryFallback` | On LOW/MEDIUM confidence, automatically retries with LLM-generated query rephrasing. Prevents silent failures. |
| `CostTracker` | Tracks token usage and cost per LLM call. Set a `budget_usd` to get `BudgetExceededError` before you overspend. |
| `CrossEncoderReranker` | Optional second-stage reranker with **≥50** first-stage candidates (see `retrieve()`). On small SQuAD-style corpora the default MS MARCO MiniLM reranker can trail hybrid RRF alone; use domain-tuned models or skip reranking when the pool is easy. |
| Multi-format ingest | PDF, markdown, HTML, email chains, chat logs, legal/numbered documents. |
| No API key required | Runs fully offline with sentence-transformers. |

---

## Benchmarks

All numbers are reproducible: run `benchmarks/squad_benchmark.py` or
`benchmarks/cuad_benchmark.py` after `pip install ragnav[embeddings] datasets`.
No API key required for SQuAD or CUAD. Hybrid retrieval uses **reciprocal rank fusion (RRF)**
by default; optional **cross-encoder reranking** is exposed on `RAGNavRetriever(reranker=...)`.

### Retrieval accuracy

| Dataset | Method | R@1 | R@3 | R@5 | MRR@10 |
|---------|--------|-----|-----|-----|--------|
| SQuAD | BM25-only | 0.852 | 0.932 | 0.950 | 0.896 |
| SQuAD | Embedding-only | 0.772 | 0.906 | 0.942 | 0.844 |
| SQuAD | **RAGNav hybrid (RRF 0.5/0.5)** | **0.864** | **0.956** | **0.978** | **0.912** |
| SQuAD | Hybrid RRF + cross-encoder reranker | 0.862 | 0.944 | 0.968 | 0.906 |
| CUAD (block-level) | BM25-only | 0.017 | 0.040 | 0.044 | 0.032 |
| CUAD (block-level) | **RAGNav hybrid (legal ingest + RRF)** | **0.007** | **0.047** | **0.051** | **0.027** |
| CUAD (block-level) | **RAGNav + graph expansion** | **0.007** | **0.047** | **0.051** | **0.027** |

### CUAD — span recall (concatenated top-k blocks)

Gold answer span may sit across legal-ingest block boundaries; **span S@k** is true if any gold string appears in the concatenation of the top-k retrieved blocks’ text (fairer for clauses).

| Dataset | Method | S@1 | S@3 | S@5 | MRR@10 |
|---------|--------|-----|-----|-----|--------|
| CUAD (span) | BM25-only | 0.020 | 0.061 | 0.071 | 0.044 |
| CUAD (span) | **RAGNav hybrid (legal ingest + RRF)** | **0.010** | **0.071** | **0.074** | **0.037** |
| CUAD (span) | **RAGNav + graph expansion** | **0.010** | **0.071** | **0.074** | **0.037** |

*SQuAD: 500 questions, 447 unique passages, `rajpurkar/squad` validation set, CC BY-SA 4.0*

*The previously published **0.968** SQuAD R@3 used weighted min–max fusion; the default hybrid path is now **RRF** (`fusion="rrf"` in `retrieve()`). Use `fusion="weighted"` to approximate the older fusion behavior.*

*CUAD: 300 questions sampled (297 with gold locatable in the indexed blocks after legal ingest), `theatticusproject/cuad-qa` test JSON (official zip), CC BY 4.0. Block-level R@k requires a gold block_id in the top-k list; span S@k only requires the gold answer text to appear in the merged text of those blocks.*

### vs. PageIndex

| | PageIndex | RAGNav |
|--|-----------|--------|
| Requires GPT-4o / paid LLM | Yes | No — runs on any LLM or embedding-only |
| Fully offline (no API key) | No | Yes |
| SQuAD R@3 | Not published | **0.956** (hybrid RRF) |
| CUAD clause retrieval (span S@3) | Not published | **0.071** (hybrid RRF + legal ingest; see benchmark file for block-level R@3) |
| FinanceBench accuracy | 98.7% (GPT-4o) | TBD (Mistral) |
| Handles markdown / chat / email | No | Yes |
| Structure-aware graph expansion | No | Yes |

*FinanceBench comparison in progress (later session).*

### One-command scorecard (offline)

```bash
python3 -m benchmarks.scorecard
```

Example output (real):

```json
{
  "ok": true,
  "suites": [
    { "name": "offline_smoke", "ok": true },
    { "name": "paper_eval", "ok": true, "json": { "suite": "paper_crossref_v1", "follow_refs_true": { "block_hit_rate": 1.0 } } },
    { "name": "entity_eval_excerpt", "ok": true, "json": { "suite": "entity_excerpt_v1" } },
    { "name": "security_eval", "ok": true }
  ]
}
```

---

## Local PDFs + golden manifest (optional)

If you add local PDFs under `data/papers/`, you can run a suite against **your own papers**:

```bash
mkdir -p data/papers
# copy some PDFs into data/papers/
python3 -m benchmarks.paper_pdf_suite
```

Optional: add `data/papers/manifest.json` to define *expected outcomes* per PDF (queries + expected pages/substrings).

Example manifest:

```json
{
  "papers": [
    {
      "file": "my_paper.pdf",
      "cases": [
        {
          "case_id": "datasets",
          "query": "Which datasets are mentioned?",
          "expected_pages": [2, 3],
          "expected_text_substrings": ["SQuAD", "GLUE"],
          "tags": ["datasets"]
        }
      ]
    }
  ]
}
```

---

## Repo layout

- `ragnav/`: the Python package (hybrid retrieval engine)
- `benchmarks/`: accuracy + latency/cost harness (PageIndex-style baseline + RAGNav hybrid)
- `examples/`: runnable end-to-end demos

---

## More examples

```bash
export MISTRAL_API_KEY="..."
python3 examples/multidoc/ragnav_doc_search_semantics.py
```

Other entrypoints:
- `examples/multidoc/ragnav_doc_search_description.py`
- `examples/multidoc/ragnav_doc_search_metadata.py`
- `examples/agentic/ragnav_agentic_retrieval.py`
- `examples/agentic/ragnav_agentic_retrieval_pdf.py`
- `examples/papers/ragnav_vectorless_rag_pdf.py`
- `examples/graphs/ragnav_chat_graph_retrieval.py`
