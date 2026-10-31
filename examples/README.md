## Examples

The `examples/` folder is organized by use case.

- **`examples/offline/`**: runnable with `pip install ragnav` only (no API key, no downloads)
- **`examples/basic/`**: smallest end-to-end demo
- **`examples/papers/`**: paper-first / PDF workflows (page routing + cross-reference following)
- **`examples/multidoc/`**: multi-document routing (semantics / description / metadata)
- **`examples/agentic/`**: agentic retrieve-then-answer loops
- **`examples/graphs/`**: graph-aware retrieval over non-linear sources (chat/thread/link edges)

Run any example from the repo root, for example:

```bash
python3 examples/basic/ragnav_rag_simple.py
```

### Script index

Offline (no key, run in CI):

- `examples/offline/help_center_search.py`: heading-aware search over support articles, metadata filters, confidence (`--embeddings` for hybrid)
- `examples/offline/contract_clause_lookup.py`: numbered contract -> clause blocks with section provenance and neighbour expansion
- `examples/offline/incident_thread_qa.py`: Slack incident thread recovered via `reply_to` graph expansion
- `examples/offline/retrieval_eval_and_guardrails.py`: `EvalCase` / `score_retrieval` plus `SimpleInjectionPolicy`

Need `MISTRAL_API_KEY` and/or network:

- `examples/multidoc/ragnav_doc_search_semantics.py` (set `MISTRAL_API_KEY` if using Mistral)
- `examples/multidoc/ragnav_doc_search_description.py`
- `examples/multidoc/ragnav_doc_search_metadata.py`
- `examples/agentic/ragnav_agentic_retrieval.py`
- `examples/agentic/ragnav_agentic_retrieval_pdf.py`
- `examples/papers/ragnav_vectorless_rag_pdf.py`
- `examples/graphs/ragnav_chat_graph_retrieval.py`
