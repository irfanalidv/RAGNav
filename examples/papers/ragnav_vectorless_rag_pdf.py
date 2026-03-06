def main() -> None:
    from ragnav.llm.mistral import MistralClient
    from ragnav.pipelines.vectorless import VectorlessRagConfig, vectorless_answer, vectorless_rag_pdf_url

    PDF_URL = "https://arxiv.org/pdf/2507.13334.pdf"
    query = "What are the evaluation methods used in this paper?"

    llm = MistralClient()
    retrieve_fn, _doc_id = vectorless_rag_pdf_url(
        pdf_url=PDF_URL,
        llm=llm,
        cfg=VectorlessRagConfig(max_pages=25, max_blocks=6, k_final=6),
    )
    ans = vectorless_answer(query=query, retrieve_fn=retrieve_fn, llm=llm)
    print(ans)


if __name__ == "__main__":
    main()

