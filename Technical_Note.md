# Technical Note

## 1. Architecture

DocuMind follows a Retrieval-Augmented Generation (RAG) architecture designed to keep answers grounded in the uploaded PDF.

The system consists of the following pipeline:

1. **PDF Ingestion**
   - Text is extracted page-wise using PyMuPDF with fallback support.
   - Spacing and formatting artifacts are repaired to improve readability.

2. **Structure-Aware Chunking**
   - Text is split into meaningful chunks using headings, lists, and paragraph boundaries.
   - Each chunk stores metadata such as page number, section title, and a contextual window.

3. **Embedding and Storage**
   - Chunks are converted into vector embeddings using Sentence Transformers.
   - Embeddings and metadata are stored in ChromaDB for efficient retrieval.

4. **Hybrid Retrieval**
   - Combines semantic search with keyword-based ranking.
   - Reranking uses query intent, phrase matching, section cues, and evidence quality.

5. **Guardrails**
   - Filters low-quality, irrelevant, noisy, or unsupported chunks.
   - Ensures only evidence-supported context is passed to the generator.

6. **Answer Generation**
   - Uses strict instructions to enforce document grounding.
   - Uses OpenAI generation when available.
   - Falls back to local seq2seq or extractive methods.
   - Refuses questions when the answer is not supported by the retrieved context.

7. **Citation Selection**
   - Selects citation pages based on overlap between the answer, query, and retrieved evidence.

## 2. Key Design Decisions

**a. Strict Grounding via Prompt and Code**

- Grounding is enforced through both system instructions and programmatic checks.
- This reduces hallucination and improves refusal behavior.

**b. Hybrid Retrieval instead of Pure Vector Search**

- Pure vector search was not reliable enough for structured queries such as lists, definitions, and section-based questions.
- Keyword scoring, section-aware boosts, and reranking improve retrieval precision.

**c. Structure-Aware Chunking**

- Fixed-size chunking can split definitions, lists, or headings in the wrong place.
- Structure-aware chunking keeps related information together and improves answer quality.

**d. Multi-Stage Answer Generation**

- The system first attempts deterministic extraction for recognizable patterns.
- If that is not enough, it uses LLM generation.
- If generation is unavailable or unsuitable, it falls back to extractive methods.

**e. Query-Type Awareness**

- The system detects definition, list, explanatory, comparison, and out-of-scope questions.
- Retrieval and answer formatting are adjusted based on the query type.

**f. Guardrails before Generation**

- Filtering happens before generation to avoid the garbage-in, garbage-out problem.
- This improves answer quality and prevents unrelated chunks from influencing the response.

## 3. Trade-offs

**a. Accuracy vs Complexity**

- Reranking and guardrails improve accuracy.
- They also increase system complexity and add a small amount of latency.

**b. Flexibility vs Strictness**

- Strict refusal logic reduces hallucinations.
- It may refuse borderline questions when only partial evidence is available.

**c. Chunk Size vs Context Quality**

- Larger chunks preserve more context.
- Smaller chunks improve retrieval precision.
- The system balances both using overlap and context windows.

**d. Model Choice**

- OpenAI generation provides stronger answers when available.
- Local fallback improves robustness when external generation is unavailable.
- This creates a trade-off between answer quality and dependency on external services.

**e. Single Document Scope**

- The current implementation indexes one active PDF at a time.
- This simplifies the user flow and vector store logic but limits multi-document querying.

## 4. Observability and Testability

The system includes mechanisms to inspect and validate its behavior:

- A debug retrieval endpoint is provided to inspect retrieved chunks, scores, and filtering decisions.
- Predefined test cases, including valid and invalid queries, are used to evaluate:
  - correctness of answers
  - proper refusal behavior
  - grounding quality
- The system ensures:
  - valid queries return grounded answers with citations
  - invalid queries consistently return the refusal response

This makes the system easier to evaluate, debug, and improve.

## 5. Limitations

- The system may struggle with very complex reasoning questions that require multi-hop understanding across distant sections.
- Poorly formatted or noisy PDFs can affect text extraction and retrieval quality.
- The system currently supports one active document at a time.

These limitations can be improved using better document parsing, stronger reranking models, and multi-document retrieval support.

## 6. Conclusion

DocuMind prioritizes correctness, grounding, and reliability over aggressive answer generation. The combination of structure-aware chunking, hybrid retrieval, guardrails, and controlled generation helps the system answer from the document while refusing unsupported questions.
