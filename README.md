# DocuMind

DocuMind is a document question-answering application that lets users upload a PDF and ask natural-language questions about its contents. It uses a retrieval-augmented generation pipeline with PDF extraction, structure-aware chunking, vector search, keyword reranking, guardrails, and grounded answer generation with page citations.

Deployed link: `PASTE_YOUR_DEPLOYED_LINK_HERE`

## Overview

DocuMind is designed for PDFs such as textbooks, research papers, reports, and technical documents. Instead of sending the whole PDF to a model, it extracts the document text, indexes meaningful chunks, retrieves only the most relevant evidence for each question, and generates a concise answer using that evidence.

The project focuses on reliable document-grounded QA:

- Answers should come only from the uploaded PDF.
- Irrelevant questions should be refused.
- Citations should point to the pages used as evidence.
- Retrieval should work across different PDF styles, not just one fixed document.

## Features

- PDF upload and indexing through a FastAPI backend.
- Text extraction using PyMuPDF with fallback support through `pypdf`.
- Structure-aware chunking for headings, lists, paragraphs, and page metadata.
- ChromaDB vector storage with Sentence Transformers embeddings.
- Hybrid retrieval using semantic search, keyword matching, section-aware scoring, and reranking.
- Guardrails to filter noisy chunks, references, boilerplate, and unrelated context.
- Grounded answer generation with strict refusal for unsupported questions.
- Page-level citations for answers.
- Debug retrieval endpoint for inspecting selected chunks.
- React + Vite frontend for uploading PDFs and chatting with documents.
- Evaluation script with valid and invalid test cases.

## System Architecture

```text
User
  |
  v
React Frontend
  |
  v
FastAPI Backend
  |
  +--> PDF Loader
  |      - extracts text page by page
  |      - repairs common spacing/artifact issues
  |
  +--> Chunker
  |      - creates sentence/section-aware chunks
  |      - stores page, chunk index, section title, and context windows
  |
  +--> Embedder + Vector Store
  |      - generates embeddings with Sentence Transformers
  |      - stores chunks in ChromaDB
  |
  +--> Retriever
  |      - combines vector search and keyword ranking
  |      - reranks chunks using query type, section cues, and phrase matches
  |
  +--> Guardrails
  |      - removes unrelated, noisy, low-value, or unsupported chunks
  |
  +--> Generator
         - creates a grounded answer from retrieved context
         - refuses questions not supported by the document
         - returns answer with citations
```

## Tech Stack

Backend:

- Python
- FastAPI
- Uvicorn
- ChromaDB
- Sentence Transformers
- PyMuPDF
- pypdf
- OpenAI API support
- python-dotenv

Frontend:

- React
- Vite
- JavaScript

Evaluation and utilities:

- Custom evaluator script
- JSON test cases
- Retrieval debug endpoint

## How It Works

1. A user uploads a PDF.
2. The backend saves the PDF in `data/`.
3. The PDF loader extracts text page by page.
4. The chunker cleans and splits text into meaningful chunks while preserving page numbers and context windows.
5. The embedder converts chunks into vector embeddings.
6. ChromaDB stores the chunks, metadata, and embeddings.
7. When the user asks a question, the retriever searches for candidate chunks.
8. The retriever reranks candidates using semantic similarity, keyword overlap, section cues, query type, and phrase matches.
9. Guardrails filter unsupported or noisy context.
10. The generator creates a concise answer using only the selected context.
11. The API returns the answer and page citations.

## Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd DocuMind
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install backend dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=google/flan-t5-base
OPENAI_LLM_MODEL=gpt-4o-mini
```

`OPENAI_API_KEY` is optional if you want to rely on the local fallback model, but OpenAI generation gives better answer quality.

### 5. Start the backend

```bash
uvicorn app.main:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

### 6. Install frontend dependencies

```bash
cd frontend
npm install
```

### 7. Start the frontend

```bash
npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

## Usage

1. Open the frontend in the browser.
2. Upload a PDF document.
3. Wait for the PDF to be processed and indexed.
4. Ask questions about the uploaded document.
5. Read the generated answer and citations.

Example questions:

```text
What is entrepreneurship?
What are the main sections of a scientific research paper?
What problem does the proposed framework solve?
Which dataset is used in the study?
What are the limitations mentioned in the document?
```

Unsupported question example:

```text
What is artificial intelligence?
```

If the uploaded PDF does not contain the answer, DocuMind should respond:

```text
I cannot answer this question from the provided document.
```

## Example Output

Question:

```text
What are the types of entrepreneurship based on ownership?
```

Answer:

```text
Entrepreneurship can be classified by ownership into:
- Founders or pure entrepreneurs: start and build the business from their own idea
- Second-generation operators of family-owned businesses: inherit and continue an existing family business
- Franchisees: operate a licensed business using the franchiser's proven name, methods, and support
- Owner-managers: buy an existing business and then manage it with their own time and resources

Citations: Page 23
```

## Evaluation

The project includes an evaluator that indexes `data/sample.pdf` and runs valid and invalid test cases from `app/evaluation/test_cases.json`.

Run:

```bash
python -m app.evaluation.evaluator
```

Latest evaluator result:

```text
Score: 8/8
Accuracy: 100.00%
```

The evaluator checks:

- whether valid document questions receive grounded answers
- whether unsupported questions are refused
- whether the retrieval and generation pipeline works after indexing

## Limitations

- PDF extraction quality depends on the structure of the source PDF.
- Scanned image-only PDFs require OCR, which is not currently included.
- Very complex tables, formulas, and multi-column layouts may still produce noisy text.
- The current vector database stores one active indexed PDF at a time.
- Some domain-specific answers may require stronger reranking or a more capable LLM.
- Citations are page-level, not sentence-level.

## Future Improvements

- Add OCR support for scanned PDFs.
- Support multiple PDFs and document collections.
- Add cross-encoder reranking for stronger retrieval precision.
- Improve table and figure extraction.
- Add sentence-level citation grounding.
- Add user authentication and document history.
- Add deployment configuration for production hosting.
- Add automated test coverage for retrieval, generation, and API endpoints.

## Folder Structure

```text
DocuMind/
├── app/
│   ├── agent/
│   │   ├── generator.py        # Grounded answer generation
│   │   ├── guardrails.py       # Context filtering and relevance checks
│   │   └── promt.py            # System prompt configuration
│   ├── evaluation/
│   │   ├── evaluator.py        # Evaluation runner
│   │   └── test_cases.json     # Valid and invalid evaluation queries
│   ├── ingestion/
│   │   ├── chunker.py          # Structure-aware chunking
│   │   ├── embedder.py         # Embedding model wrapper
│   │   └── pdf_loader.py       # PDF text extraction
│   ├── retrieval/
│   │   ├── retriever.py        # Hybrid retrieval and reranking
│   │   └── vector_store.py     # ChromaDB storage layer
│   ├── utils/
│   │   ├── helpers.py
│   │   └── logger.py
│   ├── config.py               # Environment and pipeline settings
│   └── main.py                 # FastAPI application
├── data/
│   ├── sample.pdf              # Sample evaluation PDF
│   └── db/                     # Local ChromaDB files, ignored by Git
├── frontend/
│   ├── src/                    # React frontend source
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── notebooks/
├── requirements.txt
├── test_pipeline.py
└── README.md
```

## API Endpoints

Upload a PDF:

```text
POST /upload
```

Ask a question:

```text
POST /chat?query=your_question
```

Debug retrieval:

```text
GET /debug/retrieve?query=your_question
```

## Notes

- Re-upload a PDF after changing ingestion, chunking, retrieval, or guardrail logic so the vector database is rebuilt with the latest pipeline.
- Do not commit local ChromaDB files from `data/db/`.
- Keep API keys in `.env`, not in source code.
