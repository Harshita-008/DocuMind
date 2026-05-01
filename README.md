# DocuMind

DocuMind is a document question-answering application that lets users upload a PDF and ask natural-language questions about its contents. It uses a retrieval-augmented generation pipeline with PDF extraction, structure-aware chunking, hybrid retrieval, guardrails, and grounded answer generation with page citations.

Deployed link: https://documind-ysjx.onrender.com

Demo Link: https://drive.google.com/drive/folders/1Ia0iNkUnB3VJSVd9X-XZzffRIVoPdTeD?usp=sharing

## Overview

DocuMind is built for PDFs such as textbooks, research papers, reports, and technical documents. Instead of sending the whole document to a model, it extracts text, indexes meaningful chunks, retrieves the most relevant evidence, and generates concise answers from that evidence.

The system focuses on:

- Correct answers grounded in the uploaded PDF.
- Clear refusal when the answer is not present in the document.
- Page citations for retrieved evidence.
- Reliable behavior across different PDF formats and writing styles.

The system was tested using predefined valid and invalid queries to ensure correctness, grounding, and refusal behavior.

## Features

- Upload and process PDF documents.
- Extract text using PyMuPDF with fallback support through `pypdf`.
- Clean and repair common PDF extraction artifacts.
- Split documents using structure-aware chunking for paragraphs, headings, and lists.
- Store embeddings and metadata in ChromaDB.
- Retrieve context using vector search, keyword scoring, section cues, and reranking.
- Filter irrelevant or noisy chunks with guardrails.
- Generate concise, document-grounded answers.
- Refuse unsupported questions instead of hallucinating.
- Return page-level citations.
- Inspect retrieval behavior through a debug endpoint.
- Use a React + Vite frontend for document upload and chat.

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
  +-- PDF Loader
  |   +-- extracts text page by page
  |   +-- repairs common spacing and extraction issues
  |
  +-- Chunker
  |   +-- creates section-aware chunks
  |   +-- stores page, chunk index, section title, and context windows
  |
  +-- Embedder and Vector Store
  |   +-- generates embeddings with Sentence Transformers
  |   +-- stores chunks and metadata in ChromaDB
  |
  +-- Retriever
  |   +-- combines vector search and keyword ranking
  |   +-- reranks chunks using query type, phrase matches, and section cues
  |
  +-- Guardrails
  |   +-- removes unrelated, noisy, low-value, or unsupported chunks
  |
  +-- Generator
      +-- answers using only retrieved context
      +-- refuses unsupported questions
      +-- returns answer with citations
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

Testing and utilities:

- Predefined valid and invalid queries
- JSON test cases
- Retrieval debug endpoint

## Folder Structure

```text
DocuMind/
|-- app/
|   |-- agent/
|   |   |-- generator.py        # Grounded answer generation
|   |   |-- guardrails.py       # Context filtering and relevance checks
|   |   `-- prompt.py            # System prompt configuration
|   |-- evaluation/
|   |   |-- evaluator.py        # Testability runner for predefined queries
|   |   `-- test_cases.json     # Valid and invalid test cases
|   |-- ingestion/
|   |   |-- chunker.py          # Structure-aware chunking
|   |   |-- embedder.py         # Embedding model wrapper
|   |   `-- pdf_loader.py       # PDF text extraction
|   |-- retrieval/
|   |   |-- retriever.py        # Hybrid retrieval and reranking
|   |   `-- vector_store.py     # ChromaDB storage layer
|   |-- utils/
|   |   |-- helpers.py
|   |   `-- logger.py
|   |-- config.py               # Environment and pipeline settings
|   `-- main.py                 # FastAPI application
|-- data/
|   |-- sample.pdf              # Sample PDF used for testing
|   `-- db/                     # Local ChromaDB files, ignored by Git
|-- frontend/
|   |-- src/                    # React frontend source
|   |-- index.html
|   |-- package.json
|   `-- vite.config.js
|-- sample/                     # Sample documents for testing/demo
|-- notebooks/
|-- requirements.txt
|-- test_pipeline.py
`-- README.md
```

## How It Works

1. The user uploads a PDF.
2. The backend saves the PDF in `data/`.
3. The PDF loader extracts text page by page.
4. The chunker cleans and splits the text into meaningful chunks.
5. The embedder converts chunks into vector embeddings.
6. ChromaDB stores chunks, embeddings, and metadata.
7. For each question, the retriever selects candidate chunks.
8. The reranker prioritizes chunks using semantic similarity, keyword overlap, query type, and section cues.
9. Guardrails remove unrelated or noisy context.
10. The generator answers only from the selected context.
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
ENABLE_SENTENCE_TRANSFORMERS=true
EMBEDDING_BATCH_SIZE=32
MAX_UPLOAD_MB=10
LLM_MODEL=google/flan-t5-base
OPENAI_LLM_MODEL=gpt-4o-mini
```

`OPENAI_API_KEY` is optional if you want to rely on the local fallback model, but OpenAI generation generally gives better answer quality.

For small Render backend instances, keep upload processing lightweight:

```env
FRONTEND_ORIGINS=https://documind-ysjx.onrender.com
ENABLE_SENTENCE_TRANSFORMERS=false
EMBEDDING_BATCH_SIZE=16
MAX_UPLOAD_MB=10
```

If uploads still close the connection, check the Render backend logs during an
upload. A restart or "out of memory" message means the instance needs a smaller
PDF, lower `MAX_UPLOAD_MB`, or a larger Render plan.

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
3. Wait for indexing to complete.
4. Ask questions about the uploaded document.
5. Review the answer and page citations.

Unsupported questions should return:

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

## Testability

The project includes a sample PDF and predefined valid and invalid queries to verify correctness, grounding, and refusal behavior.

Sample PDF:

```text
sample/sample.pdf
```

Valid queries:

1. What is entrepreneurship?
   Expected: Definition with citation.

2. What are types of entrepreneurship?
   Expected: List of types with citations.

3. What are the problems faced by entrepreneurs in India?
   Expected: Bullet points.

4. Why is entrepreneurship important for economic development?
   Expected: Bullet points.

5. What was the main issue in the Satyam case study?
   Expected: Critical thinking answer grounded in the document.

Invalid queries:

1. What is machine learning?
   Expected: Refusal.

2. Who is Elon Musk?
   Expected: Refusal.

3. Write a Python program.
   Expected: Refusal.

## Test Instructions

1. Start the backend:

```bash
uvicorn app.main:app --reload
```

2. Start the frontend:

```bash
cd frontend
npm install
npm run dev
```

3. Open the application in the browser:

```text
http://localhost:5173
```

4. Upload the sample PDF:

```text
sample/sample.pdf
```

5. Test valid queries:

```text
What is entrepreneurship?
What are types of entrepreneurship?
What are the problems faced by entrepreneurs in India?
Why is entrepreneurship important for economic development?
What was the main issue in the Satyam case study?
```

Expected:

- Answers should be correct.
- Answers must include citations.
- Answers should be grounded in the document.

6. Test invalid queries:

```text
What is machine learning?
Who is Elon Musk?
Write a Python program
```

Expected:

```text
I cannot answer this question from the provided document.
```

7. Optional: run the automated test runner:

```bash
python -m app.evaluation.evaluator
```

The purpose of this runner is to verify behavior on predefined queries, not to provide a full scoring or metrics dashboard.

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
- Add automated tests for retrieval, generation, and API endpoints.

## Notes

- Re-upload a PDF after changing ingestion, chunking, retrieval, or guardrail logic so the vector database is rebuilt with the latest pipeline.
- Do not commit local ChromaDB files from `data/db/`.
- Keep API keys in `.env`, not in source code.
