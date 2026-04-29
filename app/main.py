import os
import re
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.agent.generator import REFUSAL, generate_answer
from app.agent.guardrails import filter_relevant_chunks
from app.config import MAX_CONTEXT_CHUNKS
from app.ingestion.chunker import chunk_text
from app.ingestion.pdf_loader import load_pdf
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore


DB = None
retriever = None

FRONTEND_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.getenv("FRONTEND_ORIGINS", "").split(",")
    if origin.strip()
]

app = FastAPI(title="DocuMind")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        *FRONTEND_ORIGINS,
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB = VectorStore()
retriever = Retriever()


@app.get("/")
async def health_check():
    return {"status": "ok", "service": "DocuMind"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global DB, retriever

    os.makedirs("data", exist_ok=True)
    file_path = f"data/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    docs = load_pdf(file_path)
    chunks = chunk_text(docs)
    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text could be extracted from this PDF.")

    DB = VectorStore(reset=True)
    DB.add_documents(chunks)
    retriever = Retriever()

    return {
        "message": "PDF uploaded and processed successfully",
        "pages": len(docs),
        "chunks": len(chunks),
    }


@app.post("/chat")
async def chat(query: str):
    try:
        if not query or not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        if retriever is None:
            raise HTTPException(status_code=400, detail="Please upload a PDF first.")

        results = retriever.retrieve(query)
        filtered = filter_relevant_chunks(results, query=query, max_chunks=MAX_CONTEXT_CHUNKS + 2)

        if not filtered:
            return {"answer": REFUSAL, "citations": []}

        context = "\n\n".join([
            f"Page {chunk['page']}:\n{chunk.get('window_text') or chunk['text']}"
            for chunk in filtered
        ])
        if not context.strip():
            return {"answer": REFUSAL, "citations": []}

        answer = generate_answer(context, query)
        if not answer or len(answer.split()) < 3 or answer == REFUSAL:
            return {"answer": REFUSAL, "citations": []}

        return {
            "answer": answer,
            "citations": _select_citations(answer, query, filtered),
        }

    except HTTPException:
        raise
    except Exception as exc:
        print("ERROR:", str(exc))
        return {"answer": "An error occurred", "citations": []}


@app.get("/debug/retrieve")
async def debug_retrieve(query: str):
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if retriever is None:
        raise HTTPException(status_code=400, detail="Please upload a PDF first.")

    results = retriever.retrieve(query)
    filtered = filter_relevant_chunks(results, query=query, max_chunks=MAX_CONTEXT_CHUNKS + 2)
    return {
        "query": query,
        "chunks": [
            {
                "page": chunk.get("page"),
                "chunk_index": chunk.get("chunk_index"),
                "score": chunk.get("score"),
                "guardrail_score": chunk.get("guardrail_score"),
                "text": (chunk.get("text") or "")[:500],
                "window_text": (chunk.get("window_text") or "")[:800],
            }
            for chunk in filtered
        ],
    }


STOPWORDS = {
    "a", "an", "and", "are", "as", "by", "for", "from", "in", "is", "it",
    "of", "on", "or", "that", "the", "this", "to", "was", "were", "with",
    "what", "which", "who", "why", "how", "does", "did", "do", "page",
}

GENERIC_CITATION_TERMS = {
    "document", "paper", "papers", "research", "scientific", "science",
    "section", "sections", "writing", "written", "should", "main",
    "question", "answer",
}


def _select_citations(answer, query, chunks):
    anchored_pages = _anchored_citation_pages(query, chunks)
    if anchored_pages:
        max_pages = 3 if answer.count("- ") >= 3 else 2
        return [f"Page {page}" for page in sorted(anchored_pages[:max_pages])]

    answer_terms = set(_content_terms(answer))
    distinctive_answer_terms = answer_terms - GENERIC_CITATION_TERMS
    query_terms = set(_content_terms(query))
    page_scores = {}
    page_order = {}

    has_distinctive_overlap = any(
        distinctive_answer_terms.intersection(set(_content_terms(chunk.get("text", ""))))
        for chunk in chunks
    )

    for order, chunk in enumerate(chunks):
        page = int(chunk.get("page", 0) or 0)
        if page <= 0:
            continue

        evidence_text = f"{chunk.get('text', '')}\n{chunk.get('window_text', '')}"
        chunk_terms = set(_content_terms(evidence_text))
        answer_overlap_terms = (
            distinctive_answer_terms.intersection(chunk_terms)
            if has_distinctive_overlap
            else answer_terms.intersection(chunk_terms)
        )
        answer_overlap = len(answer_overlap_terms)
        query_overlap = len(query_terms.intersection(chunk_terms))
        if has_distinctive_overlap and answer_overlap <= 0:
            continue

        score = answer_overlap * 2 + query_overlap + float(chunk.get("guardrail_score", 0) or 0)

        if score <= 0:
            continue
        page_scores[page] = max(page_scores.get(page, 0), score)
        page_order.setdefault(page, order)

    if not page_scores and chunks:
        first_page = int(chunks[0].get("page", 0) or 0)
        return [f"Page {first_page}"] if first_page > 0 else []

    ordered_pages = sorted(
        page_scores,
        key=lambda page: (-page_scores[page], page_order.get(page, 9999), page),
    )
    max_pages = 3 if answer.count("- ") >= 3 else 2
    return [f"Page {page}" for page in sorted(ordered_pages[:max_pages])]


def _anchored_citation_pages(query, chunks):
    """Find pages whose text shares the most content terms with the query.

    Returns at most 2 candidate pages with high query-term overlap, or an
    empty list when no chunk clears the threshold (citation falls back to the
    general scoring path in _select_citations).
    """
    query_terms = set(_content_terms(query))
    if not query_terms or not chunks:
        return []

    page_hits = {}
    for chunk in chunks:
        text = f"{chunk.get('text', '')}\n{chunk.get('window_text', '')}".lower()
        chunk_terms = set(_content_terms(text))
        overlap = len(query_terms & chunk_terms)
        page = int(chunk.get("page", 0) or 0)
        if page > 0 and overlap > 0:
            page_hits[page] = max(page_hits.get(page, 0), overlap)

    if not page_hits:
        return []

    threshold = max(page_hits.values()) * 0.6
    strong = [p for p, hits in page_hits.items() if hits >= threshold]
    return sorted(strong)[:2]


def _content_terms(text):
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", (text or "").lower())
        if token not in STOPWORDS
    ]
