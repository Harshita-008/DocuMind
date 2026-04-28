from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_text
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import Retriever

from app.agent.guardrails import filter_relevant_chunks
from app.agent.generator import generate_answer

# Step 1: Load + chunk
docs = load_pdf("data/sample.pdf")
chunks = chunk_text(docs)

# Step 2: Store in vector DB
vs = VectorStore()
vs.add_documents(chunks)

# Step 3: Retrieval
retriever = Retriever()

queries = [
    # ✅ Valid queries
    "What is phishing?",
    "How do phishing attacks work?",
    "What do cybercriminals use phishing for?",
    "What solutions are suggested to prevent phishing?",
    "What is the impact of phishing attacks?",

    # ❌ Invalid queries
    "Who is the president of India?",
    "What is machine learning?",
    "Explain quantum physics"
]

for query in queries:
    print("\n==============================")
    print(f"QUESTION: {query}")

    results = retriever.retrieve(query)
    filtered = filter_relevant_chunks(results)

    if not filtered:
        print("ANSWER: I cannot answer this question from the provided document.")
        print("CITATIONS: None")
        continue

    context = "\n\n".join([
        f"Page {c['page']}:\n{c['text']}" for c in filtered
    ])

    answer = generate_answer(context, query)

    pages = list(set([c["page"] for c in filtered]))
    citations = ", ".join([f"Page {p}" for p in pages])

    print("\nANSWER:")
    print(answer)

    print("\nCITATIONS:")
    print(citations)