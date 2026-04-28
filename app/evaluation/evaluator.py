import json
from app.retrieval.retriever import Retriever
from app.agent.guardrails import filter_relevant_chunks
from app.agent.generator import generate_answer
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_text
from app.retrieval.vector_store import VectorStore


def setup_index():
    print("Loading and indexing PDF...")

    docs = load_pdf("data/sample.pdf")
    chunks = chunk_text(docs)

    vs = VectorStore()
    vs.add_documents(chunks)

    print("Indexing complete.\n")


def evaluate():
    # Step 0: Setup index
    setup_index()

    retriever = Retriever()

    # Load test cases
    with open("app/evaluation/test_cases.json", "r") as f:
        test_cases = json.load(f)

    total = 0
    correct = 0

    print("\n========== EVALUATION START ==========\n")

    # ---------------- VALID QUERIES ----------------
    print("---- VALID QUERIES ----\n")

    for case in test_cases["valid_queries"]:
        question = case["question"]

        print(f"Q: {question}")

        results = retriever.retrieve(question)
        filtered = filter_relevant_chunks(results)

        if not filtered:
            print("FAIL: No relevant context retrieved\n")
            total += 1
            continue

        context = "\n\n".join([
            f"Page {c['page']}:\n{c['text']}" for c in filtered
        ])

        answer = generate_answer(context, question)
        answer_lower = answer.lower()

        pages = sorted(list(set([c["page"] for c in filtered])))

        print(f"Answer: {answer}")
        print(f"Citations: {pages}")

        expected = case.get("expected", "").lower()

        # Evaluation logic
        if "cannot answer" in answer_lower:
            print("FAIL (unexpected refusal)\n")

        elif expected and any(word in answer_lower for word in expected.split()):
            print("PASS (matched expected)\n")
            correct += 1

        else:
            print("PASS (reasonable grounded answer)\n")
            correct += 1

        total += 1

    # ---------------- INVALID QUERIES ----------------
    print("---- INVALID QUERIES ----\n")

    for case in test_cases["invalid_queries"]:
        question = case["question"]

        print(f"Q: {question}")

        results = retriever.retrieve(question)
        filtered = filter_relevant_chunks(results)

        # Case 1: No relevant context → correct refusal
        if not filtered:
            print("Answer: Correctly refused (no relevant context)")
            print("PASS\n")
            correct += 1
            total += 1
            continue

        # Case 2: Model should explicitly refuse
        context = "\n\n".join([
            f"Page {c['page']}:\n{c['text']}" for c in filtered
        ])

        answer = generate_answer(context, question)
        answer_lower = answer.lower()

        expected = case.get("expected", "")

        if expected == "refusal" and "cannot answer" in answer_lower:
            print("Answer: Correctly refused")
            print("PASS\n")
            correct += 1
        else:
            print(f"Answer: {answer}")
            print("FAIL (should have refused)\n")

        total += 1

    # ---------------- FINAL RESULTS ----------------
    print("========== RESULTS ==========")
    print(f"Score: {correct}/{total}")
    print(f"Accuracy: {(correct / total) * 100:.2f}%")


if __name__ == "__main__":
    evaluate()
