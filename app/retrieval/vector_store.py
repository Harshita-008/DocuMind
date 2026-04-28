import chromadb

from app.ingestion.embedder import get_embeddings


COLLECTION_NAME = "pdf_docs"
DB_PATH = "./data/db"


class VectorStore:
    def __init__(self, reset=False):
        self.client = chromadb.PersistentClient(path=DB_PATH)

        if reset:
            try:
                self.client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, chunks):
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        metadatas = [
            {
                "page": int(c["page"]),
                "chunk_index": int(c.get("chunk_index", i)),
                "section_title": str(c.get("section_title") or ""),
            }
            for i, c in enumerate(chunks)
        ]
        ids = [
            f"page-{c['page']}-chunk-{c.get('chunk_index', i)}"
            for i, c in enumerate(chunks)
        ]

        embeddings = get_embeddings(texts)

        self.collection.upsert(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )

    def query(self, query_text, top_k=5):
        query_embedding = get_embeddings([query_text])[0]
        count = self.collection.count()

        if count == 0:
            return {"documents": [[]], "metadatas": [[]], "ids": [[]], "distances": [[]]}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"]
        )

        return results

    def get_all(self):
        return self.collection.get(include=["documents", "metadatas"])

    def count(self):
        return self.collection.count()
