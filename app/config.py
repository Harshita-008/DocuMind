import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "google/flan-t5-base")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "280"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "55"))

TOP_K = int(os.getenv("TOP_K", "14"))
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "14"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.16"))
