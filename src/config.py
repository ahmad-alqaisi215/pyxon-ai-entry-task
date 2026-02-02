import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    DATA: Path = Path(__file__).parent.parent.parent / "data"

    PARAGRAPH_VARIATION_THRESH: float = 0.2
    INDEX: str = "pyxon"

    EMBEDDING_MODEL_NAME: str = "text-embedding-3-large"
    CROSS_ENCODER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    LLM_MODEL_NAME: str = "llama-3.3-70b-versatile"
    DIMENSIONS: int = 1024
    CHUNK_OVERLAP: float = 0.2
    PINECONE_API_KEY: str = os.environ.get("PINECONE_API_KEY")
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY")
    DATABASE_URL: str = os.environ.get("DATABASE_URL")
    LLAMAINDEX_API_KEY = os.environ.get("LLAMAINDEX_API_KEY")
    LANGSMITH_API_KEY: str = os.environ.get("LANGSMITH_API_KEY")

    PERCENTILE_THRESH: float = 0.9
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.8
    MAX_RAG_ITERATIONS: int = 6

