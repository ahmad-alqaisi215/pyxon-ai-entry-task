import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    DATA: Path = Path(__file__).parent.parent.parent / "data"

    PARAGRAPH_Variation_THRESH: float = 0.2
    INDEX: str = "pyxon"

    EMBEDDING_MODEL_NAME: str = "text-embedding-3-large"
    DIMENSIONS: int = 1024
    CHUNK_OVERLAP: float = 0.2
    PINECONE_API_KEY: str = os.environ.get("PINECONE_API_KEY")
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY")
    DATABASE_URL: str = os.environ.get("DATABASE_URL")
