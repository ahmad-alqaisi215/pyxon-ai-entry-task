from dataclasses import dataclass


@dataclass
class Settings:
    PARAGRAPH_Variation_THRESH: float = 0.2
    INDEX_NAME: str = "pyxon"
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-large"
    CHUNK_OVERLAP: float = 0.2
