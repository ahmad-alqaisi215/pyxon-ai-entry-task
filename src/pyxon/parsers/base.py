from abc import ABC, abstractmethod
from pathlib import Path

from langchain_core.documents import Document


class BaseParser(ABC):
    SUPPORTED_EXTENSIONS: list[str] = ["doc", "docx", "pdf", "txt"]

    def __init__(self, file_path: Path):
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"{self.__class__.__name__} does not support "
                f"'{file_path.suffix}'. Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        self._path = file_path
        super().__init__()

    @abstractmethod
    def parse(self) -> Document:
        pass
