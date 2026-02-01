from langchain_core.documents import Document

from src.pyxon.parsers.base import BaseParser


class PyxonTextParser(BaseParser):
    SUPPORTED_EXTENSIONS: list[str] = [".txt"]

    def __init__(self, file_path):
        super().__init__(file_path)

    def parse(self) -> Document:

        content = self._file_path.read_text()

        return Document(page_content=content, metadata={"source": str(self._file_path)})
