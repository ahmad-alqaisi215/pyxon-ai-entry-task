from typing import List

from langchain_community.vectorstores import Pinecone
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.pyxon.config import Settings

index_name = "pyxon-docs"

_embedding_func = OpenAIEmbeddings(model=Settings.EMBEDDING_MODEL_NAME)
_vs = Pinecone.from_existing_index(
    index_name=Settings.INDEX_NAME, embedding=_embedding_func
)


def get_vs() -> Pinecone:
    return _vs


def chunk_document(doc: Document):
    chunker = _get_chunker(doc)
    return chunker.split_documents([doc])


def _get_chunker(doc: Document):
    if doc.metadata["chunking_strategy"] == "FIXED":
        return RecursiveCharacterTextSplitter(
            chunk_size=doc.metadata["chunk_size"],
            chunk_overlap=doc.metadata["chunk_overlap"],
        )

    return SemanticChunker(
        _embedding_func, breakpoint_threshold_type="standard_deviation"
    )


def add_documents(docs: List[Document], document_id: str):
    texts = [d.page_content for d in docs]
    metadatas = []

    for i, doc in enumerate(docs):
        metadatas.append(
            {
                "source": doc.metadata["source"],
                "document_id": document_id,
                "chunk_index": i,
                "total_chunks": len(docs),
                "chunking_strategy": doc.metadata.get("chunking_strategy"),
            }
        )

    ids = [f"{document_id}_{i}" for i in range(len(docs))]

    _vs.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    return ids
