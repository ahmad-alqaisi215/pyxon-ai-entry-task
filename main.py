from src.pyxon.parsers import parse_document
from src.pyxon.storage.database.repository import SQLStore
from src.pyxon.storage.database.schemas import ChunkCreate, DocumentCreate
from src.pyxon.storage.vs import VectorStore

doc = parse_document("/home/amq/Desktop/CV/AhmadAlqaisi_AI_Engineer_Resume.pdf")
repo = SQLStore()

doc_create = DocumentCreate(
    filename="AhmadAlqaisi_AI_Engineer_Resume.pdf",
    source_path="/home/amq/Desktop/CV/AhmadAlqaisi_AI_Engineer_Resume.pdf",
    doc_type="pdf",
)

doc_id = repo.save_document(doc_create)

print(f"Doc was saved with id: {doc_id}")

vs = VectorStore()

chunks = vs.chunk_document(doc)
print(f"Doc chunking completed: len={len(chunks)}")

vs.add_documents(chunks, doc_id)

print("Chunks added to vs")

repo.save_chunks(
    doc_id,
    [
        ChunkCreate(chunk_index=i, chunk_text=chunk.page_content)
        for i, chunk in enumerate(chunks)
    ],
)

print("Chunks saved to db")
