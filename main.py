from src.pyxon.parsers import parse_document
from src.pyxon.storage.vs import VectorStore

doc = parse_document("/home/amq/Downloads/file-sample_100kB.docx")


vs = VectorStore()

chunks = vs.chunk_document(doc)
vs.add_documents(chunks, "tst_no_1_")

print(len(chunks))
print(chunks[0].metadata)
