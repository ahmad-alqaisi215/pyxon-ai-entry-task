from src.pyxon.parsers import parse_document

print(
    parse_document(
        "/home/amq/Downloads/Feature Planning & Project Management (1).pdf"
    ).page_content
)
