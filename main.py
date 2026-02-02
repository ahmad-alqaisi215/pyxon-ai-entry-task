# main
import streamlit as st
from pathlib import Path
import tempfile
import uuid

from src.pyxon.parsers import parse_document
from src.pyxon.storage.vs import VectorStore
from src.pyxon.storage.database.repository import SQLStore
from src.pyxon.storage.database.schemas import DocumentCreate, ChunkCreate
from src.pyxon.rag.graph import app as rag_app
from src.pyxon.rag.state import AgentState
from langchain_core.messages import HumanMessage


st.set_page_config(
    page_title="Pyxon AI - Document Chat",
    page_icon="🤖",
    layout="wide"
)

if "document_id" not in st.session_state:
    st.session_state.document_id = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def process_uploaded_file(uploaded_file) -> str:
    """Save uploaded file, parse it, chunk it, store in both DBs."""
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    
    with st.spinner("📄 Parsing document..."):
        doc = parse_document(tmp_path, advanced=True)
    
    vs = VectorStore()
    sql_store = SQLStore()
    
    doc_id = str(uuid.uuid4())
    
    with st.spinner("💾 Saving to SQL database..."):
        doc_schema = DocumentCreate(
            filename=uploaded_file.name,
            source_path=tmp_path,
            doc_type=Path(uploaded_file.name).suffix.lstrip(".")
        )
        sql_doc_id = sql_store.save_document(doc_schema)
    
    with st.spinner("✂️ Chunking document..."):
        chunks = vs.chunk_document(doc)
    
    with st.spinner("🔢 Embedding and storing in vector database..."):
        vs.add_documents(chunks, sql_doc_id)
    
    with st.spinner("💾 Saving chunks to SQL..."):
        chunk_schemas = [
            ChunkCreate(
                chunk_index=i,
                chunk_text=chunk.page_content
            )
            for i, chunk in enumerate(chunks)
        ]
        sql_store.save_chunks(sql_doc_id, chunk_schemas)
    
    Path(tmp_path).unlink()
    
    return sql_doc_id


def run_rag_query(question: str, document_id: str) -> str:
    """Run the RAG pipeline on the question."""
    
    initial_state: AgentState = {
        "messages": [HumanMessage(content=question)],
        "queries": [question],
        "critiques": [],
        "iteration": 0,
        "should_continue": True,
        "retrieved_docs": [],
        "reranked_docs": [],
        "top_k": 5,
        "metadata_filter": {"document_id": document_id},
        "answer": ""
    }
    
    with st.spinner("🤔 Thinking..."):
        final_state = rag_app.invoke(initial_state)
    
    return final_state["answer"]


st.title("🤖 Pyxon AI - Document Chat")
st.markdown("Upload a document and chat with it using advanced RAG")

with st.sidebar:
    st.header("📁 Document Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["txt", "pdf", "doc", "docx"],
        help="Upload one document at a time"
    )
    
    if uploaded_file and st.button("Process Document", type="primary"):
        try:
            doc_id = process_uploaded_file(uploaded_file)
            st.session_state.document_id = doc_id
            st.session_state.filename = uploaded_file.name
            st.session_state.chat_history = []
            st.success(f"✅ Document processed: {uploaded_file.name}")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error processing document: {str(e)}")
    
    st.divider()
    
    if st.session_state.document_id:
        st.success(f"📄 Active: {st.session_state.filename}")
        if st.button("Clear Document"):
            st.session_state.document_id = None
            st.session_state.filename = None
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("No document loaded")

if st.session_state.document_id:
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("Ask a question about the document..."):
        
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        try:
            answer = run_rag_query(prompt, st.session_state.document_id)
            
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
                
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.error(error_msg)

else:
    st.info("👈 Upload a document from the sidebar to start chatting")
    
    st.markdown("""
    ### How it works:
    
    1. **Upload** a document (TXT, PDF, DOC, DOCX)
    2. **Wait** for processing (parsing, chunking, embedding)
    3. **Ask** questions about the content
    4. **Get** intelligent answers powered by RAG
    
    ### Features:
    - 🧠 Intelligent chunking (fixed/semantic)
    - 🔍 Semantic search with reranking
    - 🔁 Self-reflective retrieval loop
    - 📊 Dual storage (Vector + SQL)
    - 🌍 Full Arabic support
    """)