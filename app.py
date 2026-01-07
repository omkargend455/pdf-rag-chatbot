import os
import streamlit as st

# ----------------- API KEY (SAFE) -----------------
try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass  # allows local dev without secrets

# ----------------- LOCAL IMPORTS -----------------
from lc_utils import load_and_split_pdf_from_upload
from rag_utils import (
    build_faiss_index_from_chunks,
    get_similar_chunks,
    answer_question_with_rag,
)

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="PDF Q&A RAG Chatbot",
    page_icon="📄",
    layout="wide",
)

# ----------------- SESSION STATE -----------------
def init_session_state():
    if "chunks" not in st.session_state:
        st.session_state.chunks = []

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

init_session_state()

# ----------------- UI HEADER -----------------
st.title("📄 PDF Q&A Chatbot (RAG + LangChain + Groq)")
st.write(
    "Upload a PDF → split into chunks → build embeddings (FAISS) → "
    "chat with the document using a Groq LLM."
)
st.divider()

# ----------------- STEP 1: PDF UPLOAD -----------------
st.subheader("1️⃣ Upload and Process PDF")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    st.success("✅ PDF uploaded successfully!")

    if st.button("📥 Process PDF"):
        with st.spinner("Reading and splitting PDF..."):
            try:
                chunks = load_and_split_pdf_from_upload(
                    uploaded_file,
                    chunk_size=800,
                    chunk_overlap=200,
                )
                st.session_state.chunks = chunks
                st.session_state.vector_store = None
                st.session_state.chat_history = []
            except Exception as e:
                st.error(f"Error while processing PDF: {e}")
                chunks = []

        if chunks:
            st.success(f"✅ PDF split into {len(chunks)} chunks")

            with st.expander("🔍 Preview first 3 chunks"):
                for i, chunk in enumerate(chunks[:3], start=1):
                    st.markdown(f"**Chunk {i}:**")
                    st.write(chunk)
                    st.divider()
        else:
            st.error("No text could be extracted from this PDF.")
else:
    st.info("👆 Upload a PDF to begin")

st.divider()

# ----------------- STEP 2: BUILD FAISS -----------------
st.subheader("2️⃣ Build FAISS Vector Index")

if not st.session_state.chunks:
    st.warning("Please upload and process a PDF first.")
else:
    if st.session_state.vector_store is None:
        if st.button("⚙️ Build FAISS Index"):
            with st.spinner("Creating embeddings..."):
                try:
                    st.session_state.vector_store = build_faiss_index_from_chunks(
                        st.session_state.chunks
                    )
                    st.success("✅ FAISS index created")
                except Exception as e:
                    st.error(f"Error building FAISS index: {e}")
    else:
        st.success("✅ FAISS index already exists")

st.divider()

# ----------------- STEP 3: OPTIONAL RETRIEVAL -----------------
st.subheader("3️⃣ Test Retrieval (No LLM)")

test_query = st.text_input("Enter a query to retrieve relevant chunks")

if st.button("🔎 Retrieve Chunks"):
    if not st.session_state.vector_store:
        st.error("Build the FAISS index first.")
    elif not test_query.strip():
        st.error("Please enter a query.")
    else:
        with st.spinner("Searching vector store..."):
            docs = get_similar_chunks(
                st.session_state.vector_store,
                test_query,
                k=4,
            )

        if docs:
            st.success(f"Found {len(docs)} relevant chunks")
            for i, doc in enumerate(docs, start=1):
                st.markdown(f"**Chunk {i}:**")
                st.write(doc.page_content)
                st.divider()
        else:
            st.warning("No relevant chunks found")

st.divider()

# ----------------- STEP 4: CHAT WITH PDF -----------------
st.subheader("4️⃣ Chat with the PDF")

if not st.session_state.vector_store:
    st.warning("Upload a PDF and build the FAISS index to start chatting.")
    st.stop()

# Render existing chat history FIRST
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
user_query = st.chat_input("Ask a question about the PDF")

if user_query:
    # User message
    st.session_state.chat_history.append(
        {"role": "user", "content": user_query}
    )
    with st.chat_message("user"):
        st.write(user_query)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = answer_question_with_rag(
                st.session_state.vector_store,
                user_query,
                chat_history=st.session_state.chat_history,
            )
            st.write(answer)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer}
    )

# Clear chat
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

# ----------------- DEBUG -----------------
st.divider()
st.caption(f"Chunks loaded: {len(st.session_state.chunks)}")
st.caption(f"FAISS index ready: {st.session_state.vector_store is not None}")
