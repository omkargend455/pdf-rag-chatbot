import os
import streamlit as st

# ----------------- API KEY -----------------
try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

# ----------------- LOCAL IMPORTS -----------------
from lc_utils import load_and_split_pdf_from_upload
from rag_utils import (
    build_faiss_index_from_documents,
    get_embedding_model,
    retrieve_with_scores,
    answer_question_with_rag,
)

# FIX: cache the embedding model as a Streamlit resource so it is only
# loaded once per server process, not on every script re-run.
get_embedding_model = st.cache_resource(get_embedding_model)

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="PDF Q&A RAG Chatbot",
    page_icon="📄",
    layout="wide",
)

# ----------------- SESSION STATE -----------------
def init_session_state():
    st.session_state.setdefault("docs", [])
    st.session_state.setdefault("vector_store", None)
    st.session_state.setdefault("chat_history", [])

init_session_state()

# ----------------- HEADER -----------------
st.title("📄 PDF Q&A Chatbot (RAG + Confidence + Citations)")
st.divider()

# ----------------- STEP 1: PDF UPLOAD -----------------
st.subheader("1️⃣ Upload and Process PDF")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    st.success("✅ PDF uploaded successfully!")

    if st.button("📥 Process PDF"):
        with st.spinner("Reading and splitting PDF..."):
            try:
                docs = load_and_split_pdf_from_upload(
                    uploaded_file,
                    chunk_size=800,
                    chunk_overlap=200,
                )
                # FIX: update session state only after a successful load
                st.session_state.docs = docs
                st.session_state.vector_store = None
                st.session_state.chat_history = []
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
                docs = []

        if docs:
            st.success(f"✅ PDF split into {len(docs)} chunks")
            with st.expander("🔍 Preview first 3 chunks"):
                for i, doc in enumerate(docs[:3], start=1):
                    page = doc.metadata.get("page")
                    if page is not None:
                        page += 1
                    st.markdown(f"**Chunk {i} | Page {page}:**")
                    st.write(doc.page_content[:500])
                    st.divider()
        else:
            st.error("No text extracted from PDF.")
else:
    st.info("👆 Upload a PDF to begin")

st.divider()

# ----------------- STEP 2: BUILD FAISS -----------------
st.subheader("2️⃣ Build FAISS Vector Index")

# FIX: use st.stop() to prevent later sections from rendering when prereqs
# are unmet, instead of relying only on st.warning() and manual checks.
if not st.session_state.docs:
    st.warning("Process a PDF first.")
    st.stop()

if st.session_state.vector_store is None:
    if st.button("⚙️ Build FAISS Index"):
        with st.spinner("Creating embeddings..."):
            try:
                st.session_state.vector_store = build_faiss_index_from_documents(
                    st.session_state.docs
                )
                st.success("✅ FAISS index created")
            except Exception as e:
                st.error(f"Error building FAISS index: {e}")
else:
    st.success("✅ FAISS index already exists")

st.divider()

# ----------------- STEP 3: RETRIEVAL TEST -----------------
st.subheader("3️⃣ Test Retrieval (No LLM)")

test_query = st.text_input("Enter a query to retrieve relevant chunks")

if st.button("🔎 Retrieve Chunks"):
    if not st.session_state.vector_store:
        st.error("Build the FAISS index first.")
    elif not test_query.strip():
        st.error("Please enter a query.")
    else:
        with st.spinner("Searching..."):
            results = retrieve_with_scores(
                st.session_state.vector_store,
                test_query,
                k=4,
            )

        if results:
            for i, (doc, score) in enumerate(results, start=1):
                page = doc.metadata.get("page")
                if page is not None:
                    page += 1
                st.markdown(f"**Result {i} | Page {page} | Raw Score: {round(score, 4)}**")
                st.write(doc.page_content[:500])
                st.divider()
        else:
            st.warning("No relevant chunks found.")

st.divider()

# ----------------- STEP 4: CHAT -----------------
st.subheader("4️⃣ Chat with the PDF")

if not st.session_state.vector_store:
    st.warning("Build the FAISS index first.")
    st.stop()

# Render chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_query = st.chat_input("Ask a question about the PDF")

if user_query:
    # Append user message first so it appears in the UI immediately
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # NOTE: answer_question_with_rag already strips the last history
            # entry (the current user turn) to avoid prompt duplication.
            result = answer_question_with_rag(
                st.session_state.vector_store,
                user_query,
                chat_history=st.session_state.chat_history,
            )

        st.write(result["answer"])

        st.progress(float(result["confidence"]))
        st.caption(f"Confidence Score: {result['confidence']}")

        with st.expander("📚 Sources"):
            for src in result["sources"]:
                st.markdown(
                    f"**Page {src['page']} | Retrieval Score: {src['retrieval_score']}**"
                )
                st.write(src["content"][:500])
                st.divider()

    st.session_state.chat_history.append(
        {"role": "assistant", "content": result["answer"]}
    )

# Clear chat button
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

# ----------------- DEBUG -----------------
st.divider()
st.caption(f"Documents loaded: {len(st.session_state.docs)}")
st.caption(f"FAISS ready: {st.session_state.vector_store is not None}")