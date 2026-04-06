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
    get_groq_llm,
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
    # Tracks which filenames have already been processed so re-runs don't
    # re-process the same files on every Streamlit interaction.
    st.session_state.setdefault("processed_file_names", [])
    # Stores one LLM-generated summary string per uploaded filename.
    # Populated once after each new upload; never regenerated on reruns.
    st.session_state.setdefault("pdf_summaries", {})

init_session_state()


# ----------------- SIDEBAR HELPER -----------------

def generate_pdf_summary(file_name: str, all_docs: list) -> str:
    """
    Generate a 2-3 sentence summary for one PDF using its first few chunks.
    Called at most once per file — result is stored in session state.
    """
    file_chunks = [d for d in all_docs if d.metadata.get("source") == file_name]
    sample_text = "\n\n".join(d.page_content for d in file_chunks[:3])
    prompt = (
        "Summarize the following document in 2-3 simple sentences. "
        "Focus on what the document is about. Do not be too detailed.\n\n"
        f"Document:\n{sample_text}"
    )
    try:
        response = get_groq_llm().invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"Summary unavailable ({e})"


# ----------------- SIDEBAR -----------------

with st.sidebar:
    st.title("📂 Uploaded Documents")

    if st.session_state.pdf_summaries:
        for file_name, summary in st.session_state.pdf_summaries.items():
            st.markdown(f"### 📄 {file_name}")
            st.write(summary)
            st.divider()
    else:
        st.info("Upload PDFs to see summaries here.")


# ----------------- HEADER -----------------
st.title("📄 PDF Q&A Chatbot (RAG + Confidence + Citations)")
st.divider()

# ----------------- STEP 1: UPLOAD + AUTO-PROCESS -----------------
st.subheader("1️⃣ Upload PDFs (up to 4)")

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    current_names = sorted(f.name for f in uploaded_files)

    # Only (re-)process when the uploaded file set has actually changed.
    if current_names != st.session_state.processed_file_names:
        st.session_state.docs = []
        st.session_state.vector_store = None
        st.session_state.chat_history = []
        st.session_state.processed_file_names = []  # reset until fully done
        st.session_state.pdf_summaries = {}          # clear stale summaries

        # --- Process all PDFs ---
        all_docs = []
        with st.spinner(f"Reading and splitting {len(uploaded_files)} PDF(s)…"):
            for f in uploaded_files:
                try:
                    docs = load_and_split_pdf_from_upload(
                        f,
                        source_name=f.name,
                        chunk_size=800,
                        chunk_overlap=200,
                    )
                    all_docs.extend(docs)
                except Exception as e:
                    st.error(f"Error processing {f.name}: {e}")

        if not all_docs:
            st.error("No text could be extracted from the uploaded files.")
            st.stop()

        st.session_state.docs = all_docs

        # --- Build FAISS index automatically ---
        with st.spinner("Building FAISS index…"):
            try:
                st.session_state.vector_store = build_faiss_index_from_documents(all_docs)
                # Only mark as done after both steps succeed
                st.session_state.processed_file_names = current_names
            except Exception as e:
                st.error(f"Error building FAISS index: {e}")
                st.stop()

        # --- Generate one summary per PDF (LLM called once per file) ---
        with st.spinner("Summarising documents for sidebar…"):
            for f in uploaded_files:
                if f.name not in st.session_state.pdf_summaries:
                    st.session_state.pdf_summaries[f.name] = generate_pdf_summary(
                        f.name, all_docs
                    )

        # Rerun so the sidebar re-renders with the newly populated summaries.
        # Without this, the sidebar block (which runs before processing) has
        # already painted with an empty pdf_summaries dict.
        st.rerun()

    # Always show a status summary (works on first run and subsequent reruns)
    names_display = ", ".join(current_names)
    total_chunks = len(st.session_state.docs)
    st.success(f"✅ Ready — {len(current_names)} PDF(s) · {total_chunks} chunks · {names_display}")

    with st.expander("🔍 Preview first 3 chunks"):
        for i, doc in enumerate(st.session_state.docs[:3], start=1):
            page = doc.metadata.get("page")
            if page is not None:
                page += 1
            source = doc.metadata.get("source", "—")
            st.markdown(f"**Chunk {i} | {source} | Page {page}:**")
            st.write(doc.page_content[:500])
            st.divider()

else:
    st.info("👆 Upload 1–4 PDFs to begin")
    st.stop()

st.divider()

# ----------------- STEP 2: RETRIEVAL TEST -----------------
st.subheader("2️⃣ Test Retrieval (No LLM)")

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
                source = doc.metadata.get("source", "—")
                st.markdown(f"**Result {i} | {source} | Page {page} | Raw Score: {round(score, 4)}**")
                st.write(doc.page_content[:500])
                st.divider()
        else:
            st.warning("No relevant chunks found.")

st.divider()

# ----------------- STEP 3: CHAT -----------------
st.subheader("3️⃣ Chat with the PDFs")

if not st.session_state.vector_store:
    st.warning("Build the FAISS index first.")
    st.stop()

# Render chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_query = st.chat_input("Ask a question about the PDFs")

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
                    f"**{src['source']} | Page {src['page']} | Retrieval Score: {src['retrieval_score']}**"
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