import textwrap
import json
import math
import re
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


# ===================== EMBEDDINGS =====================
# FIX: cached at module level so the model is only loaded once per process.
# (If using Streamlit, wrap the call site with @st.cache_resource instead.)

@lru_cache(maxsize=1)
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


# ===================== LLM =====================
# FIX: cached so we don't re-instantiate the client on every RAG call.

@lru_cache(maxsize=1)
def get_groq_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=512,
    )


# ===================== VECTOR STORE =====================

def build_faiss_index_from_documents(docs):
    """Build a FAISS index from LangChain Document objects."""
    embeddings = get_embedding_model()
    return FAISS.from_documents(documents=docs, embedding=embeddings)


def retrieve_with_scores(vector_store, query: str, k: int = 4):
    """
    Returns List[(Document, score)].
    Lower L2 distance = better match.
    """
    return vector_store.similarity_search_with_score(query, k=k)


# ===================== CONFIDENCE =====================

def normalize_faiss_score(score: float) -> float:
    """
    Convert FAISS L2 distance to a 0-1 confidence value.

    FIX: The original 1/(1+score) formula produces very low values for
    typical L2 distances (0.5-2.0), making the progress bar almost always
    near-zero. exp(-score) gives a more intuitive curve:
        score=0.0  → 1.00  (perfect match)
        score=0.5  → 0.61
        score=1.0  → 0.37
        score=2.0  → 0.14
    """
    return round(max(0.0, min(1.0, math.exp(-score))), 4)


# ===================== JSON SAFE PARSER =====================

def extract_json(text: str) -> dict:
    # FIX: bare `except:` → `except Exception:` to avoid catching SystemExit etc.
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    return {"answer": text.strip(), "llm_confidence": 0.5}


# ===================== MAIN RAG FUNCTION =====================

def answer_question_with_rag(
    vector_store,
    query: str,
    chat_history: list,
    k: int = 6,
) -> dict:
    if vector_store is None:
        raise ValueError("Vector store is not initialized.")

    results = retrieve_with_scores(vector_store, query, k=k)

    if not results:
        return {
            "answer": "I couldn't find anything relevant in the document.",
            "confidence": 0.0,
            "sources": [],
        }

    sources = []
    retrieval_confidences = []

    for doc, score in results:
        confidence = normalize_faiss_score(score)
        retrieval_confidences.append(confidence)

        page_number = doc.metadata.get("page")
        if page_number is not None:
            page_number += 1  # convert to 1-based

        sources.append({
            "page": page_number,
            "content": doc.page_content,
            "retrieval_score": round(confidence, 2),
        })

    context = "\n\n".join(s["content"] for s in sources)

    # FIX: exclude the most recent message (the current user query) from
    # history, because it is already injected as `User Question` below.
    # Without this, the query appears twice in the prompt.
    prior_history = chat_history[:-1]  # drop the just-appended user turn
    history_lines = []
    for msg in prior_history[-6:]:
        role = msg["role"].capitalize()
        history_lines.append(f"{role}: {msg['content']}")
    history_text = "\n".join(history_lines)

    # FIX: use textwrap.dedent so indentation inside the function body
    # does not bleed into the prompt string sent to the model.
    prompt = textwrap.dedent(f"""
        You are a document-grounded assistant.

        STRICT RULES:
        1. Answer ONLY using the provided document context.
        2. Extract the most direct definition from the document.
        3. Do NOT summarize broadly or add external examples.
        4. If the answer is not explicitly defined, say:
           "The document does not explicitly define this."
        5. Prefer the sentence that directly defines the concept.

        Chat History:
        {history_text}

        Document Context:
        {context}

        User Question:
        {query}

        After answering, rate your confidence from 0 to 1.

        Respond strictly in JSON:
        {{
          "answer": "...",
          "llm_confidence": 0.0
        }}
    """).strip()

    llm = get_groq_llm()
    response = llm.invoke(prompt)
    llm_output = extract_json(response.content)

    llm_confidence = float(llm_output.get("llm_confidence", 0.5))
    avg_retrieval_conf = sum(retrieval_confidences) / len(retrieval_confidences)
    final_confidence = round((avg_retrieval_conf + llm_confidence) / 2, 2)

    return {
        "answer": llm_output.get("answer", ""),
        "confidence": final_confidence,
        "sources": sources,
    }