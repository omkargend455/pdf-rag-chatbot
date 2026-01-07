# rag_utils.py

from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq  # NEW


def get_embedding_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
    )
    return embeddings


def build_faiss_index_from_chunks(chunks: List[str]):
    embeddings = get_embedding_model()
    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings,
    )
    return vector_store


def get_similar_chunks(vector_store, query: str, k: int = 4):
    docs = vector_store.similarity_search(query, k=k)
    return docs


# ============== NEW: LLM + RAG ANSWER FUNCTION ==============

def get_groq_llm():
    """
    Returns a Groq Llama model via LangChain.
    Requires GROQ_API_KEY env variable to be set.
    """
    llm = ChatGroq(
        model="llama-3.1-8b-instant",  # or llama-3.3-70b-versatile for higher quality
        temperature=0.2,
        max_tokens=512,
    )
    return llm


def answer_question_with_rag(
    vector_store,
    query: str,
    chat_history: list,
    k: int = 4,
) -> str:
    if vector_store is None:
        raise ValueError("Vector store is not initialized.")

    # 1. Retrieve relevant chunks
    docs = get_similar_chunks(vector_store, query, k=k)
    if not docs:
        return "I couldn't find anything relevant in the document."

    context = "\n\n".join([d.page_content for d in docs])

    # 2. Build chat history string (memory)
    history_text = ""
    for msg in chat_history[-6:]:  # limit memory
        role = msg["role"].capitalize()
        history_text += f"{role}: {msg['content']}\n"

    # 3. Prompt (Memory + RAG)
    prompt = f"""
You are a helpful assistant answering questions based ONLY on the document context.
Use the chat history to understand follow-up questions.
If the answer is not in the document, say you don't know.

Chat History:
-------------
{history_text}
-------------

Document Context:
-----------------
{context}
-----------------

User Question:
{query}

Answer clearly and concisely:
"""

    llm = get_groq_llm()
    response = llm.invoke(prompt)
    return response.content

