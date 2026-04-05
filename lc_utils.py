import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_pdf_from_upload(
    uploaded_file,
    chunk_size: int = 1200,
    chunk_overlap: int = 300,
):
    """
    Loads an uploaded PDF and returns a list of LangChain Document objects.
    Page metadata is preserved. Temp file is always cleaned up.
    """
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = tmp.name

        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return splitter.split_documents(docs)

    finally:
        # FIX: always delete the temp file regardless of success or failure
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)