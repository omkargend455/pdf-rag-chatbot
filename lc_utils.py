import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_pdf_from_upload(
    uploaded_file,
    chunk_size: int = 1200,
    chunk_overlap: int = 300,
    source_name: str = None,
):
    """
    Loads an uploaded PDF and returns a list of LangChain Document objects.
    Page metadata is preserved. If source_name is provided it is injected
    into every chunk's metadata so multi-PDF setups can trace each chunk
    back to its originating file.
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
        split_docs = splitter.split_documents(docs)

        # Inject source filename so every chunk carries its origin.
        if source_name:
            for doc in split_docs:
                doc.metadata["source"] = source_name

        return split_docs

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)