import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_pdf_from_upload(uploaded_file,
                                   chunk_size: int = 800,
                                   chunk_overlap: int = 200):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path = tmp.name

    loader = PyPDFLoader(temp_path)
    docs = loader.load()  # list of Documents

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    split_docs = splitter.split_documents(docs)
    chunks = [d.page_content for d in split_docs]
    return chunks
