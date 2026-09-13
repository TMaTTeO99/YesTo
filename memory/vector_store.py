import os
import uuid
from datetime import datetime
from typing import Optional

from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector as PGVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_MEMORY = "memoria_conversazionale"
COLLECTION_DOCS = "documenti_esterni"

_embeddings: Optional[HuggingFaceEmbeddings] = None
_store_memory: Optional[PGVectorStore] = None
_store_docs: Optional[PGVectorStore] = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    """
        Function to create the embedding interface.
        HugginFaceEmbeddings is used to create vector representations of text.
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def _get_store(collection: str, connection_string: str) -> PGVectorStore:
    """
        Function to wrap the tabel in the DB as a vectore store.
    """
    return PGVector(
        embeddings=_get_embeddings(),
        collection_name=collection,
        connection=connection_string,
        use_jsonb=True,
    )


def init_vector_stores(connection_string: str):
    """
        Function created to initialize the vectore store at the start of the application.
    """
    global _store_memory, _store_docs
    _store_memory = _get_store(COLLECTION_MEMORY, connection_string)
    _store_docs = _get_store(COLLECTION_DOCS, connection_string)


def _ensure_initialized():
    """
        Function to check if the vector store is initialized before using it.
    """
    if _store_memory is None or _store_docs is None:
        raise RuntimeError("Vector store non inizializzato. Chiama init_vector_stores() prima.")


def save_conversation(session_id: str, human_msg: str, ai_msg: str):
    """
        Function to save the conversation in the vectore store.
        Each message are saved as a Document with real information about the session
        and the metadata of the message.
    """
    _ensure_initialized()
    doc = Document(
        page_content=f"Utente: {human_msg}\nAssistente: {ai_msg}",
        metadata={
            "session_id": session_id,
            "type": "conversazione",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )
    _store_memory.add_documents([doc], ids=[str(uuid.uuid4())])


def search_memory(query: str, session_id: Optional[str] = None, k: int = 3) -> list[Document]:
    """
        Function to search in the conversation memory.
    """
    _ensure_initialized()
    filter_dict = {"session_id": session_id} if session_id else None
    return _store_memory.similarity_search(query, k=k, filter=filter_dict)


def ingest_text(text: str, source: str, extra_metadata: Optional[dict] = None):
    """
        Function to save a text in the vectore store.
    """
    _ensure_initialized()
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    metadata = {"source": source, "type": "documento", **(extra_metadata or {})}
    docs = [Document(page_content=c, metadata=metadata) for c in chunks]
    ids = [str(uuid.uuid4()) for _ in docs]
    _store_docs.add_documents(docs, ids=ids)
    return len(docs)


def ingest_pdf(pdf_path: str) -> int:
    """
        Function to load a PDF and save it in the vector store.
    """
    _ensure_initialized()
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    for chunk in chunks:
        chunk.metadata["source"] = os.path.basename(pdf_path)
        chunk.metadata["type"] = "documento"

    ids = [str(uuid.uuid4()) for _ in chunks]
    _store_docs.add_documents(chunks, ids=ids)
    return len(chunks)

def search_documents(query: str, k: int = 3) -> list[Document]:
    _ensure_initialized()
    return _store_docs.similarity_search(query, k=k)
