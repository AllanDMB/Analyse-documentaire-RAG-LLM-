from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from config import EMBEDDING_MODEL, RETRIEVER_K
from typing import List


def get_embeddings() -> HuggingFaceEmbeddings:
    """Initialise et retourne le modèle d'embeddings."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vectorstore(chunks: List[Document]) -> FAISS:
    """
    Crée un vectorstore FAISS en mémoire uniquement (rien n'est écrit sur le disque).

    Args:
        chunks: Documents découpés par le loader.

    Returns:
        Instance FAISS prête à l'emploi.
    """
    embeddings = get_embeddings()
    db = FAISS.from_documents(chunks, embeddings)
    return db  # pas de save_local → 100% en mémoire


def get_retriever(db: FAISS) -> VectorStoreRetriever:
    """
    Retourne un retriever configuré depuis le vectorstore.

    Args:
        db: Instance FAISS.

    Returns:
        Retriever LangChain.
    """
    return db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVER_K, "fetch_k": RETRIEVER_K * 2},
    )