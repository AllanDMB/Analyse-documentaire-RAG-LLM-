from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP
from typing import List
from io import BytesIO
import hashlib
import pypdf


def get_pdf_hash(file_bytes: bytes) -> str:
    """Retourne un hash MD5 du contenu PDF pour détecter les changements."""
    return hashlib.md5(file_bytes).hexdigest()


def load_and_split_pdf_from_bytes(file_bytes: bytes, filename: str = "document.pdf") -> List[Document]:
    """
    Charge un PDF depuis des bytes en mémoire (sans écrire sur le disque)
    et le découpe en chunks.

    Args:
        file_bytes: Contenu brut du PDF.
        filename: Nom du fichier pour les métadonnées.

    Returns:
        Liste de Documents LangChain avec métadonnées.
    """
    reader = pypdf.PdfReader(BytesIO(file_bytes))

    docs = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():  # ignorer les pages vides
            docs.append(Document(
                page_content=text,
                metadata={"source": filename, "page": page_num + 1},
            ))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    return chunks