from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()


def get_secret(key: str) -> str | None:
    """Lit depuis st.secrets (prod) ou .env (local)."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key)



# === LLM ===
GROQ_API_KEY = get_secret("GROQ_API_KEY")
LLM_MODEL = "openai/gpt-oss-120b"
LLM_TEMPERATURE = 0

# === Embeddings ===
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# === Chunking ===
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# === Retriever ===
RETRIEVER_K = 5  # réduit de 10 à 5 pour éviter de noyer le contexte

