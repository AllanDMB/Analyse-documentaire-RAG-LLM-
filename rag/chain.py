from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE
from typing import List, Generator


SYSTEM_PROMPT = """Tu es un assistant expert en analyse de documents.
Réponds uniquement à partir du contexte fourni. Si l'information n'est pas dans le contexte, dis-le clairement.
Sois précis, structuré et concis.

Contexte extrait du document :
{context}"""


def get_llm() -> ChatGroq:
    """Initialise et retourne le modèle LLM Groq."""
    if not GROQ_API_KEY:
        raise ValueError(
            "Clé API Groq manquante. Ajoute GROQ_API_KEY dans ton fichier .env"
        )
    return ChatGroq(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=GROQ_API_KEY,
    )


def format_docs(docs: List[Document]) -> str:
    """Formate les documents récupérés en un seul bloc de contexte."""
    return "\n\n---\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}" for doc in docs
    )


def format_history(messages: List[dict]) -> List[HumanMessage | AIMessage]:
    """
    Convertit l'historique Streamlit en messages LangChain.

    Args:
        messages: Liste de dicts {"role": ..., "content": ...}

    Returns:
        Liste de HumanMessage / AIMessage.
    """
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history


def build_chain(context: str):
    """
    Construit la chain RAG avec historique.

    Args:
        context: Contexte formaté issu du retriever.

    Returns:
        Chain LangChain (prompt | llm | parser).
    """
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()
    return chain


def stream_response(question: str, docs: List[Document], history: List[dict]) -> Generator:
    """
    Stream la réponse du LLM token par token.

    Args:
        question: Question de l'utilisateur.
        docs: Documents récupérés par le retriever.
        history: Historique de conversation.

    Yields:
        Tokens de la réponse.
    """
    context = format_docs(docs)
    lc_history = format_history(history)
    chain = build_chain(context)

    yield from chain.stream({
        "context": context,
        "history": lc_history,
        "question": question,
    })
