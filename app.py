import streamlit as st

from rag.loader import get_pdf_hash, load_and_split_pdf_from_bytes
from rag.vectorstore import build_vectorstore, get_retriever
from rag.chain import stream_response


# =========================
# ⚙️ Config page
# =========================
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 RAG Chatbot")
st.caption("Pose des questions sur ton document PDF")

# =========================
# 📂 Sidebar - Upload PDF
# =========================
with st.sidebar:
    st.header("📂 Document")
    uploaded_file = st.file_uploader("Upload un PDF", type="pdf")

    if uploaded_file:
        file_bytes = uploaded_file.read()
        pdf_hash = get_pdf_hash(file_bytes)

        # Recharger le vectorstore uniquement si le PDF a changé
        if st.session_state.get("pdf_hash") != pdf_hash:
            with st.spinner("⚙️ Indexation du document..."):
                chunks = load_and_split_pdf_from_bytes(file_bytes, uploaded_file.name)
                db = build_vectorstore(chunks)

                st.session_state.pdf_hash = pdf_hash
                st.session_state.db = db
                st.session_state.messages = []  # reset historique sur nouveau doc

            st.success(f"✅ {len(chunks)} chunks indexés")

        st.info(f"📄 **{uploaded_file.name}**")

    # Reset conversation
    if st.button("🗑️ Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()

# =========================
# 💬 Zone de chat
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# =========================
# ✍️ Input utilisateur
# =========================
if prompt := st.chat_input("Pose ta question sur le document..."):

    if "db" not in st.session_state:
        st.warning("⚠️ Upload un PDF d'abord.")
        st.stop()

    # Afficher message user
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # =========================
    # 🔎 Retrieval
    # =========================
    retriever = get_retriever(st.session_state.db)
    retrieved_docs = retriever.invoke(prompt)  # variable distincte de 'docs' du loader

    # =========================
    # ⚡ Réponse streamée
    # =========================
    with st.chat_message("assistant"):
        response = st.write_stream(
            stream_response(
                question=prompt,
                docs=retrieved_docs,
                history=st.session_state.messages[:-1],  # exclure le message user courant
            )
        )

        # Sources
        with st.expander("📚 Sources utilisées"):
            for i, doc in enumerate(retrieved_docs):
                page = doc.metadata.get("page", "?")
                st.markdown(f"**Source {i+1} — Page {page}**")
                st.caption(doc.page_content)

    st.session_state.messages.append({"role": "assistant", "content": response})