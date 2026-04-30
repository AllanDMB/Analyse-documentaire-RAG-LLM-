# 🤖 RAG Chatbot

Un chatbot RAG (Retrieval-Augmented Generation) permettant de poser des questions sur n'importe quel document PDF.

## 🏗️ Architecture

```
rag-chatbot/
├── app.py              # UI Streamlit (couche présentation uniquement)
├── config.py           # Paramètres centralisés
├── rag/
│   ├── loader.py       # Chargement + chunking PDF
│   ├── vectorstore.py  # FAISS + embeddings HuggingFace
│   └── chain.py        # Prompt + LLM Groq + streaming
├── .env.example
├── requirements.txt
└── README.md
```

## ✨ Fonctionnalités

- 📄 Upload de PDF avec détection de changement (pas de réindexation inutile)
- 🔎 Retrieval **MMR** (Maximum Marginal Relevance) pour des résultats diversifiés
- ⚡ **Streaming** de la réponse token par token
- 💬 Mémoire de conversation avec `MessagesPlaceholder` LangChain
- 📚 Sources affichées avec numéro de page
- 🗑️ Reset de conversation en un clic

## 🚀 Installation

### 1. Cloner le repo

```bash
git clone https://github.com/ton-user/rag-chatbot.git
cd rag-chatbot
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les variables d'environnement

```bash
cp .env.example .env
# Édite .env et ajoute ta clé API Groq
```

Obtenir une clé Groq gratuitement : https://console.groq.com

### 5. Lancer l'application

```bash
streamlit run app.py
```

## ⚙️ Configuration

Tous les paramètres sont centralisés dans `config.py` :

| Paramètre | Valeur par défaut | Description |
|---|---|---|
| `LLM_MODEL` | `llama-3.1-8b-instant` | Modèle Groq utilisé |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Modèle d'embeddings HuggingFace |
| `CHUNK_SIZE` | `500` | Taille des chunks en tokens |
| `CHUNK_OVERLAP` | `50` | Chevauchement entre chunks |
| `RETRIEVER_K` | `5` | Nombre de chunks récupérés |

## 🔧 Choix techniques

### MMR vs Similarity Search
Le retriever utilise **Maximum Marginal Relevance** (`search_type="mmr"`) plutôt qu'une simple similarité cosinus. Cela garantit des chunks à la fois **pertinents** ET **diversifiés**, évitant les redondances.

### MessagesPlaceholder
L'historique est géré via `MessagesPlaceholder` LangChain plutôt qu'une concaténation de strings, ce qui permet une gestion propre des rôles et une meilleure utilisation du contexte.

### Hash PDF
Le PDF est hashé (MD5) à l'upload. Si le même fichier est uploadé deux fois, le vectorstore n'est **pas** recréé, ce qui évite un calcul inutile des embeddings.

## 📦 Stack

- [Streamlit](https://streamlit.io/) — UI
- [LangChain](https://langchain.com/) — Orchestration RAG
- [Groq](https://groq.com/) — LLM (Llama 3.1)
- [FAISS](https://github.com/facebookresearch/faiss) — Vector store local
- [HuggingFace](https://huggingface.co/) — Embeddings (all-MiniLM-L6-v2)

## 📝 Licence

MIT
