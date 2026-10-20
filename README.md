# Assistant d'analyse documentaire (RAG)

Ce projet est une application Streamlit qui permet de poser des questions sur un document PDF chargé au préalable. Les réponses sont générées par un LLM à partir des passages du document les plus pertinents, et chaque réponse affiche les extraits utilisés avec leur numéro de page pour pouvoir la vérifier.

J'ai commencé ce projet pour comprendre en pratique comment fonctionne un pipeline RAG (Retrieval-Augmented Generation), de l'indexation d'un document à la génération de la réponse. Je continuerai de le faire évoluer. A terme je souhaite créer un agent de discussion léger et fiable pouvant être utilisé pour divers domaines.
*

## Fonctionnement

1. Le PDF est lu page par page, puis découpé en morceaux d'environ 500 caractères (avec 50 caractères de chevauchement).
2. Chaque morceau est transformé en vecteur avec le modèle d'embeddings `all-MiniLM-L6-v2` (HuggingFace), puis indexé dans FAISS.
3. Pour une question, le retriever sélectionne 5 passages avec la méthode MMR (Maximal Marginal Relevance), qui évite de renvoyer plusieurs passages quasi identiques.
4. Les passages et l'historique de la conversation sont envoyés au modèle `llama-3.1-8b-instant` via l'API Groq, avec la consigne de ne répondre qu'à partir du contexte fourni.
5. La réponse s'affiche au fur et à mesure (streaming), suivie des sources utilisées.

Le PDF est identifié par un hash MD5 : si le même fichier est chargé à nouveau, l'index n'est pas recalculé.

## Structure du code

```
app.py              interface Streamlit
config.py           paramètres (modèles, découpage, retriever)
rag/loader.py       lecture et découpage du PDF
rag/vectorstore.py  embeddings, index FAISS et retriever
rag/chain.py        prompt, appel au LLM et streaming
```

## Lancer le projet en local

Prérequis : Python 3.11 ou plus récent et une clé API Groq (gratuite sur https://console.groq.com).

```bash
git clone https://github.com/AllanDMB/Analyse-documentaire-RAG-LLM-.git
cd Analyse-documentaire-RAG-LLM-
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
echo "GROQ_API_KEY=votre_cle" > .env
streamlit run app.py
```

Les paramètres (modèle, taille des morceaux, nombre de passages récupérés) se modifient dans `config.py`.

## Limites actuelles

- Un seul document à la fois, et uniquement des PDF contenant du texte (pas de PDF scannés).
- Pas encore d'évaluation chiffrée de la qualité des réponses.

## Pistes d'évolution

- Prise en charge des fichiers Word (.docx) et de plusieurs documents à la fois.
- Métadonnées sur les passages (document, section) pour filtrer la recherche.
- Jeu de questions de test pour mesurer la qualité des réponses et des sources.

## Technologies

Python, Streamlit, LangChain, FAISS, HuggingFace (sentence-transformers), Groq (Llama 3.1).
