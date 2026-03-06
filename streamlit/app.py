"""
Application Streamlit — Assistant Virtuel de la Mairie.

Ce fichier est l'orchestrateur principal. Il importe les modules :
  - llm_config  : gestion du client LLM et génération de réponse
  - prompts     : prompt système et construction des messages
  - rag_context : recherche dans la base vectorielle FAISS
"""

import streamlit as st
import logging

from llm_config import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    init_client,
    generer_reponse,
)
from prompts import (
    MESSAGE_ACCUEIL,
    construire_prompt_session,
    construire_prompt_rag,
)
from rag_context import obtenir_contexte

# Configuration du logging
logging.basicConfig(level=logging.INFO)

# ── Configuration de la page ─────────────────────────────────────────────────
st.set_page_config(page_title="Assistant Mairie", page_icon="🏛️")

# ── Initialisation du client LLM ─────────────────────────────────────────────
try:
    client = init_client()
except ValueError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Erreur lors de l'initialisation du client Mistral : {e}")
    st.stop()

# ── Sidebar : choix du modèle ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    # Sélection du modèle
    model_labels = list(AVAILABLE_MODELS.keys())
    default_index = list(AVAILABLE_MODELS.values()).index(DEFAULT_MODEL)

    selected_label = st.selectbox(
        "Modèle LLM",
        model_labels,
        index=default_index,
        help="Choisissez le modèle de langage à utiliser pour les réponses.",
    )
    model = AVAILABLE_MODELS[selected_label]

    st.caption(f"📡 Modèle actif : `{model}`")

    st.divider()

    # Bouton d'effacement
    # st.session_stateest un dictionnaire spécial de Streamlit qui persiste entre les rechargements de page
    if st.button("🗑️ Effacer la conversation", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": MESSAGE_ACCUEIL}
        ]
        st.rerun()

# ── Initialisation de l'historique ────────────────────────────────────────────
# Nous vérifions si la clé "messages" existe déjà, sinon nous l'initialisons
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": MESSAGE_ACCUEIL}
    ]

# ── Interface principale ─────────────────────────────────────────────────────
st.title("🏛️ Assistant Virtuel de la Mairie")
st.caption(f"Trifouillis-sur-Loire — Modèle : {model}")

# Affichage de l'historique des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ── Traitement de la question utilisateur ─────────────────────────────────────
if prompt := st.chat_input("Posez votre question ici..."):
    # Ajouter le message de l'utilisateur à l'historique
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Affichage du message de l'utilisateur
    with st.chat_message("user"):
        st.write(prompt)

    # Recherche de contexte RAG
    contexte_texte, sources = obtenir_contexte(prompt)

    # Construction du prompt (avec ou sans contexte RAG)
    # Construit un prompt enrichi avec les segments pertinents et l'historique récent
    if contexte_texte:
        prompt_messages = construire_prompt_rag(
            st.session_state.messages, prompt, contexte_texte
        )
    else:
        prompt_messages = construire_prompt_session(st.session_state.messages)

    # Génération de la réponse
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.text("...")

        response_content = generer_reponse(client, model, prompt_messages)
        # Affichage de la réponse
        message_placeholder.write(response_content)

        # Affichage des sources
        if sources:
            st.caption(f"📚 **Sources :** {', '.join(sources)}")
            response_history = (
                response_content + f"\n\n*Sources : {', '.join(sources)}*"
            )
        else:
            response_history = response_content

    # Sauvegarder la réponse dans l'historique
    st.session_state.messages.append(
        {"role": "assistant", "content": response_history}
    )