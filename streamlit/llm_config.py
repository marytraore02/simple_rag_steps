"""
Module de configuration des LLMs.

Centralise la gestion des modèles de langage :
  - Liste des modèles disponibles
  - Initialisation du client Mistral
  - Génération de réponses

Pour changer de modèle, modifier DEFAULT_MODEL ou ajouter
de nouveaux modèles dans AVAILABLE_MODELS.
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ── Modèles disponibles ─────────────────────────────────────────────────────
# Clé = nom affiché dans l'interface, Valeur = identifiant API
AVAILABLE_MODELS = {
    "Mistral Small (rapide, économique)": "mistral-small-latest",
    "Mistral Large (puissant, précis)": "mistral-large-latest",
}

# Modèle par défaut (doit correspondre à une valeur dans AVAILABLE_MODELS)
DEFAULT_MODEL = "mistral-large-latest"


# ── Paramètres de génération ─────────────────────────────────────────────────

GENERATION_PARAMS = {
    "temperature": 0.2,      # Factuel, peu créatif
    "top_p": 0.9,            # Cohérent, filtre les options improbables
    "max_tokens": 500,       # Réponses concises
}


# ── Client API ───────────────────────────────────────────────────────────────


def get_api_key():
    """Récupère la clé API Mistral depuis les variables d'environnement."""
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError(
            "Clé API Mistral non trouvée. "
            "Veuillez définir la variable d'environnement MISTRAL_API_KEY."
        )
    return api_key


def init_client():
    """
    Initialise et retourne le client Mistral.

    Retourne :
        Mistral : instance du client API
    """
    from mistralai import Mistral

    api_key = get_api_key()
    client = Mistral(api_key=api_key)
    logger.info("Client Mistral initialisé avec succès.")
    return client


def generer_reponse(client, model, prompt_messages):
    """
    Appelle l'API Mistral pour générer une réponse.

    Args:
        client         : instance du client Mistral
        model (str)    : identifiant du modèle à utiliser
        prompt_messages (list) : messages formatés à envoyer à l'API

    Returns:
        str : le contenu de la réponse générée ou un message d'erreur
    """
    try:
        response = client.chat.complete(
            model=model,
            messages=prompt_messages,
            **GENERATION_PARAMS,
        )

        if response.choices:
            return response.choices[0].message.content
        else:
            logger.error("L'API Mistral n'a retourné aucun choix.")
            return "Je suis désolé, je n'ai pas pu générer de réponse."

    except Exception as e:
        logger.error(f"Erreur lors de l'appel à l'API Mistral: {e}")
        return "Je suis désolé, j'ai rencontré un problème technique. Veuillez réessayer plus tard."
