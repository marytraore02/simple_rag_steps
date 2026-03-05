"""
Module de recherche contextuelle (RAG).

Gère la connexion à la base vectorielle FAISS et la recherche
de documents pertinents pour enrichir les réponses de l'assistant.
"""

import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Ajouter le répertoire parent au PYTHONPATH pour importer 'src'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from src.step_4_store.vector_store import search
    FAISS_AVAILABLE = True
    logger.info("Module de recherche FAISS chargé avec succès.")
except ImportError as e:
    logger.warning(f"Impossible d'importer le module de recherche : {e}")
    FAISS_AVAILABLE = False


def obtenir_contexte(question, top_k=3):
    """
    Cherche dans la base de données vectorielle les documents les plus pertinents.

    Args:
        question (str) : question de l'utilisateur
        top_k (int)    : nombre de résultats à retourner

    Returns:
        tuple : (contexte_texte, sources)
            - contexte_texte (str) : extraits de documents formatés
            - sources (list[str])  : noms des fichiers sources
    """
    if not FAISS_AVAILABLE:
        return "", []

    try:
        resultats = search(question, top_k=top_k)

        contexte_texte = ""
        sources = []

        for r in resultats:
            contexte_texte += (
                f"---\n"
                f"Document: {r['metadata']['source']}\n"
                f"Extrait: {r['text']}\n"
            )
            # Garder juste le nom du fichier (sans le chemin du dossier)
            source_nom = r['metadata']['source'].split('/')[-1]
            if source_nom not in sources:
                sources.append(source_nom)

        return contexte_texte, sources

    except Exception as e:
        logger.error(f"Erreur lors de la recherche dans l'index : {e}")
        return "", []
