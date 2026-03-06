"""
Étape 4 — Stockage vectoriel FAISS et recherche sémantique.

Construit un index FAISS à partir des embeddings, et fournit
une interface de recherche par similarité pour le futur RAG.

Usage autonome :
    python -m src.step_4_store.vector_store

Usage depuis le pipeline :
    from src.step_4_store.vector_store import run_store, search
    run_store(embeddings, chunks)
    results = search("Ma question", top_k=5)
"""

import json
import sys
import logging
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (
    FAISS_INDEX_FILE, FAISS_METADATA_FILE, FAISS_TOP_K,
    EMBEDDINGS_OUTPUT_FILE, CHUNKS_OUTPUT_FILE,
    SBERT_MODEL_NAME,
    LOG_LEVEL, LOG_FORMAT, LOG_DATEFMT,
)

logger = logging.getLogger(__name__)


# ── Construction de l'index ──────────────────────────────────────────────────


def build_faiss_index(embeddings: np.ndarray) -> "faiss.IndexFlatIP":
    """
    Construit un index FAISS à partir d'embeddings.

    Utilise IndexFlatIP (Inner Product) avec normalisation L2 préalable
    → équivalent à la cosine similarity.
    """
    import faiss

    # Normaliser les vecteurs pour que le dot product = cosine similarity
    embeddings_normalized = embeddings.copy().astype("float32")
    faiss.normalize_L2(embeddings_normalized)

    # Créer l'index flat (recherche exacte, pas d'approximation)
    dimension = embeddings_normalized.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_normalized)

    logger.info("   Index FAISS créé : %d vecteurs, dimension=%d",
                index.ntotal, dimension)
    return index


def save_index(index: "faiss.IndexFlatIP", chunks: list[dict],
               index_file: Path | None = None,
               metadata_file: Path | None = None) -> None:
    """Sauvegarde l'index FAISS et les métadonnées associées."""
    import faiss

    index_file = index_file or FAISS_INDEX_FILE
    metadata_file = metadata_file or FAISS_METADATA_FILE

    index_file.parent.mkdir(parents=True, exist_ok=True)

    # Sauvegarder l'index
    faiss.write_index(index, str(index_file))
    logger.info("   💾 Index FAISS sauvegardé → %s", index_file)

    # Sauvegarder les métadonnées (texte + metadata de chaque chunk)
    metadata = [{"text": c["text"], "metadata": c["metadata"]} for c in chunks]
    metadata_file.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("   💾 Métadonnées sauvegardées → %s", metadata_file)


def load_index(index_file: Path | None = None,
               metadata_file: Path | None = None) -> tuple:
    """
    Charge l'index FAISS et les métadonnées sauvegardées.

    Retourne (index, metadata_list).
    """
    import faiss

    index_file = index_file or FAISS_INDEX_FILE
    metadata_file = metadata_file or FAISS_METADATA_FILE

    if not index_file.exists() or not metadata_file.exists():
        logger.error("Index FAISS ou métadonnées introuvables.")
        logger.error("Exécutez d'abord le pipeline complet.")
        return None, None

    index = faiss.read_index(str(index_file))
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info("   Index FAISS chargé : %d vecteurs", index.ntotal)
    return index, metadata


# ── Recherche sémantique ─────────────────────────────────────────────────────


def search(query: str, top_k: int = FAISS_TOP_K,
           index_file: Path | None = None,
           metadata_file: Path | None = None,
           model_name: str = SBERT_MODEL_NAME) -> list[dict]:
    """
    Recherche les chunks les plus similaires à une requête.

    Paramètres :
        query         : texte de la requête
        top_k         : nombre de résultats à retourner
        index_file    : chemin de l'index FAISS
        metadata_file : chemin des métadonnées

    Retourne :
        Liste de dicts : {text, metadata, score}
    """
    import faiss
    from sentence_transformers import SentenceTransformer

    # Charger l'index
    index, metadata = load_index(index_file, metadata_file)
    if index is None:
        return []

    # Encoder la requête avec le même modèle que les chunks. 
    model = SentenceTransformer(model_name)
    # Obtenir l'embedding de la question
    query_embedding = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_embedding)

    # Recherche dans l'index
    scores, indices = index.search(query_embedding, top_k)

    # Récupération des segments pertinents
    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        if idx < 0 or idx >= len(metadata):
            continue
        results.append({
            "rank": rank + 1,
            "score": float(score),
            "text": metadata[idx]["text"],
            "metadata": metadata[idx]["metadata"],
        })

    return results


# ── Point d'entrée ───────────────────────────────────────────────────────────


def run_store(embeddings: np.ndarray | None = None,
              chunks: list[dict] | None = None,
              embedding_type: str = "sbert") -> dict:
    """
    Point d'entrée de l'étape 4 : construction du vector store FAISS.

    Paramètres :
        embeddings     : matrice d'embeddings (si None, charge depuis .npz)
        chunks         : liste de chunks (si None, charge depuis JSON)
        embedding_type : "sbert" ou "mistral" (quel embedding utiliser)

    Retourne :
        {"index": faiss.Index, "total_vectors": int}
    """
    logger.info("=" * 65)
    logger.info("🗄️  ÉTAPE 4 — STOCKAGE VECTORIEL (FAISS)")
    logger.info("=" * 65)

    # Charger les embeddings si pas fournis
    if embeddings is None:
        if not EMBEDDINGS_OUTPUT_FILE.exists():
            logger.error("Fichier d'embeddings introuvable : %s", EMBEDDINGS_OUTPUT_FILE)
            logger.error("Exécutez d'abord l'étape 3 (embeddings).")
            return {"index": None, "total_vectors": 0}

        data = np.load(str(EMBEDDINGS_OUTPUT_FILE))
        if embedding_type in data:
            embeddings = data[embedding_type]
            logger.info("   Embeddings '%s' chargés : shape=%s",
                        embedding_type, embeddings.shape)
        else:
            available = list(data.keys())
            logger.error("Type '%s' introuvable. Disponibles : %s",
                         embedding_type, available)
            return {"index": None, "total_vectors": 0}

    # Charger les chunks si pas fournis
    if chunks is None:
        if not CHUNKS_OUTPUT_FILE.exists():
            logger.error("Fichier de chunks introuvable : %s", CHUNKS_OUTPUT_FILE)
            return {"index": None, "total_vectors": 0}

        with open(CHUNKS_OUTPUT_FILE, "r", encoding="utf-8") as f:
            chunks = json.load(f).get("chunks", [])

    # Vérifier la cohérence
    if len(embeddings) != len(chunks):
        logger.error("Incohérence : %d embeddings ≠ %d chunks",
                     len(embeddings), len(chunks))
        return {"index": None, "total_vectors": 0}

    # Construire l'index FAISS
    index = build_faiss_index(embeddings)
    save_index(index, chunks)

    # ── Test de recherche ────────────────────────────────────────────────
    logger.info("")
    logger.info("🔍 Test de recherche :")
    test_queries = [
        "éclairage public",
        "budget 2024",
        "centre culturel innovant",
    ]

    for query in test_queries:
        results = search(query, top_k=3)
        logger.info("")
        logger.info("   🔎 Requête : \"%s\"", query)
        for r in results:
            text_preview = r["text"][:80].replace("\n", " ")
            logger.info("      #%d (%.4f) [%s] %s...",
                        r["rank"], r["score"],
                        r["metadata"]["category"], text_preview)

    logger.info("")
    logger.info("✅ Étape 4 terminée ! Index FAISS prêt.")
    return {"index": index, "total_vectors": index.ntotal}


# ── Exécution autonome ──────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATEFMT)
    run_store()
