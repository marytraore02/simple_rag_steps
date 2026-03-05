"""
Étape 3 — Vectorisation (embeddings) des chunks.

Génère les embeddings pour chaque chunk à l'aide de :
  - SBERT (Sentence-BERT)  : modèle local, gratuit, rapide
  - Mistral Embeddings     : API cloud, plus puissant (si clé API disponible)

Inclut aussi l'analyse de similarité et la visualisation PCA.

Usage autonome :
    python -m src.step_3_embed.embedder

Usage depuis le pipeline :
    from src.step_3_embed.embedder import run_embedding
    result = run_embedding(chunks)
"""

import json
import sys
import logging
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (
    CHUNKS_OUTPUT_FILE, EMBEDDINGS_OUTPUT_FILE, PLOTS_DIR,
    SBERT_MODEL_NAME, MISTRAL_API_KEY, MISTRAL_EMBED_MODEL, MISTRAL_BATCH_SIZE,
    LOG_LEVEL, LOG_FORMAT, LOG_DATEFMT,
)

logger = logging.getLogger(__name__)


# ── Fonctions d'embedding ────────────────────────────────────────────────────


def load_chunks(chunks_file: Path) -> list[dict]:
    """Charge les chunks depuis le fichier JSON."""
    with open(chunks_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("chunks", [])


def embed_with_sbert(texts: list[str],
                     model_name: str = SBERT_MODEL_NAME) -> np.ndarray:
    """
    Génère les embeddings avec un modèle SBERT local.

    Modèle par défaut : all-MiniLM-L6-v2
      - Dimension : 384
      - Rapide et léger
      - Bon pour la similarité sémantique
    """
    from sentence_transformers import SentenceTransformer

    logger.info("🧠 Génération des embeddings SBERT (%s)...", model_name)
    logger.info("   %d textes à vectoriser", len(texts))

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    logger.info("   ✅ Embeddings SBERT : shape=%s", embeddings.shape)
    return embeddings


def embed_with_mistral(texts: list[str],
                       api_key: str | None = MISTRAL_API_KEY) -> np.ndarray | None:
    """
    Génère les embeddings via l'API Mistral.

    Modèle : mistral-embed
      - Dimension : 1024
      - Plus puissant, nécessite une clé API
    """
    if not api_key:
        logger.warning("⚠️ Pas de clé API Mistral → embeddings Mistral ignorés")
        return None

    from mistralai import Mistral

    logger.info("🧠 Génération des embeddings Mistral (%s)...", MISTRAL_EMBED_MODEL)
    logger.info("   %d textes à vectoriser (batch_size=%d)", len(texts), MISTRAL_BATCH_SIZE)

    client = Mistral(api_key=api_key)
    all_embeddings = []

    for i in range(0, len(texts), MISTRAL_BATCH_SIZE):
        batch = texts[i:i + MISTRAL_BATCH_SIZE]
        response = client.embeddings.create(
            model=MISTRAL_EMBED_MODEL,
            inputs=batch,
        )
        all_embeddings.extend([d.embedding for d in response.data])
        logger.info("   Batch %d/%d traité",
                     i // MISTRAL_BATCH_SIZE + 1,
                     (len(texts) - 1) // MISTRAL_BATCH_SIZE + 1)

    embeddings = np.array(all_embeddings)
    logger.info("   ✅ Embeddings Mistral : shape=%s", embeddings.shape)
    return embeddings


# ── Analyse & Visualisation ──────────────────────────────────────────────────


def analyze_similarity(embeddings: np.ndarray, chunks: list[dict],
                       model_name: str, top_n: int = 10,
                       threshold: float = 0.8) -> None:
    """Affiche les paires de chunks les plus similaires."""
    from sklearn.metrics.pairwise import cosine_similarity

    logger.info("")
    logger.info("📐 Analyse de similarité (%s, seuil=%.2f)", model_name, threshold)

    sim_matrix = cosine_similarity(embeddings)
    pairs = []

    for i in range(len(sim_matrix)):
        for j in range(i + 1, len(sim_matrix)):
            if sim_matrix[i, j] >= threshold:
                pairs.append((i, j, sim_matrix[i, j]))

    pairs.sort(key=lambda x: x[2], reverse=True)

    if not pairs:
        logger.info("   Aucune paire avec similarité ≥ %.2f", threshold)
        return

    logger.info("   %d paires avec similarité ≥ %.2f", len(pairs), threshold)
    for i, j, sim in pairs[:top_n]:
        src_i = chunks[i]["metadata"]["source"]
        src_j = chunks[j]["metadata"]["source"]
        logger.info("   %.4f : '%s' (chunk %d) ↔ '%s' (chunk %d)",
                     sim, src_i, chunks[i]["metadata"]["chunk_index"],
                     src_j, chunks[j]["metadata"]["chunk_index"])


def plot_embeddings(embeddings: np.ndarray, chunks: list[dict],
                    title: str, output_file: Path) -> None:
    """Réduit avec PCA et visualise les embeddings en 2D."""
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    logger.info("📊 Génération du plot : %s", output_file.name)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)

    categories = [c["metadata"]["category"] for c in chunks]

    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "category": categories,
    })

    plt.figure(figsize=(14, 9))
    sns.scatterplot(data=df, x="x", y="y", hue="category", style="category",
                    s=80, alpha=0.8)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(f"PCA 1 (var. expliquée : {pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PCA 2 (var. expliquée : {pca.explained_variance_ratio_[1]:.1%})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150)
    plt.close()
    logger.info("   ✅ Plot sauvegardé → %s", output_file)


# ── Point d'entrée ───────────────────────────────────────────────────────────


def run_embedding(chunks: list[dict] | None = None,
                  chunks_file: Path | None = None,
                  output_file: Path | None = None,
                  use_mistral: bool = True,
                  use_sbert: bool = True,
                  visualize: bool = True) -> dict:
    """
    Point d'entrée de l'étape 3 : génération des embeddings.

    Paramètres :
        chunks      : liste de chunks (si None, charge depuis le fichier JSON)
        chunks_file : chemin du fichier JSON des chunks
        output_file : fichier .npz de sortie pour les embeddings
        use_mistral : génère les embeddings Mistral (si clé API dispo)
        use_sbert   : génère les embeddings SBERT
        visualize   : génère les plots PCA

    Retourne :
        {
            "sbert_embeddings": np.ndarray | None,
            "mistral_embeddings": np.ndarray | None,
            "chunks": list[dict],
            "texts": list[str],
        }
    """
    chunks_file = chunks_file or CHUNKS_OUTPUT_FILE
    output_file = output_file or EMBEDDINGS_OUTPUT_FILE

    logger.info("=" * 65)
    logger.info("🧠 ÉTAPE 3 — VECTORISATION (EMBEDDINGS)")
    logger.info("=" * 65)

    # Charger les chunks si pas fournis directement
    if chunks is None:
        if not chunks_file.exists():
            logger.error("Fichier de chunks introuvable : %s", chunks_file)
            logger.error("Exécutez d'abord l'étape 2 (chunking).")
            return {"sbert_embeddings": None, "mistral_embeddings": None,
                    "chunks": [], "texts": []}
        chunks = load_chunks(chunks_file)

    texts = [c["text"] for c in chunks]
    logger.info("   %d chunks à vectoriser", len(texts))

    result = {
        "sbert_embeddings": None,
        "mistral_embeddings": None,
        "chunks": chunks,
        "texts": texts,
    }

    # ── SBERT ────────────────────────────────────────────────────────────
    if use_sbert:
        sbert_emb = embed_with_sbert(texts)
        result["sbert_embeddings"] = sbert_emb

        if visualize:
            plot_embeddings(
                sbert_emb, chunks,
                f"Embeddings SBERT ({SBERT_MODEL_NAME}) — {len(chunks)} chunks",
                PLOTS_DIR / "sbert_embeddings.png",
            )

        analyze_similarity(sbert_emb, chunks, "SBERT")

    # ── Mistral ──────────────────────────────────────────────────────────
    if use_mistral:
        mistral_emb = embed_with_mistral(texts)
        result["mistral_embeddings"] = mistral_emb

        if mistral_emb is not None and visualize:
            plot_embeddings(
                mistral_emb, chunks,
                f"Embeddings Mistral ({MISTRAL_EMBED_MODEL}) — {len(chunks)} chunks",
                PLOTS_DIR / "mistral_embeddings.png",
            )

            analyze_similarity(mistral_emb, chunks, "Mistral")

    # ── Sauvegarde des embeddings ────────────────────────────────────────
    save_data = {}
    if result["sbert_embeddings"] is not None:
        save_data["sbert"] = result["sbert_embeddings"]
    if result["mistral_embeddings"] is not None:
        save_data["mistral"] = result["mistral_embeddings"]

    if save_data:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(output_file), **save_data)
        logger.info("💾 Embeddings sauvegardés → %s", output_file)

    logger.info("✅ Étape 3 terminée !")
    return result


# ── Exécution autonome ──────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATEFMT)
    run_embedding()
