"""
Pipeline RAG — Orchestrateur principal.

Exécute les étapes du pipeline dans l'ordre :

  1. EXTRACTION   : Documents sources → Markdown (Docling + Whisper)
  2. CHUNKING     : Markdown → Chunks avec chevauchement
  3. EMBEDDING    : Chunks → Vecteurs (SBERT + Mistral)
  4. STOCKAGE     : Vecteurs → Index FAISS (recherche sémantique)

Usage :
    python main.py                    # Pipeline complet (étapes 2-3-4)
    python main.py --step extract     # Étape 1 uniquement
    python main.py --step chunk       # Étape 2 uniquement
    python main.py --step embed       # Étape 3 uniquement
    python main.py --step store       # Étape 4 uniquement
    python main.py --step all         # Toutes les étapes (1-2-3-4)
    python main.py --search "ma question"  # Recherche dans l'index

Notes :
    - L'étape 1 (extraction) est longue et consomme des ressources.
      Elle n'est pas exécutée par défaut (les .md existent déjà).
    - Par défaut, seules les étapes 2-3-4 sont exécutées (chunk → embed → store).
"""

import sys
import logging
import argparse
from pathlib import Path

# Ajouter le répertoire du projet au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from config import LOG_LEVEL, LOG_FORMAT, LOG_DATEFMT

# Configuration du logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATEFMT)
logger = logging.getLogger(__name__)


def run_pipeline(steps: list[str] | None = None, search_query: str | None = None):
    """
    Exécute le pipeline RAG.

    Paramètres :
        steps        : liste des étapes à exécuter. Ex: ["chunk", "embed", "store"]
                       Si None, exécute chunk → embed → store (par défaut).
        search_query : si fourni, effectue une recherche dans l'index existant.
    """

    # ── Mode recherche ───────────────────────────────────────────────────
    if search_query:
        from src.step_4_store.vector_store import search

        logger.info("=" * 65)
        logger.info("🔍 RECHERCHE SÉMANTIQUE")
        logger.info("=" * 65)
        logger.info("   Requête : \"%s\"", search_query)
        logger.info("")

        results = search(search_query, top_k=5)

        if not results:
            logger.warning("Aucun résultat. L'index FAISS existe-t-il ?")
            return

        for r in results:
            logger.info("─" * 65)
            logger.info("  #%d — Score : %.4f", r["rank"], r["score"])
            logger.info("  📁 Source   : %s", r["metadata"]["source"])
            logger.info("  🏷️  Catégorie : %s", r["metadata"]["category"])
            logger.info("  📑 Section  : %s", r["metadata"].get("section", "—"))
            logger.info("  📝 Texte    :")
            # Afficher le texte du chunk (limité à 500 car.)
            text = r["text"][:500]
            for line in text.split("\n"):
                logger.info("     %s", line)
        logger.info("─" * 65)
        return

    # ── Mode pipeline ────────────────────────────────────────────────────
    if steps is None:
        steps = ["chunk", "embed", "store"]

    logger.info("")
    logger.info("╔" + "═" * 63 + "╗")
    logger.info("║   🚀 PIPELINE RAG — Trifouillis-sur-Loire" + " " * 20 + "║")
    logger.info("╠" + "═" * 63 + "╣")
    logger.info("║   Étapes à exécuter : %-40s║", ", ".join(steps))
    logger.info("╚" + "═" * 63 + "╝")
    logger.info("")

    chunks = None
    embeddings_result = None

    # ── Étape 1 : Extraction ─────────────────────────────────────────────
    if "extract" in steps or "all" in steps:
        from src.step_1_extract.extractor import run_extraction
        result = run_extraction()
        logger.info("")

    # ── Étape 2 : Chunking ───────────────────────────────────────────────
    if "chunk" in steps or "all" in steps:
        from src.step_2_chunk.chunker import run_chunking
        chunks = run_chunking()
        logger.info("")

    # ── Étape 3 : Embeddings ─────────────────────────────────────────────
    if "embed" in steps or "all" in steps:
        from src.step_3_embed.embedder import run_embedding
        embeddings_result = run_embedding(
            chunks=chunks,
            use_sbert=True,
            use_mistral=True,
            visualize=True,
        )
        logger.info("")

    # ── Étape 4 : Vector Store ───────────────────────────────────────────
    if "store" in steps or "all" in steps:
        from src.step_4_store.vector_store import run_store

        sbert_emb = None
        if embeddings_result and embeddings_result.get("sbert_embeddings") is not None:
            sbert_emb = embeddings_result["sbert_embeddings"]

        store_chunks = None
        if embeddings_result:
            store_chunks = embeddings_result.get("chunks")

        run_store(
            embeddings=sbert_emb,
            chunks=store_chunks,
            embedding_type="sbert",
        )
        logger.info("")

    # ── Résumé final ─────────────────────────────────────────────────────
    logger.info("╔" + "═" * 63 + "╗")
    logger.info("║   ✅ PIPELINE TERMINÉ" + " " * 42 + "║")
    logger.info("╚" + "═" * 63 + "╝")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline RAG — Trifouillis-sur-Loire",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python main.py                         Pipeline par défaut (chunk → embed → store)
  python main.py --step all              Pipeline complet (extract → chunk → embed → store)
  python main.py --step extract          Étape 1 : extraction uniquement
  python main.py --step chunk            Étape 2 : chunking uniquement
  python main.py --step embed            Étape 3 : embedding uniquement
  python main.py --step store            Étape 4 : stockage FAISS uniquement
  python main.py --step chunk embed      Étapes 2+3 seulement
  python main.py --search "éclairage"    Recherche sémantique dans l'index
        """,
    )

    parser.add_argument(
        "--step", nargs="+",
        choices=["extract", "chunk", "embed", "store", "all"],
        help="Étape(s) du pipeline à exécuter",
    )

    parser.add_argument(
        "--search",
        type=str,
        help="Recherche sémantique dans l'index FAISS existant",
    )

    args = parser.parse_args()

    if args.search:
        run_pipeline(search_query=args.search)
    else:
        run_pipeline(steps=args.step)


if __name__ == "__main__":
    main()
