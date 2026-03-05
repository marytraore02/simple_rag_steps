"""
Étape 2 — Découpage récursif avec chevauchement (chunking).

Pipeline :
  1. Charge tous les fichiers .md depuis le répertoire Markdown
  2. Pré-découpe par sections Markdown (titres #, ##, ###)
  3. Découpe récursivement (RecursiveCharacterTextSplitter) avec chevauchement
  4. Enrichit les métadonnées (source, catégorie, section, index)
  5. Exporte en JSON

Usage autonome :
    python -m src.step_2_chunk.chunker

Usage depuis le pipeline :
    from src.step_2_chunk.chunker import run_chunking
    chunks = run_chunking()
"""

import json
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (
    MARKDOWN_DIR, CHUNKS_OUTPUT_FILE,
    CHUNK_SIZE, CHUNK_OVERLAP, HEADERS_TO_SPLIT_ON, SEPARATORS,
    LOG_LEVEL, LOG_FORMAT, LOG_DATEFMT,
)

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

logger = logging.getLogger(__name__)


# ── Fonctions ────────────────────────────────────────────────────────────────


def load_markdown_files(input_dir: Path) -> list[dict]:
    """
    Charge tous les fichiers .md depuis le répertoire d'entrée.

    Retourne une liste de dicts :
      {content, filename, category, filepath}
    """
    documents = []

    for md_file in sorted(input_dir.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()

        if len(content) < 10:
            logger.warning("Fichier trop court, ignoré : %s (%d car.)",
                           md_file.name, len(content))
            continue

        relative = md_file.relative_to(input_dir)
        category = relative.parts[0] if len(relative.parts) > 1 else "racine"

        documents.append({
            "content": content,
            "filename": md_file.stem,
            "category": category,
            "filepath": str(relative),
        })

    return documents


def chunk_document(
    doc: dict,
    md_splitter: MarkdownHeaderTextSplitter,
    text_splitter: RecursiveCharacterTextSplitter,
) -> list[dict]:
    """
    Découpe un document Markdown en chunks enrichis.

    Étape 1 : Pré-découpage par headers Markdown
    Étape 2 : Découpage récursif avec chevauchement

    Retourne une liste de dicts :
      {text, metadata: {source, filename, category, section, chunk_index, chunk_size, ...}}
    """
    content = doc["content"]
    chunks = []

    # Pré-découpage par headers
    md_sections = md_splitter.split_text(content)
    if not md_sections:
        md_sections = [
            type("Section", (), {"page_content": content, "metadata": {}})()
        ]

    chunk_index = 0

    for section in md_sections:
        section_text = section.page_content
        section_metadata = section.metadata

        # Titre de section pour le contexte
        section_title_parts = []
        for key in ["titre_h1", "titre_h2", "titre_h3"]:
            if key in section_metadata:
                section_title_parts.append(section_metadata[key])
        section_title = " > ".join(section_title_parts) if section_title_parts else ""

        base_meta = {
            "source": doc["filepath"],
            "filename": doc["filename"],
            "category": doc["category"],
            "section": section_title,
            **section_metadata,
        }

        if len(section_text) <= CHUNK_SIZE:
            chunks.append({
                "text": section_text,
                "metadata": {
                    **base_meta,
                    "chunk_index": chunk_index,
                    "chunk_size": len(section_text),
                },
            })
            chunk_index += 1
        else:
            sub_chunks = text_splitter.split_text(section_text)
            for sub_chunk in sub_chunks:
                chunks.append({
                    "text": sub_chunk,
                    "metadata": {
                        **base_meta,
                        "chunk_index": chunk_index,
                        "chunk_size": len(sub_chunk),
                    },
                })
                chunk_index += 1

    return chunks


def run_chunking(input_dir: Path | None = None,
                 output_file: Path | None = None) -> list[dict]:
    """
    Point d'entrée de l'étape 2 : chunking des documents Markdown.

    Paramètres :
        input_dir   : répertoire Markdown (défaut : config.MARKDOWN_DIR)
        output_file : fichier JSON de sortie (défaut : config.CHUNKS_OUTPUT_FILE)

    Retourne :
        La liste de tous les chunks générés.
    """
    input_dir = input_dir or MARKDOWN_DIR
    output_file = output_file or CHUNKS_OUTPUT_FILE

    logger.info("=" * 65)
    logger.info("✂️  ÉTAPE 2 — DÉCOUPAGE RÉCURSIF AVEC CHEVAUCHEMENT")
    logger.info("=" * 65)

    if not input_dir.exists():
        logger.error("Le répertoire source n'existe pas : %s", input_dir)
        return []

    # Charger les documents
    logger.info("📂 Chargement depuis : %s", input_dir)
    documents = load_markdown_files(input_dir)

    if not documents:
        logger.warning("Aucun fichier Markdown trouvé !")
        return []

    logger.info("   %d documents chargés", len(documents))

    # Initialiser les splitters
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
        is_separator_regex=False,
        add_start_index=True,
    )

    logger.info("✂️  Splitters initialisés (chunk_size=%d, overlap=%d)",
                CHUNK_SIZE, CHUNK_OVERLAP)

    # Traiter chaque document
    all_chunks = []
    for doc in documents:
        doc_chunks = chunk_document(doc, md_splitter, text_splitter)
        all_chunks.extend(doc_chunks)
        logger.info("   %-50s → %2d chunks",
                     doc["filepath"][:50], len(doc_chunks))

    # Statistiques
    sizes = [c["metadata"]["chunk_size"] for c in all_chunks]
    categories = {}
    for chunk in all_chunks:
        cat = chunk["metadata"]["category"]
        categories[cat] = categories.get(cat, 0) + 1

    logger.info("")
    logger.info("=" * 65)
    logger.info("📊 STATISTIQUES")
    logger.info("=" * 65)
    logger.info("  chunk_size=%d | chunk_overlap=%d (%.0f%%)",
                CHUNK_SIZE, CHUNK_OVERLAP, (CHUNK_OVERLAP / CHUNK_SIZE) * 100)
    logger.info("  Documents : %d → Chunks : %d (ratio %.1f)",
                len(documents), len(all_chunks),
                len(all_chunks) / len(documents) if documents else 0)
    logger.info("  Taille : min=%d | max=%d | moy=%.0f",
                min(sizes), max(sizes), sum(sizes) / len(sizes))
    for cat, count in sorted(categories.items()):
        logger.info("    • %-25s : %d chunks", cat, count)
    logger.info("=" * 65)

    # Export JSON
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "config": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "separators": SEPARATORS,
            "headers_to_split_on": [h[0] for h in HEADERS_TO_SPLIT_ON],
            "total_documents": len(documents),
            "total_chunks": len(all_chunks),
        },
        "chunks": all_chunks,
    }
    output_file.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("💾 %d chunks sauvegardés → %s", len(all_chunks), output_file)
    return all_chunks


# ── Exécution autonome ──────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATEFMT)
    run_chunking()
