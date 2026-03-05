"""
Étape 1 — Extraction de documents sources vers Markdown.

Parcourt récursivement le répertoire d'entrée et convertit chaque fichier
supporté en Markdown :
  - Documents (PDF, DOCX, PPTX, etc.) → via Docling
  - Audio (WAV, MP3, M4A, OGG, FLAC) → via Faster-Whisper (modèle local)

Usage autonome :
    python -m src.step_1_extract.extractor

Usage depuis le pipeline :
    from src.step_1_extract.extractor import run_extraction
    documents = run_extraction()
"""

import os
import sys
import logging
from pathlib import Path

# Ajout du répertoire racine au PYTHONPATH pour les imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (
    INPUTS_DIR, MARKDOWN_DIR,
    DOCLING_EXTENSIONS, AUDIO_EXTENSIONS, SUPPORTED_EXTENSIONS,
    WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE, WHISPER_BEAM_SIZE,
    LOG_LEVEL, LOG_FORMAT, LOG_DATEFMT,
)

logger = logging.getLogger(__name__)


# ── Fonctions ────────────────────────────────────────────────────────────────


def discover_files(input_dir: Path) -> list[Path]:
    """
    Parcourt récursivement le répertoire et retourne la liste des fichiers
    dont l'extension est supportée.
    """
    files: list[Path] = []
    for root, _dirs, filenames in os.walk(input_dir):
        for filename in sorted(filenames):
            filepath = Path(root) / filename
            if filepath.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(filepath)
            else:
                logger.warning("Extension non supportée, ignoré : %s", filepath.name)
    return files


def transcribe_audio(whisper_model, source_path: Path, output_path: Path,
                     input_dir: Path) -> bool:
    """
    Transcrit un fichier audio en Markdown via Faster-Whisper.

    Le VAD (Voice Activity Detection) est activé pour :
      - Ignorer les silences et bruits de fond
      - Accélérer la transcription
      - Réduire les hallucinations

    Retourne True si la transcription a réussi, False sinon.
    """
    try:
        logger.info("🎤 Transcription audio : %s", source_path.relative_to(input_dir))
        logger.info("   Modèle : %s | Device : %s | Compute : %s",
                     WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE)

        segments, info = whisper_model.transcribe(
            str(source_path),
            beam_size=WHISPER_BEAM_SIZE,
            vad_filter=True,
        )

        logger.info("   Langue détectée : %s (probabilité: %.1f%%)",
                     info.language, info.language_probability * 100)
        logger.info("   Durée audio : %.1f secondes", info.duration)

        all_segments = list(segments)
        transcription_text = " ".join([s.text.strip() for s in all_segments])

        if not transcription_text.strip():
            logger.warning("   ⚠️ Transcription vide pour %s", source_path.name)
            transcription_text = "*Aucun contenu vocal détecté dans ce fichier audio.*"

        markdown_lines = [
            f"# Transcription : {source_path.stem}",
            "",
            f"- **Fichier source** : `{source_path.name}`",
            f"- **Langue détectée** : {info.language} ({info.language_probability * 100:.1f}%)",
            f"- **Durée** : {info.duration:.1f} secondes ({info.duration / 60:.1f} minutes)",
            f"- **Modèle utilisé** : Whisper {WHISPER_MODEL_SIZE} (faster-whisper, {WHISPER_COMPUTE_TYPE})",
            f"- **Nombre de segments** : {len(all_segments)}",
            "",
            "---",
            "",
            "## Transcription complète",
            "",
            transcription_text,
            "",
            "---",
            "",
            "## Segments détaillés",
            "",
        ]

        for seg in all_segments:
            start_min, start_sec = divmod(seg.start, 60)
            end_min, end_sec = divmod(seg.end, 60)
            markdown_lines.append(
                f"- **[{int(start_min):02d}:{start_sec:05.2f} → "
                f"{int(end_min):02d}:{end_sec:05.2f}]** {seg.text.strip()}"
            )

        markdown_content = "\n".join(markdown_lines) + "\n"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown_content, encoding="utf-8")

        logger.info("  ✅ Transcription sauvegardée → %s",
                     output_path.relative_to(MARKDOWN_DIR))
        return True

    except Exception as e:
        logger.error("  ❌ Échec transcription pour %s : %s", source_path.name, e)
        return False


def convert_file(converter, source_path: Path, output_path: Path,
                 input_dir: Path) -> bool:
    """
    Convertit un fichier document en Markdown via Docling.

    Retourne True si la conversion a réussi, False sinon.
    """
    try:
        logger.info("📄 Conversion : %s", source_path.relative_to(input_dir))
        result = converter.convert(str(source_path))
        markdown_content = result.document.export_to_markdown()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown_content, encoding="utf-8")

        logger.info("  ✅ Sauvegardé → %s", output_path.relative_to(MARKDOWN_DIR))
        return True

    except Exception as e:
        logger.error("  ❌ Échec pour %s : %s", source_path.name, e)
        return False


def run_extraction(input_dir: Path | None = None,
                   output_dir: Path | None = None) -> dict:
    """
    Point d'entrée de l'étape 1 : extraction de documents vers Markdown.

    Paramètres :
        input_dir  : répertoire des documents sources (défaut : config.INPUTS_DIR)
        output_dir : répertoire de sortie Markdown (défaut : config.MARKDOWN_DIR)

    Retourne :
        Un dictionnaire avec les statistiques :
        {
            "success_count": int,
            "fail_count": int,
            "total_files": int,
            "output_dir": str,
        }
    """
    input_dir = input_dir or INPUTS_DIR
    output_dir = output_dir or MARKDOWN_DIR

    if not input_dir.exists():
        logger.error("Le répertoire source n'existe pas : %s", input_dir)
        return {"success_count": 0, "fail_count": 0, "total_files": 0, "output_dir": str(output_dir)}

    files = discover_files(input_dir)
    if not files:
        logger.warning("Aucun fichier supporté trouvé dans %s", input_dir)
        return {"success_count": 0, "fail_count": 0, "total_files": 0, "output_dir": str(output_dir)}

    audio_files = [f for f in files if f.suffix.lower() in AUDIO_EXTENSIONS]
    doc_files = [f for f in files if f.suffix.lower() in DOCLING_EXTENSIONS]

    logger.info("=" * 65)
    logger.info("📄 ÉTAPE 1 — EXTRACTION VERS MARKDOWN")
    logger.info("=" * 65)
    logger.info("  📂 Source  : %s", input_dir)
    logger.info("  📁 Sortie  : %s", output_dir)
    logger.info("  📄 Documents (Docling) : %d", len(doc_files))
    logger.info("  🎤 Audios (Whisper)    : %d", len(audio_files))
    logger.info("  📊 Total fichiers      : %d", len(files))
    logger.info("=" * 65)

    output_dir.mkdir(parents=True, exist_ok=True)
    success_count = 0
    fail_count = 0

    # ── Audio avec Faster-Whisper ────────────────────────────────────────
    if audio_files:
        from faster_whisper import WhisperModel

        logger.info("")
        logger.info("🎤 Chargement du modèle Whisper '%s'...", WHISPER_MODEL_SIZE)
        whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        logger.info("✅ Modèle Whisper chargé !")

        for source_path in audio_files:
            relative_path = source_path.relative_to(input_dir)
            output_path = output_dir / relative_path.with_suffix(".md")
            if transcribe_audio(whisper_model, source_path, output_path, input_dir):
                success_count += 1
            else:
                fail_count += 1

        del whisper_model
        logger.info("🧹 Modèle Whisper déchargé.")

    # ── Documents avec Docling ───────────────────────────────────────────
    if doc_files:
        from docling.document_converter import DocumentConverter

        logger.info("")
        logger.info("📄 Initialisation de Docling...")
        converter = DocumentConverter()
        logger.info("✅ Docling initialisé !")

        for source_path in doc_files:
            relative_path = source_path.relative_to(input_dir)
            output_path = output_dir / relative_path.with_suffix(".md")
            if convert_file(converter, source_path, output_path, input_dir):
                success_count += 1
            else:
                fail_count += 1

    # ── Résumé ───────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 65)
    logger.info("🏁 Extraction terminée !")
    logger.info("   ✅ Réussies : %d", success_count)
    if fail_count:
        logger.info("   ❌ Échouées : %d", fail_count)
    logger.info("   📁 Résultats dans : %s", output_dir)
    logger.info("=" * 65)

    return {
        "success_count": success_count,
        "fail_count": fail_count,
        "total_files": len(files),
        "output_dir": str(output_dir),
    }


# ── Exécution autonome ──────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATEFMT)
    run_extraction()
