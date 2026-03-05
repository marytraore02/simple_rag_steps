"""
Configuration centralisée du pipeline RAG.

Toutes les constantes, chemins et paramètres sont regroupés ici
pour faciliter la maintenance et l'expérimentation.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# ── Chemins du projet ────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent

# Données d'entrée (documents sources à convertir)
DATA_DIR = PROJECT_ROOT / "data"
# Chemin configurable via .env (DATA_SOURCE_DIR), sinon fallback local
INPUTS_DIR = Path(os.getenv("DATA_SOURCE_DIR", str(DATA_DIR / "inputs")))

# Données intermédiaires & de sortie
MARKDOWN_DIR = DATA_DIR / "markdown"
CHUNKS_DIR = DATA_DIR / "chunks"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

# Résultats d'analyse (plots, rapports)
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"

# ── Étape 1 : Extraction (Docling + Whisper) ────────────────────────────────

# Extensions traitées par Docling
DOCLING_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm",
    ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp",
    ".csv", ".tex", ".vtt",
}

# Extensions audio traitées par Faster-Whisper
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}

# Union de toutes les extensions supportées
SUPPORTED_EXTENSIONS = DOCLING_EXTENSIONS | AUDIO_EXTENSIONS

# Configuration Whisper
WHISPER_MODEL_SIZE = "small"
WHISPER_DEVICE = "cpu"            # "cpu" ou "cuda" si GPU NVIDIA
WHISPER_COMPUTE_TYPE = "int8"     # "int8" pour réduire la RAM, "float16" pour GPU
WHISPER_BEAM_SIZE = 5             # 1=rapide, 5=précis

# ── Étape 2 : Chunking ──────────────────────────────────────────────────────

# Taille maximale d'un chunk en caractères
# 1500 caractères ≈ 300-350 tokens (bon pour Mistral/SBERT embeddings et évite de couper les tableaux Markdown)
CHUNK_SIZE = 1500

# Chevauchement entre chunks adjacents (20% du chunk_size)
CHUNK_OVERLAP = 300

# Headers Markdown pour le pré-découpage structurel
HEADERS_TO_SPLIT_ON = [
    ("#", "titre_h1"),
    ("##", "titre_h2"),
    ("###", "titre_h3"),
]

# Séparateurs récursifs (du plus large au plus fin)
SEPARATORS = ["\n\n", "\n", ". ", ", ", " ", ""]

# Fichier de sortie des chunks
CHUNKS_OUTPUT_FILE = CHUNKS_DIR / "chunks.json"

# ── Étape 3 : Embeddings ────────────────────────────────────────────────────

# Modèle SBERT local (Sentence-BERT)
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

# Clé API Mistral (depuis .env)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Modèle d'embedding Mistral
MISTRAL_EMBED_MODEL = "mistral-embed"

# Taille de batch pour l'API Mistral
MISTRAL_BATCH_SIZE = 50

# Fichier de sortie des embeddings
EMBEDDINGS_OUTPUT_FILE = CHUNKS_DIR / "embeddings.npz"

# ── Étape 4 : Vector Store (FAISS) ──────────────────────────────────────────

# Fichier de l'index FAISS
FAISS_INDEX_FILE = VECTORSTORE_DIR / "faiss_index.bin"

# Fichier des métadonnées associées à l'index
FAISS_METADATA_FILE = VECTORSTORE_DIR / "metadata.json"

# Nombre de résultats par défaut pour une recherche
FAISS_TOP_K = 5

# ── Logging ──────────────────────────────────────────────────────────────────

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s │ %(levelname)-7s │ %(message)s"
LOG_DATEFMT = "%H:%M:%S"
