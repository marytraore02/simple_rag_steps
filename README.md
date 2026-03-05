# 🏛️ Pipeline RAG — Trifouillis-sur-Loire

Pipeline de **Retrieval-Augmented Generation (RAG)** pour les documents municipaux de Trifouillis-sur-Loire.

## 📁 Architecture du projet

```
extract_divers_source/
│
├── main.py                      ← Orchestrateur principal
├── config.py                    ← Configuration centralisée
├── requirements.txt             ← Dépendances Python
├── .env                         ← Variables d'environnement (clés API)
│
├── src/                         ← Code source modulaire
│   ├── step_1_extract/          ← Extraction de documents → Markdown
│   │   └── extractor.py         │  Docling (PDF, DOCX...) + Whisper (audio)
│   ├── step_2_chunk/            ← Découpage récursif avec chevauchement
│   │   └── chunker.py           │  MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter
│   ├── step_3_embed/            ← Vectorisation (embeddings)
│   │   └── embedder.py          │  SBERT (local) + Mistral (API)
│   └── step_4_store/            ← Stockage vectoriel & recherche
│       └── vector_store.py      │  FAISS (IndexFlatIP, cosine similarity)
│
├── data/                        ← Données (entrées/sorties)
│   ├── inputs/                  │  Documents sources (PDF, DOCX, audio...)
│   ├── markdown/                │  Fichiers Markdown extraits
│   ├── chunks/                  │  Chunks JSON + embeddings .npz
│   └── vectorstore/             │  Index FAISS + métadonnées
│
└── outputs/                     ← Résultats d'analyse
    └── plots/                   │  Visualisations PCA des embeddings
```

## 🚀 Pipeline

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ 1. EXTRACTION   │────→│ 2. CHUNKING  │────→│ 3. EMBEDDING │────→│ 4. STOCKAGE  │
│ Docling+Whisper │     │ Récursif     │     │ SBERT/Mistral│     │ FAISS        │
│ → Markdown      │     │ + overlap    │     │ → Vecteurs   │     │ → Index      │
└─────────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
     PDF, DOCX              1000 car.           384/1024 dim.        Cosine Sim.
     Audio (MP3)            200 overlap         Plots PCA            Recherche
```

## 📋 Usage

### Pipeline complet

```bash
# Installer les dépendances
pip install -r requirements.txt

# Pipeline par défaut (chunk → embed → store)
# Utilise les .md déjà extraits dans data/markdown/
python main.py

# Pipeline complet (extract → chunk → embed → store)
python main.py --step all
```

### Étapes individuelles

```bash
python main.py --step extract     # Étape 1 uniquement
python main.py --step chunk       # Étape 2 uniquement
python main.py --step embed       # Étape 3 uniquement
python main.py --step store       # Étape 4 uniquement
python main.py --step chunk embed # Étapes 2+3 seulement
```

### Recherche sémantique

```bash
python main.py --search "éclairage public"
python main.py --search "budget 2024"
python main.py --search "centre culturel innovant"
```

### Modules autonomes

```bash
python -m src.step_1_extract.extractor
python -m src.step_2_chunk.chunker
python -m src.step_3_embed.embedder
python -m src.step_4_store.vector_store
```

## ⚙️ Configuration

Tous les paramètres sont dans `config.py` :

| Paramètre | Valeur | Description |
|---|---|---|
| `CHUNK_SIZE` | 1000 | Taille max d'un chunk (caractères) |
| `CHUNK_OVERLAP` | 200 | Chevauchement entre chunks (20%) |
| `SBERT_MODEL_NAME` | all-MiniLM-L6-v2 | Modèle d'embedding local |
| `WHISPER_MODEL_SIZE` | small | Modèle de transcription audio |
| `FAISS_TOP_K` | 5 | Résultats par recherche |

## 🔑 Variables d'environnement

```bash
# .env
MISTRAL_API_KEY='votre_clé_api_mistral'
```

## 📊 Données actuelles

- **30 documents** municipaux (PV, projets, règlements, événements...)
- **304 chunks** après découpage
- **6 catégories** : budget, communication, demandes citoyennes, événements, instances, projets
# simple_rag_steps
