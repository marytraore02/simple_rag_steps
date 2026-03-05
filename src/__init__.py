"""
src — Modules du pipeline RAG de Trifouillis-sur-Loire.

Pipeline :
  1. step_1_extract : Extraction des documents sources → Markdown
  2. step_2_chunk   : Découpage récursif avec chevauchement
  3. step_3_embed   : Vectorisation (embeddings SBERT / Mistral)
  4. step_4_store   : Stockage et recherche vectorielle (FAISS)
"""
