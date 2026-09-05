"""Tri-modal hybrid retrieval: dense (FAISS + MedCPT), lexical (BM25), fusion (RRF),
and cross-encoder rerank. See docs/01_ARCHITECTURE.md and CLAUDE.md §3.

KG traversal (the third modality) lives under src/kg/ (Phase 2).
"""
