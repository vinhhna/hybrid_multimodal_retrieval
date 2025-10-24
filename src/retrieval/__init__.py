"""
Retrieval module for multimodal search.

This module contains components for:
- Bi-encoder (CLIP) for fast embedding generation
- FAISS indexing for efficient similarity search
- Cross-encoder (BLIP-2) for accurate re-ranking
"""

from .bi_encoder import BiEncoder
from .faiss_index import FAISSIndex
from .search_engine import MultimodalSearchEngine, SearchResult

__all__ = ['BiEncoder', 'FAISSIndex', 'MultimodalSearchEngine', 'SearchResult']
