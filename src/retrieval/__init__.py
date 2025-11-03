"""
Retrieval module for multimodal search.

This module contains components for:
- Bi-encoder (CLIP) for fast embedding generation
- FAISS indexing for efficient similarity search
- Cross-encoder (BLIP-2) for accurate re-ranking
- Hybrid search engine combining CLIP and BLIP-2
"""

from .bi_encoder import BiEncoder
from .faiss_index import FAISSIndex
from .cross_encoder import CrossEncoder
from .hybrid_search import HybridSearchEngine
from .search_engine import MultimodalSearchEngine, SearchResult

__all__ = [
    'BiEncoder',
    'FAISSIndex',
    'CrossEncoder',
    'HybridSearchEngine',
    'MultimodalSearchEngine',
    'SearchResult'
]
