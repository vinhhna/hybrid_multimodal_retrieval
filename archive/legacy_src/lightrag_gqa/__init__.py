"""
LightRAG-GQA: Multi-Scale Knowledge Graph System for Visual Question Answering

A comprehensive multimodal knowledge graph system built on the GQA dataset using 
LightRAG architecture. Supports 14 query types (9 basic + 5 advanced) for visual 
reasoning tasks.

Main modules:
- basic_queries: Standard 9 query types (entity search, stats, similarity, etc.)
- advanced_queries: Advanced 5 reasoning types (chain, pattern matching, etc.)
- evaluation: CQR-based evaluation framework
- datasets: GQA dataset loaders
- cli: Command-line interface tools
"""

__version__ = "1.0.0"
__author__ = "GQA LightRAG Project"

from . import basic_queries
from . import advanced_queries

__all__ = [
    'basic_queries',
    'advanced_queries',
]
