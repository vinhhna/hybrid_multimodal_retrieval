"""
LightRAG-GQA: Two-Level Knowledge Graph Reasoning on GQA Scene Graphs

A knowledge graph system for visual reasoning on the GQA dataset using 
LightRAG two-level architecture. Implements 9 query types (5 basic + 4 advanced)
for structured graph queries.

Main modules:
- basic_queries: 5 basic query types (entity search, stats, similarity, path, negative)
- advanced_queries: 4 advanced reasoning types (chain, pattern, scene, counterfactual)
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
