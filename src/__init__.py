"""
GQA LightRAG Knowledge Graph System
====================================

A Multimodal Knowledge Graph system for Visual Reasoning using the LightRAG architecture.
Built for processing and querying the GQA (Visual Reasoning) dataset.

Modules:
    - gqa_lightrag_builder: Knowledge graph construction with multi-scale support
    - gqa_reasoning_engine: Query engine with 9 query types
    - gqa_nl_parser: Natural language query parser
    - gqa_query_interface: Command-line and interactive query interface
"""

__version__ = "1.0.0"
__author__ = "IT3930E - Project III"

from .gqa_lightrag_builder import GQALightRAGGraphBuilder
from .gqa_reasoning_engine import GQA_Reasoning_Engine
from .gqa_nl_parser import NaturalLanguageParser, QueryType
from .gqa_query_interface import QueryInterface

__all__ = [
    'GQALightRAGGraphBuilder',
    'GQA_Reasoning_Engine',
    'NaturalLanguageParser',
    'QueryType',
    'QueryInterface',
]
