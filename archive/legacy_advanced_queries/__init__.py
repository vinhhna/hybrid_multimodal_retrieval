"""
Advanced Graph Reasoning Module
================================
Extension of GQA LightRAG with advanced multi-hop reasoning capabilities.

Query Types:
1. Chain Reasoning - Multi-hop traversal with constraints
2. Pattern Matching - Subgraph isomorphism
3. Scene Comparison - Structural similarity
4. Counterfactual Reasoning - What-if analysis
5. Centrality Queries - Node importance metrics
"""

from .advanced_reasoning_engine import (
    AdvancedReasoningEngine,
    AdvancedReasoningResult,
    ReasoningStep,
)

__all__ = [
    'AdvancedReasoningEngine',
    'AdvancedReasoningResult', 
    'ReasoningStep',
]
