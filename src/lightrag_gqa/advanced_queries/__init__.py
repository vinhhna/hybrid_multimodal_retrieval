"""
Advanced Queries - 4 Advanced Reasoning Types

Implements the 4 advanced query types for LightRAG-GQA:
6. Chain Reasoning - Multi-hop sequential traversal with constraints
7. Pattern Matching - Find subgraph instances matching a pattern
8. Scene Comparison - Compare structural properties between images
9. Counterfactual Reasoning - Hypothetical "what-if" analysis
"""

from .reasoning_engine import (
    AdvancedReasoningEngine,
    AdvancedReasoningResult,
    ReasoningStep,
)

__all__ = [
    'AdvancedReasoningEngine',
    'AdvancedReasoningResult',
    'ReasoningStep',
]
