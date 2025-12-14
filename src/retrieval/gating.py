"""
Score gating and fusion for hybrid retrieval.

Combines CLIP, KG, and entity-binding scores with adaptive gating.
"""

from typing import Dict, Any, Optional


class ScoreGating:
    """
    Adaptive gating mechanism for score fusion.
    
    Phase 5: Decides when to use KG scores vs. entity-binding scores.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize score gating.
        
        Args:
            config: Configuration dict with gating parameters
                - clip_weight: Weight for CLIP similarity (default: 0.4)
                - kg_weight: Weight for KG score (default: 0.3)
                - binding_weight: Weight for entity binding (default: 0.3)
                - kg_threshold: Min KG score to use (default: 0.1)
        """
        self.config = config
        self.clip_weight = config.get("clip_weight", 0.4)
        self.kg_weight = config.get("kg_weight", 0.3)
        self.binding_weight = config.get("binding_weight", 0.3)
        self.kg_threshold = config.get("kg_threshold", 0.1)
    
    def fuse_scores(self,
                   clip_score: float,
                   kg_score: Optional[float] = None,
                   binding_score: Optional[float] = None) -> float:
        """
        Fuse multiple retrieval scores with adaptive gating.
        
        Phase 5: Gracefully handles missing scores (KG or binding).
        
        Args:
            clip_score: CLIP similarity score (always present)
            kg_score: Knowledge graph score (optional)
            binding_score: Entity binding score (optional)
            
        Returns:
            Fused final score
        """
        # Start with CLIP as base
        total_weight = self.clip_weight
        fused = clip_score * self.clip_weight
        
        # Add KG score if available and above threshold
        if kg_score is not None and kg_score >= self.kg_threshold:
            fused += kg_score * self.kg_weight
            total_weight += self.kg_weight
        
        # Add binding score if available
        if binding_score is not None:
            fused += binding_score * self.binding_weight
            total_weight += self.binding_weight
        
        # Normalize by actual total weight used
        return fused / total_weight if total_weight > 0 else clip_score
    
    def should_use_entity_binding(self, query: str) -> bool:
        """
        Decide if entity-binding scoring should be used for this query.
        
        Phase 5 TODO: Implement heuristics (e.g., check for adjectives/attributes).
        
        Args:
            query: Natural language query
            
        Returns:
            True if entity binding should be computed, False otherwise
        """
        # Placeholder: Always return False (CLIP + KG only)
        return False


# ============================================================================
# Phase 5 v3.1 Gating Functions (Day 1: Stubs + pure functions)
# ============================================================================

def compute_s_clip(
    query_emb,  # np.ndarray or torch.Tensor
    image_emb,  # np.ndarray or torch.Tensor
) -> float:
    """
    Compute normalized CLIP similarity score.
    
    Phase 5 Day 1: Stub (raises NotImplementedError).
    Phase 5 Day 8+: Full implementation with cosine similarity.
    
    Args:
        query_emb: Query embedding vector
        image_emb: Image embedding vector
    
    Returns:
        Normalized CLIP similarity score in [0, 1].
    
    Raises:
        NotImplementedError: Always on Day 1 (deferred to Day 8+).
    """
    raise NotImplementedError(
        "compute_s_clip() implementation deferred to Phase 5 Day 8+. "
        "Day 1 provides only the function signature for scaffolding."
    )


def compute_s_cov(
    caption_entities: list[int],
    kg_expanded_entities: list[int],
) -> float:
    """
    Compute coverage score (fraction of caption entities in KG expansion).
    
    Phase 5 Day 1: Stub (raises NotImplementedError).
    Phase 5 Day 8+: Full implementation with set intersection.
    
    Args:
        caption_entities: List of entity IDs extracted from captions
        kg_expanded_entities: List of entity IDs from KG expansion
    
    Returns:
        Coverage score in [0, 1] (0 if no caption entities).
    
    Raises:
        NotImplementedError: Always on Day 1 (deferred to Day 8+).
    """
    raise NotImplementedError(
        "compute_s_cov() implementation deferred to Phase 5 Day 8+. "
        "Day 1 provides only the function signature for scaffolding."
    )


def should_use_kg(
    s_clip: float,
    s_cov: float,
    alpha: float = 0.5,
    tau_gate: float = 0.5,
) -> bool:
    """
    Decide whether to use KG expansion for this query-image pair.
    
    Phase 5 Day 1: Implemented (pure function, safe to use).
    Phase 5 Day 8+: Used in gating logic.
    
    Gating decision:
      - Use KG if: alpha * s_clip + (1 - alpha) * s_cov >= tau_gate
      - Otherwise: CLIP-only (KG may be noisy or uninformative)
    
    Args:
        s_clip: CLIP similarity score (normalized to [0, 1])
        s_cov: Coverage score (fraction of caption entities in KG expansion)
        alpha: Weight for CLIP score (1 - alpha is weight for coverage)
        tau_gate: Gating threshold (default: 0.5)
    
    Returns:
        True if KG should be used, False for CLIP-only.
    
    Example:
        >>> # High CLIP, high coverage -> use KG
        >>> should_use_kg(s_clip=0.8, s_cov=0.7, alpha=0.5, tau_gate=0.5)
        True
        
        >>> # Low CLIP, low coverage -> skip KG
        >>> should_use_kg(s_clip=0.3, s_cov=0.2, alpha=0.5, tau_gate=0.5)
        False
        
        >>> # Borderline case (weighted sum = threshold)
        >>> should_use_kg(s_clip=0.5, s_cov=0.5, alpha=0.5, tau_gate=0.5)
        True
    """
    # Compute weighted gating score
    gating_score = alpha * s_clip + (1.0 - alpha) * s_cov
    
    # Use KG if gating score meets threshold
    return gating_score >= tau_gate
