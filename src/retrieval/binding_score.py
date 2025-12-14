"""
Binding score computation for entity-image matching.

Computes how well query entities bind to detected image regions.
"""

from typing import List, Dict, Any, Tuple


class BindingScorer:
    """
    Computes binding scores between query slots and image detections.
    
    Phase 5: Combines detection confidence + attribute verification scores.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize binding scorer.
        
        Args:
            config: Configuration dict with weighting parameters
                - detection_weight: Weight for detection confidence (default: 0.5)
                - attribute_weight: Weight for attribute verification (default: 0.5)
        """
        self.config = config
        self.detection_weight = config.get("detection_weight", 0.5)
        self.attribute_weight = config.get("attribute_weight", 0.5)
    
    def compute_binding_score(self,
                             detection_score: float,
                             attribute_scores: List[float]) -> float:
        """
        Compute overall binding score.
        
        Phase 5: Weighted combination of detection and attribute scores.
        
        Args:
            detection_score: Confidence of entity detection (0-1)
            attribute_scores: List of attribute verification scores
            
        Returns:
            Combined binding score (0-1)
        """
        if not attribute_scores:
            return detection_score * self.detection_weight
        
        avg_attr_score = sum(attribute_scores) / len(attribute_scores)
        
        return (self.detection_weight * detection_score + 
                self.attribute_weight * avg_attr_score)
    
    def compute_image_score(self,
                           slot_bindings: Dict[str, float]) -> float:
        """
        Compute overall image score from slot bindings.
        
        Phase 5 TODO: Implement aggregation strategy (min, max, avg, etc.)
        
        Args:
            slot_bindings: Dict mapping slot_id -> binding_score
            
        Returns:
            Overall image relevance score
        """
        if not slot_bindings:
            return 0.0
        
        # Placeholder: Use minimum (all slots must match)
        return min(slot_bindings.values())


# ============================================================================
# Phase 5 v3.1 Binding Score Computation (Day 1: Stub, Day 10+: Implementation)
# ============================================================================

"""
Phase 5 Day 10+: Entity-Attribute Binding Verification

TODO: Implement binding score computation for entity-attribute queries:
  - Given query slots (e.g., "red car") and detection boxes
  - Verify attribute bindings (is the car actually red?)
  - Compute binding score based on:
    1. Detection confidence (entity presence)
    2. Attribute verification score (HSV color matching or CLIP region similarity)
  
Implementation approach:
  1. For each query slot, find top-K detection boxes (topB_boxes per image)
  2. For each box, verify attributes:
     - Color attributes: HSV color histogram matching in box region
     - Other attributes: CLIP similarity with "{attr} {obj}" prompt
  3. Combine detection confidence + attribute scores
  4. Aggregate across all slots (min/avg/max strategy)
  
Related to:
  - Attribute verification (attribute_verifier.py)
  - Query slot parsing (query_slots.py)
  - Phase 5 Day 10: Binding Verification (see Implementation Plan v3.1)
  
Configuration:
  - topN_images: Number of top-ranked images to verify (from config)
  - topB_boxes: Number of top-scored boxes per image (from config)
  - hsv_ranges: Color ranges for HSV matching (from config)
  - clip_fallback: Use CLIP if HSV fails (from config)
"""
