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
