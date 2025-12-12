"""
Post-processing utilities for object detections.

Includes filtering, NMS, and box merging operations.
"""

from typing import List, Tuple
from .detector_base import Detection


def filter_detections(detections: List[Detection], threshold: float = 0.3) -> List[Detection]:
    """
    Filter detections by confidence threshold.
    
    Args:
        detections: List of Detection objects
        threshold: Minimum confidence score
        
    Returns:
        Filtered list of detections
    """
    return [d for d in detections if d.score >= threshold]


def compute_iou(bbox1: Tuple[float, float, float, float], 
                bbox2: Tuple[float, float, float, float]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: First bbox as (x1, y1, x2, y2)
        bbox2: Second bbox as (x1, y1, x2, y2)
        
    Returns:
        IoU score between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Compute intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Compute union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def merge_overlapping_boxes(detections: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
    """
    Merge highly overlapping detections of the same class.
    
    Phase 5 TODO: Implement proper NMS or box merging strategy.
    
    Args:
        detections: List of Detection objects
        iou_threshold: IoU threshold for considering boxes as overlapping
        
    Returns:
        List of merged detections
    """
    # Placeholder: Return as-is
    # Proper implementation would group by label, apply NMS, etc.
    return detections
