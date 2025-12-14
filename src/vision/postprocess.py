"""
Post-processing utilities for object detections.

Includes filtering, NMS, box merging operations, and Phase 5 confidence adjustment.

Phase 5 Day 1: Add postprocess_conf() pure function for noise floor correction.
Phase 5 Day 4: Use in postprocessing pipeline after threshold fitting.
"""

from __future__ import annotations

from typing import List, Tuple
from .detector_base import Detection


# ============================================================================
# Phase 5 Confidence Postprocessing (Day 1: Implemented)
# ============================================================================

def postprocess_conf(
    conf_raw: float,
    tau_present: float,
    noise_floor: float,
    delta: float,
) -> tuple[float, float, bool]:
    """
    Adjust raw confidence scores for noise floor and presence threshold.
    
    Phase 5 Day 1: Pure function implementation (safe, no dependencies).
    Phase 5 Day 4: Used in threshold-based filtering pipeline.
    
    Algorithm:
      1. Compute effective threshold: tau_eff = max(tau_present, noise_floor + delta)
      2. Compute effective confidence: conf_eff = max(0, conf_raw - tau_eff)
      3. Determine presence: is_present = (conf_eff > 0)
    
    The effective threshold ensures we stay safely above the noise floor
    estimated from negative control phrases, plus a safety margin (delta).
    
    Args:
        conf_raw: Raw confidence score from detector (uncalibrated)
        tau_present: Phrase-specific presence threshold (e.g., from ROC curve fitting)
        noise_floor: Global or phrase-type-specific noise floor estimate
        delta: Safety margin above noise floor (e.g., 0.05)
    
    Returns:
        Tuple of (tau_eff, conf_eff, is_present):
          - tau_eff: Effective threshold used for filtering
          - conf_eff: Effective confidence after noise correction
          - is_present: Boolean indicator (True if entity is present)
    
    Example:
        >>> # Case 1: Clear positive (high confidence above threshold)
        >>> tau_eff, conf_eff, is_present = postprocess_conf(0.8, 0.3, 0.2, 0.05)
        >>> print(f"tau_eff={tau_eff:.2f}, conf_eff={conf_eff:.2f}, present={is_present}")
        tau_eff=0.30, conf_eff=0.50, present=True
        
        >>> # Case 2: Below noise floor + delta (rejected)
        >>> tau_eff, conf_eff, is_present = postprocess_conf(0.15, 0.1, 0.2, 0.05)
        >>> print(f"tau_eff={tau_eff:.2f}, conf_eff={conf_eff:.2f}, present={is_present}")
        tau_eff=0.25, conf_eff=0.00, present=False
        
        >>> # Case 3: Between tau_present and noise_floor + delta (use noise floor)
        >>> tau_eff, conf_eff, is_present = postprocess_conf(0.28, 0.20, 0.22, 0.05)
        >>> print(f"tau_eff={tau_eff:.2f}, conf_eff={conf_eff:.2f}, present={is_present}")
        tau_eff=0.27, conf_eff=0.01, present=True
    
    References:
        - Phase 5 Implementation Plan v3.1, Day 4: Postprocessing Pipeline
        - Negative Controls & Noise Floor Estimation (Day 7)
    """
    # Step 1: Compute effective threshold (max of presence threshold and noise floor + delta)
    tau_eff = max(tau_present, noise_floor + delta)
    
    # Step 2: Compute effective confidence (subtract effective threshold, clamp to 0)
    conf_eff = max(0.0, conf_raw - tau_eff)
    
    # Step 3: Determine presence (positive if effective confidence > 0)
    is_present = conf_eff > 0.0
    
    return (tau_eff, conf_eff, is_present)


# ============================================================================
# Phase 4 Legacy Functions (kept for compatibility)
# ============================================================================


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
