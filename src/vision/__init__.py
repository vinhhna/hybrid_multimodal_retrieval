"""
Vision Module for Phase 5

Handles object detection, attribute extraction, and visual grounding
for entity-based image retrieval.
"""

__version__ = "0.1.0"

from .detector_base import DetectorBase
from .detector_owlvit import OWLViTDetector
from .storage import DetectionStorage
from .postprocess import filter_detections, merge_overlapping_boxes

__all__ = [
    'DetectorBase',
    'OWLViTDetector',
    'DetectionStorage',
    'filter_detections',
    'merge_overlapping_boxes',
]
