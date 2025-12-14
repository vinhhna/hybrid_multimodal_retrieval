"""
Base class for open-vocabulary object detectors.

Defines the interface for all detector implementations in Phase 5.
Supports detection with text-based entity prompts (open-vocabulary detection).

Phase 5 Day 1: Updated interface using RawDetection types for v3.1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Phase 5 imports
from .types import RawDetection


# ============================================================================
# Legacy Detection class (backward compatibility with Phase 4)
# ============================================================================

@dataclass
class Detection:
    """
    Legacy detection format (Phase 4 compatibility).
    
    DEPRECATED in Phase 5: Use RawDetection instead.
    Kept for backward compatibility with existing code.
    """
    label: str
    score: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized coordinates
    attributes: Dict[str, Any] = None  # Optional attributes extracted from the region


# ============================================================================
# Phase 4 DetectorBase (legacy, kept for compatibility)
# ============================================================================

class DetectorBase(ABC):
    """
    Abstract base class for object detectors (Phase 4 legacy).
    
    DEPRECATED in Phase 5: Use OpenVocabDetector instead.
    Kept for backward compatibility with existing Phase 4 code.
    
    All detector implementations (OWL-ViT, YOLO, etc.) should inherit from this class
    and implement the detect() method.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the detector.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        self.config = config
    
    @abstractmethod
    def detect(self, image_path: str, entities: List[str]) -> List[Detection]:
        """
        Detect entities in an image.
        
        Args:
            image_path: Path to the image file
            entities: List of entity labels to detect
            
        Returns:
            List of Detection objects
        """
        pass
    
    @abstractmethod
    def batch_detect(self, image_paths: List[str], entities: List[str]) -> Dict[str, List[Detection]]:
        """
        Batch detect entities across multiple images.
        
        Args:
            image_paths: List of image file paths
            entities: List of entity labels to detect
            
        Returns:
            Dictionary mapping image_path to list of Detection objects
        """
        pass


# ============================================================================
# Phase 5 OpenVocabDetector (new interface)
# ============================================================================

class OpenVocabDetector(ABC):
    """
    Abstract base class for open-vocabulary object detectors (Phase 5).
    
    Open-vocabulary detection allows prompting with arbitrary text phrases
    without requiring predefined class labels or fine-tuning.
    
    Phase 5 implementations:
      - OWL-ViT (google/owlvit-base-patch32, etc.)
      - Future: CLIP-based detectors, Grounding DINO, etc.
    
    Key differences from Phase 4 DetectorBase:
      - Uses pathlib.Path for cross-platform compatibility
      - Returns RawDetection objects with rich metadata
      - Supports flexible phrase_type tagging for candidate tracking
      - Designed for batch processing with large phrase lists
    """
    
    @property
    @abstractmethod
    def model_id(self) -> str:
        """
        Return the unique model identifier for this detector.
        
        Examples: "owlvit-base", "owlvit-large", "grounding-dino"
        
        Used for detection provenance tracking and caching.
        """
        pass
    
    @abstractmethod
    def detect(
        self,
        images: List[Path],
        phrases: List[str],
        phrase_types: List[str],
        image_ids: List[str | int],
        entity_ids: List[int],
    ) -> List[RawDetection]:
        """
        Detect entities in a batch of images with open-vocabulary phrases.
        
        Phase 5 Day 6+: Implement actual detection logic with OWL-ViT.
        Day 1: Placeholder signature only (NotImplementedError).
        
        Args:
            images: List of image file paths to process
            phrases: List of text phrases to detect (entity names)
            phrase_types: List of phrase type labels (e.g., "caption", "visual_prior")
            image_ids: List of unique image identifiers
            entity_ids: List of entity vocabulary IDs corresponding to phrases
        
        Returns:
            List of RawDetection objects (one per image-phrase-box combination).
            May be empty if no detections found.
        
        Notes:
            - All input lists must have compatible lengths:
              len(images) = len(image_ids)
              len(phrases) = len(phrase_types) = len(entity_ids)
            - Each phrase is tested against all images (cross-product)
            - Multiple boxes per (image, phrase) pair should generate multiple RawDetections
        
        Example:
            >>> detector = OwlViTDetector(config)
            >>> images = [Path("img1.jpg"), Path("img2.jpg")]
            >>> phrases = ["red car", "blue dog"]
            >>> types = ["caption", "visual_prior"]
            >>> img_ids = [0, 1]
            >>> ent_ids = [10, 25]
            >>> detections = detector.detect(images, phrases, types, img_ids, ent_ids)
            >>> print(len(detections))  # e.g., 3 boxes found across images and phrases
        """
        pass
