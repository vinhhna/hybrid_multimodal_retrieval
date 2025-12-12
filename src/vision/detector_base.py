"""
Base class for object detectors.

Defines the interface for all detector implementations in Phase 5.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class Detection:
    """Represents a single object detection."""
    label: str
    score: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized coordinates
    attributes: Dict[str, Any] = None  # Optional attributes extracted from the region


class DetectorBase(ABC):
    """
    Abstract base class for object detectors.
    
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
