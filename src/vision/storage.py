"""
Detection storage and caching utilities.

Handles saving/loading detection results to avoid repeated inference.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from .detector_base import Detection


class DetectionStorage:
    """
    Manages storage and retrieval of detection results.
    
    Phase 5: Cache detection results to disk for reuse.
    """
    
    def __init__(self, storage_dir: str):
        """
        Initialize detection storage.
        
        Args:
            storage_dir: Directory to store detection cache files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def save_detections(self, image_id: str, detections: List[Detection]) -> None:
        """
        Save detections for an image.
        
        Phase 5 TODO: Implement serialization logic.
        
        Args:
            image_id: Unique image identifier
            detections: List of Detection objects
        """
        pass
    
    def load_detections(self, image_id: str) -> Optional[List[Detection]]:
        """
        Load cached detections for an image.
        
        Phase 5 TODO: Implement deserialization logic.
        
        Args:
            image_id: Unique image identifier
            
        Returns:
            List of Detection objects if cached, None otherwise
        """
        return None
    
    def has_cached(self, image_id: str) -> bool:
        """
        Check if detections are cached for an image.
        
        Args:
            image_id: Unique image identifier
            
        Returns:
            True if cached, False otherwise
        """
        cache_file = self.storage_dir / f"{image_id}.pkl"
        return cache_file.exists()
    
    def clear_cache(self) -> int:
        """
        Clear all cached detections.
        
        Returns:
            Number of cache files deleted
        """
        count = 0
        for cache_file in self.storage_dir.glob("*.pkl"):
            cache_file.unlink()
            count += 1
        return count
