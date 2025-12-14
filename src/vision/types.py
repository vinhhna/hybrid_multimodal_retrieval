"""
Type definitions for Phase 5 vision-grounded detection.

Defines lightweight dataclasses for detection rows and bounding boxes,
using only stdlib dependencies for maximum portability.

Phase 5 Day 1: Scaffold types for use in later days.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BoxXYXY:
    """
    Bounding box in absolute XYXY format (x1, y1, x2, y2).
    
    Coordinates are in pixels, not normalized.
    Top-left is (x1, y1), bottom-right is (x2, y2).
    """
    x1: float
    y1: float
    x2: float
    y2: float
    
    def area(self) -> float:
        """Compute box area in pixels."""
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)
    
    def to_tuple(self) -> tuple[float, float, float, float]:
        """Convert to tuple (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)
    
    @classmethod
    def from_tuple(cls, t: tuple[float, float, float, float]) -> BoxXYXY:
        """Create from tuple (x1, y1, x2, y2)."""
        return cls(x1=t[0], y1=t[1], x2=t[2], y2=t[3])


@dataclass
class RawDetection:
    """
    Raw detection output from open-vocabulary detector (OWL-ViT, etc.).
    
    Represents a single (image, entity_phrase, box, confidence) tuple
    before any postprocessing or threshold application.
    
    Attributes:
        image_id: Unique image identifier (str or int)
        entity_id: Entity/phrase ID in vocabulary
        conf_raw: Raw confidence score from detector (before postprocessing)
        box: Bounding box (None if no box, e.g., global detection)
        phrase_type: Category of phrase (e.g., "caption", "visual_prior", "safe_neighbor")
        model_id: Detector model identifier (e.g., "owlvit-base")
    
    Phase 5 usage:
        - Day 3: Store raw detections to sharded files
        - Day 4: Postprocess conf_raw -> conf_eff and filter by thresholds
        - Day 5: Aggregate to build visual priors P(vis|entity)
    """
    image_id: str | int
    entity_id: int
    conf_raw: float
    box: Optional[BoxXYXY]
    phrase_type: str
    model_id: str
    
    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON/Parquet storage."""
        return {
            "image_id": self.image_id,
            "entity_id": self.entity_id,
            "conf_raw": self.conf_raw,
            "box": self.box.to_tuple() if self.box else None,
            "phrase_type": self.phrase_type,
            "model_id": self.model_id,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> RawDetection:
        """Deserialize from dictionary."""
        box = BoxXYXY.from_tuple(d["box"]) if d["box"] is not None else None
        return cls(
            image_id=d["image_id"],
            entity_id=d["entity_id"],
            conf_raw=d["conf_raw"],
            box=box,
            phrase_type=d["phrase_type"],
            model_id=d["model_id"],
        )
