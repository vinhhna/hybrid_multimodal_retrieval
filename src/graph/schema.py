"""
Phase 4 Graph Schema
Minimal node/edge type declarations for Image and Caption.
Region is declared as a stub only (not implemented in Phase 4 Day 1-2).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# === PHASE4 GRAPH SCHEMA (ANCHOR) ===

# StrEnum (py311+) fallback for py310/py39
try:
    from enum import StrEnum  # Python 3.11+
except ImportError:  # Python <= 3.10
    from enum import Enum
    class StrEnum(str, Enum):  # minimal polyfill
        pass


class NodeType(StrEnum):
    """
    Node types for the multimodal knowledge graph.
    
    Phase 4 Day 1-2 implements:
    - IMAGE: Image nodes with CLIP image embeddings
    - CAPTION: Caption nodes with CLIP text embeddings
    - REGION: Stub declaration only (not implemented this phase)
    """
    IMAGE = "image"
    CAPTION = "caption"
    REGION = "region"  # Stub only; not implemented in Phase 4 Day 1-2


class EdgeType(StrEnum):
    """
    Edge types for the multimodal knowledge graph.
    
    Semantic edges (SEM_SIM):
    - Within-type k-NN by cosine similarity in CLIP space
    - image ↔ image: top-k semantically similar images
    - caption ↔ caption: top-k semantically similar captions
    
    Co-occurrence edges:
    - PAIRED_WITH: caption ↔ image (paired in dataset)
    - COOCCUR: caption ↔ caption (captions of the same image)
    """
    SEM_SIM = "sem_sim"        # Semantic similarity (within-type k-NN)
    PAIRED_WITH = "paired_with"  # Caption-image pairing from dataset
    COOCCUR = "cooccur"        # Caption-caption co-occurrence (same image)


@dataclass(slots=True)
class ImageNodeMeta:
    """
    Metadata for Image nodes.
    
    Attributes:
        image_id: Unique identifier for the image (e.g., filename without extension)
        path: Path to the image file (str or Path object)
        size: Optional (width, height) tuple of image dimensions
    
    Note:
        All image nodes carry L2-normalized CLIP image embeddings (512-dim float32).
    """
    image_id: str
    path: str | Path
    size: Optional[tuple[int, int]] = None
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "image_id": self.image_id,
            "path": str(self.path),
            "size": None if self.size is None else (int(self.size[0]), int(self.size[1])),
        }


@dataclass(slots=True)
class CaptionNodeMeta:
    """
    Metadata for Caption nodes.
    
    Attributes:
        caption_id: Unique identifier for the caption (e.g., image_id + caption_index)
        image_id: ID of the associated image
        text: The caption text content
    
    Note:
        All caption nodes carry L2-normalized CLIP text embeddings (512-dim float32).
    """
    caption_id: str
    image_id: str
    text: str
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "caption_id": self.caption_id,
            "image_id": self.image_id,
            "text": self.text,
        }
