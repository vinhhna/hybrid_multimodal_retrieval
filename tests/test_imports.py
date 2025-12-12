"""
Import tests for Phase 5

Validates that all core modules can be imported without errors.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_import_flickr30k():
    """Test that flickr30k dataset module imports correctly."""
    from src.flickr30k import Flickr30KDataset
    from src.flickr30k.dataset import Flickr30KDataset as Dataset2
    from src.flickr30k.utils import load_config, get_project_root
    assert Flickr30KDataset is not None
    assert Dataset2 is not None
    assert load_config is not None
    assert get_project_root is not None


def test_import_retrieval():
    """Test that retrieval modules import correctly."""
    from src.retrieval.bi_encoder import BiEncoder
    from src.retrieval.faiss_index import FAISSIndex
    from src.retrieval.cross_encoder import CrossEncoder
    from src.retrieval.hybrid_search import HybridSearchEngine
    from src.retrieval.search_engine import SearchEngine
    
    assert BiEncoder is not None
    assert FAISSIndex is not None
    assert CrossEncoder is not None
    assert HybridSearchEngine is not None
    assert SearchEngine is not None


def test_import_graph():
    """Test that graph modules import correctly."""
    from src.graph.entities import EntityExtractor
    from src.graph.context import EntityContextBuilder
    from src.graph.build_entity_graph import build_entity_graph
    from src.graph.graph_search import GraphSearchEngine
    from src.graph.config import GraphConfig
    
    assert EntityExtractor is not None
    assert EntityContextBuilder is not None
    assert build_entity_graph is not None
    assert GraphSearchEngine is not None
    assert GraphConfig is not None


def test_import_vision():
    """Test that Phase 5 vision modules import correctly."""
    from src.vision import DetectorBase, OWLViTDetector, DetectionStorage
    from src.vision import filter_detections, merge_overlapping_boxes
    from src.vision.detector_base import Detection
    
    assert DetectorBase is not None
    assert OWLViTDetector is not None
    assert DetectionStorage is not None
    assert filter_detections is not None
    assert merge_overlapping_boxes is not None
    assert Detection is not None


def test_import_phase5_retrieval():
    """Test that Phase 5 retrieval modules import correctly."""
    from src.retrieval.query_slots import QuerySlotFiller, QuerySlot
    from src.retrieval.attribute_verifier import AttributeVerifier
    from src.retrieval.binding_score import BindingScorer
    from src.retrieval.gating import ScoreGating
    
    assert QuerySlotFiller is not None
    assert QuerySlot is not None
    assert AttributeVerifier is not None
    assert BindingScorer is not None
    assert ScoreGating is not None


def test_vision_detection_dataclass():
    """Test that Detection dataclass works correctly."""
    from src.vision.detector_base import Detection
    
    det = Detection(
        label="dog",
        score=0.95,
        bbox=(0.1, 0.2, 0.8, 0.9),
        attributes={"color": "brown"}
    )
    
    assert det.label == "dog"
    assert det.score == 0.95
    assert det.bbox == (0.1, 0.2, 0.8, 0.9)
    assert det.attributes["color"] == "brown"


def test_query_slot_dataclass():
    """Test that QuerySlot dataclass works correctly."""
    from src.retrieval.query_slots import QuerySlot
    
    slot = QuerySlot(
        entity_type="dog",
        attributes={"color": "brown", "action": "running"},
        relations=["on"]
    )
    
    assert slot.entity_type == "dog"
    assert slot.attributes["color"] == "brown"
    assert slot.attributes["action"] == "running"
    assert "on" in slot.relations


def test_postprocess_filter():
    """Test detection filtering."""
    from src.vision.detector_base import Detection
    from src.vision.postprocess import filter_detections
    
    detections = [
        Detection("dog", 0.9, (0, 0, 1, 1)),
        Detection("cat", 0.2, (0, 0, 1, 1)),
        Detection("bird", 0.5, (0, 0, 1, 1)),
    ]
    
    filtered = filter_detections(detections, threshold=0.4)
    assert len(filtered) == 2
    assert filtered[0].label == "dog"
    assert filtered[1].label == "bird"


def test_postprocess_iou():
    """Test IoU computation."""
    from src.vision.postprocess import compute_iou
    
    # Identical boxes
    iou1 = compute_iou((0, 0, 1, 1), (0, 0, 1, 1))
    assert abs(iou1 - 1.0) < 1e-6
    
    # No overlap
    iou2 = compute_iou((0, 0, 0.5, 0.5), (0.6, 0.6, 1, 1))
    assert abs(iou2 - 0.0) < 1e-6
    
    # Partial overlap
    iou3 = compute_iou((0, 0, 0.6, 0.6), (0.4, 0.4, 1, 1))
    assert 0 < iou3 < 1


def test_binding_scorer():
    """Test binding score computation."""
    from src.retrieval.binding_score import BindingScorer
    
    scorer = BindingScorer({"detection_weight": 0.6, "attribute_weight": 0.4})
    
    score = scorer.compute_binding_score(
        detection_score=0.8,
        attribute_scores=[0.9, 0.7]
    )
    
    # Should be weighted average: 0.6*0.8 + 0.4*0.8 = 0.8
    assert 0.0 <= score <= 1.0


def test_score_gating():
    """Test score gating and fusion."""
    from src.retrieval.gating import ScoreGating
    
    gating = ScoreGating({
        "clip_weight": 0.5,
        "kg_weight": 0.3,
        "binding_weight": 0.2,
        "kg_threshold": 0.1
    })
    
    # All scores present
    score1 = gating.fuse_scores(
        clip_score=0.8,
        kg_score=0.6,
        binding_score=0.7
    )
    assert 0.0 <= score1 <= 1.0
    
    # Only CLIP (fallback)
    score2 = gating.fuse_scores(clip_score=0.8)
    assert abs(score2 - 0.8) < 1e-6
    
    # CLIP + KG (no binding)
    score3 = gating.fuse_scores(clip_score=0.8, kg_score=0.6)
    assert 0.0 <= score3 <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
