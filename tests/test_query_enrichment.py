"""
Unit tests for Phase 4 query enrichment pipeline.

Tests the enrich_query function and its helper functions using
synthetic fixtures to ensure correct behavior without requiring the full dataset.
"""

import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

# Import functions to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graph.graph_search import (
    EnrichmentResult,
    enrich_query,
    _l2_normalize,
    _collect_candidate_entities,
    _score_entities,
    _top_k_entities,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def synthetic_entity_context():
    """
    Create a small synthetic entity context for testing.
    
    Entities:
      0: "dog" - appears in images 0, 1
      1: "cat" - appears in images 1, 2
      2: "tree" - appears in images 0, 2, 3
    """
    return {
        0: {
            "entity": "dog",
            "image_ids": ["img_0.jpg", "img_1.jpg"],
            "caption_ids": ["cap_0", "cap_1"]
        },
        1: {
            "entity": "cat",
            "image_ids": ["img_1.jpg", "img_2.jpg"],
            "caption_ids": ["cap_2", "cap_3"]
        },
        2: {
            "entity": "tree",
            "image_ids": ["img_0.jpg", "img_2.jpg", "img_3.jpg"],
            "caption_ids": ["cap_0", "cap_4", "cap_5"]
        }
    }


@pytest.fixture
def synthetic_entity_embeddings():
    """
    Create synthetic entity embeddings (3 entities, 512-dim, L2-normalized).
    """
    # Create random embeddings and normalize
    torch.manual_seed(42)
    embeddings = torch.randn(3, 512, dtype=torch.float32)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    return embeddings


@pytest.fixture
def synthetic_entity_meta():
    """
    Create synthetic entity metadata.
    """
    return {
        0: {"entity": "dog", "df_caption": 2, "df_image": 2},
        1: {"entity": "cat", "df_caption": 2, "df_image": 2},
        2: {"entity": "tree", "df_caption": 3, "df_image": 3}
    }


@pytest.fixture
def synthetic_config():
    """
    Create a minimal configuration for testing.
    """
    return {
        "query_enrichment": {
            "enabled": True,
            "K_seed_raw": 32,
            "M_enrich": 2,
            "w_freq": 0.5,
            "w_sim": 0.5,
            "log_examples": False,
            "text_template": "{query}. Related: {entities}",
            "image_template": "photo of {entities}"
        },
        "entity_graph": {
            "entity_embeddings_path": "dummy_path.pt",
            "entity_meta_path": "dummy_meta.json"
        }
    }


@pytest.fixture
def mock_encoder():
    """
    Create a mock BiEncoder that returns deterministic embeddings.
    """
    encoder = MagicMock()
    
    # Mock encode_texts to return L2-normalized random vectors
    def encode_texts(texts, normalize=True, show_progress=False):
        torch.manual_seed(len(texts[0]))  # Seed based on query length for determinism
        embeddings = torch.randn(len(texts), 512, dtype=torch.float32)
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings.numpy()
    
    encoder.encode_texts = encode_texts
    return encoder


@pytest.fixture
def synthetic_seeds():
    """
    Create synthetic CLIP search seeds.
    
    Returns seeds that will map to entities in synthetic_entity_context:
      - img_0.jpg (score 0.9) -> entities 0, 2
      - img_1.jpg (score 0.8) -> entities 0, 1
      - img_2.jpg (score 0.7) -> entities 1, 2
    """
    return [
        ("img_0.jpg", 0.9),
        ("img_1.jpg", 0.8),
        ("img_2.jpg", 0.7),
    ]


# ============================================================================
# Test helper functions
# ============================================================================

def test_l2_normalize_1d():
    """Test L2 normalization for 1D tensors."""
    x = torch.tensor([3.0, 4.0], dtype=torch.float32)
    normalized = _l2_normalize(x)
    
    assert normalized.shape == x.shape
    assert torch.allclose(torch.norm(normalized, p=2), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(normalized, torch.tensor([0.6, 0.8], dtype=torch.float32), atol=1e-6)


def test_l2_normalize_2d():
    """Test L2 normalization for 2D tensors."""
    x = torch.tensor([[3.0, 4.0], [5.0, 12.0]], dtype=torch.float32)
    normalized = _l2_normalize(x)
    
    assert normalized.shape == x.shape
    # Check each row is normalized
    for i in range(normalized.shape[0]):
        assert torch.allclose(torch.norm(normalized[i], p=2), torch.tensor(1.0), atol=1e-6)


def test_collect_candidate_entities(synthetic_entity_context):
    """Test candidate entity collection from seeds."""
    # Build reverse index manually for testing
    from collections import defaultdict
    image_to_entities = defaultdict(set)
    caption_to_entities = defaultdict(set)
    
    for eid, ctx in synthetic_entity_context.items():
        for img_id in ctx["image_ids"]:
            image_to_entities[img_id].add(eid)
        for cap_id in ctx["caption_ids"]:
            caption_to_entities[cap_id].add(eid)
    
    image_to_entities = dict(image_to_entities)
    caption_to_entities = dict(caption_to_entities)
    
    # Seeds that will find entities 0, 1, 2
    seeds = [
        ("img_0.jpg", 0.9),
        ("img_1.jpg", 0.8),
        ("img_2.jpg", 0.7),
    ]
    
    freq = _collect_candidate_entities(seeds, image_to_entities, caption_to_entities)
    
    # Check frequencies (calculated from images only, as per current implementation)
    # img_0 -> entities 0, 2
    # img_1 -> entities 0, 1
    # img_2 -> entities 1, 2
    # So: entity 0 appears 2 times, entity 1 appears 2 times, entity 2 appears 3 times
    assert freq[0] == 2
    assert freq[1] == 2
    assert freq[2] == 3


def test_score_entities(synthetic_entity_embeddings):
    """Test entity scoring by frequency and similarity."""
    # Create a query embedding
    torch.manual_seed(100)
    q0 = torch.randn(512, dtype=torch.float32)
    q0 = torch.nn.functional.normalize(q0, p=2, dim=-1)
    
    # Entity IDs and frequencies
    entity_ids = [0, 1, 2]
    freq = {0: 2, 1: 2, 2: 3}  # entity 2 has highest frequency
    
    # Score entities
    w_freq = 0.5
    w_sim = 0.5
    scores = _score_entities(q0, entity_ids, synthetic_entity_embeddings, freq, w_freq, w_sim)
    
    # Check that all entities are scored
    assert len(scores) == 3
    assert all(eid in scores for eid in entity_ids)
    
    # Check that scores are in valid range
    for score in scores.values():
        assert 0.0 <= score <= 1.0


def test_top_k_entities():
    """Test top-k entity selection with deterministic tie-breaking."""
    scored_entities = {
        0: 0.9,
        1: 0.8,
        2: 0.8,  # Tie with entity 1
        3: 0.7
    }
    
    # Select top 2
    top_ids, top_scores = _top_k_entities(scored_entities, k=2)
    
    assert len(top_ids) == 2
    assert len(top_scores) == 2
    
    # Entity 0 should be first (highest score)
    assert top_ids[0] == 0
    assert top_scores[0] == 0.9
    
    # For tie between 1 and 2, lower entity_id wins (deterministic)
    assert top_ids[1] == 1
    assert top_scores[1] == 0.8


def test_top_k_entities_empty():
    """Test top-k with empty input."""
    top_ids, top_scores = _top_k_entities({}, k=5)
    
    assert top_ids == []
    assert top_scores == []


# ============================================================================
# Test enrich_query_with_seeds
# ============================================================================

@patch('graph.graph_search._load_entity_embeddings')
@patch('graph.graph_search._load_entity_meta')
@patch('graph.graph_search._build_reverse_index')
def test_enrich_query_basic(
    mock_build_reverse_index,
    mock_load_meta,
    mock_load_embeddings,
    synthetic_entity_context,
    synthetic_entity_embeddings,
    synthetic_entity_meta,
    synthetic_config,
    mock_encoder,
    synthetic_seeds
):
    """Test basic query enrichment with synthetic data."""
    # Setup mocks
    mock_load_embeddings.return_value = synthetic_entity_embeddings
    mock_load_meta.return_value = synthetic_entity_meta
    
    # Build reverse index
    from collections import defaultdict
    image_to_entities = defaultdict(set)
    caption_to_entities = defaultdict(set)
    for eid, ctx in synthetic_entity_context.items():
        for img_id in ctx["image_ids"]:
            image_to_entities[img_id].add(eid)
        for cap_id in ctx["caption_ids"]:
            caption_to_entities[cap_id].add(eid)
    
    mock_build_reverse_index.return_value = (dict(image_to_entities), dict(caption_to_entities))
    
    # Run enrichment
    result = enrich_query(
        query="a red car",
        seeds=synthetic_seeds,
        encoders=mock_encoder,
        entity_context=synthetic_entity_context,
        cfg=synthetic_config
    )
    
    # Check result structure
    assert isinstance(result, EnrichmentResult)
    assert result.original_query == "a red car"
    assert result.enriched_query != result.original_query
    assert "a red car" in result.enriched_query
    
    # Check embeddings
    assert result.q0.shape == torch.Size([512])
    assert result.q_enriched.shape == torch.Size([512])
    assert torch.allclose(torch.norm(result.q0, p=2), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(torch.norm(result.q_enriched, p=2), torch.tensor(1.0), atol=1e-5)
    
    # Check entities are selected (up to M_enrich=2)
    assert len(result.entity_ids) > 0
    assert len(result.entity_ids) <= 2
    assert len(result.entity_names) == len(result.entity_ids)
    assert len(result.entity_scores) == len(result.entity_ids)
    
    # Check entity names are from our synthetic set
    valid_names = {"dog", "cat", "tree"}
    for name in result.entity_names:
        assert name in valid_names
    
    # Check enriched query contains entity names
    for name in result.entity_names:
        assert name in result.enriched_query


@patch('graph.graph_search._load_entity_embeddings')
@patch('graph.graph_search._load_entity_meta')
@patch('graph.graph_search._build_reverse_index')
def test_enrich_query_no_entities(
    mock_build_reverse_index,
    mock_load_meta,
    mock_load_embeddings,
    synthetic_entity_context,
    synthetic_entity_embeddings,
    synthetic_entity_meta,
    synthetic_config,
    mock_encoder
):
    """Test query enrichment when no entities can be extracted from seeds."""
    # Setup mocks
    mock_load_embeddings.return_value = synthetic_entity_embeddings
    mock_load_meta.return_value = synthetic_entity_meta
    
    # Empty reverse index (no entities)
    mock_build_reverse_index.return_value = ({}, {})
    
    # Seeds that won't match any entities
    seeds = [("img_unknown.jpg", 0.9)]
    
    # Run enrichment
    result = enrich_query(
        query="a blue sky",
        seeds=seeds,
        encoders=mock_encoder,
        entity_context=synthetic_entity_context,
        cfg=synthetic_config
    )
    
    # Check that query is unchanged
    assert result.original_query == "a blue sky"
    assert result.enriched_query == "a blue sky"
    
    # Check that no entities were selected
    assert len(result.entity_ids) == 0
    assert len(result.entity_names) == 0
    assert len(result.entity_scores) == 0
    
    # q_enriched should be same as q0 (or very close)
    assert torch.allclose(result.q0, result.q_enriched, atol=1e-5)


@patch('graph.graph_search._load_entity_embeddings')
@patch('graph.graph_search._load_entity_meta')
@patch('graph.graph_search._build_reverse_index')
def test_enrich_query_determinism(
    mock_build_reverse_index,
    mock_load_meta,
    mock_load_embeddings,
    synthetic_entity_context,
    synthetic_entity_embeddings,
    synthetic_entity_meta,
    synthetic_config,
    mock_encoder,
    synthetic_seeds
):
    """Test that query enrichment is deterministic."""
    # Setup mocks
    mock_load_embeddings.return_value = synthetic_entity_embeddings
    mock_load_meta.return_value = synthetic_entity_meta
    
    from collections import defaultdict
    image_to_entities = defaultdict(set)
    caption_to_entities = defaultdict(set)
    for eid, ctx in synthetic_entity_context.items():
        for img_id in ctx["image_ids"]:
            image_to_entities[img_id].add(eid)
        for cap_id in ctx["caption_ids"]:
            caption_to_entities[cap_id].add(eid)
    
    mock_build_reverse_index.return_value = (dict(image_to_entities), dict(caption_to_entities))
    
    # Run enrichment twice
    result1 = enrich_query(
        query="test query",
        seeds=synthetic_seeds,
        encoders=mock_encoder,
        entity_context=synthetic_entity_context,
        cfg=synthetic_config
    )
    
    result2 = enrich_query(
        query="test query",
        seeds=synthetic_seeds,
        encoders=mock_encoder,
        entity_context=synthetic_entity_context,
        cfg=synthetic_config
    )
    
    # Check that results are identical
    assert result1.original_query == result2.original_query
    assert result1.enriched_query == result2.enriched_query
    assert result1.entity_ids == result2.entity_ids
    assert result1.entity_names == result2.entity_names
    assert torch.allclose(result1.entity_scores, result2.entity_scores, atol=1e-6)


# ============================================================================
# Integration-style tests
# ============================================================================

@patch('graph.graph_search._load_entity_embeddings')
@patch('graph.graph_search._load_entity_meta')
@patch('graph.graph_search._build_reverse_index')
def test_enrich_query_scores_sorted(
    mock_build_reverse_index,
    mock_load_meta,
    mock_load_embeddings,
    synthetic_entity_context,
    synthetic_entity_embeddings,
    synthetic_entity_meta,
    synthetic_config,
    mock_encoder,
    synthetic_seeds
):
    """Test that entity scores are sorted in descending order."""
    # Setup mocks
    mock_load_embeddings.return_value = synthetic_entity_embeddings
    mock_load_meta.return_value = synthetic_entity_meta
    
    from collections import defaultdict
    image_to_entities = defaultdict(set)
    caption_to_entities = defaultdict(set)
    for eid, ctx in synthetic_entity_context.items():
        for img_id in ctx["image_ids"]:
            image_to_entities[img_id].add(eid)
        for cap_id in ctx["caption_ids"]:
            caption_to_entities[cap_id].add(eid)
    
    mock_build_reverse_index.return_value = (dict(image_to_entities), dict(caption_to_entities))
    
    # Run enrichment
    result = enrich_query(
        query="another test",
        seeds=synthetic_seeds,
        encoders=mock_encoder,
        entity_context=synthetic_entity_context,
        cfg=synthetic_config
    )
    
    # Check that scores are sorted descending
    if len(result.entity_scores) > 1:
        scores_list = result.entity_scores.tolist()
        assert scores_list == sorted(scores_list, reverse=True), \
            "Entity scores should be sorted in descending order"
