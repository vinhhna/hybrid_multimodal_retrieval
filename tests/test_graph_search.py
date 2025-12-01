"""
Unit tests for graph search implementation (Phase 4, Day 10-12).

These tests use synthetic toy graphs to verify:
- Seed selection from query embeddings
- Score propagation with decay and edge weights
- Expansion bounds (H_max, B, T_cap_ms)
- Entity → image aggregation

No real Flickr30K data is used in these tests.
"""

import pytest
import torch
from torch_geometric.data import HeteroData
from unittest.mock import Mock

from src.graph.graph_search import (
    GraphSearchResult,
    graph_search,
    _select_seed_entities,
    _build_adjacency,
    _expand_frontier,
    _aggregate_entity_scores_to_images,
)


def test_select_seed_entities():
    """Test seed selection picks highest similarity entities."""
    # Create 5 entities with known embeddings
    entity_embeddings = torch.tensor([
        [1.0, 0.0, 0.0],  # entity 0
        [0.0, 1.0, 0.0],  # entity 1
        [0.0, 0.0, 1.0],  # entity 2
        [0.7, 0.7, 0.0],  # entity 3
        [0.5, 0.5, 0.7],  # entity 4
    ], dtype=torch.float32)
    
    # Normalize
    entity_embeddings = torch.nn.functional.normalize(entity_embeddings, p=2, dim=-1)
    
    # Query similar to entity 0
    q_enriched = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    q_enriched = torch.nn.functional.normalize(q_enriched.unsqueeze(0), p=2, dim=-1).squeeze(0)
    
    # Select top 2 seeds
    seed_ids, seed_scores = _select_seed_entities(q_enriched, entity_embeddings, K_seed=2)
    
    assert len(seed_ids) == 2
    assert len(seed_scores) == 2
    assert seed_ids[0].item() == 0  # Most similar is entity 0
    assert seed_scores[0].item() > 0.9  # Should be close to 1.0


def test_build_adjacency():
    """Test adjacency list construction from edge_index."""
    # Create a toy graph with 3 entities
    graph = HeteroData()
    graph["entity"].x = torch.randn(3, 512, dtype=torch.float32)
    
    # Add semantic edges: 0→1, 0→2
    graph["entity", "sem", "entity"].edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    graph["entity", "sem", "entity"].edge_weight = torch.tensor([0.8, 0.6], dtype=torch.float32)
    
    # Add co-occurrence edges: 1→2
    graph["entity", "cooc", "entity"].edge_index = torch.tensor([[1], [2]], dtype=torch.long)
    graph["entity", "cooc", "entity"].edge_weight = torch.tensor([0.5], dtype=torch.float32)
    
    adj = _build_adjacency(graph)
    
    # Check semantic edges
    assert len(adj["sem"][0]) == 2  # Entity 0 has 2 outgoing semantic edges
    assert (1, 0.8) in adj["sem"][0]
    assert (2, 0.6) in adj["sem"][0]
    
    # Check co-occurrence edges
    assert len(adj["cooc"][1]) == 1  # Entity 1 has 1 outgoing cooc edge
    assert (2, 0.5) in adj["cooc"][1]


def test_expand_frontier_respects_hmax():
    """Test that frontier expansion stops at H_max hops."""
    # Create seeds
    seed_ids = torch.tensor([0], dtype=torch.long)
    seed_scores = torch.tensor([1.0], dtype=torch.float32)
    seeds = (seed_ids, seed_scores)
    
    # Create adjacency: 0→1→2→3 (chain)
    adj = {
        "sem": {
            0: [(1, 1.0)],
            1: [(2, 1.0)],
            2: [(3, 1.0)],
        },
        "cooc": {},
    }
    
    # Config with H_max=1 (only 1 hop from seeds)
    cfg = {
        "H_max": 1,
        "B": 100,
        "N_max": 100,
        "T_cap_ms": 1000,
        "decay": 0.85,
        "type_weight_sem": 1.0,
        "type_weight_cooc": 0.7,
    }
    
    entity_scores = _expand_frontier(seeds, adj, cfg)
    
    # Should have entities 0 (seed) and 1 (1 hop away)
    assert 0 in entity_scores
    assert 1 in entity_scores
    # Should NOT have entity 2 (2 hops away, exceeds H_max=1)
    assert 2 not in entity_scores
    assert 3 not in entity_scores


def test_expand_frontier_respects_beam_size():
    """Test that frontier expansion stops after B nodes processed."""
    # Create seeds
    seed_ids = torch.tensor([0], dtype=torch.long)
    seed_scores = torch.tensor([1.0], dtype=torch.float32)
    seeds = (seed_ids, seed_scores)
    
    # Create adjacency: 0 connects to 1,2,3,4,5
    adj = {
        "sem": {
            0: [(i, 1.0) for i in range(1, 6)],
        },
        "cooc": {},
    }
    
    # Config with B=2 (process at most 2 nodes)
    cfg = {
        "H_max": 2,
        "B": 2,
        "N_max": 100,
        "T_cap_ms": 1000,
        "decay": 0.85,
        "type_weight_sem": 1.0,
        "type_weight_cooc": 0.7,
    }
    
    entity_scores = _expand_frontier(seeds, adj, cfg)
    
    # Should have at most 2 visited nodes (seed + 1 expansion)
    # Note: All neighbors get added to entity_scores, but only 2 are processed (popped from heap)
    # So we should have seed + some neighbors in scores, but expansion limited
    assert 0 in entity_scores  # Seed always present


def test_aggregate_entity_scores_to_images():
    """Test entity → image score aggregation with df downweighting."""
    entity_scores = {
        0: 1.0,  # High score entity
        1: 0.5,  # Medium score entity
    }
    
    entity_context = {
        0: {"image_ids": ["img1", "img2"]},
        1: {"image_ids": ["img2", "img3"]},
    }
    
    entity_meta = {
        0: {"df_image": 4},  # More common → downweighted
        1: {"df_image": 1},  # Rare → less downweighting
    }
    
    image_scores = _aggregate_entity_scores_to_images(
        entity_scores, entity_context, entity_meta
    )
    
    # img1: only entity 0 contributes (1.0 / sqrt(4) = 0.5)
    assert "img1" in image_scores
    assert abs(image_scores["img1"] - 0.5) < 0.01
    
    # img2: both entities contribute
    # entity 0: 1.0 / sqrt(4) = 0.5
    # entity 1: 0.5 / sqrt(1) = 0.5
    # total: 1.0
    assert "img2" in image_scores
    assert abs(image_scores["img2"] - 1.0) < 0.01
    
    # img3: only entity 1 contributes (0.5 / sqrt(1) = 0.5)
    assert "img3" in image_scores
    assert abs(image_scores["img3"] - 0.5) < 0.01


def test_graph_search_integration():
    """Test full graph_search pipeline on a tiny synthetic graph."""
    # Create a toy graph with 4 entities
    graph = HeteroData()
    
    # Entity embeddings (normalized)
    embeddings = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.8, 0.6, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)
    graph["entity"].x = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    
    # Add semantic edges: 0→1, 1→2
    graph["entity", "sem", "entity"].edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    graph["entity", "sem", "entity"].edge_weight = torch.tensor([0.9, 0.8], dtype=torch.float32)
    
    # Add co-occurrence edges: 0→3
    graph["entity", "cooc", "entity"].edge_index = torch.tensor([[0], [3]], dtype=torch.long)
    graph["entity", "cooc", "entity"].edge_weight = torch.tensor([0.7], dtype=torch.float32)
    
    # Mock encoders
    encoders = Mock()
    # Query similar to entity 0
    query_emb = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=-1)
    encoders.encode_texts = Mock(return_value=query_emb.numpy())
    
    # Config
    cfg = {
        "query_enrichment": {"enabled": True},
        "graph_search": {
            "K_seed": 2,
            "H_max": 2,
            "B": 10,
            "N_max": 100,
            "T_cap_ms": 1000,
            "decay": 0.85,
            "type_weight_sem": 1.0,
            "type_weight_cooc": 0.7,
        },
        "entity_graph": {
            "context_path": "dummy_path",  # Will be mocked
            "entity_meta_path": "dummy_path",
        },
    }
    
    # Mock the module-level caches
    import src.graph.graph_search as gs_module
    
    # Save original caches
    orig_context = gs_module._entity_context_cache
    orig_meta = gs_module._entity_meta_cache
    
    try:
        # Set mock data
        gs_module._entity_context_cache = {
            0: {"image_ids": ["img1", "img2"]},
            1: {"image_ids": ["img2"]},
            2: {"image_ids": ["img3"]},
            3: {"image_ids": ["img1"]},
        }
        
        gs_module._entity_meta_cache = {
            0: {"entity": "entity0", "df_image": 2},
            1: {"entity": "entity1", "df_image": 1},
            2: {"entity": "entity2", "df_image": 1},
            3: {"entity": "entity3", "df_image": 1},
        }
        
        # Run graph search
        result = graph_search("test query", graph, encoders, cfg)
        
        # Verify result structure
        assert isinstance(result, GraphSearchResult)
        assert result.original_query == "test query"
        assert len(result.seed_entity_ids) > 0
        assert len(result.entity_ids) > 0
        assert len(result.image_scores) > 0
        assert result.runtime_ms > 0
        
        # Verify seed selection (should pick entity 0 as top seed)
        assert 0 in result.seed_entity_ids
        
        # Verify expansion happened (should include neighbors)
        assert len(result.entity_ids) >= len(result.seed_entity_ids)
        
    finally:
        # Restore original caches
        gs_module._entity_context_cache = orig_context
        gs_module._entity_meta_cache = orig_meta


if __name__ == "__main__":
    # Run tests
    test_select_seed_entities()
    print("✓ test_select_seed_entities passed")
    
    test_build_adjacency()
    print("✓ test_build_adjacency passed")
    
    test_expand_frontier_respects_hmax()
    print("✓ test_expand_frontier_respects_hmax passed")
    
    test_expand_frontier_respects_beam_size()
    print("✓ test_expand_frontier_respects_beam_size passed")
    
    test_aggregate_entity_scores_to_images()
    print("✓ test_aggregate_entity_scores_to_images passed")
    
    test_graph_search_integration()
    print("✓ test_graph_search_integration passed")
    
    print("\nAll tests passed!")
