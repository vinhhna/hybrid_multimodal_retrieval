"""
Smoke tests for entity graph construction (Phase 4 Day 5-7).

These tests validate the graph construction logic without owning the canonical
entity_graph.pt artifact. Tests use in-memory graphs or temporary files.

Run with pytest:
    pytest tests/test_entity_graph_build.py -v
    pytest tests/test_entity_graph_build.py::test_entity_graph_build_smoke -v
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch


def _get_project_root() -> Path:
    """Resolve project root from test file location."""
    return Path(__file__).resolve().parents[1]


def _load_config_and_artifacts():
    """
    Load configuration and entity artifacts for testing.
    
    Returns:
        Tuple of (project_root, cfg, cfg_entity_graph, entity_embeddings, entity_context)
    """
    project_root = _get_project_root()
    
    # Ensure src is in path
    if project_root not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from src.graph.config import load_entity_graph_config, get_entity_graph_config
    
    # Load configuration
    config_path = project_root / "configs" / "entity_graph.yaml"
    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")
    
    cfg = load_entity_graph_config(str(config_path))
    cfg_entity_graph = get_entity_graph_config(cfg)
    
    # Load entity embeddings
    embeddings_path = project_root / cfg_entity_graph["entity_embeddings_path"]
    if not embeddings_path.exists():
        pytest.skip(f"Entity embeddings not found: {embeddings_path}. Run build_entity_vocabulary first.")
    
    entity_embeddings = torch.load(str(embeddings_path), map_location="cpu")
    
    # Load entity context
    context_path = project_root / cfg_entity_graph["context_path"]
    if not context_path.exists():
        pytest.skip(f"Entity context not found: {context_path}. Run build_entity_vocabulary first.")
    
    with context_path.open("r", encoding="utf-8") as f:
        entity_context_raw = json.load(f)
    
    # Convert string keys to int keys
    entity_context = {int(eid): ctx for eid, ctx in entity_context_raw.items()}
    
    return project_root, cfg, cfg_entity_graph, entity_embeddings, entity_context


def test_entity_graph_build_smoke():
    """
    Smoke test for entity graph construction.
    
    Validates:
    - Graph can be built from embeddings and context
    - Basic structure and dtypes are correct
    - Node and edge counts are reasonable
    - Save/load round-trip preserves graph structure
    """
    # Load artifacts
    project_root, cfg, cfg_entity_graph, entity_embeddings, entity_context = _load_config_and_artifacts()
    
    from src.graph.build_entity_graph import build_entity_graph, save_entity_graph, load_entity_graph
    
    # Build graph in memory
    graph = build_entity_graph(entity_embeddings, entity_context, cfg)
    
    # Validate node features
    assert graph["entity"].x.size(0) > 0, "Graph should have at least one entity"
    assert graph["entity"].x.size(1) == 512, "Entity embeddings should be 512-dimensional"
    assert graph["entity"].x.dtype == torch.float32, "Entity embeddings should be float32"
    
    # Validate semantic edges
    sem_edge_index = graph["entity", "sem", "entity"].edge_index
    sem_edge_weight = graph["entity", "sem", "entity"].edge_weight
    
    assert sem_edge_index.size(0) == 2, "Edge index should have shape [2, num_edges]"
    assert sem_edge_index.dtype == torch.long, "Edge indices should be long"
    assert sem_edge_index.size(1) > 0, "Should have at least some semantic edges"
    
    if sem_edge_weight.numel() > 0:
        assert sem_edge_weight.dtype in {torch.float32, torch.float16}, \
            f"Edge weights should be float32 or float16, got {sem_edge_weight.dtype}"
        assert sem_edge_weight.size(0) == sem_edge_index.size(1), \
            "Edge weight count should match edge index count"
    
    # Validate co-occurrence edges
    cooc_edge_index = graph["entity", "cooc", "entity"].edge_index
    cooc_edge_weight = graph["entity", "cooc", "entity"].edge_weight
    
    assert cooc_edge_index.size(0) == 2, "Edge index should have shape [2, num_edges]"
    assert cooc_edge_index.dtype == torch.long, "Edge indices should be long"
    
    # Co-occurrence edges may be zero (depending on data), but validate dtype if present
    if cooc_edge_weight.numel() > 0:
        assert cooc_edge_weight.dtype in {torch.float32, torch.float16}, \
            f"Edge weights should be float32 or float16, got {cooc_edge_weight.dtype}"
        assert cooc_edge_weight.size(0) == cooc_edge_index.size(1), \
            "Edge weight count should match edge index count"
    
    # Test save/load round-trip with temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "entity_graph_test.pt"
        
        # Save to temporary location
        save_entity_graph(graph, tmp_path)
        assert tmp_path.exists(), "Graph file should be created"
        
        # Load from temporary location
        loaded_graph = load_entity_graph(tmp_path)
        
        # Validate loaded graph structure matches original
        assert loaded_graph["entity"].x.size(0) == graph["entity"].x.size(0), \
            "Loaded graph should have same number of entities"
        
        assert loaded_graph["entity", "sem", "entity"].edge_index.size(1) == sem_edge_index.size(1), \
            "Loaded graph should have same number of semantic edges"
        
        assert loaded_graph["entity", "cooc", "entity"].edge_index.size(1) == cooc_edge_index.size(1), \
            "Loaded graph should have same number of co-occurrence edges"
        
        # Validate dtypes preserved
        assert loaded_graph["entity"].x.dtype == graph["entity"].x.dtype, \
            "Node feature dtype should be preserved"
        
        assert loaded_graph["entity", "sem", "entity"].edge_index.dtype == sem_edge_index.dtype, \
            "Semantic edge index dtype should be preserved"
        
        assert loaded_graph["entity", "cooc", "entity"].edge_index.dtype == cooc_edge_index.dtype, \
            "Co-occurrence edge index dtype should be preserved"


def test_entity_graph_config_validation():
    """
    Test that config loading works and provides expected keys.
    """
    project_root = _get_project_root()
    
    if project_root not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from src.graph.config import load_entity_graph_config, get_entity_graph_config
    
    config_path = project_root / "configs" / "entity_graph.yaml"
    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")
    
    cfg = load_entity_graph_config(str(config_path))
    cfg_entity_graph = get_entity_graph_config(cfg)
    
    # Validate required config keys
    required_keys = ["k_sem", "degree_cap", "entity_embeddings_path", "context_path", "entity_graph_path"]
    for key in required_keys:
        assert key in cfg_entity_graph, f"Config should contain key: {key}"
    
    # Validate config value types
    assert isinstance(cfg_entity_graph["k_sem"], int), "k_sem should be int"
    assert isinstance(cfg_entity_graph["degree_cap"], int), "degree_cap should be int"
    assert cfg_entity_graph["k_sem"] > 0, "k_sem should be positive"
    assert cfg_entity_graph["degree_cap"] > 0, "degree_cap should be positive"


if __name__ == "__main__":
    # Allow running tests directly with python
    pytest.main([__file__, "-v"])
