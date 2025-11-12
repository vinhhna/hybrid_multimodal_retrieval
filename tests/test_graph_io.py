"""
Phase 4 Graph I/O Acceptance Tests

Tests:
- Build tiny graph, save to temp dir, load back
- Assert x_image/x_caption shapes, dtypes (fp32); edge weights fp16; edge_index dtype long
- Assert out-degree per semantic edge type ≤ degree_cap
"""

# === PHASE4: TEST GRAPH IO ===

import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch
import pytest

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from graph.build import (
    l2_normalize,
    build_semantic_edges,
    assemble_hetero_graph,
    GRAPH_DEFAULTS,
)
from graph.store import save_graph, load_graph


@pytest.fixture
def temp_graph_dir():
    """Create a temporary directory for graph storage."""
    temp_dir = tempfile.mkdtemp(prefix="test_graph_")
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_l2_normalize():
    """Test L2 normalization function."""
    X = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    X_norm = l2_normalize(X)
    
    # Check shape preserved
    assert X_norm.shape == X.shape
    
    # Check dtype
    assert X_norm.dtype == np.float32
    
    # Check normalization (row-wise L2 norm should be 1.0 or near-zero for zero vectors)
    norms = np.sqrt((X_norm * X_norm).sum(axis=1))
    assert np.allclose(norms[0], 1.0, atol=1e-6)
    assert np.allclose(norms[2], 1.0, atol=1e-6)
    # Zero vector stays near-zero (with eps)
    assert norms[1] < 1e-6


def test_semantic_edges_shape_and_dtype():
    """Test semantic edges builder returns correct shapes and dtypes."""
    np.random.seed(42)
    N = 50
    D = 512
    X = np.random.randn(N, D).astype(np.float32)
    X = l2_normalize(X)
    
    k_sem = 10
    degree_cap = 8
    
    edge_index, edge_weight = build_semantic_edges(
        node_type="image",
        X=X,
        k_sem=k_sem,
        degree_cap=degree_cap,
    )
    
    # Check edge_index shape
    assert edge_index.shape[0] == 2
    assert edge_index.dtype == torch.long
    
    # Check edge_weight shape and dtype
    assert edge_weight.ndim == 1
    assert edge_weight.dtype == torch.float16
    assert edge_weight.shape[0] == edge_index.shape[1]
    
    # Check degree cap
    src = edge_index[0].numpy()
    degrees = np.bincount(src, minlength=N)
    max_degree = degrees.max()
    assert max_degree <= degree_cap, f"Max degree {max_degree} exceeds cap {degree_cap}"


def test_graph_save_and_load(temp_graph_dir):
    """Test saving and loading a graph with validation."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create small synthetic graph
    N_img = 30
    N_cap = 90  # 3 captions per image
    D = 512
    k_sem = 8
    degree_cap = 8
    
    # Generate embeddings
    x_image = l2_normalize(np.random.randn(N_img, D).astype(np.float32))
    x_caption = l2_normalize(np.random.randn(N_cap, D).astype(np.float32))
    
    # Build semantic edges
    eidx_ii, w_ii = build_semantic_edges("image", x_image, k_sem, degree_cap)
    eidx_cc, w_cc = build_semantic_edges("caption", x_caption, k_sem, degree_cap)
    
    # Build paired edges (caption -> image)
    src_ci = list(range(N_cap))
    tgt_ci = [i // 3 for i in range(N_cap)]  # 3 captions per image
    eidx_ci = torch.tensor([src_ci, tgt_ci], dtype=torch.long)
    
    # Build co-occur edges (captions of same image)
    src_co, tgt_co = [], []
    for img_idx in range(N_img):
        cap_indices = list(range(img_idx * 3, (img_idx + 1) * 3))
        for i in cap_indices:
            for j in cap_indices:
                if i != j:
                    src_co.append(i)
                    tgt_co.append(j)
    eidx_cc_co = torch.tensor([src_co, tgt_co], dtype=torch.long)
    
    # Build node maps
    nid_maps = {
        'image': {f"img_{i}": i for i in range(N_img)},
        'caption': {f"cap_{i}": i for i in range(N_cap)},
    }
    id_maps = {
        'image': {i: f"img_{i}" for i in range(N_img)},
        'caption': {i: f"cap_{i}" for i in range(N_cap)},
    }
    
    # Assemble graph
    graph = assemble_hetero_graph(
        x_image=x_image,
        x_caption=x_caption,
        eidx_ii=eidx_ii,
        w_ii=w_ii,
        eidx_cc=eidx_cc,
        w_cc=w_cc,
        eidx_ci=eidx_ci,
        eidx_cc_co=eidx_cc_co,
        nid_maps=nid_maps,
        id_maps=id_maps,
    )
    
    # Save graph
    save_graph(graph, temp_graph_dir)
    
    # Load graph
    graph_loaded = load_graph(temp_graph_dir)
    
    # === Validate shapes ===
    assert graph_loaded['image'].x.shape == (N_img, D)
    assert graph_loaded['caption'].x.shape == (N_cap, D)
    
    # === Validate dtypes ===
    assert graph_loaded['image'].x.dtype == torch.float32
    assert graph_loaded['caption'].x.dtype == torch.float32
    assert graph_loaded['image', 'sem_sim', 'image'].edge_weight.dtype == torch.float16
    assert graph_loaded['caption', 'sem_sim', 'caption'].edge_weight.dtype == torch.float16
    assert graph_loaded['image', 'sem_sim', 'image'].edge_index.dtype == torch.long
    assert graph_loaded['caption', 'sem_sim', 'caption'].edge_index.dtype == torch.long
    assert graph_loaded['caption', 'paired_with', 'image'].edge_index.dtype == torch.long
    
    # === Validate degree caps ===
    # Image-image semantic edges
    src_ii = graph_loaded['image', 'sem_sim', 'image'].edge_index[0].numpy()
    degrees_ii = np.bincount(src_ii, minlength=N_img)
    assert degrees_ii.max() <= degree_cap
    
    # Caption-caption semantic edges
    src_cc = graph_loaded['caption', 'sem_sim', 'caption'].edge_index[0].numpy()
    degrees_cc = np.bincount(src_cc, minlength=N_cap)
    assert degrees_cc.max() <= degree_cap
    
    # === Validate edge counts ===
    assert graph_loaded['image', 'sem_sim', 'image'].edge_index.shape[1] > 0
    assert graph_loaded['caption', 'sem_sim', 'caption'].edge_index.shape[1] > 0
    assert graph_loaded['caption', 'paired_with', 'image'].edge_index.shape[1] == N_cap
    assert graph_loaded['caption', 'cooccur', 'caption'].edge_index.shape[1] > 0
    
    # === Validate maps ===
    assert len(graph_loaded._nid_maps['image']) == N_img
    assert len(graph_loaded._nid_maps['caption']) == N_cap
    assert len(graph_loaded._id_maps['image']) == N_img
    assert len(graph_loaded._id_maps['caption']) == N_cap


def test_graph_round_trip_values(temp_graph_dir):
    """Test that values are preserved in save/load round-trip."""
    np.random.seed(123)
    torch.manual_seed(123)
    
    N_img = 10
    N_cap = 30
    D = 512
    
    x_image = l2_normalize(np.random.randn(N_img, D).astype(np.float32))
    x_caption = l2_normalize(np.random.randn(N_cap, D).astype(np.float32))
    
    eidx_ii, w_ii = build_semantic_edges("image", x_image, k_sem=5, degree_cap=5)
    eidx_cc, w_cc = build_semantic_edges("caption", x_caption, k_sem=5, degree_cap=5)
    
    src_ci = list(range(N_cap))
    tgt_ci = [i // 3 for i in range(N_cap)]
    eidx_ci = torch.tensor([src_ci, tgt_ci], dtype=torch.long)
    
    nid_maps = {
        'image': {f"i{i}": i for i in range(N_img)},
        'caption': {f"c{i}": i for i in range(N_cap)},
    }
    id_maps = {
        'image': {i: f"i{i}" for i in range(N_img)},
        'caption': {i: f"c{i}" for i in range(N_cap)},
    }
    
    graph = assemble_hetero_graph(
        x_image=x_image,
        x_caption=x_caption,
        eidx_ii=eidx_ii,
        w_ii=w_ii,
        eidx_cc=eidx_cc,
        w_cc=w_cc,
        eidx_ci=eidx_ci,
        eidx_cc_co=None,
        nid_maps=nid_maps,
        id_maps=id_maps,
    )
    
    save_graph(graph, temp_graph_dir)
    graph_loaded = load_graph(temp_graph_dir)
    
    # Check feature values preserved (within fp32 precision)
    assert torch.allclose(graph['image'].x, graph_loaded['image'].x, atol=1e-6)
    assert torch.allclose(graph['caption'].x, graph_loaded['caption'].x, atol=1e-6)
    
    # Check edge indices preserved
    assert torch.equal(
        graph['image', 'sem_sim', 'image'].edge_index,
        graph_loaded['image', 'sem_sim', 'image'].edge_index
    )
    assert torch.equal(
        graph['caption', 'sem_sim', 'caption'].edge_index,
        graph_loaded['caption', 'sem_sim', 'caption'].edge_index
    )
    assert torch.equal(
        graph['caption', 'paired_with', 'image'].edge_index,
        graph_loaded['caption', 'paired_with', 'image'].edge_index
    )
    
    # Check edge weights preserved (within fp16 precision)
    assert torch.allclose(
        graph['image', 'sem_sim', 'image'].edge_weight,
        graph_loaded['image', 'sem_sim', 'image'].edge_weight,
        atol=1e-3  # fp16 has lower precision
    )


def test_missing_files_error(temp_graph_dir):
    """Test that loading from empty directory raises appropriate error."""
    with pytest.raises(FileNotFoundError, match="Missing required graph files"):
        load_graph(temp_graph_dir)


def test_invalid_dtype_error(temp_graph_dir):
    """Test that loading graph with wrong dtype raises error."""
    np.random.seed(42)
    N_img = 10
    N_cap = 30
    D = 512
    
    x_image = l2_normalize(np.random.randn(N_img, D).astype(np.float32))
    x_caption = l2_normalize(np.random.randn(N_cap, D).astype(np.float32))
    
    eidx_ii, w_ii = build_semantic_edges("image", x_image, k_sem=5, degree_cap=5)
    eidx_cc, w_cc = build_semantic_edges("caption", x_caption, k_sem=5, degree_cap=5)
    
    src_ci = list(range(N_cap))
    tgt_ci = [i // 3 for i in range(N_cap)]
    eidx_ci = torch.tensor([src_ci, tgt_ci], dtype=torch.long)
    
    nid_maps = {'image': {f"i{i}": i for i in range(N_img)}, 'caption': {f"c{i}": i for i in range(N_cap)}}
    id_maps = {'image': {i: f"i{i}" for i in range(N_img)}, 'caption': {i: f"c{i}" for i in range(N_cap)}}
    
    graph = assemble_hetero_graph(
        x_image=x_image,
        x_caption=x_caption,
        eidx_ii=eidx_ii,
        w_ii=w_ii,
        eidx_cc=eidx_cc,
        w_cc=w_cc,
        eidx_ci=eidx_ci,
        eidx_cc_co=None,
        nid_maps=nid_maps,
        id_maps=id_maps,
    )
    
    save_graph(graph, temp_graph_dir)
    
    # Corrupt dtype by saving wrong type
    x_corrupt = graph['image'].x.to(torch.float16)  # Wrong dtype
    torch.save(x_corrupt, Path(temp_graph_dir) / 'x_image.pt')
    
    with pytest.raises(ValueError, match="must be float32"):
        load_graph(temp_graph_dir)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
