"""
Performance and correctness tests for Phase 5 Day 5 candidate generation.

Tests:
  - Candidate generation runs without Python loops over entities
  - Average candidate gen time is stable and acceptable (target: <10ms/image excluding disk IO)
  - Output dtype is int32
  - Output length ≤ C_max
  - Stable ordering respects priority merge (caption > visual prior > safe neighbors > global)

Kaggle-first: loads Day 4 artifacts from config paths; skips if absent.
"""

from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pytest


# ============================================================================
# Fixtures for loading artifacts
# ============================================================================

@pytest.fixture(scope="module")
def config_path() -> Path:
    """Get path to entity_graph.yaml config file."""
    # Assume test runs from project root
    return Path("configs/entity_graph.yaml")


@pytest.fixture(scope="module")
def candidate_generator(config_path):
    """
    Load candidate generator with all artifacts.
    
    Skip test if Day 4 artifacts are missing (Kaggle-first design).
    """
    try:
        from src.graph.phrase_candidates_v31 import create_candidate_generator_from_config
    except ImportError as e:
        pytest.skip(f"Cannot import phrase_candidates_v31: {e}")
    
    # Check if config exists
    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")
    
    # Try to create generator (will skip if artifacts missing)
    try:
        gen = create_candidate_generator_from_config(config_path)
        return gen
    except FileNotFoundError as e:
        pytest.skip(f"Day 4 artifacts not found: {e}")
    except Exception as e:
        pytest.fail(f"Failed to create candidate generator: {e}")


@pytest.fixture(scope="module")
def synthetic_workload(candidate_generator):
    """
    Generate synthetic workload for 5,000 images.
    
    Returns:
        Tuple of (cap_ids_list, image_vecs):
          - cap_ids_list: List of 5000 arrays of variable length (3-12) sampled from [0, n_entities)
          - image_vecs: Random float32 [5000, 512] L2-normalized (FAISS IP)
    """
    n_images = 5000
    n_entities = candidate_generator.n_entities
    
    # Seed for determinism
    rng = np.random.RandomState(42)
    
    # Generate caption entity IDs (variable length 3-12)
    cap_ids_list = []
    for _ in range(n_images):
        n_cap = rng.randint(3, 13)
        cap_ids = rng.randint(0, n_entities, size=n_cap, dtype=np.int32)
        cap_ids_list.append(cap_ids)
    
    # Generate random image vectors (512-dim, L2-normalized)
    image_vecs = rng.randn(n_images, 512).astype(np.float32)
    norms = np.linalg.norm(image_vecs, axis=1, keepdims=True)
    image_vecs = image_vecs / (norms + 1e-12)
    
    return cap_ids_list, image_vecs


# ============================================================================
# Performance test
# ============================================================================

def test_candidate_generation_performance(candidate_generator, synthetic_workload):
    """
    Test candidate generation performance on 5,000 synthetic images.
    
    Measures average ms/image excluding disk IO (artifacts already loaded).
    
    Target: <10ms/image
    """
    cap_ids_list, image_vecs = synthetic_workload
    
    # Warmup run (1 image)
    _ = candidate_generator.generate_for_image(
        cap_ids=cap_ids_list[0],
        image_vec=image_vecs[0],
    )
    
    # Timed run (all 5000 images)
    start = time.perf_counter()
    
    results = candidate_generator.generate_for_batch(
        cap_ids_list=cap_ids_list,
        image_vecs=image_vecs,
    )
    
    elapsed = time.perf_counter() - start
    
    # Compute average ms/image
    n_images = len(cap_ids_list)
    avg_ms_per_image = (elapsed * 1000) / n_images
    
    print(f"\n=== Performance Results ===")
    print(f"Total images: {n_images}")
    print(f"Total time: {elapsed:.3f}s")
    print(f"Average ms/image: {avg_ms_per_image:.2f}ms")
    print(f"Target: <10ms/image")
    
    # Assert performance target
    assert avg_ms_per_image < 10.0, (
        f"Candidate generation too slow: {avg_ms_per_image:.2f}ms/image (target: <10ms)"
    )
    
    # Assert we got results for all images
    assert len(results) == n_images, f"Expected {n_images} results, got {len(results)}"


# ============================================================================
# Correctness tests
# ============================================================================

def test_candidate_output_dtype(candidate_generator, synthetic_workload):
    """Test that candidate output has int32 dtype."""
    cap_ids_list, image_vecs = synthetic_workload
    
    # Test first 10 images
    for i in range(10):
        candidates = candidate_generator.generate_for_image(
            cap_ids=cap_ids_list[i],
            image_vec=image_vecs[i],
        )
        assert candidates.dtype == np.int32, f"Expected int32, got {candidates.dtype}"


def test_candidate_output_length_capped(candidate_generator, synthetic_workload):
    """Test that candidate output length ≤ C_max."""
    cap_ids_list, image_vecs = synthetic_workload
    C_max = candidate_generator.cfg.C_max
    
    # Test first 100 images
    for i in range(100):
        candidates = candidate_generator.generate_for_image(
            cap_ids=cap_ids_list[i],
            image_vec=image_vecs[i],
        )
        assert len(candidates) <= C_max, (
            f"Candidate list exceeds C_max: {len(candidates)} > {C_max}"
        )


def test_stable_priority_ordering():
    """
    Test that stable unique preserves priority ordering.
    
    Controlled case: 1 image with known sources.
    """
    from src.graph.phrase_candidates_v31 import (
        PhraseCandidateGeneratorV31,
        CandidateGenConfigV31,
    )
    
    # Create minimal config
    cfg = CandidateGenConfigV31(
        C_max=128,
        visual_prior_topK=50,
        visual_prior_tau_vis=None,  # No tau_vis filtering
        global_topN=50,
        priority_order=["caption", "visual_prior", "safe_neighbors", "global"],
        safe_neighbors_enabled=False,  # Disable for simplicity
        seed_sources=["caption"],
        use_mutual_knn=True,
        require_in_vis_prior=False,
        tau_sim=None,
        entity_faiss_index_path=Path("dummy"),
        entity_neighbors_path=Path("dummy"),
        entity_neighbors_mutual_path=Path("dummy"),
        entity_global_list_path=None,
    )
    
    # Create minimal artifacts (100 entities, K=16)
    n_entities = 100
    K = 16
    neighbors = np.zeros((n_entities, K), dtype=np.int32)
    mutual = np.ones((n_entities, K), dtype=bool)
    
    # Global list: [90, 91, 92, 93, 94]
    global_list = np.array([90, 91, 92, 93, 94], dtype=np.int32)
    
    gen = PhraseCandidateGeneratorV31(
        cfg=cfg,
        neighbors=neighbors,
        mutual=mutual,
        faiss_index=None,  # No visual prior for this test
        global_list=global_list,
        n_entities=n_entities,
    )
    
    # Test case: caption = [1, 2, 3], no visual prior, no safe neighbors
    # Expected: [1, 2, 3, 90, 91, 92, 93, 94] (caption first, then global)
    cap_ids = np.array([1, 2, 3], dtype=np.int32)
    candidates = gen.generate_for_image(cap_ids=cap_ids, image_vec=None)
    
    # Check that caption entities appear first
    assert candidates[0] == 1, f"Expected caption entity 1 first, got {candidates[0]}"
    assert candidates[1] == 2, f"Expected caption entity 2 second, got {candidates[1]}"
    assert candidates[2] == 3, f"Expected caption entity 3 third, got {candidates[2]}"
    
    # Check that global entities appear after caption
    assert 90 in candidates[3:], "Global entity 90 should appear after caption entities"
    assert 91 in candidates[3:], "Global entity 91 should appear after caption entities"
    
    print(f"\n=== Priority Ordering Test Passed ===")
    print(f"Candidates: {candidates[:10]}")


def test_stable_priority_ordering_with_duplicates():
    """
    Test that duplicates across sources are handled correctly (first occurrence wins).
    """
    from src.graph.phrase_candidates_v31 import (
        PhraseCandidateGeneratorV31,
        CandidateGenConfigV31,
    )
    
    # Create minimal config
    cfg = CandidateGenConfigV31(
        C_max=128,
        visual_prior_topK=50,
        visual_prior_tau_vis=None,  # No tau_vis filtering
        global_topN=50,
        priority_order=["caption", "visual_prior", "safe_neighbors", "global"],
        safe_neighbors_enabled=False,
        seed_sources=["caption"],
        use_mutual_knn=True,
        require_in_vis_prior=False,
        tau_sim=None,
        entity_faiss_index_path=Path("dummy"),
        entity_neighbors_path=Path("dummy"),
        entity_neighbors_mutual_path=Path("dummy"),
        entity_global_list_path=None,
    )
    
    # Create minimal artifacts
    n_entities = 100
    K = 16
    neighbors = np.zeros((n_entities, K), dtype=np.int32)
    mutual = np.ones((n_entities, K), dtype=bool)
    
    # Global list: [1, 2, 90, 91] (overlaps with caption)
    global_list = np.array([1, 2, 90, 91], dtype=np.int32)
    
    gen = PhraseCandidateGeneratorV31(
        cfg=cfg,
        neighbors=neighbors,
        mutual=mutual,
        faiss_index=None,
        global_list=global_list,
        n_entities=n_entities,
    )
    
    # Test case: caption = [1, 2, 3]
    # Expected: [1, 2, 3, 90, 91] (duplicates from global removed)
    cap_ids = np.array([1, 2, 3], dtype=np.int32)
    candidates = gen.generate_for_image(cap_ids=cap_ids, image_vec=None)
    
    # Check that duplicates are removed
    assert len(candidates) == len(np.unique(candidates)), "Duplicates not removed"
    
    # Check that caption entities appear first (only once)
    expected_first_three = [1, 2, 3]
    assert list(candidates[:3]) == expected_first_three, (
        f"Expected {expected_first_three}, got {list(candidates[:3])}"
    )
    
    # Check that global entities 90, 91 appear after caption (but 1, 2 should not appear again)
    assert 90 in candidates[3:], "Global entity 90 should appear after caption"
    assert 91 in candidates[3:], "Global entity 91 should appear after caption"
    assert len(candidates) == 5, f"Expected 5 unique candidates, got {len(candidates)}"
    
    print(f"\n=== Duplicate Handling Test Passed ===")
    print(f"Candidates: {candidates}")


def test_no_candidates_edge_case(candidate_generator):
    """Test edge case: empty caption, no visual prior."""
    cap_ids = np.array([], dtype=np.int32)
    candidates = candidate_generator.generate_for_image(cap_ids=cap_ids, image_vec=None)
    
    # Should still get global entities
    assert len(candidates) > 0, "Should get global entities even with empty caption"
    assert candidates.dtype == np.int32, f"Expected int32, got {candidates.dtype}"


def test_large_caption_edge_case(candidate_generator):
    """Test edge case: caption larger than C_max."""
    C_max = candidate_generator.cfg.C_max
    n_entities = candidate_generator.n_entities
    
    # Create caption with C_max + 50 entities
    rng = np.random.RandomState(123)
    cap_ids = rng.choice(n_entities, size=C_max + 50, replace=False).astype(np.int32)
    
    candidates = candidate_generator.generate_for_image(cap_ids=cap_ids, image_vec=None)
    
    # Should be capped at C_max
    assert len(candidates) <= C_max, (
        f"Candidate list exceeds C_max: {len(candidates)} > {C_max}"
    )
    
    # First C_max candidates should be from caption (priority)
    assert np.all(candidates[:C_max] == cap_ids[:C_max]), (
        "Caption entities should appear first (priority)"
    )


# ============================================================================
# Module-level test for stable_unique helper
# ============================================================================

def test_stable_unique_preserve_order():
    """Test stable_unique_preserve_order function."""
    from src.graph.phrase_candidates_v31 import stable_unique_preserve_order
    
    # Test case 1: Simple duplicates
    ids = np.array([3, 1, 4, 1, 5, 9, 3], dtype=np.int32)
    result = stable_unique_preserve_order(ids)
    expected = np.array([3, 1, 4, 5, 9], dtype=np.int32)
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"
    
    # Test case 2: No duplicates
    ids = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    result = stable_unique_preserve_order(ids)
    assert np.array_equal(result, ids), "No duplicates case failed"
    
    # Test case 3: All duplicates
    ids = np.array([7, 7, 7, 7], dtype=np.int32)
    result = stable_unique_preserve_order(ids)
    expected = np.array([7], dtype=np.int32)
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"
    
    # Test case 4: Empty array
    ids = np.array([], dtype=np.int32)
    result = stable_unique_preserve_order(ids)
    assert result.size == 0, "Empty array case failed"
    assert result.dtype == np.int32, f"Expected int32, got {result.dtype}"
    
    print(f"\n=== stable_unique_preserve_order Tests Passed ===")


# ============================================================================
# Module-level test for pixel_anchoring_mask helper
# ============================================================================

def test_pixel_anchoring_mask():
    """Test pixel_anchoring_mask function."""
    from src.graph.phrase_candidates_v31 import pixel_anchoring_mask
    
    # Test case 1: Simple case
    neighbors = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    vis_prior = np.array([2, 5, 10], dtype=np.int32)
    mask = pixel_anchoring_mask(neighbors, vis_prior)
    
    expected = np.array([[False, True, False],
                         [False, True, False]])
    assert np.array_equal(mask, expected), f"Expected {expected}, got {mask}"
    
    # Test case 2: No matches
    neighbors = np.array([[1, 2, 3]], dtype=np.int32)
    vis_prior = np.array([10, 20, 30], dtype=np.int32)
    mask = pixel_anchoring_mask(neighbors, vis_prior)
    
    expected = np.array([[False, False, False]])
    assert np.array_equal(mask, expected), f"Expected {expected}, got {mask}"
    
    # Test case 3: All matches
    neighbors = np.array([[1, 2, 3]], dtype=np.int32)
    vis_prior = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    mask = pixel_anchoring_mask(neighbors, vis_prior)
    
    expected = np.array([[True, True, True]])
    assert np.array_equal(mask, expected), f"Expected {expected}, got {mask}"
    
    print(f"\n=== pixel_anchoring_mask Tests Passed ===")


if __name__ == "__main__":
    # Run pytest programmatically
    pytest.main([__file__, "-v", "-s"])
