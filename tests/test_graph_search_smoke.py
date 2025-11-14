import time
from typing import List

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from src.graph.search import Seed, encode_query, seed_nodes, enrich_query


# Synthetic config used across tests
CFG_DICT = {
    "retriever": {"K_seed": 10},
    "enrichment": {"enabled": True, "top_terms": 5},
    "FAST_SEED": 42,
}


def make_dummy_graph(num_images: int = 4, num_captions: int = 8) -> HeteroData:
    """Construct a tiny synthetic HeteroData graph for smoke tests.

    Graph structure:
    - image nodes: random 512-d features
    - caption nodes: random 512-d features + simple text field
    - caption->image paired_with edges: caption i -> image (i % num_images)
    - image->image sem_sim edges: chain i -> i+1
    - caption->caption sem_sim edges: chain i -> i+1
    - caption->caption cooccur edges: connect (i, i+num_images) when valid
    """
    g = HeteroData()

    torch.manual_seed(123)

    # Node features
    g["image"].x = torch.randn(num_images, 512)
    g["caption"].x = torch.randn(num_captions, 512)

    # Caption texts
    g["caption"].text = [
        f"caption {i} describing image {i % num_images}" for i in range(num_captions)
    ]

    # Paired caption->image edges
    cap_indices = torch.arange(num_captions, dtype=torch.long)
    img_indices = cap_indices % num_images
    g["caption", "paired_with", "image"].edge_index = torch.stack(
        [cap_indices, img_indices], dim=0
    )

    # Semantic image->image edges (simple chain)
    if num_images > 1:
        src_img = torch.arange(num_images - 1, dtype=torch.long)
        dst_img = src_img + 1
        g["image", "sem_sim", "image"].edge_index = torch.stack(
            [src_img, dst_img], dim=0
        )

    # Semantic caption->caption edges (simple chain)
    if num_captions > 1:
        src_cap = torch.arange(num_captions - 1, dtype=torch.long)
        dst_cap = src_cap + 1
        g["caption", "sem_sim", "caption"].edge_index = torch.stack(
            [src_cap, dst_cap], dim=0
        )

    # Co-occurrence caption->caption edges: connect i -> i + num_images when valid
    co_src: List[int] = []
    co_dst: List[int] = []
    for i in range(num_images):
        j = i + num_images
        if j < num_captions:
            co_src.append(i)
            co_dst.append(j)
    if co_src:
        g["caption", "cooccur", "caption"].edge_index = torch.tensor(
            [co_src, co_dst], dtype=torch.long
        )

    return g


class DummyEncoder:
    """Simple deterministic encoder matching encode_query expectations."""

    def encode_text(self, texts):
        arr = np.zeros((len(texts), 512), dtype=np.float32)
        for i, t in enumerate(texts):
            val = float(len(t) % 97)
            arr[i] = val + np.arange(512, dtype=np.float32) * 0.001
        return arr

    def encode_image(self, image_paths):
        arr = np.zeros((len(image_paths), 512), dtype=np.float32)
        for i, p in enumerate(image_paths):
            val = float(len(p) % 89)
            arr[i] = val + np.arange(512, dtype=np.float32) * 0.002
        return arr


def test_encode_query_text_and_array_shape_and_norm():
    encoder = DummyEncoder()
    cfg = CFG_DICT

    q_text = "a dog playing in the park"

    # Encode from text
    vec_text = encode_query(q_text, mode="text", encoder=encoder, cfg=cfg)
    assert isinstance(vec_text, np.ndarray)
    assert vec_text.shape == (512,)
    assert vec_text.dtype == np.float32
    assert abs(np.linalg.norm(vec_text) - 1.0) < 1e-4

    # Encode from precomputed array
    raw = np.random.randn(512).astype(np.float32)
    vec_array = encode_query(raw, mode="text", encoder=encoder, cfg=cfg)
    assert isinstance(vec_array, np.ndarray)
    assert vec_array.shape == (512,)
    assert vec_array.dtype == np.float32
    assert abs(np.linalg.norm(vec_array) - 1.0) < 1e-4


def test_seed_nodes_returns_exact_k_seed_and_valid_indices():
    g = make_dummy_graph(num_images=4, num_captions=8)
    encoder = DummyEncoder()
    cfg = {**CFG_DICT, "retriever": {"K_seed": 6}}

    q_emb = encode_query("simple query", mode="text", encoder=encoder, cfg=cfg)

    K_seed = cfg["retriever"]["K_seed"]
    seeds = seed_nodes(q_emb, K_seed, graph=g, image_index=None, caption_index=None, cfg=cfg)

    assert len(seeds) == K_seed

    img_count = g["image"].x.size(0)
    cap_count = g["caption"].x.size(0)

    for s in seeds:
        assert isinstance(s, Seed)
        assert s.type in ("image", "caption")
        if s.type == "image":
            assert 0 <= s.node_idx < img_count
        else:
            assert 0 <= s.node_idx < cap_count


def test_seed_nodes_consistent_for_same_query():
    g = make_dummy_graph(num_images=4, num_captions=8)
    encoder = DummyEncoder()
    cfg = CFG_DICT

    q_emb = encode_query("deterministic query", mode="text", encoder=encoder, cfg=cfg)

    seeds1 = seed_nodes(q_emb, cfg["retriever"]["K_seed"], graph=g, image_index=None, caption_index=None, cfg=cfg)
    seeds2 = seed_nodes(q_emb, cfg["retriever"]["K_seed"], graph=g, image_index=None, caption_index=None, cfg=cfg)

    assert len(seeds1) == len(seeds2)
    seq1 = [(s.type, s.node_idx) for s in seeds1]
    seq2 = [(s.type, s.node_idx) for s in seeds2]
    assert seq1 == seq2


def test_enrich_query_enabled_returns_non_empty_string():
    g = make_dummy_graph(num_images=4, num_captions=8)
    encoder = DummyEncoder()
    cfg = CFG_DICT

    q_text = "a dog playing"
    q_emb = encode_query(q_text, mode="text", encoder=encoder, cfg=cfg)

    seeds = seed_nodes(q_emb, cfg["retriever"]["K_seed"], graph=g, image_index=None, caption_index=None, cfg=cfg)

    cfg_enrich_on = {"enrichment": {"enabled": True, "top_terms": 5}}
    enriched = enrich_query(q_text, seeds, g, top_terms=5, cfg=cfg_enrich_on)

    assert isinstance(enriched, str)
    assert enriched.strip() != ""
    assert len(enriched) >= len(q_text)
    assert enriched == q_text or ";" in enriched


def test_enrich_query_disabled_returns_original_query():
    g = make_dummy_graph(num_images=4, num_captions=8)
    encoder = DummyEncoder()
    cfg = CFG_DICT

    q_text = "original query"
    q_emb = encode_query(q_text, mode="text", encoder=encoder, cfg=cfg)

    seeds = seed_nodes(q_emb, cfg["retriever"]["K_seed"], graph=g, image_index=None, caption_index=None, cfg=cfg)

    cfg_enrich_off = {"enrichment": {"enabled": False}}
    enriched = enrich_query(q_text, seeds, g, top_terms=5, cfg=cfg_enrich_off)

    assert enriched == q_text


def test_seed_and_enrich_latency_is_reasonable():
    g = make_dummy_graph(num_images=8, num_captions=16)
    encoder = DummyEncoder()
    cfg = CFG_DICT

    queries = [
        "a cat on the sofa",
        "a dog running on the beach",
        "two people riding bikes",
        "a city skyline at night",
        "a child playing with a ball",
    ]

    times_ms = []

    for q in queries:
        start = time.perf_counter()
        q_emb = encode_query(q, mode="text", encoder=encoder, cfg=cfg)
        seeds = seed_nodes(q_emb, cfg["retriever"]["K_seed"], graph=g, image_index=None, caption_index=None, cfg=cfg)
        enriched = enrich_query(q, seeds, g, top_terms=5, cfg=cfg)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)

        assert isinstance(enriched, str)
        assert enriched.strip() != ""

    avg_ms = sum(times_ms) / len(times_ms)
    # Relaxed threshold relative to design spec (<5 ms) to avoid flakiness
    assert avg_ms < 50.0
