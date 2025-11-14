# === PHASE4_DAY5_7_SEARCH_BEGIN ===
"""Phase 4: Minimal graph search scaffolding and enrichment interface.

This module provides:
- encode_query: CLIP-space query encoding (text/image)
- seed_nodes: ANN/brute-force seeding over image and caption embeddings
- enrich_query: optional query enrichment from graph caption neighbors
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Union, Optional

import numpy as np
import torch
from torch_geometric.data import HeteroData

from src.encoders.clip_space import l2_normalize, to_numpy_f32, assert_clip_shape


@dataclass
class Seed:
    type: Literal["image", "caption"]
    node_idx: int
    score: float


def _maybe_set_seed(cfg) -> None:
    """Set numpy RNG seed if FAST_SEED is present in cfg.

    cfg is expected to behave like a dict or have attribute access.
    """
    fast_seed: Optional[int] = None
    # Support both dict-like and attribute-like configs
    if cfg is not None:
        if hasattr(cfg, "get"):
            fast_seed = cfg.get("FAST_SEED", None)
        elif hasattr(cfg, "FAST_SEED"):
            fast_seed = getattr(cfg, "FAST_SEED")
    if fast_seed is not None:
        np.random.seed(int(fast_seed))


def encode_query(
    q: Union[str, np.ndarray],
    mode: Literal["text", "image"],
    encoder,
    cfg,
) -> np.ndarray:
    """Encode a text string or image path using CLIP.

    Return a 512-dim L2-normalized numpy array (float32).

    Args:
        q: Text string or image path, or pre-computed 512-dim embedding.
        mode: "text" or "image".
        encoder: Existing CLIP bi-encoder (repo's BiEncoder-like), expected
                 to expose `encode_text` / `encode_image` returning torch.Tensor
                 or np.ndarray in CLIP space.
        cfg: Config object/dict; may contain `FAST_SEED` for determinism.

    Returns:
        1D numpy.ndarray of shape (512,) dtype float32, L2-normalized.
    """
    _maybe_set_seed(cfg)

    if isinstance(q, np.ndarray):
        emb = to_numpy_f32(q)
    else:
        if mode == "text":
            vec = encoder.encode_text([q])
        elif mode == "image":
            vec = encoder.encode_image([q])
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if torch.is_tensor(vec):
            vec = vec.detach().cpu().numpy()
        emb = to_numpy_f32(vec[0]) if vec.ndim == 2 else to_numpy_f32(vec)

    assert_clip_shape(emb, dim=512)

    emb = emb.astype(np.float32)
    norm = np.linalg.norm(emb) + 1e-12
    emb = emb / norm

    if not np.all(np.isfinite(emb)):
        raise ValueError("encode_query produced non-finite values")

    print(f"[encode] mode={mode} norm={np.linalg.norm(emb):.4f}")
    return emb


def _faiss_search(index, q_emb: np.ndarray, k: int):
    """Small helper to run FAISS ANN search if index is provided.

    Expects index to expose a `.search(X, k)` API, where X is (1, D).
    Returns (scores, ids) with scores as numpy 1D arrays.
    """
    if index is None:
        return None, None
    X = q_emb.reshape(1, -1).astype(np.float32)
    try:
        scores, ids = index.search(X, k)
    except Exception:
        return None, None
    if scores is None or ids is None:
        return None, None
    return scores[0], ids[0]


def _brute_force_cosine(X: np.ndarray, q_emb: np.ndarray, K_seed: int) -> np.ndarray:
    """Brute-force cosine similarity fallback using matrix multiply.

    X and q_emb are assumed L2-normalized.
    Returns indices of top-K_seed items.
    """
    X = X.astype(np.float32, copy=False)
    q_emb = q_emb.astype(np.float32, copy=False)
    scores = X @ q_emb
    K_seed = min(K_seed, scores.shape[0])
    if K_seed <= 0:
        return np.empty((0,), dtype=np.int64)
    topk = np.argpartition(scores, -K_seed)[-K_seed:]
    return topk


def seed_nodes(
    q_emb: np.ndarray,
    K_seed: int,
    graph: HeteroData,
    image_index,
    caption_index,
    cfg,
) -> List[Seed]:
    """Find top-K_seed nearest neighbors in shared CLIP space.

    Searches over both image and caption embeddings.
    Uses FAISS indices when provided (image_index, caption_index) and
    falls back to brute-force cosine similarity otherwise.

    Paired captions for any seeded images are added *before* the final
    ranking; then the unified list is sorted and truncated so that the
    final result always has exactly ``K_seed`` items.

    Args:
        q_emb: Query embedding, shape (512,), L2-normalized float32.
        K_seed: Number of primary seeds to retrieve.
        graph: HeteroData graph containing `image` and `caption` nodes.
        image_index: Optional FAISS index for image embeddings.
        caption_index: Optional FAISS index for caption embeddings.
        cfg: Config object/dict; may contain `FAST_SEED` for determinism.

    Returns:
        List[Seed]: exactly K_seed seeds after ranking. Paired captions for
        seeded images are added before the final ranking; the top-K are kept.
    """
    _maybe_set_seed(cfg)

    q_emb = to_numpy_f32(q_emb)
    assert_clip_shape(q_emb, dim=512)
    if q_emb.ndim != 1:
        q_emb = q_emb.reshape(-1)

    q_emb = q_emb.astype(np.float32)
    norm = np.linalg.norm(q_emb) + 1e-12
    q_emb = q_emb / norm

    img_x = graph["image"].x.detach().cpu().numpy().astype(np.float32)
    cap_x = graph["caption"].x.detach().cpu().numpy().astype(np.float32)

    img_x = l2_normalize(img_x)
    cap_x = l2_normalize(cap_x)

    img_scores, img_ids = _faiss_search(image_index, q_emb, K_seed)
    cap_scores, cap_ids = _faiss_search(caption_index, q_emb, K_seed)

    use_faiss_img = img_scores is not None and img_ids is not None
    use_faiss_cap = cap_scores is not None and cap_ids is not None

    if not use_faiss_img:
        if img_x.shape[0] > 0:
            img_ids = _brute_force_cosine(img_x, q_emb, K_seed)
            img_scores = img_x[img_ids] @ q_emb
        else:
            img_ids = np.empty((0,), dtype=np.int64)
            img_scores = np.empty((0,), dtype=np.float32)

    if not use_faiss_cap:
        if cap_x.shape[0] > 0:
            cap_ids = _brute_force_cosine(cap_x, q_emb, K_seed)
            cap_scores = cap_x[cap_ids] @ q_emb
        else:
            cap_ids = np.empty((0,), dtype=np.int64)
            cap_scores = np.empty((0,), dtype=np.float32)

    base_seeds: List[Seed] = []
    for idx, score in zip(img_ids, img_scores):
        base_seeds.append(Seed(type="image", node_idx=int(idx), score=float(score)))
    for idx, score in zip(cap_ids, cap_scores):
        base_seeds.append(Seed(type="caption", node_idx=int(idx), score=float(score)))

    extra_caps: List[Seed] = []
    edge_dict = graph.edge_index_dict
    if ("caption", "paired_with", "image") in edge_dict:
        eidx = graph["caption", "paired_with", "image"].edge_index
        cap_to_img = {}
        for cap_idx, img_idx in zip(eidx[0].tolist(), eidx[1].tolist()):
            cap_to_img.setdefault(img_idx, []).append(cap_idx)

        existing_pairs = {(s.type, s.node_idx) for s in base_seeds}
        for s in base_seeds:
            if s.type == "image" and s.node_idx in cap_to_img:
                for cap_idx in cap_to_img[s.node_idx]:
                    if ("caption", cap_idx) not in existing_pairs:
                        extra_caps.append(
                            Seed(
                                type="caption",
                                node_idx=int(cap_idx),
                                score=float(s.score),
                            )
                        )
                        existing_pairs.add(("caption", cap_idx))
    seeds: List[Seed] = base_seeds + extra_caps
    seeds.sort(key=lambda s: s.score, reverse=True)
    seeds = seeds[:K_seed]

    print(f"[seed] K_seed={K_seed} final_seeds={len(seeds)}")
    return seeds


def _tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for raw in text.lower().split():
        cleaned = "".join(ch for ch in raw if ch.isalnum())
        if cleaned:
            tokens.append(cleaned)
    return tokens


def enrich_query(
    q_text: str,
    seeds: List[Seed],
    g: HeteroData,
    top_terms: int,
    cfg,
) -> str:
    """Enrich a query using caption neighbors from the graph.

    Collect caption neighbors via semantic and co-occurrence edges, tokenize
    into unigrams/bigrams, select top-N most frequent terms, and append to
    the original query string.

    If enrichment is disabled via cfg (enrichment.enabled=False), returns
    the original query unchanged.
    """
    enabled = True
    if cfg is not None:
        if hasattr(cfg, "get"):
            enabled = bool(cfg.get("enrichment", {}).get("enabled", True))
        else:
            enrichment = getattr(cfg, "enrichment", None)
            if enrichment is not None and hasattr(enrichment, "enabled"):
                enabled = bool(enrichment.enabled)
    if not enabled:
        return q_text

    _maybe_set_seed(cfg)

    caps: List[str] = []
    cap_text_attr = "text"
    if hasattr(g["caption"], cap_text_attr):
        cap_texts_tensor = getattr(g["caption"], cap_text_attr)
        if isinstance(cap_texts_tensor, list):
            cap_texts = cap_texts_tensor
        else:
            cap_texts = list(cap_texts_tensor)
    else:
        cap_texts = ["" for _ in range(g["caption"].num_nodes)]

    edge_dict = g.edge_index_dict

    # Preload commonly used edges once
    ci = (
        g["caption", "paired_with", "image"].edge_index
        if ("caption", "paired_with", "image") in edge_dict
        else None
    )
    ii_sem = (
        g["image", "sem_sim", "image"].edge_index
        if ("image", "sem_sim", "image") in edge_dict
        else None
    )
    cc_sem = (
        g["caption", "sem_sim", "caption"].edge_index
        if ("caption", "sem_sim", "caption") in edge_dict
        else None
    )
    cc_co = (
        g["caption", "cooccur", "caption"].edge_index
        if ("caption", "cooccur", "caption") in edge_dict
        else None
    )

    for seed in seeds:
        if seed.type == "image":
            # Step 1: captions directly paired with this image
            if ci is not None:
                c_src, c_dst = ci
                mask = c_dst == seed.node_idx
                for cap_idx in c_src[mask].tolist():
                    if 0 <= cap_idx < len(cap_texts):
                        caps.append(str(cap_texts[cap_idx]))

            # Step 2: captions from semantic neighbor images
            if ii_sem is not None and ci is not None:
                src_img, dst_img = ii_sem
                img_mask = src_img == seed.node_idx
                neighbor_imgs = dst_img[img_mask]
                c_src, c_dst = ci
                for img_idx in neighbor_imgs.tolist():
                    cap_mask = c_dst == img_idx
                    for cap_idx in c_src[cap_mask].tolist():
                        if 0 <= cap_idx < len(cap_texts):
                            caps.append(str(cap_texts[cap_idx]))

        elif seed.type == "caption":
            # Semantic neighbor captions
            if cc_sem is not None:
                src_cap, dst_cap = cc_sem
                mask = src_cap == seed.node_idx
                for cap_idx in dst_cap[mask].tolist():
                    if 0 <= cap_idx < len(cap_texts):
                        caps.append(str(cap_texts[cap_idx]))

            # Co-occurrence neighbor captions
            if cc_co is not None:
                src_cap, dst_cap = cc_co
                mask = src_cap == seed.node_idx
                for cap_idx in dst_cap[mask].tolist():
                    if 0 <= cap_idx < len(cap_texts):
                        caps.append(str(cap_texts[cap_idx]))

            # Captions paired with the same image(s) as this caption
            if ci is not None:
                c_src, c_dst = ci
                img_mask = c_src == seed.node_idx
                img_idxs = c_dst[img_mask]
                for img_idx in img_idxs.tolist():
                    cap_mask = c_dst == img_idx
                    for cap_idx in c_src[cap_mask].tolist():
                        if 0 <= cap_idx < len(cap_texts):
                            caps.append(str(cap_texts[cap_idx]))

    from collections import Counter

    tokens: List[str] = []
    for c in caps:
        tokens.extend(_tokenize(c))

    bigrams = [tokens[i] + " " + tokens[i + 1] for i in range(len(tokens) - 1)]
    all_terms = tokens + bigrams

    if not all_terms:
        enriched = q_text or ""
        print(f"[enrich] caps={len(caps)} terms={0}")
        return enriched

    terms = Counter(all_terms).most_common(top_terms)
    top_terms_tokens = [t for t, _ in terms]

    enriched = q_text or ""
    for t in top_terms_tokens:
        enriched += ("; " if enriched else "") + t

    print(f"[enrich] caps={len(caps)} terms={len(top_terms_tokens)}")
    return enriched


# === PHASE4_DAY5_7_SEARCH_END ===
