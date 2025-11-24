"""
Entity vocabulary and context building for Phase 4 entity-centric graph.

This module implements the core entity extraction, normalization, and vocabulary
building pipeline. It processes Flickr30K captions to:
  - Extract noun phrases and entities (via spaCy or fallback rule-based)
  - Normalize entity strings (lowercase, punctuation removal, etc.)
  - Count frequencies (corpus, caption-level, image-level)
  - Filter by minimum document frequency
  - Build context mappings (entity -> images, captions)
  - Serialize vocabulary and context to JSON artifacts
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple, TYPE_CHECKING

# Type checking imports
if TYPE_CHECKING:
    import torch

# ============================================================================
# Type aliases for Phase 4
# ============================================================================

EntityId = int  # Integer entity ID assigned after filtering by min_df


# ============================================================================
# Normalization regex patterns
# ============================================================================

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


# ============================================================================
# Data structures
# ============================================================================

@dataclass(frozen=True)
class EntityStats:
    """
    Aggregated statistics for a single normalized entity string.

    This is the primary vocabulary entry for Phase 4 entity-centric graph.
    Each entity will be represented as a node in the entity graph.

    Attributes:
        id: Integer entity ID assigned after filtering by min_df.
            This ID is used as the node index in the entity graph.
        df_caption: Number of distinct captions containing this entity.
        df_image: Number of distinct images containing this entity.
        cf: Total count of this entity across all captions (corpus frequency).
    """
    id: int
    df_caption: int
    df_image: int
    cf: int


# ============================================================================
# Entity normalization
# ============================================================================

def normalize_entity(text: str) -> Optional[str]:
    """
    Normalize a raw entity candidate string.

    Steps:
        - Lowercase.
        - Strip leading/trailing whitespace.
        - Remove most punctuation characters.
        - Collapse multiple spaces into one.
        - Return None if the result is empty or too short.

    This function must be deterministic and idempotent.

    Args:
        text: Raw entity string candidate.

    Returns:
        Normalized string, or None if invalid.
    """
    text = text.lower().strip()
    if not text:
        return None
    
    # Remove punctuation
    text = _PUNCT_RE.sub(" ", text)
    
    # Collapse whitespace
    text = _WHITESPACE_RE.sub(" ", text).strip()
    
    # Filter too short
    if len(text) < 2:
        return None
    
    return text


# ============================================================================
# Entity extraction
# ============================================================================

def _try_load_spacy(model_name: str = "en_core_web_sm") -> Optional[Any]:
    """
    Try to load a spaCy language model. Return the nlp object if successful,
    otherwise return None without raising.

    No network calls are allowed; do not attempt to download models.

    Args:
        model_name: Name of the spaCy model to load.

    Returns:
        spaCy nlp object if successful, None otherwise.
    """
    try:
        import spacy
        nlp = spacy.load(model_name)
        return nlp
    except (ImportError, OSError):
        # ImportError: spacy not installed
        # OSError: model not found
        return None


def extract_entities_from_caption(
    caption: str,
    nlp: Optional[Any] = None,
) -> List[str]:
    """
    Extract a list of raw entity candidate strings from a caption.

    If `nlp` is provided and looks like a spaCy language model, use:
        - doc = nlp(caption)
        - noun phrases (doc.noun_chunks)
        - plus possibly individual nouns/proper nouns.
    Otherwise, fall back to a simple rule-based tokenizer.

    Returned strings are raw candidates and still need to be normalized
    via `normalize_entity`.

    Args:
        caption: Input caption text.
        nlp: Optional spaCy language model.

    Returns:
        List of raw entity candidate strings.
    """
    candidates = []
    
    if nlp is not None:
        try:
            doc = nlp(caption)
            
            # Extract noun chunks
            for chunk in doc.noun_chunks:
                candidates.append(chunk.text)
            
            # Extract individual nouns and proper nouns
            for token in doc:
                if token.pos_ in ("NOUN", "PROPN"):
                    candidates.append(token.text)
        except Exception:
            # If spaCy processing fails, fall back to rule-based
            pass
    
    # Fallback or additional candidates from simple tokenization
    if not candidates:
        # Simple rule-based: split on whitespace
        tokens = caption.split()
        for token in tokens:
            # Filter out very short tokens or pure punctuation
            stripped = token.strip()
            if len(stripped) >= 2:
                # Skip tokens that are all punctuation (no alphanumeric chars)
                if not any(ch.isalnum() for ch in stripped):
                    continue
                candidates.append(stripped)
    
    return candidates


# ============================================================================
# Configuration helpers
# ============================================================================

def _get_entity_cfg(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Extract entity graph configuration with defaults.

    Args:
        cfg: Full configuration mapping.

    Returns:
        Dictionary with entity graph config keys and defaults applied.
    """
    entity_cfg = cfg.get("entity_graph", {})
    
    return {
        "min_df": entity_cfg.get("min_df", 5),
        "vocab_path": entity_cfg.get("vocab_path", "data/entities/entity_vocab.json"),
        "context_path": entity_cfg.get("context_path", "data/entities/entity_context.json"),
        "max_samples": entity_cfg.get("max_samples"),
        "verbose": entity_cfg.get("verbose", False),
    }


# ============================================================================
# Main vocabulary builder
# ============================================================================

def build_entity_vocabulary(
    dataset: Any,
    cfg: Mapping[str, Any],
) -> Tuple[Dict[str, EntityStats], Dict[int, Dict[str, Any]]]:
    """
    Build an entity vocabulary and entity context from all captions in the dataset.

    Args:
        dataset: Flickr30K dataset instance from `src.flickr30k.dataset`.
            Must provide a way to iterate over all images and their captions.
        cfg: Configuration mapping. Expected keys:
            - cfg["entity_graph"]["min_df"]: int, minimum caption frequency.
            - cfg["entity_graph"]["vocab_path"]: str, path to vocab JSON.
            - cfg["entity_graph"]["context_path"]: str, path to context JSON.
            - cfg["entity_graph"].get("max_samples"): optional int for debugging.
            - cfg["entity_graph"].get("verbose"): optional bool.

    Returns:
        entity_vocab: Mapping from normalized entity string to EntityStats.
        entity_context: Mapping from entity_id to a dict containing:
            {
                "entity": <entity_str>,
                "image_ids": [ ... ],
                "caption_ids": [ ... ],
            }

    Side effects:
        - Writes `entity_vocab.json` and `entity_context.json` to disk.
        - Prints summary statistics for manual inspection.
    """
    # Resolve configuration
    entity_cfg = _get_entity_cfg(cfg)
    min_df = entity_cfg["min_df"]
    vocab_path = Path(entity_cfg["vocab_path"])
    context_path = Path(entity_cfg["context_path"])
    max_samples = entity_cfg["max_samples"]
    verbose = entity_cfg["verbose"]
    
    print(f"Building entity vocabulary with min_df={min_df}")
    
    # Try to load spaCy model once
    nlp = _try_load_spacy("en_core_web_sm")
    if nlp is not None:
        print("Using spaCy for entity extraction")
    else:
        print("spaCy not available, using rule-based extraction")
    
    # Initialize counters and mappings
    entity_cf = Counter()
    entity_df_caption = Counter()
    entity_df_image = Counter()
    entity_to_images: Dict[str, Set[Any]] = defaultdict(set)
    entity_to_captions: Dict[str, Set[str]] = defaultdict(set)
    
    # Iterate over dataset
    num_samples = len(dataset)
    if max_samples is not None and max_samples > 0:
        num_samples = min(num_samples, max_samples)
        print(f"Processing first {num_samples} samples (debug mode)")
    
    total_captions = 0
    
    for idx in range(num_samples):
        sample = dataset[idx]
        
        # Extract image_id (handle both string and int)
        image_id = sample.get("image_id")
        if image_id is None:
            image_id = sample.get("id", idx)
        
        # Extract captions (handle different key names)
        captions = sample.get("captions")
        if captions is None:
            captions = sample.get("sentences")
        if captions is None:
            captions = sample.get("caption")
        if captions is None:
            continue
        
        # Ensure captions is iterable
        if isinstance(captions, str):
            captions = [captions]
        
        # Track entities seen in this image
        seen_in_image: Set[str] = set()
        
        for caption_idx, caption in enumerate(captions):
            # Handle different caption formats
            if not isinstance(caption, str):
                # Handle caption dicts like {"raw": "...", "sentence": "..."}
                if isinstance(caption, dict):
                    if "raw" in caption:
                        caption = caption["raw"]
                    elif "sentence" in caption:
                        caption = caption["sentence"]
                    else:
                        continue
                else:
                    continue
            
            total_captions += 1
            
            # Construct deterministic caption ID
            caption_id = f"{image_id}#{caption_idx}"
            
            # Extract raw entity candidates
            raw_candidates = extract_entities_from_caption(caption, nlp)
            
            # Track entities in this caption (for DF counting)
            entities_in_caption: Set[str] = set()
            
            for raw_entity in raw_candidates:
                # Normalize
                norm = normalize_entity(raw_entity)
                if norm is None:
                    continue
                
                # Count corpus frequency (every occurrence)
                entity_cf[norm] += 1
                
                # Track for caption-level DF
                entities_in_caption.add(norm)
            
            # Update caption-level DF and context mappings
            for ent in entities_in_caption:
                entity_df_caption[ent] += 1
                entity_to_captions[ent].add(caption_id)
                
                # Update image-level DF only once per image
                if ent not in seen_in_image:
                    entity_df_image[ent] += 1
                    entity_to_images[ent].add(image_id)
                    seen_in_image.add(ent)
        
        # Progress logging
        if verbose and (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{num_samples} images")
    
    print(f"\nProcessed {total_captions} captions from {num_samples} images")
    print(f"Found {len(entity_cf)} unique normalized entities")
    
    # Filter by min_df and assign IDs
    filtered_entities = [
        ent for ent in entity_df_caption.keys()
        if entity_df_caption[ent] >= min_df
    ]
    
    # Sort for deterministic ID assignment
    filtered_entities.sort()
    
    print(f"After filtering (min_df={min_df}): {len(filtered_entities)} entities")
    
    # Build vocabulary and context
    entity_vocab: Dict[str, EntityStats] = {}
    entity_context: Dict[int, Dict[str, Any]] = {}
    
    for ent_id, ent in enumerate(filtered_entities):
        stats = EntityStats(
            id=ent_id,
            df_caption=entity_df_caption[ent],
            df_image=entity_df_image[ent],
            cf=entity_cf[ent],
        )
        entity_vocab[ent] = stats
        
        entity_context[ent_id] = {
            "entity": ent,
            "image_ids": entity_to_images[ent],
            "caption_ids": entity_to_captions[ent],
        }
    
    # Save artifacts
    save_entity_artifacts(entity_vocab, entity_context, vocab_path, context_path)
    
    print(f"\nSaved vocabulary to: {vocab_path}")
    print(f"Saved context to: {context_path}")
    print(f"Final vocabulary size: {len(entity_vocab)}")
    
    # Print sample entries for manual inspection
    if entity_vocab:
        print("\nSample entities (first 10):")
        for i, (ent, stats) in enumerate(list(entity_vocab.items())[:10]):
            print(f"  {stats.id:4d}: '{ent}' (df_cap={stats.df_caption}, df_img={stats.df_image}, cf={stats.cf})")
    
    return entity_vocab, entity_context


# ============================================================================
# Serialization
# ============================================================================

def save_entity_artifacts(
    entity_vocab: Dict[str, EntityStats],
    entity_context: Dict[int, Dict[str, Any]],
    vocab_path: Path,
    context_path: Path,
) -> None:
    """
    Serialize entity vocabulary and context to JSON files.

    The JSON formats must match the Phase 4 plan:

        - entity_vocab.json: entity_name -> {id, df_caption, df_image, cf}
        - entity_context.json: entity_id(str) -> {entity, image_ids, caption_ids}

    Overwrites existing files (idempotent).

    Args:
        entity_vocab: Vocabulary mapping entity strings to statistics.
        entity_context: Context mapping entity IDs to image/caption sets.
        vocab_path: Output path for vocabulary JSON.
        context_path: Output path for context JSON.
    """
    # Ensure output directories exist
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    context_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Serialize vocabulary
    vocab_out = {
        name: {
            "id": stats.id,
            "df_caption": stats.df_caption,
            "df_image": stats.df_image,
            "cf": stats.cf,
        }
        for name, stats in entity_vocab.items()
    }
    
    # Serialize context (convert sets to sorted lists for JSON)
    context_out = {
        str(eid): {
            "entity": ctx["entity"],
            "image_ids": sorted([str(img_id) for img_id in ctx["image_ids"]]),
            "caption_ids": sorted(ctx["caption_ids"]),
        }
        for eid, ctx in entity_context.items()
    }
    
    # Write JSON files
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(vocab_out, f, ensure_ascii=False, indent=2)
    
    with context_path.open("w", encoding="utf-8") as f:
        json.dump(context_out, f, ensure_ascii=False, indent=2)


# ============================================================================
# Entity embeddings and metadata (Phase 4 Day 3-4)
# ============================================================================

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    L2-normalize a tensor along a specified dimension.
    
    Args:
        x: Input tensor.
        dim: Dimension along which to normalize.
        eps: Small constant for numerical stability.
    
    Returns:
        L2-normalized tensor with the same shape as x.
    """
    import torch as torch_module
    
    norm = torch_module.linalg.norm(x, dim=dim, keepdim=True)
    return x / (norm + eps)


def build_entity_embeddings_and_meta(
    text_encoder: Any,
    entity_vocab: Dict[str, EntityStats],
    cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[int, Dict[str, Any]]]:
    """
    Build entity embeddings and metadata from entity vocabulary.
    
    This function encodes each entity name using a CLIP text encoder,
    L2-normalizes the embeddings, and produces:
      - A tensor of shape [N_entities, d_model] with L2-normalized embeddings
      - A metadata dict mapping entity_id to {name, df_caption, df_image, cf}
    
    The embeddings are built deterministically by iterating entities in ID order.
    
    Args:
        text_encoder: CLIP text encoder instance with an encode_text() method
            that accepts a list of strings and returns a tensor [B, d_model].
            Expected to be a BiEncoder or similar wrapper from src.retrieval.
        entity_vocab: Entity vocabulary mapping entity names to EntityStats.
        cfg: Configuration dictionary with keys:
            - entity_graph.entity_text_template: str, template for entity text
            - entity_graph.batch_size: optional int, batch size for encoding
    
    Returns:
        embeddings: Tensor of shape [N_entities, 512], dtype=float32, L2-normalized.
        entity_meta: Dict mapping entity_id (int) to metadata dict with keys:
            {name, df_caption, df_image, cf}
    
    Raises:
        ValueError: If embeddings contain NaNs or Infs.
        RuntimeError: If embedding dimension is not 512.
    """
    import torch as torch_module
    
    # Extract config parameters
    entity_cfg = cfg.get("entity_graph", {})
    template = entity_cfg.get("entity_text_template", "{}")
    batch_size = entity_cfg.get("batch_size", 64)
    
    print("\n" + "=" * 70)
    print("BUILDING ENTITY EMBEDDINGS AND METADATA")
    print("=" * 70)
    print(f"  Entity text template: '{template}'")
    print(f"  Batch size: {batch_size}")
    
    # Build deterministic ID-ordered lists
    num_entities = max(stats.id for stats in entity_vocab.values()) + 1
    id_to_name = [None] * num_entities
    id_to_stats = [None] * num_entities
    
    for name, stats in entity_vocab.items():
        eid = stats.id
        id_to_name[eid] = name
        id_to_stats[eid] = stats
    
    # Validate that all IDs are assigned
    if any(name is None for name in id_to_name):
        raise ValueError("Entity vocabulary has missing IDs")
    
    print(f"\n  Total entities: {num_entities}")
    print(f"  Sample entities (first 5):")
    for i in range(min(5, num_entities)):
        print(f"    ID {i}: '{id_to_name[i]}'")
    
    # Build text inputs using template
    texts = [template.format(name) for name in id_to_name]
    
    # Encode in batches
    print(f"\n  Encoding entities in batches...")
    all_embs = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Call text_encoder.encode_text()
        # Suppress gradients for efficiency
        with torch_module.no_grad():
            batch_emb = text_encoder.encode_text(batch_texts)
        
        # Move to CPU and convert to float32
        batch_emb = batch_emb.cpu().to(torch_module.float32)
        all_embs.append(batch_emb)
        
        if (i // batch_size + 1) % 10 == 0 or (i + batch_size) >= len(texts):
            print(f"    Encoded {min(i + batch_size, len(texts))}/{len(texts)} entities")
    
    # Concatenate all batches
    embeddings = torch_module.cat(all_embs, dim=0)
    
    # Validate shape
    if embeddings.ndim != 2:
        raise RuntimeError(f"Expected 2D embeddings, got shape {embeddings.shape}")
    
    if embeddings.shape[0] != num_entities:
        raise RuntimeError(
            f"Expected {num_entities} embeddings, got {embeddings.shape[0]}"
        )
    
    d_model = embeddings.shape[1]
    if d_model != 512:
        raise RuntimeError(
            f"Expected embedding dimension 512, got {d_model}"
        )
    
    print(f"\n  Embedding shape: {list(embeddings.shape)}")
    
    # Compute norm statistics before normalization
    norms_before = torch_module.linalg.norm(embeddings, dim=1)
    print(f"\n  Norms before L2 normalization:")
    print(f"    Mean: {norms_before.mean().item():.4f}")
    print(f"    Min:  {norms_before.min().item():.4f}")
    print(f"    Max:  {norms_before.max().item():.4f}")
    
    # L2-normalize embeddings
    embeddings = l2_normalize(embeddings, dim=1)
    
    # Make contiguous for efficient storage
    embeddings = embeddings.contiguous()
    
    # Validate for NaNs/Infs
    if not torch_module.isfinite(embeddings).all():
        num_bad = (~torch_module.isfinite(embeddings)).sum().item()
        raise ValueError(
            f"Entity embeddings contain {num_bad} non-finite values (NaNs/Infs)"
        )
    
    # Compute norm statistics after normalization
    norms_after = torch_module.linalg.norm(embeddings, dim=1)
    print(f"\n  Norms after L2 normalization:")
    print(f"    Mean: {norms_after.mean().item():.4f}")
    print(f"    Min:  {norms_after.min().item():.4f}")
    print(f"    Max:  {norms_after.max().item():.4f}")
    
    # Build entity metadata
    entity_meta: Dict[int, Dict[str, Any]] = {}
    for eid, (name, stats) in enumerate(zip(id_to_name, id_to_stats)):
        entity_meta[eid] = {
            "name": name,
            "df_caption": int(stats.df_caption),
            "df_image": int(stats.df_image),
            "cf": int(stats.cf),
        }
    
    print(f"\n  ✓ Built embeddings and metadata for {num_entities} entities")
    print("=" * 70)
    
    return embeddings, entity_meta


def save_entity_embeddings_and_meta(
    embeddings: torch.Tensor,
    entity_meta: Dict[int, Dict[str, Any]],
    embeddings_path: Path,
    meta_path: Path,
) -> None:
    """
    Save entity embeddings and metadata to disk.
    
    Args:
        embeddings: Tensor of shape [N_entities, d_model] with entity embeddings.
        entity_meta: Dictionary mapping entity_id to metadata.
        embeddings_path: Output path for embeddings tensor (will use torch.save).
        meta_path: Output path for metadata JSON.
    
    Side effects:
        Creates parent directories if needed.
        Overwrites existing files (idempotent).
    """
    import torch as torch_module
    
    # Ensure output directories exist
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings tensor
    torch_module.save(embeddings, str(embeddings_path))
    print(f"\n✓ Saved embeddings to: {embeddings_path}")
    
    # Save metadata JSON (convert int keys to strings for JSON compatibility)
    meta_out = {str(eid): meta for eid, meta in entity_meta.items()}
    
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved metadata to: {meta_path}")
