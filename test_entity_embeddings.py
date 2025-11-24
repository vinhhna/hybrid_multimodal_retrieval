#!/usr/bin/env python
"""
Quick validation script for entity embeddings and metadata.

This script tests the Phase 4 Day 3-4 implementation by:
  1. Loading entity_vocab.json
  2. Loading entity_embeddings.pt
  3. Loading entity_meta.json
  4. Running acceptance tests

Run from repo root:
    python test_entity_embeddings.py
"""

import json
import torch
from pathlib import Path


def main():
    print("=" * 70)
    print("ENTITY EMBEDDINGS VALIDATION")
    print("=" * 70)
    
    # Paths
    project_root = Path(__file__).resolve().parent
    vocab_path = project_root / "data" / "entities" / "entity_vocab.json"
    embeddings_path = project_root / "data" / "entities" / "entity_embeddings.pt"
    meta_path = project_root / "data" / "entities" / "entity_meta.json"
    
    # Check files exist
    print("\n1. Checking file existence...")
    if not vocab_path.exists():
        print(f"  ✗ Missing: {vocab_path}")
        return False
    print(f"  ✓ Found: {vocab_path}")
    
    if not embeddings_path.exists():
        print(f"  ✗ Missing: {embeddings_path}")
        return False
    print(f"  ✓ Found: {embeddings_path}")
    
    if not meta_path.exists():
        print(f"  ✗ Missing: {meta_path}")
        return False
    print(f"  ✓ Found: {meta_path}")
    
    # Load files
    print("\n2. Loading files...")
    with vocab_path.open("r", encoding="utf-8") as f:
        entity_vocab = json.load(f)
    print(f"  ✓ Loaded entity_vocab.json: {len(entity_vocab)} entities")
    
    embeddings = torch.load(str(embeddings_path))
    print(f"  ✓ Loaded entity_embeddings.pt: shape {list(embeddings.shape)}")
    
    with meta_path.open("r", encoding="utf-8") as f:
        entity_meta = json.load(f)
    print(f"  ✓ Loaded entity_meta.json: {len(entity_meta)} entries")
    
    # Validate embeddings
    print("\n3. Validating embeddings...")
    
    # Check dtype
    if embeddings.dtype != torch.float32:
        print(f"  ✗ Wrong dtype: {embeddings.dtype} (expected torch.float32)")
        return False
    print(f"  ✓ dtype: {embeddings.dtype}")
    
    # Check shape
    if embeddings.ndim != 2:
        print(f"  ✗ Wrong ndim: {embeddings.ndim} (expected 2)")
        return False
    print(f"  ✓ ndim: {embeddings.ndim}")
    
    if embeddings.shape[1] != 512:
        print(f"  ✗ Wrong embedding dimension: {embeddings.shape[1]} (expected 512)")
        return False
    print(f"  ✓ Embedding dimension: {embeddings.shape[1]}")
    
    # Check for NaNs/Infs
    if not torch.isfinite(embeddings).all():
        num_bad = (~torch.isfinite(embeddings)).sum().item()
        print(f"  ✗ Contains {num_bad} non-finite values (NaNs/Infs)")
        return False
    print(f"  ✓ No NaNs/Infs")
    
    # Check norms
    norms = torch.linalg.norm(embeddings, dim=1)
    mean_norm = norms.mean().item()
    min_norm = norms.min().item()
    max_norm = norms.max().item()
    
    print(f"\n4. Norm statistics:")
    print(f"  Mean: {mean_norm:.4f}")
    print(f"  Min:  {min_norm:.4f}")
    print(f"  Max:  {max_norm:.4f}")
    
    if not (0.95 <= mean_norm <= 1.05):
        print(f"  ⚠️  Mean norm is outside expected range [0.95, 1.05]")
    else:
        print(f"  ✓ Mean norm is close to 1.0")
    
    # Check consistency
    print("\n5. Checking consistency...")
    
    # Check that number of embeddings matches vocab size
    if embeddings.shape[0] != len(entity_vocab):
        print(f"  ✗ Embeddings count {embeddings.shape[0]} != vocab size {len(entity_vocab)}")
        return False
    print(f"  ✓ Embeddings count matches vocab size")
    
    # Check that meta size matches vocab size
    if len(entity_meta) != len(entity_vocab):
        print(f"  ✗ Metadata count {len(entity_meta)} != vocab size {len(entity_vocab)}")
        return False
    print(f"  ✓ Metadata count matches vocab size")
    
    # Check a few random entities for metadata correctness
    print("\n6. Checking metadata correctness...")
    sample_ids = [0, 1, 2, min(10, len(entity_vocab) - 1)]
    
    for eid in sample_ids:
        # Get entity name from meta
        meta_entry = entity_meta.get(str(eid))
        if meta_entry is None:
            print(f"  ✗ Missing metadata for entity ID {eid}")
            return False
        
        entity_name = meta_entry["name"]
        
        # Check that this name exists in vocab
        if entity_name not in entity_vocab:
            print(f"  ✗ Entity '{entity_name}' from meta not found in vocab")
            return False
        
        vocab_entry = entity_vocab[entity_name]
        
        # Check that IDs match
        if vocab_entry["id"] != eid:
            print(f"  ✗ ID mismatch for '{entity_name}': meta={eid}, vocab={vocab_entry['id']}")
            return False
        
        # Check that stats match
        if meta_entry["df_caption"] != vocab_entry["df_caption"]:
            print(f"  ✗ df_caption mismatch for '{entity_name}'")
            return False
        
        if meta_entry["df_image"] != vocab_entry["df_image"]:
            print(f"  ✗ df_image mismatch for '{entity_name}'")
            return False
        
        if meta_entry["cf"] != vocab_entry["cf"]:
            print(f"  ✗ cf mismatch for '{entity_name}'")
            return False
        
        print(f"  ✓ Entity ID {eid}: '{entity_name}' - metadata matches vocab")
    
    print("\n" + "=" * 70)
    print("✅ ALL VALIDATION TESTS PASSED")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
