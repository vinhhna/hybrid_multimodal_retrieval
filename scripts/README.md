# Scripts Guide - Phase 5

Essential rebuild and evaluation scripts for the Phase 5 hybrid multimodal retrieval system.

---

## 🎯 What Are These Scripts?

These scripts rebuild the core artifacts (embeddings, indices, entity vocab, entity graph) and test the system.

**Workflow:** Download → Generate Embeddings → Build Indices → Build KG → Test

---

## 🚀 Essential Scripts

### 1. `smoke_run.py` - Smoke Test ⭐

**What it does:** Validates the retrieval pipeline is runnable with graceful fallbacks.

**When to use:** After setup, after changes, before deployment.

**Run it:**
```bash
python scripts\\smoke_run.py
python scripts\\smoke_run.py --config configs\\phase5.yaml
python scripts\\smoke_run.py --query "a brown dog running"
```

**What you'll see:**
```
✓ Config loaded
✓ FAISS: Available
✓ KG: Available
✓ CLIP loaded
✓ Query executed
📊 Top-5 Results:
  1. 12345.jpg (score: 0.8234)
  ...
✓ SMOKE TEST PASSED
```

**Fallback behavior:**
- ⚠️ KG missing → Uses CLIP + BLIP-2 only (no crash)
- ⚠️ BLIP-2 unavailable → Skips reranking (no crash)
- ✗ FAISS missing → Fails (cannot retrieve)

---

### 2. `download_flickr30k.py` - Download Dataset

**What it does:** Downloads and extracts the Flickr30K dataset.

**When to use:** First-time setup.

**Run it:**
```bash
python scripts\\download_flickr30k.py
```

**Output:**
- `data/images/` - 31,783 image files
- `data/results.csv` - Image-caption mappings

**Note:** Requires Kaggle API credentials. See KAGGLE_SETUP.md.

---

### 3. `generate_image_embeddings.py` - Generate Image Embeddings

**What it does:** Encodes all images using CLIP into 512-dim embeddings.

**When to use:** After downloading dataset, before building FAISS indices.

**Run it:**
```bash
python scripts\\generate_image_embeddings.py
```

**Output:**
- `data/embeddings/image_embeddings.npy` - NumPy array (31783 × 512)
- `data/embeddings/image_embeddings.json` - Image ID metadata

**Time:** ~30 minutes on GPU, ~2 hours on CPU.

---

### 4. `generate_text_embeddings.py` - Generate Text Embeddings

**What it does:** Encodes all captions using CLIP into 512-dim embeddings.

**When to use:** After downloading dataset, before building FAISS indices.

**Run it:**
```bash
python scripts\\generate_text_embeddings.py
```

**Output:**
- `data/embeddings/text_embeddings.npy` - NumPy array (158914 × 512)
- `data/embeddings/text_embeddings.json` - Caption metadata

**Time:** ~15 minutes on GPU, ~1 hour on CPU.

---

### 5. `build_faiss_indices.py` - Build FAISS Indices

**What it does:** Builds FAISS search indices from embeddings.

**When to use:** After generating embeddings.

**Run it:**
```bash
python scripts\\build_faiss_indices.py
```

**Output:**
- `data/indices/image_index.faiss` - Image search index
- `data/indices/text_index.faiss` - Text search index
- `data/indices/image_index.json` - Image metadata
- `data/indices/text_index.json` - Text metadata

**Time:** ~1 minute.

---

### 6. `build_entity_vocabulary.py` - Build Entity Vocabulary (Phase 4)

**What it does:** Extracts entities from captions, builds vocabulary with CLIP embeddings.

**When to use:** After dataset download, before building entity graph.

**Run it:**
```bash
python scripts\\build_entity_vocabulary.py
```

**Output:**
- `data/entities/entity_vocab.json` - Entity IDs and statistics
- `data/entities/entity_context.json` - Entity-image-caption mappings
- `data/entities/entity_embeddings.pt` - CLIP embeddings (torch tensor)
- `data/entities/entity_meta.json` - Entity metadata

**Time:** ~10 minutes.

---

### 7. `build_entity_graph.py` - Build Entity Graph (Phase 4)

**What it does:** Constructs PyTorch Geometric HeteroData graph with semantic + co-occurrence edges.

**When to use:** After building entity vocabulary.

**Run it:**
```bash
python scripts\\build_entity_graph.py
```

**Output:**
- `data/graph/entity_graph.pt` - Entity graph (PyG HeteroData)

**Time:** ~5-10 minutes.

---

### 8. `evaluate_accuracy.py` - Evaluate Retrieval Accuracy

**What it does:** Benchmarks retrieval accuracy using Recall@K and MRR metrics.

**When to use:** After building all artifacts, to validate performance.

**Run it:**
```bash
python scripts\\evaluate_accuracy.py
python scripts\\evaluate_accuracy.py --mode hybrid  # CLIP + BLIP-2
python scripts\\evaluate_accuracy.py --mode kg      # CLIP + KG + BLIP-2
```

**Output:**
```
Recall@1: 0.45
Recall@5: 0.68
Recall@10: 0.75
MRR: 0.58
```

**Time:** ~30-60 minutes depending on test set size.

---

## 🔄 Full Rebuild Workflow

If you need to regenerate everything from scratch:

```bash
# 1. Download dataset
python scripts\\download_flickr30k.py

# 2. Generate embeddings
python scripts\\generate_image_embeddings.py
python scripts\\generate_text_embeddings.py

# 3. Build FAISS indices
python scripts\\build_faiss_indices.py

# 4. Build entity vocabulary and graph
python scripts\\build_entity_vocabulary.py
python scripts\\build_entity_graph.py

# 5. Test the pipeline
python scripts\\smoke_run.py

# 6. Evaluate accuracy (optional)
python scripts\\evaluate_accuracy.py
```

**Total time:** ~2-3 hours on GPU, ~5-6 hours on CPU.

---

## 📝 Script Status

| Script | Status | Purpose |
|--------|--------|---------|
| `smoke_run.py` | ✅ Ready | End-to-end smoke test |
| `download_flickr30k.py` | ✅ Ready | Dataset download |
| `generate_image_embeddings.py` | ✅ Ready | CLIP image encoding |
| `generate_text_embeddings.py` | ✅ Ready | CLIP text encoding |
| `build_faiss_indices.py` | ✅ Ready | FAISS index construction |
| `build_entity_vocabulary.py` | ✅ Ready | Entity vocab + embeddings |
| `build_entity_graph.py` | ✅ Ready | Entity graph construction |
| `build_full_graph.py` | ✅ Ready | Alternative full graph builder |
| `evaluate_accuracy.py` | ✅ Ready | Retrieval benchmarking |

---

## 🆘 Troubleshooting

**"FAISS index not found"**
- Run: `python scripts\\build_faiss_indices.py`

**"Entity graph not found"**
- Run: `python scripts\\build_entity_vocabulary.py`
- Then: `python scripts\\build_entity_graph.py`

**"Out of memory"**
- Use GPU if available
- Reduce batch size in config files
- Close other applications

**Smoke test fails**
- Check that dataset is downloaded (`data/images/`)
- Check that FAISS index exists (`data/indices/`)
- Read error message carefully

---

## 📚 Configuration Files

Scripts use YAML configs from `configs/`:

- `configs/phase5.yaml` - Phase 5 entity-based retrieval
- `configs/hybrid_config.yaml` - Hybrid CLIP + BLIP-2
- `configs/entity_graph.yaml` - Phase 4 KG config
- `configs/clip_config.yaml` - CLIP model settings

Edit these to customize behavior.

---

## 🎓 For Development

When adding new scripts:
1. Follow the naming convention: `verb_noun.py`
2. Add docstring at top explaining purpose
3. Include `--help` argument support
4. Update this README

---

Phase 5 Baseline - December 2025
