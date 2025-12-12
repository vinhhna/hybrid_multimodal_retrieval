# Scripts Guide - Helper Tools

Easy guide to all the helper scripts in this project! 🛠️

---

## 🎯 What Are These Scripts?

Think of scripts as little programs that do one specific job. You run them when you need to:
- Build the search database
- Test if things work
- Generate AI embeddings
- Download the dataset

**Most people only need 2-3 of these!** 

---

## 🚀 The Scripts You'll Actually Use

### 1. `test_blip2.py` - Test the Smart AI ⭐

**What it does:** Tests if the BLIP-2 AI model works correctly.

**When to use:** Phase 3, when we add the smarter search.

**Run it:**
```bash
python scripts/test_blip2.py
```

**On Kaggle:**
```bash
python scripts/test_blip2.py --data-dir /kaggle/input/flickr30k/data
```

**What you'll see:**
```
✓ BLIP-2 model loaded
✓ Single pair scoring works
✓ Batch scoring works
✓ Memory handling OK
✓ ALL TESTS PASSED!
```

**If something fails:** Check if you have GPU available. BLIP-2 is a big model!

---

### 2. `build_faiss_indices.py` - Build Search Database 🔨

**What it does:** Creates the fast search database from AI embeddings.

**When to use:** After you've generated embeddings (see notebooks).

**Run it:**
```bash
python scripts/build_faiss_indices.py
```

**What happens:**
1. Loads image embeddings (31,783 images)
2. Loads text embeddings (158,914 captions)
3. Builds fast search index
4. Saves to `data/indices/`

**Time:** 1-2 minutes

**What you'll see:**
```
Building image index...
✓ Image index built: 31,783 vectors
✓ Saved to data/indices/image_index.faiss (62 MB)

Building text index...
✓ Text index built: 158,914 vectors
✓ Saved to data/indices/text_index.faiss (310 MB)

✓ All done!
```

---

### 3. `test_stage1_clip.py` - Test Fast Search (CLIP) ⚡

**What it does:** Tests Stage 1 CLIP retrieval speed and accuracy.

**When to use:** Phase 3, testing two-stage hybrid search (T2.2).

**Run it:**
```bash
python scripts/test_stage1_clip.py --num-test-queries 5
```

**On Kaggle:**
```bash
python scripts/test_stage1_clip.py --data-dir /kaggle/input/flickr30k --num-test-queries 5
```

**What you'll see:**
```
Stage 1 CLIP Retrieval Test
✓ Retrieved 100 candidates in 45ms
✓ Mean latency: 47ms < 100ms target
✓ T2.2 COMPLETE!
```

**Target:** < 100ms latency per query

---

### 4. `test_stage2_blip2.py` - Test Smart Re-ranking 🧠

**What it does:** Tests Stage 2 BLIP-2 re-ranking of CLIP candidates.

**When to use:** Phase 3, testing hybrid search accuracy (T2.3).

**Run it:**
```bash
python scripts/test_stage2_blip2.py --num-test-queries 3
```

**On Kaggle:**
```bash
python scripts/test_stage2_blip2.py --k1 50 --k2 10 --batch-size 4 --data-dir /kaggle/input/flickr30k
```

**Options:**
- `--k1` - Number of Stage 1 candidates (default: 100)
- `--k2` - Number of final results (default: 10)
- `--batch-size` - BLIP-2 batch size (default: 4)

**What you'll see:**
```
Stage 2 BLIP-2 Re-ranking Test
Stage 1: 100 candidates in 45ms
Stage 2: Re-ranked to 10 results in 1200ms
✓ Total: 1245ms < 2000ms target
✓ T2.3 COMPLETE!
```

**Target:** < 2000ms for 100 candidates

**Note:** Requires GPU! Use smaller k1 (50) for faster testing.

---

### 5. `test_image_to_image.py` - Test Image Similarity 🖼️

**What it does:** Tests image-to-image search using CLIP similarity.

**When to use:** Phase 3, testing visual similarity search (T2.4).

**Run it:**
```bash
python scripts/test_image_to_image.py --num-test-queries 5
```

**On Kaggle:**
```bash
python scripts/test_image_to_image.py --data-dir /kaggle/input/flickr30k
```

**With specific query image:**
```bash
python scripts/test_image_to_image.py --query-image data/images/12345.jpg --k 10
```

**What you'll see:**
```
Image-to-Image Search Test
Query: 12345.jpg
✓ Found 10 similar images in 52ms
Top result: 67890.jpg (similarity: 0.9234)
✓ T2.4 COMPLETE!
```

**Target:** < 100ms latency per query

**Note:** Uses CLIP only (Stage 1). BLIP-2 Stage 2 not needed for image similarity.

---

### 5. `test_batch_search.py` - Test Batch Processing 📦🔍

**Purpose:** Tests optimized batch processing for multiple queries (T2.5)

**What it tests:**
- **Batch vs Sequential:** Compares performance (typically 2-3x speedup)
- **Scalability:** Tests with 10, 25, 50 queries
- **Batch Sizes:** Compares BLIP-2 batch_size=2, 4, 8
- **Result Quality:** Validates batch mode produces good results

**Run locally:**
```bash
python scripts/test_batch_search.py
```

**Quick test (Kaggle-compatible):**
```bash
python scripts/test_batch_quick.py
```

**On Kaggle:**
```python
import sys
sys.path.append('/kaggle/working/hybrid_multimodal_retrieval')
%run scripts/test_batch_quick.py
```

**Expected results:**
- Batch processing 2-3x faster than sequential
- Efficient for multi-query scenarios
- Same result quality as single queries

**Performance targets:**
- Stage 1 (CLIP): Parallel encoding for all queries
- Stage 2 (BLIP-2): Batched re-ranking across all candidates
- Overall: ~100-200ms per query for batches of 10+

**Note:** Batch mode parallelizes Stage 1 and efficiently batches Stage 2 for maximum throughput.

---

### 6. `test_configuration.py` - Test Configuration & Optimization ⚙️

**Purpose:** Tests configuration management and optimization features (T2.6)

**What it tests:**
- **Runtime Config Updates:** Change k1, k2, batch_size on the fly
- **Config Validation:** Reject invalid configurations
- **Cache Management:** Enable/disable/clear query cache
- **Performance Profiling:** Test different configurations
- **Auto-Optimization:** Find optimal config for target latency

**Run locally:**
```bash
python scripts/test_configuration.py
```

**Quick test (Kaggle-compatible):**
```bash
python scripts/test_config_quick.py
```

**On Kaggle:**
```python
import sys
sys.path.append('/kaggle/working/hybrid_multimodal_retrieval')
%run scripts/test_config_quick.py
```

**Key features:**
```python
# Update configuration
engine.update_config(k1=200, k2=20, batch_size=8)

# Enable caching
engine.update_config(use_cache=True)

# Profile performance
results = engine.profile_search(
    test_queries=queries,
    k1_values=[50, 100, 200],
    k2_values=[5, 10, 20],
    batch_sizes=[2, 4, 8]
)

# Auto-optimize for target latency
result = engine.optimize_config(target_latency_ms=400)
engine.update_config(**result['recommended_config'])

# Get statistics
stats = engine.get_statistics()
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
```

**Expected results:**
- Config updates take effect immediately
- Invalid configs rejected with rollback
- Cache provides 100-1000x speedup for repeated queries
- Profiling identifies best config for your use case

**Note:** Profiling tests multiple configurations to find optimal settings for your specific queries and latency requirements.

---

### 7. `test_hybrid_search.py` - Comprehensive Test Suite 🧪

**Purpose:** Complete test suite for all hybrid search features (T2.7)

**What it tests:**
1. **Stage 1 (CLIP) Retrieval** - Target: <100ms
2. **Stage 2 (BLIP-2) Re-ranking** - Target: <2000ms
3. **End-to-End Hybrid Search** - Target: <2000ms total
4. **Batch Hybrid Search** - Efficiency test
5. **Image-to-Image Search** - Similarity search
6. **Configuration Tests** - Different k1, k2, batch_size values
7. **Latency Benchmark** - 100 queries for statistics
8. **CLIP-only vs Hybrid** - Performance comparison
9. **Memory Usage Tracking** - Resource monitoring
10. **Edge Cases** - Error handling and special inputs

**Run locally:**
```bash
python scripts/test_hybrid_search.py
```

**Quick test (Kaggle-compatible):**
```bash
python scripts/test_hybrid_quick.py
```

**On Kaggle:**
```python
import sys
sys.path.append('/kaggle/working/hybrid_multimodal_retrieval')
%run scripts/test_hybrid_quick.py
```

**Expected results:**
- All 10 test categories pass
- Stage 1: <100ms average
- End-to-end: <2000ms average
- Latency benchmark: P95 <2000ms
- Memory usage tracked
- Edge cases handled gracefully

**Performance targets:**
```
Stage 1 (CLIP):      < 100ms
Stage 2 (BLIP-2):    < 2000ms (for 100 candidates)
End-to-end:          < 2000ms total
Benchmark P95:       < 2000ms
Benchmark P99:       < 2500ms
```

**Test coverage:**
- ✅ Stage 1 CLIP retrieval
- ✅ Stage 2 BLIP-2 re-ranking
- ✅ Full hybrid pipeline
- ✅ Batch processing
- ✅ Image-to-image search
- ✅ Multiple configurations
- ✅ Performance benchmarking
- ✅ Method comparison
- ✅ Memory tracking
- ✅ Edge case handling

**Note:** This comprehensive suite validates all hybrid search capabilities and ensures production readiness.

---

### 8. `evaluate_accuracy.py` - Accuracy Evaluation 📊

**Purpose:** Comprehensive accuracy evaluation for T2.8

**What it does:**
- Selects 100 diverse test queries from Flickr30K dataset
- Evaluates CLIP-only search (baseline)
- Evaluates Hybrid search (CLIP + BLIP-2)
- Calculates accuracy metrics:
  - **Recall@1, Recall@5, Recall@10**
  - **Mean Reciprocal Rank (MRR)**
  - **Mean Average Precision (MAP)**
- Compares methods with improvement statistics
- Tracks latency for each method
- Validates performance targets

**Run locally:**
```bash
python scripts/evaluate_accuracy.py
```

**On Kaggle:**
```python
import sys
sys.path.append('/kaggle/working/hybrid_multimodal_retrieval')
%run scripts/evaluate_accuracy.py
```

**Expected output:**
```
CLIP-only Results:
  Recall@1:  45.00%
  Recall@5:  62.00%
  Recall@10: 70.00%
  MRR:       0.5234
  MAP:       0.5012
  Latency:   82.45ms (avg)

Hybrid Results:
  Recall@1:  52.00%
  Recall@5:  71.00%
  Recall@10: 78.00%
  MRR:       0.5891
  MAP:       0.5678
  Latency:   398.23ms (avg)

METHOD COMPARISON
  Recall@10 improvement: +11.4%
  Latency overhead: +315.78ms
```

**Metrics explained:**
- **Recall@k:** Percentage of queries where ground truth is in top-k results
- **MRR:** Average of 1/rank for each query (higher = better)
- **MAP:** Average precision across all queries (higher = better)

**Performance targets:**
- ✅ Hybrid Recall@10 > 65%
- ✅ End-to-end latency < 2000ms
- ✅ Measurable improvement over CLIP-only

**Saved outputs:**
- `data/evaluation/accuracy_evaluation_results.json` - Full results

**Note:** This provides quantitative evidence of hybrid search improvements over baseline CLIP-only search.

---

### 9. `test_search_engine.py` - Test Everything 🧪

**What it does:** Makes sure the complete search system works.

**When to use:** After building indices, to verify everything works.

**Run it:**
```bash
python scripts/test_search_engine.py
```

**On Kaggle:**
```bash
python scripts/test_search_engine.py --kaggle-input /kaggle/input/flickr30k
```

**What it tests:**
- ✅ Text → Image search
- ✅ Image → Image search
- ✅ Image → Caption search
- ✅ Batch searches
- ✅ Speed benchmarks

---

## 🧬 Phase 4 Entity Graph Scripts

### `build_entity_vocabulary.py` - Build Entity Vocabulary & Embeddings 📚

**What it does:** Extracts entities from Flickr30K captions and builds CLIP embeddings.

**When to use:** Phase 4 setup - builds entity vocabulary, context, embeddings, and metadata.

**Run it:**
```bash
python -m scripts.build_entity_vocabulary
```

**On Kaggle:**
```bash
python -m scripts.build_entity_vocabulary
```

**What happens:**
1. **Entity Extraction:** Processes all 158,914 captions
   - Uses spaCy for noun phrase extraction (with fallback)
   - Normalizes entity names (lowercase, strip punctuation)
   - Filters by minimum document frequency (min_df=5)
2. **Context Building:** Maps entities to images and captions
3. **Entity Embeddings:** Encodes entities with CLIP text encoder
   - Template: "a photo of {entity_name}"
   - L2-normalized 512-dim vectors
   - Validates for NaNs/Infs
4. **Saves Artifacts:**
   - `data/entities/entity_vocab.json` (~12,872 entities)
   - `data/entities/entity_context.json` (entity→images/captions)
   - `data/entities/entity_embeddings.pt` (CLIP embeddings)
   - `data/entities/entity_meta.json` (metadata with stats)

**Time:** 10-20 minutes (depends on spaCy availability)

**Configuration:** `configs/entity_graph.yaml`
- `min_df`: Minimum caption frequency (default: 5)
- `entity_text_template`: Template for CLIP encoding (default: "a photo of {}")
- `build_entity_embeddings`: Toggle embedding generation (default: true)
- `batch_size`: Encoding batch size (default: 64)

**What you'll see:**
```
======================================================================
FLICKR30K ENTITY VOCABULARY BUILDER
======================================================================

✓ Using local data: data/
📂 Loading Flickr30K dataset...
✓ Adapter ready: 31,783 images

======================================================================
BUILDING ENTITY VOCABULARY
======================================================================
Building entity vocabulary with min_df=5
Processed 158,914 captions from 31,783 images
Found 45,123 unique normalized entities
After filtering (min_df=5): 12,872 entities

✓ Saved vocabulary to: data/entities/entity_vocab.json
✓ Saved context to: data/entities/entity_context.json

======================================================================
BUILDING ENTITY EMBEDDINGS AND METADATA
======================================================================
  Entity text template: 'a photo of {}'
  Batch size: 64
  Total entities: 12,872
  
  Encoding entities in batches...
    Encoded 12,872/12,872 entities
  
  Norms after L2 normalization:
    Mean: 1.0000
    Min:  0.9999
    Max:  1.0001
  
✓ Saved embeddings to: data/entities/entity_embeddings.pt
✓ Saved metadata to: data/entities/entity_meta.json

✅ ENTITY VOCABULARY BUILD COMPLETE
```

**Requirements:**
- CLIP model (ViT-B/32)
- GPU recommended for faster embedding generation
- Optional: spaCy with en_core_web_sm model
  ```bash
  pip install spacy
  python -m spacy download en_core_web_sm
  ```

**Phase 4 Status:**
- ✅ Day 0: Config & skeletons
- ✅ Day 1-2: Entity vocabulary & context
- ✅ Day 3-4: Entity embeddings & metadata
- 🚧 Day 5-7: Graph construction (semantic + co-occurrence edges)
- 🚧 Day 8+: Query enrichment & graph search

---

## 📦 Setup Scripts

### `download_flickr30k.py` - Download Dataset 📥

**What it does:** Downloads the Flickr30K dataset from Kaggle.

**When to use:** If you don't want to download manually.

**Prerequisites:** Kaggle API setup (see [data/README.md](../data/README.md))

**Run it:**
```bash
python scripts/download_flickr30k.py
```

**Note:** Manual download is usually easier! See the [data README](../data/README.md).

---

## 🤖 Advanced Scripts (For Power Users)

### `generate_image_embeddings.py` - Create Image Embeddings

**What it does:** Turns all images into AI embeddings (numbers).

**When to use:** Only if you need to regenerate embeddings.

**Run it:**
```bash
python scripts/generate_image_embeddings.py \
    --images-dir data/images \
    --output data/embeddings/image_embeddings.npy
```

**Options:**
- `--images-dir` - Where your images are
- `--output` - Where to save embeddings
- `--batch-size` - How many at once (default: 32)

**Time:** 10-20 minutes for all 31,783 images

**Better option:** Use the notebook `notebooks/01_clip_embeddings.ipynb` instead!

---

### `generate_text_embeddings.py` - Create Text Embeddings

**What it does:** Turns all captions into AI embeddings.

**Run it:**
```bash
python scripts/generate_text_embeddings.py \
    --captions-file data/results.csv \
    --output data/embeddings/text_embeddings.npy
```

**Time:** 5-10 minutes for all 158,914 captions

**Better option:** Use the notebook `notebooks/01_clip_embeddings.ipynb` instead!

---

## 📝 Typical Workflow

Here's what most people do:

### First Time Setup

```bash
# 1. Download dataset (or do it manually - easier!)
# See data/README.md

# 2. Generate embeddings (use notebook instead!)
# Run: notebooks/01_clip_embeddings.ipynb

# 3. Build search indices
python scripts/build_faiss_indices.py

# 4. Test everything
python scripts/test_search_engine.py

# Done! Now try: notebooks/05_search_demo.ipynb
```

### Testing (Phase 3)

```bash
# Test BLIP-2 model
python scripts/test_blip2.py

# If on Kaggle
python scripts/test_blip2.py --data-dir /kaggle/input/flickr30k/data
```

---

## 🎮 On Kaggle

Running scripts on Kaggle is slightly different:

```python
# 1. Clone the repo
!git clone https://github.com/vinhhna/hybrid_multimodal_retrieval.git
%cd hybrid_multimodal_retrieval

# 2. Install
!pip install -q -e .

# 3. Run any script
!python scripts/test_blip2.py --data-dir /kaggle/input/flickr30k/data
```

**See [KAGGLE_SETUP.md](../KAGGLE_SETUP.md) for complete Kaggle guide!**

---

## 🆘 Troubleshooting

### "Module not found"

**Problem:** Project not installed  
**Solution:**
```bash
pip install -e .
```

### "No images found"

**Problem:** Wrong data path  
**Solution:**
```bash
# Check what's there
ls data/images/

# Or specify path explicitly
python script.py --data-dir /path/to/images
```

### "CUDA out of memory"

**Problem:** GPU doesn't have enough memory  
**Solutions:**
1. Use smaller batch size
2. Use CPU instead (slower but works)
3. Use a machine with more GPU memory

### "File not found: embeddings"

**Problem:** Haven't generated embeddings yet  
**Solution:**
1. Run `notebooks/01_clip_embeddings.ipynb` first
2. Or use the generation scripts above

---

## 📊 What Each Script Creates

| Script | Creates | Size | Location |
|--------|---------|------|----------|
| `generate_image_embeddings.py` | Image embeddings | ~64 MB | `data/embeddings/` |
| `generate_text_embeddings.py` | Text embeddings | ~320 MB | `data/embeddings/` |
| `build_faiss_indices.py` | Search indices | ~400 MB | `data/indices/` |
| `test_*.py` | Nothing (just tests!) | - | - |

---

## 💡 Pro Tips

**Tip 1:** Use notebooks instead of scripts for embeddings  
→ Notebooks show progress and are easier to debug

**Tip 2:** Run tests after each major step  
→ Catch problems early!

**Tip 3:** On Kaggle, specify paths explicitly  
→ Avoids path confusion

**Tip 4:** Check script help with `--help`  
```bash
python scripts/test_blip2.py --help
```

---

## 🎓 When to Use What

**Just starting?**
→ Use notebooks in `notebooks/`, not scripts

**Building search database?**
→ Use `build_faiss_indices.py`

**Testing Phase 3?**
→ Use `test_blip2.py`

**Something not working?**
→ Use `test_search_engine.py` to diagnose

**Running on Kaggle?**
→ Check [KAGGLE_SETUP.md](../KAGGLE_SETUP.md) first

---

## 📚 More Help

- **New to this?** Start with notebooks: `notebooks/05_search_demo.ipynb`
- **On Kaggle?** See: [KAGGLE_SETUP.md](../KAGGLE_SETUP.md)
- **Want details?** See: [API_REFERENCE.md](../API_REFERENCE.md)
- **General help?** See: [README.md](../README.md)

---

**Questions?** Open an issue on GitHub! 🙂
