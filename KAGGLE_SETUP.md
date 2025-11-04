# Running on Kaggle - Quick Setup

Simple guide to run this project on Kaggle.

**⚠️ GPU Required**: Enable GPU (P100 or T4) in Kaggle notebook settings before running.

## 🚀 Setup for Notebooks (Run These First)

When importing any notebook from the `notebooks/` directory, **run these two cells first** to set up your Kaggle environment:

### Cell 1: Clone Repository and Install Dependencies

```python
# KAGGLE ONLY: Clone repository and install dependencies
!rm -rf hybrid_multimodal_retrieval
!git clone https://github.com/vinhhna/hybrid_multimodal_retrieval.git
%cd hybrid_multimodal_retrieval
!pip install -q transformers accelerate open-clip-torch pyyaml tqdm pillow faiss-cpu
!pip install -e .
```

### Cell 2: Setup Data Paths

```python
# KAGGLE ONLY: Setup data paths
from pathlib import Path

# Set paths based on Kaggle dataset location
IMAGES_DIR = Path('/kaggle/input/flickr30k/data/images')
CAPTIONS_FILE = Path('/kaggle/input/flickr30k/data/results.csv')

# Verify paths
print(f"Images dir exists: {IMAGES_DIR.exists()} - {IMAGES_DIR}")
print(f"Captions file exists: {CAPTIONS_FILE.exists()} - {CAPTIONS_FILE}")

if IMAGES_DIR.exists():
    num_images = len(list(IMAGES_DIR.glob('*.jpg')))
    print(f"Found {num_images} images")
```

**Important:** Run both cells at the start of any notebook for Kaggle compatibility.

---

## 🧪 Running Scripts

### Test BLIP-2 Cross-Encoder (Phase 3 - Week 1)

```python
# Option 1: Auto-detect data paths
!python scripts/test_blip2.py

# Option 2: Specify data directory explicitly
!python scripts/test_blip2.py --data-dir /kaggle/input/flickr30k/data
```

**What it does:** Tests BLIP-2 model loading, single pair scoring, batch scoring, memory handling. Prints comprehensive test report.

---

### Test Stage 1 - CLIP Retrieval (Phase 3 - T2.2)

```python
# Test Stage 1 fast retrieval with CLIP
!python scripts/test_stage1_clip.py --num-test-queries 5

# With custom parameters
!python scripts/test_stage1_clip.py --k1 100 --num-test-queries 10 --data-dir /kaggle/input/flickr30k
```

**What it does:** Tests Stage 1 CLIP retrieval speed and accuracy. Target: <100ms latency.

---

### Test Stage 2 - BLIP-2 Re-ranking (Phase 3 - T2.3)

```python
# Test Stage 2 re-ranking with BLIP-2
!python scripts/test_stage2_blip2.py --num-test-queries 3

# With custom parameters
!python scripts/test_stage2_blip2.py --k1 50 --k2 10 --batch-size 4 --data-dir /kaggle/input/flickr30k
```

**What it does:** Tests Stage 2 BLIP-2 re-ranking of CLIP candidates. Target: <2000ms for 100 candidates.

**Note:** This requires GPU. Use smaller k1 (e.g., 50) for faster testing.

---

### Test Image-to-Image Search (Phase 3 - T2.4)

```python
# Test image similarity search
!python scripts/test_image_to_image.py --num-test-queries 5

# With custom parameters
!python scripts/test_image_to_image.py --k 10 --data-dir /kaggle/input/flickr30k

# With specific query image
!python scripts/test_image_to_image.py --query-image /kaggle/input/flickr30k/images/12345.jpg
```

**What it does:** Tests CLIP-based image-to-image similarity search. Target: <100ms latency.

**Note:** Uses CLIP only (Stage 1). BLIP-2 not used for image similarity.

---

### Test Batch Hybrid Search (Phase 3 - T2.5)

```python
# Quick test - batch vs sequential (recommended)
import sys
sys.path.append('/kaggle/working/hybrid_multimodal_retrieval')
%run scripts/test_batch_quick.py

# Full test suite (comprehensive)
!python scripts/test_batch_search.py
```

**What it does:** Tests optimized batch processing for multiple queries

**Tests performed:**
- Batch vs Sequential comparison (typically 2-3x speedup)
- Scalability with 10, 25, 50 queries
- Different BLIP-2 batch sizes (2, 4, 8)
- Result quality validation

**Expected performance:**
- Batch mode: ~100-200ms per query (for batches of 10+)
- Stage 1: Parallel CLIP encoding for all queries
- Stage 2: Efficient BLIP-2 batching across all candidates

**Note:** Batch processing parallelizes Stage 1 and batches Stage 2 for maximum throughput.

---

### Test Configuration & Optimization (Phase 3 - T2.6)

```python
# Quick test - configuration features (recommended)
import sys
sys.path.append('/kaggle/working/hybrid_multimodal_retrieval')
%run scripts/test_config_quick.py

# Full test suite (comprehensive)
!python scripts/test_configuration.py
```

**What it does:** Tests configuration management and optimization capabilities

**Features tested:**
- Runtime configuration updates (k1, k2, batch_size)
- Configuration validation (rejects invalid configs)
- Cache management (enable/disable/clear)
- Performance profiling (test multiple configs)
- Automatic optimization (find best config for target latency)

**Usage examples:**
```python
from src.retrieval.hybrid_search import HybridSearchEngine

# Update configuration at runtime
engine.update_config(k1=200, k2=20, batch_size=8)

# Enable caching for repeated queries
engine.update_config(use_cache=True)

# Profile different configurations
results = engine.profile_search(
    test_queries=["a dog", "a cat", "a bird"],
    k1_values=[50, 100, 200],
    k2_values=[5, 10, 20],
    batch_sizes=[2, 4, 8]
)

# Auto-optimize for target latency
result = engine.optimize_config(target_latency_ms=400)
engine.update_config(**result['recommended_config'])

# Get cache and performance statistics
stats = engine.get_statistics()
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
print(f"Avg latency: {stats['latency']['total_ms']['mean']:.2f}ms")
```

**Expected results:**
- Config updates take effect immediately
- Cache provides 100-1000x speedup for repeated queries
- Profiling identifies optimal config for your queries
- Auto-optimization finds best config for target latency

**Note:** Use profiling to find optimal k1, k2, and batch_size for your specific use case.

---

### Test Comprehensive Hybrid Search (Phase 3 - T2.7)

```python
# Quick test - comprehensive suite (recommended)
import sys
sys.path.append('/kaggle/working/hybrid_multimodal_retrieval')
%run scripts/test_hybrid_quick.py

# Full test suite (all 10 tests)
!python scripts/test_hybrid_search.py
```

**What it does:** Comprehensive testing of all hybrid search features

**Tests performed:**
1. Stage 1 (CLIP) retrieval - Target <100ms
2. Stage 2 (BLIP-2) re-ranking - Target <2000ms
3. End-to-end hybrid search - Target <2000ms
4. Batch processing efficiency
5. Image-to-image similarity search
6. Configuration tests (k1, k2, batch_size)
7. Latency benchmark (100 queries)
8. CLIP-only vs Hybrid comparison
9. Memory usage tracking
10. Edge cases and error handling

**Expected results:**
- All tests pass
- Stage 1: <100ms average
- End-to-end: <2000ms average
- P95 latency: <2000ms
- P99 latency: <2500ms
- Memory tracking shows resource usage
- Edge cases handled gracefully

**Performance validation:**
```
✓ Stage 1 (CLIP):      ~70-90ms
✓ Stage 2 (BLIP-2):    ~300-400ms (for 100 candidates)
✓ End-to-end:          ~400-500ms total
✓ Benchmark P95:       <2000ms
✓ Benchmark P99:       <2500ms
```

**Note:** This suite validates production-readiness with comprehensive testing across all features.

---

### Evaluate Accuracy (Phase 3 - T2.8)

```python
# Run accuracy evaluation
import sys
sys.path.append('/kaggle/working/hybrid_multimodal_retrieval')
%run scripts/evaluate_accuracy.py
```

**What it does:** Comprehensive accuracy evaluation

**Evaluation process:**
1. Selects 100 diverse test queries from dataset
2. Runs CLIP-only search (baseline)
3. Runs Hybrid search (CLIP + BLIP-2)
4. Calculates accuracy metrics for both methods
5. Compares results with improvement statistics
6. Tracks latency for performance analysis

**Metrics calculated:**
- **Recall@1, Recall@5, Recall@10** - How often ground truth appears in top-k results
- **Mean Reciprocal Rank (MRR)** - Average inverse rank of ground truth
- **Mean Average Precision (MAP)** - Overall precision quality
- **Latency statistics** - Mean, median, P95, P99

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

SUMMARY
✓ Hybrid Search Improvements:
  • Recall@10: +11.4% improvement
  • MRR: 0.5891 (vs 0.5234)
  • Latency overhead: +315.78ms
  • Trade-off: 11.4% better accuracy for 316ms overhead

✓ Performance Targets:
  • Recall@10 >65%: ✓ MET (78.00%)
  • Latency <2000ms: ✓ MET (398.23ms)
```

**Saved outputs:**
- `data/evaluation/accuracy_evaluation_results.json` - Complete results

**Note:** This provides quantitative evidence of hybrid search improvements over baseline CLIP-only search.

---

### Demo Notebook (Phase 3 - T2.9)

**Interactive demo with visualizations:**

```python
# Open the demo notebook
# notebooks/07_hybrid_search_demo.ipynb
```

**Notebook sections:**
1. Setup and imports
2. Load models and data
3. Single query examples
4. Side-by-side CLIP vs Hybrid comparison
5. Batch search demonstrations
6. Image-to-image search
7. Performance benchmarks
8. Accuracy evaluation
9. Visualizations
10. Statistics and caching

**Features demonstrated:**
- ✅ Text-to-image hybrid search
- ✅ Batch processing (2-6x speedup)
- ✅ Image-to-image similarity
- ✅ Method comparisons with visualizations
- ✅ Configuration tuning
- ✅ Performance profiling
- ✅ Cache management
- ✅ Statistics tracking

**On Kaggle:**
- Upload notebook to Kaggle Notebooks
- Set GPU accelerator
- Add Flickr30K as input dataset
- Run all cells

**Note:** The notebook includes all visualizations and interactive examples for complete demonstration of hybrid search capabilities.

---

### Test Search Engine (Full Pipeline)

**Prerequisites:** Must generate embeddings and build indices first.

```python
# Test complete search pipeline
!python scripts/test_search_engine.py --kaggle-input /kaggle/input/flickr30k
```

Replace `/kaggle/input/flickr30k` with your actual dataset path.

---

### Generate Embeddings

```python
# Generate image embeddings
!python scripts/generate_image_embeddings.py

# Generate text embeddings  
!python scripts/generate_text_embeddings.py
```

**Time:** ~10-20 minutes for full dataset

---

### Build FAISS Indices

```python
# Build search indices
!python scripts/build_faiss_indices.py
```

**Time:** ~1-2 minutes

---

## 💾 Save Results

```python
# Redirect script output to file
!python scripts/test_blip2.py > test_results.txt 2>&1

# View the results
!cat test_results.txt

# Files are saved to /kaggle/working/ and can be downloaded
```

---

## ❗ Common Issues

### "CUDA out of memory"
```python
# Use smaller batch size or specify data directory
!python scripts/test_blip2.py --data-dir /kaggle/input/flickr30k/data
```

### "Module not found"
```python
# Make sure you're in the project directory
%cd /kaggle/working/hybrid_multimodal_retrieval

# Reinstall if needed
!pip install -e .
```

### "No images found"
```python
# Specify data directory explicitly
!python scripts/test_blip2.py --data-dir /kaggle/input/flickr30k/data
```

---

## 📚 Additional Resources

- **Project README**: [README.md](README.md)
- **Scripts Documentation**: [scripts/README.md](scripts/README.md)
- **Notebooks Documentation**: [notebooks/README.md](notebooks/README.md)

---

**That's it!** Simple setup, then run any script you need. 🎉
