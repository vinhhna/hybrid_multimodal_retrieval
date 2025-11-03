# Phase 3 Detailed Plan - Machine Readable

**Phase:** 3 of 5  
**Name:** Smarter Search (Hybrid Retrieval)  
**Duration:** November 17-30, 2025 (2 weeks)  
**Status:** IN_PROGRESS  
**Current:** Week 1 COMPLETE

---

## Week 1: BLIP-2 Integration (Nov 17-23, 2025)

**Status:** COMPLETE  
**Completion Date:** November 3, 2025

### Tasks

#### T1.1: BLIP-2 Model Setup
- [x] Install transformers, accelerate dependencies
- [x] Create CrossEncoder class (src/retrieval/cross_encoder.py)
- [x] Implement model loading with FP16 support
- [x] Configure for 16GB GPU (P100/T4)
- [x] Test model instantiation

#### T1.2: Scoring Implementation
- [x] Implement score_pair() for single query-image scoring
- [x] Implement score_pairs() for batch scoring
- [x] Add configurable batch sizes (2, 4, 8)
- [x] Implement automatic GPU memory cleanup
- [x] Add error handling and validation

#### T1.3: Testing Suite
- [x] Create scripts/test_blip2.py
- [x] Test 1: Model loading verification
- [x] Test 2: Single pair scoring
- [x] Test 3: Batch scoring (multiple batch sizes)
- [x] Test 4: Memory handling
- [x] Test 5: Score quality (relevant vs irrelevant)
- [x] Automated test report generation

#### T1.4: Kaggle Compatibility
- [x] Add --data-dir argument to test_blip2.py
- [x] Fix import paths (try/except pattern)
- [x] Test on Kaggle GPU (P100/T4)
- [x] Update KAGGLE_SETUP.md

#### T1.5: Documentation
- [x] Create notebooks/06_blip2_exploration.ipynb (20+ cells)
- [x] Add Kaggle setup cells to notebook
- [x] Document API in docstrings
- [x] Update README.md with Phase 3 info
- [x] Create scripts/README.md
- [x] Create notebooks/README.md

### Deliverables
- [x] src/retrieval/cross_encoder.py (210 lines)
- [x] scripts/test_blip2.py (automated tests)
- [x] notebooks/06_blip2_exploration.ipynb (interactive demo)
- [x] Documentation updates (3 new READMEs)

### Metrics
- Tests Passing: 5/5 (100%)
- Model Size: ~15GB
- Single Pair Latency: ~300ms
- Batch Latency: ~150ms per pair (batch_size=4)

---

## Week 2: Hybrid Search Pipeline (Nov 4-10, 2025)

**Status:** NOT_STARTED  
**Start Date:** November 4, 2025  
**Target Completion:** November 10, 2025

### Tasks

#### T2.1: HybridSearchEngine Class Design
- [ ] Create src/retrieval/hybrid_search.py
- [ ] Define HybridSearchEngine class
- [ ] Add __init__(bi_encoder, cross_encoder, image_index, text_index, dataset)
- [ ] Define configuration parameters (k1, k2, batch_size)
- [ ] Add logging and progress tracking

**Files to Create:**
- src/retrieval/hybrid_search.py

**Dependencies:**
- src/retrieval/bi_encoder.py (CLIP - existing)
- src/retrieval/cross_encoder.py (BLIP-2 - Week 1)
- src/retrieval/faiss_index.py (existing)
- src/flickr30k/dataset.py (existing)

#### T2.2: Stage 1 - CLIP Retrieval
- [ ] Implement text_to_image_hybrid_search()
- [ ] Use BiEncoder to get query embedding
- [ ] Search FAISS index for top k1 candidates (k1=100)
- [ ] Return candidate image IDs and CLIP scores
- [ ] Measure Stage 1 latency (target: <100ms)

**Methods to Implement:**
```python
def _stage1_retrieve(self, query: str, k1: int = 100) -> List[Tuple[str, float]]
```

#### T2.3: Stage 2 - BLIP-2 Re-ranking
- [ ] Take k1 candidates from Stage 1
- [ ] Load candidate images from dataset
- [ ] Use CrossEncoder to re-score all candidates
- [ ] Sort by BLIP-2 scores
- [ ] Return top k2 results (k2=10)
- [ ] Measure Stage 2 latency (target: <2s for 100 candidates)

**Methods to Implement:**
```python
def _stage2_rerank(self, query: str, candidates: List[Tuple[str, float]], k2: int = 10) -> List[Tuple[str, float]]
```

#### T2.4: Image-to-Image Hybrid Search
- [ ] Implement image_to_image_hybrid_search()
- [ ] Use BiEncoder to get image embedding
- [ ] Search FAISS index for top k1 candidates
- [ ] Use CrossEncoder to re-rank candidates
- [ ] Return top k2 results

**Methods to Implement:**
```python
def image_to_image_hybrid_search(self, query_image: str, k1: int = 100, k2: int = 10) -> List[Tuple[str, float]]
```

#### T2.5: Batch Hybrid Search
- [ ] Implement batch_text_to_image_search()
- [ ] Process multiple queries efficiently
- [ ] Parallelize Stage 1 (CLIP) operations
- [ ] Batch Stage 2 (BLIP-2) re-ranking
- [ ] Return results for all queries

**Methods to Implement:**
```python
def batch_text_to_image_search(self, queries: List[str], k1: int = 100, k2: int = 10) -> List[List[Tuple[str, float]]]
```

#### T2.6: Configuration & Optimization
- [ ] Add configurable k1 (candidate count: 50, 100, 200)
- [ ] Add configurable k2 (final results: 5, 10, 20)
- [ ] Add configurable batch_size for BLIP-2 (2, 4, 8)
- [ ] Implement caching for frequent queries (optional)
- [ ] Add progress bars for long operations
- [ ] Profile and optimize bottlenecks

**Configuration Parameters:**
```python
k1: int = 100          # Stage 1 candidates
k2: int = 10           # Final results
batch_size: int = 4    # BLIP-2 batch size
use_cache: bool = False
show_progress: bool = True
```

#### T2.7: Testing Suite
- [ ] Create scripts/test_hybrid_search.py
- [ ] Test Stage 1 retrieval (CLIP)
- [ ] Test Stage 2 re-ranking (BLIP-2)
- [ ] Test end-to-end hybrid search
- [ ] Test batch hybrid search
- [ ] Benchmark latency (target: <2s total)
- [ ] Compare results: CLIP-only vs Hybrid vs BLIP-2-only

**Test Cases:**
```
1. Single query hybrid search
2. Batch queries hybrid search
3. Image-to-image hybrid search
4. Different k1 values (50, 100, 200)
5. Different k2 values (5, 10, 20)
6. Performance benchmark (100 queries)
7. Memory usage tracking
```

#### T2.8: Accuracy Evaluation
- [ ] Select 100 test queries
- [ ] Run CLIP-only search (baseline)
- [ ] Run hybrid search
- [ ] Run BLIP-2-only search (upper bound)
- [ ] Calculate Recall@10 for each method
- [ ] Calculate MRR (Mean Reciprocal Rank)
- [ ] Calculate latency statistics
- [ ] Create comparison table

**Metrics to Track:**
```
- Recall@1, Recall@5, Recall@10
- Mean Reciprocal Rank (MRR)
- Mean Average Precision (MAP)
- Latency (mean, p50, p95, p99)
- Memory usage
```

#### T2.9: Documentation & Demo
- [ ] Create notebooks/07_hybrid_search_demo.ipynb
- [ ] Show side-by-side comparisons (CLIP vs Hybrid)
- [ ] Visualize top-10 results for sample queries
- [ ] Add performance benchmarks
- [ ] Show latency breakdown (Stage 1 vs Stage 2)
- [ ] Update README.md with hybrid search usage
- [ ] Update API_REFERENCE.md

**Notebook Sections:**
```
1. Setup and imports
2. Load models and data
3. Single query example
4. Side-by-side comparison
5. Batch search example
6. Performance benchmarking
7. Accuracy evaluation
8. Visualizations
```

### Deliverables
- [ ] src/retrieval/hybrid_search.py (~300 lines)
- [ ] scripts/test_hybrid_search.py (~200 lines)
- [ ] notebooks/07_hybrid_search_demo.ipynb (15+ cells)
- [ ] Documentation updates
- [ ] Performance benchmark results

### Success Criteria
- [ ] End-to-end search completes in <2 seconds
- [ ] Recall@10 improved by 15-20% vs CLIP-only
- [ ] All tests passing
- [ ] Demo notebook runs successfully
- [ ] Kaggle compatible

### Metrics Targets
- Total Latency: <2000ms (Stage 1: <100ms, Stage 2: <1900ms)
- Recall@10: >65% (baseline CLIP: ~50%)
- Memory Usage: <20GB GPU
- Throughput: >0.5 queries/second

---

## Week 3: Polish & Integration (Nov 11-17, 2025)

**Status:** NOT_STARTED  
**Start Date:** November 11, 2025  
**Target Completion:** November 17, 2025

### Tasks

#### T3.1: Integration with Existing Code
- [ ] Update src/retrieval/__init__.py exports
- [ ] Add HybridSearchEngine to main API
- [ ] Update MultimodalSearchEngine to use hybrid search
- [ ] Ensure backward compatibility with existing code
- [ ] Update all example notebooks

**Files to Update:**
- src/retrieval/__init__.py
- src/retrieval/search_engine.py (optional: add hybrid mode)
- notebooks/05_search_demo.ipynb (add hybrid examples)

#### T3.2: Configuration Management
- [ ] Create configs/hybrid_config.yaml
- [ ] Add configuration loading
- [ ] Document all configuration options
- [ ] Add configuration validation
- [ ] Create config examples for different use cases

**Config Structure:**
```yaml
hybrid_search:
  stage1:
    k1: 100
    model: "ViT-B-32"
  stage2:
    k2: 10
    model: "Salesforce/blip2-flan-t5-xl"
    batch_size: 4
    use_fp16: true
  performance:
    use_cache: false
    show_progress: true
```

#### T3.3: Error Handling & Robustness
- [ ] Add comprehensive error handling
- [ ] Handle OOM errors gracefully
- [ ] Add fallback to CLIP-only if BLIP-2 fails
- [ ] Add retry logic for model loading
- [ ] Validate input parameters
- [ ] Add helpful error messages

#### T3.4: CLI Tool
- [ ] Create scripts/hybrid_search_cli.py
- [ ] Add command-line interface for searches
- [ ] Support text-to-image search
- [ ] Support image-to-image search
- [ ] Add output formatting options (JSON, table)
- [ ] Add --save-results flag

**CLI Usage:**
```bash
python scripts/hybrid_search_cli.py "a dog playing" --k1 100 --k2 10
python scripts/hybrid_search_cli.py --image photo.jpg --k2 20
python scripts/hybrid_search_cli.py "sunset beach" --output results.json
```

#### T3.5: Performance Optimization
- [ ] Profile code with cProfile
- [ ] Optimize BLIP-2 batch processing
- [ ] Add GPU memory pooling
- [ ] Implement query result caching
- [ ] Optimize image loading pipeline
- [ ] Reduce memory copies

**Optimization Targets:**
- Stage 2 latency: <1500ms (from <1900ms)
- Memory usage: <18GB (from <20GB)
- Cache hit speedup: >10x for repeated queries

#### T3.6: Additional Features
- [ ] Add score fusion options (weighted average, rank fusion)
- [ ] Add diversity re-ranking (MMR - Maximal Marginal Relevance)
- [ ] Add filtering by metadata (optional)
- [ ] Add explain_results() for debugging
- [ ] Add visualize_scores() for analysis

**Score Fusion Methods:**
```python
- "replace": Use only BLIP-2 scores (default)
- "weighted": Combine CLIP and BLIP-2 scores
- "rank_fusion": Fuse based on ranks
```

#### T3.7: Comprehensive Testing
- [ ] Add unit tests for all methods
- [ ] Add integration tests
- [ ] Add regression tests
- [ ] Test edge cases (empty results, OOM, etc.)
- [ ] Test with different datasets (small, medium, large)
- [ ] Stress test with 1000+ queries

#### T3.8: Final Documentation
- [ ] Complete API documentation
- [ ] Add architecture diagram
- [ ] Create usage examples
- [ ] Update QUICK_START.md
- [ ] Update CHANGELOG.md
- [ ] Create HYBRID_SEARCH_GUIDE.md (detailed guide)

**Documentation Structure:**
```
HYBRID_SEARCH_GUIDE.md:
  - Overview
  - Architecture
  - Usage examples
  - Configuration
  - Performance tuning
  - Troubleshooting
  - FAQ
```

#### T3.9: Quality Assurance
- [ ] Code review and refactoring
- [ ] Check code style (PEP 8)
- [ ] Add type hints
- [ ] Update docstrings
- [ ] Run linting (flake8, pylint)
- [ ] Check test coverage (target: >80%)

#### T3.10: Kaggle Deployment
- [ ] Test all scripts on Kaggle
- [ ] Update KAGGLE_SETUP.md with hybrid search
- [ ] Create Kaggle-specific notebook
- [ ] Test with different GPU types (P100, T4)
- [ ] Verify all imports work on Kaggle
- [ ] Create Kaggle deployment checklist

### Deliverables
- [ ] scripts/hybrid_search_cli.py
- [ ] configs/hybrid_config.yaml
- [ ] HYBRID_SEARCH_GUIDE.md
- [ ] Unit tests (tests/test_hybrid_search.py)
- [ ] Updated documentation
- [ ] Performance report

### Success Criteria
- [ ] All tests passing (unit + integration)
- [ ] Code coverage >80%
- [ ] All documentation complete
- [ ] Kaggle deployment successful
- [ ] Performance targets met

---

## Phase 3 Completion Checklist

### Code Completeness
- [x] CrossEncoder implemented (Week 1)
- [ ] HybridSearchEngine implemented (Week 2)
- [ ] CLI tool implemented (Week 3)
- [ ] Configuration system implemented (Week 3)
- [ ] All tests passing (Week 3)

### Documentation Completeness
- [x] BLIP-2 documentation (Week 1)
- [ ] Hybrid search documentation (Week 2)
- [ ] API reference complete (Week 3)
- [ ] User guides complete (Week 3)
- [ ] Kaggle guide updated (Week 3)

### Performance Targets
- [x] BLIP-2 single pair: <500ms (Week 1)
- [ ] Hybrid search total: <2000ms (Week 2)
- [ ] Recall@10: >65% (Week 2)
- [ ] Memory usage: <20GB (Week 2-3)

### Testing Coverage
- [x] BLIP-2 tests: 5/5 passing (Week 1)
- [ ] Hybrid search tests: TBD (Week 2)
- [ ] Integration tests: TBD (Week 3)
- [ ] Kaggle tests: TBD (Week 3)

### Deliverables
- [x] Week 1: BLIP-2 integration complete
- [ ] Week 2: Hybrid search working
- [ ] Week 3: Production-ready system

---

## File Structure After Phase 3

```
hybrid_multimodal_retrieval/
├── src/
│   └── retrieval/
│       ├── bi_encoder.py          [✓ Existing]
│       ├── cross_encoder.py       [✓ Week 1]
│       ├── hybrid_search.py       [□ Week 2]
│       ├── faiss_index.py         [✓ Existing]
│       └── search_engine.py       [✓ Existing]
│
├── scripts/
│   ├── test_blip2.py             [✓ Week 1]
│   ├── test_hybrid_search.py     [□ Week 2]
│   ├── hybrid_search_cli.py      [□ Week 3]
│   └── ...
│
├── notebooks/
│   ├── 06_blip2_exploration.ipynb          [✓ Week 1]
│   ├── 07_hybrid_search_demo.ipynb         [□ Week 2]
│   └── ...
│
├── configs/
│   ├── hybrid_config.yaml        [□ Week 3]
│   └── ...
│
├── tests/
│   ├── test_hybrid_search.py     [□ Week 3]
│   └── ...
│
├── PHASE3_PLAN.md               [✓ This file]
├── HYBRID_SEARCH_GUIDE.md       [□ Week 3]
├── KAGGLE_SETUP.md              [✓ Updated Week 1]
└── ...
```

---

## Dependencies

### Python Packages
- transformers>=4.30.0 [✓ Installed]
- accelerate>=0.20.0 [✓ Installed]
- torch>=2.0.0 [✓ Installed]
- open-clip-torch [✓ Installed]
- faiss-cpu or faiss-gpu [✓ Installed]
- pillow [✓ Installed]
- numpy [✓ Installed]
- pyyaml [✓ Installed]
- tqdm [✓ Installed]

### Hardware Requirements
- GPU: 16GB VRAM minimum (P100, T4, V100)
- RAM: 32GB recommended
- Disk: 50GB for models and data

### Model Downloads
- CLIP ViT-B/32 (~350MB) [✓ Downloaded]
- BLIP-2 flan-t5-xl (~15GB) [✓ Downloaded]

---

## Risk Mitigation

### Technical Risks
1. **OOM Errors:** Use smaller batch sizes, FP16, gradient checkpointing
2. **Slow Re-ranking:** Optimize batch processing, use GPU pooling
3. **Poor Accuracy:** Fine-tune models, adjust k1/k2, try different fusion methods

### Schedule Risks
1. **Week 2 Overrun:** Simplify implementation, skip optional features
2. **Testing Delays:** Automate testing, parallel testing on multiple machines
3. **Documentation Delays:** Write docs alongside code, use templates

---

## Notes

- All times are estimates; adjust based on progress
- Optional features marked clearly in each week
- Focus on core functionality first, polish later
- Test incrementally, don't wait until end of week
- Ask for help if blocked for >1 day
- Commit and push code daily

---

**Last Updated:** November 3, 2025  
**Next Review:** November 10, 2025 (End of Week 2)  
**Status:** Week 1 COMPLETE, Week 2 READY TO START
