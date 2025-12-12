# Hybrid Multimodal Retrieval System - Phase 5

🔍 Entity-based image retrieval with vision grounding and knowledge graphs

## 🎯 What Does This Do?

Advanced multimodal retrieval combining:
- 📝 **CLIP** → Fast semantic text-to-image matching
- 🧠 **Knowledge Graphs** → Entity relationship understanding  
- 👁️ **Vision Grounding** → Object detection & attribute verification (Phase 5)
- 🎯 **BLIP-2** → Accurate cross-modal re-ranking

**Dataset:** Flickr30K (31,000+ images with 158,000+ captions)  
**Phase:** 5 - Entity-based retrieval with vision grounding  
**Status:** Baseline ready, Phase 5 modules scaffolded

---

## 🚀 Quick Start

### Step 1: Setup

```bash
# Install Python packages
pip install -r requirements.txt

# Install the project
pip install -e .
```

### Step 2: Get the Dataset

**Option A - Kaggle:**
1. Download from [Kaggle Flickr30K](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)
2. Extract images to `data/images/`
3. Put `results.csv` in `data/`

**Option B - Use our script:**
```bash
python scripts\download_flickr30k.py
```

### Step 3: Build Artifacts

```bash
# Generate CLIP embeddings
python scripts\generate_image_embeddings.py
python scripts\generate_text_embeddings.py

# Build FAISS indices
python scripts\build_faiss_indices.py

# Build entity vocabulary and graph (Phase 4)
python scripts\build_entity_vocabulary.py
python scripts\build_entity_graph.py
```

### Step 4: Test the Pipeline

```bash
# Smoke test (validates pipeline is runnable)
python scripts\smoke_run.py

# Run imports test
pytest tests\test_imports.py -v

# Evaluate accuracy
python scripts\evaluate_accuracy.py
```

---

## 📖 Project Structure

```
hybrid_multimodal_retrieval/
├── configs/                    # YAML configuration files
│   ├── phase5.yaml            # Phase 5: Entity-based retrieval config
│   ├── entity_graph.yaml      # Phase 4: KG config
│   ├── hybrid_config.yaml     # Hybrid search config
│   └── clip_config.yaml       # CLIP model config
├── data/                       # Dataset and artifacts
│   ├── images/                # Flickr30K images
│   ├── embeddings/            # CLIP embeddings (.npy, .json)
│   ├── entities/              # Entity vocab, embeddings, metadata
│   ├── graph/                 # Entity graph (entity_graph.pt)
│   └── indices/               # FAISS indices
├── src/                        # Core library
│   ├── flickr30k/             # Dataset loading
│   ├── retrieval/             # CLIP, BLIP-2, hybrid search, Phase 5 modules
│   ├── graph/                 # Entity extraction, context, KG search
│   └── vision/                # Phase 5: Object detection, attribute verification
├── scripts/                    # Rebuild and evaluation scripts
│   ├── smoke_run.py           # Smoke test entrypoint
│   ├── generate_*_embeddings.py
│   ├── build_faiss_indices.py
│   ├── build_entity_vocabulary.py
│   ├── build_entity_graph.py
│   └── evaluate_accuracy.py
└── tests/                      # Pytest tests
    └── test_imports.py        # Import validation tests
```

---

## 🏗️ Phase 5 Features (Scaffolded)

Phase 5 adds **entity-based retrieval with vision grounding**:

- **Vision Module** (`src/vision/`):
  - Object detection (OWL-ViT, placeholder)
  - Attribute verification (BLIP-2 VQA, placeholder)
  - Detection storage/caching
  
- **Query Processing** (`src/retrieval/`):
  - Slot filling: Extract entities + attributes from queries
  - Binding scorer: Match query slots to image detections
  - Score gating: Adaptive fusion of CLIP, KG, and entity scores

**Status:** Skeleton modules created (importable, not yet implemented).

---

## 🧪 Testing

```bash
# Import tests (validates all modules load)
pytest tests\test_imports.py -v

# Smoke test (runs end-to-end retrieval)
python scripts\smoke_run.py --query "a dog running on grass"
```

**Expected behavior:**
- ✓ Imports succeed
- ✓ Smoke test runs query and returns top-5 results
- ⚠️ Graceful fallbacks if KG or BLIP-2 artifacts missing

---

## 🔧 Rebuild from Scratch

If you need to regenerate all artifacts:

```bash
# 1. Download dataset
python scripts\download_flickr30k.py

# 2. Generate embeddings
python scripts\generate_image_embeddings.py
python scripts\generate_text_embeddings.py

# 3. Build FAISS indices
python scripts\build_faiss_indices.py

# 4. Build entity vocab and graph (Phase 4)
python scripts\build_entity_vocabulary.py
python scripts\build_entity_graph.py

# 5. Test
python scripts\smoke_run.py
```

---

## 📋 TODO (Phase 5 Implementation)

- [ ] Implement OWL-ViT detector (`src/vision/detector_owlvit.py`)
- [ ] Implement BLIP-2 attribute verifier (`src/retrieval/attribute_verifier.py`)
- [ ] Implement query slot filler (`src/retrieval/query_slots.py`)
- [ ] Integrate entity-binding into `HybridSearchEngine`
- [ ] Add detection caching (`src/vision/storage.py`)
- [ ] Benchmark Phase 5 vs. Phase 4

---

## 📄 License

MIT

---

## 👤 Author

Phase 5 Baseline - December 2025
