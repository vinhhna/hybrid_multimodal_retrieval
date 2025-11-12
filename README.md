# Hybrid Multimodal Retrieval System

🔍 Search images using text. Find captions for images. Discover similar images. All blazingly fast!

## 🎯 What Does This Do?

Think of it like Google Images, but smarter! This project lets you:

- 📝 **Type a description** → Get matching images ("dog playing in park")
- 🖼️ **Upload an image** → Get similar images  
- 🔍 **Upload an image** → Get descriptive captions

**Dataset:** 31,000+ images from Flickr with 158,000+ captions  
**Speed:** Lightning fast (~11ms per search!)  
**Powered by:** AI models (CLIP) + super-fast search (FAISS)

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

**Option A - Kaggle (Easiest):**
1. Download from [Kaggle Flickr30K](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)
2. Extract images to `data/images/`
3. Put `results.csv` in `data/`

**Option B - Use our script:**
```bash
python scripts/download_flickr30k.py
```

### Step 3: Try It Out!

```python
from retrieval import BiEncoder, FAISSIndex, MultimodalSearchEngine
from flickr30k import Flickr30KDataset

# Load everything
encoder = BiEncoder()
image_index = FAISSIndex.load('data/indices/image_index.faiss')
text_index = FAISSIndex.load('data/indices/text_index.faiss')
dataset = Flickr30KDataset('data/images', 'data/results.csv')

# Create search engine
engine = MultimodalSearchEngine(encoder, image_index, text_index, dataset)

# Search!
results = engine.text_to_image_search("A dog playing in the park", k=10)
print(f"Found {len(results)} images!")
```

**That's it!** 🎉

---

## 📖 What's Inside?

```
hybrid_multimodal_retrieval/
├── configs/                 # YAML configuration files
│   ├── graph_config.yaml   # Phase 4 graph retrieval settings
│   ├── clip_config.yaml    # CLIP model configuration
│   ├── faiss_config.yaml   # FAISS index settings
│   └── blip2_config.yaml   # BLIP-2 model configuration
├── data/                    # Dataset and generated files
│   ├── images/             # Flickr30K image files
│   ├── embeddings/         # Pre-computed CLIP embeddings
│   └── indices/            # FAISS search indices
├── src/                     # Core source code
│   ├── encoders/           # CLIP-space utilities (Phase 4)
│   ├── flickr30k/          # Dataset handling
│   ├── graph/              # Graph schema (Phase 4)
│   └── retrieval/          # Search engines and indexing
├── notebooks/               # Interactive Jupyter demos
├── scripts/                 # Utility scripts
└── tests/                   # Test files
```

**Start here:** Check out the notebooks in `notebooks/` for interactive examples!

---

## 💡 Simple Examples

### Example 1: Find Images by Description

```python
# "Show me dogs!"
results = engine.text_to_image_search("dogs playing", k=5)
for img_name, score in results:
    print(f"✓ {img_name}")
```

### Example 2: Describe an Image

```python
# "What's in this image?"
captions = engine.image_to_text_search("my_photo.jpg", k=3)
for caption, score in captions:
    print(f"📝 {caption}")
```

### Example 3: Find Similar Images

```python
# "Find images like this one"
similar = engine.image_to_image_search("vacation.jpg", k=10)
print(f"Found {len(similar)} similar images!")
```

### Example 4: Hybrid Search (Smarter!)

```python
from src.retrieval.hybrid_search import HybridSearchEngine

# Create hybrid engine (CLIP + BLIP-2)
hybrid_engine = HybridSearchEngine(
    bi_encoder=bi_encoder,
    cross_encoder=cross_encoder,
    image_index=image_index,
    dataset=dataset
)

# Two-stage search: Fast CLIP → Accurate BLIP-2 re-ranking
results = hybrid_engine.text_to_image_hybrid_search(
    query="a dog running on the beach",
    k1=100,  # Stage 1: Get 100 candidates quickly
    k2=10    # Stage 2: Re-rank to top 10
)

for img_id, score in results:
    print(f"✓ {img_id} - Relevance: {score:.4f}")
```

### Example 5: Batch Search (Super Fast!)

```python
# Search multiple queries at once (2-6x faster!)
queries = [
    "a dog playing in the park",
    "sunset over the ocean",
    "a group of people at a party"
]

batch_results = hybrid_engine.batch_text_to_image_search(
    queries=queries,
    k1=100,
    k2=5
)

for query, results in batch_results.items():
    print(f"\nQuery: {query}")
    for img_id, score in results[:3]:
        print(f"  • {img_id}")
```

### Example 6: Phase 4 Graph Utilities

```python
from src.graph.schema import NodeType, EdgeType, ImageNodeMeta, CaptionNodeMeta
from src.encoders.clip_space import l2_normalize, ensure_clip_aligned

# Define graph nodes
img_node = ImageNodeMeta(
    image_id="img_001",
    path="data/images/photo.jpg",
    size=(640, 480)
)

cap_node = CaptionNodeMeta(
    caption_id="cap_001",
    image_id="img_001",
    text="A beautiful sunset over the ocean"
)

# CLIP-space utilities
import numpy as np
embeddings = np.random.rand(10, 512).astype(np.float32)
normalized = l2_normalize(embeddings)  # L2-normalize with NaN/Inf guards
ensure_clip_aligned(normalized, dim=512, check_unit_norm=True)
```

---

## 🎓 Project Progress

### ✅ What's Done
- **Phase 1**: Project setup ✅
- **Phase 2**: Fast search system working! ✅  
  - Can search 31,000 images in 11 milliseconds!
- **Phase 3**: Hybrid Search Pipeline! ✅  
  - Two-stage search: CLIP + BLIP-2 re-ranking
  - Batch processing (2-6x faster)
  - Accuracy: ~65-70% Recall@10
  - Speed: <2000ms end-to-end
- **Phase 4 (Day 1-2)**: Graph-based retrieval foundation! ✅
  - Schema definitions (NodeType, EdgeType, metadata)
  - CLIP-space utilities (L2-norm, shape validation, NaN/Inf guards)
  - Configuration system (YAML-based, merged configs)
  - Phase 4 validation script
- **Phase 4 (Day 3-4)**: PyG containers and serialization! ✅
  - PyTorch Geometric HeteroData containers
  - Chunked k-NN semantic edge builder with degree caps
  - Co-occurrence edge builder (paired & cooccur relations)
  - Atomic save/load with comprehensive validation
  - Smoke test script (100-image graph slice)
  - Test suite: 7 tests, all passing
  
### 🚧 What's Next
- **Phase 4 (Day 5-6)**: Graph retrieval with multi-hop traversal
- **Phase 4 (Day 7-10)**: LightRAG integration and evaluation
- **Phase 5**: Final polish and deployment

---

## ⚡ Performance

| What | How Fast | Notes |
|------|----------|-------|
| CLIP Search (Stage 1) | ~80ms | Fast bi-encoder retrieval |
| Hybrid Search | ~400ms | CLIP + BLIP-2 re-ranking |
| Batch Search (10 queries) | ~2000ms | 2-6x faster than sequential |
| Images | 31,783 | Entire Flickr30K dataset |
| Captions | 158,914 | About 5 per image |

**Accuracy:**
- CLIP-only: ~55-60% Recall@10
- Hybrid: ~65-70% Recall@10 (10-15% improvement!)

**Runs on:** GPU recommended (CUDA) for best performance

---

## 🆘 Need Help?

**New to this?**
- Start with `notebooks/05_search_demo.ipynb` - it's interactive and easy!
- Check out `KAGGLE_SETUP.md` if running on Kaggle
- Read `scripts/README.md` for all available utility scripts

**Something not working?**
- Make sure you downloaded the dataset (Step 2 above)
- Check that Python 3.9+ is installed
- Try the notebooks - they have all the examples

**Phase 4 Development?**
- See `PHASE_4_PLAN.md` for implementation details
- Graph schema: `src/graph/schema.py`
- Graph building: `src/graph/build.py`, `src/graph/store.py`
- CLIP utilities: `src/encoders/clip_space.py`
- Configuration: `configs/graph_config.yaml`
- Day 3-4 docs: `docs/PHASE4_DAY3_4_*.md`

**Still stuck?**
- Open an issue on GitHub
- Check the documentation files

---

## 📚 More Documentation

- **[scripts/README.md](scripts/README.md)** - Complete guide to all utility scripts
- **[KAGGLE_SETUP.md](KAGGLE_SETUP.md)** - Running on Kaggle (cloud, free GPU!)
- **[PHASE_4_PLAN.md](PHASE_4_PLAN.md)** - Graph-based retrieval implementation plan
- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** - Overall project roadmap
- **[PROMPT_TEMPLATE.md](PROMPT_TEMPLATE.md)** - Development guidelines

---

## 🎓 About This Project

**Course:** IT3930E - Project III  
**School:** Hanoi University of Science and Technology  
**Goal:** Build a smart image search system using AI

---

## 📄 License

Educational project. Please respect the Flickr30K dataset license.

---

**Questions?** Check the documentation or open an issue! 🙂
