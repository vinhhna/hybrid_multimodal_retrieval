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
│   ├── entity_graph.yaml   # Phase 4 entity graph + query enrichment + search config
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
│   ├── graph/              # Entity vocabulary + entity graph (Phase 4)
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

### Example 6: Phase 4 Entity Vocabulary & Config

```python
from src.graph.config import load_entity_graph_config, print_config_summary
from src.graph import build_entity_vocabulary, EntityStats
from src.flickr30k.dataset import Flickr30KDataset

# Load Phase 4 configuration
cfg = load_entity_graph_config("configs/entity_graph.yaml")
print_config_summary(cfg)  # Shows entity_graph, query_enrichment, graph_search, fusion

# Build entity vocabulary (already implemented)
dataset = Flickr30KDataset('data/images', 'data/results.csv', auto_load=True)
entity_vocab, entity_context = build_entity_vocabulary(dataset, cfg)

print(f"Built vocabulary with {len(entity_vocab)} entities")

# Access entity stats
for entity_name, stats in list(entity_vocab.items())[:5]:
    print(f"{entity_name}: df_caption={stats.df_caption}, df_image={stats.df_image}")

# Note: Graph building, query enrichment, and graph search are under development
# See src/graph/build_entity_graph.py, graph_search.py, context.py (skeletons)
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
- **Phase 4 (Day 0)**: Entity-centric graph setup ✅
  - Created `configs/entity_graph.yaml` with 4 config sections:
    - `entity_graph`: paths, min_df, k_sem, degree_cap
    - `query_enrichment`: K_seed_raw, M_enrich, templates
    - `graph_search`: K_seed, H_max, B, decay, edge weights
    - `fusion`: w_clip, w_kg, w_blip2
  - Created Phase 4 module skeletons with detailed design docs
  - Config helpers implemented in `src/graph/config.py`
  - Verification script: all tests passing
- **Phase 4 (Day 1-2)**: Entity vocabulary & context ✅
  - Implemented entity extraction & normalization pipeline
  - Built vocabulary from full Flickr30K (31,783 images, 158,914 captions)
  - Generated `data/entities/entity_vocab.json` (entity IDs, stats)
  - Generated `data/entities/entity_context.json` (entity→images/captions)
  - Code: `src/graph/entities.py`, `scripts/build_entity_vocabulary.py`
  
### 🚧 What's Next (Phase 4 - In Progress)
- **Entity graph construction**: Build PyG HeteroData with semantic & co-occurrence edges
  - Skeleton ready: `src/graph/build_entity_graph.py`
- **Query enrichment**: CLIP-based entity seeding for query expansion
  - Skeleton ready: `src/graph/graph_search.py`
- **Graph search**: LightRAG-style beam search with decay and edge weighting
  - Skeleton ready: `src/graph/graph_search.py`
- **Context synthesis**: Explainable retrieval with graph reasoning chains
  - Skeleton ready: `src/graph/context.py`
- **Evaluation**: Compare CLIP-only vs Hybrid vs KG-augmented modes
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
- See `PHASE_4_PLAN.md` for entity-centric design details
- Configuration: `configs/entity_graph.yaml` (4 sections: entity_graph, query_enrichment, graph_search, fusion)
- Entity vocabulary: `src/graph/entities.py` (✅ implemented)
- Config helpers: `src/graph/config.py` (✅ implemented)
- Graph construction: `src/graph/build_entity_graph.py` (🚧 skeleton, under development)
- Query enrichment & search: `src/graph/graph_search.py` (🚧 skeleton, under development)
- Context synthesis: `src/graph/context.py` (🚧 skeleton, under development)
- Module exports & status: `src/graph/__init__.py`
- Setup verification: `scripts/verify_day0_setup.py`
- Day 0 completion: `PHASE_4_DAY_0_COMPLETE.md`

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
