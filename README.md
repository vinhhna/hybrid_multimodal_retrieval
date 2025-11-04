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
├── data/                    # Your images and search indices
├── src/                     # The smart code that makes it work
├── notebooks/               # Interactive demos you can play with
├── scripts/                 # Helper scripts
└── configs/                 # Settings and configurations
```

**Don't worry about the details!** Check out the notebooks in `notebooks/` for easy examples.

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

---

## 🎓 Project Progress

### ✅ What's Done
- **Phase 1**: Project setup ✅
- **Phase 2**: Fast search system working! ✅  
  - Can search 31,000 images in 11 milliseconds!
- **Phase 3 Week 2**: Hybrid Search Pipeline! ✅  
  - Two-stage search: CLIP + BLIP-2 re-ranking
  - Batch processing (2-6x faster)
  - Accuracy: ~65-70% Recall@10
  - Speed: <2000ms end-to-end
  
### 🚧 What's Next
- **Phase 3 Week 3**: Knowledge-enhanced retrieval
- **Phase 4**: Adding knowledge graphs (Dec 2025)
- **Phase 5**: Final polish (Jan-Feb 2026)

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

**Something not working?**
- Make sure you downloaded the dataset (Step 2 above)
- Check that Python 3.9+ is installed
- Try the notebooks - they have all the examples

**Still stuck?**
- Open an issue on GitHub
- Check the documentation files

---

## 📚 More Documentation

- **[KAGGLE_SETUP.md](KAGGLE_SETUP.md)** - Running on Kaggle (cloud, free GPU!)
- **[API_REFERENCE.md](API_REFERENCE.md)** - Detailed code documentation
- **[CHANGELOG.md](CHANGELOG.md)** - What changed in each version
- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** - Project roadmap

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
