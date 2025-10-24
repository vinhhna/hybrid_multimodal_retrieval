# Quick Start Guide

Get started with the Hybrid Multimodal Retrieval System in minutes! This guide will walk you through setup, usage, and common tasks.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Usage](#quick-usage)
4. [Detailed Examples](#detailed-examples)
5. [Troubleshooting](#troubleshooting)
6. [Performance Tips](#performance-tips)

---

## ✅ Prerequisites

### System Requirements
- **Python**: 3.9 or higher (tested with 3.13.7)
- **GPU**: NVIDIA GPU with 4GB+ VRAM (optional, but recommended)
  - With GPU: ~11ms search latency
  - CPU only: Still fast, but slower embedding generation
- **RAM**: 8GB+ recommended
- **Disk Space**: ~5GB for dataset + ~500MB for indices

### Software
- Git (for cloning)
- pip (Python package manager)
- CUDA 11.7+ (if using GPU)

---

## 🚀 Installation

### Step 1: Clone the Repository

```powershell
# Navigate to your project directory
cd "d:\Giáo trình 20251\IT3930E - Project III"

# Clone or navigate to the repo
cd hybrid_multimodal_retrieval
```

### Step 2: Create Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate it (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate it (Windows CMD)
venv\Scripts\activate.bat

# Activate it (Linux/Mac)
source venv/bin/activate
```

### Step 3: Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA (for GPU support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install all project dependencies
pip install -r requirements.txt

# Install project in development mode
pip install -e .
```

> **Note**: If you don't have a GPU, PyTorch will automatically use CPU. Everything will still work, just slower.

### Step 4: Verify Installation

```powershell
# Test PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Test CLIP
python -c "import open_clip; print('✓ CLIP installed')"

# Test FAISS
python -c "import faiss; print('✓ FAISS installed')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA available: True  (or False if no GPU)
✓ CLIP installed
✓ FAISS installed
```

---

## ⚡ Quick Usage

### Option 1: Run the Demo Notebook (Recommended for First-Time Users)

```powershell
# Launch Jupyter
jupyter notebook notebooks/05_search_demo.ipynb
```

Then run all cells to see:
- Text-to-image search examples
- Image-to-text search examples
- Image-to-image similarity search
- Batch processing
- Performance benchmarks

### Option 2: Python Script (Fast Start)

Create a file `quick_test.py`:

```python
from retrieval import BiEncoder, FAISSIndex, MultimodalSearchEngine
from flickr30k import Flickr30KDataset

# Load components (this takes ~10 seconds first time)
print("Loading components...")
encoder = BiEncoder(model_name='ViT-B-32', pretrained='openai')
image_index = FAISSIndex.load('data/indices/image_index.faiss')
text_index = FAISSIndex.load('data/indices/text_index.faiss')
dataset = Flickr30KDataset('data/images', 'data/results.csv')

# Create search engine
engine = MultimodalSearchEngine(encoder, image_index, text_index, dataset)
print("✓ Ready!")

# Try a search
results = engine.text_to_image_search("A dog playing in the park", k=5)

print(f"\nTop 5 images:")
for i, (img_name, score) in enumerate(results, 1):
    print(f"  {i}. {img_name} (score: {score:.4f})")

# Check performance
stats = engine.get_performance_stats()
print(f"\nSearch took {stats['total_time_ms']:.2f}ms")
```

Run it:
```powershell
python quick_test.py
```

---

## 📖 Detailed Examples

### Example 1: Text-to-Image Search

Find images matching a text description:

```python
from retrieval import MultimodalSearchEngine, BiEncoder, FAISSIndex

# Setup (do once)
encoder = BiEncoder(model_name='ViT-B-32', pretrained='openai')
image_index = FAISSIndex.load('data/indices/image_index.faiss')
text_index = FAISSIndex.load('data/indices/text_index.faiss')
engine = MultimodalSearchEngine(encoder, image_index, text_index)

# Search!
results = engine.text_to_image_search(
    query_text="Children playing on the beach",
    k=10  # Return top 10 results
)

# Display results
for i, (image_name, score) in enumerate(results, 1):
    print(f"{i}. {image_name} (similarity: {score:.4f})")
    # Image is at: data/images/{image_name}
```

### Example 2: Image-to-Text Search

Find captions describing an image:

```python
from pathlib import Path

# Path to your query image
query_image = Path('data/images/1000092795.jpg')

# Search for captions
results = engine.image_to_text_search(
    query_image=query_image,
    k=5
)

print(f"Top 5 captions for {query_image.name}:")
for i, (caption, score) in enumerate(results, 1):
    print(f"{i}. [{score:.4f}] {caption}")
```

You can also use a PIL Image directly:

```python
from PIL import Image

img = Image.open('data/images/1000092795.jpg')
results = engine.image_to_text_search(query_image=img, k=5)
```

### Example 3: Image-to-Image Search

Find visually similar images:

```python
query_image = 'data/images/1000092795.jpg'

results = engine.image_to_image_search(
    query_image=query_image,
    k=10  # Top 10 similar images
)

print("Most similar images:")
for i, (img_name, score) in enumerate(results, 1):
    if i == 1:
        print(f"  {i}. {img_name} (QUERY - self match)")
    else:
        print(f"  {i}. {img_name} (similarity: {score:.4f})")
```

> **Note**: The first result is usually the query image itself (score ~1.0). Skip it if you want only different images.

### Example 4: Batch Search

Process multiple queries efficiently:

```python
# Multiple text queries
queries = [
    "A dog running",
    "People eating at a restaurant",
    "Mountain landscape",
    "City street at night"
]

# Batch search (faster than individual queries)
results_list = engine.batch_search(
    queries=queries,
    search_type='text_to_image',
    k=3  # Top 3 for each query
)

# Display results
for query, results in zip(queries, results_list):
    print(f"\n'{query}':")
    for i, (img_name, score) in enumerate(results, 1):
        print(f"  {i}. {img_name} ({score:.3f})")
```

### Example 5: With Visualization

```python
import matplotlib.pyplot as plt
from flickr30k import Flickr30KDataset

# Load dataset for image access
dataset = Flickr30KDataset('data/images', 'data/results.csv')

# Search
results = engine.text_to_image_search("Sunset at the beach", k=5)

# Visualize
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, (img_name, score) in enumerate(results):
    img = dataset.get_image(img_name)
    axes[i].imshow(img)
    axes[i].axis('off')
    axes[i].set_title(f"#{i+1}: {score:.3f}")

plt.suptitle("Query: 'Sunset at the beach'", fontsize=14)
plt.tight_layout()
plt.show()
```

---

## 🔧 Troubleshooting

### Issue 1: CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size when encoding:
   ```python
   # In BiEncoder, adjust batch_size
   embeddings = encoder.encode_images(images, batch_size=32)  # Try 16 or 8
   ```

2. Use CPU instead:
   ```python
   encoder = BiEncoder(model_name='ViT-B-32', pretrained='openai', device='cpu')
   ```

### Issue 2: Index Files Not Found

**Symptom**: `RuntimeError: Error: 'f' failed: could not open data\indices\image_index.faiss`

**Solutions**:
1. Check if indices exist:
   ```powershell
   ls data\indices\
   ```

2. If missing, build them:
   ```powershell
   python scripts/build_faiss_indices.py
   ```

3. In notebooks, use relative paths:
   ```python
   # If running from notebooks/
   image_index = FAISSIndex.load('../data/indices/image_index.faiss')
   ```

### Issue 3: Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'retrieval'`

**Solutions**:
1. Make sure virtual environment is activated:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. Install in development mode:
   ```powershell
   pip install -e .
   ```

3. Add src to Python path (temporary):
   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path.cwd().parent / 'src'))
   ```

### Issue 4: Slow Performance

**Symptom**: Search takes > 100ms

**Possible Causes**:
1. **First query**: Model loading takes ~443ms. Subsequent queries are fast (~11ms)
2. **CPU mode**: Without GPU, encoding is slower
3. **Large k**: Larger k values take slightly longer

**Solutions**:
- Warm up the model first:
  ```python
  _ = engine.text_to_image_search("warmup", k=1)  # Discard result
  ```
- Use GPU if available
- Use batch search for multiple queries

### Issue 5: Notebook Path Issues

**Symptom**: File not found errors when running notebooks

**Solution**: Notebooks run from `notebooks/` directory, so use `../` for parent:
```python
# Correct paths from notebooks/
encoder = BiEncoder(...)
image_index = FAISSIndex.load('../data/indices/image_index.faiss')
dataset = Flickr30KDataset('../data/images', '../data/results.csv')
```

---

## ⚡ Performance Tips

### 1. Use GPU Acceleration
```python
# Check if GPU is available
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# BiEncoder automatically uses GPU if available
encoder = BiEncoder(model_name='ViT-B-32', pretrained='openai')
print(f"Using device: {encoder.device}")  # Should show 'cuda'
```

### 2. Batch Processing
```python
# ❌ Slow: Individual queries
for query in queries:
    result = engine.text_to_image_search(query, k=10)

# ✅ Fast: Batch query
results = engine.batch_search(queries, 'text_to_image', k=10)
```

### 3. Reuse Components
```python
# ❌ Slow: Load every time
def search(query):
    encoder = BiEncoder(...)  # Loads model every time!
    results = engine.text_to_image_search(query)
    return results

# ✅ Fast: Load once, reuse
encoder = BiEncoder(...)  # Load once
engine = MultimodalSearchEngine(...)

def search(query):
    return engine.text_to_image_search(query)
```

### 4. Choose Appropriate k
```python
# Smaller k = faster
results = engine.text_to_image_search(query, k=10)   # Fast
results = engine.text_to_image_search(query, k=100)  # Still fast, but slightly slower
```

### 5. Disable Metadata When Not Needed
```python
# Slightly faster without metadata
results = engine.text_to_image_search(query, k=10, return_metadata=False)
```

---

## 📊 Expected Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Model loading | ~5-10s | One-time cost |
| First query | ~443ms | Includes GPU warmup |
| Subsequent queries (k=10) | ~11ms | Typical performance |
| Batch (10 queries, k=10) | ~110ms | ~11ms per query |
| Index loading | ~1-2s | One-time cost |

**Hardware Tested**: RTX 3050 Laptop GPU, Python 3.13.7, Windows 11

---

## 🎯 Next Steps

After completing this quick start:

1. **Explore the Demo Notebook**: `notebooks/05_search_demo.ipynb` has comprehensive examples
2. **Read the API Reference**: See `API_REFERENCE.md` for detailed API documentation
3. **Check Examples**: See `notebooks/` for more use cases
4. **Build Your App**: Integrate the search engine into your application

---

## 📚 Additional Resources

- **README.md** - Project overview
- **IMPLEMENTATION_PLAN.md** - Detailed project timeline
- **API_REFERENCE.md** - Complete API documentation
- **CHANGELOG.md** - Version history
- **data/README.md** - Dataset documentation

---

## 🤝 Getting Help

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section above
2. Review the demo notebooks in `notebooks/`
3. Check if FAISS indices exist in `data/indices/`
4. Verify virtual environment is activated
5. Ensure all dependencies are installed: `pip install -r requirements.txt`

---

**Last Updated**: October 24, 2025  
**Version**: 1.0  
**Status**: Phase 2 Complete - Fully Functional Search Engine
