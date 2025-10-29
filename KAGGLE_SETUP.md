# Running on Kaggle - Quick Setup

Simple guide to run this project on Kaggle.

**⚠️ GPU Required**: Enable GPU (P100 or T4) in Kaggle notebook settings before running.

## 🚀 Setup (Run Once)

Copy and run this in your Kaggle notebook:

```python
# Remove old version (if exists) and clone fresh
!rm -rf hybrid_multimodal_retrieval
!git clone https://github.com/vinhhna/hybrid_multimodal_retrieval.git

# Navigate to project directory
%cd hybrid_multimodal_retrieval

# Install dependencies
!pip install -q transformers accelerate open-clip-torch pyyaml tqdm pillow faiss-cpu

# Install project
!pip install -e .

# Verify installation
import torch
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
```

---

## 🧪 Running Scripts

### Test BLIP-2 Cross-Encoder (Phase 3)

```python
# Option 1: Auto-detect data paths
!python scripts/test_blip2.py

# Option 2: Specify data directory explicitly
!python scripts/test_blip2.py --data-dir /kaggle/input/flickr30k/data
```

**What it does:** Tests BLIP-2 model loading, single pair scoring, batch scoring, memory handling. Prints comprehensive test report.

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
