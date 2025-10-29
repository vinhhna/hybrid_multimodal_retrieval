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

### 3. `test_search_engine.py` - Test Everything 🧪

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
