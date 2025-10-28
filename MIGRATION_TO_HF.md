# Migration from salesforce-lavis to Hugging Face Transformers

**Date**: October 28, 2025  
**Reason**: User preference for Hugging Face ecosystem

---

## ✅ What Changed

### Dependencies

**Before (salesforce-lavis)**:
```bash
pip install salesforce-lavis accelerate
```

**After (Hugging Face)**:
```bash
pip install transformers accelerate
```

### Model Initialization

**Before**:
```python
from lavis.models import load_model_and_preprocess

encoder = CrossEncoder(
    model_name='blip2_opt',
    model_type='pretrain_opt2.7b'
)
```

**After**:
```python
from transformers import Blip2Processor, Blip2ForConditionalGeneration

encoder = CrossEncoder(
    model_name='Salesforce/blip2-opt-2.7b'  # Hugging Face model ID
)
```

---

## 📝 Files Updated

### 1. Core Implementation
- **`src/retrieval/cross_encoder.py`**
  - Replaced `lavis` imports with `transformers`
  - Changed model loading from LAVIS API to Hugging Face
  - Updated preprocessing to use `Blip2Processor`
  - Simplified scoring methods for HF API
  - Removed `model_type` parameter (now in model name)

### 2. Dependencies
- **`requirements.txt`**
  - `salesforce-lavis>=1.0.0` → `transformers>=4.30.0`
  - Kept `accelerate>=0.20.0` (still needed)

### 3. Documentation
- **`README.md`**
  - Updated dependency list
  
- **`IMPLEMENTATION_PLAN.md`**
  - Updated Phase 3 installation instructions
  - Updated requirements.txt example

- **`KAGGLE_SETUP.md`**
  - All cells updated to use `transformers`
  - Simplified code examples with HF API
  - Updated troubleshooting section

### 4. Testing
- **`scripts/test_blip2.py`**
  - Updated model initialization
  - Changed to HF model name

### 5. Notebooks
- **`notebooks/06_blip2_exploration.ipynb`**
  - Cell 1: Check for `transformers` instead of `lavis`
  - Cell 2: Updated model loading with HF model name

---

## 🔑 Key Differences

### API Changes

| Aspect | salesforce-lavis | Hugging Face |
|--------|------------------|--------------|
| **Import** | `from lavis.models import load_model_and_preprocess` | `from transformers import Blip2Processor, Blip2ForConditionalGeneration` |
| **Model ID** | `'blip2_opt'` + `'pretrain_opt2.7b'` | `'Salesforce/blip2-opt-2.7b'` |
| **Processor** | `vis_processors["eval"]`, `txt_processors["eval"]` | `processor(images=..., text=...)` |
| **Device** | Set in `load_model_and_preprocess()` | `.to(device)` after loading |
| **FP16** | `.half()` after loading | `torch_dtype` in `from_pretrained()` |

### Advantages of Hugging Face

✅ **Better ecosystem integration** - Works with other HF models  
✅ **More familiar API** - Standard transformers interface  
✅ **Better documentation** - Extensive HF docs and examples  
✅ **Model hub** - Easy access to all BLIP-2 variants  
✅ **Community support** - Larger user base  

### Scoring Method (Unchanged)

Both versions use the same yes/no probability scoring:
```python
prompt = f"Question: Does this image show {text.lower()}? Answer:"
# Get P(yes) and P(no), compute score = P(yes) / (P(yes) + P(no))
```

---

## 🚀 Usage After Migration

### Quick Start

```python
from src.retrieval.cross_encoder import CrossEncoder

# Initialize (downloads ~5GB on first run)
encoder = CrossEncoder(
    model_name='Salesforce/blip2-opt-2.7b',
    use_fp16=True
)

# Score single pair
score = encoder.score_pair("A dog playing", "image.jpg")

# Score batch
scores = encoder.score_pairs(queries, images, batch_size=8)
```

### Available Models

You can now easily switch BLIP-2 variants:

```python
# Smaller, faster (2.7B params)
encoder = CrossEncoder('Salesforce/blip2-opt-2.7b')

# Larger, more accurate (11B params) - requires more GPU memory
encoder = CrossEncoder('Salesforce/blip2-flan-t5-xl')
```

---

## ✅ Verification

Run tests to verify migration:

```bash
# Local testing
python scripts/test_blip2.py

# Kaggle testing
# Follow KAGGLE_SETUP.md (already updated)
```

---

## 📊 Performance Expectations

**No change expected** - Both implementations use the same BLIP-2 model weights from Salesforce.

| Metric | Expected Value |
|--------|----------------|
| Model size | ~5GB download |
| GPU memory | ~6-8GB (FP16) |
| 100 pairs | < 2 seconds (target) |
| Single pair | ~20-50ms |

---

## 🔄 Rollback (if needed)

To revert to salesforce-lavis:

```bash
# 1. Uninstall transformers
pip uninstall transformers

# 2. Install lavis
pip install salesforce-lavis

# 3. Revert files (use git)
git checkout HEAD -- src/retrieval/cross_encoder.py
git checkout HEAD -- requirements.txt
# ... (revert other files)
```

---

## 📌 Notes

- Model weights are identical (both from Salesforce BLIP-2)
- Scoring logic unchanged (yes/no probability method)
- Performance should be similar
- HF version is now the standard for this project
- All documentation updated to reflect HF usage

---

**Migration completed successfully! ✅**
