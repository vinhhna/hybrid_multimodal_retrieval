# Getting Started Guide

## Quick Setup (5 minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

Download Flickr30K from Kaggle:
https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset

Extract the files:
- Put all images in `data/images/`
- Put `results.csv` in `data/`

### 3. Build Search Indices

This will take 10-15 minutes:

```bash
python build_indices.py
```

### 4. Run Demo

```bash
python demo.py
```

Or open `demo.ipynb` in Jupyter to try interactive examples.

## What Each File Does

- **src/dataset.py** - Loads images and captions
- **src/encoder.py** - CLIP model for encoding images and text
- **src/index.py** - FAISS for fast search
- **src/search.py** - Main search engine

- **build_indices.py** - Run once to create search indices
- **demo.py** - Example usage script
- **demo.ipynb** - Interactive notebook demo

## Troubleshooting

**No CUDA device found?**
- It's fine! The code will use CPU automatically
- Just slower (few seconds instead of milliseconds)

**Out of memory?**
- Reduce batch_size in encoder.py (line 28 and 44)
- Default is 32 for images, 64 for text

**Index files not found?**
- Make sure you ran `build_indices.py` first
- Check that files exist in `data/` folder

## Tips

- Search is fast after indices are built (~100ms)
- Try different queries to see what works
- Check `demo.ipynb` for visualization examples
- Captions are pre-computed, so image-to-text is instant
