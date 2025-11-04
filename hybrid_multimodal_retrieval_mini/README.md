# Hybrid Multimodal Retrieval - Mini Version

A simple image search system using CLIP and FAISS, with optional BLIP-2 re-ranking for better accuracy.

## What it does

- Search images using text descriptions
- Find captions for images
- Find similar images
- Hybrid search (CLIP + BLIP-2) for improved accuracy

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Basic Search (CLIP only)

```python
from src.dataset import Flickr30KDataset
from src.encoder import CLIPEncoder
from src.index import FAISSIndex
from src.search import SearchEngine

# Load components
dataset = Flickr30KDataset('data/images', 'data/results.csv')
encoder = CLIPEncoder()

# Load pre-built indices
image_index = FAISSIndex()
image_index.load('data/image_index.faiss')

text_index = FAISSIndex()
text_index.load('data/text_index.faiss')

# Create search engine
engine = SearchEngine(encoder, image_index, text_index, dataset)

# Search!
results = engine.text_to_image("a dog playing in the park", k=5)
for img_name, score in results:
    print(f"{img_name}: {score:.4f}")
```

### Hybrid Search (CLIP + BLIP-2)

For better accuracy, use two-stage hybrid search:

```python
from src.reranker import BLIP2Reranker
from src.hybrid_search import HybridSearchEngine

# Load BLIP-2 re-ranker
reranker = BLIP2Reranker()

# Create hybrid engine
hybrid_engine = HybridSearchEngine(encoder, reranker, image_index, dataset)

# Two-stage search: CLIP retrieves 100 candidates, BLIP-2 re-ranks to top 5
results = hybrid_engine.search("a dog playing in the park", k1=100, k2=5)
for img_name, score in results:
    print(f"{img_name}: {score:.4f}")
```

## Building Indices

If you need to build the indices from scratch:

```python
# Generate embeddings
image_names = dataset.get_all_images()
images = [dataset.get_image(name) for name in image_names]
image_embeddings = encoder.encode_images(images)

captions = dataset.df['caption'].tolist()
text_embeddings = encoder.encode_texts(captions)

# Build indices
image_index = FAISSIndex()
image_index.add(image_embeddings, ids=image_names)
image_index.save('data/image_index.faiss')

text_index = FAISSIndex()
text_index.add(text_embeddings)
text_index.save('data/text_index.faiss')
```

## Project Structure

```
hybrid_multimodal_retrieval_mini/
├── src/
│   ├── dataset.py       # Load Flickr30K data
│   ├── encoder.py       # CLIP encoder
│   ├── index.py         # FAISS index
│   ├── search.py        # Basic search engine
│   ├── reranker.py      # BLIP-2 re-ranker
│   └── hybrid_search.py # Hybrid search (CLIP + BLIP-2)
├── data/
│   ├── images/          # Image files (download separately)
│   └── results.csv      # Captions file
├── demo.py              # Example usage
├── demo.ipynb           # Interactive demo
└── requirements.txt     # Dependencies
```

## Dataset

Download Flickr30K dataset from Kaggle:
https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset

Place images in `data/images/` and `results.csv` in `data/`.
