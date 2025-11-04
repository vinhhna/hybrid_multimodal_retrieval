# Hybrid Multimodal Retrieval - Mini Version

A simple image search system using CLIP and FAISS.

## What it does

- Search images using text descriptions
- Find captions for images
- Find similar images

## Setup

```bash
pip install -r requirements.txt
```

## Usage

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
│   ├── dataset.py      # Load Flickr30K data
│   ├── encoder.py      # CLIP encoder
│   ├── index.py        # FAISS index
│   └── search.py       # Search engine
├── data/
│   ├── images/         # Image files (download separately)
│   └── results.csv     # Captions file
├── demo.py             # Example usage
└── requirements.txt    # Dependencies
```

## Dataset

Download Flickr30K dataset from Kaggle:
https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset

Place images in `data/images/` and `results.csv` in `data/`.
