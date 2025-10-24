# API Reference

Complete API documentation for the Hybrid Multimodal Retrieval System.

---

## Table of Contents

1. [retrieval Module](#retrieval-module)
   - [BiEncoder](#biencoder)
   - [FAISSIndex](#faissindex)
   - [MultimodalSearchEngine](#multimodalsearchengine)
   - [SearchResult](#searchresult)
2. [flickr30k Module](#flickr30k-module)
   - [Flickr30KDataset](#flickr30kdataset)
   - [Utility Functions](#utility-functions)
3. [Configuration](#configuration)

---

## retrieval Module

The `retrieval` module provides classes for multimodal search and retrieval.

```python
from retrieval import BiEncoder, FAISSIndex, MultimodalSearchEngine, SearchResult
```

---

### BiEncoder

CLIP bi-encoder wrapper for generating image and text embeddings.

#### Class Definition

```python
class BiEncoder:
    """
    Wrapper for CLIP model to generate embeddings.
    
    Attributes:
        model: CLIP model instance
        preprocess: Image preprocessing function
        tokenizer: Text tokenizer
        device: torch.device ('cuda' or 'cpu')
        model_name: Name of the CLIP model
        pretrained: Pretrained weights identifier
    """
```

#### Constructor

```python
BiEncoder(model_name='ViT-B-32', pretrained='openai', device=None)
```

**Parameters:**
- `model_name` (str, optional): CLIP model architecture. Default: `'ViT-B-32'`
- `pretrained` (str, optional): Pretrained weights. Default: `'openai'`
- `device` (str or torch.device, optional): Device to use. Default: Auto-detect (CUDA if available)

**Returns:**
- `BiEncoder` instance

**Example:**
```python
# Use GPU if available
encoder = BiEncoder(model_name='ViT-B-32', pretrained='openai')

# Force CPU
encoder = BiEncoder(model_name='ViT-B-32', pretrained='openai', device='cpu')
```

---

#### encode_images()

Generate embeddings for a list of images.

```python
encode_images(images, batch_size=32, show_progress=True, normalize=True)
```

**Parameters:**
- `images` (List[PIL.Image or str or Path]): List of PIL Images or image paths
- `batch_size` (int, optional): Batch size for processing. Default: `32`
- `show_progress` (bool, optional): Show progress bar. Default: `True`
- `normalize` (bool, optional): L2-normalize embeddings. Default: `True`

**Returns:**
- `np.ndarray`: Array of shape `(n_images, 512)` with image embeddings

**Example:**
```python
from PIL import Image

# From PIL Images
images = [Image.open(f'image{i}.jpg') for i in range(10)]
embeddings = encoder.encode_images(images)

# From paths
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
embeddings = encoder.encode_images(image_paths, batch_size=16)

print(embeddings.shape)  # (3, 512)
```

---

#### encode_texts()

Generate embeddings for a list of text strings.

```python
encode_texts(texts, batch_size=32, show_progress=True, normalize=True)
```

**Parameters:**
- `texts` (List[str]): List of text strings
- `batch_size` (int, optional): Batch size for processing. Default: `32`
- `show_progress` (bool, optional): Show progress bar. Default: `True`
- `normalize` (bool, optional): L2-normalize embeddings. Default: `True`

**Returns:**
- `np.ndarray`: Array of shape `(n_texts, 512)` with text embeddings

**Example:**
```python
texts = [
    "A dog playing in the park",
    "Children at the beach",
    "Mountain landscape"
]
embeddings = encoder.encode_texts(texts)

print(embeddings.shape)  # (3, 512)
```

---

#### save_embeddings() / load_embeddings()

```python
save_embeddings(embeddings, save_path, metadata=None)
load_embeddings(load_path)
```

**Parameters (save):**
- `embeddings` (np.ndarray): Embeddings array to save
- `save_path` (str or Path): Path to save (`.npy` file)
- `metadata` (dict, optional): Additional metadata to save as JSON

**Returns (load):**
- `tuple`: `(embeddings, metadata)` - Embeddings array and metadata dict

**Example:**
```python
# Save
metadata = {'model': 'ViT-B-32', 'num_images': len(images)}
encoder.save_embeddings(embeddings, 'image_embeds.npy', metadata)

# Load
embeddings, meta = encoder.load_embeddings('image_embeds.npy')
print(f"Loaded {meta['num_images']} embeddings")
```

---

### FAISSIndex

FAISS index manager for fast similarity search.

#### Class Definition

```python
class FAISSIndex:
    """
    Wrapper for FAISS index operations.
    
    Attributes:
        index: faiss.Index instance
        dimension: Embedding dimension (512)
        index_type: Type of index ('flat', 'ivf', 'hnsw')
        metric: Distance metric ('cosine', 'euclidean')
        metadata: Dictionary with index metadata
        is_trained: Whether index is trained
    """
```

#### Constructor

```python
FAISSIndex(dimension=512, index_type='flat', metric='cosine', nlist=100)
```

**Parameters:**
- `dimension` (int, optional): Embedding dimension. Default: `512`
- `index_type` (str, optional): Index type - `'flat'`, `'ivf'`, or `'hnsw'`. Default: `'flat'`
- `metric` (str, optional): Distance metric - `'cosine'` or `'euclidean'`. Default: `'cosine'`
- `nlist` (int, optional): Number of clusters for IVF. Default: `100`

**Returns:**
- `FAISSIndex` instance

**Example:**
```python
# Exact search (slower, more accurate)
index = FAISSIndex(dimension=512, index_type='flat', metric='cosine')

# Approximate search (faster, slightly less accurate)
index = FAISSIndex(dimension=512, index_type='ivf', metric='cosine', nlist=100)
```

---

#### add()

Add embeddings to the index.

```python
add(embeddings, ids=None)
```

**Parameters:**
- `embeddings` (np.ndarray): Embeddings array of shape `(n, dimension)`
- `ids` (List[str], optional): List of IDs for each embedding

**Example:**
```python
# Add embeddings
index.add(embeddings, ids=['img1.jpg', 'img2.jpg', 'img3.jpg'])
print(f"Index now has {index.index.ntotal} vectors")
```

---

#### search()

Search for k nearest neighbors.

```python
search(query_embeddings, k=10)
```

**Parameters:**
- `query_embeddings` (np.ndarray): Query embeddings of shape `(n_queries, dimension)`
- `k` (int, optional): Number of results to return. Default: `10`

**Returns:**
- `tuple`: `(scores, indices)` - Similarity scores and indices arrays
  - `scores`: shape `(n_queries, k)`
  - `indices`: shape `(n_queries, k)`

**Example:**
```python
# Single query
query_emb = encoder.encode_texts(["A dog playing"])
scores, indices = index.search(query_emb, k=5)

print("Top 5 results:")
for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
    print(f"  {i+1}. Index {idx}: score {score:.4f}")

# Batch query
query_embs = encoder.encode_texts(["dogs", "cats", "birds"])
scores, indices = index.search(query_embs, k=10)
print(scores.shape)  # (3, 10)
```

---

#### save() / load()

```python
save(index_path, metadata_path=None)
load(index_path, metadata_path=None)
```

**Parameters:**
- `index_path` (str or Path): Path to save/load FAISS index (`.faiss` file)
- `metadata_path` (str or Path, optional): Path to save/load metadata JSON

**Example:**
```python
# Save
index.save('my_index.faiss')

# Load
index = FAISSIndex()
index.load('my_index.faiss')
print(f"Loaded index with {index.index.ntotal} vectors")
```

---

#### get_stats()

Get index statistics.

```python
get_stats()
```

**Returns:**
- `dict`: Dictionary with index statistics

**Example:**
```python
stats = index.get_stats()
print(f"Index type: {stats['type']}")
print(f"Vectors: {stats['n_vectors']}")
print(f"Dimension: {stats['dimension']}")
```

---

### MultimodalSearchEngine

High-level search engine for multimodal retrieval.

#### Class Definition

```python
class MultimodalSearchEngine:
    """
    Multimodal search engine combining BiEncoder and FAISS indices.
    
    Attributes:
        encoder: BiEncoder instance
        image_index: FAISSIndex for images
        text_index: FAISSIndex for text
        dataset: Flickr30KDataset (optional)
    """
```

#### Constructor

```python
MultimodalSearchEngine(encoder, image_index, text_index, dataset=None)
```

**Parameters:**
- `encoder` (BiEncoder): Encoder for generating embeddings
- `image_index` (FAISSIndex): FAISS index containing image embeddings
- `text_index` (FAISSIndex): FAISS index containing text embeddings
- `dataset` (Flickr30KDataset, optional): Dataset for accessing captions/images

**Returns:**
- `MultimodalSearchEngine` instance

**Example:**
```python
from retrieval import BiEncoder, FAISSIndex, MultimodalSearchEngine
from flickr30k import Flickr30KDataset

# Load components
encoder = BiEncoder(model_name='ViT-B-32', pretrained='openai')
image_index = FAISSIndex.load('data/indices/image_index.faiss')
text_index = FAISSIndex.load('data/indices/text_index.faiss')
dataset = Flickr30KDataset('data/images', 'data/results.csv')

# Create engine
engine = MultimodalSearchEngine(encoder, image_index, text_index, dataset)
```

---

#### text_to_image_search()

Search images using text query.

```python
text_to_image_search(query_text, k=10, return_metadata=True)
```

**Parameters:**
- `query_text` (str or List[str]): Text query or list of queries
- `k` (int, optional): Number of results. Default: `10`
- `return_metadata` (bool, optional): Include metadata in results. Default: `True`

**Returns:**
- `SearchResult` or `List[SearchResult]`: Results (single or list depending on input)

**Example:**
```python
# Single query
result = engine.text_to_image_search("A dog playing in the park", k=5)

print(f"Found {len(result)} images:")
for img_name, score in result:
    print(f"  {img_name}: {score:.4f}")

# Multiple queries
queries = ["dogs", "cats", "birds"]
results = engine.text_to_image_search(queries, k=3)

for query, result in zip(queries, results):
    print(f"\n{query}: {result.ids[0]} ({result.scores[0]:.3f})")
```

---

#### image_to_text_search()

Search captions using image query.

```python
image_to_text_search(query_image, k=10, return_metadata=True)
```

**Parameters:**
- `query_image` (str, Path, PIL.Image, or List): Image path, PIL Image, or list
- `k` (int, optional): Number of results. Default: `10`
- `return_metadata` (bool, optional): Include metadata. Default: `True`

**Returns:**
- `SearchResult` or `List[SearchResult]`: Results with captions (if dataset provided)

**Example:**
```python
# From file path
result = engine.image_to_text_search('data/images/123.jpg', k=5)

print("Top 5 captions:")
for i, (caption, score) in enumerate(result, 1):
    print(f"  {i}. [{score:.4f}] {caption}")

# From PIL Image
from PIL import Image
img = Image.open('my_image.jpg')
result = engine.image_to_text_search(img, k=3)
```

---

#### image_to_image_search()

Find similar images.

```python
image_to_image_search(query_image, k=10, return_metadata=True)
```

**Parameters:**
- `query_image` (str, Path, PIL.Image, or List): Image path, PIL Image, or list
- `k` (int, optional): Number of results. Default: `10`
- `return_metadata` (bool, optional): Include metadata. Default: `True`

**Returns:**
- `SearchResult` or `List[SearchResult]`: Similar images

**Example:**
```python
result = engine.image_to_image_search('query_image.jpg', k=6)

print("Similar images:")
for i, (img_name, score) in enumerate(result):
    if i == 0:
        print(f"  {i+1}. {img_name} (QUERY - {score:.4f})")
    else:
        print(f"  {i+1}. {img_name} ({score:.4f})")
```

---

#### batch_search()

Batch search for multiple queries.

```python
batch_search(queries, search_type='text_to_image', k=10, return_metadata=True)
```

**Parameters:**
- `queries` (List): List of queries (text or images)
- `search_type` (str): `'text_to_image'`, `'image_to_text'`, or `'image_to_image'`
- `k` (int, optional): Number of results per query. Default: `10`
- `return_metadata` (bool, optional): Include metadata. Default: `True`

**Returns:**
- `List[SearchResult]`: List of results for each query

**Example:**
```python
queries = ["dogs playing", "cats sleeping", "birds flying"]
results = engine.batch_search(queries, 'text_to_image', k=5)

for query, result in zip(queries, results):
    print(f"{query}: {len(result)} results")
```

---

#### get_performance_stats()

Get performance metrics from last search.

```python
get_performance_stats()
```

**Returns:**
- `dict`: Performance statistics

**Example:**
```python
result = engine.text_to_image_search("dogs", k=10)
stats = engine.get_performance_stats()

print(f"Encoding: {stats['encode_time_ms']:.2f}ms")
print(f"Search: {stats['search_time_ms']:.2f}ms")
print(f"Total: {stats['total_time_ms']:.2f}ms")
print(f"QPS: {stats['qps']:.1f}")
```

---

### SearchResult

Container class for search results.

#### Class Definition

```python
class SearchResult:
    """
    Container for search results.
    
    Attributes:
        ids: List of result IDs (image names or captions)
        scores: List of similarity scores
        metadata: Optional metadata dictionary
    """
```

#### Usage

```python
result = engine.text_to_image_search("dogs", k=5)

# Access as iterable
for img_name, score in result:
    print(f"{img_name}: {score}")

# Access by index
first_id, first_score = result[0]

# Get length
print(f"Found {len(result)} results")

# String representation
print(result)  # SearchResult(n=5, top_score=0.3456)
```

---

## flickr30k Module

Dataset utilities for Flickr30K.

```python
from flickr30k import Flickr30KDataset
from flickr30k.utils import load_config
```

---

### Flickr30KDataset

Dataset loader and accessor for Flickr30K.

#### Constructor

```python
Flickr30KDataset(images_dir, captions_file)
```

**Parameters:**
- `images_dir` (str or Path): Path to images directory
- `captions_file` (str or Path): Path to `results.csv`

**Example:**
```python
dataset = Flickr30KDataset(
    images_dir='data/images',
    captions_file='data/results.csv'
)

print(f"Loaded {len(dataset)} captions")
print(f"Unique images: {dataset.num_images}")
```

---

#### get_captions()

Get all captions for an image.

```python
get_captions(image_name)
```

**Parameters:**
- `image_name` (str): Image filename

**Returns:**
- `List[str]`: List of captions (typically 5)

**Example:**
```python
captions = dataset.get_captions('1000092795.jpg')
for i, caption in enumerate(captions, 1):
    print(f"  {i}. {caption}")
```

---

#### get_image()

Load and return an image.

```python
get_image(image_name)
```

**Parameters:**
- `image_name` (str): Image filename

**Returns:**
- `PIL.Image`: Loaded image

**Example:**
```python
import matplotlib.pyplot as plt

img = dataset.get_image('1000092795.jpg')
plt.imshow(img)
plt.axis('off')
plt.show()
```

---

#### get_unique_images()

Get list of all unique image names.

```python
get_unique_images()
```

**Returns:**
- `List[str]`: List of image filenames

**Example:**
```python
images = dataset.get_unique_images()
print(f"Total images: {len(images)}")
print(f"First 5: {images[:5]}")
```

---

### Utility Functions

#### load_config()

Load YAML configuration file.

```python
from flickr30k.utils import load_config

config = load_config('configs/faiss_config.yaml')
print(config['index_type'])
```

---

## Configuration

### FAISS Configuration (`configs/faiss_config.yaml`)

```yaml
# Index configuration
index:
  type: "flat"          # flat, ivf, or hnsw
  metric: "cosine"      # cosine or euclidean
  dimension: 512        # Embedding dimension
  
  # IVF parameters (only used if type=ivf)
  ivf:
    nlist: 100          # Number of clusters
    nprobe: 10          # Number of clusters to search
  
  # HNSW parameters (only used if type=hnsw)
  hnsw:
    M: 32               # Number of connections
    efConstruction: 40  # Construction time search depth
    efSearch: 16        # Search time depth

# Paths
paths:
  image_index: "data/indices/image_index.faiss"
  text_index: "data/indices/text_index.faiss"
```

---

## Common Patterns

### Pattern 1: Complete Search Pipeline

```python
from retrieval import BiEncoder, FAISSIndex, MultimodalSearchEngine
from flickr30k import Flickr30KDataset

# Setup (do once)
encoder = BiEncoder()
image_index = FAISSIndex.load('data/indices/image_index.faiss')
text_index = FAISSIndex.load('data/indices/text_index.faiss')
dataset = Flickr30KDataset('data/images', 'data/results.csv')
engine = MultimodalSearchEngine(encoder, image_index, text_index, dataset)

# Use repeatedly
results = engine.text_to_image_search("your query", k=10)
```

### Pattern 2: Batch Processing

```python
queries = ["query1", "query2", "query3"]
results = engine.batch_search(queries, 'text_to_image', k=5)

for query, result in zip(queries, results):
    print(f"\n{query}:")
    for img_name, score in result:
        print(f"  {img_name}: {score:.3f}")
```

### Pattern 3: Visualization

```python
import matplotlib.pyplot as plt

result = engine.text_to_image_search("dogs", k=5)

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, (img_name, score) in enumerate(result):
    img = dataset.get_image(img_name)
    axes[i].imshow(img)
    axes[i].axis('off')
    axes[i].set_title(f"{score:.3f}")
plt.show()
```

---

**Version**: 1.0  
**Last Updated**: October 24, 2025  
**Phase**: 2 Complete
