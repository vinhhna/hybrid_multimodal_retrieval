# API Reference - Simple Guide

Quick guide to using the code in this project. Don't worry, it's easier than it looks!

---

## 🎯 The Main Classes You'll Use

There are just 3 main things you need to know about:

1. **BiEncoder** - Turns images and text into numbers (embeddings)
2. **FAISSIndex** - Stores and searches those numbers super fast
3. **MultimodalSearchEngine** - The easy way to search (uses both above)

**Pro tip:** Just use `MultimodalSearchEngine` and you're good to go! 🚀

---

## 🔧 MultimodalSearchEngine - The Easy One

This is all you need for most tasks!

### Setup (Do Once)

```python
from retrieval import BiEncoder, FAISSIndex, MultimodalSearchEngine
from flickr30k import Flickr30KDataset

# Load everything
encoder = BiEncoder()
image_index = FAISSIndex.load('data/indices/image_index.faiss')
text_index = FAISSIndex.load('data/indices/text_index.faiss')
dataset = Flickr30KDataset('data/images', 'data/results.csv')

# Create your search engine
engine = MultimodalSearchEngine(encoder, image_index, text_index, dataset)
```

### Search by Text

```python
# Find images matching your description
results = engine.text_to_image_search("A dog playing", k=10)

# See what you found
for img_name, score in results:
    print(f"{img_name} - Match: {score:.2f}")
```

**Parameters:**
- `query_text` - What you're looking for (string)
- `k` - How many results you want (default: 10)

### Search by Image (Find Captions)

```python
# Get captions for an image
captions = engine.image_to_text_search("my_image.jpg", k=5)

# Print them
for caption, score in captions:
    print(f"📝 {caption}")
```

### Search by Image (Find Similar)

```python
# Find images that look similar
similar = engine.image_to_image_search("vacation.jpg", k=10)

# First result is the image itself
for i, (img_name, score) in enumerate(similar):
    if i == 0:
        print(f"Original: {img_name}")
    else:
        print(f"Similar #{i}: {img_name}")
```

### Batch Search (Multiple Queries at Once)

```python
# Search for multiple things
queries = [
    "Dogs playing",
    "People swimming",
    "City at night"
]

results = engine.batch_search(queries, search_type='text_to_image', k=5)

# Look at results
for query, result in zip(queries, results):
    print(f"\n{query}:")
    for img, score in result:
        print(f"  - {img}")
```

**Pro tip:** Batch search is faster than doing searches one by one!

---

## 📊 Flickr30KDataset - Working with the Data

### Load the Dataset

```python
from flickr30k import Flickr30KDataset

dataset = Flickr30KDataset('data/images', 'data/results.csv')
print(f"Loaded {dataset.num_images} images!")
```

### Get Captions for an Image

```python
captions = dataset.get_captions('1000092795.jpg')

for i, caption in enumerate(captions, 1):
    print(f"{i}. {caption}")
```

### Load an Image

```python
img = dataset.get_image('1000092795.jpg')
# Now you can display it with matplotlib or PIL
```

### Get All Image Names

```python
all_images = dataset.get_unique_images()
print(f"Total images: {len(all_images)}")
```

---

## 🤖 BiEncoder - The AI Part (Advanced)

Only use this if you need to generate embeddings yourself.

### Create Encoder

```python
from retrieval import BiEncoder

encoder = BiEncoder()
# That's it! It automatically uses GPU if available
```

### Turn Images into Embeddings

```python
from PIL import Image

# Load some images
images = [Image.open(f) for f in ['img1.jpg', 'img2.jpg']]

# Get embeddings (numbers that represent the images)
embeddings = encoder.encode_images(images)
print(f"Shape: {embeddings.shape}")  # (2, 512)
```

### Turn Text into Embeddings

```python
texts = [
    "A dog playing",
    "People at the beach",
    "City skyline"
]

embeddings = encoder.encode_texts(texts)
print(f"Shape: {embeddings.shape}")  # (3, 512)
```

**What's an embedding?** It's just a list of 512 numbers that captures the "meaning" of an image or text. Similar things have similar numbers!

---

## 🗂️ FAISSIndex - The Fast Search (Advanced)

Only use this if you want to build your own index.

### Create a New Index

```python
from retrieval import FAISSIndex

index = FAISSIndex(dimension=512, index_type='flat')
```

### Add Embeddings

```python
# Add your embeddings
index.add(embeddings, ids=['img1.jpg', 'img2.jpg', 'img3.jpg'])
```

### Search

```python
# Create a query embedding
query_emb = encoder.encode_texts(["dogs"])

# Search!
scores, indices = index.search(query_emb, k=5)

print(f"Top 5 results:")
for score, idx in zip(scores[0], indices[0]):
    print(f"  Index {idx}: score {score:.3f}")
```

### Save and Load

```python
# Save for later
index.save('my_index.faiss')

# Load it back
index = FAISSIndex()
index.load('my_index.faiss')
```

---

## 💡 Common Patterns

### Pattern 1: Simple Text Search

```python
# This is all you need!
results = engine.text_to_image_search("cats", k=10)
```

### Pattern 2: Display Results with Images

```python
import matplotlib.pyplot as plt

results = engine.text_to_image_search("sunset", k=6)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, (img_name, score) in enumerate(results):
    img = dataset.get_image(img_name)
    axes[i//3, i%3].imshow(img)
    axes[i//3, i%3].set_title(f"Score: {score:.2f}")
    axes[i//3, i%3].axis('off')
plt.show()
```

### Pattern 3: Find Images Like Yours

```python
# Upload your image
results = engine.image_to_image_search("my_photo.jpg", k=10)

# Get the similar ones (skip first - it's your image)
similar = results[1:]  # Skip the first result
```

---

## 🔍 Understanding the Results

### What's a Score?

Each result comes with a score between 0 and 1:
- **1.0** = Perfect match
- **0.8-0.9** = Very similar
- **0.6-0.7** = Somewhat similar
- **< 0.5** = Not very similar

### SearchResult Object

```python
result = engine.text_to_image_search("dogs", k=5)

# You can use it like a list
for img_name, score in result:
    print(img_name, score)

# Or access parts directly
print(result.ids)      # ['img1.jpg', 'img2.jpg', ...]
print(result.scores)   # [0.85, 0.78, ...]
print(len(result))     # 5
```

---

## 🎨 Complete Example

Here's a complete example that does everything:

```python
from retrieval import BiEncoder, FAISSIndex, MultimodalSearchEngine
from flickr30k import Flickr30KDataset
import matplotlib.pyplot as plt

# 1. Setup (do once)
print("Loading models...")
encoder = BiEncoder()
image_index = FAISSIndex.load('data/indices/image_index.faiss')
text_index = FAISSIndex.load('data/indices/text_index.faiss')
dataset = Flickr30KDataset('data/images', 'data/results.csv')
engine = MultimodalSearchEngine(encoder, image_index, text_index, dataset)

# 2. Search
print("Searching...")
results = engine.text_to_image_search("children playing on the beach", k=6)

# 3. Display
print("Displaying results...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, (img_name, score) in enumerate(results):
    img = dataset.get_image(img_name)
    axes[i//3, i%3].imshow(img)
    axes[i//3, i%3].set_title(f"{score:.2f}")
    axes[i//3, i%3].axis('off')
plt.suptitle("Children playing on the beach", fontsize=16)
plt.show()

# Done!
print("✓ Search complete!")
```

---

## 🆘 Troubleshooting

### "Module not found"
```bash
pip install -e .
```

### "CUDA out of memory"
```python
# Use CPU instead
encoder = BiEncoder(device='cpu')
```

### "File not found"
Make sure you:
1. Downloaded the dataset
2. Built the indices (see `scripts/build_faiss_indices.py`)

---

## 📚 Want More Details?

Check out the notebooks:
- `notebooks/05_search_demo.ipynb` - Interactive examples
- `notebooks/04_test_faiss_indices.ipynb` - How the search works

---

**That's all you need to know!** Start with `MultimodalSearchEngine` and you'll be searching in no time. 🎉

For the original detailed documentation, check the Git history.
