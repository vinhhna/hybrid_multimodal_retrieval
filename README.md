# LightRAG-VG150

A LightRAG-style knowledge graph implementation using the Visual Genome 150 (VG150) dataset.

## Overview

This project builds a document-centric knowledge graph from Visual Genome scene graphs, enabling semantic retrieval over images via their scene graph representations.

### Graph Structure

Following the LightRAG paradigm:

- **Document nodes**: Images (each image is a document)
- **Chunk nodes**: Relationship triples (`person riding horse`) and attribute assertions (`person is tall`)
- **Entity nodes**: Canonical labels for objects, predicates, and attributes

**Edges**:
- `Document → HAS_CHUNK → Chunk`: Links images to their scene graph facts
- `Chunk → MENTIONS → Entity`: Links facts to the entities they mention
- `Entity → RELATED_TO → Entity`: Derived from relationship triples

## Installation

```bash
# Clone the repository
git clone https://github.com/vinhhna/hybrid_multimodal_retrieval.git
cd hybrid_multimodal_retrieval

# Checkout this branch
git checkout lightrag-vg150

# Install in development mode
pip install -e ".[dev]"
```

## Dataset: VG150

This project uses the VG150 preprocessed scene graph files from the [Scene Graph Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).

### Required Files

| File | Description |
|------|-------------|
| `VG-SGG-with-attri.h5` | HDF5 file with scene graph annotations |
| `VG-SGG-dicts-with-attri.json` | Label dictionaries (objects, predicates, attributes) |
| `image_data.json` | *(Optional)* Image metadata |

### Obtaining the Data

1. Download from the Scene Graph Benchmark repository:
   ```bash
   # Example (check the repository for current links)
   mkdir -p data
   cd data
   # Download VG-SGG-with-attri.h5 (~1.1GB)
   # Download VG-SGG-dicts-with-attri.json (~200KB)
   ```

2. Or use your own copy and point to it with `--data_dir`

### Validate Dataset

```bash
python scripts/fetch_vg150.py --data_dir ./data
```

Expected output:
```
✓ All required VG150 files found in ./data
  - Object classes: 150
  - Predicate classes: 50
  - Attribute classes: 200
  - Images: 108073
```

## Usage

### 1. Build the Knowledge Graph

```bash
# Full dataset
python scripts/build_graph.py --data_dir ./data --out_dir ./output

# Sample mode (first 1000 images, for testing)
python scripts/build_graph.py --data_dir ./data --out_dir ./output --sample 1000
```

Expected output:
```
Loading VG150 dataset from ./data...
  Loaded 1000 images
Building knowledge graph...
100%|██████████████████████████████| 1000/1000 [00:05<00:00, 189.23it/s]
  Nodes: 45230
  Edges: 156789
  Node types: {'document': 1000, 'chunk': 32456, 'entity': 11774}
Saving graph to ./output/graph.db...
✓ Graph built successfully: ./output/graph.db
```

### 2. Query the Graph

```bash
python scripts/query_graph.py --graph_dir ./output --query "person riding horse"
```

Expected output:
```
Loading graph from ./output/graph.db...
Building query indices...
Indexed 32456 chunks, 11774 entities with chunks

Query: person riding horse
------------------------------------------------------------

[1] Image 2456 (score: 0.8934)
    Evidence:
      - person riding horse
      - horse on grass
      - person wearing helmet
    Matched: person, riding, horse

[2] Image 8921 (score: 0.7123)
    Evidence:
      - man riding horse
      - horse near fence
    Matched: horse, riding

[3] Image 1234 (score: 0.6547)
    Evidence:
      - woman on horse
      - horse in field
    Matched: horse
```

## Example Queries

### Query 1: Relationship search
```bash
python scripts/query_graph.py --graph_dir ./output --query "dog running on grass"
```
```
[1] Image 5432 (score: 0.9012)
    Evidence:
      - dog running on grass
      - dog is brown
    Matched: dog, running, grass

[2] Image 7821 (score: 0.7234)
    Evidence:
      - dog on grass
      - dog playing
    Matched: dog, grass
```

### Query 2: Object + attribute
```bash
python scripts/query_graph.py --graph_dir ./output --query "red car on street"
```
```
[1] Image 3421 (score: 0.8756)
    Evidence:
      - car on street
      - car is red
      - car near building
    Matched: car, street, red

[2] Image 9087 (score: 0.6543)
    Evidence:
      - car parked on street
    Matched: car, street
```

### Query 3: Multi-hop exploration
```bash
python scripts/query_graph.py --graph_dir ./output --query "kitchen table food" --hops 2
```
```
[1] Image 4567 (score: 0.8234)
    Evidence:
      - food on table
      - table in kitchen
      - plate on table
    Matched: kitchen, table, food

[2] Image 2341 (score: 0.7123)
    Evidence:
      - table near window
      - food on plate
    Matched: table, food
```

### Query 4: Attribute-focused
```bash
python scripts/query_graph.py --graph_dir ./output --query "tall building"
```
```
[1] Image 8765 (score: 0.8901)
    Evidence:
      - building is tall
      - building in city
    Matched: tall, building

[2] Image 1234 (score: 0.7654)
    Evidence:
      - building near street
      - building is large
    Matched: building
```

### Query 5: Action-based
```bash
python scripts/query_graph.py --graph_dir ./output --query "person eating pizza"
```
```
[1] Image 6789 (score: 0.9234)
    Evidence:
      - person eating pizza
      - pizza on table
    Matched: person, eating, pizza

[2] Image 3456 (score: 0.7890)
    Evidence:
      - man holding pizza
      - person at table
    Matched: person, pizza
```

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_smoke.py -v
```

## Project Structure

```
lightrag-vg150/
├── pyproject.toml          # Package configuration
├── README.md               # This file
├── .gitignore
├── src/
│   └── lightrag_vg150/
│       ├── __init__.py
│       ├── dataset.py      # VG150 data loader
│       ├── graph.py        # Graph builder and storage
│       ├── query.py        # Query engine (BM25 + entity expansion)
│       └── cli.py          # CLI entrypoints
├── scripts/
│   ├── fetch_vg150.py      # Validate/fetch dataset
│   ├── build_graph.py      # Build knowledge graph
│   └── query_graph.py      # Query the graph
└── tests/
    └── test_smoke.py       # Smoke tests (synthetic data)
```

## How It Works

### Graph Building

1. **Load VG150**: Parse the HDF5 file to extract scene graphs per image
2. **Create Document Nodes**: One node per image
3. **Create Chunk Nodes**: 
   - Relationship chunks: `"person riding horse"` (subject-predicate-object)
   - Attribute chunks: `"person is tall"` (object-attribute)
4. **Create Entity Nodes**: Deduplicated canonical labels
5. **Create Edges**: Link documents → chunks → entities

### Querying

1. **Local Retrieval**: BM25 search over chunk text
2. **Entity Matching**: Find entities that match query terms
3. **Graph Expansion**: Traverse `RELATED_TO` edges (1-2 hops)
4. **Scoring**: Combine local (BM25) and global (expansion) signals
5. **Ranking**: Return top-k images with evidence chunks

## License

MIT License
