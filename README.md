# GQA LightRAG: Multimodal Knowledge Graph System

A comprehensive Multimodal Knowledge Graph system built on the GQA (Visual Reasoning) dataset using LightRAG architecture. Supports natural language queries in English with 9 distinct query types for visual reasoning tasks.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.0+-green.svg)](https://networkx.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Features

- **Multi-Scale Knowledge Graphs**: Build graphs at 1K, 10K, or full scale (74,942 images)
- **LightRAG Architecture**: 2-tier structure with Instance and Global levels
- **9 Query Types**: Entity search, statistical, similarity, relational paths, and more
- **Natural Language Interface**: Query in plain English without writing code
- **Comprehensive Reasoning**: Full reasoning traces for all queries
- **Production Ready**: Tested on full GQA dataset with optimized performance

## 📁 Project Structure

```
hybrid_multimodal_retrieval/
├── src/                          # Core source modules
│   ├── __init__.py              # Package initialization
│   ├── gqa_lightrag_builder.py  # Multi-scale KG builder
│   ├── gqa_reasoning_engine.py  # Query engine (9 types)
│   ├── gqa_nl_parser.py         # Natural language parser
│   └── gqa_query_interface.py   # CLI and interactive interface
├── scripts/                      # Demo and test scripts
│   ├── demo_nl_interface.py     # Comprehensive demo
│   ├── test_all_queries.py      # Automated testing
│   └── run_demo_and_save.py     # UTF-8 demo wrapper
├── experiments/                  # Saved knowledge graphs
│   ├── sample_1k/               # 1,000 image graph (6.15 MB)
│   ├── sample_10k/              # 10,000 image graph (61.86 MB)
│   └── full/                    # 74,942 image graph (470.61 MB)
├── sceneGraphs/                 # GQA dataset files
│   ├── train_sceneGraphs.json
│   └── val_sceneGraphs.json
└── data/                        # Output and cache directory
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/vinhhna/hybrid_multimodal_retrieval.git
cd hybrid_multimodal_retrieval

# Install dependencies
pip install networkx tqdm

# Verify installation
python -c "import networkx; import tqdm; print('✓ All dependencies installed')"
```

### Build Knowledge Graph

```bash
# Build 1K sample (fast, for testing)
python src/gqa_lightrag_builder.py --scale 1k

# Build 10K sample (medium scale)
python src/gqa_lightrag_builder.py --scale 10k

# Build full graph (production)
python src/gqa_lightrag_builder.py --scale full
```

### Query the Knowledge Graph

#### Option 1: Natural Language Interface (Recommended)

```bash
# Interactive mode
python src/gqa_query_interface.py --scale full

# Single query
python src/gqa_query_interface.py --scale full --query "Find all red cars"
```

#### Option 2: Python API

```python
from src.gqa_reasoning_engine import GQA_Reasoning_Engine

# Initialize engine
engine = GQA_Reasoning_Engine(scale='full')

# Entity search
results = engine.entity_search(
    concept='car',
    attributes=['red', 'large'],
    limit=10
)

# Statistical knowledge
stats = engine.statistical_knowledge(
    concept_a='man',
    concept_b='shirt',
    relation='wearing'
)

# Print results
results.print_summary()
```

### Run Demo

```bash
# Run all 9 query types
python scripts/demo_nl_interface.py --scale full

# Run with UTF-8 output
python scripts/run_demo_and_save.py --scale full --output demo_results.txt
```

## 📊 Supported Query Types

| # | Query Type | Description | Example |
|---|------------|-------------|---------|
| 1 | **Entity Search** | Find objects by concept and attributes | "Find red cars" |
| 2 | **Statistical Knowledge** | Calculate co-occurrence probabilities | "Probability of shirt near man" |
| 3 | **Similarity Search** | Find similar objects | "Find objects like green trees" |
| 4 | **Relational Path** | Discover paths between objects | "Path from man to shirt" |
| 5 | **Negative Constraints** | Find images with/without objects | "Images with man but no woman" |
| 6 | **Comparative** | Compare object distributions | "More white walls or white beds?" |
| 7 | **Hierarchical** | Explore category hierarchies | "All furniture types" |
| 8 | **Anomaly Detection** | Find unusual relations | "Dogs on tables" |
| 9 | **Multi-Attribute** | Complex attribute combinations | "Red plastic cups" |

## 📈 Performance Statistics

### Full Dataset (74,942 images)

```
Nodes:              1,233,453
  ├── Instance:     1,231,134  (99.8%)
  ├── Concept:          1,702  (0.14%)
  └── Attribute:          617  (0.05%)

Edges:              5,634,778
  ├── instance_of:  1,231,134  (21.9%)
  ├── has_attribute:  675,340  (12.0%)
  └── semantic_rel: 3,795,907  (67.4%)

Build Time:            ~25 seconds
Query Time:            <1 second/query
Memory Usage:          2-3 GB RAM
```

### Scaling Comparison

| Scale | Images | Nodes | Edges | File Size | Build Time |
|-------|--------|-------|-------|-----------|------------|
| 1K | 1,000 | 17,463 | 72,713 | 6.15 MB | ~1s |
| 10K | 10,000 | 164,585 | 738,736 | 61.86 MB | ~5s |
| Full | 74,942 | 1,233,453 | 5,634,778 | 470.61 MB | ~25s |

## 🏗️ Architecture

### LightRAG 2-Tier Structure

```
┌─────────────────────────────────────────────────────────┐
│                    Global Level                         │
│  ┌──────────────┐           ┌──────────────┐          │
│  │   Concept    │           │  Attribute   │          │
│  │   Nodes      │           │    Nodes     │          │
│  └──────┬───────┘           └──────┬───────┘          │
│         │ instance_of              │ has_attribute     │
└─────────┼──────────────────────────┼───────────────────┘
          │                          │
┌─────────┴──────────────────────────┴───────────────────┐
│                  Instance Level                         │
│  ┌──────────┐   semantic    ┌──────────┐              │
│  │ Object 1 │ ────────────→ │ Object 2 │              │
│  │  (car)   │   relation    │ (street) │              │
│  └──────────┘               └──────────┘              │
│        Image: 2417431.jpg                              │
└─────────────────────────────────────────────────────────┘
```

### Edge Types

- **instance_of**: Links instance nodes to concept nodes
- **has_attribute**: Links instance nodes to attribute nodes
- **semantic_relation**: Links related instance nodes (on, near, wearing, etc.)

## 💡 Example Queries

### Natural Language Examples

```bash
# Entity Search
"Find all red cars"
"Show me large green trees"

# Statistical
"What is the probability of finding a shirt near a man?"
"How often do windows appear near buildings?"

# Similarity
"Find objects similar to white shirts"

# Relational Paths
"Show paths between man and shirt"
"How is plate connected to table?"

# Negative Constraints
"Find images with man but no woman"
"Show images with tree but no car"

# Comparative
"Compare white walls vs white beds"
"Which is more common: chair in kitchen or chair in living room?"

# Hierarchical
"Show all types of furniture"
"List all electronic devices"

# Anomaly Detection
"Find unusual cases of dog on table"

# Multi-Attribute
"Find red plastic cups"
"Show tall men wearing black shirts"
```

## 🛠️ Development

### Running Tests

```bash
# Test all 9 query types
python scripts/test_all_queries.py --scale 10k

# Test specific query type
python src/gqa_reasoning_engine.py --scale 10k --demo part1
```

### Building Custom Graphs

```python
from src.gqa_lightrag_builder import GQALightRAGGraphBuilder

# Initialize builder
builder = GQALightRAGGraphBuilder(
    scene_graphs_dir="sceneGraphs",
    output_dir="experiments"
)

# Load custom data
scene_graphs = builder.load_scene_graphs(max_images=5000)

# Build graph
graph = builder.build_graph(scene_graphs)

# Save to custom location
builder.save_graph("experiments/custom/my_graph.gpickle")
```

## 📚 Documentation

- [Full Dataset Results](experiments/full/README.md) - Detailed results and statistics
- [API Documentation](docs/API.md) - Complete API reference (coming soon)
- [Query Examples](docs/QUERIES.md) - Comprehensive query examples (coming soon)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **GQA Dataset**: [Visual Reasoning in the Real World](https://cs.stanford.edu/people/dorarad/gqa/)
- **LightRAG**: Lightweight architecture for knowledge graph reasoning
- **IT3930E - Project III**: Hanoi University of Science and Technology

## 📧 Contact

**Author**: vinhhna  
**Repository**: [hybrid_multimodal_retrieval](https://github.com/vinhhna/hybrid_multimodal_retrieval)  
**Branch**: lightrag-vg150

---

**Built with ❤️ for Visual Reasoning Research**