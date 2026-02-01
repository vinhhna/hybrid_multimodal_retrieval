# LightRAG-GQA: Multi-Scale Knowledge Graph for Visual QA

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.0+-green.svg)](https://networkx.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive multimodal knowledge graph system built on the GQA dataset using LightRAG architecture. Supports **14 query types** (9 basic + 5 advanced) with natural language interface and full reasoning traces.

## 🎯 Features

- **Multi-Scale Knowledge Graphs**: 1K, 10K, or full scale (74,942 images)
- **LightRAG Architecture**: 2-tier structure (Instance + Global levels)
- **14 Query Types**: Complete coverage from entity search to counterfactual reasoning
- **Natural Language Interface**: Query in plain English
- **CLI Tools**: One-command graph building, querying, and evaluation
- **Production Ready**: Optimized for large-scale graphs (100K+ nodes)

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/vinhhna/hybrid_multimodal_retrieval.git
cd hybrid_multimodal_retrieval

# Install package in development mode
pip install -e .

# Or install with optional dependencies
pip install -e ".[dev]"
```

## 🚀 Quick Start

### 1. Build a Knowledge Graph

```bash
# Build 10K scale graph (recommended for testing)
lightrag-gqa-build --scale 10k

# Build 1K scale graph (fastest)
lightrag-gqa-build --scale 1k

# Build full scale graph (requires ~8GB RAM)
lightrag-gqa-build --scale full --input sceneGraphs/train_sceneGraphs.json
```

### 2. Query the Graph

```bash
# Interactive mode
lightrag-gqa-query --scale 10k --interactive

# Single query
lightrag-gqa-query --scale 10k --query "Find all images with dogs wearing hats"

# Query specific graph file
lightrag-gqa-query --graph experiments/sample_10k/gqa_lightrag.gpickle --query "How many people are wearing red shirts?"
```

### 3. Run Evaluation

```bash
# Run comprehensive evaluation
lightrag-gqa-eval --config evaluation_v0_1/configs/eval.yaml
```

## 📊 Query Types

### Basic Queries (9 types)

1. **Entity Search** - Find entities by concept/attributes
   ```python
   from lightrag_gqa.basic_queries import GQA_Reasoning_Engine
   engine = GQA_Reasoning_Engine(scale='10k')
   results = engine.entity_search(concept='dog', limit=10)
   ```

2. **Statistical Queries** - Count, aggregate
3. **Similarity & Pattern Matching** - Semantic similarity
4. **Relational Path Discovery** - Find connections
5. **Negative Constraints** - NOT queries
6. **Comparative Queries** - Entity comparisons
7. **Hierarchical Queries** - Concept hierarchies
8. **Anomaly Detection** - Unusual patterns
9. **Visual-Attribute Constraints** - Complex constraints

### Advanced Queries (5 types)

10. **Chain Reasoning** - Multi-hop traversal
    ```python
    from lightrag_gqa.advanced_queries import AdvancedReasoningEngine
    engine = AdvancedReasoningEngine(scale='10k')
    result = engine.chain_reasoning(
        start_concept='person',
        chain=[{'relation': 'wearing', 'concept': 'shirt'}],
        limit=10
    )
    ```

11. **Pattern Matching** - Subgraph isomorphism
12. **Scene Comparison** - Structural similarity
13. **Counterfactual Reasoning** - What-if analysis
14. **Centrality Queries** - Node importance

## 📁 Project Structure

```
hybrid_multimodal_retrieval/
├── src/lightrag_gqa/          # Main package
│   ├── basic_queries/         # 9 standard query types
│   │   ├── builder.py         # Graph builder
│   │   ├── reasoning_engine.py
│   │   ├── nl_parser.py
│   │   └── query_interface.py
│   ├── advanced_queries/      # 5 advanced reasoning types
│   │   ├── reasoning_engine.py
│   │   ├── demo_advanced_reasoning.ipynb
│   │   └── evaluation_advanced_reasoning.ipynb
│   ├── evaluation/            # CQR-based evaluation
│   ├── datasets/              # GQA data loaders
│   ├── utils/                 # Shared utilities
│   └── cli/                   # Command-line tools
│       ├── build_graph.py
│       ├── query.py
│       └── eval.py
├── experiments/               # Saved graphs
│   ├── sample_1k/            # 1K graph (6.15 MB)
│   ├── sample_10k/           # 10K graph (61.86 MB)
│   └── full/                 # Full graph (470.61 MB)
├── sceneGraphs/              # GQA dataset
├── evaluation_v0_1/          # Legacy evaluation (still usable)
├── pyproject.toml            # Package configuration
└── README.md
```

## 💻 Python API Examples

### Basic Usage

```python
from lightrag_gqa.basic_queries import GQA_Reasoning_Engine

# Initialize engine
engine = GQA_Reasoning_Engine(scale='10k')

# Entity search
results = engine.entity_search(concept='person', limit=10)

# Statistical query
stats = engine.statistical_query(
    query_type='count_by_concept',
    concept='dog'
)

# Path discovery
paths = engine.relational_path_query(
    source='person',
    target='table',
    max_depth=3
)
```

### Advanced Reasoning

```python
from lightrag_gqa.advanced_queries import AdvancedReasoningEngine

# Initialize advanced engine
engine = AdvancedReasoningEngine(scale='10k')

# Multi-hop chain reasoning
result = engine.chain_reasoning(
    start_concept='person',
    chain=[
        {'relation': 'wearing', 'concept': 'shirt'},
        {'relation': 'to the left of', 'concept': 'table'}
    ],
    limit=5
)

# Pattern matching
result = engine.pattern_matching(
    pattern_nodes=['person', 'shirt', 'hat'],
    pattern_edges=[
        ('person', 'wearing', 'shirt'),
        ('person', 'wearing', 'hat')
    ],
    limit=10
)

# Scene comparison
result = engine.scene_comparison(
    image_id_1='2386621',
    image_id_2='2373554'
)

# Print reasoning trace
result.print_result(verbose=True)
```

### Natural Language Queries

```python
from lightrag_gqa.basic_queries import QueryInterface

# Initialize interface
interface = QueryInterface(scale='10k')

# Execute natural language query
response = interface.execute_natural_language_query(
    nl_query="Find all images with dogs wearing hats",
    limit=10
)

print(response.formatted_output)
```

## 📈 Performance

| Scale | Nodes | Edges | Build Time | Query Time (avg) | Memory |
|-------|-------|-------|------------|------------------|--------|
| 1K | 16,458 | 73,873 | ~30s | <50ms | ~1GB |
| 10K | 164,585 | 738,736 | ~5min | <100ms | ~3GB |
| Full | 1.2M+ | 5.5M+ | ~40min | <200ms | ~8GB |

## 🔬 Evaluation

The system includes comprehensive evaluation:

```bash
# Run full evaluation suite
lightrag-gqa-eval

# Results saved to: evaluation_v0_1/results/
```

**Latest Results (10K scale)**:
- MRR: 0.952
- NDCG@10: 0.890
- MAP: 0.904
- Advanced queries: 88.7% success rate

## 📚 Documentation

- **Basic Queries**: See inline docstrings in `src/lightrag_gqa/basic_queries/`
- **Advanced Queries**: See inline docstrings in `src/lightrag_gqa/advanced_queries/`
- **Notebooks**: Interactive demos in `src/lightrag_gqa/advanced_queries/*.ipynb`

## 🛠️ Development

```bash
# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Run tests (if available)
pytest

# Format code
black src/

# Type checking
mypy src/
```

## 📄 Dataset

This project uses the [GQA dataset](https://cs.stanford.edu/people/dorarad/gqa/):
- Training: 74,942 scene graphs
- Validation: 10,234 scene graphs

Place scene graph files in `sceneGraphs/`:
- `train_sceneGraphs.json`
- `val_sceneGraphs.json`

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 Citation

If you use this work, please cite:

```bibtex
@misc{lightrag-gqa,
  title={LightRAG-GQA: Multi-Scale Knowledge Graph for Visual Question Answering},
  author={GQA LightRAG Project},
  year={2026},
  howpublished={\url{https://github.com/vinhhna/hybrid_multimodal_retrieval}}
}
```

## 📜 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- GQA Dataset: Stanford Vision Lab
- LightRAG Architecture: Inspired by retrieval-augmented generation frameworks
- NetworkX: Graph algorithms library

---

**Status**: Production-ready ✅  
**Maintained**: Yes  
**Python**: 3.8+  
**Last Updated**: February 2026
