# Evaluation Framework v0.1 for LightRAG-GQA

A comprehensive evaluation framework for the GQA LightRAG knowledge graph system.  
**Track R**: Retrieval/Query Engine Evaluation + Parser Comparison.

## Overview

This framework evaluates the query engine and parsers using ground truth derived **solely from GQA Scene Graphs** (not from questions1.2/).

### Two-Layer Evaluation Design

- **Layer A (Engine-only)**: Evaluate engine execution using GOLD structured queries (CQR) generated from sceneGraphs.
- **Layer B (Parser comparison)**: Compare heuristic vs LLM parser:
  1. Parser-only: NL → CQR accuracy against GOLD CQR
  2. End-to-end: NL → parsed CQR → engine → outputs against GOLD outputs

### Supported Output Types

| Output Type | Description | Metrics |
|-------------|-------------|---------|
| `ranked_set` | Ranked list of image_ids | P@k, R@k, NDCG@k, MRR |
| `scalar` | Counts/probabilities/ratios | MAE, RMSE, Spearman |
| `path` | Graph paths between concepts | Validity rate, Hop accuracy |
| `subgraph` | Evidence subgraph | Node/Edge recall, Compactness |

---

## Repo Integration Notes

### Classes and Functions Used

#### 1. Knowledge Graph Loading
```python
# From: src/gqa_reasoning_engine.py
from src.gqa_reasoning_engine import GQA_Reasoning_Engine

# Initialize with scale
engine = GQA_Reasoning_Engine(scale='10k')  # Options: '1k', '10k', 'full'

# Or with explicit path
engine = GQA_Reasoning_Engine(graph_path='experiments/sample_10k/gqa_lightrag.gpickle')

# Graph is a networkx.DiGraph
graph = engine.graph
```

#### 2. Query Execution
```python
# From: src/gqa_query_interface.py
from src.gqa_query_interface import QueryInterface, QueryResponse

# Initialize
interface = QueryInterface(scale='10k', verbose=False)

# Execute NL query
response: QueryResponse = interface.query("Find all red cars", limit=50)

# Response fields:
# - success: bool
# - query_type: str (e.g., "entity_search", "negative_constraints")
# - results: Any (list of dicts with 'image_id', 'node_id', etc.)
# - reasoning_trace: List[str]
# - metadata: Dict
```

#### 3. Natural Language Parser
```python
# From: src/gqa_nl_parser.py
from src.gqa_nl_parser import NaturalLanguageParser, ParseResult, QueryType

parser = NaturalLanguageParser()
result: ParseResult = parser.parse("Find all red cars")

# ParseResult fields:
# - query_type: QueryType enum
# - params: Dict[str, Any] (e.g., {'concept': 'car', 'attributes': ['red']})
# - confidence: float
# - original_query: str
# - normalized_query: str
```

#### 4. Engine Methods (Direct Access)
```python
# Entity search
result = engine.entity_search(concept='car', attributes=['red'], limit=20)

# Statistical knowledge
result = engine.statistical_knowledge(concept_a='man', concept_b='shirt', relation='wearing')

# Relational path
result = engine.relational_path(source_concept='man', target_concept='shirt', max_hops=3)

# Negative constraints
result = engine.negative_constraints(concept_present='tree', concept_absent='sky', limit=20)
```

#### 5. Result Format (ReasoningResult)
```python
# From: src/gqa_reasoning_engine.py
@dataclass
class ReasoningResult:
    query_type: str
    question: str
    results: Any  # List[Dict] or Dict depending on query type
    reasoning_trace: List[str]
    metadata: Dict[str, Any]

# For entity_search, results is List[Dict]:
# [{'node_id': '2345:obj1', 'image_id': '2345', 'object_id': 'obj1', 'name': 'car', 'attributes': ['red']}]
```

### Graph Node Structure

Instance nodes (objects):
- `node_type`: 'instance'
- `image_id`: str
- `name`: str (concept name)
- `attributes`: List[str]

Global concept nodes:
- `node_type`: 'global_concept'
- Node ID format: `Concept:{name}`

Global attribute nodes:
- `node_type`: 'global_attribute'  
- Node ID format: `Attr:{name}`

Edge types:
- `instance_of`: Instance → Global Concept
- `has_attribute`: Instance → Global Attribute
- `semantic_relation`: Instance ↔ Instance (with `relation` attribute)

### KG Scale Selection

| Scale | Path | Approx Size |
|-------|------|-------------|
| 1k | `experiments/sample_1k/gqa_lightrag.gpickle` | ~6 MB |
| 10k | `experiments/sample_10k/gqa_lightrag.gpickle` | ~62 MB |
| full | `experiments/full/gqa_lightrag.gpickle` | ~471 MB |

---

## Folder Structure

```
evaluation_v0_1/
├── README.md                    # This file
├── configs/
│   └── eval.yaml               # Configuration file
├── data/
│   ├── suites/                 # Generated query suites (JSONL)
│   │   ├── suite_ranked_entity_attr.jsonl
│   │   ├── suite_ranked_negative.jsonl
│   │   ├── suite_scalar_stats.jsonl
│   │   ├── suite_path.jsonl
│   │   └── suite_subgraph.jsonl
│   └── synonym_map.json        # Term normalization mappings
├── scripts/
│   ├── cqr.py                  # Canonical Query Representation
│   ├── normalize.py            # Term normalization utilities
│   ├── metrics.py              # Evaluation metrics
│   ├── oracle_scenegraphs.py   # Ground truth from scene graphs
│   ├── generate_suites.py      # Query suite generator
│   ├── adapter_engine.py       # CQR → Engine adapter
│   ├── llm_parser_stub.py      # LLM parser placeholder
│   ├── evaluate_engine.py      # Layer A evaluation
│   ├── evaluate_parser.py      # Parser-only evaluation
│   ├── evaluate_e2e.py         # End-to-end evaluation
│   ├── run_all.py              # Main orchestration script
│   └── test_metrics.py         # Unit tests for metrics
└── results/                    # Generated at runtime
    ├── engine_*.{csv,json,md}
    ├── parser_*.{csv,json}
    ├── e2e_*.{csv,json,md}
    └── summary.{md,json}
```

---

## Quick Start

```bash
# Run full evaluation pipeline (recommended)
python evaluation_v0_1/scripts/run_all.py --config evaluation_v0_1/configs/eval.yaml --verbose

# Force regeneration of suites
python evaluation_v0_1/scripts/run_all.py --config evaluation_v0_1/configs/eval.yaml --regen

# Run specific step only
python evaluation_v0_1/scripts/run_all.py --config evaluation_v0_1/configs/eval.yaml --step generate
python evaluation_v0_1/scripts/run_all.py --config evaluation_v0_1/configs/eval.yaml --step engine
python evaluation_v0_1/scripts/run_all.py --config evaluation_v0_1/configs/eval.yaml --step parser
python evaluation_v0_1/scripts/run_all.py --config evaluation_v0_1/configs/eval.yaml --step e2e

# Run individual evaluation scripts
python -m evaluation_v0_1.scripts.generate_suites --config evaluation_v0_1/configs/eval.yaml
python -m evaluation_v0_1.scripts.evaluate_engine --config evaluation_v0_1/configs/eval.yaml -v
python -m evaluation_v0_1.scripts.evaluate_parser --config evaluation_v0_1/configs/eval.yaml -v
python -m evaluation_v0_1.scripts.evaluate_e2e --config evaluation_v0_1/configs/eval.yaml -v
```

---

## Canonical Query Representation (CQR)

### Structure

```python
@dataclass
class CQR:
    query_id: str
    output_type: str  # "ranked_set", "scalar", "path", "subgraph"
    op: str           # "retrieve", "stats", "path", "similarity", "compare"
    must: List[Constraint]
    must_not: List[Constraint]
    k: int = 50
    return_unit: str = "images"
    meta: Dict = field(default_factory=dict)
```

### Constraint Types

```python
# Concept constraint
{"type": "concept", "value": "car"}

# Attribute constraint
{"type": "attr", "value": "red"}

# Relation constraint
{"type": "rel", "value": {"name": "on", "obj": "road"}}
```

---

## Ground Truth Generation

Ground truth is derived **exclusively** from GQA Scene Graphs:
- `sceneGraphs/train_sceneGraphs.json`
- `sceneGraphs/val_sceneGraphs.json`

### Scene Graph Format

```json
{
  "image_id": {
    "objects": {
      "obj_id": {
        "name": "car",
        "attributes": ["red", "large"],
        "relations": [
          {"name": "on", "object": "other_obj_id"}
        ]
      }
    }
  }
}
```

### Oracle Output Types

1. **ranked_set**: Set of image_ids where constraint is satisfied
2. **scalar**: Computed probability P(B|A,rel) = count(A rel B) / count(A)
3. **path**: Existence and shortest hops on concept graph
4. **subgraph**: Minimal evidence nodes and edges

---

## Negative Constraint Safeguards

Because missing annotations can cause false negatives, we apply:
- Minimum concept frequency threshold (default: 200 occurrences)
- Minimum gold set size threshold (default: 50 images)
- Only use high-frequency concepts for `must_not` constraints

---

## Future Extensions

### Adding External Scene Graph Datasets

To plug in Visual Genome or other scene graph datasets:

1. Implement a loader in `oracle_scenegraphs.py`:
```python
def load_visual_genome(path: str) -> Dict[str, SceneGraph]:
    # Convert VG format to internal SceneGraph format
    pass
```

2. Update `eval.yaml`:
```yaml
paths:
  external_scenegraphs: "path/to/visual_genome.json"
  scenegraph_format: "visual_genome"  # or "gqa"
```

3. The suite generator and oracle will work unchanged if the loader normalizes to the same internal format.

---

## Running Tests

```bash
# Run metrics unit tests
python -m pytest evaluation_v0_1/scripts/test_metrics.py -v

# Or directly
python evaluation_v0_1/scripts/test_metrics.py
```

---

## Configuration Reference

See `configs/eval.yaml` for all configuration options including:
- Paths to scene graphs and KG
- Suite sizes
- k-values for ranking metrics
- Random seed for reproducibility
- Negative constraint safeguards
- Normalization settings

