# LightRAG-GQA: Two-Level Knowledge Graph Reasoning on GQA Scene Graphs

**Academic Technical Report**

*Project: IT3930E - Project III (2026)*

---

## Abstract

This project presents **LightRAG-GQA**, a two-level knowledge graph system for structured visual reasoning on the GQA dataset. The system ingests scene graphs (JSON representations of objects, attributes, and relations within images) and constructs a normalized knowledge graph with distinct **instance-level** (per-image object occurrences) and **global-level** (aggregated concepts/attributes) nodes. We implement **9 query types** (5 basic + 4 advanced) spanning entity retrieval, statistical analysis, path traversal, pattern matching, and counterfactual reasoning, accessible via both programmatic API and rule-based natural language parsing.

We evaluate the reasoning engine (Layer A) on gold-standard Canonical Query Representations (CQR) using standard information retrieval metrics: NDCG@k, MRR, F1 for ranked outputs; MAE, RMSE for scalar outputs; and path validity rate for graph traversals. The natural language parsing layer (Layer B) remains **unevaluated** due to absence of annotated NL-to-CQR ground truth. This document provides a complete technical specification aligned with implementation reality.

### Key Claims (Verified Against Code)

| Claim | Status | Evidence |
|-------|--------|----------|
| 9 query types implemented (5 basic + 4 advanced) | ✅ Verified | [traceability.md](docs/traceability.md) |
| Two-level graph architecture (instance + global nodes) | ✅ Verified | [definitions.md](docs/definitions.md) |
| Deterministic query execution on fixed scene graphs | ✅ Verified | BFS/set operations are deterministic |
| Traceable reasoning steps | ✅ Verified | `ReasoningTrace` objects returned |
| Rule-based NL parser | ✅ Verified | [nl_parser.py](src/lightrag_gqa/basic_queries/nl_parser.py) |
| Layer A evaluation (engine-only) implemented | ✅ Verified | [evaluation_v0_1/](evaluation_v0_1/) |
| Layer B evaluation (parser) implemented | ❌ Not Implemented | No gold NL→CQR annotations |
| 14 query types as claimed in old README | ❌ Incorrect | Only 9 fully implemented |

---

## 1. Scope & Assumptions

### 1.1 In-Scope

| Aspect | Description |
|--------|-------------|
| **Data** | GQA scene graphs (JSON): objects with names, bounding boxes, attributes, and relations |
| **Queries** | Entity search, statistical, similarity, path, negative constraints (basic); chain, pattern, scene comparison, counterfactual (advanced) |
| **Graph Construction** | Two-level LightRAG architecture with normalization |
| **Parsing** | Rule-based regex pattern matching for natural language queries |
| **Evaluation** | Layer A (engine-only with gold CQR); IR metrics for ranked/scalar/path outputs |
| **Scales** | 1K, 10K, and full scene graph collections |

### 1.2 Out-of-Scope

| Aspect | Description |
|--------|-------------|
| **Image Features** | No visual feature extraction (RGB pixels, embeddings) |
| **End-to-End VQA** | Not a full VQA system; answers structured graph queries, not free-form questions |
| **LLM-Based Parsing** | Only heuristic parser; no trained or LLM-based query understanding |
| **Cross-Image Reasoning** | Relations are image-scoped; no global trends across images |
| **Temporal/Causal** | Scene graphs are static snapshots; no temporal ordering |
| **Layer B Evaluation** | No ground-truth annotations for parser accuracy |

### 1.3 Input Assumptions

1. **Schema Compliance**: Input JSON conforms to GQA scene graph schema (see [definitions.md § 1](docs/definitions.md#1-input-data-format-gqa-scene-graphs))
2. **Image Closure**: All `semantic_relation` edges connect objects within the same image
3. **Attribute Vocabulary**: Attributes are string-valued; out-of-vocabulary terms return empty results
4. **Relation Directionality**: Relations are directed (e.g., "wearing" is not symmetric)
5. **No External Knowledge**: The system operates only on the provided scene graphs

---

## 2. Formal Definitions

### 2.1 Two-Level Graph Schema

The LightRAG architecture defines three node types and three edge types:

#### Node Types

| Type | ID Format | Level | Description |
|------|-----------|-------|-------------|
| **Instance** | `{image_id}:{object_id}` | Instance | Single object occurrence in a specific image |
| **Global Concept** | `Concept:{normalized_name}` | Global | Aggregated category across all images |
| **Global Attribute** | `Attr:{normalized_name}` | Global | Aggregated visual property across all images |

#### Edge Types

| Type | Direction | Meaning | Cardinality |
|------|-----------|---------|-------------|
| `instance_of` | Instance → Concept | "This object is a {concept}" | Many-to-one |
| `has_attribute` | Instance → Attribute | "This object has property {attr}" | Many-to-many |
| `semantic_relation` | Instance → Instance | Spatial/semantic relation | Many-to-many (within-image only) |

#### Normalization

All concept and attribute names are normalized: `normalize(text) = lowercase(strip(text))`

### 2.2 Canonical Query Representation (CQR)

Each query type has a defined JSON schema. Queries are executed by the reasoning engine using these structured inputs.

#### CQR JSON Schema (Common Structure)

```json
{
  "query_type": "string",       // One of the 9 implemented types
  "parameters": { ... },        // Type-specific parameters
  "limit": "integer",           // Optional: max results to return
  "output_type": "string"       // One of: ranked_set, scalar, path, subgraph
}
```

#### CQR Examples by Query Type

**Type 1: Entity Search** (`output_type: ranked_set`)
```json
{
  "query_type": "entity_search",
  "parameters": {
    "concept": "dog",
    "attributes": ["brown", "large"]
  },
  "limit": 50
}
```

**Type 2: Statistical Knowledge** (`output_type: scalar`)
```json
{
  "query_type": "statistical_knowledge",
  "parameters": {
    "concept_a": "person",
    "concept_b": "shirt",
    "relation": "wearing"
  }
}
```

**Type 3: Similarity Search** (`output_type: ranked_set`)
```json
{
  "query_type": "similarity_search",
  "parameters": {
    "concept": "tree",
    "attributes": ["large", "green"],
    "min_common_attributes": 1
  },
  "limit": 10
}
```

**Type 4: Relational Path** (`output_type: path`)
```json
{
  "query_type": "relational_path",
  "parameters": {
    "source_concept": "person",
    "target_concept": "table",
    "via_relation": null,
    "max_hops": 3
  },
  "limit": 5
}
```

**Type 5: Negative Constraints** (`output_type: ranked_set`)
```json
{
  "query_type": "negative_constraints",
  "parameters": {
    "concept_present": "dog",
    "concept_absent": "cat"
  },
  "limit": 100
}
```

**Type 6: Chain Reasoning** (`output_type: path`)
```json
{
  "query_type": "chain_reasoning",
  "parameters": {
    "start_concept": "man",
    "chain": [
      {"relation": "wearing", "concept": "shirt", "attribute": "red"},
      {"relation": "near", "concept": "table"}
    ]
  },
  "limit": 10
}
```

**Type 7: Pattern Matching** (`output_type: subgraph`)
```json
{
  "query_type": "pattern_matching",
  "parameters": {
    "pattern_nodes": ["person", "shirt"],
    "pattern_edges": [
      {"source": "person", "relation": "wearing", "target": "shirt"}
    ]
  },
  "limit": 20
}
```

**Type 8: Scene Comparison** (`output_type: subgraph`)
```json
{
  "query_type": "scene_comparison",
  "parameters": {
    "image_id_1": "2386621",
    "image_id_2": "2373554"
  }
}
```

**Type 9: Counterfactual Reasoning** (`output_type: subgraph`)
```json
{
  "query_type": "counterfactual_reasoning",
  "parameters": {
    "initial_instance": "2386621:0",
    "remove_relation": "wearing",
    "target_concept": "jacket"
  }
}
```

### 2.3 Output Types & Ordering Semantics

| Output Type | Structure | Ordering | Semantics |
|-------------|-----------|----------|-----------|
| `ranked_set` | `List[InstanceDict]` | Insertion order (unranked) | Set-based; order is arbitrary unless explicitly sorted |
| `scalar` | `Dict[str, float]` | N/A | Single numeric value (count, probability, ratio) |
| `path` | `List[List[NodeID]]` | Sorted by path length | Sequence of connected nodes |
| `subgraph` | `Dict[nodes, edges]` | N/A | Unordered subset of knowledge graph |

**Important**: Current implementation does **not** provide relevance-based ranking for `ranked_set` outputs. Results are returned in graph traversal order (effectively arbitrary).

---

## 3. Query Taxonomy

### 3.1 Summary Table

| # | Query Type | Category | Intent | Inputs | Output | Algorithm | Implementation |
|---|------------|----------|--------|--------|--------|-----------|----------------|
| 1 | Entity Search | Basic | Find objects by concept+attributes | concept, attributes[], limit | ranked_set | Set intersection | [reasoning_engine.py#L288](src/lightrag_gqa/basic_queries/reasoning_engine.py#L288) |
| 2 | Statistical Knowledge | Basic | Co-occurrence probability P(B\|A) | concept_a, concept_b, relation? | scalar | Neighbor counting | [reasoning_engine.py#L397](src/lightrag_gqa/basic_queries/reasoning_engine.py#L397) |
| 3 | Similarity Search | Basic | Find similar attribute profiles | concept, attributes[], min_common | ranked_set | Jaccard similarity | [reasoning_engine.py#L508](src/lightrag_gqa/basic_queries/reasoning_engine.py#L508) |
| 4 | Relational Path | Basic | Find paths between concepts | source, target, via_rel?, max_hops | path | BFS traversal | [reasoning_engine.py#L643](src/lightrag_gqa/basic_queries/reasoning_engine.py#L643) |
| 5 | Negative Constraints | Basic | Find A but not B | concept_present, concept_absent | ranked_set | Set difference | [reasoning_engine.py#L818](src/lightrag_gqa/basic_queries/reasoning_engine.py#L818) |
| 6 | Chain Reasoning | Advanced | Multi-hop with constraints | start, chain[], limit | path | Sequential filtering | [advanced/reasoning_engine.py#L188](src/lightrag_gqa/advanced_queries/reasoning_engine.py#L188) |
| 7 | Pattern Matching | Advanced | Find subgraph instances | pattern_nodes[], pattern_edges[] | subgraph | Subgraph enumeration | [advanced/reasoning_engine.py#L386](src/lightrag_gqa/advanced_queries/reasoning_engine.py#L386) |
| 8 | Scene Comparison | Advanced | Compare two images | image_id_1, image_id_2 | subgraph | Structural comparison | [advanced/reasoning_engine.py#L535](src/lightrag_gqa/advanced_queries/reasoning_engine.py#L535) |
| 9 | Counterfactual | Advanced | "What-if" analysis | instance, remove_rel, target | subgraph | Graph modification | [advanced/reasoning_engine.py#L713](src/lightrag_gqa/advanced_queries/reasoning_engine.py#L713) |

### 3.2 Methods Existing in Code but Not Fully Specified

The following methods exist in code but lack complete evaluation and specification:

| Method | Location | Status |
|--------|----------|--------|
| `compare_contexts()` | reasoning_engine.py | Partial implementation |
| `get_hierarchical_entities()` | reasoning_engine.py | Not in evaluation |
| `find_anomalies()` | reasoning_engine.py | Not in evaluation |
| `multi_constraint_search()` | reasoning_engine.py | Not in evaluation |
| `centrality_query()` | advanced/reasoning_engine.py | Not in evaluation |

**Note**: Previous README claimed 14 query types; only 9 are fully implemented and documented. See [limitations_and_failure_modes.md](docs/limitations_and_failure_modes.md#1-claimed-vs-implemented-functionality) for details.

---

## 4. Architecture

### 4.1 Data Flow Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           LightRAG-GQA System                               │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────┐     ┌──────────────────────────────────────────┐ │
│   │   GQA Scene Graphs  │     │              Knowledge Graph              │ │
│   │   (JSON files)      │────►│   ┌────────────────────────────────────┐ │ │
│   │   train/val splits  │     │   │       GLOBAL LEVEL                 │ │ │
│   └─────────────────────┘     │   │  ┌──────────┐    ┌──────────┐     │ │ │
│             │                 │   │  │Concept:  │    │Attr:     │     │ │ │
│             ▼                 │   │  │  dog     │    │  red     │     │ │ │
│   ┌─────────────────────┐     │   │  └────┬─────┘    └────┬─────┘     │ │ │
│   │      Builder        │     │   │       │instance_of    │has_attr   │ │ │
│   │  (normalize, link)  │────►│   │       ▼               ▼           │ │ │
│   │                     │     │   │  ┌────────────────────────────┐   │ │ │
│   └─────────────────────┘     │   │  │    INSTANCE LEVEL          │   │ │ │
│                               │   │  │  2386621:0 ──wearing──►    │   │ │ │
│                               │   │  │              2386621:1     │   │ │ │
│                               │   │  └────────────────────────────┘   │ │ │
│                               │   └────────────────────────────────────┘ │ │
│                               └──────────────────────────────────────────┘ │
│                                               │                             │
│                                               ▼                             │
│   ┌─────────────────────┐     ┌──────────────────────────────────────────┐ │
│   │   Natural Language  │     │           Reasoning Engines               │ │
│   │      Query          │────►│  ┌─────────────┐  ┌─────────────────────┐│ │
│   │  "Find red dogs"    │     │  │ NL Parser   │  │ Basic Engine        ││ │
│   └─────────────────────┘     │  │ (heuristic) │─►│ (5 query types)     ││ │
│                               │  └─────────────┘  └─────────────────────┘│ │
│   ┌─────────────────────┐     │                   ┌─────────────────────┐│ │
│   │   Structured Query  │────►│                   │ Advanced Engine     ││ │
│   │   (CQR JSON)        │     │                   │ (4 query types)     ││ │
│   └─────────────────────┘     │                   └─────────────────────┘│ │
│                               └──────────────────────────────────────────┘ │
│                                               │                             │
│                                               ▼                             │
│                               ┌──────────────────────────────────────────┐ │
│                               │              Outputs                      │ │
│                               │  • ranked_set: List[Instance]            │ │
│                               │  • scalar: {probability, count}          │ │
│                               │  • path: List[NodeSequence]              │ │
│                               │  • subgraph: {nodes, edges}              │ │
│                               │  + ReasoningTrace for debugging          │ │
│                               └──────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Module Structure

| Module | Path | Responsibility |
|--------|------|----------------|
| **Builder** | [src/lightrag_gqa/basic_queries/builder.py](src/lightrag_gqa/basic_queries/builder.py) | Construct knowledge graph from scene graphs |
| **Basic Engine** | [src/lightrag_gqa/basic_queries/reasoning_engine.py](src/lightrag_gqa/basic_queries/reasoning_engine.py) | Execute 5 basic query types |
| **Advanced Engine** | [src/lightrag_gqa/advanced_queries/reasoning_engine.py](src/lightrag_gqa/advanced_queries/reasoning_engine.py) | Execute 4 advanced query types |
| **NL Parser** | [src/lightrag_gqa/basic_queries/nl_parser.py](src/lightrag_gqa/basic_queries/nl_parser.py) | Rule-based NL→CQR conversion |
| **Query Interface** | [src/lightrag_gqa/basic_queries/query_interface.py](src/lightrag_gqa/basic_queries/query_interface.py) | Unified query dispatch |
| **Evaluation** | [src/lightrag_gqa/evaluation/](src/lightrag_gqa/evaluation/) | Suite generation and metrics |
| **CLI** | [src/lightrag_gqa/cli/](src/lightrag_gqa/cli/) | Command-line entrypoints |

---

## 5. Evaluation

### 5.1 Evaluation Layer Design

| Layer | Description | Status | Ground Truth Source |
|-------|-------------|--------|---------------------|
| **Layer A** | Engine-only evaluation using gold CQR | ✅ Implemented | Derived from scene graphs |
| **Layer B** | Parser evaluation (NL → CQR) | ❌ Not Implemented | Would require human annotation |
| **End-to-End** | Full NL → Engine → Results | ⚠️ Partial | Uses heuristic parser (unvalidated) |

### 5.2 Metrics by Output Type

#### Type A: Ranked Set Queries

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **P@k** | `|correct ∩ top-k| / k` | Precision of top-k results |
| **R@k** | `|correct ∩ top-k| / |correct|` | Recall within top-k |
| **F1@k** | `2 × P@k × R@k / (P@k + R@k)` | Harmonic mean |
| **MRR** | `mean(1/rank_first_correct)` | Mean reciprocal rank |
| **NDCG@k** | Normalized DCG | Ranking quality (position-weighted) |

**Ground Truth**: For entity search, ground truth = all `(image_id, object_id)` pairs where `object.name == concept` AND `∀attr ∈ attributes: attr ∈ object.attributes`

**Note**: Since output is **unranked**, NDCG/MRR may not be meaningful. P@k, R@k, F1@k are appropriate.

#### Type B: Scalar Queries

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | `mean(|predicted - expected|)` | Average absolute error |
| **RMSE** | `sqrt(mean((predicted - expected)²))` | Root mean squared error |

**Ground Truth**: For statistical knowledge, `P(B|A,rel) = |{a ∈ A : ∃b ∈ B, (a,b) ∈ E(rel)}| / |A|`

#### Type C: Path Queries

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Validity Rate** | `|valid_paths| / |returned_paths|` | Fraction of paths that exist in graph |
| **Hop Accuracy** | `|paths_with_correct_hops| / |returned_paths|` | Hop count correctness |
| **Completeness** | `|found_paths| / |all_valid_paths|` | Coverage of BFS solution space |

**Ground Truth**: All valid BFS paths from `instances_of(source)` to `instances_of(target)` with length ≤ `max_hops`, optionally filtered by `via_relation`.

#### Type D: Subgraph Queries

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Node Recall** | `|returned_nodes ∩ expected_nodes| / |expected_nodes|` | Node coverage |
| **Edge Recall** | `|returned_edges ∩ expected_edges| / |expected_edges|` | Edge coverage |
| **Compactness** | `|returned_nodes| / |full_graph_nodes|` | Subgraph size (smaller is better) |

### 5.3 Query Types: Evaluation Status

| # | Query Type | Layer A | Layer B | Notes |
|---|------------|---------|---------|-------|
| 1 | Entity Search | ✅ Evaluated | ❌ No labels | P@k, R@k, F1 metrics |
| 2 | Statistical Knowledge | ✅ Evaluated | ❌ No labels | MAE, RMSE metrics |
| 3 | Similarity Search | ✅ Evaluated | ❌ No labels | NDCG@k (but output is unranked) |
| 4 | Relational Path | ✅ Evaluated | ❌ No labels | Validity rate, hop accuracy |
| 5 | Negative Constraints | ✅ Evaluated | ❌ No labels | Set F1 |
| 6 | Chain Reasoning | ❌ Not in v0.1 | ❌ Not in v0.1 | Would use path metrics |
| 7 | Pattern Matching | ❌ Not in v0.1 | ❌ Not in v0.1 | Would use subgraph metrics |
| 8 | Scene Comparison | ❌ Not in v0.1 | ❌ Not in v0.1 | Would use subgraph metrics |
| 9 | Counterfactual | ❌ Not in v0.1 | ❌ Not in v0.1 | Requires human judgment |

### 5.4 Baselines

| Baseline | Status | Description |
|----------|--------|-------------|
| **Heuristic Parser** | ✅ Implemented | Rule-based regex matching (current system) |
| **LLM Parser Stub** | ⚠️ Stub only | Placeholder for LLM-based parsing (not functional) |
| **Random Baseline** | ❌ Not implemented | Would sample random results for comparison |
| **Oracle Baseline** | ✅ Implicit | Gold CQR represents perfect parsing |

### 5.5 Reported Results (10K Scale, Layer A)

From [EVALUATION_RESULTS.md](evaluation_v0_1/EVALUATION_RESULTS.md):

| Suite | Query Count | Success Rate | Key Metric |
|-------|-------------|--------------|------------|
| Entity+Attr | 210 | 100% | P@50: 0.00* |
| Negative Constraints | 500 | 100% | P@50: 0.00* |
| Statistical | 200 | 100% | RMSE: TBD |
| Path | 200 | 100% | Validity: 71% |
| Subgraph | 200 | 100% | Node Recall: TBD |

*Note: Low precision indicates mismatch between engine output format and evaluation expectations. See [TROUBLESHOOTING.md](evaluation_v0_1/TROUBLESHOOTING.md).

---

## 6. Installation & Usage

### 6.1 Prerequisites

- Python ≥ 3.8
- RAM: ~3 GB (10K scale), ~8 GB (full scale)
- NetworkX ≥ 3.0

### 6.2 Installation

```bash
git clone https://github.com/vinhhna/hybrid_multimodal_retrieval.git
cd hybrid_multimodal_retrieval
pip install -e .
```

### 6.3 Dataset Setup

Download GQA scene graphs from [Stanford Vision Lab](https://cs.stanford.edu/people/dorarad/gqa/):

```bash
mkdir -p sceneGraphs
# Place train_sceneGraphs.json and val_sceneGraphs.json in sceneGraphs/
```

### 6.4 Build Knowledge Graph

```bash
lightrag-gqa-build --scale 1k    # ~30s, development
lightrag-gqa-build --scale 10k   # ~5 min, recommended
lightrag-gqa-build --scale full  # ~40 min, production
```

### 6.5 Programmatic API

```python
from lightrag_gqa.basic_queries import GQA_Reasoning_Engine
from lightrag_gqa.advanced_queries import AdvancedReasoningEngine

# Basic queries
engine = GQA_Reasoning_Engine(scale='10k')
result = engine.entity_search(concept='dog', attributes=['brown'], limit=10)

# Advanced queries
adv_engine = AdvancedReasoningEngine(engine.graph)
result = adv_engine.chain_reasoning(
    start_concept='person',
    chain=[{'relation': 'wearing', 'concept': 'shirt', 'attribute': 'red'}],
    limit=5
)
```

### 6.6 CLI Commands

```bash
# Interactive mode
lightrag-gqa-query --scale 10k --interactive

# Single query
lightrag-gqa-query --scale 10k --query "Find red cars"

# Run evaluation
lightrag-gqa-eval --config evaluation_v0_1/configs/eval.yaml
```

---

## 7. Limitations & Failure Modes

### 7.1 Critical Limitations

| Issue | Impact | Mitigation |
|-------|--------|------------|
| **No Parser Evaluation** | Cannot measure NL understanding accuracy | Implement gold NL→CQR annotations |
| **Unranked Outputs** | NDCG/MRR metrics are not meaningful | Sort results by relevance score (not implemented) |
| **Heuristic Parsing** | Paraphrases fail silently | Use LLM-based parser (not implemented) |
| **Path Explosion** | BFS on dense graphs exhausts memory | Hardcoded 500-instance limit |

### 7.2 Algorithm Limitations

| Issue | Description | Reference |
|-------|-------------|-----------|
| **Cycle Detection** | Diamond patterns may exclude valid paths | [limitations_and_failure_modes.md#7.1](docs/limitations_and_failure_modes.md#issue-71-cycle-detection-in-path-queries) |
| **Jaccard Similarity** | No semantic similarity for synonyms | [limitations_and_failure_modes.md#7.2](docs/limitations_and_failure_modes.md#issue-72-similarity-metric-jaccard-not-normalized) |
| **Pattern Matching** | O(instances^pattern_nodes) complexity | [limitations_and_failure_modes.md#5.3](docs/limitations_and_failure_modes.md#issue-53-subgraph-pattern-matching-not-scalable) |

### 7.3 Dataset Limitations

- Fixed attribute vocabulary (no OOV handling)
- Relation names not canonicalized ("on", "on top of", "placed on" are distinct)
- No cross-image edges (image-scoped only)
- Inherits GQA annotation errors (~92% inter-annotator agreement)

Full details: [limitations_and_failure_modes.md](docs/limitations_and_failure_modes.md)

---

## 8. Evaluation (v0.1.1)

### 8.1 Quick Start

```bash
# Full evaluation on validation set
python -m evaluation_v0_1.run_evaluation --split val --verbose

# Evaluation on training set
python -m evaluation_v0_1.run_evaluation --split train --verbose

# Baseline comparison
python -m evaluation_v0_1.run_evaluation --method baseline_naive_scan --split val
python -m evaluation_v0_1.run_evaluation --method baseline_relation_walk --split val

# Advanced queries (toy suite)
python -m evaluation_v0_1.run_evaluation --step advanced --verbose

# All steps with seed for reproducibility
python -m evaluation_v0_1.run_evaluation --step all --split val --seed 1337 --verbose
```

### 8.2 Data Splits

| Split | Scene Graphs | Purpose |
|-------|--------------|---------|
| `train` | `train_sceneGraphs.json` | Development |
| `val` | `val_sceneGraphs.json` | Validation |

### 8.3 Metrics by Query Type

| Query Type | Output | Metrics |
|------------|--------|---------|
| Entity Search | Unranked Set | EM, P, R, F1, Jaccard |
| Negative Constraints | Unranked Set | EM, P, R, F1, Jaccard |
| Statistical | Scalar | MAE, RMSE, Spearman |
| Relational Path | Path | Validity, Hop Accuracy |
| Subgraph | Graph | Node/Edge P/R/F1 |

**Note**: Ranking metrics (NDCG, MRR, MAP) are NOT used for entity search because outputs are unranked sets.

### 8.4 Baselines

| Baseline | Description | Command |
|----------|-------------|---------|
| `baseline_naive_scan` | Direct scene graph scan | `--method baseline_naive_scan` |
| `baseline_relation_walk` | Within-image BFS | `--method baseline_relation_walk` |
| `baseline_parser_trivial` | Simple regex parser | `--method baseline_parser_trivial` |

### 8.5 Output Structure

```
artifacts/
└── {timestamp}_{split}_{method}/
    ├── summary.json
    ├── summary.md
    ├── engine_suite_*.json
    ├── engine_suite_*.csv
    ├── baseline_*.json
    ├── advanced_*.json
    └── advanced_*.md
```

### 8.6 Configuration

See [evaluation_v0_1/configs/eval.yaml](evaluation_v0_1/configs/eval.yaml) for full options.

### 8.7 Expected Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| Built graph | `experiments/sample_{scale}/gqa_lightrag.gpickle` | NetworkX DiGraph |
| Evaluation suites | `evaluation_v0_1/data/suites/*.json` | Generated test queries |
| Evaluation results | `evaluation_v0_1/results/*.csv` | Per-query results |
| Summary | `evaluation_v0_1/results/e2e_summary.md` | Aggregate metrics |
| Timestamped runs | `artifacts/{timestamp}_{split}_{method}/` | Full output per run |

### 8.8 Determinism

- Graph construction is deterministic (fixed scene graph order)
- Query execution is deterministic (BFS, set operations)
- Evaluation suite generation uses random sampling (use `--seed` for reproducibility)

---

## 9. Project Structure

```
hybrid_multimodal_retrieval/
├── README.md                    # This document
├── pyproject.toml               # Package configuration
├── requirements.txt             # Dependencies
│
├── docs/                        # Documentation
│   ├── definitions.md           # Formal schema definitions
│   ├── traceability.md          # Query-to-code mapping
│   ├── evaluation_protocol.md   # Evaluation methodology
│   ├── limitations_and_failure_modes.md
│   └── toy_graph.json           # Minimal example
│
├── src/lightrag_gqa/            # Main package
│   ├── basic_queries/           # 5 basic query types
│   │   ├── builder.py           # Graph construction
│   │   ├── reasoning_engine.py  # Query execution
│   │   ├── nl_parser.py         # NL parsing
│   │   └── query_interface.py   # Unified interface
│   ├── advanced_queries/        # 4 advanced query types
│   │   └── reasoning_engine.py
│   ├── evaluation/              # Evaluation framework
│   ├── cli/                     # Command-line tools
│   ├── datasets/
│   └── utils/
│
├── evaluation_v0_1/             # Standalone evaluation scripts
│   ├── run_evaluation.py
│   ├── configs/eval.yaml
│   ├── data/suites/
│   ├── results/
│   └── scripts/
│
├── sceneGraphs/                 # GQA data (gitignored)
├── experiments/                 # Built graphs (gitignored)
├── notebooks/                   # Demo notebooks
└── archive/                     # Legacy code
```

---

## 10. Change Log & Future Work

### Completed (v1.0)

- ✅ Two-level LightRAG graph construction
- ✅ 9 query types (5 basic + 4 advanced)
- ✅ Rule-based NL parser
- ✅ Layer A evaluation framework
- ✅ CLI tools (build, query, eval)
- ✅ Multi-scale support (1K, 10K, full)
- ✅ Reasoning traces

### Not Implemented

- ❌ Layer B parser evaluation (no gold annotations)
- ❌ LLM-based parsing
- ❌ Result ranking/relevance scoring
- ❌ Relation canonicalization
- ❌ Cross-image reasoning
- ❌ SPARQL interface
- ❌ Distributed execution

### v0.1.1 (Evaluation Upgrade)

- ✅ Data splits (train/val) with CLI support (`--split`)
- ✅ Set-based metrics (EM, P, R, F1, Jaccard) for unranked outputs
- ✅ Baseline methods (naive_scan, relation_walk, parser_trivial)
- ✅ Advanced query evaluation (types 6-9) with curated toy suite
- ✅ Timestamped output directories (`artifacts/`)
- ✅ Reproducibility seed (`--seed`)

### Future Work

1. **Annotate** 100+ NL questions with gold CQR for Layer B evaluation
2. **Implement** relation synonym mapping (e.g., "on" = "on top of")
3. **Add** relevance scoring for ranked output types
4. **Integrate** LLM-based query parser
5. **Scale** advanced query evaluation to val set (beyond toy suite)

---

## References

- **GQA Dataset**: Hudson, D. A., & Manning, C. D. (2019). GQA: A New Dataset for Real-World Visual Reasoning. CVPR. [Link](https://cs.stanford.edu/people/dorarad/gqa/)
- **NetworkX**: Hagberg, A., Swart, P., & Chult, D. S. (2008). Exploring network structure, dynamics, and function using NetworkX.
- **LightRAG Concept**: Inspired by retrieval-augmented generation architectures for knowledge-intensive tasks.

---

**Status**: Functional (v0.1.1 - evaluation upgraded, parser layer incomplete)  
**Python**: ≥ 3.8  
**Last Updated**: January 2025
