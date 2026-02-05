# Evaluation Protocol

## Overview

The evaluation framework is located in [evaluation_v0_1/](../evaluation_v0_1/) and provides a standardized methodology for assessing query engine performance.

**Design Goal**: Separate engine correctness (Layer A) from parser accuracy (Layer B).

**Version**: 0.1.1 (Updated February 2026)

---

## Key Design Decisions

### 1. Data Splits (Task A)

The evaluation supports multiple data splits for academically rigorous assessment:

| Split | Scene Graphs | Purpose | Status |
|-------|--------------|---------|--------|
| **train** | `train_sceneGraphs.json` | Development, tuning | ✅ Implemented |
| **val** | `val_sceneGraphs.json` | Model selection | ✅ Implemented |
| **test** | N/A (GQA test is blind) | Final reporting | ❌ Not available |

**Configuration** ([eval.yaml](../evaluation_v0_1/configs/eval.yaml)):
```yaml
data_splits:
  active_split: "val"  # Options: "train", "val", "train+val"
```

### 2. Metrics Validity (Task B)

**Critical Insight**: Not all query types produce ranked outputs. Using ranking metrics (NDCG, MRR, MAP) on unranked sets is academically invalid.

#### Output Semantics by Query Type

| Query Type | Output | Semantics | Valid Metrics |
|------------|--------|-----------|---------------|
| Entity Search | Set | **Unranked** | EM, P, R, F1, Jaccard |
| Negative Constraints | Set | **Unranked** | EM, P, R, F1, Jaccard |
| Statistical | Scalar | Value | MAE, RMSE, Spearman |
| Relational Path | Path | Structure | Validity, Hop Accuracy |
| Subgraph | Graph | Structure | Node/Edge P/R/F1 |

**Why Entity Search is Unranked**: The engine returns images via set intersection without explicit scoring.

### 3. Baselines (Task C)

| Baseline | Description | Purpose |
|----------|-------------|---------|
| `baseline_naive_scan` | Direct scene graph scan | Shows KG value |
| `baseline_relation_walk` | Within-image BFS | Shows cross-image value |
| `baseline_parser_trivial` | Simple regex parser | Parser lower bound |

### 4. Advanced Queries (Task D)

Query types 6-9 use curated toy suites in [advanced_toy/](../evaluation_v0_1/data/suites/advanced_toy/).

**Trace Quality Rubric**: Placeholder for future work.

---

## Layer A: Engine-Only Evaluation

### Objective
Evaluate the reasoning engine in isolation using gold-standard structured queries (CQR = Canonical Query Representation).

### Methodology

1. **Gold Query Generation** (from sceneGraphs only):
   - Parse sceneGraph JSON files
   - Extract ground truth object/relation/attribute facts
   - Generate canonical queries for each fact type

2. **Query Execution**:
   - Execute each gold query on built knowledge graph
   - Capture output results

3. **Expected Output Derivation**:
   - For each query, ground truth is computed directly from scene graph
   - Example: Entity query `{"concept": "dog", "attributes": ["red"]}` → 
     Expected output = all image_ids where object.name == "dog" AND "red" in object.attributes

4. **Metric Computation** (SET-BASED for unranked outputs):
   - **Exact Match (EM)**: 1 if predicted set = gold set
   - **Precision**: |intersection| / |predicted|
   - **Recall**: |intersection| / |gold|
   - **F1 Score**: Harmonic mean of precision and recall
   - **Jaccard**: |intersection| / |union|

### Query Output Types and Metrics

#### Type A: Unranked Set (Entity Search, Negative Constraints)

**Output**: Set of image IDs (order NOT meaningful)

**Valid Metrics**:
- **EM** (Exact Match): Perfect set equality
- **Precision**: Fraction of predictions that are correct
- **Recall**: Fraction of gold items retrieved
- **F1**: Harmonic mean of P and R
- **Jaccard**: Set overlap (IoU)
- **P@k, R@k, F1@k**: Performance on first k results (user-facing cutoff)

**INVALID Metrics** (do not use):
- ~~NDCG@k~~ - Requires meaningful ranking
- ~~MRR~~ - Requires meaningful ranking
- ~~MAP~~ - Requires meaningful ranking

**Ground Truth**:
- For entity search: all image IDs where object matches concept+attributes

#### Type B: Scalar (e.g., Statistical Knowledge)

**Output**: Single numeric value (count, probability, ratio)

**Metrics**:
- **MAE** (Mean Absolute Error): Average |predicted - expected|
- **RMSE** (Root Mean Squared Error): $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$
- **Spearman Correlation**: Rank correlation of predicted vs expected values
- **Accuracy** (binary): Predicted value within ±threshold of expected

**Ground Truth**:
- For co-occurrence probability: count_coexisting / count_source
- For count queries: exact integer count from scene graphs

#### Type C: Path (e.g., Relational Path Discovery)

**Output**: List of graph paths

**Metrics**:
- **Path Validity Rate**: % of returned paths that exist in graph
  - Check: all edges exist, no missing intermediate nodes
- **Hop Accuracy**: % of paths with correct hop count
- **Completeness**: % of valid paths found (among all possible paths ≤ max_hops)

**Ground Truth**:
- All valid BFS paths from source_concept to target_concept
- Filtered by via_relation if specified
- Limited by max_hops

#### Type D: Subgraph (e.g., Scene Comparison Evidence)

**Output**: Subgraph nodes and edges

**Metrics**:
- **Node Recall**: % of expected nodes included in returned subgraph
- **Edge Recall**: % of expected edges included
- **Precision**: % of returned nodes/edges that are in ground truth
- **Compactness**: nodes_in_result / nodes_in_full_graph (prefer smaller subgraphs)

**Ground Truth**:
- For scene comparison: all objects and relations in the image

---

## Layer B: Parser Evaluation (Not Fully Implemented)

### Objective
Evaluate natural language parsing accuracy.

### Methodology

#### Sub-Layer B1: Parser-Only
- **Input**: Natural language question
- **Output**: Extracted query parameters (parsed CQR)
- **Comparison**: Against human-annotated CQR labels
- **Metric**: Parse accuracy (% of parameters extracted correctly)

**Status**: Heuristic parser implemented ([nl_parser.py](../src/lightrag_gqa/basic_queries/nl_parser.py)), but **no gold annotation set** for evaluation.

#### Sub-Layer B2: End-to-End
- **Input**: Natural language question
- **Output**: Final engine results (via parsed query)
- **Comparison**: Against gold query results (from Layer A)
- **Metric**: End-to-end F1, NDCG@10, etc.

**Status**: Framework exists in [evaluation_v0_1/](../evaluation_v0_1/), but **parser ground truth missing**.

---

## Supported Query Types in Evaluation

| Query Type | Status | Layer A (Gold) | Layer B (Parser) | Notes |
|------------|--------|----------------|-----------------|-------|
| 1. Entity Search | ✅ Implemented | ✅ Supported | ❌ No labels | Evaluate with perfect parse |
| 2. Statistical Knowledge | ✅ Implemented | ✅ Supported | ❌ No labels | Scalar metric (RMSE) |
| 3. Similarity Search | ✅ Implemented | ✅ Supported | ❌ No labels | Ranked metric (NDCG@10) |
| 4. Relational Path | ✅ Implemented | ✅ Supported | ❌ No labels | Path validity metric |
| 5. Negative Constraints | ✅ Implemented | ✅ Supported | ❌ No labels | Set F1 score |
| 6. Chain Reasoning | ⚠️ Partial | ❌ Not in v0.1 | ❌ Not in v0.1 | Would use ranked metric |
| 7. Pattern Matching | ⚠️ Partial | ❌ Not in v0.1 | ❌ Not in v0.1 | Would use subgraph metric |
| 8. Scene Comparison | ⚠️ Partial | ❌ Not in v0.1 | ❌ Not in v0.1 | Would use subgraph metric |
| 9. Counterfactual | ❌ Not evaluated | ❌ Not in v0.1 | ❌ Not in v0.1 | Requires human judgment |

---

## Evaluation Suite Structure

### Suite Generation ([evaluation_v0_1/scripts/generate_suites.py](../evaluation_v0_1/scripts/generate_suites.py))

**Process**:
1. Load scene graphs from `sceneGraphs/train_sceneGraphs.json`
2. Sample images (1K, 10K, or full)
3. For each query type, generate test queries:
   - Extract concepts, attributes, and relations from scene graphs
   - Create parametrized queries
   - Compute ground truth answers

**Suites Generated**:
- Entity search: concept + attributes combinations
- Statistical: concept pairs
- Similarity: concept with attributes
- Relational path: concept pairs (BFS up to max_hops)
- Negative constraints: concept pairs

**Output Format**: JSON files in [evaluation_v0_1/data/suites/](../evaluation_v0_1/data/suites/)

---

## Running Evaluation

### Prerequisites
```bash
# Build graph at desired scale
python -m lightrag_gqa.cli.build_graph --scale 10k --input sceneGraphs/train_sceneGraphs.json

# Place scene graphs
cp sceneGraphs/*.json . || download from GQA dataset
```

### Command
```bash
cd evaluation_v0_1
python run_evaluation.py --config configs/eval.yaml --scale 10k --verbose
```

### Configuration ([evaluation_v0_1/configs/eval.yaml](../evaluation_v0_1/configs/eval.yaml))
```yaml
scale: 10k                    # Graph scale
graph_path: experiments/sample_10k/gqa_lightrag.gpickle
scene_graphs_path: sceneGraphs/train_sceneGraphs.json

queries:
  entity_search:
    enabled: true
    num_queries: 100
  statistical_knowledge:
    enabled: true
    num_queries: 50
  # ... other query types

metrics:
  entity_search: [precision, recall, f1]
  statistical_knowledge: [mae, rmse]
  relational_path: [path_validity, completeness]

output_dir: results/
```

### Output
```
evaluation_v0_1/results/
├── e2e_summary.md          # Overall results
├── e2e_summary.json        # Machine-readable summary
├── engine_suite_*.csv      # Per-query results
├── engine_suite_*.json     # Detailed results
└── [query_type]_stats.json # Type-specific statistics
```

---

## Baseline and Expected Performance

### Current Results (10K Scale)

From [evaluation_v0_1/EVALUATION_RESULTS.md](../evaluation_v0_1/EVALUATION_RESULTS.md):

**Entity Search**:
- Precision@10: 0.95
- Recall@10: 0.88
- F1: 0.91

**Statistical Knowledge**:
- RMSE: 0.012
- Spearman ρ: 0.94

**Relational Path**:
- Path validity: 98%
- Completeness: 87%

**Overall**:
- MRR: 0.952
- NDCG@10: 0.890
- MAP: 0.904

### Baselines
- **Heuristic Baseline** (rule-based keyword matching): Not systematically evaluated
- **LLM Stub** (placeholder for LLM parser): Implemented in evaluation framework
- **Oracle** (perfect parse + perfect engine): Expected ≈ Layer A results

### Known Failure Modes
See [docs/limitations_and_failure_modes.md](limitations_and_failure_modes.md#evaluation-specific-limitations).

---

## Extending Evaluation

### Adding a New Query Type

1. **Implement in Engine**: Add method to `GQA_Reasoning_Engine` or `AdvancedReasoningEngine`

2. **Define Ground Truth**:
   ```python
   # evaluation_v0_1/scripts/generate_suites.py
   def generate_TYPE_queries(scene_graphs, num_queries):
       """Generate test cases and expected outputs."""
       queries = []
       for ... :
           query = {
               "type": "TYPE",
               "params": {...},
               "expected_output": compute_ground_truth(...)
           }
           queries.append(query)
       return queries
   ```

3. **Implement Metrics**:
   ```python
   # evaluation_v0_1/scripts/evaluation_runner.py
   def evaluate_TYPE(results, expected):
       """Compute metrics for TYPE queries."""
       return {
           "metric1": compute_metric1(results, expected),
           "metric2": compute_metric2(results, expected)
       }
   ```

4. **Add to Config**:
   ```yaml
   # evaluation_v0_1/configs/eval.yaml
   queries:
     TYPE:
       enabled: true
       num_queries: 50
   ```

5. **Run Evaluation**:
   ```bash
   python run_evaluation.py --config configs/eval.yaml
   ```

---

## Limitations of Current Evaluation

1. **No Parser Ground Truth**: Layer B cannot be evaluated without annotated NL-to-CQR pairs
2. **Scene Graph Bias**: Evaluation uses only training sceneGraphs; validation/test sets available but not used
3. **No Human Judgment**: Complex queries (counterfactual, anomaly) lack objective correctness criteria
4. **Missing Query Types**: Advanced types (6-9) not included in evaluation framework
5. **No Efficiency Metrics**: Runtime/memory not tracked; "sub-200ms" claim not verified
6. **No Baseline Comparison**: No comparison to other retrieval methods (e.g., SPARQL, semantic search, neural methods)

---

## Reproducibility Checklist

- [ ] GQA sceneGraphs downloaded and placed in `sceneGraphs/`
- [ ] Python ≥ 3.8, networkx ≥ 3.0 installed
- [ ] Knowledge graph built: `lightrag-gqa-build --scale 10k`
- [ ] Evaluation config reviewed and adjusted
- [ ] Results directory exists: `mkdir -p evaluation_v0_1/results`
- [ ] Run evaluation: `python evaluation_v0_1/run_evaluation.py --config evaluation_v0_1/configs/eval.yaml`
- [ ] Check output files in `evaluation_v0_1/results/`
- [ ] Compare against expected performance (see above)

