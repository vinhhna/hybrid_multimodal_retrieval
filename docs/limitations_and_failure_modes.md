# Limitations and Failure Modes

## Overview

This document catalogs known limitations, unimplemented features, and failure scenarios for the LightRAG-GQA system.

---

## 1. Claimed vs. Implemented Functionality

### Discrepancy: Query Types

**Claim** (from original README): "14 query types (9 basic + 5 advanced)"

**Reality** (from code inspection):
- **Basic queries**: 5 implemented (`entity_search`, `statistical_knowledge`, `similarity_search`, `relational_path`, `negative_constraints`)
- **Advanced queries**: 4 implemented (`chain_reasoning`, `pattern_matching`, `scene_comparison`, `counterfactual_reasoning`)
- **Total**: 9 query types

**Missing Types** (announced but not implemented):
- Basic 6: Comparative (partial; `compare_contexts()` and `compare_attribute_distribution()` exist but not full spec)
- Basic 7: Hierarchical (`get_hierarchical_entities()` exists)
- Basic 8: Anomaly Detection (`find_anomalies()` and `find_specific_anomaly()` exist)
- Basic 9: Visual-Attribute Constraint (`multi_constraint_search()` and `complex_constraint_search()` exist)
- Advanced 5: Centrality (`centrality_query()` exists in code)

**Verdict**: README overcounts. 9 types are genuinely implemented; the remaining 5 (claimed as "advanced") do exist in code but were not included in the main evaluation framework or demo scripts.

**Source**: [src/lightrag_gqa/basic_queries/reasoning_engine.py](../src/lightrag_gqa/basic_queries/reasoning_engine.py#L900-L1600) defines methods 6-9; [src/lightrag_gqa/advanced_queries/reasoning_engine.py](../src/lightrag_gqa/advanced_queries/reasoning_engine.py) contains chain, pattern, scene, counterfactual.

---

## 2. Natural Language Parsing Limitations

### Issue 2.1: Heuristic Pattern Matching
**Problem**: Query parsing uses rule-based regex patterns, not trained models.

**Consequence**: 
- Paraphrases may fail to parse (e.g., "Show objects that look similar to a tall tree" may not match expected pattern)
- No confidence threshold enforcement (high-confidence misclassifications possible)
- Ambiguous queries map to incorrect types (no multi-type hypothesis)

**Example Failure**:
```
Query: "How many red things are on a table?"
Expected: Entity search for red objects + constraint on "on table" relation
Actual: May parse as Statistical or misclassify as Similarity
```

**No Evaluation**: Layer B (parser evaluation) lacks gold annotations; parser accuracy unmeasured.

**File**: [src/lightrag_gqa/basic_queries/nl_parser.py](../src/lightrag_gqa/basic_queries/nl_parser.py)

---

## 3. Scene Graph Schema Limitations

### Issue 3.1: Fixed Attribute Vocabulary
**Problem**: Attributes are drawn from a fixed set in GQA dataset; out-of-vocabulary attributes are not handled.

**Consequence**:
- Queries with novel attributes (e.g., "metallic" if not in training data) return empty results
- No fallback to semantic similarity for attribute matching

**Partial Mitigation**: Normalization (lowercase) makes matching case-insensitive, but synonyms (e.g., "shiny" vs "glossy") are not recognized.

**File**: [src/lightrag_gqa/basic_queries/builder.py](../src/lightrag_gqa/basic_queries/builder.py#L1-50) loads attributes as-is from scene graphs.

### Issue 3.2: Relation Vocabulary Not Standardized
**Problem**: Relations in GQA vary by annotator (e.g., "on", "on top of", "placed on" may all exist).

**Consequence**:
- Relational path queries may miss paths due to relation name mismatch
- No canonicalization of relations

**Workaround**: Query parser attempts substring matching (`contains` check), but incomplete.

**File**: [src/lightrag_gqa/basic_queries/reasoning_engine.py](../src/lightrag_gqa/basic_queries/reasoning_engine.py#L269-L276)

---

## 4. Graph Construction Limitations

### Issue 4.1: Within-Image Semantics Only
**Problem**: Semantic relations only connect objects in the same image; cross-image relations are not captured.

**Consequence**:
- Queries asking about trends across images (e.g., "Do dogs appear more often in indoor scenes than outdoor?") require manual aggregation
- No global semantic hierarchy (e.g., "cat" → "animal" → "living_thing")

**Design Rationale**: Scene graphs are image-centric; cross-image reasoning would require additional structure.

**File**: [src/lightrag_gqa/basic_queries/builder.py](../src/lightrag_gqa/basic_queries/builder.py#L280-290)

### Issue 4.2: No Temporal Information
**Problem**: Scene graphs are static; no temporal or causal information.

**Consequence**:
- Cannot answer "what happens before/after this?" queries
- Causality must be inferred from spatial proximity (weak signal)

**Scope**: Out of scope for single-image datasets like GQA.

### Issue 4.3: No Scene-Level Metadata
**Problem**: Scene graphs lack scene descriptions, location context, or image source.

**Consequence**:
- Queries like "Find objects in outdoor scenes" cannot be answered directly
- Requires scene classification as preprocessing (not provided)

**File**: [sceneGraphs/README.txt](../sceneGraphs/readme.txt) documents available fields.

---

## 5. Query Execution Limitations

### Issue 5.1: No Ranking for Unranked Queries
**Problem**: Entity Search and similar queries return unranked sets; no relevance scoring.

**Consequence**:
- Results are returned in arbitrary order (insertion order in graph)
- "Top 5" results may not be most relevant

**Resolution (v0.1.1)**:
- ✅ Ranking metrics (NDCG, MRR, MAP) removed from entity search evaluation
- ✅ Set-based metrics (EM, P, R, F1, Jaccard) now used instead
- See [evaluation_protocol.md](evaluation_protocol.md) for metric justification

**Remaining Gap**: To enable ranking, implement a scoring function (future work).

**File**: [src/lightrag_gqa/basic_queries/reasoning_engine.py](../src/lightrag_gqa/basic_queries/reasoning_engine.py#L345-360)

### Issue 5.2: Path Length Explosion
**Problem**: BFS path discovery can generate exponentially many paths in dense graphs.

**Consequence**:
- For highly connected graphs (many relations), `relational_path()` may run out of memory
- `max_hops` parameter is critical; no adaptive depth limiting

**Example**: In a graph with 1M edges, BFS to depth 5 could explore 10M+ paths.

**Mitigation**: Limit initial start set size (hardcoded to 500 instances); may miss valid paths.

**File**: [src/lightrag_gqa/advanced_queries/reasoning_engine.py](../src/lightrag_gqa/advanced_queries/reasoning_engine.py#L220-225)

### Issue 5.3: Subgraph Pattern Matching Not Scalable
**Problem**: Pattern matching uses brute-force instance enumeration; no indexing.

**Consequence**:
- `pattern_matching()` complexity: O(|instances|^|pattern_nodes|)
- For large graphs, patterns with >3 nodes become intractable

**No Optimization**: No SMT solvers or constraint propagation techniques used.

**File**: [src/lightrag_gqa/advanced_queries/reasoning_engine.py](../src/lightrag_gqa/advanced_queries/reasoning_engine.py#L386-450)

---

## 6. Evaluation Framework Limitations

### Issue 6.1: No Ground Truth for NL Parsing
**Problem**: Evaluation Layer B (parser accuracy) requires annotated NL-to-CQR pairs, which are absent.

**Consequence**:
- Parser accuracy is unmeasured; only heuristic baseline tested
- No benchmark for query understanding quality

**Required to Implement**: 
- Human annotation of 100+ NL questions with correct CQR
- Agreement study (Cohen's kappa) for consistency

**File**: [evaluation_v0_1/README.md](../evaluation_v0_1/README.md) documents the gap.

### Issue 6.2: Limited Evaluation Scope
**Problem**: Evaluation v0.1 covers only 5 query types; advanced types (6-9) not evaluated.

**Consequence**:
- Chain reasoning, pattern matching, scene comparison have no quantitative metrics
- Success rates for these queries unknown

**File**: [evaluation_v0_1/EVALUATION_RESULTS.md](../evaluation_v0_1/EVALUATION_RESULTS.md) shows only 5 types.

### Issue 6.3: No Cross-Dataset Validation
**Problem**: Evaluation uses only training sceneGraphs; validation/test splits not used.

**Consequence**:
- Generalization to unseen images unknown
- Risk of overfitting (if graph construction was tuned to training set)

**Available But Unused**: `val_sceneGraphs.json` exists but is not in evaluation config.

**File**: [evaluation_v0_1/configs/eval.yaml](../evaluation_v0_1/configs/eval.yaml)

---

## 7. Algorithm Correctness Limitations

### Issue 7.1: Cycle Detection in Path Queries
**Problem**: Path queries prevent revisiting nodes to avoid infinite loops, but this is a heuristic.

**Consequence**:
- Some valid acyclic paths with "diamond" patterns may be excluded:
  - Example: A → B → D, A → C → D (both valid 2-hop paths from A to D, but one path shares D with the other)
  - Current implementation: Path [A, B, D] would exclude path [A, C, D] if D is already in the first path explored

**Formal Issue**: Paths are generated per-branch, not globally; first-found path may exclude alternatives.

**File**: [src/lightrag_gqa/advanced_queries/reasoning_engine.py](../src/lightrag_gqa/advanced_queries/reasoning_engine.py#L220-250)

### Issue 7.2: Similarity Metric (Jaccard) Not Normalized
**Problem**: Jaccard similarity treats attributes as unordered sets; no weighting by importance.

**Consequence**:
- "Large" attribute weighs same as rare "turquoise" attribute
- No semantic similarity (synonyms like "tall"/"long" treated as different)

**File**: [src/lightrag_gqa/basic_queries/reasoning_engine.py](../src/lightrag_gqa/basic_queries/reasoning_engine.py#L550-600)

### Issue 7.3: Comparative Queries Assume Image Boundaries
**Problem**: `compare_contexts()` assumes contexts are images with isolated object sets.

**Consequence**:
- If same object appears in multiple contexts, counting is incorrect
- Example: "Compare chairs in kitchen vs dining room" fails if same chair image is labeled both

**Mitigation**: None; assumes scenes are mutually exclusive (may be violated by GQA).

**File**: [src/lightrag_gqa/basic_queries/reasoning_engine.py](../src/lightrag_gqa/basic_queries/reasoning_engine.py#L922-1000)

---

## 8. Implementation Quality Limitations

### Issue 8.1: No Error Handling for Corrupt Graphs
**Problem**: Graph loading assumes pickle file is valid; no validation.

**Consequence**:
- If graph file is truncated or corrupted, error is cryptic
- No checksum or versioning

**File**: [src/lightrag_gqa/basic_queries/reasoning_engine.py](../src/lightrag_gqa/basic_queries/reasoning_engine.py#L154-170)

### Issue 8.2: No Type Hints in Core Engine
**Problem**: Most query methods use `Optional[str]`, `Dict[str, Any]` without precise typing.

**Consequence**:
- Static type checking (mypy) provides no enforcement
- Easier for bugs to slip through (e.g., passing list instead of string)

**File**: [src/lightrag_gqa/basic_queries/reasoning_engine.py](../src/lightrag_gqa/basic_queries/reasoning_engine.py) (throughout)

### Issue 8.3: Hardcoded Limits
**Problem**: Various hardcoded limits are scattered throughout code (e.g., max 500 start instances for chain reasoning).

**Consequence**:
- Not configurable; cannot adjust for different graphs
- May cause unexpected result truncation

**Hardcoded Values**:
- 500 instance limit: [advanced_queries.py](../src/lightrag_gqa/advanced_queries/reasoning_engine.py#L220)
- 1000 node limit: (implicit in some aggregations)

---

## 9. Dataset-Specific Limitations

### Issue 9.1: GQA Scene Graphs Have Annotation Errors
**Problem**: GQA scene graphs are automatically extracted; not all are accurate.

**Consequence**:
- Entity search may return false positives/negatives due to annotation errors
- No way to detect or correct errors in the knowledge graph

**Citation**: [GQA Paper](https://cs.stanford.edu/people/dorarad/gqa/) reports inter-annotator agreement ≈ 92% for some attributes.

### Issue 9.2: Missing Objects in Scene Graphs
**Problem**: Some objects in images may not be annotated in scene graphs.

**Consequence**:
- Queries asking "find all Xs" may miss instances not in scene graph
- Recall metric is actually incomplete_recall / true_recall (unmeasurable without alternative annotation)

**File**: Scene graphs are provided by GQA dataset (not this project).

### Issue 9.3: Attribute Ambiguity
**Problem**: Attributes like "transparent" or "metallic" are subjective; annotators may disagree.

**Consequence**:
- Query results depend on annotator interpretation
- No fuzzy matching for similar attributes

---

## 10. Documented Unimplemented Features

### From README (claimed but missing):

1. **Custom graph builder API** (programmatic graph creation)
   - **Status**: Only command-line builder available
   - **Would enable**: Real-time graph updates, streaming insertion

2. **Batch query execution**
   - **Status**: Only single-query interface available
   - **Would enable**: Efficient evaluation on large query sets

3. **SPARQL or other standard query language support**
   - **Status**: Not implemented
   - **Would enable**: Interoperability with other RDF/graph tools

4. **Caching for frequently-executed queries**
   - **Status**: No cache layer
   - **Would enable**: Faster repeated queries

5. **Distributed/parallel execution**
   - **Status**: Single-threaded only
   - **Would enable**: Scaling to larger graphs

---

## 11. Suggested Improvements

### High Priority (Correctness)
1. Add ground truth annotations for NL parser evaluation
2. Implement relation canonicalization (e.g., "on" = "on top of")
3. Add query result validation and error reporting
4. Implement attribute/concept fuzzy matching

### Medium Priority (Functionality)
1. Implement missing advanced query evaluation framework
2. Add scene-level metadata and cross-image reasoning
3. Implement query ranking/relevance scoring
4. Add temporal/causal reasoning

### Low Priority (Performance/UX)
1. Add caching layer for frequent queries
2. Implement distributed execution
3. Add web API wrapper
4. Implement SPARQL interface

---

## 12. Reproducibility Concerns

### What May Differ Across Runs
- **Graph build order**: If using different scene graph ordering, instance node IDs may differ (but semantics unchanged)
- **Randomness in evaluation**: ✅ Now seeded via `random_seed` in config

### What Should Be Deterministic
- **Query results** (given same graph and same query params): Should be identical
- **Evaluation metrics**: Should be identical given same test set and seed

### Verification
To verify reproducibility:
```bash
# Run evaluation twice with same seed
python -m evaluation_v0_1.run_evaluation --seed 1337 --split val
python -m evaluation_v0_1.run_evaluation --seed 1337 --split val
# Results should be identical
```

---

## 13. Evaluation-Specific Limitations (Updated v0.1.1)

### Resolved in v0.1.1

| Issue | Previous State | Current State |
|-------|----------------|---------------|
| Invalid ranking metrics | NDCG/MRR used on unranked sets | ✅ Set metrics (EM, P, R, F1) |
| No data splits | Train only | ✅ Train/val supported |
| No baselines | No comparison | ✅ 3 baselines implemented |
| No advanced eval | Types 6-9 not evaluated | ✅ Curated toy suites |

### Remaining Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| No parser ground truth | Layer B unevaluated | Trivial baseline only |
| Small advanced suites | Not statistically significant | Document limitation |
| Trace quality placeholder | No reasoning quality measure | Mark as future work |
| Cross-split KG | Val uses train-built KG | Expected for retrieval |

---

## Summary Table

| Category | # Issues | Severity | Impact |
|----------|----------|----------|--------|
| Claimed vs Actual | 1 | High | Misleading README |
| NL Parsing | 1 | High | Unparseable queries fail silently |
| Schema Limitations | 2 | Medium | Missing domain knowledge |
| Graph Construction | 3 | Medium | Limited expressiveness |
| Query Execution | 3 | Medium | Scalability/correctness concerns |
| Evaluation | 1 | Medium | ~~3~~ Reduced: metrics fixed |
| Algorithm Correctness | 3 | Low | Rare edge cases |
| Implementation Quality | 3 | Low | Brittle code |
| Dataset Issues | 3 | Low | Inherited from GQA |
| Unimplemented Features | 5 | Low | Out of scope |
| **Total** | **25** | | ~~27~~ Reduced |

---

## Changelog

### v0.1.1 (February 2026)
- ✅ Fixed invalid ranking metrics issue (Issue 5.1)
- ✅ Added data split support (train/val)
- ✅ Added baseline methods for comparison
- ✅ Added advanced query evaluation with curated suites
- ✅ Added reproducibility controls (seeded random)

---

## Conclusion

The system is functionally correct for its intended use case (entity/path queries on GQA scene graphs) but has significant limitations in parser robustness, evaluation completeness, and scalability for advanced query types. For academic/research use, the limitations should be documented in any citations or derivative work.

