# Advanced Graph Reasoning Queries

This module extends the GQA LightRAG system with **advanced reasoning capabilities** that require sophisticated graph traversal and multi-hop inference. These queries highlight the strength of knowledge graph-based retrieval by performing complex reasoning that would be difficult or impossible with simple keyword search.

## Overview

While the base system supports 9 query types (entity search, statistics, similarity, paths, etc.), these **advanced queries** require:

- **Multi-hop reasoning**: Following chains of relationships across the graph
- **Subgraph pattern matching**: Finding images that match complex relational patterns
- **Comparative analysis**: Comparing structural properties across different subgraphs
- **Inference chains**: Drawing conclusions from connected facts

## Advanced Query Types

### 1. Multi-Hop Chain Reasoning (`chain_reasoning`)

**Purpose**: Answer questions that require traversing multiple relationship hops with intermediate constraints.

**Example Questions**:
- "Find images where a person is wearing something red that is near a vehicle"
- "Which scenes have an animal that is on furniture which is inside a room?"

**Graph Operations**:
1. Start from concept A instances
2. Traverse relationship R1 to find intermediate nodes
3. Filter by attribute/concept constraints
4. Continue traversing R2 to reach concept B
5. Verify all constraints are satisfied

**Complexity**: O(|V|^k) where k is the number of hops

---

### 2. Subgraph Pattern Matching (`pattern_matching`)

**Purpose**: Find all images containing a specific relational pattern (mini-graph).

**Example Patterns**:
- Triangle: person-near-dog-near-cat-near-person
- Star: object with 3+ different relationships
- Path: A→wearing→B→on→C

**Graph Operations**:
1. Define pattern as a small graph template
2. Use subgraph isomorphism algorithms
3. Find all matches in the knowledge graph
4. Return images containing the pattern

**Complexity**: NP-complete in general, but tractable for small patterns

---

### 3. Scene Comparison Reasoning (`scene_comparison`)

**Purpose**: Compare two images/scenes based on structural properties.

**Example Questions**:
- "How similar are these two images in terms of object relationships?"
- "What objects are in image A but not in image B?"
- "Which image has more complex spatial relationships?"

**Graph Operations**:
1. Extract subgraphs for each image
2. Compute graph similarity metrics (node overlap, edge overlap)
3. Identify structural differences
4. Generate natural language comparison

**Metrics**: Jaccard similarity, graph edit distance, structural similarity

---

### 4. Counterfactual Reasoning (`counterfactual`)

**Purpose**: Explore hypothetical scenarios by modifying graph structure.

**Example Questions**:
- "If this person weren't in the image, what would change?"
- "What relationships would exist if we added a 'table' here?"
- "Which images would match if the dog were a cat instead?"

**Graph Operations**:
1. Create virtual graph with hypothetical modifications
2. Re-run queries on modified graph
3. Compare original vs. modified results
4. Identify causal dependencies

**Use Case**: Understanding object importance and relationship dependencies

---

### 5. Centrality-Based Retrieval (`centrality_query`)

**Purpose**: Find important/influential nodes using graph centrality measures.

**Example Questions**:
- "What is the most connected object type in scenes with people?"
- "Which attribute is most commonly shared across different concepts?"
- "Find 'hub' images that connect many different concepts"

**Graph Operations**:
1. Compute centrality measures (degree, betweenness, PageRank)
2. Identify hub nodes and authorities
3. Use centrality for ranking query results
4. Find bridging concepts between clusters

**Algorithms**: PageRank, betweenness centrality, eigenvector centrality

---

## Architecture

```
reasoning/
├── README.md                    # This documentation
├── advanced_reasoning_engine.py # Main implementation
├── pattern_templates.py         # Predefined graph patterns
├── graph_metrics.py             # Centrality and similarity functions
└── demo_advanced_reasoning.ipynb # Interactive demo notebook
```

## Usage

```python
from reasoning.advanced_reasoning_engine import AdvancedReasoningEngine

# Initialize with existing KG
engine = AdvancedReasoningEngine(scale='10k')

# Multi-hop chain reasoning
result = engine.chain_reasoning(
    start_concept="person",
    chain=[
        {"relation": "wearing", "concept": "shirt", "attribute": "red"},
        {"relation": "to the left of", "concept": "car"}
    ]
)
result.print_result()  # Shows detailed reasoning trace

# Pattern matching
result = engine.pattern_matching(
    pattern={
        "nodes": ["person", "dog", "ball"],
        "edges": [
            ("person", "holding", "ball"),
            ("dog", "looking at", "ball")
        ]
    }
)

# Scene comparison
result = engine.scene_comparison(
    image_id_1="2393941",
    image_id_2="2393942"
)
```

## Reasoning Trace Format

All advanced queries produce detailed reasoning traces:

```
🔍 REASONING TRACE:
   Step 1: [RETRIEVE] Found 1,234 instances of concept 'person'
   Step 2: [TRAVERSE] Following 'wearing' edges → 856 intermediate nodes
   Step 3: [FILTER] Applying attribute constraint 'red' → 142 nodes remain
   Step 4: [TRAVERSE] Following 'to the left of' edges → 45 paths found
   Step 5: [VERIFY] Checking target concept 'car' → 23 valid chains
   Step 6: [AGGREGATE] Grouping by image_id → 18 unique images
```

## Performance Considerations

| Query Type | Time Complexity | Scalability |
|------------|-----------------|-------------|
| Chain Reasoning | O(n^k) | Good for k ≤ 4 |
| Pattern Matching | O(n^p) | Good for p ≤ 5 nodes |
| Scene Comparison | O(e₁ + e₂) | Linear in edge count |
| Counterfactual | O(base query) | Same as base query |
| Centrality | O(V + E) | Linear, precomputable |

## Integration with Evaluation Framework

These advanced queries can be evaluated using the `evaluation_v0_1` framework:

```python
# Generate test suites for advanced queries
python -m evaluation_v0_1.run_evaluation --config configs/advanced_eval.yaml
```

## References

- Graph Pattern Matching: [GraphQL, SPARQL patterns]
- Subgraph Isomorphism: [VF2 algorithm]
- Knowledge Graph Reasoning: [Multi-hop QA, Chain-of-Thought]
- Graph Neural Networks: [Message Passing for reasoning]
