# Traceability Matrix: Query Types to Code Implementation

## Overview
This document maps each of the 9 supported query types to their implementation locations, algorithms, and formal definitions.

**Total Query Types Implemented**: 9 (5 basic + 4 advanced)*

*Note: README claims 14 types (9 basic + 5 advanced), but only 9 are implemented in code. See [docs/limitations_and_failure_modes.md](limitations_and_failure_modes.md#claimed-vs-implemented) for details.*

---

## Basic Query Types (5 Implemented)

### Query Type 1: Entity Search

**Intent**: Find objects matching concept and/or attributes.

**Input**: 
- `concept`: Object category name (optional)
- `attributes`: List of attribute names (optional)
- `limit`: Max results to return

**Output**: List of matching instance node dictionaries

**File**: [src/lightrag_gqa/basic_queries/reasoning_engine.py](../src/lightrag_gqa/basic_queries/reasoning_engine.py)

**Function**: `GQA_Reasoning_Engine.entity_search()` (line 288)

**Algorithm**:
```
1. If concept is provided:
   - Lookup concept node via normalize(concept)
   - Get all instances connected via instance_of edge
   - Store in candidates set
   
2. If attributes provided:
   - For each attribute:
     - Lookup attribute node
     - Get all instances connected via has_attribute edge
     - Intersect with candidates (AND semantics)
   
3. Format output:
   - For each instance in candidates (up to limit):
     - Extract node_id, image_id, object_id, name, attributes
     - Return as structured result list
```

**Formal Semantics**:
```
RESULT = {inst | 
  (concept = ∅ ∨ inst ∈ instances_of(concept)) ∧
  (attributes = ∅ ∨ ∀attr ∈ attributes: inst ∈ instances_with_attr(attr))
}
```

**Example Questions**:
- "Find all red cars"
- "Show me dogs wearing hats"
- "List all large tables in kitchens"

**Reasoning Trace Format**: 
```
HOP 1: Concept lookup → N instances via instance_of edge
HOP 2: Attribute lookup → M instances via has_attribute edge
...
FINAL: Intersection → K results
```

**Correctness Criteria**:
- All returned instances have the specified concept
- All returned instances have ALL specified attributes
- No missing instances that satisfy constraints (completeness depends on scene graph input)

---

### Query Type 2: Statistical Knowledge

**Intent**: Compute co-occurrence statistics between two concepts.

**Input**:
- `concept_a`: First concept name
- `concept_b`: Second concept name
- `relation`: Optional relation type filter

**Output**: Probability/frequency statistics

**File**: [src/lightrag_gqa/basic_queries/reasoning_engine.py](../src/lightrag_gqa/basic_queries/reasoning_engine.py)

**Function**: `GQA_Reasoning_Engine.statistical_knowledge()` (line 397)

**Algorithm**:
```
1. Get all instances of concept_a
2. Get all instances of concept_b
3. For each instance of a:
   - Find neighbors via semantic_relation edges
   - Count how many are instances of concept_b (with optional relation filter)
4. Compute statistics:
   - Total a instances: |A|
   - Total b instances: |B|
   - Co-occurrence count: |{(a, b) : a→b ∈ semantic_relations}|
   - Probability: co-occurrence_count / |A|
   - Joint probability: co-occurrence_count / (|A| × |B|)
```

**Formal Semantics**:
```
P(concept_b | concept_a) = |{inst_a ∈ A : ∃inst_b ∈ B, (inst_a, inst_b) ∈ E}| / |A|
```

**Example Questions**:
- "How likely is it to find a shirt near a person?"
- "What is the probability of seeing a window in a building?"
- "Co-occurrence of dogs and people"

**Correctness Criteria**:
- Probability ∈ [0, 1]
- Counts accurately reflect instance pairs in same image
- Only counts direct neighbor relationships (not transitive)

---

### Query Type 3: Similarity Search

**Intent**: Find instances with similar attribute profiles.

**Input**:
- `concept`: Reference concept
- `attributes`: List of reference attributes
- `min_common_attributes`: Minimum overlap threshold
- `limit`: Max results

**Output**: Ranked list of similar instances

**File**: [src/lightrag_gqa/basic_queries/reasoning_engine.py](../src/lightrag_gqa/basic_queries/reasoning_engine.py)

**Function**: `GQA_Reasoning_Engine.similarity_search()` (line 508)

**Algorithm**:
```
1. Get all instances of concept
2. For each instance:
   - Extract its attribute set
3. For each pair of instances:
   - Compute Jaccard similarity: |A ∩ B| / |A ∪ B|
4. Rank by similarity score (descending)
5. Return top-K instances with similarity ≥ min_common_attributes
```

**Formal Semantics**:
```
SIMILAR(ref_attrs) = 
  {inst ∈ instances_of(concept) :
    |attrs(inst) ∩ ref_attrs| ≥ min_common_attributes
  }
  sorted by Jaccard(attrs(inst), ref_attrs) descending
```

**Example Questions**:
- "Find trees similar to a large green one"
- "What objects resemble a red plastic cup?"
- "Find white chairs like this one"

**Correctness Criteria**:
- All returned instances belong to reference concept
- Similarity score ≥ threshold
- No instances with threshold-violating similarity included

---

### Query Type 4: Relational Path Discovery

**Intent**: Find graph paths between two concepts.

**Input**:
- `source_concept`: Starting concept
- `target_concept`: Destination concept
- `via_relation`: Optional required relation (e.g., "wearing")
- `max_hops`: Maximum path length

**Output**: List of paths (sequences of instance nodes)

**File**: [src/lightrag_gqa/basic_queries/reasoning_engine.py](../src/lightrag_gqa/basic_queries/reasoning_engine.py)

**Function**: `GQA_Reasoning_Engine.relational_path()` (line 643)

**Algorithm**: Breadth-first search (BFS)
```
1. Initialize queue with all instances of source_concept
2. For each hop (up to max_hops):
   - For each current node:
     - Get neighbors via semantic_relations
     - Filter by via_relation if specified
     - Add to queue if not in current path (prevent cycles)
3. For final hop:
   - Check if any neighbor is an instance of target_concept
   - If yes, include path in results
4. Return all found paths, sorted by length
```

**Formal Semantics**:
```
PATHS = {[n₀, n₁, ..., nₖ] |
  n₀ ∈ instances_of(source) ∧
  nₖ ∈ instances_of(target) ∧
  k ≤ max_hops ∧
  ∀i ∈ [0, k-1]: (nᵢ, nᵢ₊₁) ∈ E(via_relation) ∧
  nᵢ ∉ nⱼ for i ≠ j (acyclic)
}
```

**Example Questions**:
- "Find path from person to shirt"
- "How is dog connected to table?"
- "What connects plate to kitchen?"

**Correctness Criteria**:
- All returned paths start with source concept instance
- All returned paths end with target concept instance
- No path exceeds max_hops
- All intermediate edges exist and have correct relation type

---

### Query Type 5: Negative Constraints

**Intent**: Find instances with one concept but excluding another.

**Input**:
- `concept_present`: Required concept
- `concept_absent`: Excluded concept
- `limit`: Max results

**Output**: Images containing concept_present but not concept_absent

**File**: [src/lightrag_gqa/basic_queries/reasoning_engine.py](../src/lightrag_gqa/basic_queries/reasoning_engine.py)

**Function**: `GQA_Reasoning_Engine.negative_constraints()` (line 818)

**Algorithm**:
```
1. Get images containing concept_present:
   - instances_with_present = instances_of(concept_present)
   - images_with_present = {img_id(inst) : inst ∈ instances_with_present}

2. Get images containing concept_absent:
   - instances_with_absent = instances_of(concept_absent)
   - images_with_absent = {img_id(inst) : inst ∈ instances_with_absent}

3. Compute difference:
   - result_images = images_with_present - images_with_absent

4. Return list of image IDs (up to limit)
```

**Formal Semantics**:
```
RESULT = {img |
  (∃inst: inst ∈ instances_of(concept_present) ∧ img = img_id(inst)) ∧
  (¬∃inst: inst ∈ instances_of(concept_absent) ∧ img = img_id(inst))
}
```

**Example Questions**:
- "Find images with dogs but not cats"
- "Show me scenes with people but no vehicles"
- "Images with trees excluding cars"

**Correctness Criteria**:
- All returned images contain at least one instance of concept_present
- No returned image contains any instance of concept_absent
- No false negatives (all qualifying images included)

---

## Advanced Query Types (4 Implemented)

### Query Type 6: Chain Reasoning (Multi-Hop)

**Intent**: Multi-step reasoning with intermediate constraints.

**Input**:
- `start_concept`: Starting concept
- `chain`: List of hop specifications, each with:
  - `relation`: Relation name to traverse
  - `concept`: Target concept (optional)
  - `attribute`: Required attribute (optional)
- `limit`: Max results

**Output**: Complete chains of instances satisfying all constraints

**File**: [src/lightrag_gqa/advanced_queries/reasoning_engine.py](../src/lightrag_gqa/advanced_queries/reasoning_engine.py)

**Function**: `AdvancedReasoningEngine.chain_reasoning()` (line 188)

**Algorithm**:
```
1. Initialize paths with all instances of start_concept
2. For each hop in chain:
   - For each current path endpoint:
     - Traverse via specified relation
     - Get neighboring nodes
     - Filter by concept if specified (intersection with instances_of)
     - Filter by attribute if specified (intersection with instances_with_attr)
     - Add to new paths (without revisiting nodes in path)
   - Replace paths with filtered paths
3. Return all surviving paths that completed all hops
```

**Formal Semantics**:
```
CHAINS = {[n₀, n₁, ..., nₖ] |
  n₀ ∈ instances_of(start) ∧
  ∀i ∈ [1, k]:
    (nᵢ₋₁, nᵢ) ∈ E(chain[i].relation) ∧
    (chain[i].concept = ∅ ∨ nᵢ ∈ instances_of(chain[i].concept)) ∧
    (chain[i].attribute = ∅ ∨ nᵢ ∈ instances_with_attr(chain[i].attribute)) ∧
    nᵢ ∉ {n₀, ..., nᵢ₋₁}
}
```

**Example Questions**:
- "Find men wearing red shirts near tables"
- "Person → wearing → black shirt → on → table"

**Reasoning Steps Tracked**:
- RETRIEVE: Get starting instances
- TRAVERSE: Follow relation edges
- FILTER: Apply concept/attribute constraints
- AGGREGATE: Combine constraints

**Correctness Criteria**:
- All chains follow specified relations in order
- All intermediate nodes satisfy optional concept/attribute filters
- No node appears twice in same chain (acyclic)
- Semantics: AND between constraints within a hop, sequential hops

---

### Query Type 7: Pattern Matching

**Intent**: Find subgraph instances matching a pattern.

**Input**:
- `pattern_nodes`: List of target concepts
- `pattern_edges`: List of (source_concept, relation, target_concept) tuples
- `limit`: Max patterns

**Output**: Matching subgraph instances

**File**: [src/lightrag_gqa/advanced_queries/reasoning_engine.py](../src/lightrag_gqa/advanced_queries/reasoning_engine.py)

**Function**: `AdvancedReasoningEngine.pattern_matching()` (line 386)

**Algorithm**:
```
1. Initialize with instances of first pattern node concept
2. For each pattern edge (s_concept, relation, t_concept):
   - For each current instance of s_concept:
     - Find neighbors via relation
     - Filter to instances of t_concept
     - Verify edge exists and is in same image
3. Aggregate results:
   - For each valid subgraph:
     - Extract nodes and edges
     - Format as pattern match
4. Deduplicate and limit results
```

**Formal Semantics**:
```
MATCHES = {(N, E) |
  N = {n₁, n₂, ..., nₖ} where nᵢ ∈ instances_of(pattern_nodes[i]) ∧
  E ⊆ {(nᵢ, nⱼ) : (pattern_nodes[i], rel, pattern_nodes[j]) ∈ pattern_edges
           ∧ (nᵢ, nⱼ) ∈ semantic_relations(rel)
           ∧ img_id(nᵢ) = img_id(nⱼ)}
}
```

**Example Questions**:
- "Find patterns: person wearing shirt"
- "Find subgraphs: dog on table near person"

**Correctness Criteria**:
- All pattern node concepts present in match
- All pattern edges satisfied in match
- No spurious matches (edge must exist)
- Instances must be in same image

---

### Query Type 8: Scene Comparison

**Intent**: Compare structural properties between images.

**Input**:
- `image_id_1`, `image_id_2`: Two image IDs
- Or two concept sets to compare

**Output**: Comparative statistics (overlap, unique elements)

**File**: [src/lightrag_gqa/advanced_queries/reasoning_engine.py](../src/lightrag_gqa/advanced_queries/reasoning_engine.py)

**Function**: `AdvancedReasoningEngine.scene_comparison()` (line 535)

**Algorithm**:
```
1. Extract objects from image_id_1:
   - Get all instance nodes with img_id = image_id_1
   - Extract concepts and attributes for each

2. Extract objects from image_id_2:
   - Get all instance nodes with img_id = image_id_2
   - Extract concepts and attributes for each

3. Compute comparison metrics:
   - Total objects in each
   - Concept overlap: concepts in both
   - Concept difference: concepts only in one
   - Attribute statistics

4. Format results:
   - Percentage similarity
   - List of common/unique elements
```

**Example Questions**:
- "Compare image 2386621 and 2373554"
- "Are these two scenes similar?"

**Correctness Criteria**:
- Comparison metrics are symmetric (up to variable order)
- No instances from other images included
- Coverage is complete (all instances accounted for)

---

### Query Type 9: Counterfactual Reasoning

**Intent**: Hypothetical "what-if" analysis.

**Input**:
- `initial_instance`: Starting instance
- `remove_relation`: Relation to remove
- `target_concept`: Target for alternative relation

**Output**: Hypothetical subgraph with modification

**File**: [src/lightrag_gqa/advanced_queries/reasoning_engine.py](../src/lightrag_gqa/advanced_queries/reasoning_engine.py)

**Function**: `AdvancedReasoningEngine.counterfactual_reasoning()` (line 713)

**Algorithm**:
```
1. Load initial subgraph around instance
2. Identify edges to remove (matching remove_relation)
3. Find alternative targets (instances of target_concept)
4. Generate hypothetical graph:
   - Original subgraph - removed edges + alternative edges
5. Analyze consequences:
   - New neighbors
   - Changed relationships
   - Path accessibility
```

**Example Questions**:
- "If this person wasn't wearing a shirt, what would change?"
- "What if this dog was on a chair instead of a table?"

**Correctness Criteria**:
- Modifications are clearly marked as counterfactual
- Original graph remains unchanged (logical operation only)
- Alternative relations must be valid (target must exist in image)

---

## Query Type Mapping Summary

| # | Type | Category | File | Function | Algorithm | I/O |
|---|------|----------|------|----------|-----------|-----|
| 1 | Entity Search | Basic | reasoning_engine.py | `.entity_search()` | Set intersection | Concept+Attrs → Instances |
| 2 | Statistical | Basic | reasoning_engine.py | `.statistical_knowledge()` | Co-occurrence counting | Concepts → Probability |
| 3 | Similarity | Basic | reasoning_engine.py | `.similarity_search()` | Jaccard similarity | Concept+Attrs → Ranked instances |
| 4 | Relational Path | Basic | reasoning_engine.py | `.relational_path()` | BFS traversal | Source+Target+Relation → Paths |
| 5 | Negative Constraints | Basic | reasoning_engine.py | `.negative_constraints()` | Set difference | Concepts → Image IDs |
| 6 | Chain Reasoning | Advanced | advanced/reasoning_engine.py | `.chain_reasoning()` | Sequential filtering | Start+Chain spec → Complete chains |
| 7 | Pattern Matching | Advanced | advanced/reasoning_engine.py | `.pattern_matching()` | Subgraph isomorphism | Pattern spec → Matching subgraphs |
| 8 | Scene Comparison | Advanced | advanced/reasoning_engine.py | `.scene_comparison()` | Structural comparison | Images → Comparison metrics |
| 9 | Counterfactual | Advanced | advanced/reasoning_engine.py | `.counterfactual_reasoning()` | Graph modification | Instance+Modifications → Alt. subgraph |

---

## Natural Language Parsing

**File**: [src/lightrag_gqa/basic_queries/nl_parser.py](../src/lightrag_gqa/basic_queries/nl_parser.py)

**Class**: `NaturalLanguageParser`

**Method**: `.parse(nl_query) → ParseResult`

**Mechanism**: Rule-based pattern matching with confidence scores

**Supported Patterns** (by query type):
- **Entity Search**: "find X", "show me Y", "list Z"
- **Statistical**: "probability of X near Y", "how often X with Y"
- **Similarity**: "similar to X", "looks like Y"
- **Relational Path**: "path from X to Y", "connection between X and Y"
- **Negative Constraints**: "X but not Y", "X without Y"
- **Comparative**: "compare X in A vs B"
- **Hierarchical**: "all furniture", "types of vehicles"
- **Anomaly**: "unusual X Y relation", "rare patterns"
- **Visual-Attribute**: "red plastic cup", "tall man wearing shirt"

**Limitations**: Pattern matching is heuristic; complex queries may misclassify.

---

## Evaluation Integration

See [docs/evaluation_protocol.md](evaluation_protocol.md) for how query types are evaluated.

