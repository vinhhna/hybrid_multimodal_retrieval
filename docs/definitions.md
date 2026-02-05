# Formal Definitions and Graph Schema

## 1. Input Data Format: GQA Scene Graphs

### Definition
A **GQA Scene Graph** is a JSON object representing the spatial and semantic relationships in a single image. It defines:
- A set of **objects** (entities in the image)
- For each object: attributes and relations to other objects

### Schema
```json
{
  "image_id": {
    "objects": {
      "object_id": {
        "name": "string",           // Object category (e.g., "dog", "shirt")
        "x": int,                   // Bounding box x coordinate
        "y": int,                   // Bounding box y coordinate
        "w": int,                   // Bounding box width
        "h": int,                   // Bounding box height
        "attributes": [string],     // List of visual attributes (e.g., ["red", "large"])
        "relations": [
          {
            "name": "string",       // Relation name (e.g., "wearing", "on", "to the left of")
            "object": "object_id"   // Target object_id
          }
        ]
      }
    }
  }
}
```

### Example Input
```json
{
  "2386621": {
    "objects": {
      "0": {
        "name": "person",
        "x": 100, "y": 50, "w": 200, "h": 300,
        "attributes": ["man", "tall", "standing"],
        "relations": [
          {"name": "wearing", "object": "1"},
          {"name": "holding", "object": "2"}
        ]
      },
      "1": {
        "name": "shirt",
        "x": 120, "y": 80, "w": 160, "h": 180,
        "attributes": ["black", "cotton"],
        "relations": []
      },
      "2": {
        "name": "bag",
        "x": 250, "y": 200, "w": 80, "h": 100,
        "attributes": ["red", "leather"],
        "relations": []
      }
    }
  }
}
```

---

## 2. Knowledge Graph Representation (LightRAG)

### 2.1 Node Types

#### **Type A: Instance Nodes** (`node_type='instance'`)
- **Represents**: A single object occurrence in a specific image
- **ID Format**: `{image_id}:{object_id}` (e.g., `2386621:0`)
- **Attributes**:
  - `node_type`: `'instance'`
  - `level`: `'instance'`
  - `image_id`: The source image ID
  - `object_id`: The object ID within the image
  - `name`: Object category name (string, normalized)
  - `attributes`: List of visual attributes (strings, normalized)
  - `x, y, w, h`: Bounding box coordinates

#### **Type B: Global Concept Nodes** (`node_type='global_concept'`)
- **Represents**: An aggregated concept across all images
- **ID Format**: `Concept:{normalized_name}` (e.g., `Concept:dog`)
- **Attributes**:
  - `node_type`: `'global_concept'`
  - `level`: `'global'`
  - `name`: The concept name (normalized to lowercase)

#### **Type C: Global Attribute Nodes** (`node_type='global_attribute'`)
- **Represents**: An aggregated attribute concept across all images
- **ID Format**: `Attr:{normalized_name}` (e.g., `Attr:red`)
- **Attributes**:
  - `node_type`: `'global_attribute'`
  - `level`: `'global'`
  - `name`: The attribute name (normalized to lowercase)

### 2.2 Edge Types

#### **Edge A: `instance_of`** (Instance → Concept)
- **Direction**: Instance Node → Global Concept Node
- **Meaning**: "This instance is an example of this concept"
- **Attributes**:
  - `edge_type`: `'instance_of'`
  - `relation`: `'instance_of'`
- **Cardinality**: Many-to-one (multiple instances → one concept)
- **Example**: `2386621:0` (dog instance) → `Concept:dog`

#### **Edge B: `has_attribute`** (Instance → Attribute)
- **Direction**: Instance Node → Global Attribute Node
- **Meaning**: "This instance has this visual property"
- **Attributes**:
  - `edge_type`: `'has_attribute'`
  - `relation`: `'has_attribute'`
- **Cardinality**: Many-to-many (instances can have multiple attributes)
- **Example**: `2386621:0` (person) → `Attr:tall`

#### **Edge C: `semantic_relation`** (Instance ↔ Instance)
- **Direction**: Instance Node → Instance Node (directional)
- **Meaning**: Spatial or semantic relationship within same image
- **Attributes**:
  - `edge_type`: `'semantic_relation'`
  - `relation`: The relation name from scene graph (e.g., `"wearing"`, `"on"`, `"to the left of"`)
  - `image_id`: The image where this relation exists
- **Cardinality**: One-to-many (one instance can have multiple relations)
- **Constraint**: Both source and target must be in the same image
- **Example**: `2386621:0` (person) --[wearing]--> `2386621:1` (shirt)

### 2.3 Graph Structure

```
┌─────────────────────────────────────────────────────────────┐
│                   LightRAG Two-Layer Architecture            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  GLOBAL LEVEL (Aggregated)                                  │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │ Concept:dog  │      │ Attr:red     │                    │
│  └──────┬───────┘      └──────┬───────┘                    │
│         │                     │                             │
│    [instance_of]         [has_attribute]                   │
│         │                     │                             │
│  ┌──────┴─────────────────────┴───────┐                   │
│  │                                    │                    │
│  │        INSTANCE LEVEL               │                   │
│  │   (Image-Specific Objects)         │                   │
│  │                                    │                   │
│  │  Instance: 2386621:0 (dog)        │                   │
│  │  Attributes: [brown, large]       │                   │
│  │                                    │                   │
│  │  Instance: 2386621:1 (person)     │                   │
│  │          [wearing]                 │                   │
│  │            ────────► dog            │                   │
│  └────────────────────────────────────┘                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Normalization Rules

### Name Normalization Function
```
normalize(text) = lowercase(strip(text))
```

### Application
- All concept names are normalized: `"Dog"`, `"DOG"`, `"dog"` → `"dog"`
- All attribute names are normalized: `"Red"`, `"RED"`, `"red"` → `"red"`
- Global node IDs use normalized names:
  - Concept node: `Concept:dog` (not `Concept:Dog`)
  - Attribute node: `Attr:red` (not `Attr:Red`)

### Consequence
The knowledge graph is case-insensitive by design. Queries with different cases are automatically unified.

---

## 4. Query Result Types

### Type 1: Instance Results
- **Format**: List of instance node dictionaries
- **Fields per instance**:
  - `node_id`: The instance node ID (e.g., `2386621:0`)
  - `image_id`: Source image ID
  - `object_id`: Object ID within image
  - `name`: Object category name
  - `attributes`: List of attributes
- **Example Output**:
  ```json
  [
    {
      "node_id": "2386621:0",
      "image_id": "2386621",
      "object_id": "0",
      "name": "person",
      "attributes": ["man", "tall"]
    }
  ]
  ```

### Type 2: Image Results
- **Format**: List of unique image IDs
- **Example Output**: `["2386621", "2373554", "2380095"]`

### Type 3: Scalar Results
- **Format**: Single numeric value
- **Examples**: Count, probability, average, ratio
- **Example Output**: `{"count": 245, "percentage": 45.3}`

### Type 4: Path Results
- **Format**: List of graph paths (sequences of nodes with relations)
- **Example Path**: `2386621:0 --[wearing]--> 2386621:1 --[on]--> 2386621:2`

### Type 5: Subgraph Results
- **Format**: A subset of the knowledge graph (nodes and edges)
- **Used for**: Evidence extraction, visualization
- **Representation**: Dictionary with `nodes` and `edges`

---

## 5. Algorithm Primitives

### 5.1 Concept Lookup
```
lookup_concept(concept_name) → Global_Concept_Node_ID
  normalized_name = normalize(concept_name)
  node_id = "Concept:" + normalized_name
  return node_id if exists in graph, else null
```

### 5.2 Get All Instances of Concept
```
instances_of(concept_name) → Set[Instance_Node_IDs]
  concept_node = lookup_concept(concept_name)
  if not concept_node: return ∅
  
  result = ∅
  for edge in graph.in_edges(concept_node):
    if edge.type == "instance_of":
      result ← result ∪ {edge.source}
  return result
```

### 5.3 Get All Instances with Attribute
```
instances_with_attribute(attribute_name) → Set[Instance_Node_IDs]
  attr_node = "Attr:" + normalize(attribute_name)
  if not attr_node in graph: return ∅
  
  result = ∅
  for edge in graph.in_edges(attr_node):
    if edge.type == "has_attribute":
      result ← result ∪ {edge.source}
  return result
```

### 5.4 Set Intersection (Filtering)
```
filter_by_multiple(instances, attributes) → Set[Instance_Node_IDs]
  result = instances
  for attr in attributes:
    attr_instances = instances_with_attribute(attr)
    result = result ∩ attr_instances
  return result
```

### 5.5 Get Semantic Neighbors
```
neighbors_via_relation(instance_node, relation_name) → List[(Instance_Node, relation)]
  result = []
  
  // Forward edges
  for edge in graph.out_edges(instance_node):
    if edge.type == "semantic_relation" AND matches(edge.relation, relation_name):
      result.append((edge.target, edge.relation))
  
  // Backward edges
  for edge in graph.in_edges(instance_node):
    if edge.type == "semantic_relation" AND matches(edge.relation, relation_name):
      result.append((edge.source, edge.relation))
  
  return result
```

### 5.6 BFS Graph Traversal
```
bfs_traverse(start_node, relation, max_depth) → List[List[node]]
  paths = [[start_node]]
  
  for depth in 1..max_depth:
    new_paths = []
    for path in paths:
      current = path[-1]
      neighbors = neighbors_via_relation(current, relation)
      
      for neighbor, rel_name in neighbors:
        if neighbor not in path:  // Prevent cycles
          new_path = path + [neighbor]
          new_paths.append(new_path)
    
    paths.extend(new_paths)
  
  return paths
```

---

## 6. Formal Semantics: Example Query

### Example: Entity Search
**Question**: "Find all men wearing red shirts"

**Formal Semantics**:
```
Result = {inst | 
  inst ∈ instances_of("man") ∧
  ∃ inst' ∈ instances_of("shirt") :
    (inst, inst') ∈ edges("wearing") ∧
    inst' ∈ instances_with_attribute("red")
}
```

**Execution Algorithm** (from code):
1. Load instances of "man" via `instances_of("man")`
2. Load instances of "shirt" via `instances_of("shirt")`
3. Filter shirt instances by "red" attribute
4. For each man, find neighbors with "wearing" relation
5. Keep only men whose wearing-neighbors are in filtered shirts
6. Return as list of instance dictionaries

**Code Reference**: [reasoning_engine.py:288](../src/lightrag_gqa/basic_queries/reasoning_engine.py#L288)

---

## 7. Graph Statistics (10K Scale Example)

| Metric | Value |
|--------|-------|
| Total images | 10,000 |
| Total instance nodes | 164,585 |
| Total global concepts | ~5,000 |
| Total global attributes | ~1,500 |
| instance_of edges | ~164,585 |
| has_attribute edges | ~738,736 |
| semantic_relation edges | ~738,736 |
| **Total edges** | **~1,642,057** |
| Average instance degree | ~14.2 |
| Graph density | ~0.0001 |

---

## 8. Key Invariants

1. **Global Node Uniqueness**: For each unique normalized concept/attribute name, exactly one global node exists.
2. **Instance-Global Mapping**: Every instance node has exactly one outgoing `instance_of` edge to its concept.
3. **Attribute Bindings**: Instance-to-attribute edges preserve image boundaries (both nodes come from same image context).
4. **Relation Locality**: Every `semantic_relation` edge connects instances from the same image (enforced by builder).
5. **No Dangling References**: Every `semantic_relation` target must correspond to an actual instance node in the graph.

