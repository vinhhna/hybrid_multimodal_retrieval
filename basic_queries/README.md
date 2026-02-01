# Basic Queries - 9 Standard Query Types

This module contains the implementation of the **9 standard query types** for the GQA LightRAG system.

## 📋 Query Types

### 1. Entity Search
Search for entities (objects) by concept name or attributes.
- Example: "Find all images with dogs"
- Example: "Find red cars"

### 2. Statistical Queries
Count and aggregate information across the knowledge graph.
- Example: "How many people are wearing hats?"
- Example: "What's the most common object?"

### 3. Similarity & Pattern Matching
Find semantically similar concepts or specific patterns.
- Example: "Find concepts similar to 'vehicle'"
- Example: "Find images with pattern X"

### 4. Relational Path Discovery
Discover paths and connections between entities.
- Example: "Find all relationships between person and table"
- Example: "What connects A to B?"

### 5. Negative Constraints (NOT Queries)
Search with exclusion criteria.
- Example: "Find outdoor scenes without people"
- Example: "Images with dogs but not cats"

### 6. Comparative Queries
Compare entities or attributes.
- Example: "Find larger objects than X"
- Example: "Compare attributes of A and B"

### 7. Hierarchical Queries
Navigate concept hierarchies and taxonomies.
- Example: "Find all types of furniture"
- Example: "What are subtypes of vehicle?"

### 8. Anomaly Detection
Identify unusual patterns or outliers.
- Example: "Find unusual object combinations"
- Example: "Detect rare relationships"

### 9. Visual-Attribute Constraint Queries
Complex queries combining visual and attribute constraints.
- Example: "Find large red objects on tables"
- Example: "Person wearing white shirt near blue car"

## 🗂️ Files

- **gqa_reasoning_engine.py** - Core reasoning engine implementing all 9 query types
- **gqa_query_interface.py** - User-friendly interface for query execution
- **gqa_nl_parser.py** - Natural language query parser
- **gqa_lightrag_builder.py** - Knowledge graph builder from GQA scene graphs

## 🚀 Usage

```python
from basic_queries import GQA_Reasoning_Engine

# Initialize engine
engine = GQA_Reasoning_Engine(scale='10k')

# Execute queries
results = engine.entity_search(concept='person', limit=10)
stats = engine.statistical_query(query_type='count_by_concept')
paths = engine.relational_path_query(source='person', target='table')
```

## 📊 Performance

These queries are optimized for:
- **Fast retrieval** (< 50ms for most queries)
- **Large-scale graphs** (100k+ nodes)
- **Interactive use** (real-time response)
