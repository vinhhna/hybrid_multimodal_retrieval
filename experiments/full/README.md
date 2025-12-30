# Full Dataset Experiment Results (74,942 images)

## 📁 Files in this directory:

### 1. Knowledge Graph Files
- **gqa_lightrag.gpickle** (470.61 MB) - Complete LightRAG knowledge graph
- **gqa_lightrag.stats.json** - Detailed statistics about the graph

---

## 📊 Knowledge Graph Statistics

```
Nodes:              1,233,453
├── Instance:       1,231,134  (99.8%)
├── Concept:            1,702  (0.14%)
└── Attribute:            617  (0.05%)

Edges:              5,634,778
├── instance_of:    1,231,134  (21.9%)
├── has_attribute:    675,340  (12.0%)
└── semantic_rel:   3,795,907  (67.4%)

Images:                74,289
File Size:            470.61 MB
Build Time:              ~25s
```

---

## 🎯 Demo Results Highlights

### Query Type 1: Entity Search
- **Red cars**: 780 instances found
- **White men**: 0 instances (concept mismatch - people don't have color attributes)

### Query Type 2: Statistical Knowledge
- **P(shirt | near man)** = 76.88% (24,117 co-occurrences / 31,370 men)
- **P(window | near building)** = 82.02% (15,701 co-occurrences / 19,143 buildings)

### Query Type 3: Similarity Search
- **Green, large trees**: 5,388 similar instances from 23,608 total trees
- **White shirts**: 5,185 similar instances from 25,171 total shirts

### Query Type 4: Relational Path
- **man → wearing → shirt**: Found 5 direct paths (2 hops)
- **plate → on → table**: Found 5 paths (2-3 hops)

### Query Type 5: Negative Constraints
- **Images with man but NO woman**: 16,007 images
- **Images with tree but NO car**: 10,775 images

### Query Type 6: Comparative
- **chair in kitchen**: 276 vs **chair in living room**: 216 (kitchen wins)
- **white walls**: 11.26% vs **white beds**: 7.01% (walls more common)

### Query Type 7: Hierarchical
- **Furniture**: 29,342 instances
  - table: 10,666 (36.3%)
  - chair: 7,513 (25.6%)
  - bench: 3,572 (12.2%)
  - cabinet: 2,322, bed: 2,039, desk: 1,556, couch: 1,302, sofa: 372
  
- **Electronic devices**: 7,418 instances
  - phone: 1,643 (22.1%)
  - laptop: 1,592 (21.5%)
  - camera: 1,267, television: 1,201, computer: 1,026, monitor: 689

### Query Type 8: Anomaly Detection
- **dog on table**: 10 instances found! 🐕🍽️ (rare case detected)
- **Rare relations (frequency < 2)**: 235,599 / 475,963 total SPO types (49.5%)
  - 'straw' → 'tablecloth', 'meat' → 'plate', etc.

### Query Type 9: Visual-Attribute Constraint
- **Red plastic cups**: 28 instances
- **Tall man wearing black shirt**: 0 (no match in dataset)

---

## 📈 Comparison: 1K vs 10K vs FULL

| Metric | 1K | 10K | FULL | Scale Factor |
|--------|-------|--------|----------|--------------|
| Images | 1,000 | 10,000 | 74,942 | 74.9x |
| Nodes | 17,463 | 164,585 | 1,233,453 | 70.6x |
| Edges | 72,713 | 738,736 | 5,634,778 | 77.5x |
| Concepts | 898 | 1,525 | 1,702 | 1.9x (saturated) |
| Attributes | 369 | 607 | 617 | 1.7x (saturated) |
| File Size | 6.15 MB | 61.86 MB | 470.61 MB | 76.5x |

### Key Insights:
1. **Concept/Attribute saturation**: Only increased 1.7-1.9x when scaling from 1k → full
   → Most concepts are already covered in 10k sample
   
2. **Linear scaling**: Nodes/Edges increase linearly (~75x) with number of images
   → System scales well, no bottlenecks

3. **Anomaly detection**: Only detectable with full dataset
   - Dog on table: 0 (1k), 0 (10k) → 10 (full)
   
4. **Query performance**: All 9 query types work stably with full dataset

---

## 🚀 Usage

### Load and query the graph:
```python
from src.gqa_reasoning_engine import GQA_Reasoning_Engine

# Initialize với full dataset
engine = GQA_Reasoning_Engine(scale='full')

# Example queries
results = engine.entity_search(concept='car', attributes=['red'])
stats = engine.statistical_knowledge(concept_a='man', concept_b='shirt')
anomalies = engine.find_specific_anomaly('dog', 'on', 'table')
```

### Run demo:
```bash
# Run all 9 query types
python scripts/demo_nl_interface.py --scale full

# Run using saved output wrapper  
python scripts/run_demo_and_save.py --scale full --demo all --output results.txt
```

---

## 📝 Notes

- **Encoding**: All result files use UTF-8 encoding for proper display of emojis
- **Performance**: Graph loading takes ~5-10s, each query averages <1s
- **Memory**: Requires ~2-3 GB RAM to load full graph
- **Completeness**: All 9 query types tested and verified with 18 demo questions

---

## 📅 Build Information

- **Build Date**: December 30, 2025
- **Dataset**: GQA train_sceneGraphs.json (74,942 images)
- **Architecture**: LightRAG (2-tier: Instance + Global)
- **Builder Script**: `python src/gqa_lightrag_builder.py --scale full`
