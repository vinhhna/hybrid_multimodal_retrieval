# Troubleshooting Guide - Evaluation Framework v0.1

## 🔍 Identified Issues and Fixes

Based on the evaluation run on January 20, 2026, here are the specific issues found and how to fix them:

---

## Issue #1: Zero Precision/Recall on Ranked Queries ❌

### Symptom
```
Precision@50: 0.0000
Recall@50: 0.0000
MRR: 0.0000
```

### Root Cause Analysis

The engine is executing queries successfully (100% success rate) but returning no relevant results. This suggests one of:

1. **Adapter not lifting instance IDs to image IDs correctly**
2. **Query interface returning empty results**
3. **Mismatch between oracle and engine concept normalization**

### Debugging Steps

#### Step 1: Add Debug Logging to Adapter

Edit `evaluation_v0_1/scripts/adapter_engine.py`:

```python
def execute(self, cqr: CQR) -> Dict[str, Any]:
    """Execute CQR and return standardized results."""
    
    if cqr.output_type == OutputType.RANKED_SET:
        # Get concepts and attributes
        concepts = [c['value'] for c in cqr.must if c['type'] == 'concept']
        attrs = [a['value'] for a in cqr.must if a['type'] == 'attr']
        
        # === ADD THIS DEBUG BLOCK ===
        print(f"[Adapter Debug] Query: {concepts} + {attrs}")
        
        # Execute via query interface
        results = self.query_interface.query_entities_with_attributes(
            concept_names=concepts,
            attribute_names=attrs
        )
        
        # === ADD THIS DEBUG BLOCK ===
        print(f"[Adapter Debug] Raw results: {len(results)} items")
        print(f"[Adapter Debug] First 3 results: {results[:3]}")
        
        # Lift to image IDs
        image_ids = self._lift_to_images(results)
        
        # === ADD THIS DEBUG BLOCK ===
        print(f"[Adapter Debug] Image IDs: {len(image_ids)} images")
        print(f"[Adapter Debug] First 3 images: {list(image_ids)[:3]}")
        
        return {
            "image_ids": list(image_ids)[:cqr.k],
            "scores": [1.0] * min(len(image_ids), cqr.k)
        }
```

#### Step 2: Run Single Query Test

Create a test script `test_single_query.py`:

```python
from evaluation_v0_1.scripts.adapter_engine import create_adapter
from evaluation_v0_1.scripts.cqr import CQR, Constraint, OutputType

# Create adapter
adapter = create_adapter("experiments/sample_10k/gqa_lightrag.gpickle")

# Simple test query: "Find red cars"
cqr = CQR(
    query_id="test_001",
    output_type=OutputType.RANKED_SET,
    op="retrieve",
    must=[
        Constraint(type="concept", value="car"),
        Constraint(type="attr", value="red")
    ],
    k=10
)

print("Testing query:", cqr)
result = adapter.execute(cqr)
print("Result:", result)
print(f"Found {len(result['image_ids'])} images")
```

Run it:
```bash
python test_single_query.py
```

#### Step 3: Check Oracle Ground Truth

Compare with what oracle expects:

```python
from evaluation_v0_1.scripts.oracle_scenegraphs import load_oracle

oracle = load_oracle("sceneGraphs/val_sceneGraphs.json")
gold = oracle.get_gold_output(cqr)
print(f"Oracle expects {len(gold.image_ids)} images")
print(f"First 3: {list(gold.image_ids)[:3]}")
```

### Likely Fix

**If adapter returns empty results**: Check normalization. The query interface might expect different format:
```python
# Instead of:
results = qi.query_entities_with_attributes(["car"], ["red"])

# Try:
results = qi.retrieve_images_with_concept_and_attrs("car", ["red"])
```

**If adapter returns instances but no images**: Check `_lift_to_images()`:
```python
def _lift_to_images(self, instance_ids: List[str]) -> Set[str]:
    """Convert instance IDs to image IDs."""
    image_ids = set()
    for inst_id in instance_ids:
        if inst_id in self.instance_to_image:
            image_ids.add(self.instance_to_image[inst_id])
        else:
            # === ADD THIS CHECK ===
            print(f"[Warning] Instance {inst_id} not in mapping!")
    return image_ids
```

---

## Issue #2: Parser Can't Handle Scalar Queries ❌

### Symptom
```
suite_scalar_stats:
  Heuristic exact match: 0.0000
  LLM stub exact match: 0.0600
```

### Root Cause

The heuristic parser in `src/gqa_nl_parser.py` doesn't have patterns for statistical queries.

### Fix

Add patterns to detect scalar queries. Edit `src/gqa_nl_parser.py`:

```python
def parse_nl_query(self, nl_query: str) -> Dict[str, Any]:
    """Parse natural language query into structured format."""
    
    nl_lower = nl_query.lower().strip()
    
    # === ADD THIS BLOCK ===
    # Detect scalar/statistical queries
    scalar_patterns = [
        r"how often",
        r"what is.*probability",
        r"what is p\(",
        r"what fraction",
        r"how frequently"
    ]
    
    for pattern in scalar_patterns:
        if re.search(pattern, nl_lower):
            return self._parse_scalar_query(nl_query)
    # === END NEW BLOCK ===
    
    # ... rest of existing code ...
    
def _parse_scalar_query(self, nl_query: str) -> Dict[str, Any]:
    """Parse statistical queries like 'How often is X on Y?'"""
    
    nl_lower = nl_query.lower()
    
    # Pattern: "How often is A <relation> B?"
    match = re.search(r"how often is (?:a |an )?(\w+) (.*?) (?:a |an )?(\w+)", nl_lower)
    if match:
        concept_a = match.group(1)
        relation = match.group(2).strip()
        concept_b = match.group(3)
        
        return {
            "output_type": "scalar",
            "op": "stats",
            "concepts": [concept_a, concept_b],
            "relations": [{"name": relation, "from": concept_a, "to": concept_b}],
            "attributes": []
        }
    
    # Fallback
    return {
        "output_type": "scalar",
        "op": "stats",
        "concepts": [],
        "relations": [],
        "attributes": []
    }
```

### Test the Fix

```python
from src.gqa_nl_parser import NLParser

parser = NLParser()

# Test queries
queries = [
    "How often is a eye to the right of a ear?",
    "What is the probability of finding shirt wearing man?",
    "How frequently is tree to the left of building?"
]

for q in queries:
    result = parser.parse_nl_query(q)
    print(f"Query: {q}")
    print(f"Output type: {result.get('output_type')}")
    print(f"Operation: {result.get('op')}")
    print()
```

---

## Issue #3: Parser Can't Handle Subgraph Queries ❌

### Symptom
```
suite_subgraph:
  Heuristic exact match: 0.0000
  output_type_acc: 0.0000
```

### Root Cause

Parser doesn't recognize "Find images where X is REL Y" as subgraph query.

### Fix

Add detection for relational/evidence queries:

```python
def parse_nl_query(self, nl_query: str) -> Dict[str, Any]:
    """Parse natural language query into structured format."""
    
    nl_lower = nl_query.lower().strip()
    
    # === ADD THIS BLOCK ===
    # Detect subgraph/evidence queries
    subgraph_patterns = [
        r"find images where.*is (to the left of|to the right of|wearing|in|on)",
        r"show pictures with.*is (to the left of|to the right of|wearing|in|on)",
        r"images where.*is (to the left of|to the right of|wearing|in|on)"
    ]
    
    for pattern in subgraph_patterns:
        if re.search(pattern, nl_lower):
            return self._parse_subgraph_query(nl_query)
    # === END NEW BLOCK ===
    
    # ... rest of existing code ...

def _parse_subgraph_query(self, nl_query: str) -> Dict[str, Any]:
    """Parse subgraph queries like 'Find images where A is <rel> B'"""
    
    nl_lower = nl_query.lower()
    
    # Pattern: "where A is <relation> B"
    match = re.search(r"where (?:a |an )?(\w+) is (.*?) (?:a |an )?(\w+)", nl_lower)
    if match:
        concept_a = match.group(1)
        relation = match.group(2).strip()
        concept_b = match.group(3)
        
        return {
            "output_type": "subgraph",
            "op": "retrieve",
            "concepts": [concept_a, concept_b],
            "relations": [{"name": relation, "from": concept_a, "to": concept_b}],
            "attributes": []
        }
    
    # Fallback
    return {
        "output_type": "subgraph",
        "op": "retrieve",
        "concepts": [],
        "relations": [],
        "attributes": []
    }
```

---

## Issue #4: Path Hop Counting Inaccurate ⚠️

### Symptom
```
hop_accuracy: 0.0050
mean_hop_error: 2.7606
validity_rate: 0.7100  # This is OK
```

### Root Cause

Engine finds paths (71% validity) but hop counting differs from oracle.

### Investigation

Check how oracle counts hops vs how engine counts:

**Oracle** (in `oracle_scenegraphs.py`):
```python
# Counts edges in concept graph
shortest_path = nx.shortest_path(concept_graph, source, target)
hops = len(shortest_path) - 1  # Number of edges
```

**Engine** (in `adapter_engine.py`):
```python
# Might be counting instance nodes instead of concept nodes
paths = engine.find_paths(source, target)
# Make sure this returns CONCEPT-level paths
```

### Fix

Ensure adapter counts at concept level:

```python
def execute(self, cqr: CQR) -> Dict[str, Any]:
    if cqr.output_type == OutputType.PATH:
        # ... get paths ...
        
        # Convert instance-level paths to concept-level
        concept_paths = []
        for path in instance_paths:
            concept_path = [self._instance_to_concept(node) for node in path]
            # Remove duplicates (consecutive same concepts)
            concept_path = [concept_path[0]] + [
                concept_path[i] for i in range(1, len(concept_path))
                if concept_path[i] != concept_path[i-1]
            ]
            concept_paths.append(concept_path)
        
        # Count hops at concept level
        shortest_hops = min(len(p) - 1 for p in concept_paths) if concept_paths else None
        
        return {
            "paths": concept_paths,
            "shortest_hops": shortest_hops,
            "exists": len(concept_paths) > 0
        }
```

---

## Issue #5: E2E Evaluation Timeout ⏱️

### Symptom

Process runs for >5 minutes and gets interrupted.

### Cause

Large number of queries × parsing × execution takes time.

### Solutions

#### Option 1: Run Suites Individually

```bash
# Process one suite at a time
for suite in path ranked_entity_attr ranked_negative scalar_stats subgraph; do
    echo "Processing $suite..."
    python -m evaluation_v0_1.scripts.evaluate_e2e --suite $suite
done
```

#### Option 2: Reduce Suite Sizes During Development

Edit `evaluation_v0_1/configs/eval.yaml`:
```yaml
suite_sizes:
  ranked_entity_attr_n: 50   # Reduced from 210
  ranked_negative_n: 100      # Reduced from 500
  scalar_stats_n: 50          # Reduced from 200
  path_n: 50                  # Reduced from 200
  subgraph_n: 50              # Reduced from 200
```

Then regenerate:
```bash
python -m evaluation_v0_1.scripts.run_all --regen --step generate
```

#### Option 3: Optimize Slow Operations

Add caching to adapter:
```python
class EngineAdapter:
    def __init__(self, graph_path: str, config: dict):
        # ... existing code ...
        self._query_cache = {}  # Add cache
    
    def execute(self, cqr: CQR) -> Dict[str, Any]:
        # Check cache first
        cache_key = str(cqr)
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]
        
        # Execute query
        result = self._execute_uncached(cqr)
        
        # Store in cache
        self._query_cache[cache_key] = result
        return result
```

---

## General Debugging Tips

### Enable Verbose Logging

```bash
# Add --verbose to any command
python -m evaluation_v0_1.scripts.run_all --verbose
```

### Check Individual Query Results

```bash
# Look at CSV files for per-query details
cat evaluation_v0_1/results/engine_suite_path.csv | head -20
```

### Verify Suite Generation

```bash
# Check a generated suite file
cat evaluation_v0_1/data/suites/suite_ranked_entity_attr.jsonl | head -1 | python -m json.tool
```

### Test Oracle Directly

```python
from evaluation_v0_1.scripts.oracle_scenegraphs import load_oracle
from evaluation_v0_1.scripts.cqr import CQR, Constraint, OutputType

oracle = load_oracle("sceneGraphs/val_sceneGraphs.json")

# Test query
cqr = CQR(
    query_id="test",
    output_type=OutputType.RANKED_SET,
    op="retrieve",
    must=[Constraint(type="concept", value="car")],
    k=10
)

gold = oracle.get_gold_output(cqr)
print(f"Gold images: {len(gold.image_ids)}")
```

### Validate Normalization

```python
from evaluation_v0_1.scripts.normalize import normalize_term

# Check if normalization is consistent
terms = ["Car", "car", "cars", "automobile"]
for term in terms:
    print(f"{term} → {normalize_term(term)}")
```

---

## Quick Health Check

Run this script to check all components:

```python
#!/usr/bin/env python
"""health_check.py - Verify evaluation framework setup"""

import sys
from pathlib import Path

def check_files():
    """Check all required files exist."""
    required = [
        "evaluation_v0_1/configs/eval.yaml",
        "evaluation_v0_1/data/synonym_map.json",
        "sceneGraphs/val_sceneGraphs.json",
        "experiments/sample_10k/gqa_lightrag.gpickle"
    ]
    
    for path in required:
        if not Path(path).exists():
            print(f"❌ Missing: {path}")
            return False
        print(f"✅ Found: {path}")
    return True

def check_imports():
    """Check all modules can be imported."""
    try:
        from evaluation_v0_1.scripts import cqr
        from evaluation_v0_1.scripts import oracle_scenegraphs
        from evaluation_v0_1.scripts import adapter_engine
        from evaluation_v0_1.scripts import metrics
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def check_oracle():
    """Check oracle can load scene graphs."""
    try:
        from evaluation_v0_1.scripts.oracle_scenegraphs import load_oracle
        oracle = load_oracle("sceneGraphs/val_sceneGraphs.json")
        print(f"✅ Oracle loaded {len(oracle.scenegraphs)} images")
        return True
    except Exception as e:
        print(f"❌ Oracle failed: {e}")
        return False

def check_adapter():
    """Check adapter can load graph."""
    try:
        from evaluation_v0_1.scripts.adapter_engine import create_adapter
        adapter = create_adapter("experiments/sample_10k/gqa_lightrag.gpickle")
        print(f"✅ Adapter loaded graph with {adapter.graph.number_of_nodes()} nodes")
        return True
    except Exception as e:
        print(f"❌ Adapter failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Evaluation Framework Health Check")
    print("=" * 60)
    
    checks = [
        ("Files", check_files),
        ("Imports", check_imports),
        ("Oracle", check_oracle),
        ("Adapter", check_adapter)
    ]
    
    results = []
    for name, check_fn in checks:
        print(f"\nChecking {name}...")
        results.append(check_fn())
    
    print("\n" + "=" * 60)
    if all(results):
        print("✅ All checks passed! System is healthy.")
        sys.exit(0)
    else:
        print("❌ Some checks failed. Fix issues above.")
        sys.exit(1)
```

Run it:
```bash
python health_check.py
```

---

## When to Ask for Help

You should investigate further if:
- ✅ Health check passes but evaluation fails
- ✅ Debug logs show unexpected behavior
- ✅ Results are inconsistent across runs (with same seed)
- ✅ Memory usage keeps growing

You can proceed if:
- ⚠️ Some metrics are 0 (expected during development)
- ⚠️ Parser exact match < 50% (can be improved)
- ⚠️ E2E takes long time (normal for large graphs)

---

**Last Updated**: January 20, 2026  
**Status**: Active debugging guide  
**Next Review**: After implementing fixes
