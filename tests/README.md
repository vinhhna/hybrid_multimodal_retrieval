# Query Enrichment Tests

This directory contains tests for Phase 4 query enrichment implementation.

## Running Tests

### Prerequisites

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Run Unit Tests

From the project root:
```bash
pytest tests/test_query_enrichment.py -v
```

Or run all tests:
```bash
pytest tests/ -v
```

### Run Demo Script

From the project root:
```bash
python scripts/demo_query_enrichment.py
```

This demonstrates query enrichment with synthetic data (no dataset required).

### Run Integration Test (requires full environment)

From the project root:
```bash
python scripts/test_integration_enrichment.py
```

Note: This requires the full environment with CLIP, BLIP-2, and FAISS installed.

## Test Coverage

### `test_query_enrichment.py`

**Helper Functions:**
- `test_l2_normalize_1d()` - Test 1D L2 normalization
- `test_l2_normalize_2d()` - Test 2D L2 normalization
- `test_collect_candidate_entities()` - Test entity collection from seeds
- `test_score_entities()` - Test entity scoring by frequency + similarity
- `test_top_k_entities()` - Test top-k selection with tie-breaking
- `test_top_k_entities_empty()` - Test edge case with empty input

**Main Function:**
- `test_enrich_query_basic()` - Basic enrichment with synthetic data
- `test_enrich_query_no_entities()` - Test when no entities found
- `test_enrich_query_determinism()` - Test reproducibility
- `test_enrich_query_scores_sorted()` - Test score ordering

**Fixtures:**
- `synthetic_entity_context()` - Mock entity context (4 entities)
- `synthetic_entity_embeddings()` - Mock embeddings (4×512, L2-normalized)
- `synthetic_entity_meta()` - Mock metadata
- `synthetic_config()` - Mock configuration
- `mock_encoder()` - Mock BiEncoder (deterministic)
- `synthetic_seeds()` - Mock CLIP search seeds

## Test Strategy

All tests use **synthetic fixtures** to avoid dependencies on:
- Real Flickr30K dataset
- Pre-trained CLIP model
- Entity graph files

This allows tests to run:
- Fast (<1s for all tests)
- Deterministically
- Without large downloads

## Expected Output

```
$ pytest tests/test_query_enrichment.py -v

tests/test_query_enrichment.py::test_l2_normalize_1d PASSED
tests/test_query_enrichment.py::test_l2_normalize_2d PASSED
tests/test_query_enrichment.py::test_collect_candidate_entities PASSED
tests/test_query_enrichment.py::test_score_entities PASSED
tests/test_query_enrichment.py::test_top_k_entities PASSED
tests/test_query_enrichment.py::test_top_k_entities_empty PASSED
tests/test_query_enrichment.py::test_enrich_query_basic PASSED
tests/test_query_enrichment.py::test_enrich_query_no_entities PASSED
tests/test_query_enrichment.py::test_enrich_query_determinism PASSED
tests/test_query_enrichment.py::test_enrich_query_scores_sorted PASSED

========== 10 passed in 0.50s ==========
```

## Troubleshooting

### ModuleNotFoundError

If you see import errors, ensure `src/` is in your Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

Or run pytest from project root with the `-p` flag:
```bash
pytest -p no:cacheprovider tests/test_query_enrichment.py -v
```

### Missing Dependencies

Install all requirements:
```bash
pip install -r requirements.txt
```

For pytest specifically:
```bash
pip install pytest>=7.3.0
```

## Related Files

- `src/graph/graph_search.py` - Main implementation
- `src/graph/config.py` - Configuration helpers
- `configs/entity_graph.yaml` - Configuration file
- `QUERY_ENRICHMENT_SUMMARY.md` - Implementation summary
