# PR: Evaluation Framework Upgrade (v0.1.1)

## Summary

This PR upgrades the evaluation framework to academic standards with proper data splits, valid metrics, runnable baselines, and advanced query evaluation.

---

## Changes Overview

### Task A: Data Splits ✅

**Problem**: Evaluation only ran on training data.

**Solution**: Added train/val split support.

| File | Change |
|------|--------|
| `evaluation_v0_1/configs/eval.yaml` | Added `data_splits` configuration section |
| `evaluation_v0_1/scripts/run_all.py` | Added `--split` CLI argument |

**Usage**:
```bash
python -m evaluation_v0_1.run_evaluation --split val --verbose
python -m evaluation_v0_1.run_evaluation --split train --verbose
```

---

### Task B: Metrics Validity ✅

**Problem**: NDCG/MRR/MAP were used on unranked entity search outputs (invalid).

**Solution**: Added set-based metrics; documented which metrics apply to which output types.

| File | Change |
|------|--------|
| `evaluation_v0_1/scripts/metrics.py` | Added `exact_match()`, `jaccard_similarity()`, `compute_set_metrics()`, `aggregate_set_metrics()` |
| `evaluation_v0_1/configs/eval.yaml` | Added `metrics.output_semantics` documentation |
| `docs/evaluation_protocol.md` | Rewrote Section 4 with metrics justification |

**Valid Metrics by Query Type**:

| Query Type | Output | Metrics |
|------------|--------|---------|
| Entity Search | Unranked Set | EM, P, R, F1, Jaccard |
| Negative Constraints | Unranked Set | EM, P, R, F1, Jaccard |
| Statistical | Scalar | MAE, RMSE, Spearman |
| Relational Path | Path | Validity, Hop Accuracy |
| Subgraph | Graph | Node/Edge P/R/F1 |

---

### Task C: Baselines ✅

**Problem**: No comparison methods (baselines) were implemented.

**Solution**: Created 3 runnable baselines with unified interface.

| File | Change |
|------|--------|
| `evaluation_v0_1/scripts/baselines.py` | **New file** with `NaiveScanBaseline`, `RelationWalkBaseline`, `TrivialParserBaseline`, `BaselineAdapter` |
| `evaluation_v0_1/scripts/run_all.py` | Added `run_evaluate_baselines()`, `--method` CLI argument |

**Baselines**:

| Baseline | Description | Expected Performance |
|----------|-------------|---------------------|
| `baseline_naive_scan` | Direct scene graph scan (no KG) | High recall, low precision |
| `baseline_relation_walk` | Within-image BFS traversal | Moderate (single-hop only) |
| `baseline_parser_trivial` | Regex keyword extraction | Low (no semantic understanding) |

**Usage**:
```bash
python -m evaluation_v0_1.run_evaluation --method baseline_naive_scan --split val
python -m evaluation_v0_1.run_evaluation --method baseline_relation_walk --split val
```

---

### Task D: Advanced Query Evaluation ✅

**Problem**: Query types 6-9 (chain_reasoning, pattern_matching, scene_comparison, counterfactual) had no evaluation.

**Solution**: Created curated toy suites and evaluation runner.

| File | Change |
|------|--------|
| `evaluation_v0_1/data/suites/advanced_toy/suite_chain_reasoning.json` | **New file** - 5 curated queries |
| `evaluation_v0_1/data/suites/advanced_toy/suite_pattern_matching.json` | **New file** - 5 curated queries |
| `evaluation_v0_1/data/suites/advanced_toy/suite_scene_comparison.json` | **New file** - 3 curated queries |
| `evaluation_v0_1/data/suites/advanced_toy/suite_counterfactual.json` | **New file** - 4 curated queries |
| `evaluation_v0_1/scripts/evaluate_advanced.py` | **New file** - `AdvancedQueryEvaluator`, `TraceQualityRubric` |
| `evaluation_v0_1/scripts/run_all.py` | Added `run_evaluate_advanced()`, `--step advanced` support |

**Usage**:
```bash
python -m evaluation_v0_1.run_evaluation --step advanced --verbose
```

**Trace Quality Rubric** (placeholder for future LLM-based grading):

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Step Correctness | 40% | Each reasoning step is valid |
| Completeness | 30% | All required steps present |
| Coherence | 20% | Logical flow between steps |
| Conciseness | 10% | No redundant steps |

---

### Reproducibility ✅

| File | Change |
|------|--------|
| `evaluation_v0_1/configs/eval.yaml` | Added `reproducibility` section with `random_seed`, `timestamped_outputs` |
| `evaluation_v0_1/scripts/run_all.py` | Added `--seed` CLI argument, `get_output_dir()` for timestamped folders |
| `artifacts/` | Created output directory (gitignored) |

**Usage**:
```bash
# Fully reproducible run
python -m evaluation_v0_1.run_evaluation --step all --split val --seed 1337 --verbose
```

**Output Structure**:
```
artifacts/
└── 20250115_143022_val_main/
    ├── summary.json
    ├── summary.md
    ├── engine_suite_*.csv
    └── advanced_*.json
```

---

### Documentation Updates ✅

| File | Change |
|------|--------|
| `docs/evaluation_protocol.md` | Rewrote header, added design decisions, metrics justification, baseline documentation |
| `docs/limitations_and_failure_modes.md` | Updated Issue 5.1 (resolved), added Section 13 (resolved/remaining) |
| `README.md` | Added Section 8 (Evaluation v0.1.1), updated Section 10 (Change Log) |

---

## Files Changed Summary

| Category | Files Added | Files Modified |
|----------|-------------|----------------|
| Scripts | 2 (`baselines.py`, `evaluate_advanced.py`) | 2 (`run_all.py`, `metrics.py`) |
| Configuration | 0 | 1 (`eval.yaml`) |
| Data | 4 (advanced_toy suites) | 0 |
| Documentation | 1 (`EVALUATION_UPGRADE_PR.md`) | 3 (`evaluation_protocol.md`, `limitations.md`, `README.md`) |
| Infrastructure | 1 (`artifacts/.gitkeep`) | 0 |

---

## Testing Commands

```bash
# 1. Verify installation
pip install -e .

# 2. Build graph (if not exists)
lightrag-gqa-build --scale 10k

# 3. Run main evaluation on val
python -m evaluation_v0_1.run_evaluation --split val --verbose

# 4. Run baselines
python -m evaluation_v0_1.run_evaluation --method baseline_naive_scan --split val

# 5. Run advanced evaluation
python -m evaluation_v0_1.run_evaluation --step advanced --verbose

# 6. Check outputs
ls artifacts/
cat evaluation_v0_1/results/e2e_summary.md
```

---

## Metrics Validity Justification

### Why NOT use NDCG/MRR/MAP for Entity Search:

1. **NDCG** (Normalized Discounted Cumulative Gain): Requires relevance scores and ranked ordering. Our entity search returns unranked sets.
2. **MRR** (Mean Reciprocal Rank): Requires ranked list with single correct answer. Entity search may have multiple valid entities.
3. **MAP** (Mean Average Precision): Requires ranked ordering which we don't produce.

### Why SET metrics are appropriate:

- **Exact Match (EM)**: Binary correctness (1 if prediction == gold, else 0)
- **Precision**: What fraction of predicted entities are correct
- **Recall**: What fraction of gold entities were found
- **F1**: Harmonic mean of P and R
- **Jaccard**: |intersection| / |union| - symmetric similarity

See [evaluation_protocol.md](docs/evaluation_protocol.md) Section 4 for full justification.

---

## Known Limitations

1. **Advanced Query Suites**: Only toy examples (17 total queries); not scaled to val set yet
2. **Trace Quality Rubric**: Placeholder implementation; requires LLM for proper grading
3. **Baseline Performance**: Not benchmarked yet; expected to underperform main method
4. **Cross-validation**: Not implemented (would require k-fold splits)

---

## Backward Compatibility

- All existing CLI commands work unchanged
- New features are additive (opt-in via flags)
- Default behavior (`--split train`, `--method main`) matches previous version
