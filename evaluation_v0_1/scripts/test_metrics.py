"""
Test Metrics for Evaluation Framework v0.1

Lightweight unit tests for metric functions.

Usage:
    python test_metrics.py
    python -m pytest test_metrics.py -v
"""

import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from .metrics import (
    # Ranked set metrics
    precision_at_k, recall_at_k, ndcg_at_k, mrr, average_precision,
    compute_ranked_metrics, aggregate_ranked_metrics,
    # Scalar metrics
    mae, squared_error, compute_scalar_metrics, aggregate_scalar_metrics,
    spearman_correlation,
    # Path metrics
    path_validity, hop_error, hop_accuracy, compute_path_metrics, aggregate_path_metrics,
    # Subgraph metrics
    set_recall, set_precision, set_f1, compactness,
    compute_subgraph_metrics, aggregate_subgraph_metrics,
    # Parser metrics
    compute_parser_metrics, aggregate_parser_metrics
)


def test_precision_at_k():
    """Test precision@k calculation."""
    predicted = ["a", "b", "c", "d", "e"]
    gold = {"a", "c", "e"}

    assert precision_at_k(predicted, gold, 1) == 1.0  # "a" is relevant
    assert precision_at_k(predicted, gold, 2) == 0.5  # 1/2 relevant
    assert precision_at_k(predicted, gold, 3) == 2/3  # 2/3 relevant
    assert precision_at_k(predicted, gold, 5) == 3/5  # 3/5 relevant

    # Edge cases
    assert precision_at_k([], gold, 5) == 0.0
    assert precision_at_k(predicted, set(), 5) == 0.0
    assert precision_at_k(predicted, gold, 0) == 0.0

    print("✓ precision_at_k tests passed")


def test_recall_at_k():
    """Test recall@k calculation."""
    predicted = ["a", "b", "c", "d", "e"]
    gold = {"a", "c", "e", "f"}

    assert recall_at_k(predicted, gold, 1) == 0.25  # 1/4
    assert recall_at_k(predicted, gold, 5) == 0.75  # 3/4

    # Edge cases
    assert recall_at_k([], gold, 5) == 0.0
    assert recall_at_k(predicted, set(), 5) == 0.0

    print("✓ recall_at_k tests passed")


def test_ndcg_at_k():
    """Test NDCG@k calculation."""
    # Perfect ranking
    predicted = ["a", "b", "c"]
    gold = {"a", "b", "c"}
    assert abs(ndcg_at_k(predicted, gold, 3) - 1.0) < 0.001

    # Imperfect ranking
    predicted = ["x", "a", "y", "b", "c"]
    gold = {"a", "b", "c"}
    ndcg_val = ndcg_at_k(predicted, gold, 5)
    assert 0 < ndcg_val < 1.0  # Should be less than perfect

    # Edge cases
    assert ndcg_at_k([], {"a"}, 5) == 0.0

    print("✓ ndcg_at_k tests passed")


def test_mrr():
    """Test MRR calculation."""
    assert mrr(["a", "b", "c"], {"a"}) == 1.0
    assert mrr(["a", "b", "c"], {"b"}) == 0.5
    assert mrr(["a", "b", "c"], {"c"}) == 1/3
    assert mrr(["a", "b", "c"], {"x"}) == 0.0

    print("✓ mrr tests passed")


def test_average_precision():
    """Test average precision calculation."""
    predicted = ["a", "b", "c", "d"]
    gold = {"a", "c"}

    # AP = (1/1 + 2/3) / 2 = (1 + 0.667) / 2 ≈ 0.833
    ap = average_precision(predicted, gold)
    assert 0.8 < ap < 0.9

    # All relevant at top
    predicted = ["a", "b", "c", "d"]
    gold = {"a", "b"}
    ap = average_precision(predicted, gold)
    assert ap == 1.0  # (1/1 + 2/2) / 2 = 1.0

    print("✓ average_precision tests passed")


def test_mae():
    """Test MAE calculation."""
    assert mae(0.5, 0.5) == 0.0
    assert mae(0.7, 0.5) == 0.2
    assert mae(0.3, 0.5) == 0.2

    print("✓ mae tests passed")


def test_squared_error():
    """Test squared error calculation."""
    assert squared_error(0.5, 0.5) == 0.0
    assert abs(squared_error(0.7, 0.5) - 0.04) < 0.001

    print("✓ squared_error tests passed")


def test_spearman_correlation():
    """Test Spearman correlation."""
    # Perfect positive correlation
    x = [1, 2, 3, 4, 5]
    y = [1, 2, 3, 4, 5]
    assert abs(spearman_correlation(x, y) - 1.0) < 0.001

    # Perfect negative correlation
    x = [1, 2, 3, 4, 5]
    y = [5, 4, 3, 2, 1]
    assert abs(spearman_correlation(x, y) - (-1.0)) < 0.001

    # No correlation (roughly)
    x = [1, 2, 3, 4, 5]
    y = [3, 1, 5, 2, 4]
    corr = spearman_correlation(x, y)
    assert -0.5 < corr < 0.5

    print("✓ spearman_correlation tests passed")


def test_path_validity():
    """Test path validity check."""
    assert path_validity(True, True) == 1.0
    assert path_validity(False, False) == 1.0
    assert path_validity(True, False) == 0.0
    assert path_validity(False, True) == 0.0
    assert path_validity(None, False) == 1.0  # None treated as False

    print("✓ path_validity tests passed")


def test_hop_error():
    """Test hop error calculation."""
    assert hop_error(3, 3) == 0
    assert hop_error(5, 3) == 2
    assert hop_error(1, 3) == 2
    assert hop_error(None, 3) is None
    assert hop_error(3, None) is None

    print("✓ hop_error tests passed")


def test_hop_accuracy():
    """Test hop accuracy."""
    assert hop_accuracy(3, 3) == 1.0
    assert hop_accuracy(5, 3) == 0.0
    assert hop_accuracy(None, 3) == 0.0
    assert hop_accuracy(3, None) is None

    print("✓ hop_accuracy tests passed")


def test_set_recall():
    """Test set recall."""
    assert set_recall({"a", "b"}, {"a", "b", "c"}) == 2/3
    assert set_recall({"a", "b", "c"}, {"a", "b"}) == 1.0
    assert set_recall(set(), {"a"}) == 0.0
    assert set_recall({"a"}, set()) == 0.0

    print("✓ set_recall tests passed")


def test_set_precision():
    """Test set precision."""
    assert set_precision({"a", "b"}, {"a", "b", "c"}) == 1.0
    assert set_precision({"a", "b", "c"}, {"a", "b"}) == 2/3
    assert set_precision(set(), {"a"}) == 0.0

    print("✓ set_precision tests passed")


def test_set_f1():
    """Test set F1."""
    # When P=R=1.0, F1=1.0
    assert set_f1({"a", "b"}, {"a", "b"}) == 1.0

    # P=1, R=0.5 -> F1 = 2*1*0.5 / 1.5 = 2/3
    f1 = set_f1({"a"}, {"a", "b"})
    assert abs(f1 - 2/3) < 0.001

    print("✓ set_f1 tests passed")


def test_compactness():
    """Test compactness ratio."""
    assert compactness(5, 5) == 1.0
    assert compactness(10, 5) == 2.0
    assert compactness(2, 5) == 0.4
    assert compactness(0, 5) == 0.0
    assert compactness(5, 0) == 10.0  # Capped at 10

    print("✓ compactness tests passed")


def test_compute_ranked_metrics():
    """Test ranked metrics computation."""
    predicted = ["a", "b", "c", "d", "e"]
    gold = {"a", "c", "e"}
    k_values = [1, 3, 5]

    metrics = compute_ranked_metrics(predicted, gold, k_values)

    assert "precision@1" in metrics
    assert "recall@5" in metrics
    assert "ndcg@3" in metrics
    assert "mrr" in metrics
    assert "ap" in metrics

    print("✓ compute_ranked_metrics tests passed")


def test_compute_scalar_metrics():
    """Test scalar metrics computation."""
    metrics = compute_scalar_metrics(0.7, 0.5)

    assert "mae" in metrics
    assert "squared_error" in metrics
    assert "predicted" in metrics
    assert "gold" in metrics

    assert metrics["mae"] == 0.2

    print("✓ compute_scalar_metrics tests passed")


def test_compute_path_metrics():
    """Test path metrics computation."""
    predicted = {"paths": [["a", "b", "c"]], "shortest_hops": 2, "exists": True}
    gold = {"exists": True, "shortest_hops": 2}

    metrics = compute_path_metrics(predicted, gold)

    assert metrics["validity"] == 1.0
    assert metrics["hop_accuracy"] == 1.0
    assert metrics["hop_error"] == 0

    print("✓ compute_path_metrics tests passed")


def test_compute_subgraph_metrics():
    """Test subgraph metrics computation."""
    predicted = {
        "nodes": ["concept:car", "attr:red"],
        "edges": [["concept:car", "has_attr", "attr:red"]]
    }
    gold = {
        "nodes": ["concept:car", "attr:red", "attr:large"],
        "edges": [["concept:car", "has_attr", "attr:red"],
                 ["concept:car", "has_attr", "attr:large"]]
    }

    metrics = compute_subgraph_metrics(predicted, gold)

    assert "node_recall" in metrics
    assert "edge_recall" in metrics
    assert "compactness_nodes" in metrics

    assert metrics["node_recall"] == 2/3
    assert metrics["edge_recall"] == 0.5

    print("✓ compute_subgraph_metrics tests passed")


def test_compute_parser_metrics():
    """Test parser metrics computation."""
    predicted_cqr = {
        "output_type": "ranked_set",
        "op": "retrieve",
        "must": [
            {"type": "concept", "value": "car"},
            {"type": "attr", "value": "red"}
        ],
        "must_not": []
    }
    gold_cqr = {
        "output_type": "ranked_set",
        "op": "retrieve",
        "must": [
            {"type": "concept", "value": "car"},
            {"type": "attr", "value": "red"}
        ],
        "must_not": []
    }

    metrics = compute_parser_metrics(predicted_cqr, gold_cqr)

    assert metrics["output_type_acc"] == 1.0
    assert metrics["op_acc"] == 1.0
    assert metrics["concept_f1"] == 1.0
    assert metrics["attr_f1"] == 1.0
    assert metrics["exact_match"] == 1.0

    print("✓ compute_parser_metrics tests passed")


def test_aggregation_functions():
    """Test aggregation functions."""
    # Ranked
    results = [
        {"mrr": 1.0, "ap": 1.0, "precision@5": 0.8, "recall@5": 0.6, "ndcg@5": 0.9},
        {"mrr": 0.5, "ap": 0.7, "precision@5": 0.6, "recall@5": 0.4, "ndcg@5": 0.7}
    ]
    agg = aggregate_ranked_metrics(results, [5])
    assert agg["mrr"] == 0.75
    assert agg["map"] == 0.85

    # Scalar
    results = [
        {"mae": 0.1, "squared_error": 0.01, "predicted": 0.5, "gold": 0.4},
        {"mae": 0.2, "squared_error": 0.04, "predicted": 0.7, "gold": 0.5}
    ]
    agg = aggregate_scalar_metrics(results)
    assert agg["mae"] == 0.15

    # Path
    results = [
        {"validity": 1.0, "coverage": 1.0, "hop_error": 0, "hop_accuracy": 1.0},
        {"validity": 1.0, "coverage": 0.0}
    ]
    agg = aggregate_path_metrics(results)
    assert agg["validity_rate"] == 1.0
    assert agg["coverage"] == 0.5

    # Subgraph
    results = [
        {"node_recall": 0.8, "node_precision": 0.9, "node_f1": 0.84,
         "edge_recall": 0.7, "edge_precision": 0.8, "edge_f1": 0.74,
         "compactness_nodes": 1.0, "compactness_edges": 1.0},
    ]
    agg = aggregate_subgraph_metrics(results)
    assert agg["node_recall"] == 0.8

    # Parser
    results = [
        {"output_type_acc": 1.0, "op_acc": 1.0, "concept_f1": 1.0,
         "attr_f1": 1.0, "rel_f1": 1.0, "negation_f1": 1.0, "exact_match": 1.0},
        {"output_type_acc": 1.0, "op_acc": 0.0, "concept_f1": 0.5,
         "attr_f1": 0.5, "rel_f1": 0.0, "negation_f1": 1.0, "exact_match": 0.0}
    ]
    agg = aggregate_parser_metrics(results)
    assert agg["output_type_acc"] == 1.0
    assert agg["op_acc"] == 0.5
    assert agg["exact_match"] == 0.5

    print("✓ aggregation tests passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*50)
    print("Running Metrics Unit Tests")
    print("="*50 + "\n")

    test_precision_at_k()
    test_recall_at_k()
    test_ndcg_at_k()
    test_mrr()
    test_average_precision()
    test_mae()
    test_squared_error()
    test_spearman_correlation()
    test_path_validity()
    test_hop_error()
    test_hop_accuracy()
    test_set_recall()
    test_set_precision()
    test_set_f1()
    test_compactness()
    test_compute_ranked_metrics()
    test_compute_scalar_metrics()
    test_compute_path_metrics()
    test_compute_subgraph_metrics()
    test_compute_parser_metrics()
    test_aggregation_functions()

    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50 + "\n")


if __name__ == "__main__":
    run_all_tests()

