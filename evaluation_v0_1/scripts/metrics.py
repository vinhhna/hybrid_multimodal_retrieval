"""
Evaluation Metrics for Evaluation Framework v0.1

Implements metrics for all 4 output types:
- ranked_set: P@k, R@k, NDCG@k, MRR, constraint satisfaction
- scalar: MAE, RMSE, Spearman correlation
- path: validity rate, hop accuracy, hop error, coverage
- subgraph: node/edge recall, compactness
"""

import math
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict


# =============================================================================
# RANKED SET METRICS
# =============================================================================

def precision_at_k(predicted: List[str], gold: Set[str], k: int) -> float:
    """
    Precision@k: fraction of top-k predictions that are relevant.

    Args:
        predicted: Ranked list of predicted image_ids
        gold: Set of relevant image_ids
        k: Cutoff

    Returns:
        Precision@k value in [0, 1]
    """
    if k <= 0:
        return 0.0

    top_k = predicted[:k]
    if not top_k:
        return 0.0

    relevant = sum(1 for item in top_k if item in gold)
    return relevant / len(top_k)


def recall_at_k(predicted: List[str], gold: Set[str], k: int) -> float:
    """
    Recall@k: fraction of relevant items found in top-k.

    Args:
        predicted: Ranked list of predicted image_ids
        gold: Set of relevant image_ids
        k: Cutoff

    Returns:
        Recall@k value in [0, 1]
    """
    if not gold:
        return 1.0 if not predicted else 0.0

    top_k = set(predicted[:k])
    relevant = len(top_k & gold)
    return relevant / len(gold)


def dcg_at_k(predicted: List[str], gold: Set[str], k: int) -> float:
    """
    Discounted Cumulative Gain at k.
    Uses binary relevance (1 if in gold, 0 otherwise).
    """
    dcg = 0.0
    for i, item in enumerate(predicted[:k]):
        if item in gold:
            # Binary relevance = 1, log base 2
            dcg += 1.0 / math.log2(i + 2)  # i+2 because i is 0-indexed
    return dcg


def ideal_dcg_at_k(gold_size: int, k: int) -> float:
    """
    Ideal DCG at k (all relevant items ranked first).
    """
    idcg = 0.0
    for i in range(min(gold_size, k)):
        idcg += 1.0 / math.log2(i + 2)
    return idcg


def ndcg_at_k(predicted: List[str], gold: Set[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at k.

    Args:
        predicted: Ranked list of predicted image_ids
        gold: Set of relevant image_ids
        k: Cutoff

    Returns:
        NDCG@k value in [0, 1]
    """
    if not gold:
        return 1.0 if not predicted else 0.0

    dcg = dcg_at_k(predicted, gold, k)
    idcg = ideal_dcg_at_k(len(gold), k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def mrr(predicted: List[str], gold: Set[str]) -> float:
    """
    Mean Reciprocal Rank.
    Returns 1/rank of first relevant item, or 0 if none found.

    Args:
        predicted: Ranked list of predicted image_ids
        gold: Set of relevant image_ids

    Returns:
        MRR value in [0, 1]
    """
    for i, item in enumerate(predicted):
        if item in gold:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(predicted: List[str], gold: Set[str]) -> float:
    """
    Average Precision (AP).

    Args:
        predicted: Ranked list of predicted image_ids
        gold: Set of relevant image_ids

    Returns:
        AP value in [0, 1]
    """
    if not gold:
        return 1.0 if not predicted else 0.0

    num_relevant = 0
    precision_sum = 0.0

    for i, item in enumerate(predicted):
        if item in gold:
            num_relevant += 1
            precision_sum += num_relevant / (i + 1)

    return precision_sum / len(gold) if gold else 0.0


def compute_ranked_metrics(predicted: List[str], gold: Set[str],
                          k_values: List[int]) -> Dict[str, float]:
    """
    Compute all ranking metrics for a single query.

    Args:
        predicted: Ranked list of predicted image_ids
        gold: Set of relevant image_ids
        k_values: List of k values for P@k, R@k, NDCG@k

    Returns:
        Dictionary of metric_name -> value
    """
    metrics = {
        "mrr": mrr(predicted, gold),
        "ap": average_precision(predicted, gold),
        "gold_size": len(gold),
        "pred_size": len(predicted)
    }

    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(predicted, gold, k)
        metrics[f"recall@{k}"] = recall_at_k(predicted, gold, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(predicted, gold, k)

    return metrics


# =============================================================================
# SCALAR METRICS
# =============================================================================

def mae(predicted: float, gold: float) -> float:
    """Mean Absolute Error for a single prediction."""
    return abs(predicted - gold)


def squared_error(predicted: float, gold: float) -> float:
    """Squared error for a single prediction."""
    return (predicted - gold) ** 2


def rmse_single(predicted: float, gold: float) -> float:
    """RMSE for a single prediction (equals absolute error)."""
    return abs(predicted - gold)


def compute_scalar_metrics(predicted: float, gold: float) -> Dict[str, float]:
    """
    Compute scalar metrics for a single query.

    Args:
        predicted: Predicted scalar value
        gold: Gold scalar value

    Returns:
        Dictionary of metric_name -> value
    """
    return {
        "mae": mae(predicted, gold),
        "squared_error": squared_error(predicted, gold),
        "predicted": predicted,
        "gold": gold,
        "error": predicted - gold
    }


def aggregate_scalar_metrics(results: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate scalar metrics across multiple queries.

    Args:
        results: List of per-query metric dicts

    Returns:
        Aggregated metrics including RMSE and Spearman
    """
    if not results:
        return {"mae": 0.0, "rmse": 0.0, "count": 0}

    n = len(results)
    mae_sum = sum(r["mae"] for r in results)
    se_sum = sum(r["squared_error"] for r in results)

    aggregated = {
        "mae": mae_sum / n,
        "rmse": math.sqrt(se_sum / n),
        "count": n
    }

    # Spearman correlation
    predicted_values = [r["predicted"] for r in results]
    gold_values = [r["gold"] for r in results]
    aggregated["spearman"] = spearman_correlation(predicted_values, gold_values)

    return aggregated


def spearman_correlation(x: List[float], y: List[float]) -> float:
    """
    Compute Spearman rank correlation coefficient.

    Args:
        x: First list of values
        y: Second list of values

    Returns:
        Spearman correlation in [-1, 1]
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    n = len(x)

    # Compute ranks
    def rank(values):
        sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0] * len(values)
        for rank_val, idx in enumerate(sorted_indices):
            ranks[idx] = rank_val + 1
        return ranks

    rank_x = rank(x)
    rank_y = rank(y)

    # Compute Spearman correlation using Pearson on ranks
    mean_x = sum(rank_x) / n
    mean_y = sum(rank_y) / n

    numerator = sum((rx - mean_x) * (ry - mean_y) for rx, ry in zip(rank_x, rank_y))
    denom_x = math.sqrt(sum((rx - mean_x) ** 2 for rx in rank_x))
    denom_y = math.sqrt(sum((ry - mean_y) ** 2 for ry in rank_y))

    if denom_x == 0 or denom_y == 0:
        return 0.0

    return numerator / (denom_x * denom_y)


# =============================================================================
# PATH METRICS
# =============================================================================

def path_validity(predicted_exists: Optional[bool], gold_exists: bool) -> float:
    """
    Check if predicted path existence matches gold.

    Args:
        predicted_exists: Whether engine found a path (None = no prediction)
        gold_exists: Whether path exists in gold

    Returns:
        1.0 if match, 0.0 otherwise
    """
    if predicted_exists is None:
        predicted_exists = False
    return 1.0 if predicted_exists == gold_exists else 0.0


def hop_error(predicted_hops: Optional[int], gold_hops: Optional[int]) -> Optional[float]:
    """
    Compute absolute hop count error.

    Args:
        predicted_hops: Predicted shortest path length
        gold_hops: Gold shortest path length

    Returns:
        Absolute difference, or None if not applicable
    """
    if gold_hops is None or predicted_hops is None:
        return None
    return abs(predicted_hops - gold_hops)


def hop_accuracy(predicted_hops: Optional[int], gold_hops: Optional[int]) -> Optional[float]:
    """
    Check if predicted hop count exactly matches gold.

    Args:
        predicted_hops: Predicted shortest path length
        gold_hops: Gold shortest path length

    Returns:
        1.0 if exact match, 0.0 otherwise, None if not applicable
    """
    if gold_hops is None:
        return None
    if predicted_hops is None:
        return 0.0
    return 1.0 if predicted_hops == gold_hops else 0.0


def compute_path_metrics(predicted: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute path metrics for a single query.

    Args:
        predicted: {"paths": [...], "shortest_hops": int|None, "exists": bool}
        gold: {"exists": bool, "shortest_hops": int|None}

    Returns:
        Dictionary of metric_name -> value
    """
    pred_exists = predicted.get("exists", len(predicted.get("paths", [])) > 0)
    gold_exists = gold.get("exists", False)

    pred_hops = predicted.get("shortest_hops")
    gold_hops = gold.get("shortest_hops")

    metrics = {
        "validity": path_validity(pred_exists, gold_exists),
        "predicted_exists": pred_exists,
        "gold_exists": gold_exists,
        "predicted_hops": pred_hops,
        "gold_hops": gold_hops,
    }

    h_err = hop_error(pred_hops, gold_hops)
    h_acc = hop_accuracy(pred_hops, gold_hops)

    if h_err is not None:
        metrics["hop_error"] = h_err
    if h_acc is not None:
        metrics["hop_accuracy"] = h_acc

    # Coverage: did engine return at least one path?
    metrics["coverage"] = 1.0 if pred_exists else 0.0

    return metrics


def aggregate_path_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregate path metrics across multiple queries.
    """
    if not results:
        return {"validity_rate": 0.0, "coverage": 0.0, "count": 0}

    n = len(results)
    validity_sum = sum(r["validity"] for r in results)
    coverage_sum = sum(r["coverage"] for r in results)

    # Hop metrics only for queries where gold_exists
    hop_errors = [r["hop_error"] for r in results if "hop_error" in r]
    hop_accs = [r["hop_accuracy"] for r in results if "hop_accuracy" in r]

    aggregated = {
        "validity_rate": validity_sum / n,
        "coverage": coverage_sum / n,
        "count": n
    }

    if hop_errors:
        aggregated["mean_hop_error"] = sum(hop_errors) / len(hop_errors)
    if hop_accs:
        aggregated["hop_accuracy"] = sum(hop_accs) / len(hop_accs)

    return aggregated


# =============================================================================
# SUBGRAPH METRICS
# =============================================================================

def set_recall(predicted: Set[str], gold: Set[str]) -> float:
    """
    Recall of predicted set against gold set.

    Args:
        predicted: Set of predicted items
        gold: Set of gold items

    Returns:
        Recall in [0, 1]
    """
    if not gold:
        return 1.0 if not predicted else 0.0
    return len(predicted & gold) / len(gold)


def set_precision(predicted: Set[str], gold: Set[str]) -> float:
    """
    Precision of predicted set against gold set.
    """
    if not predicted:
        return 1.0 if not gold else 0.0
    return len(predicted & gold) / len(predicted)


def set_f1(predicted: Set[str], gold: Set[str]) -> float:
    """
    F1 score of predicted set against gold set.
    """
    p = set_precision(predicted, gold)
    r = set_recall(predicted, gold)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def compactness(predicted_size: int, gold_size: int) -> float:
    """
    Compactness ratio: predicted_size / gold_size.
    Values < 1 mean more compact, > 1 means bloated.

    Args:
        predicted_size: Size of predicted set
        gold_size: Size of gold set

    Returns:
        Ratio (capped at 10 for numerical stability)
    """
    if gold_size == 0:
        return 0.0 if predicted_size == 0 else 10.0
    return min(predicted_size / gold_size, 10.0)


def compute_subgraph_metrics(predicted: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute subgraph metrics for a single query.

    Args:
        predicted: {"nodes": [...], "edges": [...]}
        gold: {"nodes": [...], "edges": [...]}

    Returns:
        Dictionary of metric_name -> value
    """
    pred_nodes = set(predicted.get("nodes", []))
    gold_nodes = set(gold.get("nodes", []))

    # Convert edge lists to sets of tuples
    pred_edges = set(tuple(e) for e in predicted.get("edges", []))
    gold_edges = set(tuple(e) for e in gold.get("edges", []))

    metrics = {
        "node_recall": set_recall(pred_nodes, gold_nodes),
        "node_precision": set_precision(pred_nodes, gold_nodes),
        "node_f1": set_f1(pred_nodes, gold_nodes),
        "edge_recall": set_recall(pred_edges, gold_edges),
        "edge_precision": set_precision(pred_edges, gold_edges),
        "edge_f1": set_f1(pred_edges, gold_edges),
        "compactness_nodes": compactness(len(pred_nodes), len(gold_nodes)),
        "compactness_edges": compactness(len(pred_edges), len(gold_edges)),
        "pred_node_count": len(pred_nodes),
        "gold_node_count": len(gold_nodes),
        "pred_edge_count": len(pred_edges),
        "gold_edge_count": len(gold_edges)
    }

    return metrics


def aggregate_subgraph_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregate subgraph metrics across multiple queries.
    """
    if not results:
        return {"node_recall": 0.0, "edge_recall": 0.0, "count": 0}

    n = len(results)

    aggregated = {
        "node_recall": sum(r["node_recall"] for r in results) / n,
        "node_precision": sum(r["node_precision"] for r in results) / n,
        "node_f1": sum(r["node_f1"] for r in results) / n,
        "edge_recall": sum(r["edge_recall"] for r in results) / n,
        "edge_precision": sum(r["edge_precision"] for r in results) / n,
        "edge_f1": sum(r["edge_f1"] for r in results) / n,
        "mean_compactness_nodes": sum(r["compactness_nodes"] for r in results) / n,
        "mean_compactness_edges": sum(r["compactness_edges"] for r in results) / n,
        "count": n
    }

    return aggregated


# =============================================================================
# PARSER METRICS (CQR comparison)
# =============================================================================

def output_type_accuracy(predicted: str, gold: str) -> float:
    """Check if predicted output_type matches gold."""
    return 1.0 if predicted == gold else 0.0


def op_accuracy(predicted: str, gold: str) -> float:
    """Check if predicted op matches gold."""
    return 1.0 if predicted == gold else 0.0


def slot_f1(predicted_slots: List[str], gold_slots: List[str]) -> Dict[str, float]:
    """
    Compute F1 for a slot (concepts, attrs, rels).

    Args:
        predicted_slots: List of predicted values
        gold_slots: List of gold values

    Returns:
        Dict with precision, recall, f1
    """
    pred_set = set(predicted_slots)
    gold_set = set(gold_slots)

    return {
        "precision": set_precision(pred_set, gold_set),
        "recall": set_recall(pred_set, gold_set),
        "f1": set_f1(pred_set, gold_set)
    }


def compute_parser_metrics(predicted_cqr: Dict[str, Any],
                          gold_cqr: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute parser accuracy metrics.

    Args:
        predicted_cqr: Predicted CQR dict
        gold_cqr: Gold CQR dict

    Returns:
        Dictionary of metric_name -> value
    """
    metrics = {
        "output_type_acc": output_type_accuracy(
            predicted_cqr.get("output_type", ""),
            gold_cqr.get("output_type", "")
        ),
        "op_acc": op_accuracy(
            predicted_cqr.get("op", ""),
            gold_cqr.get("op", "")
        )
    }

    # Extract slots from constraints
    def extract_concepts(cqr):
        return [c["value"] for c in cqr.get("must", []) if c["type"] == "concept"]

    def extract_attrs(cqr):
        return [c["value"] for c in cqr.get("must", []) if c["type"] == "attr"]

    def extract_rels(cqr):
        rels = []
        for c in cqr.get("must", []):
            if c["type"] == "rel":
                rels.append(f"{c['value']['name']}:{c['value']['obj']}")
        return rels

    def extract_negations(cqr):
        return [c["value"] for c in cqr.get("must_not", []) if c["type"] == "concept"]

    # Slot F1s
    concept_f1 = slot_f1(extract_concepts(predicted_cqr), extract_concepts(gold_cqr))
    metrics["concept_precision"] = concept_f1["precision"]
    metrics["concept_recall"] = concept_f1["recall"]
    metrics["concept_f1"] = concept_f1["f1"]

    attr_f1 = slot_f1(extract_attrs(predicted_cqr), extract_attrs(gold_cqr))
    metrics["attr_precision"] = attr_f1["precision"]
    metrics["attr_recall"] = attr_f1["recall"]
    metrics["attr_f1"] = attr_f1["f1"]

    rel_f1 = slot_f1(extract_rels(predicted_cqr), extract_rels(gold_cqr))
    metrics["rel_precision"] = rel_f1["precision"]
    metrics["rel_recall"] = rel_f1["recall"]
    metrics["rel_f1"] = rel_f1["f1"]

    neg_f1 = slot_f1(extract_negations(predicted_cqr), extract_negations(gold_cqr))
    metrics["negation_precision"] = neg_f1["precision"]
    metrics["negation_recall"] = neg_f1["recall"]
    metrics["negation_f1"] = neg_f1["f1"]

    # Exact match (all slots match)
    exact = (
        metrics["output_type_acc"] == 1.0 and
        metrics["concept_f1"] == 1.0 and
        metrics["attr_f1"] == 1.0 and
        metrics["rel_f1"] == 1.0 and
        metrics["negation_f1"] == 1.0
    )
    metrics["exact_match"] = 1.0 if exact else 0.0

    return metrics


# =============================================================================
# AGGREGATION UTILITIES
# =============================================================================

def aggregate_ranked_metrics(results: List[Dict[str, float]],
                            k_values: List[int]) -> Dict[str, float]:
    """
    Aggregate ranking metrics across multiple queries.
    """
    if not results:
        return {"count": 0}

    n = len(results)
    aggregated = {"count": n}

    # Average MRR and AP
    aggregated["mrr"] = sum(r["mrr"] for r in results) / n
    aggregated["map"] = sum(r["ap"] for r in results) / n

    # Average P@k, R@k, NDCG@k
    for k in k_values:
        aggregated[f"precision@{k}"] = sum(r.get(f"precision@{k}", 0) for r in results) / n
        aggregated[f"recall@{k}"] = sum(r.get(f"recall@{k}", 0) for r in results) / n
        aggregated[f"ndcg@{k}"] = sum(r.get(f"ndcg@{k}", 0) for r in results) / n

    return aggregated


def aggregate_parser_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregate parser metrics across multiple queries.
    """
    if not results:
        return {"count": 0}

    n = len(results)

    aggregated = {
        "count": n,
        "output_type_acc": sum(r["output_type_acc"] for r in results) / n,
        "op_acc": sum(r["op_acc"] for r in results) / n,
        "concept_f1": sum(r["concept_f1"] for r in results) / n,
        "attr_f1": sum(r["attr_f1"] for r in results) / n,
        "rel_f1": sum(r["rel_f1"] for r in results) / n,
        "negation_f1": sum(r["negation_f1"] for r in results) / n,
        "exact_match": sum(r["exact_match"] for r in results) / n
    }

    return aggregated

