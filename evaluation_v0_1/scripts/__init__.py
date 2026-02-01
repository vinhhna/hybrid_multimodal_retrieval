"""
Evaluation Framework v0.1 Scripts

This package contains all evaluation scripts and modules:
- cqr: Canonical Query Representation
- normalize: Text normalization utilities
- metrics: Evaluation metrics for all output types
- oracle_scenegraphs: Ground truth computation from scene graphs
- generate_suites: Query suite generation
- adapter_engine: Bridge between CQR and the GQA reasoning engine
- llm_parser_stub: Stub for LLM-based parsing
- evaluate_engine: Engine-only evaluation
- evaluate_parser: Parser comparison evaluation
- evaluate_e2e: End-to-end evaluation
- run_all: Master script for complete pipeline
"""

from .cqr import CQR, Constraint, GoldOutput, QuerySuiteItem, OutputType
from .normalize import Normalizer, normalize_term
from .metrics import (
    precision_at_k, recall_at_k, ndcg_at_k, mrr, average_precision,
    compute_ranked_metrics, aggregate_ranked_metrics,
    compute_scalar_metrics, aggregate_scalar_metrics,
    compute_path_metrics, aggregate_path_metrics,
    compute_subgraph_metrics, aggregate_subgraph_metrics,
    compute_parser_metrics, aggregate_parser_metrics
)

__all__ = [
    # CQR
    'CQR', 'Constraint', 'GoldOutput', 'QuerySuiteItem', 'OutputType',
    # Normalization
    'Normalizer', 'normalize_term',
    # Metrics
    'precision_at_k', 'recall_at_k', 'ndcg_at_k', 'mrr', 'average_precision',
    'compute_ranked_metrics', 'aggregate_ranked_metrics',
    'compute_scalar_metrics', 'aggregate_scalar_metrics',
    'compute_path_metrics', 'aggregate_path_metrics',
    'compute_subgraph_metrics', 'aggregate_subgraph_metrics',
    'compute_parser_metrics', 'aggregate_parser_metrics'
]

