"""
Engine Adapter for Evaluation Framework v0.1

Bridges CQR (Canonical Query Representation) to the existing GQA query engine.
This adapter translates CQR into engine calls and standardizes outputs.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass

# Add repo root to path for imports
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from .cqr import CQR, Constraint, OutputType
from .normalize import Normalizer


@dataclass
class EngineResult:
    """Standardized result from engine execution."""
    success: bool
    output_type: str
    data: Dict[str, Any]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output_type": self.output_type,
            "data": self.data,
            "error": self.error
        }


class EngineAdapter:
    """
    Adapter that bridges CQR to the GQA reasoning engine.

    Supports:
    - ranked_set: entity_search, negative_constraints
    - scalar: statistical_knowledge
    - path: relational_path
    - subgraph: entity_search with evidence extraction
    """

    def __init__(self, graph_path: Optional[str] = None,
                 scale: str = '10k',
                 config: Optional[Dict[str, Any]] = None,
                 verbose: bool = False):
        """
        Initialize the engine adapter.

        Args:
            graph_path: Explicit path to graph pickle (overrides scale)
            scale: Scale to use if graph_path not provided ('1k', '10k', 'full')
            config: Configuration dictionary
            verbose: Whether to print status messages
        """
        self.config = config or {}
        self.verbose = verbose
        self.scale = scale

        # Import engine components
        try:
            from basic_queries.reasoning_engine import GQA_Reasoning_Engine
            from basic_queries.gqa_query_interface import QueryInterface
        except ImportError as e:
            raise ImportError(f"Failed to import engine modules: {e}")

        # Initialize engine
        if graph_path:
            if verbose:
                print(f"[Adapter] Loading graph from: {graph_path}")
            self.engine = GQA_Reasoning_Engine(graph_path=graph_path)
        else:
            if verbose:
                print(f"[Adapter] Loading graph with scale: {scale}")
            self.engine = GQA_Reasoning_Engine(scale=scale)

        # Store reference to graph
        self.graph = self.engine.graph

        if verbose:
            print(f"[Adapter] Graph loaded: {self.graph.number_of_nodes()} nodes, "
                  f"{self.graph.number_of_edges()} edges")

    def execute(self, cqr: CQR) -> EngineResult:
        """
        Execute a CQR and return standardized results.

        Args:
            cqr: Canonical Query Representation

        Returns:
            EngineResult with standardized output
        """
        output_type = cqr.output_type

        try:
            if output_type == OutputType.RANKED_SET.value:
                return self._execute_ranked_set(cqr)
            elif output_type == OutputType.SCALAR.value:
                return self._execute_scalar(cqr)
            elif output_type == OutputType.PATH.value:
                return self._execute_path(cqr)
            elif output_type == OutputType.SUBGRAPH.value:
                return self._execute_subgraph(cqr)
            else:
                return EngineResult(
                    success=False,
                    output_type=output_type,
                    data={},
                    error=f"Unknown output_type: {output_type}"
                )
        except Exception as e:
            return EngineResult(
                success=False,
                output_type=output_type,
                data={},
                error=str(e)
            )

    def _execute_ranked_set(self, cqr: CQR) -> EngineResult:
        """
        Execute ranked_set query.

        Maps to:
        - entity_search (concept + attributes)
        - negative_constraints (with must_not)
        """
        concepts = cqr.get_concepts()
        attrs = cqr.get_attributes()
        rels = cqr.get_relations()
        neg_concepts = cqr.get_negative_concepts()

        k = cqr.k

        # Determine which engine method to use
        if neg_concepts:
            # Use negative_constraints
            if not concepts:
                return EngineResult(
                    success=False,
                    output_type=OutputType.RANKED_SET.value,
                    data={"image_ids": [], "scores": []},
                    error="Negative query requires a must concept"
                )

            result = self.engine.negative_constraints(
                concept_present=concepts[0],
                concept_absent=neg_concepts[0],
                limit=k
            )

            # Extract image_ids from results
            image_ids = self._extract_image_ids(result.results)

            return EngineResult(
                success=True,
                output_type=OutputType.RANKED_SET.value,
                data={
                    "image_ids": image_ids,
                    "scores": [1.0] * len(image_ids),  # Binary relevance
                    "reasoning_trace": result.reasoning_trace
                }
            )

        elif rels:
            # Relation query - use statistical knowledge to find images
            # and then filter by the relation
            concept = concepts[0] if concepts else None
            rel = rels[0]
            rel_name = rel.get("name", "")
            rel_obj = rel.get("obj", "")

            # Try to find images with this relation pattern
            # Using a combination of entity search and relation filtering
            if concept:
                result = self.engine.entity_search(
                    concept=concept,
                    attributes=attrs,
                    limit=k * 2  # Get more to filter
                )

                # Filter by relation if possible
                image_ids = self._extract_image_ids(result.results)
                # Note: Full relation filtering would require graph traversal
                # For now, return entity search results

                return EngineResult(
                    success=True,
                    output_type=OutputType.RANKED_SET.value,
                    data={
                        "image_ids": image_ids[:k],
                        "scores": [1.0] * min(len(image_ids), k),
                        "reasoning_trace": result.reasoning_trace
                    }
                )

        else:
            # Standard entity search
            concept = concepts[0] if concepts else None

            if not concept and not attrs:
                return EngineResult(
                    success=False,
                    output_type=OutputType.RANKED_SET.value,
                    data={"image_ids": [], "scores": []},
                    error="Query requires concept or attributes"
                )

            result = self.engine.entity_search(
                concept=concept,
                attributes=attrs,
                limit=k
            )

            image_ids = self._extract_image_ids(result.results)

            return EngineResult(
                success=True,
                output_type=OutputType.RANKED_SET.value,
                data={
                    "image_ids": image_ids,
                    "scores": [1.0] * len(image_ids),
                    "reasoning_trace": result.reasoning_trace
                }
            )

    def _execute_scalar(self, cqr: CQR) -> EngineResult:
        """
        Execute scalar (statistical) query.

        Maps to statistical_knowledge for probability computation.
        """
        concepts = cqr.get_concepts()
        rels = cqr.get_relations()

        if not concepts or not rels:
            return EngineResult(
                success=False,
                output_type=OutputType.SCALAR.value,
                data={"value": 0.0},
                error="Scalar query requires concept and relation"
            )

        concept_a = concepts[0]
        rel = rels[0]
        rel_name = rel.get("name", "")
        concept_b = rel.get("obj", "")

        result = self.engine.statistical_knowledge(
            concept_a=concept_a,
            concept_b=concept_b,
            relation=rel_name
        )

        # Extract probability from result
        probability = 0.0
        if isinstance(result.results, dict):
            probability = result.results.get("probability", 0.0)

        return EngineResult(
            success=True,
            output_type=OutputType.SCALAR.value,
            data={
                "value": probability,
                "co_occurrences": result.results.get("co_occurrences", 0) if isinstance(result.results, dict) else 0,
                "total": result.results.get("total_concept_a", 0) if isinstance(result.results, dict) else 0,
                "reasoning_trace": result.reasoning_trace
            }
        )

    def _execute_path(self, cqr: CQR) -> EngineResult:
        """
        Execute path query.

        Maps to relational_path for path finding.
        """
        source = cqr.meta.get("source_concept", "")
        target = cqr.meta.get("target_concept", "")
        max_hops = cqr.meta.get("max_hops", 3)

        # Also try to get from constraints
        if not source or not target:
            concepts = cqr.get_concepts()
            if len(concepts) >= 2:
                source = concepts[0]
                target = concepts[1]

        if not source or not target:
            return EngineResult(
                success=False,
                output_type=OutputType.PATH.value,
                data={"paths": [], "exists": False, "shortest_hops": None},
                error="Path query requires source and target concepts"
            )

        result = self.engine.relational_path(
            source_concept=source,
            target_concept=target,
            max_hops=max_hops,
            limit=cqr.k
        )

        # Extract paths from result
        paths = []
        shortest_hops = None

        if isinstance(result.results, list) and result.results:
            paths = result.results
            # Compute shortest
            for path_info in paths:
                if isinstance(path_info, dict) and "path" in path_info:
                    path = path_info["path"]
                    hops = len(path) - 1 if isinstance(path, list) else 0
                    if shortest_hops is None or hops < shortest_hops:
                        shortest_hops = hops

        return EngineResult(
            success=True,
            output_type=OutputType.PATH.value,
            data={
                "paths": paths,
                "exists": len(paths) > 0,
                "shortest_hops": shortest_hops,
                "reasoning_trace": result.reasoning_trace
            }
        )

    def _execute_subgraph(self, cqr: CQR) -> EngineResult:
        """
        Execute subgraph query.

        Returns evidence nodes and edges for the query.
        """
        # First get the matching images
        ranked_result = self._execute_ranked_set(cqr)

        if not ranked_result.success:
            return EngineResult(
                success=False,
                output_type=OutputType.SUBGRAPH.value,
                data={"nodes": [], "edges": [], "image_ids": []},
                error=ranked_result.error
            )

        image_ids = ranked_result.data.get("image_ids", [])[:10]  # Limit for subgraph

        # Extract evidence nodes and edges
        nodes = set()
        edges = set()

        concepts = cqr.get_concepts()
        attrs = cqr.get_attributes()
        rels = cqr.get_relations()

        # Add concept nodes
        for concept in concepts:
            nodes.add(f"concept:{concept}")

        # Add attribute nodes and edges
        for attr in attrs:
            nodes.add(f"attr:{attr}")
            if concepts:
                edges.add((f"concept:{concepts[0]}", "has_attr", f"attr:{attr}"))

        # Add relation edges
        for rel in rels:
            rel_name = rel.get("name", "")
            rel_obj = rel.get("obj", "")
            nodes.add(f"concept:{rel_obj}")
            if concepts:
                edges.add((f"concept:{concepts[0]}", f"rel:{rel_name}", f"concept:{rel_obj}"))

        return EngineResult(
            success=True,
            output_type=OutputType.SUBGRAPH.value,
            data={
                "nodes": list(nodes),
                "edges": [list(e) for e in edges],
                "image_ids": image_ids,
                "reasoning_trace": ranked_result.data.get("reasoning_trace", [])
            }
        )

    def _extract_image_ids(self, results: Any) -> List[str]:
        """
        Extract image_ids from engine results.

        Handles various result formats from the engine.
        """
        image_ids = []
        seen = set()

        if isinstance(results, list):
            for item in results:
                if isinstance(item, dict):
                    img_id = item.get("image_id", "")
                    if img_id and img_id not in seen:
                        image_ids.append(str(img_id))
                        seen.add(img_id)
                elif isinstance(item, str):
                    # Might be in format "image_id:object_id"
                    parts = item.split(":")
                    if parts and parts[0] not in seen:
                        image_ids.append(parts[0])
                        seen.add(parts[0])

        elif isinstance(results, dict):
            # Results might be keyed by image_id
            for key, value in results.items():
                if isinstance(value, dict) and "image_id" in value:
                    img_id = str(value["image_id"])
                    if img_id not in seen:
                        image_ids.append(img_id)
                        seen.add(img_id)

        return image_ids


def create_adapter(config: Dict[str, Any], verbose: bool = False) -> EngineAdapter:
    """
    Factory function to create an EngineAdapter from config.

    Args:
        config: Configuration dictionary (from eval.yaml)
        verbose: Whether to print status messages

    Returns:
        Initialized EngineAdapter
    """
    paths = config.get("paths", {})
    graph_path = paths.get("kg_graph_path", "")

    if graph_path:
        # Make path absolute if relative
        graph_path = str(REPO_ROOT / graph_path)

    return EngineAdapter(
        graph_path=graph_path if graph_path else None,
        scale='10k',  # Default fallback
        config=config,
        verbose=verbose
    )


if __name__ == "__main__":
    # Quick test
    import yaml

    config_path = REPO_ROOT / "evaluation_v0_1" / "configs" / "eval.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create adapter
    adapter = create_adapter(config, verbose=True)

    # Test entity search
    from .cqr import CQR, Constraint, OutputType

    cqr = CQR(
        query_id="test_001",
        output_type=OutputType.RANKED_SET.value,
        op="retrieve",
        must=[Constraint.concept("car"), Constraint.attr("red")],
        must_not=[],
        k=10
    )

    result = adapter.execute(cqr)
    print(f"\nTest result: {result.to_dict()}")

