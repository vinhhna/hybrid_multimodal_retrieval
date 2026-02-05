"""
Baseline Methods for Evaluation Framework v0.1

Implements simple baseline methods for comparison with the main LightRAG-GQA system.

Baselines:
1. NaiveScanBaseline - Direct scene graph scan without knowledge graph
2. RelationWalkBaseline - Within-image relation traversal with bounded hops
3. TrivialParserBaseline - Minimal regex-based NL parser

These baselines establish lower bounds for comparison with the main method.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass

from .cqr import CQR, Constraint, OutputType
from .adapter_engine import EngineResult
from .normalize import normalize_term


# =============================================================================
# BASELINE RESULT CLASS
# =============================================================================

@dataclass
class BaselineResult:
    """Result from baseline method execution."""
    success: bool
    output_type: str
    data: Dict[str, Any]
    method: str
    error: Optional[str] = None
    
    def to_engine_result(self) -> EngineResult:
        """Convert to EngineResult for compatibility with evaluation."""
        return EngineResult(
            success=self.success,
            output_type=self.output_type,
            data=self.data,
            error=self.error
        )


# =============================================================================
# BASELINE 1: NAIVE SCAN (Entity Search)
# =============================================================================

class NaiveScanBaseline:
    """
    Naive scan baseline for entity search queries.
    
    This baseline iterates through objects/attributes in each image directly
    from scene graphs WITHOUT using the global concept/attribute index nodes.
    
    Purpose: Establish lower bound for entity retrieval tasks.
    Shows the value of the two-level knowledge graph structure.
    """
    
    def __init__(self, scene_graphs: Dict[str, Any], verbose: bool = False):
        """
        Initialize with loaded scene graphs.
        
        Args:
            scene_graphs: Dictionary of image_id -> scene graph data
            verbose: Whether to print progress
        """
        self.scene_graphs = scene_graphs
        self.verbose = verbose
        
        # No index building - that's the point of this baseline!
        if verbose:
            print(f"[NaiveScan] Initialized with {len(scene_graphs)} scene graphs")
    
    def execute(self, cqr: CQR) -> BaselineResult:
        """
        Execute a CQR query using naive scan.
        
        Args:
            cqr: Canonical Query Representation
            
        Returns:
            BaselineResult with found images
        """
        if cqr.output_type != OutputType.RANKED_SET.value:
            return BaselineResult(
                success=False,
                output_type=cqr.output_type,
                data={},
                method="naive_scan",
                error=f"NaiveScan only supports ranked_set queries, got {cqr.output_type}"
            )
        
        try:
            concepts = cqr.get_concepts()
            attrs = cqr.get_attributes()
            neg_concepts = cqr.get_negative_concepts()
            
            matching_images = []
            
            # Scan each scene graph
            for image_id, sg_data in self.scene_graphs.items():
                objects = sg_data.get("objects", {})
                
                image_matches = False
                has_negative = False
                
                for obj_id, obj_data in objects.items():
                    obj_name = normalize_term(obj_data.get("name", ""))
                    obj_attrs = [normalize_term(a) for a in obj_data.get("attributes", [])]
                    
                    # Check for negative concepts
                    if neg_concepts and obj_name in [normalize_term(c) for c in neg_concepts]:
                        has_negative = True
                    
                    # Check if object matches required concept and attributes
                    if concepts:
                        if obj_name not in [normalize_term(c) for c in concepts]:
                            continue
                    
                    # Check attributes
                    if attrs:
                        normalized_attrs = [normalize_term(a) for a in attrs]
                        if not all(a in obj_attrs for a in normalized_attrs):
                            continue
                    
                    image_matches = True
                
                # Apply negative constraint
                if image_matches and not has_negative:
                    matching_images.append(image_id)
            
            # Apply limit
            k = cqr.k or 50
            matching_images = matching_images[:k]
            
            return BaselineResult(
                success=True,
                output_type=OutputType.RANKED_SET.value,
                data={
                    "image_ids": matching_images,
                    "scores": [1.0] * len(matching_images)
                },
                method="naive_scan"
            )
            
        except Exception as e:
            return BaselineResult(
                success=False,
                output_type=cqr.output_type,
                data={},
                method="naive_scan",
                error=str(e)
            )


# =============================================================================
# BASELINE 2: SIMPLE RELATION WALK (Path Queries)
# =============================================================================

class RelationWalkBaseline:
    """
    Simple relation-walk baseline for relational path queries.
    
    This baseline only traverses within-image scene graph relations directly,
    WITHOUT using global-level concept linking or cross-image paths.
    
    Purpose: Establish lower bound for path-based queries.
    Shows the value of the global concept graph for path finding.
    """
    
    def __init__(self, scene_graphs: Dict[str, Any], max_hops: int = 3,
                 verbose: bool = False):
        """
        Initialize with loaded scene graphs.
        
        Args:
            scene_graphs: Dictionary of image_id -> scene graph data
            max_hops: Maximum number of hops to traverse
            verbose: Whether to print progress
        """
        self.scene_graphs = scene_graphs
        self.max_hops = max_hops
        self.verbose = verbose
        
        if verbose:
            print(f"[RelationWalk] Initialized with max_hops={max_hops}")
    
    def execute(self, cqr: CQR) -> BaselineResult:
        """
        Execute a path query using simple within-image relation walk.
        
        Args:
            cqr: Canonical Query Representation
            
        Returns:
            BaselineResult with path information
        """
        if cqr.output_type != OutputType.PATH.value:
            return BaselineResult(
                success=False,
                output_type=cqr.output_type,
                data={},
                method="relation_walk",
                error=f"RelationWalk only supports path queries, got {cqr.output_type}"
            )
        
        try:
            # Extract path parameters from CQR
            source_concept = None
            target_concept = None
            via_relation = None
            
            for constraint in cqr.must:
                if constraint.type == "concept":
                    if source_concept is None:
                        source_concept = normalize_term(constraint.value)
                    else:
                        target_concept = normalize_term(constraint.value)
                elif constraint.type == "relation":
                    via_relation = normalize_term(constraint.value.get("name", ""))
            
            if not source_concept or not target_concept:
                return BaselineResult(
                    success=False,
                    output_type=OutputType.PATH.value,
                    data={"exists": False, "paths": [], "shortest_hops": None},
                    method="relation_walk",
                    error="Path query requires source and target concepts"
                )
            
            # Find paths within each image
            all_paths = []
            min_hops = None
            
            for image_id, sg_data in self.scene_graphs.items():
                paths = self._find_paths_in_image(
                    sg_data, source_concept, target_concept, via_relation
                )
                
                for path in paths:
                    path_with_image = {"image_id": image_id, "path": path}
                    all_paths.append(path_with_image)
                    
                    hops = len(path) - 1
                    if min_hops is None or hops < min_hops:
                        min_hops = hops
            
            exists = len(all_paths) > 0
            
            return BaselineResult(
                success=True,
                output_type=OutputType.PATH.value,
                data={
                    "exists": exists,
                    "paths": all_paths[:10],  # Limit paths
                    "shortest_hops": min_hops,
                    "total_paths_found": len(all_paths)
                },
                method="relation_walk"
            )
            
        except Exception as e:
            return BaselineResult(
                success=False,
                output_type=OutputType.PATH.value,
                data={},
                method="relation_walk",
                error=str(e)
            )
    
    def _find_paths_in_image(self, sg_data: Dict, source: str, target: str,
                             via_relation: Optional[str]) -> List[List[str]]:
        """
        Find all paths from source concept to target concept within an image.
        
        Uses BFS with hop limit.
        """
        objects = sg_data.get("objects", {})
        
        # Build object name index
        name_to_obj_ids: Dict[str, List[str]] = defaultdict(list)
        for obj_id, obj_data in objects.items():
            name = normalize_term(obj_data.get("name", ""))
            name_to_obj_ids[name].append(obj_id)
        
        source_ids = name_to_obj_ids.get(source, [])
        target_ids = set(name_to_obj_ids.get(target, []))
        
        if not source_ids or not target_ids:
            return []
        
        # BFS from each source object
        all_paths = []
        
        for start_id in source_ids:
            # BFS state: (current_obj_id, path_so_far)
            queue = [(start_id, [start_id])]
            visited = {start_id}
            
            while queue:
                current_id, path = queue.pop(0)
                
                if len(path) > self.max_hops + 1:
                    continue
                
                if current_id in target_ids:
                    # Convert path to concept names
                    concept_path = []
                    for obj_id in path:
                        if obj_id in objects:
                            concept_path.append(objects[obj_id].get("name", obj_id))
                    all_paths.append(concept_path)
                    continue
                
                # Get relations from current object
                if current_id not in objects:
                    continue
                    
                relations = objects[current_id].get("relations", [])
                
                for rel in relations:
                    rel_name = normalize_term(rel.get("name", ""))
                    next_id = str(rel.get("object", ""))
                    
                    # Filter by via_relation if specified
                    if via_relation and rel_name != via_relation:
                        continue
                    
                    if next_id not in visited and next_id in objects:
                        visited.add(next_id)
                        queue.append((next_id, path + [next_id]))
        
        return all_paths


# =============================================================================
# BASELINE 3: TRIVIAL PARSER (NL -> CQR)
# =============================================================================

class TrivialParserBaseline:
    """
    Trivial rule-based parser baseline for NL -> CQR conversion.
    
    This baseline uses simple regex patterns to extract query parameters.
    Covers only a small subset of patterns to establish a lower bound.
    
    Purpose: Establish lower bound for NL parsing.
    Shows the value of more sophisticated parsing approaches.
    """
    
    # Minimal set of patterns (intentionally limited)
    PATTERNS = {
        "entity_with_attr": r"(?:find|show|get)\s+(?:images?\s+(?:with|of|containing))?\s*(?:a\s+)?(\w+)\s+(\w+)",
        "entity_simple": r"(?:find|show|get)\s+(?:images?\s+(?:with|of|containing))?\s*(?:a\s+)?(\w+)",
        "count": r"(?:how many|count)\s+(\w+)",
        "path": r"(?:path|connection)\s+(?:from|between)\s+(\w+)\s+(?:to|and)\s+(\w+)",
        "negative": r"(\w+)\s+(?:but not|without|excluding)\s+(\w+)",
    }
    
    def __init__(self, verbose: bool = False):
        """Initialize the trivial parser."""
        self.verbose = verbose
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.PATTERNS.items()
        }
    
    def parse(self, query: str) -> Dict[str, Any]:
        """
        Parse a natural language query using simple regex.
        
        Args:
            query: Natural language query string
            
        Returns:
            Dictionary with parsed parameters and query type
        """
        query_lower = query.lower().strip()
        
        # Try each pattern
        for pattern_name, regex in self.compiled_patterns.items():
            match = regex.search(query_lower)
            
            if match:
                return self._build_parse_result(pattern_name, match, query)
        
        # No match - return failure
        return {
            "success": False,
            "query_type": "unknown",
            "params": {},
            "method": "trivial_parser",
            "original_query": query,
            "error": "No pattern matched"
        }
    
    def _build_parse_result(self, pattern_name: str, match: re.Match,
                            original: str) -> Dict[str, Any]:
        """Build parse result from regex match."""
        groups = match.groups()
        
        if pattern_name == "entity_with_attr":
            # Pattern: "find <attr> <concept>"
            return {
                "success": True,
                "query_type": "entity_search",
                "params": {
                    "concept": groups[1] if len(groups) > 1 else groups[0],
                    "attributes": [groups[0]] if len(groups) > 1 else []
                },
                "method": "trivial_parser",
                "original_query": original,
                "matched_pattern": pattern_name
            }
        
        elif pattern_name == "entity_simple":
            # Pattern: "find <concept>"
            return {
                "success": True,
                "query_type": "entity_search",
                "params": {
                    "concept": groups[0],
                    "attributes": []
                },
                "method": "trivial_parser",
                "original_query": original,
                "matched_pattern": pattern_name
            }
        
        elif pattern_name == "count":
            # Pattern: "how many <concept>"
            return {
                "success": True,
                "query_type": "statistical_knowledge",
                "params": {
                    "concept": groups[0],
                    "operation": "count"
                },
                "method": "trivial_parser",
                "original_query": original,
                "matched_pattern": pattern_name
            }
        
        elif pattern_name == "path":
            # Pattern: "path from <source> to <target>"
            return {
                "success": True,
                "query_type": "relational_path",
                "params": {
                    "source_concept": groups[0],
                    "target_concept": groups[1]
                },
                "method": "trivial_parser",
                "original_query": original,
                "matched_pattern": pattern_name
            }
        
        elif pattern_name == "negative":
            # Pattern: "<concept> but not <negative>"
            return {
                "success": True,
                "query_type": "negative_constraints",
                "params": {
                    "concept_present": groups[0],
                    "concept_absent": groups[1]
                },
                "method": "trivial_parser",
                "original_query": original,
                "matched_pattern": pattern_name
            }
        
        return {
            "success": False,
            "query_type": "unknown",
            "params": {},
            "method": "trivial_parser",
            "original_query": original,
            "error": f"Unknown pattern: {pattern_name}"
        }
    
    def evaluate_on_suite(self, nl_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate parser on a suite of NL queries with gold CQR.
        
        Args:
            nl_queries: List of {"nl": str, "gold_cqr": dict}
            
        Returns:
            Evaluation metrics
        """
        results = {
            "total": len(nl_queries),
            "parsed": 0,
            "type_correct": 0,
            "param_correct": 0,
            "per_query": []
        }
        
        for item in nl_queries:
            nl = item.get("nl", "")
            gold = item.get("gold_cqr", {})
            
            parsed = self.parse(nl)
            
            query_result = {
                "nl": nl,
                "parsed_success": parsed.get("success", False),
                "type_match": False,
                "param_match": False
            }
            
            if parsed.get("success"):
                results["parsed"] += 1
                
                if parsed.get("query_type") == gold.get("query_type"):
                    results["type_correct"] += 1
                    query_result["type_match"] = True
                
                # Simple param matching (check if concepts match)
                parsed_concept = parsed.get("params", {}).get("concept", "")
                gold_concept = gold.get("params", {}).get("concept", "")
                
                if normalize_term(parsed_concept) == normalize_term(gold_concept):
                    results["param_correct"] += 1
                    query_result["param_match"] = True
            
            results["per_query"].append(query_result)
        
        # Compute rates
        if results["total"] > 0:
            results["parse_rate"] = results["parsed"] / results["total"]
            results["type_accuracy"] = results["type_correct"] / results["total"]
            results["param_accuracy"] = results["param_correct"] / results["total"]
        else:
            results["parse_rate"] = 0.0
            results["type_accuracy"] = 0.0
            results["param_accuracy"] = 0.0
        
        return results


# =============================================================================
# BASELINE ADAPTER (Unified Interface)
# =============================================================================

class BaselineAdapter:
    """
    Unified adapter for running baseline methods.
    
    Provides same interface as EngineAdapter for easy comparison.
    """
    
    def __init__(self, method: str, scene_graphs: Dict[str, Any],
                 config: Optional[Dict[str, Any]] = None,
                 verbose: bool = False):
        """
        Initialize baseline adapter.
        
        Args:
            method: One of "naive_scan", "relation_walk", "parser_trivial"
            scene_graphs: Loaded scene graph data
            config: Configuration dictionary
            verbose: Whether to print progress
        """
        self.method = method
        self.config = config or {}
        self.verbose = verbose
        
        # Initialize appropriate baseline
        baseline_config = self.config.get("method", {}).get("baselines", {})
        
        if method == "baseline_naive_scan":
            self.baseline = NaiveScanBaseline(scene_graphs, verbose=verbose)
        
        elif method == "baseline_relation_walk":
            max_hops = baseline_config.get("relation_walk", {}).get("max_hops", 3)
            self.baseline = RelationWalkBaseline(
                scene_graphs, max_hops=max_hops, verbose=verbose
            )
        
        elif method == "baseline_parser_trivial":
            self.baseline = TrivialParserBaseline(verbose=verbose)
        
        else:
            raise ValueError(f"Unknown baseline method: {method}")
    
    def execute(self, cqr: CQR) -> EngineResult:
        """
        Execute query using baseline method.
        
        Args:
            cqr: Canonical Query Representation
            
        Returns:
            EngineResult for compatibility with evaluation
        """
        if self.method in ("baseline_naive_scan", "baseline_relation_walk"):
            result = self.baseline.execute(cqr)
            return result.to_engine_result()
        
        else:
            # Parser baseline doesn't execute queries
            return EngineResult(
                success=False,
                output_type=cqr.output_type,
                data={},
                error="Parser baseline does not execute queries"
            )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_scene_graphs_for_baseline(path: str, max_images: Optional[int] = None
                                   ) -> Dict[str, Any]:
    """
    Load scene graphs for baseline evaluation.
    
    Args:
        path: Path to scene graphs JSON file
        max_images: Maximum images to load (for testing)
        
    Returns:
        Dictionary of image_id -> scene graph data
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_images is not None:
        # Limit number of images
        keys = list(data.keys())[:max_images]
        data = {k: data[k] for k in keys}
    
    return data


def run_baseline_evaluation(baseline_name: str, config: Dict[str, Any],
                           suites_dir: str, results_dir: str,
                           verbose: bool = False) -> Dict[str, Any]:
    """
    Run evaluation using a baseline method.
    
    Args:
        baseline_name: Name of baseline method
        config: Configuration dictionary
        suites_dir: Directory containing test suites
        results_dir: Directory to save results
        verbose: Whether to print progress
        
    Returns:
        Evaluation results
    """
    from pathlib import Path
    
    paths = config.get("paths", {})
    scene_graphs_path = paths.get("scenegraphs_train", "sceneGraphs/train_sceneGraphs.json")
    
    # Load scene graphs
    if verbose:
        print(f"[Baseline] Loading scene graphs from: {scene_graphs_path}")
    
    scene_graphs = load_scene_graphs_for_baseline(scene_graphs_path)
    
    # Create baseline adapter
    adapter = BaselineAdapter(
        method=baseline_name,
        scene_graphs=scene_graphs,
        config=config,
        verbose=verbose
    )
    
    # Note: Full evaluation would integrate with evaluate_engine.py
    # This function provides the setup; integration done in run_all.py
    
    return {
        "baseline": baseline_name,
        "scene_graphs_loaded": len(scene_graphs),
        "adapter_ready": True
    }
