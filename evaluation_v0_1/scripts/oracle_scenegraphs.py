"""
Oracle Ground Truth from Scene Graphs for Evaluation Framework v0.1

Computes gold outputs for queries using ONLY GQA Scene Graphs.
NO dependency on questions1.2/*.

Supports 4 output types:
- ranked_set: images satisfying concept/attr/relation constraints
- scalar: probability P(B|A,rel)
- path: concept graph path existence and length
- subgraph: minimal evidence nodes and edges
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field
import networkx as nx

from .cqr import CQR, Constraint, GoldOutput, OutputType
from .normalize import Normalizer, normalize_term


@dataclass
class SceneGraphObject:
    """Represents an object in a scene graph."""
    obj_id: str
    name: str
    attributes: List[str]
    relations: List[Dict[str, str]]  # [{"name": rel_name, "object": target_obj_id}]


@dataclass
class SceneGraph:
    """Represents a scene graph for an image."""
    image_id: str
    objects: Dict[str, SceneGraphObject]


class SceneGraphOracle:
    """
    Oracle that computes ground truth from scene graphs.
    """

    def __init__(self, normalizer: Optional[Normalizer] = None):
        """
        Initialize the oracle.

        Args:
            normalizer: Normalizer instance for term normalization
        """
        self.normalizer = normalizer or Normalizer()
        self.scene_graphs: Dict[str, SceneGraph] = {}

        # Indices for fast lookup
        self._concept_to_images: Dict[str, Set[str]] = defaultdict(set)
        self._attr_to_images: Dict[str, Set[str]] = defaultdict(set)
        self._concept_attr_to_images: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self._relation_to_images: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)

        # Concept graph for path queries
        self._concept_graph: Optional[nx.DiGraph] = None

        # Statistics
        self._concept_freq: Dict[str, int] = defaultdict(int)
        self._attr_freq: Dict[str, int] = defaultdict(int)
        self._relation_freq: Dict[Tuple[str, str, str], int] = defaultdict(int)

    def load_scene_graphs(self, path: str, max_images: Optional[int] = None) -> int:
        """
        Load scene graphs from a JSON file.

        Args:
            path: Path to scene graphs JSON file
            max_images: Maximum number of images to load (for testing)

        Returns:
            Number of images loaded
        """
        print(f"[Oracle] Loading scene graphs from: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        count = 0
        for image_id, sg_data in data.items():
            if max_images is not None and count >= max_images:
                break

            self._process_scene_graph(str(image_id), sg_data)
            count += 1

        print(f"[Oracle] Loaded {count} scene graphs")
        print(f"[Oracle] Unique concepts: {len(self._concept_freq)}")
        print(f"[Oracle] Unique attributes: {len(self._attr_freq)}")

        # Build concept graph
        self._build_concept_graph()

        return count

    def _process_scene_graph(self, image_id: str, sg_data: Dict) -> None:
        """Process a single scene graph and update indices."""
        objects = {}
        objects_data = sg_data.get("objects", {})

        for obj_id, obj_data in objects_data.items():
            obj_id = str(obj_id)
            name = self.normalizer.normalize_term(obj_data.get("name", ""))

            if not name:
                continue

            attrs = [self.normalizer.normalize_term(a)
                    for a in obj_data.get("attributes", [])]
            attrs = [a for a in attrs if a]  # Filter empty

            relations = []
            for rel in obj_data.get("relations", []):
                rel_name = self.normalizer.normalize_term(rel.get("name", ""))
                rel_obj = str(rel.get("object", ""))
                if rel_name and rel_obj:
                    relations.append({"name": rel_name, "object": rel_obj})

            obj = SceneGraphObject(
                obj_id=obj_id,
                name=name,
                attributes=attrs,
                relations=relations
            )
            objects[obj_id] = obj

            # Update indices
            self._concept_to_images[name].add(image_id)
            self._concept_freq[name] += 1

            for attr in attrs:
                self._attr_to_images[attr].add(image_id)
                self._attr_freq[attr] += 1
                self._concept_attr_to_images[(name, attr)].add(image_id)

        # Process relations (need object names)
        for obj_id, obj in objects.items():
            subj_name = obj.name
            for rel in obj.relations:
                rel_name = rel["name"]
                target_obj_id = rel["object"]
                if target_obj_id in objects:
                    obj_name = objects[target_obj_id].name
                    key = (subj_name, rel_name, obj_name)
                    self._relation_to_images[key].add(image_id)
                    self._relation_freq[key] += 1

        self.scene_graphs[image_id] = SceneGraph(
            image_id=image_id,
            objects=objects
        )

    def _build_concept_graph(self) -> None:
        """Build a concept graph for path queries."""
        self._concept_graph = nx.DiGraph()

        # Add all concepts as nodes
        for concept in self._concept_freq.keys():
            self._concept_graph.add_node(concept)

        # Add edges from relations
        for (subj, rel, obj), freq in self._relation_freq.items():
            if not self._concept_graph.has_edge(subj, obj):
                self._concept_graph.add_edge(subj, obj, relations={rel: freq})
            else:
                edge_data = self._concept_graph.edges[subj, obj]
                if "relations" not in edge_data:
                    edge_data["relations"] = {}
                edge_data["relations"][rel] = edge_data["relations"].get(rel, 0) + freq

        print(f"[Oracle] Concept graph: {self._concept_graph.number_of_nodes()} nodes, "
              f"{self._concept_graph.number_of_edges()} edges")

    def get_concept_frequency(self, concept: str) -> int:
        """Get frequency of a concept."""
        return self._concept_freq.get(self.normalizer.normalize_term(concept), 0)

    def get_top_concepts(self, n: int = 100) -> List[Tuple[str, int]]:
        """Get top N most frequent concepts."""
        sorted_concepts = sorted(self._concept_freq.items(),
                                key=lambda x: x[1], reverse=True)
        return sorted_concepts[:n]

    def get_top_attributes(self, n: int = 100) -> List[Tuple[str, int]]:
        """Get top N most frequent attributes."""
        sorted_attrs = sorted(self._attr_freq.items(),
                             key=lambda x: x[1], reverse=True)
        return sorted_attrs[:n]

    def get_top_relations(self, n: int = 100) -> List[Tuple[Tuple[str, str, str], int]]:
        """Get top N most frequent relation triples."""
        sorted_rels = sorted(self._relation_freq.items(),
                            key=lambda x: x[1], reverse=True)
        return sorted_rels[:n]

    def get_attributes_for_concept(self, concept: str, min_count: int = 10) -> List[Tuple[str, int]]:
        """Get frequent attributes for a concept."""
        concept = self.normalizer.normalize_term(concept)
        attr_counts = defaultdict(int)

        for (c, attr), images in self._concept_attr_to_images.items():
            if c == concept:
                attr_counts[attr] = len(images)

        sorted_attrs = sorted(attr_counts.items(), key=lambda x: x[1], reverse=True)
        return [(a, c) for a, c in sorted_attrs if c >= min_count]

    # =========================================================================
    # ORACLE METHODS BY OUTPUT TYPE
    # =========================================================================

    def compute_gold(self, cqr: CQR) -> GoldOutput:
        """
        Compute gold output for a CQR.

        Args:
            cqr: Canonical Query Representation

        Returns:
            GoldOutput for the query
        """
        output_type = cqr.output_type

        if output_type == OutputType.RANKED_SET.value:
            return self._compute_ranked_set_gold(cqr)
        elif output_type == OutputType.SCALAR.value:
            return self._compute_scalar_gold(cqr)
        elif output_type == OutputType.PATH.value:
            return self._compute_path_gold(cqr)
        elif output_type == OutputType.SUBGRAPH.value:
            return self._compute_subgraph_gold(cqr)
        else:
            raise ValueError(f"Unknown output_type: {output_type}")

    def _compute_ranked_set_gold(self, cqr: CQR) -> GoldOutput:
        """
        Compute gold image set for ranked_set queries.

        Supports:
        - Entity + attributes: images with object matching concept AND all attrs
        - Relations: images with subject-rel-object
        - Negative: images with A but not B
        """
        must_concepts = [c.value for c in cqr.must if c.type == "concept"]
        must_attrs = [c.value for c in cqr.must if c.type == "attr"]
        must_rels = [c.value for c in cqr.must if c.type == "rel"]
        must_not_concepts = [c.value for c in cqr.must_not if c.type == "concept"]

        gold_images: Set[str] = set()

        # Start with concept constraint
        if must_concepts:
            main_concept = self.normalizer.normalize_term(must_concepts[0])

            if must_attrs:
                # Entity + attribute query
                gold_images = self._get_images_with_concept_and_attrs(
                    main_concept,
                    [self.normalizer.normalize_term(a) for a in must_attrs]
                )
            elif must_rels:
                # Relation query
                rel = must_rels[0]
                rel_name = self.normalizer.normalize_term(rel["name"])
                rel_obj = self.normalizer.normalize_term(rel["obj"])
                gold_images = self._get_images_with_relation(
                    main_concept, rel_name, rel_obj
                )
            else:
                # Just concept
                gold_images = self._concept_to_images.get(main_concept, set()).copy()

        # Apply must_not constraints
        for neg_concept in must_not_concepts:
            neg_concept = self.normalizer.normalize_term(neg_concept)
            neg_images = self._concept_to_images.get(neg_concept, set())
            gold_images = gold_images - neg_images

        return GoldOutput.ranked_set(
            image_ids=list(gold_images),
            notes=f"concepts={must_concepts}, attrs={must_attrs}, "
                  f"rels={must_rels}, negations={must_not_concepts}"
        )

    def _get_images_with_concept_and_attrs(self, concept: str,
                                           attrs: List[str]) -> Set[str]:
        """Get images where an object has given concept AND all attributes."""
        if not attrs:
            return self._concept_to_images.get(concept, set()).copy()

        # For each image with the concept, check if any object has all attrs
        candidate_images = self._concept_to_images.get(concept, set())
        result = set()

        for image_id in candidate_images:
            if image_id not in self.scene_graphs:
                continue
            sg = self.scene_graphs[image_id]

            for obj in sg.objects.values():
                if obj.name == concept:
                    obj_attrs = set(obj.attributes)
                    if all(attr in obj_attrs for attr in attrs):
                        result.add(image_id)
                        break

        return result

    def _get_images_with_relation(self, subj_concept: str, rel_name: str,
                                  obj_concept: str) -> Set[str]:
        """Get images with a specific subject-relation-object triple."""
        key = (subj_concept, rel_name, obj_concept)
        return self._relation_to_images.get(key, set()).copy()

    def _compute_scalar_gold(self, cqr: CQR) -> GoldOutput:
        """
        Compute gold scalar value for probability queries.

        Computes P(B | A, rel) = count(A rel B images) / count(A images)
        """
        must_concepts = [c.value for c in cqr.must if c.type == "concept"]
        must_rels = [c.value for c in cqr.must if c.type == "rel"]

        if not must_concepts or not must_rels:
            return GoldOutput.scalar(
                value=0.0, numerator=0, denominator=0,
                notes="Missing concept or relation"
            )

        concept_a = self.normalizer.normalize_term(must_concepts[0])
        rel = must_rels[0]
        rel_name = self.normalizer.normalize_term(rel["name"])
        concept_b = self.normalizer.normalize_term(rel["obj"])

        # Denominator: images with concept A
        images_with_a = self._concept_to_images.get(concept_a, set())
        denominator = len(images_with_a)

        # Numerator: images with (A rel B)
        key = (concept_a, rel_name, concept_b)
        images_with_rel = self._relation_to_images.get(key, set())
        numerator = len(images_with_rel)

        probability = numerator / denominator if denominator > 0 else 0.0

        return GoldOutput.scalar(
            value=probability,
            numerator=numerator,
            denominator=denominator,
            notes=f"P({concept_b} | {concept_a}, {rel_name})"
        )

    def _compute_path_gold(self, cqr: CQR) -> GoldOutput:
        """
        Compute gold path existence and length on concept graph.
        """
        if self._concept_graph is None:
            return GoldOutput.path(exists=False, notes="Concept graph not built")

        # Extract source and target from meta or constraints
        source = cqr.meta.get("source_concept", "")
        target = cqr.meta.get("target_concept", "")
        max_hops = cqr.meta.get("max_hops", 3)

        # Also try to extract from constraints
        if not source or not target:
            concepts = cqr.get_concepts()
            if len(concepts) >= 2:
                source = concepts[0]
                target = concepts[1]

        if not source or not target:
            return GoldOutput.path(exists=False, notes="Missing source or target")

        source = self.normalizer.normalize_term(source)
        target = self.normalizer.normalize_term(target)

        # Check if path exists
        if source not in self._concept_graph or target not in self._concept_graph:
            return GoldOutput.path(
                exists=False,
                notes=f"Source '{source}' or target '{target}' not in graph"
            )

        try:
            path = nx.shortest_path(self._concept_graph, source, target)
            if len(path) - 1 <= max_hops:
                return GoldOutput.path(
                    exists=True,
                    shortest_hops=len(path) - 1,
                    example_path=path,
                    notes=f"Path found: {' -> '.join(path)}"
                )
            else:
                return GoldOutput.path(
                    exists=False,
                    shortest_hops=len(path) - 1,
                    notes=f"Path exists but too long ({len(path)-1} > {max_hops})"
                )
        except nx.NetworkXNoPath:
            return GoldOutput.path(
                exists=False,
                notes=f"No path from '{source}' to '{target}'"
            )

    def _compute_subgraph_gold(self, cqr: CQR) -> GoldOutput:
        """
        Compute gold subgraph (minimal evidence) for a query.

        Returns normalized node and edge sets representing the evidence.
        """
        must_concepts = [c.value for c in cqr.must if c.type == "concept"]
        must_attrs = [c.value for c in cqr.must if c.type == "attr"]
        must_rels = [c.value for c in cqr.must if c.type == "rel"]

        nodes_gold: Set[str] = set()
        edges_gold: Set[Tuple[str, str, str]] = set()
        gold_images: List[str] = []

        # Add concept nodes
        for concept in must_concepts:
            concept = self.normalizer.normalize_term(concept)
            nodes_gold.add(f"concept:{concept}")

        # Add attribute nodes and edges
        for attr in must_attrs:
            attr = self.normalizer.normalize_term(attr)
            nodes_gold.add(f"attr:{attr}")
            # Edge from concept to attr
            if must_concepts:
                concept = self.normalizer.normalize_term(must_concepts[0])
                edges_gold.add((f"concept:{concept}", "has_attr", f"attr:{attr}"))

        # Add relation edges
        for rel in must_rels:
            rel_name = self.normalizer.normalize_term(rel["name"])
            obj_concept = self.normalizer.normalize_term(rel["obj"])
            nodes_gold.add(f"concept:{obj_concept}")
            if must_concepts:
                subj_concept = self.normalizer.normalize_term(must_concepts[0])
                edges_gold.add((f"concept:{subj_concept}", f"rel:{rel_name}",
                               f"concept:{obj_concept}"))

        # Get sample gold images
        ranked_gold = self._compute_ranked_set_gold(cqr)
        gold_images = ranked_gold.data.get("image_ids", [])[:10]  # Sample

        return GoldOutput.subgraph(
            nodes=list(nodes_gold),
            edges=list(edges_gold),
            image_ids=gold_images,
            notes=f"Minimal evidence for {cqr.query_id}"
        )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def validate_query(self, cqr: CQR, min_gold_size: int = 10) -> Tuple[bool, str]:
        """
        Validate that a query has sufficient gold data.

        Returns:
            (is_valid, reason)
        """
        if cqr.output_type == OutputType.RANKED_SET.value:
            gold = self._compute_ranked_set_gold(cqr)
            gold_size = len(gold.data.get("image_ids", []))
            if gold_size < min_gold_size:
                return False, f"Gold size {gold_size} < {min_gold_size}"
            return True, f"Gold size: {gold_size}"

        elif cqr.output_type == OutputType.SCALAR.value:
            gold = self._compute_scalar_gold(cqr)
            if gold.data.get("denominator", 0) == 0:
                return False, "Denominator is 0"
            return True, f"P = {gold.data['value']:.4f}"

        elif cqr.output_type == OutputType.PATH.value:
            # Path queries don't need minimum gold size
            return True, "Path query"

        elif cqr.output_type == OutputType.SUBGRAPH.value:
            gold = self._compute_subgraph_gold(cqr)
            if not gold.data.get("nodes"):
                return False, "No gold nodes"
            return True, f"Nodes: {len(gold.data['nodes'])}"

        return False, "Unknown output type"


# Convenience function for loading
def load_oracle(val_path: str, train_path: Optional[str] = None,
                normalizer: Optional[Normalizer] = None,
                max_images: Optional[int] = None) -> SceneGraphOracle:
    """
    Load scene graph oracle.

    Args:
        val_path: Path to val_sceneGraphs.json
        train_path: Optional path to train_sceneGraphs.json
        normalizer: Optional normalizer instance
        max_images: Maximum images to load (for testing)

    Returns:
        Initialized SceneGraphOracle
    """
    oracle = SceneGraphOracle(normalizer=normalizer)

    oracle.load_scene_graphs(val_path, max_images=max_images)

    if train_path:
        oracle.load_scene_graphs(train_path, max_images=max_images)

    return oracle

