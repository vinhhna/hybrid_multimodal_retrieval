"""
Query Suite Generator for Evaluation Framework v0.1

Generates query suites from GQA Scene Graphs (NO Questions dependency).

Suites generated:
1. suite_ranked_entity_attr.jsonl - Entity + attribute queries
2. suite_ranked_negative.jsonl - Negative constraint queries
3. suite_scalar_stats.jsonl - Statistical probability queries
4. suite_path.jsonl - Path finding queries
5. suite_subgraph.jsonl - Subgraph evidence queries
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

from .cqr import CQR, Constraint, GoldOutput, QuerySuiteItem, save_suite, OutputType
from .oracle_scenegraphs import SceneGraphOracle, load_oracle
from .normalize import Normalizer


class SuiteGenerator:
    """
    Generates evaluation query suites from scene graphs.
    """

    def __init__(self, oracle: SceneGraphOracle, config: Dict[str, Any]):
        """
        Initialize suite generator.

        Args:
            oracle: SceneGraphOracle instance
            config: Configuration dictionary
        """
        self.oracle = oracle
        self.config = config
        self.random_seed = config.get("random_seed", 1337)
        random.seed(self.random_seed)

        # Suite size defaults
        suite_sizes = config.get("suite_sizes", {})
        self.n_ranked_entity_attr = suite_sizes.get("ranked_entity_attr_n", 500)
        self.n_ranked_negative = suite_sizes.get("ranked_negative_n", 500)
        self.n_scalar_stats = suite_sizes.get("scalar_stats_n", 200)
        self.n_path = suite_sizes.get("path_n", 200)
        self.n_subgraph = suite_sizes.get("subgraph_n", 200)

        # Negative constraint safeguards
        neg_safeguards = config.get("negative_safeguards", {})
        self.min_concept_freq = neg_safeguards.get("min_concept_freq", 200)
        self.min_gold_size = neg_safeguards.get("min_gold_size", 50)
        self.top_n_concepts_for_negation = neg_safeguards.get("top_n_concepts_for_negation", 100)

    def generate_all_suites(self, output_dir: str) -> Dict[str, int]:
        """
        Generate all query suites.

        Args:
            output_dir: Directory to save JSONL files

        Returns:
            Dictionary of suite_name -> count
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # 1. Ranked Entity + Attribute
        print("\n[Generator] Creating suite_ranked_entity_attr...")
        suite = self.generate_ranked_entity_attr_suite(self.n_ranked_entity_attr)
        save_suite(suite, output_path / "suite_ranked_entity_attr.jsonl")
        results["ranked_entity_attr"] = len(suite)
        print(f"[Generator] Generated {len(suite)} queries")

        # 2. Ranked Negative
        print("\n[Generator] Creating suite_ranked_negative...")
        suite = self.generate_ranked_negative_suite(self.n_ranked_negative)
        save_suite(suite, output_path / "suite_ranked_negative.jsonl")
        results["ranked_negative"] = len(suite)
        print(f"[Generator] Generated {len(suite)} queries")

        # 3. Scalar Stats
        print("\n[Generator] Creating suite_scalar_stats...")
        suite = self.generate_scalar_stats_suite(self.n_scalar_stats)
        save_suite(suite, output_path / "suite_scalar_stats.jsonl")
        results["scalar_stats"] = len(suite)
        print(f"[Generator] Generated {len(suite)} queries")

        # 4. Path
        print("\n[Generator] Creating suite_path...")
        suite = self.generate_path_suite(self.n_path)
        save_suite(suite, output_path / "suite_path.jsonl")
        results["path"] = len(suite)
        print(f"[Generator] Generated {len(suite)} queries")

        # 5. Subgraph
        print("\n[Generator] Creating suite_subgraph...")
        suite = self.generate_subgraph_suite(self.n_subgraph)
        save_suite(suite, output_path / "suite_subgraph.jsonl")
        results["subgraph"] = len(suite)
        print(f"[Generator] Generated {len(suite)} queries")

        return results

    def generate_ranked_entity_attr_suite(self, n: int) -> List[QuerySuiteItem]:
        """
        Generate entity + attribute queries.

        Example: "Find images with a red car"
        """
        suite = []
        used_pairs = set()

        # Get top concepts
        top_concepts = self.oracle.get_top_concepts(200)

        attempts = 0
        max_attempts = n * 10

        while len(suite) < n and attempts < max_attempts:
            attempts += 1

            # Pick a concept
            concept, concept_freq = random.choice(top_concepts[:100])

            # Get attributes for this concept
            attrs_for_concept = self.oracle.get_attributes_for_concept(concept, min_count=20)
            if not attrs_for_concept:
                continue

            # Pick 1-2 attributes
            num_attrs = random.randint(1, min(2, len(attrs_for_concept)))
            selected_attrs = random.sample(attrs_for_concept, num_attrs)
            attr_names = [a[0] for a in selected_attrs]

            # Check for uniqueness
            pair_key = (concept, tuple(sorted(attr_names)))
            if pair_key in used_pairs:
                continue
            used_pairs.add(pair_key)

            # Create CQR
            query_id = f"ranked_entity_attr_{len(suite):04d}"
            cqr = CQR(
                query_id=query_id,
                output_type=OutputType.RANKED_SET.value,
                op="retrieve",
                must=[
                    Constraint.concept(concept),
                    *[Constraint.attr(a) for a in attr_names]
                ],
                must_not=[],
                k=50,
                return_unit="images",
                meta={"source": "generated"}
            )

            # Validate with oracle
            is_valid, reason = self.oracle.validate_query(cqr, self.min_gold_size)
            if not is_valid:
                continue

            # Compute gold
            gold_output = self.oracle.compute_gold(cqr)

            # Generate NL templates
            if len(attr_names) == 1:
                nl_templates = [
                    f"Find images with a {attr_names[0]} {concept}.",
                    f"Retrieve pictures containing a {concept} that is {attr_names[0]}."
                ]
            else:
                attrs_str = " and ".join(attr_names)
                nl_templates = [
                    f"Find images with a {attrs_str} {concept}.",
                    f"Retrieve pictures containing a {concept} that is {attrs_str}."
                ]

            # Create suite item
            item = QuerySuiteItem(
                query_id=query_id,
                cqr_gold=cqr,
                nl_templates=nl_templates,
                gold_output=gold_output,
                gold_summary={
                    "gold_size": len(gold_output.data.get("image_ids", [])),
                    "concept": concept,
                    "attributes": attr_names
                }
            )
            suite.append(item)

        return suite

    def generate_ranked_negative_suite(self, n: int) -> List[QuerySuiteItem]:
        """
        Generate negative constraint queries.

        Example: "Find images with tree but without sky"
        """
        suite = []
        used_pairs = set()

        # Get top concepts for negation (high frequency = better annotation coverage)
        top_concepts = self.oracle.get_top_concepts(self.top_n_concepts_for_negation)
        high_freq_concepts = [(c, f) for c, f in top_concepts if f >= self.min_concept_freq]

        if len(high_freq_concepts) < 2:
            print(f"[WARN] Not enough high-frequency concepts for negative suite")
            return suite

        attempts = 0
        max_attempts = n * 10

        while len(suite) < n and attempts < max_attempts:
            attempts += 1

            # Pick two different concepts
            concept_a, freq_a = random.choice(high_freq_concepts)
            concept_b, freq_b = random.choice(high_freq_concepts)

            if concept_a == concept_b:
                continue

            # Check uniqueness
            pair_key = (concept_a, concept_b)
            if pair_key in used_pairs:
                continue
            used_pairs.add(pair_key)

            # Create CQR
            query_id = f"ranked_negative_{len(suite):04d}"
            cqr = CQR(
                query_id=query_id,
                output_type=OutputType.RANKED_SET.value,
                op="retrieve",
                must=[Constraint.concept(concept_a)],
                must_not=[Constraint.concept(concept_b)],
                k=50,
                return_unit="images",
                meta={"source": "generated"}
            )

            # Validate
            is_valid, reason = self.oracle.validate_query(cqr, self.min_gold_size)
            if not is_valid:
                continue

            # Compute gold
            gold_output = self.oracle.compute_gold(cqr)

            # NL templates
            nl_templates = [
                f"Find images with {concept_a} but without {concept_b}.",
                f"Retrieve pictures containing {concept_a} and no {concept_b}."
            ]

            item = QuerySuiteItem(
                query_id=query_id,
                cqr_gold=cqr,
                nl_templates=nl_templates,
                gold_output=gold_output,
                gold_summary={
                    "gold_size": len(gold_output.data.get("image_ids", [])),
                    "concept_present": concept_a,
                    "concept_absent": concept_b
                }
            )
            suite.append(item)

        return suite

    def generate_scalar_stats_suite(self, n: int) -> List[QuerySuiteItem]:
        """
        Generate statistical probability queries.

        Example: "What is the probability of finding a person wearing a shirt?"
        """
        suite = []
        used_triples = set()

        # Get top relation triples
        top_rels = self.oracle.get_top_relations(500)

        attempts = 0
        max_attempts = n * 10

        while len(suite) < n and attempts < max_attempts:
            attempts += 1

            if not top_rels:
                break

            triple, freq = random.choice(top_rels[:200])
            subj, rel, obj = triple

            # Check uniqueness
            if triple in used_triples:
                continue
            used_triples.add(triple)

            # Create CQR
            query_id = f"scalar_stats_{len(suite):04d}"
            cqr = CQR(
                query_id=query_id,
                output_type=OutputType.SCALAR.value,
                op="stats",
                must=[
                    Constraint.concept(subj),
                    Constraint.rel(rel, obj)
                ],
                must_not=[],
                k=1,
                return_unit="images",
                meta={
                    "source": "generated",
                    "stat_type": "conditional_probability"
                }
            )

            # Validate
            is_valid, reason = self.oracle.validate_query(cqr)
            if not is_valid:
                continue

            # Compute gold
            gold_output = self.oracle.compute_gold(cqr)

            # NL templates
            nl_templates = [
                f"How often is a {obj} {rel} a {subj}?",
                f"What is the probability of finding {obj} {rel} {subj}?"
            ]

            item = QuerySuiteItem(
                query_id=query_id,
                cqr_gold=cqr,
                nl_templates=nl_templates,
                gold_output=gold_output,
                gold_summary={
                    "probability": gold_output.data.get("value", 0),
                    "numerator": gold_output.data.get("numerator", 0),
                    "denominator": gold_output.data.get("denominator", 0),
                    "triple": f"{subj}-{rel}-{obj}"
                }
            )
            suite.append(item)

        return suite

    def generate_path_suite(self, n: int) -> List[QuerySuiteItem]:
        """
        Generate path finding queries.

        Example: "Find a path from person to car"
        """
        suite = []
        used_pairs = set()

        # Get concepts from the concept graph
        if self.oracle._concept_graph is None:
            print("[WARN] Concept graph not built, cannot generate path suite")
            return suite

        concepts = list(self.oracle._concept_graph.nodes())
        if len(concepts) < 2:
            return suite

        # Filter to high-frequency concepts
        top_concepts = self.oracle.get_top_concepts(100)
        top_concept_names = [c for c, f in top_concepts]
        concepts = [c for c in concepts if c in top_concept_names]

        attempts = 0
        max_attempts = n * 20

        while len(suite) < n and attempts < max_attempts:
            attempts += 1

            # Pick two different concepts
            source = random.choice(concepts)
            target = random.choice(concepts)

            if source == target:
                continue

            pair_key = (source, target)
            if pair_key in used_pairs:
                continue
            used_pairs.add(pair_key)

            # Create CQR
            query_id = f"path_{len(suite):04d}"
            cqr = CQR(
                query_id=query_id,
                output_type=OutputType.PATH.value,
                op="path",
                must=[
                    Constraint.concept(source),
                    Constraint.concept(target)
                ],
                must_not=[],
                k=10,
                return_unit="paths",
                meta={
                    "source": "generated",
                    "source_concept": source,
                    "target_concept": target,
                    "max_hops": 3
                }
            )

            # Compute gold (always valid for path queries)
            gold_output = self.oracle.compute_gold(cqr)

            # NL templates
            nl_templates = [
                f"Find a path from {source} to {target}.",
                f"How can {source} connect to {target}?"
            ]

            item = QuerySuiteItem(
                query_id=query_id,
                cqr_gold=cqr,
                nl_templates=nl_templates,
                gold_output=gold_output,
                gold_summary={
                    "exists": gold_output.data.get("exists", False),
                    "shortest_hops": gold_output.data.get("shortest_hops"),
                    "source": source,
                    "target": target
                }
            )
            suite.append(item)

        return suite

    def generate_subgraph_suite(self, n: int) -> List[QuerySuiteItem]:
        """
        Generate subgraph evidence queries.

        Example: "Find images where a person is wearing a shirt"
        """
        suite = []
        used_triples = set()

        # Get top relation triples
        top_rels = self.oracle.get_top_relations(500)

        attempts = 0
        max_attempts = n * 10

        while len(suite) < n and attempts < max_attempts:
            attempts += 1

            if not top_rels:
                break

            triple, freq = random.choice(top_rels[:200])
            subj, rel, obj = triple

            if triple in used_triples:
                continue
            used_triples.add(triple)

            # Create CQR
            query_id = f"subgraph_{len(suite):04d}"
            cqr = CQR(
                query_id=query_id,
                output_type=OutputType.SUBGRAPH.value,
                op="retrieve",
                must=[
                    Constraint.concept(subj),
                    Constraint.rel(rel, obj)
                ],
                must_not=[],
                k=50,
                return_unit="images",
                meta={"source": "generated"}
            )

            # Validate
            is_valid, reason = self.oracle.validate_query(cqr, min_gold_size=10)
            if not is_valid:
                continue

            # Compute gold
            gold_output = self.oracle.compute_gold(cqr)

            # NL templates
            nl_templates = [
                f"Find images where a {subj} is {rel} a {obj}.",
                f"Show pictures with {subj} {rel} {obj}."
            ]

            item = QuerySuiteItem(
                query_id=query_id,
                cqr_gold=cqr,
                nl_templates=nl_templates,
                gold_output=gold_output,
                gold_summary={
                    "nodes": gold_output.data.get("nodes", []),
                    "edges": gold_output.data.get("edges", []),
                    "num_images": len(gold_output.data.get("image_ids", [])),
                    "triple": f"{subj}-{rel}-{obj}"
                }
            )
            suite.append(item)

        return suite


def generate_suites(config_path: str, output_dir: Optional[str] = None) -> Dict[str, int]:
    """
    Main entry point for suite generation.

    Args:
        config_path: Path to eval.yaml
        output_dir: Override output directory

    Returns:
        Dictionary of suite_name -> count
    """
    import yaml

    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Get paths
    paths = config.get("paths", {})
    base_path = Path(config_path).parent.parent.parent  # Go up to repo root

    # Determine which scene graph split to use based on config
    scenegraph_config = config.get("scenegraph_loading", {})
    split_for_suites = scenegraph_config.get("split_for_suites", "train")
    
    if split_for_suites == "train":
        sg_path = base_path / paths.get("scenegraphs_train", "sceneGraphs/train_sceneGraphs.json")
    elif split_for_suites == "val":
        sg_path = base_path / paths.get("scenegraphs_val", "sceneGraphs/val_sceneGraphs.json")
    else:
        # Default to train to match KG
        sg_path = base_path / paths.get("scenegraphs_train", "sceneGraphs/train_sceneGraphs.json")

    if output_dir is None:
        output_dir = base_path / paths.get("suites_dir", "evaluation_v0_1/data/suites")

    synonym_path = base_path / paths.get("synonym_map", "evaluation_v0_1/data/synonym_map.json")

    # Initialize normalizer
    normalizer = Normalizer(str(synonym_path))

    # Load oracle
    print(f"\n[Generator] Loading scene graphs from: {sg_path}")
    print(f"[Generator] Using split: {split_for_suites}")
    oracle = load_oracle(str(sg_path), normalizer=normalizer)

    # Generate suites
    generator = SuiteGenerator(oracle, config)
    results = generator.generate_all_suites(str(output_dir))

    print(f"\n[Generator] Suite generation complete!")
    print(f"[Generator] Total queries: {sum(results.values())}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate evaluation query suites")
    parser.add_argument("--config", default="evaluation_v0_1/configs/eval.yaml",
                       help="Path to eval.yaml")
    parser.add_argument("--output", default=None,
                       help="Output directory for suites")

    args = parser.parse_args()

    generate_suites(args.config, args.output)

