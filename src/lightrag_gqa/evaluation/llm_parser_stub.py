"""
LLM Parser Stub for Evaluation Framework v0.1

This is a stub interface for future LLM-based parsing.
Currently falls back to the heuristic parser or simple regex patterns.

To integrate a real LLM parser:
1. Implement the parse() method to call your LLM API
2. Ensure the output is a valid CQR object
3. Handle errors gracefully
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import re

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from .cqr import CQR, Constraint, OutputType


class LLMParserStub:
    """
    Stub LLM parser that mimics what an LLM parser would produce.

    For now, uses simple pattern matching or falls back to heuristic parser.
    Replace the parse() method with actual LLM calls when ready.
    """

    def __init__(self, use_heuristic_fallback: bool = True, verbose: bool = False):
        """
        Initialize the LLM parser stub.

        Args:
            use_heuristic_fallback: Whether to use heuristic parser as fallback
            verbose: Whether to print debug info
        """
        self.use_heuristic_fallback = use_heuristic_fallback
        self.verbose = verbose
        self.heuristic_parser = None

        if use_heuristic_fallback:
            try:
                from basic_queries.nl_parser import NaturalLanguageParser
                self.heuristic_parser = NaturalLanguageParser()
            except ImportError:
                print("[WARN] Could not import heuristic parser for fallback")

    def parse(self, nl_query: str, query_id: str = "llm_query") -> CQR:
        """
        Parse a natural language query into CQR.

        This is a STUB - replace with actual LLM API calls.

        Args:
            nl_query: Natural language query string
            query_id: Optional query ID for the result

        Returns:
            CQR object
        """
        if self.verbose:
            print(f"[LLM Stub] Parsing: {nl_query}")

        # Try simple pattern matching first (simulating LLM output)
        cqr = self._pattern_parse(nl_query, query_id)

        if cqr is not None:
            return cqr

        # Fall back to heuristic parser
        if self.use_heuristic_fallback and self.heuristic_parser:
            return self._heuristic_fallback(nl_query, query_id)

        # Return a default empty CQR
        return CQR(
            query_id=query_id,
            output_type=OutputType.RANKED_SET.value,
            op="retrieve",
            must=[],
            must_not=[],
            k=50,
            meta={"parser": "llm_stub", "confidence": 0.0}
        )

    def _pattern_parse(self, nl_query: str, query_id: str) -> Optional[CQR]:
        """
        Simple pattern matching (placeholder for LLM).
        """
        nl_lower = nl_query.lower().strip()

        # Pattern: "Find images with X but without Y"
        neg_match = re.search(
            r'(?:find|retrieve|show|get)\s+(?:images?\s+)?with\s+(\w+)\s+(?:but\s+)?(?:without|not|no)\s+(\w+)',
            nl_lower
        )
        if neg_match:
            concept_a = neg_match.group(1)
            concept_b = neg_match.group(2)
            return CQR(
                query_id=query_id,
                output_type=OutputType.RANKED_SET.value,
                op="retrieve",
                must=[Constraint.concept(concept_a)],
                must_not=[Constraint.concept(concept_b)],
                k=50,
                meta={"parser": "llm_stub", "pattern": "negative"}
            )

        # Pattern: "Find a path from X to Y"
        path_match = re.search(
            r'(?:find|get)\s+(?:a\s+)?path\s+from\s+(\w+)\s+to\s+(\w+)',
            nl_lower
        )
        if path_match:
            source = path_match.group(1)
            target = path_match.group(2)
            return CQR(
                query_id=query_id,
                output_type=OutputType.PATH.value,
                op="path",
                must=[Constraint.concept(source), Constraint.concept(target)],
                must_not=[],
                k=10,
                meta={
                    "parser": "llm_stub",
                    "pattern": "path",
                    "source_concept": source,
                    "target_concept": target,
                    "max_hops": 3
                }
            )

        # Pattern: "How often is X rel Y" (statistical)
        stat_match = re.search(
            r'(?:how\s+often|probability|what\s+is\s+the\s+probability)\s+.*?(\w+)\s+(on|in|near|wearing|holding|riding|next to|behind|beside)\s+(?:a\s+)?(\w+)',
            nl_lower
        )
        if stat_match:
            obj = stat_match.group(1)
            rel = stat_match.group(2)
            subj = stat_match.group(3)
            return CQR(
                query_id=query_id,
                output_type=OutputType.SCALAR.value,
                op="stats",
                must=[Constraint.concept(subj), Constraint.rel(rel, obj)],
                must_not=[],
                k=1,
                meta={"parser": "llm_stub", "pattern": "statistical"}
            )

        # Pattern: "Find images with ADJ NOUN" or "Find ADJ NOUN"
        entity_match = re.search(
            r'(?:find|retrieve|show|get)\s+(?:images?\s+(?:with\s+)?(?:a\s+)?)?(\w+)\s+(\w+)(?:s)?(?:\s|$|\.)',
            nl_lower
        )
        if entity_match:
            word1 = entity_match.group(1)
            word2 = entity_match.group(2)

            # Common attributes
            attrs = {'red', 'blue', 'green', 'yellow', 'white', 'black', 'brown',
                    'large', 'small', 'big', 'tall', 'old', 'new', 'wooden', 'metal'}

            if word1 in attrs:
                return CQR(
                    query_id=query_id,
                    output_type=OutputType.RANKED_SET.value,
                    op="retrieve",
                    must=[Constraint.concept(word2), Constraint.attr(word1)],
                    must_not=[],
                    k=50,
                    meta={"parser": "llm_stub", "pattern": "entity_attr"}
                )

        # Pattern: "Find images where X is rel Y" (subgraph)
        subgraph_match = re.search(
            r'(?:find|show)\s+(?:images?\s+)?where\s+(?:a\s+)?(\w+)\s+is\s+(\w+)\s+(?:a\s+)?(\w+)',
            nl_lower
        )
        if subgraph_match:
            subj = subgraph_match.group(1)
            rel = subgraph_match.group(2)
            obj = subgraph_match.group(3)
            return CQR(
                query_id=query_id,
                output_type=OutputType.SUBGRAPH.value,
                op="retrieve",
                must=[Constraint.concept(subj), Constraint.rel(rel, obj)],
                must_not=[],
                k=50,
                meta={"parser": "llm_stub", "pattern": "subgraph"}
            )

        return None

    def _heuristic_fallback(self, nl_query: str, query_id: str) -> CQR:
        """
        Fall back to heuristic parser and convert result to CQR.
        """
        if self.heuristic_parser is None:
            return CQR(
                query_id=query_id,
                output_type=OutputType.RANKED_SET.value,
                op="retrieve",
                must=[],
                must_not=[],
                k=50,
                meta={"parser": "llm_stub", "error": "no_fallback"}
            )

        result = self.heuristic_parser.parse(nl_query)

        # Map query type to output type
        query_type = result.query_type.value
        params = result.params

        output_type = OutputType.RANKED_SET.value
        op = "retrieve"
        must = []
        must_not = []
        meta = {"parser": "llm_stub_heuristic_fallback", "confidence": result.confidence}

        if query_type == "entity_search":
            output_type = OutputType.RANKED_SET.value
            op = "retrieve"
            if params.get("concept"):
                must.append(Constraint.concept(params["concept"]))
            for attr in params.get("attributes", []):
                must.append(Constraint.attr(attr))

        elif query_type == "negative_constraints":
            output_type = OutputType.RANKED_SET.value
            op = "retrieve"
            if params.get("concept_present"):
                must.append(Constraint.concept(params["concept_present"]))
            if params.get("concept_absent"):
                must_not.append(Constraint.concept(params["concept_absent"]))

        elif query_type == "statistical_knowledge":
            output_type = OutputType.SCALAR.value
            op = "stats"
            if params.get("concept_a"):
                must.append(Constraint.concept(params["concept_a"]))
            if params.get("concept_b"):
                must.append(Constraint.rel("near", params["concept_b"]))

        elif query_type == "relational_path":
            output_type = OutputType.PATH.value
            op = "path"
            if params.get("source_concept"):
                must.append(Constraint.concept(params["source_concept"]))
            if params.get("target_concept"):
                must.append(Constraint.concept(params["target_concept"]))
            meta["source_concept"] = params.get("source_concept", "")
            meta["target_concept"] = params.get("target_concept", "")
            meta["max_hops"] = 3

        return CQR(
            query_id=query_id,
            output_type=output_type,
            op=op,
            must=must,
            must_not=must_not,
            k=50,
            meta=meta
        )


class HeuristicParserWrapper:
    """
    Wrapper for the existing heuristic parser that outputs CQR format.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the wrapper.

        Args:
            verbose: Whether to print debug info
        """
        self.verbose = verbose

        try:
            from basic_queries.nl_parser import NaturalLanguageParser
            self.parser = NaturalLanguageParser()
        except ImportError as e:
            raise ImportError(f"Failed to import heuristic parser: {e}")

    def parse(self, nl_query: str, query_id: str = "heuristic_query") -> CQR:
        """
        Parse a natural language query into CQR.

        Args:
            nl_query: Natural language query string
            query_id: Optional query ID

        Returns:
            CQR object
        """
        if self.verbose:
            print(f"[Heuristic] Parsing: {nl_query}")

        result = self.parser.parse(nl_query)

        # Map query type to output type and build CQR
        query_type = result.query_type.value
        params = result.params

        output_type = OutputType.RANKED_SET.value
        op = "retrieve"
        must = []
        must_not = []
        meta = {
            "parser": "heuristic",
            "confidence": result.confidence,
            "matched_pattern": result.matched_pattern,
            "original_query_type": query_type
        }

        if query_type == "entity_search":
            output_type = OutputType.RANKED_SET.value
            op = "retrieve"
            if params.get("concept"):
                must.append(Constraint.concept(params["concept"]))
            for attr in params.get("attributes", []):
                must.append(Constraint.attr(attr))

        elif query_type == "negative_constraints":
            output_type = OutputType.RANKED_SET.value
            op = "retrieve"
            if params.get("concept_present"):
                must.append(Constraint.concept(params["concept_present"]))
            if params.get("concept_absent"):
                must_not.append(Constraint.concept(params["concept_absent"]))

        elif query_type == "statistical_knowledge":
            output_type = OutputType.SCALAR.value
            op = "stats"
            concept_a = params.get("concept_a")
            concept_b = params.get("concept_b")
            if concept_a:
                must.append(Constraint.concept(concept_a))
            if concept_b:
                must.append(Constraint.rel("near", concept_b))

        elif query_type == "relational_path":
            output_type = OutputType.PATH.value
            op = "path"
            source = params.get("source_concept")
            target = params.get("target_concept")
            if source:
                must.append(Constraint.concept(source))
            if target:
                must.append(Constraint.concept(target))
            meta["source_concept"] = source or ""
            meta["target_concept"] = target or ""
            meta["max_hops"] = 3

        elif query_type == "similarity_search":
            output_type = OutputType.RANKED_SET.value
            op = "similarity"
            if params.get("concept"):
                must.append(Constraint.concept(params["concept"]))
            for attr in params.get("attributes", []):
                must.append(Constraint.attr(attr))

        elif query_type in ("hierarchical", "comparative", "anomaly_detection",
                           "visual_attribute_constraint"):
            # Map these to ranked_set for now
            output_type = OutputType.RANKED_SET.value
            op = "retrieve"
            if params.get("concept"):
                must.append(Constraint.concept(params["concept"]))
            if params.get("category"):
                must.append(Constraint.concept(params["category"]))

        else:  # unknown
            output_type = OutputType.RANKED_SET.value
            op = "retrieve"

        return CQR(
            query_id=query_id,
            output_type=output_type,
            op=op,
            must=must,
            must_not=must_not,
            k=50,
            meta=meta
        )


if __name__ == "__main__":
    # Quick test
    llm_parser = LLMParserStub(verbose=True)
    heuristic_parser = HeuristicParserWrapper(verbose=True)

    test_queries = [
        "Find images with red car",
        "Find images with tree but without sky",
        "Find a path from person to car",
        "How often is a shirt worn by a man?",
        "Find images where a dog is near a person"
    ]

    print("\n=== LLM Parser Stub Test ===")
    for q in test_queries:
        print(f"\nQuery: {q}")
        cqr = llm_parser.parse(q)
        print(f"  Output Type: {cqr.output_type}")
        print(f"  Op: {cqr.op}")
        print(f"  Must: {[c.to_dict() for c in cqr.must]}")
        print(f"  Must Not: {[c.to_dict() for c in cqr.must_not]}")

    print("\n=== Heuristic Parser Wrapper Test ===")
    for q in test_queries:
        print(f"\nQuery: {q}")
        cqr = heuristic_parser.parse(q)
        print(f"  Output Type: {cqr.output_type}")
        print(f"  Op: {cqr.op}")
        print(f"  Must: {[c.to_dict() for c in cqr.must]}")
        print(f"  Must Not: {[c.to_dict() for c in cqr.must_not]}")

