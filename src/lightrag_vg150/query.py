"""
Query engine for LightRAG-style graph.

Implements:
- Local retrieval: keyword/BM25 search over chunks
- Global expansion: 1-2 hop traversal over entity graph
- Result ranking and explanation generation
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from rank_bm25 import BM25Okapi

from .graph import GraphStorage, Node

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result from a graph query."""

    image_id: int
    score: float
    evidence_chunks: list[str] = field(default_factory=list)
    matched_entities: list[str] = field(default_factory=list)
    explanation: str = ""


class QueryEngine:
    """
    Query engine for LightRAG-style knowledge graph.

    Combines:
    1. Local retrieval: BM25 search over chunk text
    2. Global expansion: Entity graph traversal (1-2 hops)
    """

    def __init__(self, storage: GraphStorage):
        self.storage = storage
        self._chunks: list[Node] = []
        self._chunk_texts: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._entity_to_chunks: dict[str, list[str]] = defaultdict(list)
        self._entity_to_entity: dict[str, list[tuple[str, str]]] = defaultdict(list)

    def build_index(self) -> None:
        """Build search indices from the graph."""
        logger.info("Building query indices...")

        # Load all chunks for BM25
        self._chunks = self.storage.get_nodes_by_type("chunk")
        self._chunk_texts = [
            self._tokenize(chunk.label) for chunk in self._chunks
        ]

        if self._chunk_texts:
            self._bm25 = BM25Okapi(self._chunk_texts)

        # Build entity -> chunks mapping
        for chunk in self._chunks:
            edges = self.storage.get_outgoing_edges(chunk.id, "MENTIONS")
            for edge in edges:
                self._entity_to_chunks[edge.target_id].append(chunk.id)

        # Build entity -> entity mapping
        entities = self.storage.get_nodes_by_type("entity")
        for entity in entities:
            edges = self.storage.get_outgoing_edges(entity.id, "RELATED_TO")
            for edge in edges:
                predicate = edge.properties.get("via_predicate", "related")
                self._entity_to_entity[entity.id].append(
                    (edge.target_id, predicate)
                )

        logger.info(
            f"Indexed {len(self._chunks)} chunks, "
            f"{len(self._entity_to_chunks)} entities with chunks"
        )

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for BM25."""
        return text.lower().split()

    def _search_bm25(
        self, query: str, top_k: int = 20
    ) -> list[tuple[Node, float]]:
        """BM25 search over chunks."""
        if not self._bm25 or not self._chunks:
            return []

        query_tokens = self._tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        # Get top-k results
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in indexed_scores[:top_k]:
            if score > 0:
                results.append((self._chunks[idx], score))

        return results

    def _find_matching_entities(self, query: str) -> list[Node]:
        """Find entities that match query terms."""
        query_terms = set(self._tokenize(query))
        matching = []

        entities = self.storage.get_nodes_by_type("entity")
        for entity in entities:
            entity_terms = set(self._tokenize(entity.label))
            if query_terms & entity_terms:  # Intersection
                matching.append(entity)

        return matching

    def _expand_entities(
        self, entity_ids: set[str], hops: int = 2
    ) -> set[str]:
        """Expand entity set by traversing RELATED_TO edges."""
        expanded = set(entity_ids)
        frontier = set(entity_ids)

        for _ in range(hops):
            next_frontier = set()
            for entity_id in frontier:
                for target_id, _ in self._entity_to_entity.get(entity_id, []):
                    if target_id not in expanded:
                        next_frontier.add(target_id)
                        expanded.add(target_id)
            frontier = next_frontier
            if not frontier:
                break

        return expanded

    def _get_chunks_for_entities(self, entity_ids: set[str]) -> list[str]:
        """Get chunk IDs that mention any of the given entities."""
        chunk_ids = set()
        for entity_id in entity_ids:
            chunk_ids.update(self._entity_to_chunks.get(entity_id, []))
        return list(chunk_ids)

    def _get_image_from_chunk(self, chunk: Node) -> int | None:
        """Extract image_id from chunk properties."""
        return chunk.properties.get("image_id")

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        expansion_hops: int = 1,
        local_weight: float = 0.7,
        global_weight: float = 0.3,
    ) -> list[QueryResult]:
        """
        Execute a query against the graph.

        Args:
            query_text: Natural language query
            top_k: Number of results to return
            expansion_hops: Number of hops for entity expansion (1-2)
            local_weight: Weight for BM25 local search
            global_weight: Weight for entity expansion

        Returns:
            List of QueryResult objects ranked by score
        """
        # 1. Local retrieval: BM25 over chunks
        local_results = self._search_bm25(query_text, top_k=top_k * 3)

        # 2. Find matching entities
        matching_entities = self._find_matching_entities(query_text)
        entity_ids = {e.id for e in matching_entities}

        # 3. Expand entities
        expanded_entity_ids = self._expand_entities(entity_ids, hops=expansion_hops)

        # 4. Get chunks from expanded entities
        expanded_chunk_ids = set(self._get_chunks_for_entities(expanded_entity_ids))

        # 5. Score and aggregate by image
        image_scores: dict[int, dict[str, Any]] = defaultdict(
            lambda: {
                "local_score": 0.0,
                "global_score": 0.0,
                "chunks": [],
                "entities": set(),
            }
        )

        # Add local scores
        for chunk, score in local_results:
            image_id = self._get_image_from_chunk(chunk)
            if image_id is not None:
                image_scores[image_id]["local_score"] += score
                image_scores[image_id]["chunks"].append(chunk.label)

        # Add global scores (binary: does chunk come from expanded entities?)
        for chunk_id in expanded_chunk_ids:
            chunk = self.storage.get_node(chunk_id)
            if chunk:
                image_id = self._get_image_from_chunk(chunk)
                if image_id is not None:
                    image_scores[image_id]["global_score"] += 1.0
                    if chunk.label not in image_scores[image_id]["chunks"]:
                        image_scores[image_id]["chunks"].append(chunk.label)

        # Track matched entities per image
        for entity in matching_entities:
            for chunk_id in self._entity_to_chunks.get(entity.id, []):
                chunk = self.storage.get_node(chunk_id)
                if chunk:
                    image_id = self._get_image_from_chunk(chunk)
                    if image_id is not None:
                        image_scores[image_id]["entities"].add(entity.label)

        # Compute final scores
        results = []
        max_local = max(
            (s["local_score"] for s in image_scores.values()), default=1.0
        ) or 1.0
        max_global = max(
            (s["global_score"] for s in image_scores.values()), default=1.0
        ) or 1.0

        for image_id, scores in image_scores.items():
            norm_local = scores["local_score"] / max_local
            norm_global = scores["global_score"] / max_global
            final_score = local_weight * norm_local + global_weight * norm_global

            # Generate explanation
            entities_str = ", ".join(list(scores["entities"])[:5])
            explanation = (
                f"Matched entities: [{entities_str}]. "
                f"Local score: {norm_local:.2f}, Global score: {norm_global:.2f}"
            )

            results.append(
                QueryResult(
                    image_id=image_id,
                    score=final_score,
                    evidence_chunks=scores["chunks"][:5],
                    matched_entities=list(scores["entities"]),
                    explanation=explanation,
                )
            )

        # Sort by score and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def explain_result(self, result: QueryResult) -> str:
        """Generate a detailed explanation for a query result."""
        lines = [
            f"Image ID: {result.image_id}",
            f"Score: {result.score:.4f}",
            "",
            "Evidence chunks:",
        ]
        for i, chunk in enumerate(result.evidence_chunks, 1):
            lines.append(f"  {i}. {chunk}")

        if result.matched_entities:
            lines.append("")
            lines.append(f"Matched entities: {', '.join(result.matched_entities)}")

        lines.append("")
        lines.append(f"Explanation: {result.explanation}")

        return "\n".join(lines)
