"""
LightRAG-style knowledge graph builder.

Implements a document-centric graph structure:
- Document nodes: Images
- Chunk nodes: Relationship triples + attribute assertions
- Entity nodes: Canonical labels (objects, predicates, attributes)

Edges:
- Document -> HAS_CHUNK -> Chunk
- Chunk -> MENTIONS -> Entity
- Entity -> RELATED_TO -> Entity (derived from relationships)
"""

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .dataset import SceneGraph

logger = logging.getLogger(__name__)


@dataclass
class Node:
    """Base node in the graph."""

    id: str
    type: str  # "document", "chunk", "entity"
    label: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """Edge in the graph."""

    source_id: str
    target_id: str
    type: str  # "HAS_CHUNK", "MENTIONS", "RELATED_TO"
    properties: dict[str, Any] = field(default_factory=dict)


class GraphBuilder:
    """
    Builds a LightRAG-style knowledge graph from scene graphs.

    Graph structure:
    - Documents (images) contain chunks (scene graph facts)
    - Chunks mention entities (object labels, predicates, attributes)
    - Entities can be related to other entities
    """

    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self._entity_ids: dict[str, str] = {}  # label -> entity_id cache

    def _make_entity_id(self, entity_type: str, label: str) -> str:
        """Create a stable entity ID from type and label."""
        key = f"{entity_type}:{label}"
        if key not in self._entity_ids:
            hash_val = hashlib.md5(key.encode()).hexdigest()[:12]
            self._entity_ids[key] = f"entity_{entity_type}_{hash_val}"
        return self._entity_ids[key]

    def _make_chunk_id(self, image_id: int, chunk_type: str, idx: int) -> str:
        """Create a chunk ID."""
        return f"chunk_{image_id}_{chunk_type}_{idx}"

    def _get_or_create_entity(
        self, entity_type: str, label: str
    ) -> str:
        """Get or create an entity node, return its ID."""
        entity_id = self._make_entity_id(entity_type, label)
        if entity_id not in self.nodes:
            self.nodes[entity_id] = Node(
                id=entity_id,
                type="entity",
                label=label,
                properties={"entity_type": entity_type},
            )
        return entity_id

    def add_scene_graph(self, scene_graph: SceneGraph) -> None:
        """
        Add a scene graph to the knowledge graph.

        Creates:
        - One document node for the image
        - Chunk nodes for each relationship and attribute
        - Entity nodes for unique objects, predicates, and attributes
        """
        image_id = scene_graph.image_id
        doc_id = f"doc_{image_id}"

        # Create document node
        self.nodes[doc_id] = Node(
            id=doc_id,
            type="document",
            label=f"image_{image_id}",
            properties={
                "image_id": image_id,
                "num_objects": len(scene_graph.objects),
                "num_relationships": len(scene_graph.relationships),
                "num_attributes": len(scene_graph.attributes),
            },
        )

        # Process relationships as chunks
        for idx, rel in enumerate(scene_graph.relationships):
            chunk_id = self._make_chunk_id(image_id, "rel", idx)

            # Create chunk text: "subject predicate object"
            chunk_text = (
                f"{rel['subject_label']} {rel['predicate']} {rel['object_label']}"
            )

            self.nodes[chunk_id] = Node(
                id=chunk_id,
                type="chunk",
                label=chunk_text,
                properties={
                    "chunk_type": "relationship",
                    "subject": rel["subject_label"],
                    "predicate": rel["predicate"],
                    "object": rel["object_label"],
                    "image_id": image_id,
                },
            )

            # Document -> HAS_CHUNK -> Chunk
            self.edges.append(
                Edge(source_id=doc_id, target_id=chunk_id, type="HAS_CHUNK")
            )

            # Chunk -> MENTIONS -> Entity (for subject, predicate, object)
            subj_entity_id = self._get_or_create_entity(
                "object", rel["subject_label"]
            )
            pred_entity_id = self._get_or_create_entity(
                "predicate", rel["predicate"]
            )
            obj_entity_id = self._get_or_create_entity(
                "object", rel["object_label"]
            )

            self.edges.append(
                Edge(
                    source_id=chunk_id,
                    target_id=subj_entity_id,
                    type="MENTIONS",
                    properties={"role": "subject"},
                )
            )
            self.edges.append(
                Edge(
                    source_id=chunk_id,
                    target_id=pred_entity_id,
                    type="MENTIONS",
                    properties={"role": "predicate"},
                )
            )
            self.edges.append(
                Edge(
                    source_id=chunk_id,
                    target_id=obj_entity_id,
                    type="MENTIONS",
                    properties={"role": "object"},
                )
            )

            # Entity -> RELATED_TO -> Entity (subject <-> object via predicate)
            self.edges.append(
                Edge(
                    source_id=subj_entity_id,
                    target_id=obj_entity_id,
                    type="RELATED_TO",
                    properties={"via_predicate": rel["predicate"]},
                )
            )

        # Process attributes as chunks
        for idx, attr in enumerate(scene_graph.attributes):
            chunk_id = self._make_chunk_id(image_id, "attr", idx)

            # Create chunk text: "object is attribute"
            chunk_text = f"{attr['object_label']} is {attr['attribute']}"

            self.nodes[chunk_id] = Node(
                id=chunk_id,
                type="chunk",
                label=chunk_text,
                properties={
                    "chunk_type": "attribute",
                    "object": attr["object_label"],
                    "attribute": attr["attribute"],
                    "image_id": image_id,
                },
            )

            # Document -> HAS_CHUNK -> Chunk
            self.edges.append(
                Edge(source_id=doc_id, target_id=chunk_id, type="HAS_CHUNK")
            )

            # Chunk -> MENTIONS -> Entity
            obj_entity_id = self._get_or_create_entity(
                "object", attr["object_label"]
            )
            attr_entity_id = self._get_or_create_entity(
                "attribute", attr["attribute"]
            )

            self.edges.append(
                Edge(
                    source_id=chunk_id,
                    target_id=obj_entity_id,
                    type="MENTIONS",
                    properties={"role": "object"},
                )
            )
            self.edges.append(
                Edge(
                    source_id=chunk_id,
                    target_id=attr_entity_id,
                    type="MENTIONS",
                    properties={"role": "attribute"},
                )
            )

            # Entity -> RELATED_TO -> Entity (object has attribute)
            self.edges.append(
                Edge(
                    source_id=obj_entity_id,
                    target_id=attr_entity_id,
                    type="RELATED_TO",
                    properties={"via_predicate": "has_attribute"},
                )
            )

    def get_stats(self) -> dict[str, int]:
        """Get graph statistics."""
        node_types = {}
        for node in self.nodes.values():
            node_types[node.type] = node_types.get(node.type, 0) + 1

        edge_types = {}
        for edge in self.edges:
            edge_types[edge.type] = edge_types.get(edge.type, 0) + 1

        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": node_types,
            "edge_types": edge_types,
        }


class GraphStorage:
    """
    SQLite-based storage for the knowledge graph.

    Tables:
    - nodes: id, type, label, properties (JSON)
    - edges: source_id, target_id, type, properties (JSON)
    - chunks_fts: Full-text search index on chunk labels
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        """Open database connection and create tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> "GraphStorage":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _create_tables(self) -> None:
        """Create database schema."""
        assert self.conn is not None
        cursor = self.conn.cursor()

        # Nodes table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                label TEXT NOT NULL,
                properties TEXT
            )
        """
        )

        # Edges table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                type TEXT NOT NULL,
                properties TEXT,
                FOREIGN KEY (source_id) REFERENCES nodes(id),
                FOREIGN KEY (target_id) REFERENCES nodes(id)
            )
        """
        )

        # Indices for faster lookups
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(type)"
        )

        # Full-text search for chunks (standalone FTS5 table)
        cursor.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id,
                label
            )
        """
        )

        self.conn.commit()

    def save_graph(self, builder: GraphBuilder) -> None:
        """Save graph from builder to database."""
        assert self.conn is not None
        cursor = self.conn.cursor()

        # Clear existing data
        cursor.execute("DELETE FROM edges")
        cursor.execute("DELETE FROM nodes")
        cursor.execute("DELETE FROM chunks_fts")

        # Insert nodes
        for node in builder.nodes.values():
            cursor.execute(
                "INSERT INTO nodes (id, type, label, properties) VALUES (?, ?, ?, ?)",
                (node.id, node.type, node.label, json.dumps(node.properties)),
            )

        # Insert edges
        for edge in builder.edges:
            cursor.execute(
                "INSERT INTO edges (source_id, target_id, type, properties) VALUES (?, ?, ?, ?)",
                (
                    edge.source_id,
                    edge.target_id,
                    edge.type,
                    json.dumps(edge.properties),
                ),
            )

        # Build FTS index for chunks
        cursor.execute(
            """
            INSERT INTO chunks_fts (chunk_id, label)
            SELECT id, label FROM nodes WHERE type = 'chunk'
        """
        )

        self.conn.commit()
        logger.info(
            f"Saved graph: {len(builder.nodes)} nodes, {len(builder.edges)} edges"
        )

    def get_node(self, node_id: str) -> Node | None:
        """Get a node by ID."""
        assert self.conn is not None
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM nodes WHERE id = ?", (node_id,))
        row = cursor.fetchone()
        if row:
            return Node(
                id=row["id"],
                type=row["type"],
                label=row["label"],
                properties=json.loads(row["properties"] or "{}"),
            )
        return None

    def get_nodes_by_type(self, node_type: str) -> list[Node]:
        """Get all nodes of a given type."""
        assert self.conn is not None
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM nodes WHERE type = ?", (node_type,))
        return [
            Node(
                id=row["id"],
                type=row["type"],
                label=row["label"],
                properties=json.loads(row["properties"] or "{}"),
            )
            for row in cursor.fetchall()
        ]

    def get_outgoing_edges(
        self, node_id: str, edge_type: str | None = None
    ) -> list[Edge]:
        """Get outgoing edges from a node."""
        assert self.conn is not None
        cursor = self.conn.cursor()
        if edge_type:
            cursor.execute(
                "SELECT * FROM edges WHERE source_id = ? AND type = ?",
                (node_id, edge_type),
            )
        else:
            cursor.execute(
                "SELECT * FROM edges WHERE source_id = ?", (node_id,)
            )
        return [
            Edge(
                source_id=row["source_id"],
                target_id=row["target_id"],
                type=row["type"],
                properties=json.loads(row["properties"] or "{}"),
            )
            for row in cursor.fetchall()
        ]

    def get_incoming_edges(
        self, node_id: str, edge_type: str | None = None
    ) -> list[Edge]:
        """Get incoming edges to a node."""
        assert self.conn is not None
        cursor = self.conn.cursor()
        if edge_type:
            cursor.execute(
                "SELECT * FROM edges WHERE target_id = ? AND type = ?",
                (node_id, edge_type),
            )
        else:
            cursor.execute(
                "SELECT * FROM edges WHERE target_id = ?", (node_id,)
            )
        return [
            Edge(
                source_id=row["source_id"],
                target_id=row["target_id"],
                type=row["type"],
                properties=json.loads(row["properties"] or "{}"),
            )
            for row in cursor.fetchall()
        ]

    def search_chunks_fts(self, query: str, limit: int = 20) -> list[Node]:
        """Full-text search over chunk labels."""
        assert self.conn is not None
        cursor = self.conn.cursor()

        # Escape special FTS5 characters
        safe_query = query.replace('"', '""')

        try:
            cursor.execute(
                """
                SELECT n.* FROM chunks_fts f
                JOIN nodes n ON f.chunk_id = n.id
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """,
                (f'"{safe_query}"', limit),
            )
        except sqlite3.OperationalError:
            # Fallback to LIKE search if FTS fails
            cursor.execute(
                """
                SELECT * FROM nodes
                WHERE type = 'chunk' AND label LIKE ?
                LIMIT ?
            """,
                (f"%{query}%", limit),
            )

        return [
            Node(
                id=row["id"],
                type=row["type"],
                label=row["label"],
                properties=json.loads(row["properties"] or "{}"),
            )
            for row in cursor.fetchall()
        ]

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        assert self.conn is not None
        cursor = self.conn.cursor()

        cursor.execute("SELECT type, COUNT(*) as cnt FROM nodes GROUP BY type")
        node_counts = {row["type"]: row["cnt"] for row in cursor.fetchall()}

        cursor.execute("SELECT type, COUNT(*) as cnt FROM edges GROUP BY type")
        edge_counts = {row["type"]: row["cnt"] for row in cursor.fetchall()}

        return {
            "node_counts": node_counts,
            "edge_counts": edge_counts,
            "total_nodes": sum(node_counts.values()),
            "total_edges": sum(edge_counts.values()),
        }
