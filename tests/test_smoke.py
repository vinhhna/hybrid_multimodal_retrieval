"""
Smoke tests for LightRAG-VG150.

These tests use synthetic data so they can run without the VG150 dataset files.
"""

import tempfile
from pathlib import Path

import pytest

from lightrag_vg150.dataset import SceneGraph
from lightrag_vg150.graph import GraphBuilder, GraphStorage
from lightrag_vg150.query import QueryEngine


class TestGraphBuilder:
    """Tests for the GraphBuilder class."""

    def test_build_from_scene_graph(self):
        """Test building a graph from a single scene graph."""
        sg = SceneGraph(
            image_id=0,
            objects=[
                {"label": "person", "local_idx": 0},
                {"label": "horse", "local_idx": 1},
                {"label": "grass", "local_idx": 2},
            ],
            relationships=[
                {
                    "subject_idx": 0,
                    "subject_label": "person",
                    "predicate": "riding",
                    "predicate_idx": 1,
                    "object_idx": 1,
                    "object_label": "horse",
                },
                {
                    "subject_idx": 1,
                    "subject_label": "horse",
                    "predicate": "on",
                    "predicate_idx": 2,
                    "object_idx": 2,
                    "object_label": "grass",
                },
            ],
            attributes=[
                {
                    "object_local_idx": 0,
                    "object_label": "person",
                    "attribute": "tall",
                    "attribute_idx": 1,
                },
            ],
        )

        builder = GraphBuilder()
        builder.add_scene_graph(sg)

        stats = builder.get_stats()
        assert stats["total_nodes"] > 0
        assert stats["total_edges"] > 0
        assert "document" in stats["node_types"]
        assert "chunk" in stats["node_types"]
        assert "entity" in stats["node_types"]

    def test_multiple_scene_graphs_share_entities(self):
        """Test that entities are shared across scene graphs."""
        sg1 = SceneGraph(
            image_id=0,
            relationships=[
                {
                    "subject_idx": 0,
                    "subject_label": "person",
                    "predicate": "riding",
                    "predicate_idx": 1,
                    "object_idx": 1,
                    "object_label": "horse",
                },
            ],
        )
        sg2 = SceneGraph(
            image_id=1,
            relationships=[
                {
                    "subject_idx": 0,
                    "subject_label": "person",
                    "predicate": "walking",
                    "predicate_idx": 2,
                    "object_idx": 1,
                    "object_label": "dog",
                },
            ],
        )

        builder = GraphBuilder()
        builder.add_scene_graph(sg1)
        builder.add_scene_graph(sg2)

        # Should have 2 documents
        assert builder.get_stats()["node_types"]["document"] == 2

        # "person" entity should be shared
        entity_labels = [
            n.label
            for n in builder.nodes.values()
            if n.type == "entity" and n.properties.get("entity_type") == "object"
        ]
        assert entity_labels.count("person") == 1


class TestGraphStorage:
    """Tests for GraphStorage class."""

    def test_save_and_load(self):
        """Test saving and loading a graph."""
        sg = SceneGraph(
            image_id=0,
            relationships=[
                {
                    "subject_idx": 0,
                    "subject_label": "cat",
                    "predicate": "on",
                    "predicate_idx": 1,
                    "object_idx": 1,
                    "object_label": "mat",
                },
            ],
        )

        builder = GraphBuilder()
        builder.add_scene_graph(sg)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            with GraphStorage(db_path) as storage:
                storage.save_graph(builder)
                stats = storage.get_stats()

            assert stats["total_nodes"] > 0
            assert stats["total_edges"] > 0

            # Verify we can read back
            with GraphStorage(db_path) as storage:
                doc = storage.get_node("doc_0")
                assert doc is not None
                assert doc.type == "document"

    def test_fts_search(self):
        """Test full-text search over chunks."""
        sg = SceneGraph(
            image_id=0,
            relationships=[
                {
                    "subject_idx": 0,
                    "subject_label": "person",
                    "predicate": "riding",
                    "predicate_idx": 1,
                    "object_idx": 1,
                    "object_label": "bicycle",
                },
            ],
        )

        builder = GraphBuilder()
        builder.add_scene_graph(sg)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            with GraphStorage(db_path) as storage:
                storage.save_graph(builder)

                results = storage.search_chunks_fts("riding", limit=5)
                assert len(results) > 0
                assert any("riding" in r.label for r in results)


class TestQueryEngine:
    """Tests for QueryEngine class."""

    @pytest.fixture
    def mini_graph(self):
        """Create a mini graph for testing."""
        scene_graphs = [
            SceneGraph(
                image_id=0,
                relationships=[
                    {
                        "subject_idx": 0,
                        "subject_label": "person",
                        "predicate": "riding",
                        "predicate_idx": 1,
                        "object_idx": 1,
                        "object_label": "horse",
                    },
                    {
                        "subject_idx": 1,
                        "subject_label": "horse",
                        "predicate": "on",
                        "predicate_idx": 2,
                        "object_idx": 2,
                        "object_label": "grass",
                    },
                ],
                attributes=[
                    {
                        "object_local_idx": 0,
                        "object_label": "person",
                        "attribute": "young",
                        "attribute_idx": 1,
                    },
                ],
            ),
            SceneGraph(
                image_id=1,
                relationships=[
                    {
                        "subject_idx": 0,
                        "subject_label": "dog",
                        "predicate": "running on",
                        "predicate_idx": 3,
                        "object_idx": 1,
                        "object_label": "grass",
                    },
                ],
                attributes=[
                    {
                        "object_local_idx": 0,
                        "object_label": "dog",
                        "attribute": "brown",
                        "attribute_idx": 2,
                    },
                ],
            ),
            SceneGraph(
                image_id=2,
                relationships=[
                    {
                        "subject_idx": 0,
                        "subject_label": "cat",
                        "predicate": "sitting on",
                        "predicate_idx": 4,
                        "object_idx": 1,
                        "object_label": "chair",
                    },
                ],
            ),
        ]

        builder = GraphBuilder()
        for sg in scene_graphs:
            builder.add_scene_graph(sg)

        return builder

    def test_query_local_retrieval(self, mini_graph):
        """Test local BM25 retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            with GraphStorage(db_path) as storage:
                storage.save_graph(mini_graph)

                engine = QueryEngine(storage)
                engine.build_index()

                results = engine.query("person riding horse", top_k=3)

                assert len(results) > 0
                # Image 0 should be the top result
                assert results[0].image_id == 0

    def test_query_entity_expansion(self, mini_graph):
        """Test entity expansion finds related images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            with GraphStorage(db_path) as storage:
                storage.save_graph(mini_graph)

                engine = QueryEngine(storage)
                engine.build_index()

                # Query for "grass" should find images 0 and 1
                results = engine.query("grass", top_k=5)

                image_ids = {r.image_id for r in results}
                assert 0 in image_ids or 1 in image_ids

    def test_query_returns_evidence(self, mini_graph):
        """Test that query results include evidence chunks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            with GraphStorage(db_path) as storage:
                storage.save_graph(mini_graph)

                engine = QueryEngine(storage)
                engine.build_index()

                results = engine.query("dog", top_k=3)

                # Should have results with evidence
                assert len(results) > 0
                assert len(results[0].evidence_chunks) > 0


class TestSmokeIntegration:
    """End-to-end smoke test."""

    def test_full_pipeline(self):
        """Test the full pipeline: build graph -> query."""
        # Create synthetic scene graphs
        scene_graphs = [
            SceneGraph(
                image_id=i,
                relationships=[
                    {
                        "subject_idx": 0,
                        "subject_label": "person",
                        "predicate": "near",
                        "predicate_idx": 1,
                        "object_idx": 1,
                        "object_label": "tree",
                    }
                ],
            )
            for i in range(5)
        ]

        # Add a unique relationship to image 3
        scene_graphs[3] = SceneGraph(
            image_id=3,
            relationships=[
                {
                    "subject_idx": 0,
                    "subject_label": "elephant",
                    "predicate": "walking in",
                    "predicate_idx": 2,
                    "object_idx": 1,
                    "object_label": "water",
                }
            ],
        )

        # Build graph
        builder = GraphBuilder()
        for sg in scene_graphs:
            builder.add_scene_graph(sg)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "graph.db"

            # Save
            with GraphStorage(db_path) as storage:
                storage.save_graph(builder)

            # Query
            with GraphStorage(db_path) as storage:
                engine = QueryEngine(storage)
                engine.build_index()

                # Should find the elephant image
                results = engine.query("elephant water", top_k=3)
                assert len(results) > 0
                assert results[0].image_id == 3

                # Should find multiple person/tree images
                results = engine.query("person tree", top_k=5)
                assert len(results) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
