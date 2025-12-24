"""
CLI entrypoints for LightRAG-VG150.
"""

import argparse
import logging
import sys
from pathlib import Path

from .dataset import VG150Dataset
from .graph import GraphBuilder, GraphStorage
from .query import QueryEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def fetch_main() -> None:
    """CLI entrypoint for fetching/validating VG150 data."""
    parser = argparse.ArgumentParser(
        description="Fetch or validate VG150 dataset files"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing or to download VG150 files",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only use first N images (for testing)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Attempt to download files if missing (not implemented)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset = VG150Dataset(data_dir, sample=args.sample)
    valid, missing = dataset.validate_files()

    if valid:
        print(f"✓ All required VG150 files found in {data_dir}")
        dataset.load()
        stats = dataset.get_label_stats()
        print(f"  - Object classes: {stats['num_object_classes']}")
        print(f"  - Predicate classes: {stats['num_predicate_classes']}")
        print(f"  - Attribute classes: {stats['num_attribute_classes']}")
        print(f"  - Images: {stats['num_images']}")
        dataset.close()
    else:
        print(f"✗ Missing required files:")
        for f in missing:
            print(f"  - {f}")
        print()
        print("Please download the VG150 preprocessed files:")
        print("  - VG-SGG-with-attri.h5")
        print("  - VG-SGG-dicts-with-attri.json")
        print()
        print("These files can be obtained from:")
        print(
            "  https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch"
        )
        sys.exit(1)


def build_main() -> None:
    """CLI entrypoint for building the knowledge graph."""
    parser = argparse.ArgumentParser(
        description="Build LightRAG-style knowledge graph from VG150"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing VG150 files",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for graph artifacts",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only use first N images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "graph.db"

    print(f"Loading VG150 dataset from {args.data_dir}...")
    with VG150Dataset(args.data_dir, sample=args.sample, seed=args.seed) as dataset:
        stats = dataset.get_label_stats()
        print(f"  Loaded {stats['num_images']} images")

        print("Building knowledge graph...")
        builder = GraphBuilder()

        from tqdm import tqdm

        for sg in tqdm(dataset.iter_scene_graphs(), total=stats["num_images"]):
            builder.add_scene_graph(sg)

        graph_stats = builder.get_stats()
        print(f"  Nodes: {graph_stats['total_nodes']}")
        print(f"  Edges: {graph_stats['total_edges']}")
        print(f"  Node types: {graph_stats['node_types']}")

        print(f"Saving graph to {db_path}...")
        with GraphStorage(db_path) as storage:
            storage.save_graph(builder)

    print(f"✓ Graph built successfully: {db_path}")


def query_main() -> None:
    """CLI entrypoint for querying the knowledge graph."""
    parser = argparse.ArgumentParser(
        description="Query the LightRAG-style knowledge graph"
    )
    parser.add_argument(
        "--graph_dir",
        type=str,
        required=True,
        help="Directory containing graph.db",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Natural language query",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of results to return",
    )
    parser.add_argument(
        "--hops",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of hops for entity expansion",
    )

    args = parser.parse_args()

    graph_dir = Path(args.graph_dir)
    db_path = graph_dir / "graph.db"

    if not db_path.exists():
        print(f"✗ Graph database not found: {db_path}")
        print("Run build-graph first to create the graph.")
        sys.exit(1)

    print(f"Loading graph from {db_path}...")
    with GraphStorage(db_path) as storage:
        engine = QueryEngine(storage)
        engine.build_index()

        print(f"\nQuery: {args.query}")
        print("-" * 60)

        results = engine.query(
            args.query, top_k=args.top_k, expansion_hops=args.hops
        )

        if not results:
            print("No results found.")
        else:
            for i, result in enumerate(results, 1):
                print(f"\n[{i}] Image {result.image_id} (score: {result.score:.4f})")
                print("    Evidence:")
                for chunk in result.evidence_chunks[:3]:
                    print(f"      - {chunk}")
                if result.matched_entities:
                    print(f"    Matched: {', '.join(result.matched_entities[:5])}")


if __name__ == "__main__":
    # Default to query if run directly
    query_main()
