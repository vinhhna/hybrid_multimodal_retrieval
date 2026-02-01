"""
GQA LightRAG Knowledge Graph Builder - Multi-Scale Version
============================================================
Script for building Multimodal Knowledge Graph from dataset GQA (Visual Reasoning)
with configurable number of images và output directory.

Supported scales:
- sample_1k: 1,000 images (demo/quick testing)
- sample_10k: 10,000 images (medium testing)
- full: Toàn bộ 74,942 images (production)

Author: Generated for IT3930E - Project III
Date: 2024
"""

import os
import sys
import json
import zipfile
import pickle
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional

try:
    import networkx as nx
except ImportError:
    raise ImportError("Please install networkx: pip install networkx")

try:
    from tqdm import tqdm
except ImportError:
    raise ImportError("Please install tqdm: pip install tqdm")


class GQALightRAGGraphBuilder:
    """
    Builder class for constructing Knowledge Graph using architecture LightRAG từ GQA dataset.
    
    Graph Architecture:
    - Instance Level (Instance Level): Each object in an image is a node
    - Global Level (Global Level): Global Concept and Global Attribute nodes
    
    Edge Types:
    - instance_of: Connects Instance Node → Global Concept Node
    - has_attribute: Connects Instance Node → Global Attribute Node  
    - semantic_relation: Connects Instance Nodes within the same image
    """
    
    def __init__(self, 
                 scene_graphs_dir: str = "sceneGraphs",
                 output_dir: str = "experiments"):
        """
        Initialize Graph Builder.
        
        Args:
            scene_graphs_dir: Directory containing original scene graphs files
            output_dir: Root directory for experiments
        """
        self.scene_graphs_dir = Path(scene_graphs_dir)
        self.output_dir = Path(output_dir)
        self.graph = nx.DiGraph()
        
        # Thống kê
        self.stats = {
            'total_images': 0,
            'total_objects': 0,
            'instance_nodes': 0,
            'concept_nodes': 0,
            'attribute_nodes': 0,
            'instance_of_edges': 0,
            'has_attribute_edges': 0,
            'semantic_relation_edges': 0,
        }
        
        # Cache các global nodes để tránh duplicate
        self.global_concepts: Set[str] = set()
        self.global_attributes: Set[str] = set()
    
    def reset(self):
        """Reset builder to construct new graph."""
        self.graph = nx.DiGraph()
        self.stats = {key: 0 for key in self.stats}
        self.global_concepts = set()
        self.global_attributes = set()
    
    def find_scene_graph_file(self) -> Optional[Path]:
        """Find file train_sceneGraphs.json."""
        possible_paths = [
            self.scene_graphs_dir / "train_sceneGraphs.json",
            Path("train_sceneGraphs.json"),
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        return None
    
    def load_scene_graphs(self, max_images: Optional[int] = None) -> Dict[str, Any]:
        """
        Read data scene graphs từ file JSON.
        
        Args:
            max_images: Số lượng images tối đa (None = all)
            
        Returns:
            Dictionary containing scene graphs of images
        """
        json_path = self.find_scene_graph_file()
        
        if json_path is None:
            raise FileNotFoundError(
                "File not found train_sceneGraphs.json."
            )
        
        print(f"📂 Reading file: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
        total_images = len(all_data)
        
        if max_images is None or max_images >= total_images:
            data = all_data
            print(f"   ✅ Read all {len(data):,} images")
        else:
            image_ids = list(all_data.keys())[:max_images]
            data = {img_id: all_data[img_id] for img_id in image_ids}
            print(f"   ✅ Read {len(data):,} images (from total of {total_images:,} images)")
        
        return data
    
    def _normalize_name(self, name: str) -> str:
        """Chuẩn hóa tên."""
        return name.lower().strip()
    
    def _create_instance_node_id(self, image_id: str, object_id: str) -> str:
        return f"{image_id}:{object_id}"
    
    def _create_concept_node_id(self, concept_name: str) -> str:
        return f"Concept:{self._normalize_name(concept_name)}"
    
    def _create_attribute_node_id(self, attribute_name: str) -> str:
        return f"Attr:{self._normalize_name(attribute_name)}"
    
    def _add_global_concept_node(self, concept_name: str) -> str:
        node_id = self._create_concept_node_id(concept_name)
        
        if node_id not in self.global_concepts:
            self.graph.add_node(
                node_id,
                node_type='global_concept',
                name=self._normalize_name(concept_name),
                level='global'
            )
            self.global_concepts.add(node_id)
            self.stats['concept_nodes'] += 1
        
        return node_id
    
    def _add_global_attribute_node(self, attribute_name: str) -> str:
        node_id = self._create_attribute_node_id(attribute_name)
        
        if node_id not in self.global_attributes:
            self.graph.add_node(
                node_id,
                node_type='global_attribute',
                name=self._normalize_name(attribute_name),
                level='global'
            )
            self.global_attributes.add(node_id)
            self.stats['attribute_nodes'] += 1
        
        return node_id
    
    def _add_instance_node(self, image_id: str, object_id: str, 
                           obj_data: Dict[str, Any]) -> str:
        node_id = self._create_instance_node_id(image_id, object_id)
        
        self.graph.add_node(
            node_id,
            node_type='instance',
            level='instance',
            image_id=image_id,
            object_id=object_id,
            name=obj_data.get('name', 'unknown'),
            attributes=obj_data.get('attributes', []),
            x=obj_data.get('x', 0),
            y=obj_data.get('y', 0),
            w=obj_data.get('w', 0),
            h=obj_data.get('h', 0)
        )
        
        self.stats['instance_nodes'] += 1
        return node_id
    
    def _add_instance_of_edge(self, instance_node_id: str, concept_node_id: str):
        self.graph.add_edge(
            instance_node_id,
            concept_node_id,
            edge_type='instance_of',
            relation='instance_of'
        )
        self.stats['instance_of_edges'] += 1
    
    def _add_has_attribute_edge(self, instance_node_id: str, attribute_node_id: str):
        self.graph.add_edge(
            instance_node_id,
            attribute_node_id,
            edge_type='has_attribute',
            relation='has_attribute'
        )
        self.stats['has_attribute_edges'] += 1
    
    def _add_semantic_relation_edge(self, source_node_id: str, target_node_id: str,
                                     relation_name: str, image_id: str):
        self.graph.add_edge(
            source_node_id,
            target_node_id,
            edge_type='semantic_relation',
            relation=relation_name,
            image_id=image_id
        )
        self.stats['semantic_relation_edges'] += 1
    
    def build_graph(self, scene_graphs: Dict[str, Any]) -> nx.DiGraph:
        """Build Knowledge Graph từ scene graphs data."""
        print("\n🔨 Starting Knowledge Graph construction...")
        print("=" * 60)
        
        self.stats['total_images'] = len(scene_graphs)
        
        for image_id, image_data in tqdm(scene_graphs.items(), 
                                          desc="📊 Processing images",
                                          unit="images"):
            objects = image_data.get('objects', {})
            self.stats['total_objects'] += len(objects)
            
            # Add Instance Nodes
            for object_id, obj_data in objects.items():
                instance_node_id = self._add_instance_node(
                    image_id, object_id, obj_data
                )
                
                # Add Global Concept Node and instance_of edge
                obj_name = obj_data.get('name', '')
                if obj_name:
                    concept_node_id = self._add_global_concept_node(obj_name)
                    self._add_instance_of_edge(instance_node_id, concept_node_id)
                
                # Add Global Attribute Nodes and has_attribute edges
                attributes = obj_data.get('attributes', [])
                for attr in attributes:
                    if attr:
                        attr_node_id = self._add_global_attribute_node(attr)
                        self._add_has_attribute_edge(instance_node_id, attr_node_id)
            
            # Add semantic_relation edges
            for object_id, obj_data in objects.items():
                source_node_id = self._create_instance_node_id(image_id, object_id)
                relations = obj_data.get('relations', [])
                
                for rel in relations:
                    target_object_id = rel.get('object', '')
                    relation_name = rel.get('name', '')
                    
                    if target_object_id and relation_name:
                        if target_object_id in objects:
                            target_node_id = self._create_instance_node_id(
                                image_id, target_object_id
                            )
                            self._add_semantic_relation_edge(
                                source_node_id, 
                                target_node_id,
                                relation_name,
                                image_id
                            )
        
        print("\n✅ Knowledge Graph construction completed!")
        return self.graph
    
    def print_statistics(self):
        """Print detailed statistics about graph."""
        print("\n" + "=" * 60)
        print("📈 KNOWLEDGE GRAPH STATISTICS")
        print("=" * 60)
        
        print("\n📷 Input Data:")
        print(f"   • Number of images processed: {self.stats['total_images']:,}")
        print(f"   • Total objects: {self.stats['total_objects']:,}")
        
        print("\n🔵 Nodes:")
        total_nodes = self.graph.number_of_nodes()
        print(f"   • Total nodes: {total_nodes:,}")
        print(f"   • Instance Nodes: {self.stats['instance_nodes']:,}")
        print(f"   • Global Concept Nodes: {self.stats['concept_nodes']:,}")
        print(f"   • Global Attribute Nodes: {self.stats['attribute_nodes']:,}")
        
        print("\n🔗 Edges:")
        total_edges = self.graph.number_of_edges()
        print(f"   • Total edges: {total_edges:,}")
        print(f"   • instance_of edges: {self.stats['instance_of_edges']:,}")
        print(f"   • has_attribute edges: {self.stats['has_attribute_edges']:,}")
        print(f"   • semantic_relation edges: {self.stats['semantic_relation_edges']:,}")
        
        print("\n" + "=" * 60)
    
    def save_graph(self, output_path: str) -> str:
        """Save graph to pickle file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving graph to: {output_path}")
        
        with open(output_path, 'wb') as f:
            pickle.dump(self.graph, f, pickle.HIGHEST_PROTOCOL)
        
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ Saved successfully! Size: {file_size:.2f} MB")
        
        # Save statistics
        stats_path = output_path.with_suffix('.stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)
        print(f"   📊 Saved statistics to: {stats_path}")
        
        return str(output_path)
    
    @staticmethod
    def load_graph(input_path: str) -> nx.DiGraph:
        """Load graph from pickle file."""
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        
        print(f"📂 Reading graph from: {input_path}")
        
        with open(input_path, 'rb') as f:
            graph = pickle.load(f)
        
        print(f"   ✅ Nodes: {graph.number_of_nodes():,}, Edges: {graph.number_of_edges():,}")
        
        return graph


def build_sample_1k():
    """Build graph with 1,000 images."""
    print("\n" + "=" * 60)
    print("🚀 BUILD SAMPLE 1K (1,000 images)")
    print("=" * 60)
    
    builder = GQALightRAGGraphBuilder()
    scene_graphs = builder.load_scene_graphs(max_images=1000)
    builder.build_graph(scene_graphs)
    builder.print_statistics()
    builder.save_graph("experiments/sample_1k/gqa_lightrag.gpickle")
    
    return builder.graph


def build_sample_10k():
    """Build graph with 10,000 images."""
    print("\n" + "=" * 60)
    print("🚀 BUILD SAMPLE 10K (10,000 images)")
    print("=" * 60)
    
    builder = GQALightRAGGraphBuilder()
    scene_graphs = builder.load_scene_graphs(max_images=10000)
    builder.build_graph(scene_graphs)
    builder.print_statistics()
    builder.save_graph("experiments/sample_10k/gqa_lightrag.gpickle")
    
    return builder.graph


def build_full():
    """Build graph with all images."""
    print("\n" + "=" * 60)
    print("🚀 BUILD FULL DATASET (74,942 images)")
    print("=" * 60)
    
    builder = GQALightRAGGraphBuilder()
    scene_graphs = builder.load_scene_graphs(max_images=None)
    builder.build_graph(scene_graphs)
    builder.print_statistics()
    builder.save_graph("experiments/full/gqa_lightrag.gpickle")
    
    return builder.graph


def main():
    parser = argparse.ArgumentParser(
        description="GQA LightRAG Knowledge Graph Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  python gqa_lightrag_builder.py --scale 1k      # Build with 1,000 images
  python gqa_lightrag_builder.py --scale 10k     # Build with 10,000 images
  python gqa_lightrag_builder.py --scale full    # Build with all images
  python gqa_lightrag_builder.py --scale all     # Build tất cả các phiên bản
        """
    )
    
    parser.add_argument(
        '--scale',
        type=str,
        choices=['1k', '10k', 'full', 'all'],
        default='10k',
        help='Scale của dataset: 1k (1,000), 10k (10,000), full (74,942), all (tất cả)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 GQA LIGHTRAG KNOWLEDGE GRAPH BUILDER")
    print("=" * 60)
    print(f"Scale: {args.scale}")
    print("=" * 60)
    
    if args.scale == '1k':
        build_sample_1k()
    elif args.scale == '10k':
        build_sample_10k()
    elif args.scale == 'full':
        build_full()
    elif args.scale == 'all':
        build_sample_1k()
        build_sample_10k()
        build_full()
    
    print("\n" + "=" * 60)
    print("🎉 COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
