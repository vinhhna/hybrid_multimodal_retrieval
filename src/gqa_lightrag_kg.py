"""
GQA LightRAG Knowledge Graph Builder
=====================================
Script for building Multimodal Knowledge Graph from GQA (Visual Reasoning) dataset
using LightRAG architecture with 2 tiers: Instance Level and Global Level.

Author: Generated for IT3930E - Project III
Date: 2024
"""

import os
import json
import zipfile
import pickle
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
    Builder class for constructing Knowledge Graph using LightRAG architecture from GQA dataset.
    
    Graph Architecture:
    - Instance Level: Each object in an image is a node
    - Global Level: Global Concept and Global Attribute nodes
    
    Edge Types:
    - instance_of: Connects Instance Node → Global Concept Node
    - has_attribute: Connects Instance Node → Global Attribute Node  
    - semantic_relation: Connects Instance Nodes within the same image
    """
    
    def __init__(self, data_dir: str = "data/gqa", scene_graphs_dir: str = "sceneGraphs"):
        """
        Initialize Graph Builder.
        
        Args:
            data_dir: Directory containing output data
            scene_graphs_dir: Directory containing original scene graphs files
        """
        self.data_dir = Path(data_dir)
        self.scene_graphs_dir = Path(scene_graphs_dir)
        self.graph = nx.DiGraph()  # Directed graph for directional relationships
        
        # Statistics
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
        
    def extract_zip_if_exists(self) -> bool:
        """
        Extract sceneGraphs.json.zip file if it exists.
        
        Returns:
            True if extraction successful or file already extracted
        """
        zip_patterns = [
            self.scene_graphs_dir / "sceneGraphs.json.zip",
            self.scene_graphs_dir / "train_sceneGraphs.json.zip",
            self.data_dir / "sceneGraphs.json.zip",
        ]
        
        for zip_path in zip_patterns:
            if zip_path.exists():
                print(f"📦 Found zip file: {zip_path}")
                try:
                    # Create destination directory if not exists
                    self.data_dir.mkdir(parents=True, exist_ok=True)
                    
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        print(f"   Extracting to {self.data_dir}...")
                        zip_ref.extractall(self.data_dir)
                    print("   ✅ Extraction successful!")
                    return True
                except zipfile.BadZipFile:
                    print(f"   ❌ Error: Corrupted zip file: {zip_path}")
                    return False
                except Exception as e:
                    print(f"   ❌ Error during extraction: {e}")
                    return False
        
        print("ℹ️  No zip file found, will read directly from JSON file.")
        return True
    
    def find_scene_graph_file(self) -> Optional[Path]:
        """
        Find train_sceneGraphs.json file in possible locations.
        
        Returns:
            Path to file if found, None otherwise
        """
        possible_paths = [
            self.scene_graphs_dir / "train_sceneGraphs.json",
            self.data_dir / "train_sceneGraphs.json",
            self.data_dir / "sceneGraphs" / "train_sceneGraphs.json",
            Path("train_sceneGraphs.json"),
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return None
    
    def load_scene_graphs(self, max_images: int = 1000) -> Dict[str, Any]:
        """
        Read scene graphs data from JSON file.
        
        Args:
            max_images: Maximum number of images to process (default 1000)
            
        Returns:
            Dictionary containing scene graphs of images
        """
        # Find file
        json_path = self.find_scene_graph_file()
        
        if json_path is None:
            raise FileNotFoundError(
                "train_sceneGraphs.json file not found. "
                "Please ensure the file exists in sceneGraphs/ or data/gqa/ directory"
            )
        
        print(f"📂 Reading file: {json_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parse error: {e}")
        except Exception as e:
            raise IOError(f"File read error: {e}")
        
        # Get first max_images
        image_ids = list(all_data.keys())[:max_images]
        data = {img_id: all_data[img_id] for img_id in image_ids}
        
        print(f"   ✅ Read {len(data)} images (from total of {len(all_data)} images)")
        
        return data
    
    def _normalize_name(self, name: str) -> str:
        """Normalize name (lowercase, strip)."""
        return name.lower().strip()
    
    def _create_instance_node_id(self, image_id: str, object_id: str) -> str:
        """Create ID for Instance Node."""
        return f"{image_id}:{object_id}"
    
    def _create_concept_node_id(self, concept_name: str) -> str:
        """Create ID for Global Concept Node."""
        return f"Concept:{self._normalize_name(concept_name)}"
    
    def _create_attribute_node_id(self, attribute_name: str) -> str:
        """Create ID for Global Attribute Node."""
        return f"Attr:{self._normalize_name(attribute_name)}"
    
    def _add_global_concept_node(self, concept_name: str) -> str:
        """
        Add Global Concept Node to graph if not already exists.
        
        Returns:
            Node ID
        """
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
        """
        Add Global Attribute Node to graph if not already exists.
        
        Returns:
            Node ID
        """
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
        """
        Add Instance Node (specific object in image) to graph.
        
        Returns:
            Node ID
        """
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
        """Add instance_of edge from Instance Node to Global Concept Node."""
        self.graph.add_edge(
            instance_node_id,
            concept_node_id,
            edge_type='instance_of',
            relation='instance_of'
        )
        self.stats['instance_of_edges'] += 1
    
    def _add_has_attribute_edge(self, instance_node_id: str, attribute_node_id: str):
        """Add has_attribute edge from Instance Node to Global Attribute Node."""
        self.graph.add_edge(
            instance_node_id,
            attribute_node_id,
            edge_type='has_attribute',
            relation='has_attribute'
        )
        self.stats['has_attribute_edges'] += 1
    
    def _add_semantic_relation_edge(self, source_node_id: str, target_node_id: str,
                                     relation_name: str, image_id: str):
        """Add semantic_relation edge between 2 Instance Nodes."""
        self.graph.add_edge(
            source_node_id,
            target_node_id,
            edge_type='semantic_relation',
            relation=relation_name,
            image_id=image_id
        )
        self.stats['semantic_relation_edges'] += 1
    
    def build_graph(self, scene_graphs: Dict[str, Any]) -> nx.DiGraph:
        """
        Build Knowledge Graph from scene graphs data.
        
        Args:
            scene_graphs: Dictionary containing scene graphs of images
            
        Returns:
            Built NetworkX DiGraph
        """
        print("\n🔨 Starting Knowledge Graph construction...")
        print("=" * 60)
        
        self.stats['total_images'] = len(scene_graphs)
        
        # Iterate through each image
        for image_id, image_data in tqdm(scene_graphs.items(), 
                                          desc="📊 Processing images",
                                          unit="images"):
            objects = image_data.get('objects', {})
            self.stats['total_objects'] += len(objects)
            
            # Step 1: Add all Instance Nodes for this image
            for object_id, obj_data in objects.items():
                # Create Instance Node
                instance_node_id = self._add_instance_node(
                    image_id, object_id, obj_data
                )
                
                # Step 2: Thêm Global Concept Node và tạo cạnh instance_of
                obj_name = obj_data.get('name', '')
                if obj_name:
                    concept_node_id = self._add_global_concept_node(obj_name)
                    self._add_instance_of_edge(instance_node_id, concept_node_id)
                
                # Step 3: Thêm Global Attribute Nodes và tạo cạnh has_attribute
                attributes = obj_data.get('attributes', [])
                for attr in attributes:
                    if attr:  # Bỏ qua attribute rỗng
                        attr_node_id = self._add_global_attribute_node(attr)
                        self._add_has_attribute_edge(instance_node_id, attr_node_id)
            
            # Step 4: Thêm các cạnh semantic_relation giữa các objects
            for object_id, obj_data in objects.items():
                source_node_id = self._create_instance_node_id(image_id, object_id)
                relations = obj_data.get('relations', [])
                
                for rel in relations:
                    target_object_id = rel.get('object', '')
                    relation_name = rel.get('name', '')
                    
                    if target_object_id and relation_name:
                        # Check if target object exists in same image
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
        """Print detailed statistics about the graph."""
        print("\n" + "=" * 60)
        print("📈 KNOWLEDGE GRAPH STATISTICS")
        print("=" * 60)
        
        print("\n📷 Input Data:")
        print(f"   • Number of images processed: {self.stats['total_images']:,}")
        print(f"   • Total objects: {self.stats['total_objects']:,}")
        
        print("\n🔵 Nodes:")
        total_nodes = self.graph.number_of_nodes()
        print(f"   • Total nodes: {total_nodes:,}")
        print(f"   • Instance Nodes (specific objects): {self.stats['instance_nodes']:,}")
        print(f"   • Global Concept Nodes: {self.stats['concept_nodes']:,}")
        print(f"   • Global Attribute Nodes: {self.stats['attribute_nodes']:,}")
        
        print("\n🔗 Edges:")
        total_edges = self.graph.number_of_edges()
        print(f"   • Total edges: {total_edges:,}")
        print(f"   • instance_of edges: {self.stats['instance_of_edges']:,}")
        print(f"   • has_attribute edges: {self.stats['has_attribute_edges']:,}")
        print(f"   • semantic_relation edges: {self.stats['semantic_relation_edges']:,}")
        
        # Additional statistics
        print("\n📊 Additional Statistics:")
        if self.stats['instance_nodes'] > 0:
            avg_attrs = self.stats['has_attribute_edges'] / self.stats['instance_nodes']
            avg_rels = self.stats['semantic_relation_edges'] / self.stats['instance_nodes']
            print(f"   • Average attributes/object: {avg_attrs:.2f}")
            print(f"   • Average relations/object: {avg_rels:.2f}")
        
        # Top concepts
        print("\n🏆 Top 10 Global Concepts Most Common:")
        concept_counts = defaultdict(int)
        for node_id in self.global_concepts:
            count = self.graph.in_degree(node_id)
            concept_name = self.graph.nodes[node_id]['name']
            concept_counts[concept_name] = count
        
        for i, (concept, count) in enumerate(
            sorted(concept_counts.items(), key=lambda x: -x[1])[:10], 1
        ):
            print(f"   {i:2}. {concept}: {count:,} instances")
        
        # Top attributes
        print("\n🎨 Top 10 Most Common Global Attributes:")
        attr_counts = defaultdict(int)
        for node_id in self.global_attributes:
            count = self.graph.in_degree(node_id)
            attr_name = self.graph.nodes[node_id]['name']
            attr_counts[attr_name] = count
        
        for i, (attr, count) in enumerate(
            sorted(attr_counts.items(), key=lambda x: -x[1])[:10], 1
        ):
            print(f"   {i:2}. {attr}: {count:,} objects")
        
        # Top relations
        print("\n🔀 Top 10 Most Common Semantic Relations:")
        relation_counts = defaultdict(int)
        for u, v, data in self.graph.edges(data=True):
            if data.get('edge_type') == 'semantic_relation':
                relation_counts[data['relation']] += 1
        
        for i, (rel, count) in enumerate(
            sorted(relation_counts.items(), key=lambda x: -x[1])[:10], 1
        ):
            print(f"   {i:2}. '{rel}': {count:,} times")
        
        print("\n" + "=" * 60)
    
    def save_graph(self, output_path: str = "gqa_lightrag.gpickle") -> str:
        """
        Save graph to pickle file.
        
        Args:
            output_path: Output file path
            
        Returns:
            Saved file path
        """
        output_path = Path(output_path)
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving graph to: {output_path}")
        
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(self.graph, f, pickle.HIGHEST_PROTOCOL)
            
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            print(f"   ✅ Saved successfully! File size: {file_size:.2f} MB")
            
            return str(output_path)
        except Exception as e:
            print(f"   ❌ Error saving: {e}")
            raise
    
    @staticmethod
    def load_graph(input_path: str = "gqa_lightrag.gpickle") -> nx.DiGraph:
        """
        Load graph from pickle file.
        
        Args:
            input_path: Input file path
            
        Returns:
            NetworkX DiGraph
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        
        print(f"📂 Reading graph from: {input_path}")
        
        with open(input_path, 'rb') as f:
            graph = pickle.load(f)
        
        print(f"   ✅ Read successfully! Nodes: {graph.number_of_nodes():,}, Edges: {graph.number_of_edges():,}")
        
        return graph


def main():
    """Main function to run the entire pipeline."""
    print("=" * 60)
    print("🚀 GQA LIGHTRAG KNOWLEDGE GRAPH BUILDER")
    print("=" * 60)
    print("Building Multimodal Knowledge Graph from GQA Dataset")
    print("Architecture: LightRAG (Instance Level + Global Level)")
    print("=" * 60)
    
    # Initialize builder
    builder = GQALightRAGGraphBuilder(
        data_dir="data/gqa",
        scene_graphs_dir="sceneGraphs"
    )
    
    # Step 1: Extract zip file if exists
    print("\n📦 Step 1: Checking and extracting data...")
    builder.extract_zip_if_exists()
    
    # Step 2: Load data
    print("\n📂 Step 2: Reading scene graphs data...")
    try:
        scene_graphs = builder.load_scene_graphs(max_images=1000)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unknown error: {e}")
        return None
    
    # Step 3: Build graph
    print("\n🔨 Step 3: Building Knowledge Graph...")
    graph = builder.build_graph(scene_graphs)
    
    # Step 4: Print statistics
    print("\n📈 Step 4: Graph statistics...")
    builder.print_statistics()
    
    # Step 5: Save graph
    print("\n💾 Step 5: Saving graph...")
    output_file = builder.save_graph("gqa_lightrag.gpickle")
    
    print("\n" + "=" * 60)
    print("🎉 COMPLETED!")
    print(f"   Graph saved at: {output_file}")
    print("=" * 60)
    
    return graph


if __name__ == "__main__":
    graph = main()
