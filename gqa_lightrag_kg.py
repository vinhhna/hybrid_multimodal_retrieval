"""
GQA LightRAG Knowledge Graph Builder
=====================================
Script xây dựng Multimodal Knowledge Graph từ tập dữ liệu GQA (Visual Reasoning)
sử dụng kiến trúc LightRAG với 2 tầng: Instance Level và Global Level.

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
    raise ImportError("Vui lòng cài đặt networkx: pip install networkx")

try:
    from tqdm import tqdm
except ImportError:
    raise ImportError("Vui lòng cài đặt tqdm: pip install tqdm")


class GQALightRAGGraphBuilder:
    """
    Builder class để xây dựng Knowledge Graph theo kiến trúc LightRAG từ GQA dataset.
    
    Kiến trúc đồ thị:
    - Tầng Thực thể (Instance Level): Mỗi vật thể trong ảnh là một node
    - Tầng Khái niệm (Global Level): Các node Global Concept và Global Attribute
    
    Các loại cạnh:
    - instance_of: Nối Instance Node → Global Concept Node
    - has_attribute: Nối Instance Node → Global Attribute Node  
    - semantic_relation: Nối các Instance Nodes trong cùng một ảnh
    """
    
    def __init__(self, data_dir: str = "data/gqa", scene_graphs_dir: str = "sceneGraphs"):
        """
        Khởi tạo Graph Builder.
        
        Args:
            data_dir: Thư mục chứa dữ liệu đầu ra
            scene_graphs_dir: Thư mục chứa file scene graphs gốc
        """
        self.data_dir = Path(data_dir)
        self.scene_graphs_dir = Path(scene_graphs_dir)
        self.graph = nx.DiGraph()  # Directed graph cho các quan hệ có hướng
        
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
        
    def extract_zip_if_exists(self) -> bool:
        """
        Giải nén file sceneGraphs.json.zip nếu tồn tại.
        
        Returns:
            True nếu giải nén thành công hoặc file đã được giải nén trước đó
        """
        zip_patterns = [
            self.scene_graphs_dir / "sceneGraphs.json.zip",
            self.scene_graphs_dir / "train_sceneGraphs.json.zip",
            self.data_dir / "sceneGraphs.json.zip",
        ]
        
        for zip_path in zip_patterns:
            if zip_path.exists():
                print(f"📦 Tìm thấy file zip: {zip_path}")
                try:
                    # Tạo thư mục đích nếu chưa tồn tại
                    self.data_dir.mkdir(parents=True, exist_ok=True)
                    
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        print(f"   Đang giải nén vào {self.data_dir}...")
                        zip_ref.extractall(self.data_dir)
                    print("   ✅ Giải nén thành công!")
                    return True
                except zipfile.BadZipFile:
                    print(f"   ❌ Lỗi: File zip bị hỏng: {zip_path}")
                    return False
                except Exception as e:
                    print(f"   ❌ Lỗi khi giải nén: {e}")
                    return False
        
        print("ℹ️  Không tìm thấy file zip, sẽ đọc trực tiếp từ file JSON.")
        return True
    
    def find_scene_graph_file(self) -> Optional[Path]:
        """
        Tìm file train_sceneGraphs.json trong các vị trí có thể.
        
        Returns:
            Path đến file nếu tìm thấy, None nếu không
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
        Đọc dữ liệu scene graphs từ file JSON.
        
        Args:
            max_images: Số lượng ảnh tối đa để xử lý (mặc định 1000)
            
        Returns:
            Dictionary chứa scene graphs của các ảnh
        """
        # Tìm file
        json_path = self.find_scene_graph_file()
        
        if json_path is None:
            raise FileNotFoundError(
                "Không tìm thấy file train_sceneGraphs.json. "
                "Vui lòng đảm bảo file tồn tại trong thư mục sceneGraphs/ hoặc data/gqa/"
            )
        
        print(f"📂 Đang đọc file: {json_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Lỗi parse JSON: {e}")
        except Exception as e:
            raise IOError(f"Lỗi đọc file: {e}")
        
        # Lấy max_images ảnh đầu tiên
        image_ids = list(all_data.keys())[:max_images]
        data = {img_id: all_data[img_id] for img_id in image_ids}
        
        print(f"   ✅ Đã đọc {len(data)} ảnh (từ tổng số {len(all_data)} ảnh)")
        
        return data
    
    def _normalize_name(self, name: str) -> str:
        """Chuẩn hóa tên (lowercase, strip)."""
        return name.lower().strip()
    
    def _create_instance_node_id(self, image_id: str, object_id: str) -> str:
        """Tạo ID cho Instance Node."""
        return f"{image_id}:{object_id}"
    
    def _create_concept_node_id(self, concept_name: str) -> str:
        """Tạo ID cho Global Concept Node."""
        return f"Concept:{self._normalize_name(concept_name)}"
    
    def _create_attribute_node_id(self, attribute_name: str) -> str:
        """Tạo ID cho Global Attribute Node."""
        return f"Attr:{self._normalize_name(attribute_name)}"
    
    def _add_global_concept_node(self, concept_name: str) -> str:
        """
        Thêm Global Concept Node vào đồ thị nếu chưa tồn tại.
        
        Returns:
            ID của node
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
        Thêm Global Attribute Node vào đồ thị nếu chưa tồn tại.
        
        Returns:
            ID của node
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
        Thêm Instance Node (vật thể cụ thể trong ảnh) vào đồ thị.
        
        Returns:
            ID của node
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
        """Thêm cạnh instance_of từ Instance Node đến Global Concept Node."""
        self.graph.add_edge(
            instance_node_id,
            concept_node_id,
            edge_type='instance_of',
            relation='instance_of'
        )
        self.stats['instance_of_edges'] += 1
    
    def _add_has_attribute_edge(self, instance_node_id: str, attribute_node_id: str):
        """Thêm cạnh has_attribute từ Instance Node đến Global Attribute Node."""
        self.graph.add_edge(
            instance_node_id,
            attribute_node_id,
            edge_type='has_attribute',
            relation='has_attribute'
        )
        self.stats['has_attribute_edges'] += 1
    
    def _add_semantic_relation_edge(self, source_node_id: str, target_node_id: str,
                                     relation_name: str, image_id: str):
        """Thêm cạnh semantic_relation giữa 2 Instance Nodes."""
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
        Xây dựng Knowledge Graph từ scene graphs data.
        
        Args:
            scene_graphs: Dictionary chứa scene graphs của các ảnh
            
        Returns:
            NetworkX DiGraph đã xây dựng
        """
        print("\n🔨 Bắt đầu xây dựng Knowledge Graph...")
        print("=" * 60)
        
        self.stats['total_images'] = len(scene_graphs)
        
        # Duyệt qua từng ảnh
        for image_id, image_data in tqdm(scene_graphs.items(), 
                                          desc="📊 Xử lý ảnh",
                                          unit="ảnh"):
            objects = image_data.get('objects', {})
            self.stats['total_objects'] += len(objects)
            
            # Bước 1: Thêm tất cả Instance Nodes cho ảnh này
            for object_id, obj_data in objects.items():
                # Tạo Instance Node
                instance_node_id = self._add_instance_node(
                    image_id, object_id, obj_data
                )
                
                # Bước 2: Thêm Global Concept Node và tạo cạnh instance_of
                obj_name = obj_data.get('name', '')
                if obj_name:
                    concept_node_id = self._add_global_concept_node(obj_name)
                    self._add_instance_of_edge(instance_node_id, concept_node_id)
                
                # Bước 3: Thêm Global Attribute Nodes và tạo cạnh has_attribute
                attributes = obj_data.get('attributes', [])
                for attr in attributes:
                    if attr:  # Bỏ qua attribute rỗng
                        attr_node_id = self._add_global_attribute_node(attr)
                        self._add_has_attribute_edge(instance_node_id, attr_node_id)
            
            # Bước 4: Thêm các cạnh semantic_relation giữa các objects
            for object_id, obj_data in objects.items():
                source_node_id = self._create_instance_node_id(image_id, object_id)
                relations = obj_data.get('relations', [])
                
                for rel in relations:
                    target_object_id = rel.get('object', '')
                    relation_name = rel.get('name', '')
                    
                    if target_object_id and relation_name:
                        # Kiểm tra target object tồn tại trong cùng ảnh
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
        
        print("\n✅ Hoàn thành xây dựng Knowledge Graph!")
        return self.graph
    
    def print_statistics(self):
        """In thống kê chi tiết về đồ thị."""
        print("\n" + "=" * 60)
        print("📈 THỐNG KÊ KNOWLEDGE GRAPH")
        print("=" * 60)
        
        print("\n📷 Dữ liệu đầu vào:")
        print(f"   • Số lượng ảnh xử lý: {self.stats['total_images']:,}")
        print(f"   • Tổng số objects: {self.stats['total_objects']:,}")
        
        print("\n🔵 Nodes:")
        total_nodes = self.graph.number_of_nodes()
        print(f"   • Tổng số nodes: {total_nodes:,}")
        print(f"   • Instance Nodes (vật thể cụ thể): {self.stats['instance_nodes']:,}")
        print(f"   • Global Concept Nodes: {self.stats['concept_nodes']:,}")
        print(f"   • Global Attribute Nodes: {self.stats['attribute_nodes']:,}")
        
        print("\n🔗 Edges:")
        total_edges = self.graph.number_of_edges()
        print(f"   • Tổng số edges: {total_edges:,}")
        print(f"   • instance_of edges: {self.stats['instance_of_edges']:,}")
        print(f"   • has_attribute edges: {self.stats['has_attribute_edges']:,}")
        print(f"   • semantic_relation edges: {self.stats['semantic_relation_edges']:,}")
        
        # Thống kê thêm
        print("\n📊 Thống kê bổ sung:")
        if self.stats['instance_nodes'] > 0:
            avg_attrs = self.stats['has_attribute_edges'] / self.stats['instance_nodes']
            avg_rels = self.stats['semantic_relation_edges'] / self.stats['instance_nodes']
            print(f"   • Trung bình attributes/object: {avg_attrs:.2f}")
            print(f"   • Trung bình relations/object: {avg_rels:.2f}")
        
        # Top concepts
        print("\n🏆 Top 10 Global Concepts phổ biến nhất:")
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
        print("\n🎨 Top 10 Global Attributes phổ biến nhất:")
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
        print("\n🔀 Top 10 Semantic Relations phổ biến nhất:")
        relation_counts = defaultdict(int)
        for u, v, data in self.graph.edges(data=True):
            if data.get('edge_type') == 'semantic_relation':
                relation_counts[data['relation']] += 1
        
        for i, (rel, count) in enumerate(
            sorted(relation_counts.items(), key=lambda x: -x[1])[:10], 1
        ):
            print(f"   {i:2}. '{rel}': {count:,} lần")
        
        print("\n" + "=" * 60)
    
    def save_graph(self, output_path: str = "gqa_lightrag.gpickle") -> str:
        """
        Lưu đồ thị vào file pickle.
        
        Args:
            output_path: Đường dẫn file đầu ra
            
        Returns:
            Đường dẫn file đã lưu
        """
        output_path = Path(output_path)
        
        # Đảm bảo thư mục tồn tại
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Đang lưu đồ thị vào: {output_path}")
        
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(self.graph, f, pickle.HIGHEST_PROTOCOL)
            
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            print(f"   ✅ Đã lưu thành công! Kích thước file: {file_size:.2f} MB")
            
            return str(output_path)
        except Exception as e:
            print(f"   ❌ Lỗi khi lưu: {e}")
            raise
    
    @staticmethod
    def load_graph(input_path: str = "gqa_lightrag.gpickle") -> nx.DiGraph:
        """
        Đọc đồ thị từ file pickle.
        
        Args:
            input_path: Đường dẫn file đầu vào
            
        Returns:
            NetworkX DiGraph
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {input_path}")
        
        print(f"📂 Đang đọc đồ thị từ: {input_path}")
        
        with open(input_path, 'rb') as f:
            graph = pickle.load(f)
        
        print(f"   ✅ Đã đọc thành công! Nodes: {graph.number_of_nodes():,}, Edges: {graph.number_of_edges():,}")
        
        return graph


def main():
    """Hàm main để chạy toàn bộ pipeline."""
    print("=" * 60)
    print("🚀 GQA LIGHTRAG KNOWLEDGE GRAPH BUILDER")
    print("=" * 60)
    print("Xây dựng Multimodal Knowledge Graph từ GQA Dataset")
    print("Kiến trúc: LightRAG (Instance Level + Global Level)")
    print("=" * 60)
    
    # Khởi tạo builder
    builder = GQALightRAGGraphBuilder(
        data_dir="data/gqa",
        scene_graphs_dir="sceneGraphs"
    )
    
    # Bước 1: Giải nén file zip nếu có
    print("\n📦 Bước 1: Kiểm tra và giải nén dữ liệu...")
    builder.extract_zip_if_exists()
    
    # Bước 2: Đọc dữ liệu
    print("\n📂 Bước 2: Đọc dữ liệu scene graphs...")
    try:
        scene_graphs = builder.load_scene_graphs(max_images=1000)
    except FileNotFoundError as e:
        print(f"❌ Lỗi: {e}")
        return None
    except Exception as e:
        print(f"❌ Lỗi không xác định: {e}")
        return None
    
    # Bước 3: Xây dựng đồ thị
    print("\n🔨 Bước 3: Xây dựng Knowledge Graph...")
    graph = builder.build_graph(scene_graphs)
    
    # Bước 4: In thống kê
    print("\n📈 Bước 4: Thống kê đồ thị...")
    builder.print_statistics()
    
    # Bước 5: Lưu đồ thị
    print("\n💾 Bước 5: Lưu đồ thị...")
    output_file = builder.save_graph("gqa_lightrag.gpickle")
    
    print("\n" + "=" * 60)
    print("🎉 HOÀN THÀNH!")
    print(f"   Đồ thị đã được lưu tại: {output_file}")
    print("=" * 60)
    
    return graph


if __name__ == "__main__":
    graph = main()
