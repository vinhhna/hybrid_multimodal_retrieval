"""
GQA LightRAG Reasoning Engine - Part 1
=======================================
Công cụ suy luận trên Multimodal Knowledge Graph từ GQA Dataset.
Sử dụng kiến trúc LightRAG với Graph Traversal algorithms.

Các loại truy vấn được hỗ trợ trong Part 1:
1. Entity Search - Tìm vật thể theo thuộc tính
2. Statistical Knowledge - Tính xác suất xuất hiện chung
3. Similarity Search - Tìm vật thể tương đồng
4. Relational Path - Tìm đường đi giữa các vật thể
5. Negative Constraints - Tìm images có A nhưng không có B

Author: Generated for IT3930E - Project III
Date: 2024
"""

import pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional, Union
from dataclasses import dataclass, field

try:
    import networkx as nx
except ImportError:
    raise ImportError("Vui lòng cài đặt networkx: pip install networkx")


# ============================================================================
# DATA CLASSES CHO KẾT QUẢ TRẢ VỀ
# ============================================================================

@dataclass
class ReasoningResult:
    """
    Cấu trúc dữ liệu chuẩn cho kết quả suy luận.
    
    Attributes:
        query_type: Loại truy vấn (Entity, Statistical, etc.)
        question: Câu hỏi dạng text
        results: Dữ liệu kết quả (Image IDs, Object IDs, etc.)
        reasoning_trace: Danh sách các bước suy luận trên đồ thị
        metadata: Thông tin bổ sung (số lượng kết quả, thời gian, etc.)
    """
    query_type: str
    question: str
    results: Any
    reasoning_trace: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def print_result(self):
        """In kết quả theo định dạng chuẩn."""
        print("\n" + "=" * 70)
        print(f"📋 QUERY TYPE: {self.query_type}")
        print("=" * 70)
        print(f"\n❓ QUESTION: {self.question}")
        
        print(f"\n📊 RESULTS:")
        if isinstance(self.results, list):
            if len(self.results) == 0:
                print("   (Không có kết quả)")
            elif len(self.results) <= 10:
                for i, item in enumerate(self.results, 1):
                    print(f"   {i}. {item}")
            else:
                for i, item in enumerate(self.results[:10], 1):
                    print(f"   {i}. {item}")
                print(f"   ... và {len(self.results) - 10} kết quả khác")
        elif isinstance(self.results, dict):
            for key, value in list(self.results.items())[:10]:
                print(f"   • {key}: {value}")
        else:
            print(f"   {self.results}")
        
        print(f"\n🔍 REASONING TRACE:")
        for i, step in enumerate(self.reasoning_trace, 1):
            print(f"   Step {i}: {step}")
        
        if self.metadata:
            print(f"\n📈 METADATA:")
            for key, value in self.metadata.items():
                print(f"   • {key}: {value}")
        
        print("=" * 70)


# ============================================================================
# GQA REASONING ENGINE CLASS
# ============================================================================

# Định nghĩa các đường dẫn đồ thị theo scale
GRAPH_PATHS = {
    '1k': 'experiments/sample_1k/gqa_lightrag.gpickle',
    '10k': 'experiments/sample_10k/gqa_lightrag.gpickle',
    'full': 'experiments/full/gqa_lightrag.gpickle',
}

def get_graph_path(scale: str = '1k') -> str:
    """
    Lấy đường dẫn đồ thị theo scale.
    
    Args:
        scale: '1k', '10k', hoặc 'full'
    """
    return GRAPH_PATHS.get(scale, GRAPH_PATHS['1k'])


class GQA_Reasoning_Engine:
    """
    Engine suy luận trên GQA Knowledge Graph.
    
    Sử dụng kiến trúc LightRAG với 2 tầng:
    - Instance Level: Các specific objects trong images (node_type='instance')
    - Global Level: Concepts và Attributes (node_type='global_concept', 'global_attribute')
    
    Edge Types:
    - instance_of: Instance → Concept
    - has_attribute: Instance → Attribute
    - semantic_relation: Instance ↔ Instance
    """
    
    def __init__(self, graph_path: Optional[str] = None, scale: str = '1k'):
        """
        Initialize Reasoning Engine.
        
        Args:
            graph_path: Path to graph pickle file (ưu tiên nếu có)
            scale: Scale của dataset ('1k', '10k', 'full') - dùng nếu graph_path=None
        """
        if graph_path:
            self.graph_path = Path(graph_path)
        else:
            self.graph_path = Path(get_graph_path(scale))
        
        self.scale = scale
        self.graph: nx.DiGraph = None
        
        # Cache để tăng tốc truy vấn
        self._instance_nodes: Set[str] = set()
        self._concept_nodes: Set[str] = set()
        self._attribute_nodes: Set[str] = set()
        self._image_to_objects: Dict[str, Set[str]] = defaultdict(set)
        self._concept_to_instances: Dict[str, Set[str]] = defaultdict(set)
        self._attribute_to_instances: Dict[str, Set[str]] = defaultdict(set)
        
        # Load đồ thị
        self._load_graph()
        self._build_cache()
    
    def _load_graph(self):
        """Load đồ thị từ file pickle."""
        if not self.graph_path.exists():
            raise FileNotFoundError(
                f"File not found đồ thị: {self.graph_path}\n"
                "Vui lòng chạy gqa_lightrag_kg.py trước để tạo đồ thị."
            )
        
        print(f"📂 Đang load đồ thị từ: {self.graph_path}")
        
        with open(self.graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        
        print(f"   ✅ Đã load thành công!")
        print(f"   📊 Nodes: {self.graph.number_of_nodes():,}")
        print(f"   🔗 Edges: {self.graph.number_of_edges():,}")
    
    def _build_cache(self):
        """Build cache để tăng tốc truy vấn."""
        print("   🔧 Đang xây dựng cache...")
        
        for node_id, data in self.graph.nodes(data=True):
            node_type = data.get('node_type', '')
            
            if node_type == 'instance':
                self._instance_nodes.add(node_id)
                image_id = data.get('image_id', '')
                if image_id:
                    self._image_to_objects[image_id].add(node_id)
            
            elif node_type == 'global_concept':
                self._concept_nodes.add(node_id)
                # Lấy tất cả instances của concept này
                for pred in self.graph.predecessors(node_id):
                    edge_data = self.graph.edges[pred, node_id]
                    if edge_data.get('edge_type') == 'instance_of':
                        self._concept_to_instances[node_id].add(pred)
            
            elif node_type == 'global_attribute':
                self._attribute_nodes.add(node_id)
                # Lấy tất cả instances có attribute này
                for pred in self.graph.predecessors(node_id):
                    edge_data = self.graph.edges[pred, node_id]
                    if edge_data.get('edge_type') == 'has_attribute':
                        self._attribute_to_instances[node_id].add(pred)
        
        print(f"   ✅ Cache đã sẵn sàng!")
        print(f"      • Instance nodes: {len(self._instance_nodes):,}")
        print(f"      • Concept nodes: {len(self._concept_nodes):,}")
        print(f"      • Attribute nodes: {len(self._attribute_nodes):,}")
        print(f"      • Images: {len(self._image_to_objects):,}")
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _normalize(self, text: str) -> str:
        """Chuẩn hóa text (lowercase, strip)."""
        return text.lower().strip()
    
    def _get_concept_node_id(self, concept_name: str) -> str:
        """Tạo ID cho Global Concept Node."""
        return f"Concept:{self._normalize(concept_name)}"
    
    def _get_attribute_node_id(self, attr_name: str) -> str:
        """Tạo ID cho Global Attribute Node."""
        return f"Attr:{self._normalize(attr_name)}"
    
    def _parse_instance_node_id(self, node_id: str) -> Tuple[str, str]:
        """
        Parse Instance Node ID thành image_id và object_id.
        
        Returns:
            Tuple (image_id, object_id)
        """
        parts = node_id.split(':')
        if len(parts) == 2:
            return parts[0], parts[1]
        return '', ''
    
    def _get_node_info(self, node_id: str) -> Dict[str, Any]:
        """Lấy thông tin chi tiết của một node."""
        if node_id in self.graph:
            return dict(self.graph.nodes[node_id])
        return {}
    
    def _get_instances_by_concept(self, concept_name: str) -> Set[str]:
        """Lấy tất cả instance nodes thuộc về một concept."""
        concept_node_id = self._get_concept_node_id(concept_name)
        return self._concept_to_instances.get(concept_node_id, set())
    
    def _get_instances_by_attribute(self, attr_name: str) -> Set[str]:
        """Lấy tất cả instance nodes có một attribute."""
        attr_node_id = self._get_attribute_node_id(attr_name)
        return self._attribute_to_instances.get(attr_node_id, set())
    
    def _get_semantic_neighbors(self, instance_node_id: str, 
                                 relation: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        Lấy các neighbors của một instance node qua semantic relations.
        
        Args:
            instance_node_id: ID của instance node
            relation: Lọc theo loại relation (optional)
            
        Returns:
            List of tuples (neighbor_node_id, relation_name)
        """
        neighbors = []
        
        # Duyệt các successors (outgoing edges)
        for succ in self.graph.successors(instance_node_id):
            if succ in self._instance_nodes:
                edge_data = self.graph.edges[instance_node_id, succ]
                if edge_data.get('edge_type') == 'semantic_relation':
                    rel_name = edge_data.get('relation', '')
                    if relation is None or self._normalize(relation) in self._normalize(rel_name):
                        neighbors.append((succ, rel_name))
        
        # Duyệt các predecessors (incoming edges)
        for pred in self.graph.predecessors(instance_node_id):
            if pred in self._instance_nodes:
                edge_data = self.graph.edges[pred, instance_node_id]
                if edge_data.get('edge_type') == 'semantic_relation':
                    rel_name = edge_data.get('relation', '')
                    if relation is None or self._normalize(relation) in self._normalize(rel_name):
                        neighbors.append((pred, rel_name))
        
        return neighbors
    
    # ========================================================================
    # QUERY 1: ENTITY SEARCH
    # ========================================================================
    
    def entity_search(self, 
                      concept: Optional[str] = None,
                      attributes: Optional[List[str]] = None,
                      limit: int = 20) -> ReasoningResult:
        """
        TÌM VẬT THỂ THEO THUỘC TÍNH VÀ BỐI CẢNH
        
        Logic suy luận:
        1. Nếu có concept: Tìm Global Concept Node → Lấy tất cả instances
        2. Nếu có attributes: Tìm Global Attribute Nodes → Lấy instances giao nhau
        3. Kết hợp: Lấy giao của 2 tập trên
        
        Args:
            concept: Tên loại vật thể (vd: "dog", "car")
            attributes: Danh sách thuộc tính (vd: ["red", "large"])
            limit: Số lượng kết quả tối đa
            
        Returns:
            ReasoningResult với danh sách instances phù hợp
        """
        trace = []
        candidates = None
        
        # Step 1: Tìm theo concept nếu có
        if concept:
            concept_node_id = self._get_concept_node_id(concept)
            
            if concept_node_id in self._concept_nodes:
                concept_instances = self._get_instances_by_concept(concept)
                trace.append(
                    f"HOP 1: Từ Global Concept '{concept_node_id}' → "
                    f"Tìm thấy {len(concept_instances)} instances qua cạnh 'instance_of'"
                )
                candidates = concept_instances
            else:
                trace.append(f"HOP 1: Không tìm thấy Concept Node '{concept_node_id}' trong đồ thị")
                candidates = set()
        
        # Step 2: Lọc theo attributes nếu có
        if attributes:
            for i, attr in enumerate(attributes):
                attr_node_id = self._get_attribute_node_id(attr)
                
                if attr_node_id in self._attribute_nodes:
                    attr_instances = self._get_instances_by_attribute(attr)
                    trace.append(
                        f"HOP {len(trace)+1}: Từ Global Attribute '{attr_node_id}' → "
                        f"Tìm thấy {len(attr_instances)} instances qua cạnh 'has_attribute'"
                    )
                    
                    if candidates is None:
                        candidates = attr_instances
                    else:
                        candidates = candidates.intersection(attr_instances)
                        trace.append(
                            f"         → Giao với tập hiện tại → Còn {len(candidates)} instances"
                        )
                else:
                    trace.append(f"HOP {len(trace)+1}: Không tìm thấy Attribute Node '{attr_node_id}'")
        
        # Step 3: Format kết quả
        if candidates is None:
            candidates = set()
        
        results = []
        for node_id in list(candidates)[:limit]:
            info = self._get_node_info(node_id)
            image_id, object_id = self._parse_instance_node_id(node_id)
            results.append({
                'node_id': node_id,
                'image_id': image_id,
                'object_id': object_id,
                'name': info.get('name', ''),
                'attributes': info.get('attributes', [])
            })
        
        trace.append(f"FINAL: Trả về {len(results)} kết quả (giới hạn {limit})")
        
        # Create câu hỏi mô tả
        question_parts = []
        if concept:
            question_parts.append(f"loại '{concept}'")
        if attributes:
            question_parts.append(f"thuộc tính {attributes}")
        question = f"Tìm vật thể có {' và '.join(question_parts)}"
        
        return ReasoningResult(
            query_type="1. Entity Search",
            question=question,
            results=results,
            reasoning_trace=trace,
            metadata={
                'total_matches': len(candidates),
                'returned': len(results),
                'concept': concept,
                'attributes': attributes
            }
        )
    
    # ========================================================================
    # QUERY 2: STATISTICAL KNOWLEDGE
    # ========================================================================
    
    def statistical_knowledge(self, 
                              concept_a: str, 
                              concept_b: str,
                              relation: Optional[str] = None) -> ReasoningResult:
        """
        TÍNH XÁC SUẤT XUẤT HIỆN CHUNG GIỮA CÁC CONCEPT
        
        Logic suy luận:
        1. Tìm tất cả instances của Concept A
        2. Với mỗi instance A, tìm các neighbors qua semantic relations
        3. Kiểm tra xem neighbor có phải là instance của Concept B không
        4. Tính P(B | gần A) = Số cặp (A,B) / Total A
        
        Args:
            concept_a: Concept nguồn (vd: "person")
            concept_b: Concept đích (vd: "chair")
            relation: Lọc theo loại quan hệ cụ thể (optional)
            
        Returns:
            ReasoningResult với xác suất và thống kê
        """
        trace = []
        
        # Step 1: Lấy instances của Concept A
        concept_a_node = self._get_concept_node_id(concept_a)
        instances_a = self._get_instances_by_concept(concept_a)
        trace.append(
            f"HOP 1: Global Concept '{concept_a_node}' → "
            f"{len(instances_a)} instances qua 'instance_of'"
        )
        
        # Step 2: Lấy instances của Concept B
        concept_b_node = self._get_concept_node_id(concept_b)
        instances_b = self._get_instances_by_concept(concept_b)
        trace.append(
            f"HOP 2: Global Concept '{concept_b_node}' → "
            f"{len(instances_b)} instances qua 'instance_of'"
        )
        
        if not instances_a or not instances_b:
            trace.append("FINAL: Không đủ dữ liệu để tính xác suất")
            return ReasoningResult(
                query_type="2. Statistical Knowledge",
                question=f"Tính P({concept_b} | gần {concept_a})",
                results={'probability': 0, 'co_occurrences': 0},
                reasoning_trace=trace,
                metadata={'error': 'Insufficient data'}
            )
        
        # Step 3: Đếm số cặp (A, B) có quan hệ semantic
        co_occurrences = 0
        relation_counts = defaultdict(int)
        example_pairs = []
        
        for inst_a in instances_a:
            # Lấy neighbors của instance A
            neighbors = self._get_semantic_neighbors(inst_a, relation)
            
            for neighbor_id, rel_name in neighbors:
                if neighbor_id in instances_b:
                    co_occurrences += 1
                    relation_counts[rel_name] += 1
                    
                    if len(example_pairs) < 5:
                        example_pairs.append({
                            'instance_a': inst_a,
                            'instance_b': neighbor_id,
                            'relation': rel_name
                        })
        
        trace.append(
            f"HOP 3: Duyệt {len(instances_a)} instances của '{concept_a}' → "
            f"Tìm thấy {co_occurrences} cặp có quan hệ với instances của '{concept_b}'"
        )
        
        # Step 4: Tính xác suất
        probability = co_occurrences / len(instances_a) if instances_a else 0
        trace.append(
            f"FINAL: P({concept_b} | gần {concept_a}) = "
            f"{co_occurrences}/{len(instances_a)} = {probability:.4f}"
        )
        
        # Create câu hỏi
        question = f"Xác suất tìm thấy '{concept_b}' gần '{concept_a}'"
        if relation:
            question += f" với quan hệ '{relation}'"
        question += " là bao nhiêu?"
        
        return ReasoningResult(
            query_type="2. Statistical Knowledge",
            question=question,
            results={
                'probability': round(probability, 4),
                'co_occurrences': co_occurrences,
                'total_concept_a': len(instances_a),
                'total_concept_b': len(instances_b),
                'relation_breakdown': dict(relation_counts),
                'example_pairs': example_pairs
            },
            reasoning_trace=trace,
            metadata={
                'concept_a': concept_a,
                'concept_b': concept_b,
                'relation_filter': relation
            }
        )
    
    # ========================================================================
    # QUERY 3: SIMILARITY SEARCH
    # ========================================================================
    
    def similarity_search(self, 
                          reference_node_id: Optional[str] = None,
                          concept: Optional[str] = None,
                          attributes: Optional[List[str]] = None,
                          min_common_attributes: int = 2,
                          limit: int = 20) -> ReasoningResult:
        """
        TÌM VẬT THỂ TƯƠNG ĐỒNG QUA THUỘC TÍNH/CONCEPT
        
        Logic suy luận:
        1. Nếu có reference_node_id: Lấy concept và attributes của node đó
        2. Tìm tất cả instances có cùng concept
        3. Tính similarity dựa trên số attributes chung (Jaccard-like)
        4. Sắp xếp theo độ tương đồng giảm dần
        
        Args:
            reference_node_id: Node ID tham chiếu (nếu có)
            concept: Tên concept để tìm kiếm
            attributes: Danh sách attributes mong muốn
            min_common_attributes: Số attributes chung tối thiểu
            limit: Số lượng kết quả tối đa
            
        Returns:
            ReasoningResult với danh sách instances tương đồng
        """
        trace = []
        
        # Step 1: Xác định concept và attributes tham chiếu
        if reference_node_id:
            ref_info = self._get_node_info(reference_node_id)
            if not ref_info:
                return ReasoningResult(
                    query_type="3. Similarity Search",
                    question=f"Tìm vật thể tương đồng với {reference_node_id}",
                    results=[],
                    reasoning_trace=[f"ERROR: Không tìm thấy node '{reference_node_id}'"],
                    metadata={'error': 'Node not found'}
                )
            
            concept = ref_info.get('name', '')
            attributes = ref_info.get('attributes', [])
            trace.append(
                f"HOP 1: Từ reference node '{reference_node_id}' → "
                f"Concept='{concept}', Attributes={attributes}"
            )
        else:
            trace.append(
                f"HOP 1: Sử dụng tham số đầu vào → "
                f"Concept='{concept}', Attributes={attributes}"
            )
        
        if not concept:
            return ReasoningResult(
                query_type="3. Similarity Search",
                question="Tìm vật thể tương đồng",
                results=[],
                reasoning_trace=["ERROR: Cần cung cấp concept hoặc reference_node_id"],
                metadata={'error': 'Missing concept'}
            )
        
        ref_attrs_set = set(self._normalize(a) for a in (attributes or []))
        
        # Step 2: Tìm tất cả instances cùng concept
        candidates = self._get_instances_by_concept(concept)
        trace.append(
            f"HOP 2: Global Concept 'Concept:{self._normalize(concept)}' → "
            f"{len(candidates)} candidate instances"
        )
        
        # Loại bỏ reference node khỏi candidates nếu có
        if reference_node_id and reference_node_id in candidates:
            candidates = candidates - {reference_node_id}
        
        # Step 3: Tính similarity cho mỗi candidate
        similarities = []
        
        for cand_id in candidates:
            cand_info = self._get_node_info(cand_id)
            cand_attrs = cand_info.get('attributes', [])
            cand_attrs_set = set(self._normalize(a) for a in cand_attrs)
            
            # Tính số attributes chung
            common_attrs = ref_attrs_set.intersection(cand_attrs_set)
            
            # Jaccard similarity
            union_size = len(ref_attrs_set.union(cand_attrs_set))
            similarity = len(common_attrs) / union_size if union_size > 0 else 0
            
            if len(common_attrs) >= min_common_attributes or (not ref_attrs_set and not cand_attrs_set):
                image_id, object_id = self._parse_instance_node_id(cand_id)
                similarities.append({
                    'node_id': cand_id,
                    'image_id': image_id,
                    'object_id': object_id,
                    'name': cand_info.get('name', ''),
                    'attributes': cand_attrs,
                    'common_attributes': list(common_attrs),
                    'similarity_score': round(similarity, 4)
                })
        
        trace.append(
            f"HOP 3: Tính similarity dựa trên attributes chung → "
            f"{len(similarities)} instances có ≥{min_common_attributes} attributes chung"
        )
        
        # Step 4: Sắp xếp và trả về
        similarities.sort(key=lambda x: -x['similarity_score'])
        results = similarities[:limit]
        
        trace.append(f"FINAL: Sắp xếp theo similarity score → Trả về top {len(results)} kết quả")
        
        # Create câu hỏi
        question = f"Tìm vật thể tương đồng với concept='{concept}'"
        if attributes:
            question += f" và attributes={attributes}"
        
        return ReasoningResult(
            query_type="3. Similarity Search",
            question=question,
            results=results,
            reasoning_trace=trace,
            metadata={
                'reference_node': reference_node_id,
                'concept': concept,
                'reference_attributes': attributes,
                'total_candidates': len(candidates),
                'total_similar': len(similarities),
                'min_common_attributes': min_common_attributes
            }
        )
    
    # ========================================================================
    # QUERY 4: RELATIONAL PATH
    # ========================================================================
    
    def relational_path(self,
                        source_concept: str,
                        target_concept: str,
                        via_attribute: Optional[str] = None,
                        via_relation: Optional[str] = None,
                        max_hops: int = 3,
                        limit: int = 10) -> ReasoningResult:
        """
        TÌM ĐƯỜNG ĐI GIỮA 2 VẬT THỂ KHÁC LOẠI
        
        Logic suy luận (Graph Traversal):
        1. Tìm tất cả instances của source_concept
        2. Với mỗi source instance, thực hiện DFS/BFS để tìm đường đi
        3. Kiểm tra điểm đến có phải là instance của target_concept không
        4. Nếu có via_attribute: Đường đi phải qua Global Attribute Node
        5. Nếu có via_relation: Đường đi phải có cạnh với relation cụ thể
        
        Ví dụ: Tìm đường "person" → "wearing" → "shirt" → "cùng màu" → "car"
        
        Args:
            source_concept: Concept nguồn (vd: "person")
            target_concept: Concept đích (vd: "car")
            via_attribute: Đi qua thuộc tính trung gian (vd: "red")
            via_relation: Loại quan hệ cần có trên đường đi
            max_hops: Số bước nhảy tối đa
            limit: Số đường đi tối đa trả về
            
        Returns:
            ReasoningResult với danh sách paths tìm được
        """
        trace = []
        
        # Step 1: Lấy instances của source và target concepts
        source_instances = self._get_instances_by_concept(source_concept)
        target_instances = self._get_instances_by_concept(target_concept)
        
        trace.append(
            f"HOP 1: Source '{source_concept}' → {len(source_instances)} instances, "
            f"Target '{target_concept}' → {len(target_instances)} instances"
        )
        
        if not source_instances or not target_instances:
            return ReasoningResult(
                query_type="4. Relational Path",
                question=f"Tìm đường đi từ '{source_concept}' đến '{target_concept}'",
                results=[],
                reasoning_trace=trace + ["ERROR: Không tìm thấy instances cho source hoặc target"],
                metadata={'error': 'No instances found'}
            )
        
        # Nếu có via_attribute, tìm các instances có attribute đó
        via_instances = None
        if via_attribute:
            via_instances = self._get_instances_by_attribute(via_attribute)
            trace.append(
                f"HOP 2: Via Attribute '{via_attribute}' → "
                f"{len(via_instances)} instances có thuộc tính này"
            )
        
        # Step 2: Tìm paths bằng DFS
        found_paths = []
        visited_pairs = set()  # Tránh trùng lặp
        
        def dfs_find_paths(current_node: str, target_set: Set[str], 
                           path: List[str], depth: int) -> List[List[str]]:
            """DFS để tìm tất cả đường đi từ current đến target."""
            paths = []
            
            if depth > max_hops:
                return paths
            
            if current_node in target_set:
                paths.append(path.copy())
                return paths
            
            # Lấy semantic neighbors
            neighbors = self._get_semantic_neighbors(current_node, via_relation)
            
            for neighbor_id, rel_name in neighbors:
                if neighbor_id not in path:  # Tránh cycle
                    new_path = path + [f"--[{rel_name}]-->", neighbor_id]
                    paths.extend(dfs_find_paths(neighbor_id, target_set, new_path, depth + 1))
            
            return paths
        
        trace.append(f"HOP 3: Thực hiện DFS từ mỗi source instance (max_hops={max_hops})")
        
        # Duyệt qua một số source instances (giới hạn để tránh quá lâu)
        source_sample = list(source_instances)[:100]
        
        for source_id in source_sample:
            # Nếu có via_attribute, chỉ tìm từ các source có attribute đó
            if via_instances and source_id not in via_instances:
                continue
            
            paths = dfs_find_paths(source_id, target_instances, [source_id], 0)
            
            for path in paths:
                # Create unique key để tránh duplicate
                path_key = (path[0], path[-1])
                if path_key not in visited_pairs:
                    visited_pairs.add(path_key)
                    
                    # Check via_attribute nếu có
                    if via_attribute:
                        # Ít nhất một node trong path phải có via_attribute
                        has_via_attr = False
                        for node in path:
                            if not node.startswith('--['):
                                if node in via_instances:
                                    has_via_attr = True
                                    break
                        if not has_via_attr:
                            continue
                    
                    found_paths.append(path)
                    
                    if len(found_paths) >= limit:
                        break
            
            if len(found_paths) >= limit:
                break
        
        trace.append(f"HOP 4: Tìm thấy {len(found_paths)} đường đi thỏa mãn")
        
        # Step 3: Format kết quả
        results = []
        for path in found_paths[:limit]:
            # Parse path để lấy thông tin chi tiết
            path_info = {
                'path': path,
                'path_length': len([p for p in path if not p.startswith('--[')]),
                'source_node': path[0],
                'target_node': path[-1],
                'relations': [p for p in path if p.startswith('--[')]
            }
            
            # Thêm thông tin về các nodes
            source_info = self._get_node_info(path[0])
            target_info = self._get_node_info(path[-1])
            path_info['source_name'] = source_info.get('name', '')
            path_info['target_name'] = target_info.get('name', '')
            path_info['source_image'] = source_info.get('image_id', '')
            path_info['target_image'] = target_info.get('image_id', '')
            
            results.append(path_info)
        
        trace.append(f"FINAL: Trả về {len(results)} đường đi")
        
        # Create câu hỏi
        question = f"Tìm đường đi từ '{source_concept}' đến '{target_concept}'"
        if via_attribute:
            question += f" qua thuộc tính '{via_attribute}'"
        if via_relation:
            question += f" với quan hệ '{via_relation}'"
        
        return ReasoningResult(
            query_type="4. Relational Path",
            question=question,
            results=results,
            reasoning_trace=trace,
            metadata={
                'source_concept': source_concept,
                'target_concept': target_concept,
                'via_attribute': via_attribute,
                'via_relation': via_relation,
                'max_hops': max_hops,
                'total_paths_found': len(found_paths)
            }
        )
    
    # ========================================================================
    # QUERY 5: NEGATIVE CONSTRAINTS
    # ========================================================================
    
    def negative_constraints(self,
                             concept_present: str,
                             concept_absent: str,
                             limit: int = 20) -> ReasoningResult:
        """
        TÌM ẢNH CÓ VẬT THỂ A NHƯNG KHÔNG CÓ VẬT THỂ B
        
        Logic suy luận:
        1. Tìm tất cả images chứa instances của concept_present
        2. Tìm tất cả images chứa instances của concept_absent
        3. Lấy hiệu: images_with_A - images_with_B
        
        Args:
            concept_present: Concept phải có mặt (vd: "dog")
            concept_absent: Concept không được có (vd: "cat")
            limit: Số lượng kết quả tối đa
            
        Returns:
            ReasoningResult với danh sách images thỏa mãn
        """
        trace = []
        
        # Step 1: Tìm images chứa concept_present
        instances_present = self._get_instances_by_concept(concept_present)
        images_with_present = set()
        
        for inst_id in instances_present:
            image_id, _ = self._parse_instance_node_id(inst_id)
            if image_id:
                images_with_present.add(image_id)
        
        trace.append(
            f"HOP 1: Concept '{concept_present}' → "
            f"{len(instances_present)} instances trong {len(images_with_present)} images"
        )
        
        # Step 2: Tìm images chứa concept_absent
        instances_absent = self._get_instances_by_concept(concept_absent)
        images_with_absent = set()
        
        for inst_id in instances_absent:
            image_id, _ = self._parse_instance_node_id(inst_id)
            if image_id:
                images_with_absent.add(image_id)
        
        trace.append(
            f"HOP 2: Concept '{concept_absent}' → "
            f"{len(instances_absent)} instances trong {len(images_with_absent)} images"
        )
        
        # Step 3: Lấy hiệu
        result_images = images_with_present - images_with_absent
        trace.append(
            f"HOP 3: Phép hiệu {len(images_with_present)} - {len(images_with_absent)} "
            f"= {len(result_images)} images thỏa mãn"
        )
        
        # Step 4: Thu thập thông tin chi tiết
        results = []
        for image_id in list(result_images)[:limit]:
            # Lấy các objects của concept_present trong images này
            objects_in_image = self._image_to_objects.get(image_id, set())
            present_objects = []
            
            for obj_id in objects_in_image:
                if obj_id in instances_present:
                    obj_info = self._get_node_info(obj_id)
                    _, object_id = self._parse_instance_node_id(obj_id)
                    present_objects.append({
                        'object_id': object_id,
                        'name': obj_info.get('name', ''),
                        'attributes': obj_info.get('attributes', [])
                    })
            
            results.append({
                'image_id': image_id,
                'total_objects': len(objects_in_image),
                f'{concept_present}_objects': present_objects
            })
        
        trace.append(f"FINAL: Trả về {len(results)} images (giới hạn {limit})")
        
        return ReasoningResult(
            query_type="5. Negative Constraints",
            question=f"Tìm images có '{concept_present}' nhưng KHÔNG có '{concept_absent}'",
            results=results,
            reasoning_trace=trace,
            metadata={
                'concept_present': concept_present,
                'concept_absent': concept_absent,
                'images_with_present': len(images_with_present),
                'images_with_absent': len(images_with_absent),
                'result_count': len(result_images)
            }
        )
    
    # ========================================================================
    # QUERY 6: COMPARATIVE (SO SÁNH)
    # ========================================================================
    
    def compare_contexts(self,
                         context_a: str,
                         context_b: str,
                         target_concept: str,
                         compare_attribute: Optional[str] = None) -> ReasoningResult:
        """
        SO SÁNH SỐ LƯỢNG HOẶC THUỘC TÍNH CỦA TARGET_CONCEPT GIỮA 2 BỐI CẢNH
        
        Logic suy luận:
        1. Tìm tất cả images chứa context_a (vd: kitchen) → Lọc instances của target_concept
        2. Tìm tất cả images chứa context_b (vd: living room) → Lọc instances của target_concept  
        3. So sánh số lượng hoặc phân bố thuộc tính giữa 2 tập
        
        Lưu ý: "Bối cimages" được xác định bằng cách tìm các images có chứa concept bối cimages đó.
        
        Args:
            context_a: Bối cimages thứ nhất (vd: "kitchen", "bedroom")
            context_b: Bối cimages thứ hai (vd: "living room", "bathroom")
            target_concept: Concept cần so sánh (vd: "chair", "table")
            compare_attribute: Thuộc tính cần so sánh (optional, vd: "white")
            
        Returns:
            ReasoningResult với kết quả so sánh
        """
        trace = []
        
        # Step 1: Tìm images chứa context_a
        context_a_instances = self._get_instances_by_concept(context_a)
        images_context_a = set()
        for inst in context_a_instances:
            img_id, _ = self._parse_instance_node_id(inst)
            if img_id:
                images_context_a.add(img_id)
        
        trace.append(
            f"HOP 1: Tìm bối cimages '{context_a}' → "
            f"{len(context_a_instances)} instances trong {len(images_context_a)} images"
        )
        
        # Step 2: Tìm images chứa context_b
        context_b_instances = self._get_instances_by_concept(context_b)
        images_context_b = set()
        for inst in context_b_instances:
            img_id, _ = self._parse_instance_node_id(inst)
            if img_id:
                images_context_b.add(img_id)
        
        trace.append(
            f"HOP 2: Tìm bối cimages '{context_b}' → "
            f"{len(context_b_instances)} instances trong {len(images_context_b)} images"
        )
        
        # Step 3: Tìm target_concept trong mỗi bối cimages
        target_instances = self._get_instances_by_concept(target_concept)
        trace.append(
            f"HOP 3: Tìm target concept '{target_concept}' → {len(target_instances)} instances tổng cộng"
        )
        
        # Đếm target trong context_a
        targets_in_a = []
        for inst in target_instances:
            img_id, obj_id = self._parse_instance_node_id(inst)
            if img_id in images_context_a:
                info = self._get_node_info(inst)
                targets_in_a.append({
                    'node_id': inst,
                    'image_id': img_id,
                    'object_id': obj_id,
                    'attributes': info.get('attributes', [])
                })
        
        # Đếm target trong context_b
        targets_in_b = []
        for inst in target_instances:
            img_id, obj_id = self._parse_instance_node_id(inst)
            if img_id in images_context_b:
                info = self._get_node_info(inst)
                targets_in_b.append({
                    'node_id': inst,
                    'image_id': img_id,
                    'object_id': obj_id,
                    'attributes': info.get('attributes', [])
                })
        
        trace.append(
            f"HOP 4: Lọc '{target_concept}' theo bối cimages → "
            f"Trong '{context_a}': {len(targets_in_a)}, Trong '{context_b}': {len(targets_in_b)}"
        )
        
        # Step 4: So sánh thuộc tính nếu có
        comparison_result = {
            'context_a': context_a,
            'context_b': context_b,
            'target_concept': target_concept,
            f'count_in_{context_a}': len(targets_in_a),
            f'count_in_{context_b}': len(targets_in_b),
            'winner_by_count': context_a if len(targets_in_a) > len(targets_in_b) 
                               else (context_b if len(targets_in_b) > len(targets_in_a) else 'tie')
        }
        
        if compare_attribute:
            # Đếm instances có attribute trong mỗi context
            attr_in_a = sum(1 for t in targets_in_a 
                           if self._normalize(compare_attribute) in 
                           [self._normalize(a) for a in t['attributes']])
            attr_in_b = sum(1 for t in targets_in_b 
                           if self._normalize(compare_attribute) in 
                           [self._normalize(a) for a in t['attributes']])
            
            comparison_result[f'{compare_attribute}_in_{context_a}'] = attr_in_a
            comparison_result[f'{compare_attribute}_in_{context_b}'] = attr_in_b
            comparison_result[f'winner_by_{compare_attribute}'] = (
                context_a if attr_in_a > attr_in_b 
                else (context_b if attr_in_b > attr_in_a else 'tie')
            )
            
            trace.append(
                f"HOP 5: So sánh thuộc tính '{compare_attribute}' → "
                f"Trong '{context_a}': {attr_in_a}, Trong '{context_b}': {attr_in_b}"
            )
        
        # Kết luận
        if len(targets_in_a) == 0 and len(targets_in_b) == 0:
            trace.append(
                f"FINAL: Không tìm thấy '{target_concept}' trong cả hai bối cimages trong tập mẫu"
            )
        else:
            winner = comparison_result['winner_by_count']
            trace.append(
                f"FINAL: '{context_a}' có {len(targets_in_a)} '{target_concept}', "
                f"'{context_b}' có {len(targets_in_b)} '{target_concept}' → "
                f"{'Hòa' if winner == 'tie' else f'{winner} nhiều hơn'}"
            )
        
        # Create câu hỏi
        question = f"So sánh số lượng '{target_concept}' trong bối cimages '{context_a}' và '{context_b}'"
        if compare_attribute:
            question = f"So sánh thuộc tính '{compare_attribute}' của '{target_concept}' trong '{context_a}' và '{context_b}'"
        
        return ReasoningResult(
            query_type="6. Comparative",
            question=question,
            results=comparison_result,
            reasoning_trace=trace,
            metadata={
                'context_a': context_a,
                'context_b': context_b,
                'target_concept': target_concept,
                'compare_attribute': compare_attribute,
                'sample_targets_a': targets_in_a[:3],
                'sample_targets_b': targets_in_b[:3]
            }
        )
    
    def compare_attribute_distribution(self,
                                        concept_a: str,
                                        concept_b: str,
                                        attribute: str) -> ReasoningResult:
        """
        SO SÁNH PHÂN BỐ THUỘC TÍNH GIỮA 2 LOẠI VẬT THỂ
        
        Logic suy luận:
        1. Tìm tất cả instances của concept_a có attribute
        2. Tìm tất cả instances của concept_b có attribute
        3. Tính tỷ lệ và so sánh
        
        Args:
            concept_a: Loại vật thể thứ nhất (vd: "wall")
            concept_b: Loại vật thể thứ hai (vd: "bed")
            attribute: Thuộc tính cần so sánh (vd: "white")
        """
        trace = []
        
        # Step 1: Tìm instances của concept_a
        instances_a = self._get_instances_by_concept(concept_a)
        trace.append(f"HOP 1: Concept '{concept_a}' → {len(instances_a)} instances")
        
        # Step 2: Tìm instances của concept_b
        instances_b = self._get_instances_by_concept(concept_b)
        trace.append(f"HOP 2: Concept '{concept_b}' → {len(instances_b)} instances")
        
        # Step 3: Tìm instances có attribute
        attr_instances = self._get_instances_by_attribute(attribute)
        trace.append(f"HOP 3: Attribute '{attribute}' → {len(attr_instances)} instances")
        
        # Step 4: Tính giao
        a_with_attr = instances_a.intersection(attr_instances)
        b_with_attr = instances_b.intersection(attr_instances)
        
        ratio_a = len(a_with_attr) / len(instances_a) if instances_a else 0
        ratio_b = len(b_with_attr) / len(instances_b) if instances_b else 0
        
        trace.append(
            f"HOP 4: Giao với attribute → "
            f"'{concept_a}' + '{attribute}': {len(a_with_attr)} ({ratio_a:.2%}), "
            f"'{concept_b}' + '{attribute}': {len(b_with_attr)} ({ratio_b:.2%})"
        )
        
        winner = concept_a if ratio_a > ratio_b else (concept_b if ratio_b > ratio_a else 'tie')
        trace.append(
            f"FINAL: Thuộc tính '{attribute}' phổ biến hơn ở "
            f"'{winner if winner != 'tie' else 'cả hai (hòa)'}'"
        )
        
        return ReasoningResult(
            query_type="6. Comparative",
            question=f"So sánh: Thuộc tính '{attribute}' phổ biến hơn ở '{concept_a}' hay '{concept_b}'?",
            results={
                'concept_a': concept_a,
                'concept_b': concept_b,
                'attribute': attribute,
                f'{concept_a}_total': len(instances_a),
                f'{concept_a}_with_{attribute}': len(a_with_attr),
                f'{concept_a}_ratio': round(ratio_a, 4),
                f'{concept_b}_total': len(instances_b),
                f'{concept_b}_with_{attribute}': len(b_with_attr),
                f'{concept_b}_ratio': round(ratio_b, 4),
                'winner': winner
            },
            reasoning_trace=trace,
            metadata={
                'comparison_type': 'attribute_distribution',
                'sample_a': list(a_with_attr)[:3],
                'sample_b': list(b_with_attr)[:3]
            }
        )
    
    # ========================================================================
    # QUERY 7: HIERARCHICAL (PHÂN CẤP)
    # ========================================================================
    
    # Dictionary ánh xạ phân cấp nội bộ
    HIERARCHY_MAPPING = {
        'furniture': ['chair', 'table', 'bed', 'sofa', 'cabinet', 'desk', 'couch', 'bench'],
        'electronic_devices': ['television', 'phone', 'laptop', 'camera', 'computer', 'tv', 'monitor'],
        'vehicle': ['car', 'bus', 'truck', 'motorcycle', 'bicycle', 'bike', 'train', 'airplane'],
        'animal': ['dog', 'cat', 'bird', 'horse', 'elephant', 'cow', 'sheep', 'bear'],
        'food': ['pizza', 'cake', 'bread', 'fruit', 'vegetable', 'meat', 'banana', 'apple'],
        'clothing': ['shirt', 'pants', 'jacket', 'hat', 'shoes', 'dress', 'coat', 'shorts'],
        'body_part': ['head', 'hand', 'arm', 'leg', 'face', 'hair', 'eye', 'nose'],
        'nature': ['tree', 'grass', 'flower', 'sky', 'cloud', 'mountain', 'water', 'sun'],
        'building_structure': ['building', 'house', 'wall', 'window', 'door', 'roof', 'floor', 'ceiling'],
        'kitchenware': ['plate', 'bowl', 'cup', 'glass', 'fork', 'knife', 'spoon', 'pot']
    }
    
    def get_hierarchical_entities(self,
                                   parent_category: str,
                                   limit: int = 50) -> ReasoningResult:
        """
        TÌM TẤT CẢ INSTANCES THUỘC MỘT CATEGORY CHA
        
        Logic suy luận:
        1. Tra cứu dictionary phân cấp để lấy danh sách child concepts
        2. Với mỗi child concept, tìm tất cả instances
        3. Gộp kết quả và trả về
        
        Args:
            parent_category: Tên category cha (vd: "furniture", "electronic_devices")
            limit: Số lượng kết quả tối đa cho mỗi child concept
            
        Returns:
            ReasoningResult với danh sách instances theo từng child concept
        """
        trace = []
        
        # Step 1: Tra cứu hierarchy mapping
        parent_key = self._normalize(parent_category).replace(' ', '_')
        child_concepts = self.HIERARCHY_MAPPING.get(parent_key, [])
        
        if not child_concepts:
            trace.append(f"HOP 1: Không tìm thấy category '{parent_category}' trong hierarchy mapping")
            trace.append(f"       Các categories có sẵn: {list(self.HIERARCHY_MAPPING.keys())}")
            return ReasoningResult(
                query_type="7. Hierarchical",
                question=f"Liệt kê tất cả thực thể thuộc loại '{parent_category}'",
                results=[],
                reasoning_trace=trace,
                metadata={'error': 'Category not found', 'available': list(self.HIERARCHY_MAPPING.keys())}
            )
        
        trace.append(
            f"HOP 1: Category '{parent_category}' → "
            f"Bao gồm các child concepts: {child_concepts}"
        )
        
        # Step 2: Tìm instances cho từng child concept
        results_by_concept = {}
        total_instances = 0
        
        for child in child_concepts:
            instances = self._get_instances_by_concept(child)
            results_by_concept[child] = {
                'count': len(instances),
                'sample_instances': []
            }
            total_instances += len(instances)
            
            # Lấy sample instances
            for inst_id in list(instances)[:5]:
                info = self._get_node_info(inst_id)
                img_id, obj_id = self._parse_instance_node_id(inst_id)
                results_by_concept[child]['sample_instances'].append({
                    'node_id': inst_id,
                    'image_id': img_id,
                    'object_id': obj_id,
                    'attributes': info.get('attributes', [])
                })
        
        trace.append(
            f"HOP 2: Duyệt {len(child_concepts)} child concepts → "
            f"Tổng cộng {total_instances} instances"
        )
        
        # Thống kê chi tiết
        concept_counts = [(c, d['count']) for c, d in results_by_concept.items()]
        concept_counts.sort(key=lambda x: -x[1])
        
        trace.append("HOP 3: Thống kê số lượng theo từng loại:")
        for concept, count in concept_counts:
            if count > 0:
                trace.append(f"       • {concept}: {count} instances")
        
        if total_instances == 0:
            trace.append(f"FINAL: Không tìm thấy thực thể nào thuộc '{parent_category}' trong tập mẫu")
        else:
            trace.append(f"FINAL: Tìm thấy tổng cộng {total_instances} thực thể thuộc '{parent_category}'")
        
        return ReasoningResult(
            query_type="7. Hierarchical",
            question=f"Liệt kê tất cả thực thể thuộc loại '{parent_category}' (bao gồm: {', '.join(child_concepts)})",
            results=results_by_concept,
            reasoning_trace=trace,
            metadata={
                'parent_category': parent_category,
                'child_concepts': child_concepts,
                'total_instances': total_instances,
                'concept_breakdown': dict(concept_counts)
            }
        )
    
    # ========================================================================
    # QUERY 8: ANOMALY DETECTION (PHÁT HIỆN BẤT THƯỜNG)
    # ========================================================================
    
    def find_anomalies(self,
                       min_frequency: int = 2,
                       limit: int = 20) -> ReasoningResult:
        """
        TÌM CÁC QUAN HỆ HIẾM GẶP (S-P-O) XUẤT HIỆN ÍT HƠN MIN_FREQUENCY LẦN
        
        Logic suy luận:
        1. Duyệt tất cả các cạnh semantic_relation trong đồ thị
        2. Đếm tần suất của mỗi bộ ba (subject_concept, relation, object_concept)
        3. Lọc các bộ ba có tần suất < min_frequency
        
        Args:
            min_frequency: Ngưỡng tần suất tối thiểu để coi là "bình thường"
            limit: Số lượng anomalies tối đa trả về
            
        Returns:
            ReasoningResult với danh sách các quan hệ hiếm gặp
        """
        trace = []
        
        # Step 1: Thu thập tất cả các bộ ba (S, P, O)
        trace.append("HOP 1: Duyệt tất cả các cạnh semantic_relation trong đồ thị...")
        
        spo_counts = defaultdict(list)  # (s_concept, relation, o_concept) -> list of instances
        
        for u, v, data in self.graph.edges(data=True):
            if data.get('edge_type') == 'semantic_relation':
                # Lấy concept của subject và object
                u_info = self._get_node_info(u)
                v_info = self._get_node_info(v)
                
                s_concept = u_info.get('name', 'unknown')
                o_concept = v_info.get('name', 'unknown')
                relation = data.get('relation', 'unknown')
                
                spo_key = (s_concept, relation, o_concept)
                spo_counts[spo_key].append({
                    'subject_node': u,
                    'object_node': v,
                    'subject_image': u_info.get('image_id', ''),
                    'object_image': v_info.get('image_id', '')
                })
        
        trace.append(f"HOP 2: Tìm thấy {len(spo_counts)} loại bộ ba (S-P-O) khác nhau")
        
        # Step 2: Lọc các bộ ba có tần suất thấp
        anomalies = []
        for spo, instances in spo_counts.items():
            if len(instances) < min_frequency:
                anomalies.append({
                    'subject_concept': spo[0],
                    'relation': spo[1],
                    'object_concept': spo[2],
                    'frequency': len(instances),
                    'instances': instances[:3]  # Lấy tối đa 3 ví dụ
                })
        
        # Sắp xếp theo tần suất tăng dần
        anomalies.sort(key=lambda x: x['frequency'])
        
        trace.append(
            f"HOP 3: Lọc các quan hệ có tần suất < {min_frequency} → "
            f"Tìm thấy {len(anomalies)} quan hệ hiếm gặp"
        )
        
        results = anomalies[:limit]
        
        if not results:
            trace.append(f"FINAL: Không tìm thấy quan hệ nào có tần suất < {min_frequency}")
        else:
            trace.append(f"FINAL: Trả về {len(results)} quan hệ hiếm gặp nhất")
            # Liệt kê top 5
            for i, anom in enumerate(results[:5], 1):
                trace.append(
                    f"       {i}. '{anom['subject_concept']}' --[{anom['relation']}]--> "
                    f"'{anom['object_concept']}' (xuất hiện {anom['frequency']} times)"
                )
        
        return ReasoningResult(
            query_type="8. Anomaly Detection",
            question=f"Tìm các quan hệ (S-P-O) có tần suất xuất hiện < {min_frequency} times",
            results=results,
            reasoning_trace=trace,
            metadata={
                'min_frequency': min_frequency,
                'total_spo_types': len(spo_counts),
                'total_anomalies': len(anomalies),
                'returned': len(results)
            }
        )
    
    def find_specific_anomaly(self,
                              subject_concept: str,
                              relation: str,
                              object_concept: str) -> ReasoningResult:
        """
        TÌM MỘT TRƯỜNG HỢP BẤT THƯỜNG CỤ THỂ
        
        Logic suy luận:
        1. Tìm tất cả instances của subject_concept
        2. Với mỗi instance, kiểm tra có quan hệ 'relation' với object_concept không
        3. Trả về các trường hợp tìm thấy
        
        Args:
            subject_concept: Loại vật thể chủ ngữ (vd: "dog")
            relation: Quan hệ cần tìm (vd: "on")
            object_concept: Loại vật thể tân ngữ (vd: "table")
        """
        trace = []
        
        # Step 1: Tìm instances của subject và object
        subject_instances = self._get_instances_by_concept(subject_concept)
        object_instances = self._get_instances_by_concept(object_concept)
        
        trace.append(
            f"HOP 1: Subject '{subject_concept}' → {len(subject_instances)} instances, "
            f"Object '{object_concept}' → {len(object_instances)} instances"
        )
        
        # Step 2: Tìm các cặp có quan hệ
        found_cases = []
        
        for subj_id in subject_instances:
            # Lấy neighbors qua relation cụ thể
            neighbors = self._get_semantic_neighbors(subj_id, relation)
            
            for neighbor_id, rel_name in neighbors:
                if neighbor_id in object_instances:
                    subj_info = self._get_node_info(subj_id)
                    obj_info = self._get_node_info(neighbor_id)
                    
                    found_cases.append({
                        'subject_node': subj_id,
                        'subject_name': subj_info.get('name', ''),
                        'subject_image': subj_info.get('image_id', ''),
                        'relation': rel_name,
                        'object_node': neighbor_id,
                        'object_name': obj_info.get('name', ''),
                        'object_image': obj_info.get('image_id', ''),
                        'subject_attributes': subj_info.get('attributes', []),
                        'object_attributes': obj_info.get('attributes', [])
                    })
        
        trace.append(
            f"HOP 2: Tìm kiếm quan hệ '{relation}' giữa '{subject_concept}' và '{object_concept}'"
        )
        
        if not found_cases:
            trace.append(
                f"FINAL: Không tìm thấy trường hợp '{subject_concept}' --[{relation}]--> "
                f"'{object_concept}' trong tập mẫu"
            )
        else:
            trace.append(
                f"FINAL: Tìm thấy {len(found_cases)} trường hợp "
                f"'{subject_concept}' --[{relation}]--> '{object_concept}'"
            )
        
        return ReasoningResult(
            query_type="8. Anomaly Detection",
            question=f"Tìm trường hợp '{subject_concept}' có quan hệ '{relation}' với '{object_concept}'",
            results=found_cases,
            reasoning_trace=trace,
            metadata={
                'subject_concept': subject_concept,
                'relation': relation,
                'object_concept': object_concept,
                'found_count': len(found_cases)
            }
        )
    
    # ========================================================================
    # QUERY 9: VISUAL-ATTRIBUTE CONSTRAINT (RÀNG BUỘC ĐA ĐIỀU KIỆN)
    # ========================================================================
    
    def multi_constraint_search(self,
                                 name: str,
                                 required_attributes: List[str],
                                 limit: int = 20) -> ReasoningResult:
        """
        TÌM VẬT THỂ KHỚP VỚI TÊN VÀ TẤT CẢ CÁC THUỘC TÍNH YÊU CẦU
        
        Logic suy luận:
        1. Tìm tất cả instances có concept = name
        2. Với mỗi attribute trong required_attributes, lấy instances có attribute đó
        3. Lấy giao của tất cả các tập → Chỉ giữ instances thỏa mãn TẤT CẢ điều kiện
        
        Args:
            name: Tên loại vật thể (vd: "cup", "man")
            required_attributes: Danh sách thuộc tính BẮT BUỘC phải có (vd: ["red", "plastic"])
            limit: Số lượng kết quả tối đa
            
        Returns:
            ReasoningResult với danh sách instances thỏa mãn tất cả điều kiện
        """
        trace = []
        
        # Step 1: Tìm instances của concept
        concept_instances = self._get_instances_by_concept(name)
        trace.append(
            f"HOP 1: Tìm concept '{name}' → {len(concept_instances)} instances"
        )
        
        if not concept_instances:
            trace.append(f"FINAL: Không tìm thấy concept '{name}' trong tập mẫu")
            return ReasoningResult(
                query_type="9. Visual-Attribute Constraint",
                question=f"Tìm '{name}' có các thuộc tính {required_attributes}",
                results=[],
                reasoning_trace=trace,
                metadata={'error': 'Concept not found'}
            )
        
        # Step 2: Lọc theo từng attribute
        candidates = concept_instances.copy()
        
        for i, attr in enumerate(required_attributes):
            attr_instances = self._get_instances_by_attribute(attr)
            
            before_count = len(candidates)
            candidates = candidates.intersection(attr_instances)
            
            trace.append(
                f"HOP {i+2}: Lọc theo attribute '{attr}' → "
                f"{len(attr_instances)} instances có attribute này → "
                f"Giao với tập hiện tại: {before_count} → {len(candidates)}"
            )
            
            if not candidates:
                trace.append(f"FINAL: Không còn instance nào thỏa mãn sau khi lọc '{attr}'")
                break
        
        # Step 3: Thu thập kết quả chi tiết
        results = []
        for inst_id in list(candidates)[:limit]:
            info = self._get_node_info(inst_id)
            img_id, obj_id = self._parse_instance_node_id(inst_id)
            results.append({
                'node_id': inst_id,
                'image_id': img_id,
                'object_id': obj_id,
                'name': info.get('name', ''),
                'attributes': info.get('attributes', []),
                'matched_attributes': required_attributes
            })
        
        if results:
            trace.append(
                f"FINAL: Tìm thấy {len(candidates)} instances thỏa mãn tất cả điều kiện"
            )
        else:
            trace.append(
                f"FINAL: Không tìm thấy '{name}' nào có đồng thời "
                f"tất cả thuộc tính {required_attributes} trong tập mẫu"
            )
        
        return ReasoningResult(
            query_type="9. Visual-Attribute Constraint",
            question=f"Tìm '{name}' có tất cả các thuộc tính: {required_attributes}",
            results=results,
            reasoning_trace=trace,
            metadata={
                'concept': name,
                'required_attributes': required_attributes,
                'total_concept_instances': len(concept_instances),
                'matching_instances': len(candidates)
            }
        )
    
    def complex_constraint_search(self,
                                   main_concept: str,
                                   relation: str,
                                   related_concept: str,
                                   related_attributes: List[str],
                                   main_attributes: Optional[List[str]] = None,
                                   limit: int = 20) -> ReasoningResult:
        """
        TÌM VẬT THỂ CÓ QUAN HỆ VỚI VẬT THỂ KHÁC VÀ CẢ HAI ĐỀU THỎA MÃN THUỘC TÍNH
        
        Ví dụ: Tìm "man" đang "wearing" "shirt" (có thuộc tính "black")
        
        Logic suy luận:
        1. Tìm instances của main_concept (lọc theo main_attributes nếu có)
        2. Với mỗi instance, tìm neighbors qua relation
        3. Kiểm tra neighbor có phải là related_concept với related_attributes không
        
        Args:
            main_concept: Vật thể chính (vd: "man")
            relation: Quan hệ (vd: "wearing")
            related_concept: Vật thể liên quan (vd: "shirt")
            related_attributes: Thuộc tính của vật thể liên quan (vd: ["black"])
            main_attributes: Thuộc tính của vật thể chính (optional, vd: ["tall"])
            limit: Số lượng kết quả tối đa
        """
        trace = []
        
        # Step 1: Tìm instances của main_concept
        main_instances = self._get_instances_by_concept(main_concept)
        trace.append(f"HOP 1: Tìm '{main_concept}' → {len(main_instances)} instances")
        
        # Lọc theo main_attributes nếu có
        if main_attributes:
            for attr in main_attributes:
                attr_instances = self._get_instances_by_attribute(attr)
                main_instances = main_instances.intersection(attr_instances)
            trace.append(
                f"HOP 2: Lọc '{main_concept}' theo {main_attributes} → {len(main_instances)} instances"
            )
        
        # Step 2: Tìm instances của related_concept với related_attributes
        related_instances = self._get_instances_by_concept(related_concept)
        trace.append(f"HOP 3: Tìm '{related_concept}' → {len(related_instances)} instances")
        
        for attr in related_attributes:
            attr_instances = self._get_instances_by_attribute(attr)
            related_instances = related_instances.intersection(attr_instances)
        trace.append(
            f"HOP 4: Lọc '{related_concept}' theo {related_attributes} → {len(related_instances)} instances"
        )
        
        # Step 3: Tìm các cặp có quan hệ
        found_pairs = []
        
        for main_id in main_instances:
            neighbors = self._get_semantic_neighbors(main_id, relation)
            
            for neighbor_id, rel_name in neighbors:
                if neighbor_id in related_instances:
                    main_info = self._get_node_info(main_id)
                    rel_info = self._get_node_info(neighbor_id)
                    
                    found_pairs.append({
                        'main_node': main_id,
                        'main_image': main_info.get('image_id', ''),
                        'main_attributes': main_info.get('attributes', []),
                        'relation': rel_name,
                        'related_node': neighbor_id,
                        'related_image': rel_info.get('image_id', ''),
                        'related_attributes': rel_info.get('attributes', [])
                    })
        
        trace.append(
            f"HOP 5: Tìm quan hệ '{relation}' giữa hai tập → {len(found_pairs)} cặp thỏa mãn"
        )
        
        results = found_pairs[:limit]
        
        if not results:
            trace.append(
                f"FINAL: Không tìm thấy '{main_concept}' "
                f"{'với ' + str(main_attributes) + ' ' if main_attributes else ''}"
                f"đang '{relation}' '{related_concept}' với {related_attributes} trong tập mẫu"
            )
        else:
            trace.append(f"FINAL: Trả về {len(results)} kết quả")
        
        # Create câu hỏi
        question = f"Tìm '{main_concept}'"
        if main_attributes:
            question += f" (có thuộc tính {main_attributes})"
        question += f" đang '{relation}' '{related_concept}' (có thuộc tính {related_attributes})"
        
        return ReasoningResult(
            query_type="9. Visual-Attribute Constraint",
            question=question,
            results=results,
            reasoning_trace=trace,
            metadata={
                'main_concept': main_concept,
                'main_attributes': main_attributes,
                'relation': relation,
                'related_concept': related_concept,
                'related_attributes': related_attributes,
                'found_count': len(found_pairs)
            }
        )


# ============================================================================
# DEMO FUNCTION
# ============================================================================

def run_part1_demo(scale: str = '1k'):
    """
    Chạy demo cho 5 loại truy vấn đầu tiên.
    Mỗi loại có 2 câu hỏi thực tế.
    
    Args:
        scale: Scale của dataset ('1k', '10k', 'full')
    """
    print("\n" + "=" * 70)
    print(f"🚀 GQA LIGHTRAG REASONING ENGINE - PART 1 DEMO (Scale: {scale})")
    print("=" * 70)
    print("Demo 5 loại truy vấn: Entity, Statistical, Similarity,")
    print("Relational Path, Negative Constraints")
    print("=" * 70)
    
    # Initialize engine
    engine = GQA_Reasoning_Engine(scale=scale)
    
    print("\n\n" + "🔹" * 35)
    print("                    QUERY TYPE 1: ENTITY SEARCH")
    print("🔹" * 35)
    
    # Query 1.1: Tìm người mặc áo trắng
    result = engine.entity_search(
        concept="man",
        attributes=["white"],
        limit=5
    )
    result.print_result()
    
    # Query 1.2: Tìm xe màu đỏ và lớn
    result = engine.entity_search(
        concept="car",
        attributes=["red"],
        limit=5
    )
    result.print_result()
    
    print("\n\n" + "🔹" * 35)
    print("                QUERY TYPE 2: STATISTICAL KNOWLEDGE")
    print("🔹" * 35)
    
    # Query 2.1: Xác suất tìm thấy shirt gần man
    result = engine.statistical_knowledge(
        concept_a="man",
        concept_b="shirt"
    )
    result.print_result()
    
    # Query 2.2: Xác suất tìm thấy window gần building
    result = engine.statistical_knowledge(
        concept_a="building",
        concept_b="window"
    )
    result.print_result()
    
    print("\n\n" + "🔹" * 35)
    print("                  QUERY TYPE 3: SIMILARITY SEARCH")
    print("🔹" * 35)
    
    # Query 3.1: Tìm các cây tương tự (xanh, lớn)
    result = engine.similarity_search(
        concept="tree",
        attributes=["green", "large"],
        min_common_attributes=1,
        limit=5
    )
    result.print_result()
    
    # Query 3.2: Tìm các áo sơ mi tương tự (trắng)
    result = engine.similarity_search(
        concept="shirt",
        attributes=["white"],
        min_common_attributes=1,
        limit=5
    )
    result.print_result()
    
    print("\n\n" + "🔹" * 35)
    print("                   QUERY TYPE 4: RELATIONAL PATH")
    print("🔹" * 35)
    
    # Query 4.1: Tìm đường đi từ person đến shirt qua quan hệ wearing
    result = engine.relational_path(
        source_concept="man",
        target_concept="shirt",
        via_relation="wearing",
        max_hops=2,
        limit=5
    )
    result.print_result()
    
    # Query 4.2: Tìm đường đi từ plate đến table
    result = engine.relational_path(
        source_concept="plate",
        target_concept="table",
        max_hops=2,
        limit=5
    )
    result.print_result()
    
    print("\n\n" + "🔹" * 35)
    print("                QUERY TYPE 5: NEGATIVE CONSTRAINTS")
    print("🔹" * 35)
    
    # Query 5.1: Ảnh có man nhưng không có woman
    result = engine.negative_constraints(
        concept_present="man",
        concept_absent="woman",
        limit=5
    )
    result.print_result()
    
    # Query 5.2: Ảnh có tree nhưng không có car
    result = engine.negative_constraints(
        concept_present="tree",
        concept_absent="car",
        limit=5
    )
    result.print_result()
    
    print("\n\n" + "=" * 70)
    print("🎉 COMPLETED DEMO PART 1!")
    print("=" * 70)
    print("Đã demo 10 câu hỏi cho 5 loại truy vấn:")
    print("  1. Entity Search (2 câu)")
    print("  2. Statistical Knowledge (2 câu)")
    print("  3. Similarity Search (2 câu)")
    print("  4. Relational Path (2 câu)")
    print("  5. Negative Constraints (2 câu)")
    print("=" * 70)
    
    return engine


def run_part2_demo(engine: Optional[GQA_Reasoning_Engine] = None, scale: str = '1k'):
    """
    Chạy demo cho 4 loại truy vấn nâng cao (loại 6-9).
    Sử dụng 8 câu hỏi cố định theo yêu cầu.
    
    Args:
        engine: Engine đã khởi tạo sẵn (optional)
        scale: Scale của dataset ('1k', '10k', 'full')
    """
    print("\n\n" + "=" * 70)
    print(f"🚀 GQA LIGHTRAG REASONING ENGINE - PART 2 DEMO (Scale: {scale})")
    print("=" * 70)
    print("Demo 4 loại truy vấn nâng cao:")
    print("  6. Comparative (So sánh)")
    print("  7. Hierarchical (Phân cấp)")
    print("  8. Anomaly Detection (Phát hiện bất thường)")
    print("  9. Visual-Attribute Constraint (Ràng buộc đa điều kiện)")
    print("=" * 70)
    
    # Initialize engine nếu chưa có
    if engine is None:
        engine = GQA_Reasoning_Engine(scale=scale)
    
    # ========================================================================
    # QUERY TYPE 6: COMPARATIVE (SO SÁNH)
    # ========================================================================
    
    print("\n\n" + "🔹" * 35)
    print("              QUERY TYPE 6: COMPARATIVE (SO SÁNH)")
    print("🔹" * 35)
    
    # Câu 1: So sánh số lượng thực thể "chair" trong bối cimages "kitchen" và "living room"
    print("\n📝 Câu 6.1: So sánh số lượng thực thể 'chair' trong bối cimages 'kitchen' và 'living room'")
    result = engine.compare_contexts(
        context_a="kitchen",
        context_b="living room",
        target_concept="chair"
    )
    result.print_result()
    
    # Câu 2: So sánh xem màu "white" xuất hiện phổ biến hơn ở "wall" hay "bed"
    print("\n📝 Câu 6.2: So sánh xem màu 'white' xuất hiện phổ biến hơn ở 'wall' hay 'bed'")
    result = engine.compare_attribute_distribution(
        concept_a="wall",
        concept_b="bed",
        attribute="white"
    )
    result.print_result()
    
    # ========================================================================
    # QUERY TYPE 7: HIERARCHICAL (PHÂN CẤP)
    # ========================================================================
    
    print("\n\n" + "🔹" * 35)
    print("            QUERY TYPE 7: HIERARCHICAL (PHÂN CẤP)")
    print("🔹" * 35)
    
    # Câu 1: Liệt kê tất cả các thực thể thuộc loại "furniture"
    print("\n📝 Câu 7.1: Liệt kê tất cả thực thể thuộc loại 'furniture' (bao gồm: chair, table, bed, sofa, cabinet)")
    result = engine.get_hierarchical_entities(
        parent_category="furniture",
        limit=10
    )
    result.print_result()
    
    # Câu 2: Tìm các thực thể được phân loại là "electronic devices"
    print("\n📝 Câu 7.2: Tìm các thực thể được phân loại là 'electronic_devices' (bao gồm: television, phone, laptop, camera)")
    result = engine.get_hierarchical_entities(
        parent_category="electronic_devices",
        limit=10
    )
    result.print_result()
    
    # ========================================================================
    # QUERY TYPE 8: ANOMALY DETECTION (PHÁT HIỆN BẤT THƯỜNG)
    # ========================================================================
    
    print("\n\n" + "🔹" * 35)
    print("       QUERY TYPE 8: ANOMALY DETECTION (PHÁT HIỆN BẤT THƯỜNG)")
    print("🔹" * 35)
    
    # Câu 1: Tìm các trường hợp mà một "dog" có quan hệ "on" một chiếc "table"
    print("\n📝 Câu 8.1: Tìm các trường hợp 'dog' (con chó) có quan hệ 'on' (trên) một chiếc 'table' (bàn)")
    result = engine.find_specific_anomaly(
        subject_concept="dog",
        relation="on",
        object_concept="table"
    )
    result.print_result()
    
    # Câu 2: Liệt kê 5 quan hệ (S-P-O) có tần suất xuất hiện thấp nhất
    print("\n📝 Câu 8.2: Liệt kê 5 quan hệ (S-P-O) có tần suất xuất hiện thấp nhất trong hệ thống")
    result = engine.find_anomalies(
        min_frequency=2,
        limit=5
    )
    result.print_result()
    
    # ========================================================================
    # QUERY TYPE 9: VISUAL-ATTRIBUTE CONSTRAINT (RÀNG BUỘC ĐA ĐIỀU KIỆN)
    # ========================================================================
    
    print("\n\n" + "🔹" * 35)
    print("   QUERY TYPE 9: VISUAL-ATTRIBUTE CONSTRAINT (RÀNG BUỘC ĐA ĐIỀU KIỆN)")
    print("🔹" * 35)
    
    # Câu 1: Tìm một cái "cup" vừa có thuộc tính "red" vừa có thuộc tính "plastic"
    print("\n📝 Câu 9.1: Tìm một cái 'cup' (ly) vừa có thuộc tính 'red' (đỏ) vừa có thuộc tính 'plastic' (nhựa)")
    result = engine.multi_constraint_search(
        name="cup",
        required_attributes=["red", "plastic"],
        limit=10
    )
    result.print_result()
    
    # Câu 2: Tìm "man" đang "wearing" "black shirt" và có thuộc tính "tall"
    print("\n📝 Câu 9.2: Tìm 'man' (người đàn ông) đang 'wearing' (mặc) 'shirt' (áo) có thuộc tính 'black' (đen) và người đó có thuộc tính 'tall' (cao)")
    result = engine.complex_constraint_search(
        main_concept="man",
        main_attributes=["tall"],
        relation="wearing",
        related_concept="shirt",
        related_attributes=["black"],
        limit=10
    )
    result.print_result()
    
    # ========================================================================
    # TỔNG KẾT
    # ========================================================================
    
    print("\n\n" + "=" * 70)
    print("🎉 COMPLETED DEMO PART 2!")
    print("=" * 70)
    print("Đã demo 8 câu hỏi cho 4 loại truy vấn nâng cao:")
    print("  6. Comparative - So sánh (2 câu)")
    print("     • So sánh 'chair' trong 'kitchen' vs 'living room'")
    print("     • So sánh 'white' ở 'wall' vs 'bed'")
    print("  7. Hierarchical - Phân cấp (2 câu)")
    print("     • Liệt kê 'furniture' (chair, table, bed, sofa, cabinet)")
    print("     • Liệt kê 'electronic_devices' (television, phone, laptop, camera)")
    print("  8. Anomaly Detection - Phát hiện bất thường (2 câu)")
    print("     • Tìm 'dog' --[on]--> 'table'")
    print("     • Top 5 quan hệ hiếm gặp nhất")
    print("  9. Visual-Attribute Constraint - Ràng buộc đa điều kiện (2 câu)")
    print("     • Tìm 'cup' + [red, plastic]")
    print("     • Tìm 'man' [tall] --[wearing]--> 'shirt' [black]")
    print("=" * 70)
    
    return engine


def run_comprehensive_demo(scale: str = '1k'):
    """
    Chạy demo đầy đủ cho cả 9 loại truy vấn.
    
    Args:
        scale: Scale của dataset ('1k', '10k', 'full')
    """
    print("\n" + "=" * 70)
    print(f"🚀 GQA LIGHTRAG COMPREHENSIVE DEMO - ALL 9 QUERY TYPES (Scale: {scale})")
    print("=" * 70)
    
    # Initialize engine một times duy nhất
    engine = GQA_Reasoning_Engine(scale=scale)
    
    # Chạy Part 1 (Query Types 1-5)
    run_part1_demo_with_engine(engine)
    
    # Chạy Part 2 (Query Types 6-9)
    run_part2_demo(engine, scale=scale)
    
    print("\n\n" + "=" * 70)
    print("🎉 COMPLETED DEMO ĐẦY ĐỦ CẢ 9 LOẠI TRUY VẤN!")
    print("=" * 70)
    print("Tổng cộng: 18 câu hỏi demo")
    print("  Part 1: 10 câu (Query Types 1-5)")
    print("  Part 2: 8 câu (Query Types 6-9)")
    print("=" * 70)


def run_part1_demo_with_engine(engine: GQA_Reasoning_Engine):
    """
    Chạy demo Part 1 với engine đã có sẵn.
    """
    print("\n\n" + "🔹" * 35)
    print("                    QUERY TYPE 1: ENTITY SEARCH")
    print("🔹" * 35)
    
    result = engine.entity_search(concept="man", attributes=["white"], limit=5)
    result.print_result()
    
    result = engine.entity_search(concept="car", attributes=["red"], limit=5)
    result.print_result()
    
    print("\n\n" + "🔹" * 35)
    print("                QUERY TYPE 2: STATISTICAL KNOWLEDGE")
    print("🔹" * 35)
    
    result = engine.statistical_knowledge(concept_a="man", concept_b="shirt")
    result.print_result()
    
    result = engine.statistical_knowledge(concept_a="building", concept_b="window")
    result.print_result()
    
    print("\n\n" + "🔹" * 35)
    print("                  QUERY TYPE 3: SIMILARITY SEARCH")
    print("🔹" * 35)
    
    result = engine.similarity_search(concept="tree", attributes=["green", "large"], min_common_attributes=1, limit=5)
    result.print_result()
    
    result = engine.similarity_search(concept="shirt", attributes=["white"], min_common_attributes=1, limit=5)
    result.print_result()
    
    print("\n\n" + "🔹" * 35)
    print("                   QUERY TYPE 4: RELATIONAL PATH")
    print("🔹" * 35)
    
    result = engine.relational_path(source_concept="man", target_concept="shirt", via_relation="wearing", max_hops=2, limit=5)
    result.print_result()
    
    result = engine.relational_path(source_concept="plate", target_concept="table", max_hops=2, limit=5)
    result.print_result()
    
    print("\n\n" + "🔹" * 35)
    print("                QUERY TYPE 5: NEGATIVE CONSTRAINTS")
    print("🔹" * 35)
    
    result = engine.negative_constraints(concept_present="man", concept_absent="woman", limit=5)
    result.print_result()
    
    result = engine.negative_constraints(concept_present="tree", concept_absent="car", limit=5)
    result.print_result()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="GQA LightRAG Reasoning Engine Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  python gqa_reasoning_engine.py --scale 1k --demo part1   # Demo Part 1 với 1k images
  python gqa_reasoning_engine.py --scale 10k --demo part2  # Demo Part 2 với 10k images
  python gqa_reasoning_engine.py --scale full --demo all   # Demo tất cả với full dataset
        """
    )
    
    parser.add_argument(
        '--scale',
        type=str,
        choices=['1k', '10k', 'full'],
        default='1k',
        help='Scale của dataset: 1k (1,000), 10k (10,000), full (74,942 images)'
    )
    
    parser.add_argument(
        '--demo',
        type=str,
        choices=['part1', 'part2', 'all'],
        default='part2',
        help='Phần demo: part1 (Query 1-5), part2 (Query 6-9), all (tất cả)'
    )
    
    args = parser.parse_args()
    
    print(f"\n📌 Sử dụng scale: {args.scale}")
    print(f"📌 Demo: {args.demo}")
    
    if args.demo == 'part1':
        run_part1_demo(scale=args.scale)
    elif args.demo == 'part2':
        run_part2_demo(scale=args.scale)
    elif args.demo == 'all':
        run_comprehensive_demo(scale=args.scale)

