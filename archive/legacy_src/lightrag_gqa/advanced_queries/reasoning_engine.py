"""
Advanced Graph Reasoning Engine
================================
Extension of GQA LightRAG Reasoning Engine with advanced multi-hop reasoning,
pattern matching, and comparative analysis capabilities.

These query types require sophisticated graph traversal to highlight
the strengths of knowledge graph-based retrieval.

Advanced Query Types:
1. Chain Reasoning - Multi-hop traversal with intermediate constraints
2. Pattern Matching - Subgraph isomorphism for complex patterns
3. Scene Comparison - Structural comparison between images
4. Counterfactual Reasoning - Hypothetical "what if" queries
5. Centrality Queries - Find influential nodes using graph metrics

Author: GQA LightRAG Project
Date: 2026
"""

import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field
import heapq

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import networkx as nx
except ImportError:
    raise ImportError("Please install networkx: pip install networkx")

from lightrag_gqa.basic_queries.reasoning_engine import GQA_Reasoning_Engine, ReasoningResult


# ============================================================================
# ENHANCED REASONING RESULT WITH STEP TRACKING
# ============================================================================

@dataclass
class ReasoningStep:
    """Individual reasoning step with detailed information."""
    step_number: int
    operation: str  # RETRIEVE, TRAVERSE, FILTER, VERIFY, AGGREGATE, COMPUTE
    description: str
    input_size: int
    output_size: int
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self):
        return (f"[{self.operation}] {self.description} "
                f"({self.input_size} → {self.output_size})")


@dataclass
class AdvancedReasoningResult:
    """
    Enhanced result structure with explicit reasoning steps.
    """
    query_type: str
    question: str
    results: Any
    reasoning_steps: List[ReasoningStep] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def print_result(self, verbose: bool = True):
        """Print result with detailed reasoning visualization."""
        print("\n" + "=" * 80)
        print(f"🧠 ADVANCED QUERY: {self.query_type}")
        print("=" * 80)
        print(f"\n❓ QUESTION: {self.question}")
        
        if verbose and self.reasoning_steps:
            print(f"\n🔍 REASONING STEPS ({len(self.reasoning_steps)} steps):")
            print("-" * 60)
            for step in self.reasoning_steps:
                icon = self._get_operation_icon(step.operation)
                print(f"   Step {step.step_number}: {icon} [{step.operation}]")
                print(f"            {step.description}")
                print(f"            📊 {step.input_size:,} → {step.output_size:,} items")
                if step.details:
                    for key, value in step.details.items():
                        print(f"            • {key}: {value}")
                print()
        
        print(f"📊 RESULTS:")
        if isinstance(self.results, list):
            if len(self.results) == 0:
                print("   (No results found)")
            elif len(self.results) <= 5:
                for i, item in enumerate(self.results, 1):
                    self._print_result_item(i, item)
            else:
                for i, item in enumerate(self.results[:5], 1):
                    self._print_result_item(i, item)
                print(f"   ... and {len(self.results) - 5} more results")
        elif isinstance(self.results, dict):
            for key, value in list(self.results.items())[:10]:
                print(f"   • {key}: {value}")
        else:
            print(f"   {self.results}")
        
        if self.metadata:
            print(f"\n📈 METADATA:")
            for key, value in self.metadata.items():
                print(f"   • {key}: {value}")
        
        print("=" * 80)
    
    def _get_operation_icon(self, operation: str) -> str:
        icons = {
            'RETRIEVE': '📥',
            'TRAVERSE': '🔗',
            'FILTER': '🔍',
            'VERIFY': '✅',
            'AGGREGATE': '📊',
            'COMPUTE': '🧮',
            'COMPARE': '⚖️',
            'MATCH': '🎯',
        }
        return icons.get(operation, '•')
    
    def _print_result_item(self, index: int, item: Any):
        if isinstance(item, dict):
            print(f"   {index}. ", end="")
            key_items = list(item.items())[:4]
            print(", ".join(f"{k}={v}" for k, v in key_items))
        else:
            print(f"   {index}. {item}")
    
    def get_reasoning_summary(self) -> str:
        """Get a text summary of the reasoning chain."""
        lines = [f"Query: {self.question}", "Reasoning Chain:"]
        for step in self.reasoning_steps:
            lines.append(f"  {step.step_number}. {step}")
        lines.append(f"Final Result: {len(self.results) if isinstance(self.results, list) else 1} items")
        return "\n".join(lines)


# ============================================================================
# ADVANCED REASONING ENGINE
# ============================================================================

class AdvancedReasoningEngine(GQA_Reasoning_Engine):
    """
    Extended reasoning engine with advanced graph reasoning capabilities.
    
    Inherits from GQA_Reasoning_Engine and adds:
    - Multi-hop chain reasoning
    - Subgraph pattern matching
    - Scene comparison
    - Counterfactual reasoning
    - Centrality-based queries
    """
    
    def __init__(self, graph_path: Optional[str] = None, scale: str = '10k'):
        """Initialize advanced reasoning engine."""
        super().__init__(graph_path=graph_path, scale=scale)
        
        # Additional caches for advanced queries
        self._relation_index: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
        self._image_subgraphs: Dict[str, Set[str]] = {}
        
        # Build additional indices
        self._build_advanced_cache()
    
    def _build_advanced_cache(self):
        """Build additional caches for advanced reasoning."""
        print("   🔧 Building advanced reasoning cache...")
        
        # Index edges by relation type
        for u, v, data in self.graph.edges(data=True):
            if data.get('edge_type') == 'semantic_relation':
                relation = self._normalize(data.get('relation', ''))
                self._relation_index[relation].add((u, v))
        
        print(f"      • Relation types indexed: {len(self._relation_index)}")
        print(f"   ✅ Advanced cache ready!")
    
    # ========================================================================
    # ADVANCED QUERY 1: MULTI-HOP CHAIN REASONING
    # ========================================================================
    
    def chain_reasoning(self,
                        start_concept: str,
                        chain: List[Dict[str, Any]],
                        limit: int = 20) -> AdvancedReasoningResult:
        """
        MULTI-HOP CHAIN REASONING
        
        Find instances that satisfy a chain of relationship constraints.
        Each step in the chain specifies:
        - relation: The relationship to traverse
        - concept: The target concept (optional)
        - attribute: Required attribute (optional)
        
        Example chain:
        [
            {"relation": "wearing", "concept": "shirt", "attribute": "red"},
            {"relation": "to the left of", "concept": "car"}
        ]
        
        This finds: person → wearing → red shirt → to the left of → car
        
        Args:
            start_concept: Starting concept (e.g., "person")
            chain: List of hop specifications
            limit: Maximum results to return
            
        Returns:
            AdvancedReasoningResult with found chains and reasoning steps
        """
        steps = []
        step_num = 1
        
        # Step 1: Get starting instances
        start_instances = self._get_instances_by_concept(start_concept)
        steps.append(ReasoningStep(
            step_number=step_num,
            operation="RETRIEVE",
            description=f"Get all instances of concept '{start_concept}'",
            input_size=0,
            output_size=len(start_instances),
            details={"concept": start_concept}
        ))
        step_num += 1
        
        if not start_instances:
            return AdvancedReasoningResult(
                query_type="1. Chain Reasoning",
                question=self._format_chain_question(start_concept, chain),
                results=[],
                reasoning_steps=steps,
                metadata={"error": f"No instances found for concept '{start_concept}'"}
            )
        
        # Track chains as we traverse
        # Each chain item: (current_node, path_so_far)
        active_chains = [(inst, [inst]) for inst in list(start_instances)[:500]]
        
        # Process each hop in the chain
        for hop_idx, hop in enumerate(chain):
            relation = hop.get('relation', '')
            target_concept = hop.get('concept')
            target_attribute = hop.get('attribute')
            
            # Step: Traverse relation
            new_chains = []
            traversed_count = 0
            
            for current_node, path in active_chains:
                # Get neighbors via this relation
                neighbors = self._get_semantic_neighbors(current_node, relation)
                traversed_count += len(neighbors)
                
                for neighbor_id, rel_name in neighbors:
                    if neighbor_id not in path:  # Avoid cycles
                        new_path = path + [f"--[{rel_name}]-->", neighbor_id]
                        new_chains.append((neighbor_id, new_path))
            
            steps.append(ReasoningStep(
                step_number=step_num,
                operation="TRAVERSE",
                description=f"Follow '{relation}' edges from current nodes",
                input_size=len(active_chains),
                output_size=len(new_chains),
                details={"relation": relation, "edges_checked": traversed_count}
            ))
            step_num += 1
            
            # Step: Filter by concept if specified
            if target_concept:
                target_instances = self._get_instances_by_concept(target_concept)
                filtered_chains = [(node, path) for node, path in new_chains 
                                   if node in target_instances]
                
                steps.append(ReasoningStep(
                    step_number=step_num,
                    operation="FILTER",
                    description=f"Keep only nodes of concept '{target_concept}'",
                    input_size=len(new_chains),
                    output_size=len(filtered_chains),
                    details={"concept": target_concept, "concept_instances": len(target_instances)}
                ))
                step_num += 1
                new_chains = filtered_chains
            
            # Step: Filter by attribute if specified
            if target_attribute:
                attr_instances = self._get_instances_by_attribute(target_attribute)
                filtered_chains = [(node, path) for node, path in new_chains 
                                   if node in attr_instances]
                
                steps.append(ReasoningStep(
                    step_number=step_num,
                    operation="FILTER",
                    description=f"Keep only nodes with attribute '{target_attribute}'",
                    input_size=len(new_chains),
                    output_size=len(filtered_chains),
                    details={"attribute": target_attribute, "attr_instances": len(attr_instances)}
                ))
                step_num += 1
                new_chains = filtered_chains
            
            active_chains = new_chains
            
            if not active_chains:
                break
        
        # Step: Aggregate results
        results = []
        seen_images = set()
        
        for end_node, path in active_chains[:limit * 2]:
            # Get image ID from the starting node
            start_node = path[0]
            start_info = self._get_node_info(start_node)
            end_info = self._get_node_info(end_node)
            image_id = start_info.get('image_id', '')
            
            if image_id and image_id not in seen_images:
                seen_images.add(image_id)
                results.append({
                    'image_id': image_id,
                    'chain_path': path,
                    'chain_length': len([p for p in path if not str(p).startswith('--[')]),
                    'start_node': start_node,
                    'start_name': start_info.get('name', ''),
                    'end_node': end_node,
                    'end_name': end_info.get('name', ''),
                })
                
                if len(results) >= limit:
                    break
        
        steps.append(ReasoningStep(
            step_number=step_num,
            operation="AGGREGATE",
            description="Group valid chains by image and deduplicate",
            input_size=len(active_chains),
            output_size=len(results),
            details={"unique_images": len(seen_images)}
        ))
        
        return AdvancedReasoningResult(
            query_type="1. Chain Reasoning (Multi-Hop)",
            question=self._format_chain_question(start_concept, chain),
            results=results,
            reasoning_steps=steps,
            metadata={
                'start_concept': start_concept,
                'chain_hops': len(chain),
                'total_chains_found': len(active_chains),
                'unique_images': len(seen_images)
            }
        )
    
    def _format_chain_question(self, start_concept: str, chain: List[Dict]) -> str:
        """Format chain specification as natural language question."""
        parts = [f"Find images with '{start_concept}'"]
        for hop in chain:
            rel = hop.get('relation', '?')
            concept = hop.get('concept', '')
            attr = hop.get('attribute', '')
            
            hop_desc = f"{rel}"
            if attr and concept:
                hop_desc += f" {attr} {concept}"
            elif concept:
                hop_desc += f" {concept}"
            elif attr:
                hop_desc += f" something {attr}"
            
            parts.append(hop_desc)
        
        return " → ".join(parts)
    
    # ========================================================================
    # ADVANCED QUERY 2: SUBGRAPH PATTERN MATCHING
    # ========================================================================
    
    def pattern_matching(self,
                         pattern_nodes: List[str],
                         pattern_edges: List[Tuple[str, str, str]],
                         limit: int = 20) -> AdvancedReasoningResult:
        """
        SUBGRAPH PATTERN MATCHING
        
        Find images containing a specific relationship pattern.
        
        Args:
            pattern_nodes: List of concept names in the pattern
            pattern_edges: List of (source_concept, relation, target_concept) tuples
            limit: Maximum results
            
        Example:
            pattern_nodes = ["person", "dog", "ball"]
            pattern_edges = [
                ("person", "holding", "dog"),
                ("dog", "looking at", "ball")
            ]
            
        Returns:
            AdvancedReasoningResult with images matching the pattern
        """
        steps = []
        step_num = 1
        
        # Step 1: Get instances for each concept in pattern
        concept_instances = {}
        for concept in pattern_nodes:
            instances = self._get_instances_by_concept(concept)
            concept_instances[concept] = instances
            steps.append(ReasoningStep(
                step_number=step_num,
                operation="RETRIEVE",
                description=f"Get instances of concept '{concept}'",
                input_size=0,
                output_size=len(instances),
                details={"concept": concept}
            ))
            step_num += 1
        
        # Step 2: Group instances by image
        image_candidates = defaultdict(lambda: defaultdict(set))
        
        for concept, instances in concept_instances.items():
            for inst_id in instances:
                image_id, _ = self._parse_instance_node_id(inst_id)
                if image_id:
                    image_candidates[image_id][concept].add(inst_id)
        
        # Filter to images that have at least one instance of each concept
        valid_images = {
            img_id for img_id, concepts in image_candidates.items()
            if all(concept in concepts for concept in pattern_nodes)
        }
        
        steps.append(ReasoningStep(
            step_number=step_num,
            operation="FILTER",
            description="Keep images with all required concepts",
            input_size=len(image_candidates),
            output_size=len(valid_images),
            details={"required_concepts": pattern_nodes}
        ))
        step_num += 1
        
        # Step 3: Check edge constraints for each candidate image
        matching_images = []
        
        for image_id in valid_images:
            concepts_in_image = image_candidates[image_id]
            
            # Try to find a valid assignment of instances to pattern nodes
            match_found = self._check_pattern_match(
                concepts_in_image, pattern_edges
            )
            
            if match_found:
                matching_images.append({
                    'image_id': image_id,
                    'matched_pattern': match_found,
                    'concepts_found': {c: len(insts) for c, insts in concepts_in_image.items()}
                })
                
                if len(matching_images) >= limit:
                    break
        
        steps.append(ReasoningStep(
            step_number=step_num,
            operation="MATCH",
            description="Verify pattern edges exist in each image",
            input_size=len(valid_images),
            output_size=len(matching_images),
            details={"pattern_edges": len(pattern_edges)}
        ))
        
        # Format question
        pattern_desc = " + ".join([f"({s} {r} {t})" for s, r, t in pattern_edges])
        question = f"Find images matching pattern: {pattern_desc}"
        
        return AdvancedReasoningResult(
            query_type="2. Subgraph Pattern Matching",
            question=question,
            results=matching_images,
            reasoning_steps=steps,
            metadata={
                'pattern_nodes': pattern_nodes,
                'pattern_edges': pattern_edges,
                'images_checked': len(valid_images),
                'matches_found': len(matching_images)
            }
        )
    
    def _check_pattern_match(self, 
                             concepts_in_image: Dict[str, Set[str]],
                             pattern_edges: List[Tuple[str, str, str]]) -> Optional[Dict]:
        """Check if pattern edges exist in image."""
        matched_edges = []
        
        for source_concept, relation, target_concept in pattern_edges:
            source_instances = concepts_in_image.get(source_concept, set())
            target_instances = concepts_in_image.get(target_concept, set())
            
            # Check if any edge exists between source and target instances
            edge_found = False
            for src_inst in source_instances:
                neighbors = self._get_semantic_neighbors(src_inst, relation)
                for neighbor_id, rel_name in neighbors:
                    if neighbor_id in target_instances:
                        matched_edges.append({
                            'source': src_inst,
                            'relation': rel_name,
                            'target': neighbor_id
                        })
                        edge_found = True
                        break
                if edge_found:
                    break
            
            if not edge_found:
                return None
        
        return {'matched_edges': matched_edges}
    
    # ========================================================================
    # ADVANCED QUERY 3: SCENE COMPARISON
    # ========================================================================
    
    def scene_comparison(self,
                         image_id_1: str,
                         image_id_2: str) -> AdvancedReasoningResult:
        """
        SCENE COMPARISON REASONING
        
        Compare two images based on their graph structure:
        - Common objects/concepts
        - Unique objects to each image
        - Relationship similarity
        - Attribute overlap
        
        Args:
            image_id_1: First image ID
            image_id_2: Second image ID
            
        Returns:
            AdvancedReasoningResult with detailed comparison
        """
        steps = []
        step_num = 1
        
        # Step 1: Get objects in image 1
        objects_1 = self._image_to_objects.get(image_id_1, set())
        concepts_1 = set()
        attributes_1 = set()
        
        for obj_id in objects_1:
            info = self._get_node_info(obj_id)
            concepts_1.add(info.get('name', ''))
            attributes_1.update(info.get('attributes', []))
        
        steps.append(ReasoningStep(
            step_number=step_num,
            operation="RETRIEVE",
            description=f"Extract scene graph for image '{image_id_1}'",
            input_size=0,
            output_size=len(objects_1),
            details={
                "objects": len(objects_1),
                "concepts": len(concepts_1),
                "attributes": len(attributes_1)
            }
        ))
        step_num += 1
        
        # Step 2: Get objects in image 2
        objects_2 = self._image_to_objects.get(image_id_2, set())
        concepts_2 = set()
        attributes_2 = set()
        
        for obj_id in objects_2:
            info = self._get_node_info(obj_id)
            concepts_2.add(info.get('name', ''))
            attributes_2.update(info.get('attributes', []))
        
        steps.append(ReasoningStep(
            step_number=step_num,
            operation="RETRIEVE",
            description=f"Extract scene graph for image '{image_id_2}'",
            input_size=0,
            output_size=len(objects_2),
            details={
                "objects": len(objects_2),
                "concepts": len(concepts_2),
                "attributes": len(attributes_2)
            }
        ))
        step_num += 1
        
        # Step 3: Compute similarities
        common_concepts = concepts_1 & concepts_2
        unique_to_1 = concepts_1 - concepts_2
        unique_to_2 = concepts_2 - concepts_1
        
        common_attributes = attributes_1 & attributes_2
        
        # Jaccard similarities
        concept_jaccard = (len(common_concepts) / len(concepts_1 | concepts_2) 
                          if concepts_1 | concepts_2 else 0)
        attr_jaccard = (len(common_attributes) / len(attributes_1 | attributes_2)
                       if attributes_1 | attributes_2 else 0)
        
        steps.append(ReasoningStep(
            step_number=step_num,
            operation="COMPUTE",
            description="Calculate structural similarity metrics",
            input_size=len(concepts_1) + len(concepts_2),
            output_size=2,  # Two similarity scores
            details={
                "concept_jaccard": round(concept_jaccard, 4),
                "attribute_jaccard": round(attr_jaccard, 4)
            }
        ))
        step_num += 1
        
        # Step 4: Count relationships in each image
        relations_1 = self._count_relations_in_image(image_id_1)
        relations_2 = self._count_relations_in_image(image_id_2)
        
        common_relations = set(relations_1.keys()) & set(relations_2.keys())
        relation_jaccard = (len(common_relations) / 
                           len(set(relations_1.keys()) | set(relations_2.keys()))
                           if relations_1 or relations_2 else 0)
        
        steps.append(ReasoningStep(
            step_number=step_num,
            operation="COMPARE",
            description="Compare relationship types between images",
            input_size=len(relations_1) + len(relations_2),
            output_size=len(common_relations),
            details={
                "relation_types_1": len(relations_1),
                "relation_types_2": len(relations_2),
                "common_relations": len(common_relations)
            }
        ))
        
        # Overall similarity score
        overall_similarity = (concept_jaccard + attr_jaccard + relation_jaccard) / 3
        
        results = {
            'image_1': {
                'id': image_id_1,
                'objects': len(objects_1),
                'concepts': list(concepts_1),
                'attributes': list(attributes_1)[:10],
                'relations': dict(list(relations_1.items())[:5])
            },
            'image_2': {
                'id': image_id_2,
                'objects': len(objects_2),
                'concepts': list(concepts_2),
                'attributes': list(attributes_2)[:10],
                'relations': dict(list(relations_2.items())[:5])
            },
            'comparison': {
                'common_concepts': list(common_concepts),
                'unique_to_image_1': list(unique_to_1),
                'unique_to_image_2': list(unique_to_2),
                'concept_similarity': round(concept_jaccard, 4),
                'attribute_similarity': round(attr_jaccard, 4),
                'relation_similarity': round(relation_jaccard, 4),
                'overall_similarity': round(overall_similarity, 4)
            }
        }
        
        return AdvancedReasoningResult(
            query_type="3. Scene Comparison",
            question=f"Compare scenes: image '{image_id_1}' vs image '{image_id_2}'",
            results=results,
            reasoning_steps=steps,
            metadata={
                'image_1': image_id_1,
                'image_2': image_id_2,
                'overall_similarity': round(overall_similarity, 4)
            }
        )
    
    def _count_relations_in_image(self, image_id: str) -> Dict[str, int]:
        """Count relation types in an image."""
        objects = self._image_to_objects.get(image_id, set())
        relation_counts = defaultdict(int)
        
        for obj_id in objects:
            for succ in self.graph.successors(obj_id):
                if succ in objects:  # Only count within-image relations
                    edge_data = self.graph.edges[obj_id, succ]
                    if edge_data.get('edge_type') == 'semantic_relation':
                        rel = edge_data.get('relation', 'unknown')
                        relation_counts[rel] += 1
        
        return dict(relation_counts)
    
    # ========================================================================
    # ADVANCED QUERY 4: COUNTERFACTUAL REASONING
    # ========================================================================
    
    def counterfactual_reasoning(self,
                                 image_id: str,
                                 remove_concept: Optional[str] = None,
                                 add_concept: Optional[str] = None,
                                 query_type: str = "impact") -> AdvancedReasoningResult:
        """
        COUNTERFACTUAL REASONING
        
        Explore hypothetical scenarios:
        - What if we removed all instances of a concept?
        - What if we added a new concept to the scene?
        
        Args:
            image_id: Target image
            remove_concept: Concept to hypothetically remove
            add_concept: Concept to hypothetically add
            query_type: "impact" (analyze effects) or "similar" (find similar scenes)
            
        Returns:
            AdvancedReasoningResult with counterfactual analysis
        """
        steps = []
        step_num = 1
        
        # Step 1: Get current scene
        objects = self._image_to_objects.get(image_id, set())
        current_concepts = set()
        current_relations = []
        
        for obj_id in objects:
            info = self._get_node_info(obj_id)
            current_concepts.add(info.get('name', ''))
            
            # Get relations involving this object
            for succ in self.graph.successors(obj_id):
                if succ in objects:
                    edge_data = self.graph.edges[obj_id, succ]
                    if edge_data.get('edge_type') == 'semantic_relation':
                        succ_info = self._get_node_info(succ)
                        current_relations.append({
                            'source': info.get('name', ''),
                            'relation': edge_data.get('relation', ''),
                            'target': succ_info.get('name', '')
                        })
        
        steps.append(ReasoningStep(
            step_number=step_num,
            operation="RETRIEVE",
            description=f"Load current scene for image '{image_id}'",
            input_size=0,
            output_size=len(objects),
            details={
                "concepts": list(current_concepts),
                "relations": len(current_relations)
            }
        ))
        step_num += 1
        
        results = {
            'original_scene': {
                'image_id': image_id,
                'concepts': list(current_concepts),
                'relation_count': len(current_relations)
            }
        }
        
        # Step 2: Apply counterfactual modification
        if remove_concept:
            # Find objects of this concept
            removed_objects = set()
            for obj_id in objects:
                info = self._get_node_info(obj_id)
                if info.get('name', '').lower() == remove_concept.lower():
                    removed_objects.add(obj_id)
            
            # Count affected relations
            affected_relations = [
                r for r in current_relations
                if r['source'].lower() == remove_concept.lower() or
                   r['target'].lower() == remove_concept.lower()
            ]
            
            steps.append(ReasoningStep(
                step_number=step_num,
                operation="COMPUTE",
                description=f"Simulate removal of concept '{remove_concept}'",
                input_size=len(objects),
                output_size=len(objects) - len(removed_objects),
                details={
                    "objects_removed": len(removed_objects),
                    "relations_affected": len(affected_relations)
                }
            ))
            step_num += 1
            
            # Find images similar to the counterfactual scene
            counterfactual_concepts = current_concepts - {remove_concept}
            similar_images = self._find_images_with_concepts(
                must_have=list(counterfactual_concepts),
                must_not_have=[remove_concept],
                limit=10
            )
            
            steps.append(ReasoningStep(
                step_number=step_num,
                operation="RETRIEVE",
                description=f"Find images matching counterfactual scene",
                input_size=len(self._image_to_objects),
                output_size=len(similar_images),
                details={"matching_criteria": f"has {counterfactual_concepts}, lacks {remove_concept}"}
            ))
            
            results['counterfactual'] = {
                'modification': f"Remove all '{remove_concept}' objects",
                'objects_affected': len(removed_objects),
                'relations_broken': len(affected_relations),
                'remaining_concepts': list(counterfactual_concepts),
                'similar_scenes': similar_images
            }
        
        if add_concept:
            # Find what relations this concept typically has
            concept_instances = self._get_instances_by_concept(add_concept)
            typical_relations = defaultdict(int)
            
            for inst_id in list(concept_instances)[:100]:
                neighbors = self._get_semantic_neighbors(inst_id)
                for _, rel_name in neighbors:
                    typical_relations[rel_name] += 1
            
            # Find concepts it typically relates to
            related_concepts = defaultdict(int)
            for inst_id in list(concept_instances)[:100]:
                neighbors = self._get_semantic_neighbors(inst_id)
                for neighbor_id, _ in neighbors:
                    neighbor_info = self._get_node_info(neighbor_id)
                    related_concepts[neighbor_info.get('name', '')] += 1
            
            steps.append(ReasoningStep(
                step_number=step_num,
                operation="COMPUTE",
                description=f"Analyze typical relationships of '{add_concept}'",
                input_size=len(concept_instances),
                output_size=len(typical_relations),
                details={
                    "instances_analyzed": min(100, len(concept_instances)),
                    "relation_types_found": len(typical_relations)
                }
            ))
            step_num += 1
            
            # Predict new relations if added
            potential_relations = []
            for concept in current_concepts:
                if concept in related_concepts:
                    potential_relations.append({
                        'with_concept': concept,
                        'frequency': related_concepts[concept]
                    })
            
            potential_relations.sort(key=lambda x: -x['frequency'])
            
            steps.append(ReasoningStep(
                step_number=step_num,
                operation="COMPUTE",
                description=f"Predict relations if '{add_concept}' added to scene",
                input_size=len(current_concepts),
                output_size=len(potential_relations),
                details={"existing_concepts_analyzed": len(current_concepts)}
            ))
            
            results['counterfactual'] = results.get('counterfactual', {})
            results['counterfactual']['addition'] = {
                'concept_to_add': add_concept,
                'typical_relations': dict(list(typical_relations.items())[:5]),
                'predicted_interactions': potential_relations[:5]
            }
        
        question_parts = [f"Counterfactual analysis for image '{image_id}'"]
        if remove_concept:
            question_parts.append(f"if '{remove_concept}' removed")
        if add_concept:
            question_parts.append(f"if '{add_concept}' added")
        
        return AdvancedReasoningResult(
            query_type="4. Counterfactual Reasoning",
            question=" - ".join(question_parts),
            results=results,
            reasoning_steps=steps,
            metadata={
                'image_id': image_id,
                'remove_concept': remove_concept,
                'add_concept': add_concept
            }
        )
    
    def _find_images_with_concepts(self,
                                   must_have: List[str],
                                   must_not_have: List[str],
                                   limit: int = 10) -> List[str]:
        """Find images with specific concept constraints."""
        matching_images = []
        
        for image_id, objects in self._image_to_objects.items():
            concepts_in_image = set()
            for obj_id in objects:
                info = self._get_node_info(obj_id)
                concepts_in_image.add(info.get('name', '').lower())
            
            # Check must_have
            has_all = all(c.lower() in concepts_in_image for c in must_have)
            # Check must_not_have
            has_none = all(c.lower() not in concepts_in_image for c in must_not_have)
            
            if has_all and has_none:
                matching_images.append(image_id)
                if len(matching_images) >= limit:
                    break
        
        return matching_images
    
    # ========================================================================
    # ADVANCED QUERY 5: CENTRALITY-BASED QUERIES
    # ========================================================================
    
    def centrality_query(self,
                         centrality_type: str = "degree",
                         node_filter: Optional[str] = None,
                         top_k: int = 20) -> AdvancedReasoningResult:
        """
        CENTRALITY-BASED RETRIEVAL
        
        Find important/influential nodes using graph centrality measures.
        
        Args:
            centrality_type: "degree", "betweenness", "pagerank", or "hub_images"
            node_filter: Filter to specific node types ("concept", "attribute", "instance")
            top_k: Number of top results
            
        Returns:
            AdvancedReasoningResult with centrality rankings
        """
        steps = []
        step_num = 1
        
        # Step 1: Select nodes to analyze
        if node_filter == "concept":
            nodes = list(self._concept_nodes)
            node_type_desc = "concept nodes"
        elif node_filter == "attribute":
            nodes = list(self._attribute_nodes)
            node_type_desc = "attribute nodes"
        elif node_filter == "instance":
            nodes = list(self._instance_nodes)[:10000]  # Limit for performance
            node_type_desc = "instance nodes (sampled)"
        else:
            # Use concepts and attributes for global analysis
            nodes = list(self._concept_nodes) + list(self._attribute_nodes)
            node_type_desc = "concept + attribute nodes"
        
        steps.append(ReasoningStep(
            step_number=step_num,
            operation="RETRIEVE",
            description=f"Select {node_type_desc} for centrality analysis",
            input_size=self.graph.number_of_nodes(),
            output_size=len(nodes),
            details={"node_filter": node_filter or "global"}
        ))
        step_num += 1
        
        # Step 2: Compute centrality
        centrality_scores = {}
        
        if centrality_type == "degree":
            # Count connections for each node
            for node in nodes:
                in_deg = self.graph.in_degree(node)
                out_deg = self.graph.out_degree(node)
                centrality_scores[node] = in_deg + out_deg
            
            steps.append(ReasoningStep(
                step_number=step_num,
                operation="COMPUTE",
                description="Calculate degree centrality (in + out edges)",
                input_size=len(nodes),
                output_size=len(centrality_scores),
                details={"metric": "degree"}
            ))
            
        elif centrality_type == "pagerank":
            # Use NetworkX PageRank on subgraph
            subgraph = self.graph.subgraph(nodes)
            try:
                pr_scores = nx.pagerank(subgraph, max_iter=50)
                centrality_scores = {n: pr_scores.get(n, 0) for n in nodes}
            except:
                # Fallback to degree if PageRank fails
                for node in nodes:
                    centrality_scores[node] = self.graph.degree(node)
            
            steps.append(ReasoningStep(
                step_number=step_num,
                operation="COMPUTE",
                description="Calculate PageRank centrality",
                input_size=len(nodes),
                output_size=len(centrality_scores),
                details={"metric": "pagerank", "iterations": 50}
            ))
            
        elif centrality_type == "hub_images":
            # Special: Find images that connect many different concepts
            image_scores = {}
            
            for image_id, objects in self._image_to_objects.items():
                concepts_in_image = set()
                relations_in_image = 0
                
                for obj_id in objects:
                    info = self._get_node_info(obj_id)
                    concepts_in_image.add(info.get('name', ''))
                    relations_in_image += len(self._get_semantic_neighbors(obj_id))
                
                # Score = unique concepts * log(relations)
                import math
                score = len(concepts_in_image) * math.log1p(relations_in_image)
                image_scores[image_id] = score
            
            centrality_scores = image_scores
            
            steps.append(ReasoningStep(
                step_number=step_num,
                operation="COMPUTE",
                description="Calculate hub score for images (concepts × log(relations))",
                input_size=len(self._image_to_objects),
                output_size=len(centrality_scores),
                details={"metric": "hub_score"}
            ))
        
        else:
            # Default to degree
            for node in nodes:
                centrality_scores[node] = self.graph.degree(node)
            
            steps.append(ReasoningStep(
                step_number=step_num,
                operation="COMPUTE",
                description=f"Calculate {centrality_type} centrality (defaulted to degree)",
                input_size=len(nodes),
                output_size=len(centrality_scores),
                details={"metric": centrality_type}
            ))
        
        step_num += 1
        
        # Step 3: Rank and return top-k
        ranked = sorted(centrality_scores.items(), key=lambda x: -x[1])[:top_k]
        
        steps.append(ReasoningStep(
            step_number=step_num,
            operation="AGGREGATE",
            description=f"Rank nodes and select top {top_k}",
            input_size=len(centrality_scores),
            output_size=len(ranked),
            details={"top_k": top_k}
        ))
        
        # Format results
        results = []
        for node_id, score in ranked:
            info = self._get_node_info(node_id)
            result_item = {
                'node_id': node_id,
                'score': round(score, 6),
                'node_type': info.get('node_type', 'unknown'),
            }
            
            if centrality_type == "hub_images":
                result_item['objects'] = len(self._image_to_objects.get(node_id, []))
            else:
                result_item['name'] = info.get('name', node_id)
            
            results.append(result_item)
        
        return AdvancedReasoningResult(
            query_type="5. Centrality-Based Query",
            question=f"Find most influential nodes by {centrality_type} centrality",
            results=results,
            reasoning_steps=steps,
            metadata={
                'centrality_type': centrality_type,
                'node_filter': node_filter,
                'nodes_analyzed': len(centrality_scores),
                'top_k': top_k
            }
        )
    
    # ========================================================================
    # CONVENIENCE METHOD: RUN ALL DEMOS
    # ========================================================================
    
    def run_demo(self):
        """Run demonstration of all advanced query types."""
        print("\n" + "=" * 80)
        print("🧠 ADVANCED GRAPH REASONING DEMO")
        print("=" * 80)
        
        # Demo 1: Chain Reasoning
        print("\n📌 Demo 1: Multi-Hop Chain Reasoning")
        result = self.chain_reasoning(
            start_concept="person",
            chain=[
                {"relation": "wearing", "concept": "shirt"},
                {"relation": "to the left of", "concept": "tree"}
            ],
            limit=5
        )
        result.print_result()
        
        # Demo 2: Pattern Matching
        print("\n📌 Demo 2: Subgraph Pattern Matching")
        result = self.pattern_matching(
            pattern_nodes=["person", "shirt", "hat"],
            pattern_edges=[
                ("person", "wearing", "shirt"),
                ("person", "wearing", "hat")
            ],
            limit=5
        )
        result.print_result()
        
        # Demo 3: Scene Comparison
        print("\n📌 Demo 3: Scene Comparison")
        # Get two random image IDs
        image_ids = list(self._image_to_objects.keys())[:2]
        if len(image_ids) >= 2:
            result = self.scene_comparison(image_ids[0], image_ids[1])
            result.print_result()
        
        # Demo 4: Centrality Query
        print("\n📌 Demo 4: Centrality-Based Query")
        result = self.centrality_query(
            centrality_type="degree",
            node_filter="concept",
            top_k=10
        )
        result.print_result()
        
        print("\n" + "=" * 80)
        print("✅ Demo Complete!")
        print("=" * 80)


# ============================================================================
# MAIN - RUN DEMO
# ============================================================================

if __name__ == "__main__":
    print("Initializing Advanced Reasoning Engine...")
    engine = AdvancedReasoningEngine(scale='10k')
    engine.run_demo()
