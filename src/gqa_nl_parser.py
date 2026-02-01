"""
Natural Language Parser for GQA LightRAG Queries
Parses English natural language queries and maps them to appropriate query types.

Query Types:
1. Entity Search - Find objects by concept/attributes
2. Statistical Knowledge - Probability of co-occurrence  
3. Similarity Search - Find similar instances
4. Relational Path - Path between concepts
5. Negative Constraints - Find with A but not B
6. Comparative - Compare contexts/attributes
7. Hierarchical - Category-based search
8. Anomaly Detection - Rare relations
9. Visual-Attribute Constraint - Multi-condition search
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple


class QueryType(Enum):
    ENTITY_SEARCH = "entity_search"
    STATISTICAL_KNOWLEDGE = "statistical_knowledge"
    SIMILARITY_SEARCH = "similarity_search"
    RELATIONAL_PATH = "relational_path"
    NEGATIVE_CONSTRAINTS = "negative_constraints"
    COMPARATIVE = "comparative"
    HIERARCHICAL = "hierarchical"
    ANOMALY_DETECTION = "anomaly_detection"
    VISUAL_ATTRIBUTE_CONSTRAINT = "visual_attribute_constraint"
    UNKNOWN = "unknown"


@dataclass
class ParseResult:
    """Result of parsing a natural language query"""
    query_type: QueryType
    params: Dict[str, Any]
    confidence: float
    original_query: str
    normalized_query: str = ""
    matched_pattern: str = ""
    
    def is_supported(self) -> bool:
        return self.query_type != QueryType.UNKNOWN
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_type": self.query_type.value,
            "params": self.params,
            "confidence": self.confidence,
            "original_query": self.original_query,
            "normalized_query": self.normalized_query,
            "matched_pattern": self.matched_pattern,
            "is_supported": self.is_supported()
        }


class NaturalLanguageParser:
    """
    Rule-based parser for natural language queries.
    Maps queries to one of 9 query types with extracted parameters.
    """
    
    # Common attributes in the dataset
    KNOWN_ATTRIBUTES = {
        'red', 'blue', 'green', 'yellow', 'white', 'black', 'brown', 'gray', 'grey',
        'orange', 'pink', 'purple', 'gold', 'silver', 'beige', 'tan',
        'large', 'small', 'big', 'tall', 'short', 'long', 'wide', 'narrow', 'thick', 'thin',
        'old', 'young', 'new', 'modern', 'ancient', 'vintage',
        'wooden', 'wood', 'metal', 'plastic', 'glass', 'leather', 'fabric', 'stone', 'brick',
        'open', 'closed', 'empty', 'full', 'round', 'square', 'flat', 'curved',
        'standing', 'sitting', 'walking', 'running', 'lying', 'hanging', 'flying',
        'wet', 'dry', 'clean', 'dirty', 'shiny', 'dull', 'bright', 'dark',
        'striped', 'spotted', 'plain', 'patterned', 'checkered', 'plaid'
    }
    
    # Common concepts/objects in the dataset
    KNOWN_CONCEPTS = {
        # People
        'man', 'woman', 'person', 'people', 'boy', 'girl', 'child', 'children', 'baby',
        # Animals
        'dog', 'cat', 'bird', 'horse', 'cow', 'sheep', 'elephant', 'giraffe', 'zebra',
        # Vehicles
        'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'bike', 'train', 'plane', 'airplane', 'boat',
        # Furniture
        'chair', 'table', 'desk', 'bed', 'sofa', 'couch', 'bench', 'cabinet', 'shelf',
        # Electronics
        'television', 'tv', 'computer', 'laptop', 'phone', 'camera', 'monitor', 'keyboard',
        # Kitchen
        'plate', 'cup', 'bowl', 'bottle', 'glass', 'fork', 'knife', 'spoon', 'pan', 'pot',
        # Nature
        'tree', 'flower', 'grass', 'plant', 'leaf', 'bush', 'mountain', 'sky', 'cloud', 'sun',
        # Buildings/Structures
        'building', 'house', 'window', 'door', 'wall', 'floor', 'ceiling', 'roof', 'fence', 'gate',
        # Clothing
        'shirt', 'pants', 'dress', 'jacket', 'coat', 'hat', 'shoes', 'glasses', 'tie', 'helmet',
        # Food
        'food', 'fruit', 'vegetable', 'bread', 'pizza', 'cake', 'sandwich', 'apple', 'banana', 'orange',
        # Others
        'bag', 'box', 'book', 'sign', 'clock', 'lamp', 'umbrella', 'ball', 'toy', 'picture', 'painting'
    }
    
    # Category hierarchies - must match HIERARCHY_MAPPING in gqa_reasoning_engine.py
    CATEGORY_HIERARCHIES = {
        'furniture': ['chair', 'table', 'desk', 'bed', 'sofa', 'couch', 'bench', 'cabinet', 'shelf'],
        'electronic_devices': ['television', 'tv', 'computer', 'laptop', 'phone', 'camera', 'monitor'],
        'electronics': ['television', 'tv', 'computer', 'laptop', 'phone', 'camera', 'monitor'],
        'vehicle': ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'bike', 'train', 'plane', 'boat'],  # singular
        'vehicles': ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'bike', 'train', 'plane', 'boat'],  # plural alias
        'animal': ['dog', 'cat', 'bird', 'horse', 'cow', 'sheep', 'elephant', 'giraffe', 'zebra'],  # singular
        'animals': ['dog', 'cat', 'bird', 'horse', 'cow', 'sheep', 'elephant', 'giraffe', 'zebra'],  # plural alias
        'people': ['man', 'woman', 'person', 'boy', 'girl', 'child', 'baby'],
        'clothing': ['shirt', 'pants', 'dress', 'jacket', 'coat', 'hat', 'shoes', 'glasses', 'tie'],
        'food': ['fruit', 'vegetable', 'bread', 'pizza', 'cake', 'sandwich', 'apple', 'banana'],
        'kitchenware': ['plate', 'cup', 'bowl', 'bottle', 'glass', 'fork', 'knife', 'spoon', 'pan', 'pot'],
        'nature': ['tree', 'flower', 'grass', 'plant', 'leaf', 'bush', 'sky', 'cloud', 'mountain'],
        'plants': ['tree', 'flower', 'grass', 'plant', 'leaf', 'bush'],
        'body_part': ['head', 'hand', 'arm', 'leg', 'face', 'eye', 'ear', 'nose', 'mouth'],
        'building_structure': ['building', 'house', 'wall', 'window', 'door', 'roof', 'floor', 'ceiling']
    }
    
    # Common relations
    KNOWN_RELATIONS = {
        'on', 'in', 'near', 'next to', 'beside', 'behind', 'in front of', 'above', 'below',
        'under', 'over', 'inside', 'outside', 'between', 'around', 'through',
        'to the left of', 'to the right of', 'on top of', 'at the bottom of',
        'wearing', 'holding', 'riding', 'eating', 'sitting on', 'standing on',
        'lying on', 'walking on', 'looking at', 'watching', 'playing with'
    }
    
    def __init__(self):
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for each query type"""
        
        # Pattern definitions for each query type
        self.patterns = {
            QueryType.ENTITY_SEARCH: [
                # "find all red cars"
                (r"find\s+(?:all\s+)?(.+?)$", 0.8),
                # "show me white shirts"
                (r"show\s+(?:me\s+)?(?:all\s+)?(.+?)$", 0.8),
                # "get all large tables"
                (r"get\s+(?:all\s+)?(.+?)$", 0.7),
                # "list red cars"
                (r"list\s+(?:all\s+)?(.+?)$", 0.7),
                # "search for blue chairs"
                (r"search\s+(?:for\s+)?(.+?)$", 0.7),
                # "what are the red cars"
                (r"what\s+(?:are\s+)?(?:the\s+)?(.+?)$", 0.6),
                # "which cars are red"
                (r"which\s+(\w+)\s+(?:are|is)\s+(.+?)$", 0.7),
            ],
            
            QueryType.STATISTICAL_KNOWLEDGE: [
                # "what is the probability of finding shirt near man"
                (r"(?:what\s+is\s+)?(?:the\s+)?probability\s+of\s+(?:finding\s+)?(\w+)\s+(?:near|with|by|next to)\s+(\w+)", 0.95),
                # "how likely is it to find a dog near a person"
                (r"how\s+likely\s+(?:is\s+it\s+)?to\s+find\s+(?:a\s+)?(\w+)\s+(?:near|with|by)\s+(?:a\s+)?(\w+)", 0.9),
                # "probability of window near building"
                (r"probability\s+(?:of\s+)?(\w+)\s+(?:near|with|by)\s+(\w+)", 0.9),
                # "how often do we see shirt with man"
                (r"how\s+often\s+(?:do\s+we\s+)?(?:see|find)\s+(\w+)\s+(?:with|near|by)\s+(\w+)", 0.85),
                # "how often are dogs near people" - NEW PATTERN
                (r"how\s+often\s+(?:are|is)\s+(\w+)\s+(?:near|with|by|next to)\s+(\w+)", 0.9),
                # "co-occurrence of A and B"
                (r"co-?occurrence\s+of\s+(\w+)\s+(?:and|with)\s+(\w+)", 0.9),
                # "how common is it to see A near B"
                (r"how\s+common\s+(?:is\s+it\s+)?to\s+(?:see|find)\s+(\w+)\s+(?:near|with)\s+(\w+)", 0.85),
                # "frequency of A with B"
                (r"frequency\s+of\s+(\w+)\s+(?:with|near)\s+(\w+)", 0.9),
                # "how frequently does A appear with B"
                (r"how\s+frequent(?:ly)?\s+(?:does|do)\s+(\w+)\s+appear\s+(?:with|near)\s+(\w+)", 0.85),
            ],
            
            QueryType.SIMILARITY_SEARCH: [
                # "find objects similar to large green tree"
                (r"(?:find\s+)?(?:objects?\s+)?similar\s+to\s+(.+?)$", 0.9),
                # "what looks like a red car"
                (r"what\s+looks?\s+like\s+(?:a\s+)?(.+?)$", 0.85),
                # "find things that resemble white shirt"
                (r"(?:find\s+)?(?:things?\s+)?(?:that\s+)?resemble\s+(?:a\s+)?(.+?)$", 0.85),
                # "similar to red cars"
                (r"similar\s+to\s+(.+?)$", 0.8),
                # "like a blue chair"
                (r"like\s+(?:a\s+)?(.+?)$", 0.6),
            ],
            
            QueryType.RELATIONAL_PATH: [
                # "find path from man to shirt via wearing"
                (r"(?:find\s+)?path\s+from\s+(\w+)\s+to\s+(\w+)(?:\s+via\s+(\w+))?", 0.95),
                # "how is man connected to shirt"
                (r"how\s+is\s+(\w+)\s+connected\s+to\s+(\w+)", 0.9),
                # "connection between plate and table"
                (r"connection\s+between\s+(\w+)\s+and\s+(\w+)", 0.9),
                # "relationship from dog to person"
                (r"relationship\s+from\s+(\w+)\s+to\s+(\w+)", 0.85),
                # "path between car and road"
                (r"path\s+between\s+(\w+)\s+and\s+(\w+)", 0.85),
                # "how does A relate to B"
                (r"how\s+does\s+(\w+)\s+relate\s+to\s+(\w+)", 0.8),
                # "what paths connect A to B" - NEW PATTERN
                (r"(?:what\s+)?paths?\s+connect\s+(\w+)\s+to\s+(\w+)", 0.9),
                # "what connects A to B"
                (r"what\s+connects?\s+(\w+)\s+to\s+(\w+)", 0.9),
                # "how are A and B connected"
                (r"how\s+are\s+(\w+)\s+and\s+(\w+)\s+connected", 0.85),
                # "link between A and B"
                (r"link\s+between\s+(\w+)\s+and\s+(\w+)", 0.85),
            ],
            
            QueryType.NEGATIVE_CONSTRAINTS: [
                # "find images with man but not woman"
                (r"(?:find\s+)?images?\s+with\s+(\w+)\s+(?:but\s+)?(?:not|without)\s+(\w+)", 0.95),
                # "images containing tree but no car"
                (r"images?\s+containing\s+(\w+)\s+(?:but\s+)?(?:no|without)\s+(\w+)", 0.9),
                # "show me photos with dog but without cat"
                (r"(?:show\s+me\s+)?photos?\s+with\s+(\w+)\s+(?:but\s+)?without\s+(\w+)", 0.9),
                # "find scenes with A excluding B"
                (r"(?:find\s+)?scenes?\s+with\s+(\w+)\s+excluding\s+(\w+)", 0.85),
                # "A but not B"
                (r"(\w+)\s+but\s+not\s+(\w+)", 0.7),
                # "A without B"
                (r"(\w+)\s+without\s+(\w+)", 0.7),
            ],
            
            QueryType.COMPARATIVE: [
                # "compare the number of chairs in kitchen and living room"
                (r"compare\s+(?:the\s+)?(?:number\s+of\s+)?(\w+)\s+in\s+(.+?)\s+(?:and|vs|versus)\s+(.+?)$", 0.95),
                # "compare the number of people indoors vs outdoors" - NEW
                (r"compare\s+(?:the\s+)?(?:number\s+of\s+)?(\w+)\s+(indoors?|outdoors?)\s+(?:vs|versus|and)\s+(indoors?|outdoors?)$", 0.95),
                # "is white more common on wall or bed"
                (r"is\s+(\w+)\s+more\s+common\s+(?:on|in|with)\s+(\w+)\s+or\s+(\w+)", 0.9),
                # "which has more chairs: kitchen or living room"
                (r"which\s+has\s+more\s+(\w+)[:\s]+(\w+)\s+or\s+(\w+)", 0.9),
                # "compare white walls vs white beds"
                (r"compare\s+(\w+)\s+(\w+)\s+(?:vs|versus|and)\s+(\w+)\s+(\w+)", 0.85),
                # "more X in A or B"
                (r"more\s+(\w+)\s+in\s+(\w+)\s+or\s+(\w+)", 0.8),
                # "A vs B for attribute X"
                (r"(\w+)\s+(?:vs|versus)\s+(\w+)\s+(?:for|with)\s+(\w+)", 0.8),
                # "compare X indoors and outdoors"
                (r"compare\s+(\w+)\s+(indoors?)\s+(?:and|vs)\s+(outdoors?)", 0.95),
                # "how many X in A vs B"
                (r"how\s+many\s+(\w+)\s+(?:in|at)\s+(\w+)\s+(?:vs|versus|compared to)\s+(\w+)", 0.85),
            ],
            
            QueryType.HIERARCHICAL: [
                # "list all furniture"
                (r"list\s+(?:all\s+)?(\w+)$", 0.5),  # Low confidence, checked against categories
                # "show all types of vehicles"
                (r"(?:show|list|get)\s+(?:all\s+)?types?\s+of\s+(\w+)", 0.9),
                # "what are the electronic devices"
                (r"what\s+are\s+(?:the\s+)?(\w+)(?:\s+types?)?$", 0.6),
                # "find all items in category furniture"
                (r"(?:find\s+)?(?:all\s+)?items?\s+in\s+category\s+(\w+)", 0.95),
                # "things that belong to furniture category"
                (r"things?\s+(?:that\s+)?belong\s+to\s+(\w+)\s+category", 0.9),
                # "all furniture items"
                (r"all\s+(\w+)\s+items?", 0.7),
                # "count furniture"
                (r"count\s+(\w+)$", 0.6),
            ],
            
            QueryType.ANOMALY_DETECTION: [
                # "find unusual cases of dog on table"
                (r"(?:find\s+)?unusual\s+(?:cases?\s+)?(?:of\s+)?(\w+)\s+(\w+)\s+(\w+)", 0.9),
                # "rare instances of cat on car"
                (r"rare\s+instances?\s+(?:of\s+)?(\w+)\s+(?:on|in|near)\s+(\w+)", 0.9),
                # "find anomalies with dog on table"
                (r"(?:find\s+)?anomal(?:y|ies)\s+(?:with\s+)?(\w+)\s+(\w+)\s+(\w+)", 0.9),
                # "unusual relationship between A and B"
                (r"unusual\s+relationship\s+between\s+(\w+)\s+and\s+(\w+)", 0.85),
                # "find cases where dog is on table"
                (r"(?:find\s+)?cases?\s+where\s+(\w+)\s+is\s+(\w+)\s+(\w+)", 0.85),
                # "what are the rarest relations"
                (r"(?:what\s+are\s+)?(?:the\s+)?rarest\s+relations?", 0.9),
                # "top N rare/unusual relations"
                (r"top\s+(\d+)\s+(?:rare|unusual)\s+relations?", 0.9),
                # "find rare relations"
                (r"(?:find\s+)?rare\s+relations?", 0.85),
                # "what unusual/rare object-relation pairs exist" - NEW PATTERN
                (r"(?:what\s+)?unusual\s+(?:object[- ]relation\s+)?pairs?\s+exist", 0.95),
                # "find unusual patterns/pairs"
                (r"(?:find\s+)?unusual\s+(?:patterns?|pairs?|combinations?)", 0.9),
                # "rare patterns in the data"
                (r"rare\s+patterns?\s+(?:in\s+(?:the\s+)?(?:data|graph|knowledge))?", 0.85),
                # "detect anomalies"
                (r"detect\s+anomal(?:y|ies)", 0.9),
            ],
            
            QueryType.VISUAL_ATTRIBUTE_CONSTRAINT: [
                # "find red plastic cups"
                (r"find\s+(.+?)\s+(\w+)s?$", 0.5),  # Very general, needs attribute check
                # "tall man wearing black shirt"
                (r"(\w+)\s+(\w+)\s+(\w+)\s+(\w+)\s+(\w+)", 0.4),  # Very general
                # "find X with attributes A and B that is related to Y"
                (r"find\s+(\w+)\s+(?:with\s+)?(?:attributes?\s+)?(.+?)\s+(?:that\s+is\s+)?(?:related\s+to|near|on)\s+(\w+)", 0.85),
                # "X that is both A and B"
                (r"(\w+)\s+that\s+is\s+both\s+(\w+)\s+and\s+(\w+)", 0.9),
                # "A and B X" (e.g., "red and large car")
                (r"(\w+)\s+and\s+(\w+)\s+(\w+)", 0.7),
                # "X with A doing/relation Y with Z"
                (r"(\w+)\s+(?:with\s+)?(\w+)\s+(\w+)\s+(\w+)\s+(?:with\s+)?(\w+)", 0.6),
            ],
        }
    
    def normalize_query(self, query: str) -> str:
        """Normalize the query for parsing"""
        # Convert to lowercase
        query = query.lower().strip()
        # Remove extra spaces
        query = re.sub(r'\s+', ' ', query)
        # Remove punctuation at the end
        query = re.sub(r'[?!.,;:]+$', '', query)
        # Standardize some phrases
        query = query.replace("what is", "what are")
        query = query.replace("show me", "show")
        return query
    
    def extract_attributes_and_concept(self, phrase: str) -> Tuple[List[str], Optional[str]]:
        """Extract attributes and concept from a phrase like 'red large car'"""
        words = phrase.lower().split()
        attributes = []
        concept = None
        
        for word in words:
            word = word.strip('.,!?;:')
            if word in self.KNOWN_ATTRIBUTES:
                attributes.append(word)
            elif word in self.KNOWN_CONCEPTS:
                concept = word
            elif word.rstrip('s') in self.KNOWN_CONCEPTS:
                concept = word.rstrip('s')  # Handle plurals
        
        # If no concept found, take the last word as concept
        if concept is None and words:
            last_word = words[-1].strip('.,!?;:')
            if last_word.rstrip('s') not in self.KNOWN_ATTRIBUTES:
                concept = last_word.rstrip('s')
        
        return attributes, concept
    
    def is_category(self, word: str) -> bool:
        """Check if a word is a known category"""
        return word.lower() in self.CATEGORY_HIERARCHIES
    
    def get_category_members(self, category: str) -> List[str]:
        """Get members of a category"""
        return self.CATEGORY_HIERARCHIES.get(category.lower(), [])
    
    def extract_relation(self, query: str) -> Optional[str]:
        """Extract relation from query"""
        query_lower = query.lower()
        for relation in sorted(self.KNOWN_RELATIONS, key=len, reverse=True):
            if relation in query_lower:
                return relation
        return None
    
    def parse(self, query: str) -> ParseResult:
        """Parse a natural language query and return a structured result"""
        original_query = query
        normalized = self.normalize_query(query)
        
        # Try each query type
        best_match = None
        best_confidence = 0
        best_pattern = ""
        best_params = {}
        best_type = QueryType.UNKNOWN
        
        for query_type, patterns in self.patterns.items():
            for pattern, base_confidence in patterns:
                match = re.search(pattern, normalized, re.IGNORECASE)
                if match:
                    params, confidence_modifier = self._extract_params(
                        query_type, match, normalized
                    )
                    
                    adjusted_confidence = base_confidence * confidence_modifier
                    
                    if adjusted_confidence > best_confidence:
                        best_confidence = adjusted_confidence
                        best_match = match
                        best_pattern = pattern
                        best_params = params
                        best_type = query_type
        
        # Post-processing for certain query types
        if best_type == QueryType.HIERARCHICAL:
            # Verify it's actually a category
            if 'category' in best_params:
                if not self.is_category(best_params['category']):
                    # Might be an entity search instead
                    attrs, concept = self.extract_attributes_and_concept(normalized)
                    if concept:
                        best_type = QueryType.ENTITY_SEARCH
                        best_params = {'concept': concept, 'attributes': attrs}
        
        elif best_type == QueryType.ENTITY_SEARCH:
            # Extract concept and attributes from the captured phrase
            if 'phrase' in best_params:
                attrs, concept = self.extract_attributes_and_concept(best_params['phrase'])
                if concept:
                    best_params = {'concept': concept, 'attributes': attrs}
                else:
                    best_confidence *= 0.5
        
        return ParseResult(
            query_type=best_type,
            params=best_params,
            confidence=best_confidence,
            original_query=original_query,
            normalized_query=normalized,
            matched_pattern=best_pattern
        )
    
    def _extract_params(
        self, 
        query_type: QueryType, 
        match: re.Match, 
        normalized: str
    ) -> Tuple[Dict[str, Any], float]:
        """Extract parameters based on query type and match"""
        params = {}
        confidence_modifier = 1.0
        groups = match.groups()
        
        if query_type == QueryType.ENTITY_SEARCH:
            if len(groups) >= 1:
                phrase = groups[0]
                attrs, concept = self.extract_attributes_and_concept(phrase)
                if concept:
                    params = {'concept': concept, 'attributes': attrs, 'phrase': phrase}
                else:
                    params = {'phrase': phrase}
                    confidence_modifier = 0.7
                    
        elif query_type == QueryType.STATISTICAL_KNOWLEDGE:
            if len(groups) >= 2:
                params = {
                    'concept_a': groups[1] if len(groups) > 1 else groups[0],
                    'concept_b': groups[0] if len(groups) > 1 else None
                }
                # Swap if needed based on pattern
                if 'near' in normalized or 'with' in normalized:
                    # "A near B" -> concept_a=B, concept_b=A for P(A|B)
                    params = {'concept_a': groups[1], 'concept_b': groups[0]}
                    
        elif query_type == QueryType.SIMILARITY_SEARCH:
            if len(groups) >= 1:
                attrs, concept = self.extract_attributes_and_concept(groups[0])
                params = {'concept': concept, 'attributes': attrs}
                if concept:
                    confidence_modifier = 1.0
                else:
                    confidence_modifier = 0.6
                    
        elif query_type == QueryType.RELATIONAL_PATH:
            if len(groups) >= 2:
                params = {
                    'source_concept': groups[0],
                    'target_concept': groups[1],
                    'via_relation': groups[2] if len(groups) > 2 and groups[2] else None
                }
                
        elif query_type == QueryType.NEGATIVE_CONSTRAINTS:
            if len(groups) >= 2:
                params = {
                    'concept_present': groups[0],
                    'concept_absent': groups[1]
                }
                
        elif query_type == QueryType.COMPARATIVE:
            if len(groups) >= 3:
                params = {
                    'target_concept': groups[0],
                    'context_a': groups[1].strip(),
                    'context_b': groups[2].strip()
                }
            elif len(groups) == 2:
                params = {
                    'concept_a': groups[0],
                    'concept_b': groups[1]
                }
                
        elif query_type == QueryType.HIERARCHICAL:
            if len(groups) >= 1:
                category = groups[0]
                # Try singular form if plural doesn't match
                if not self.is_category(category) and category.endswith('s'):
                    category = category[:-1]  # Remove trailing 's'
                if self.is_category(category):
                    params = {
                        'category': category,
                        'child_concepts': self.get_category_members(category)
                    }
                    confidence_modifier = 1.0  # Maximum confidence (valid category)
                else:
                    params = {'category': category}
                    confidence_modifier = 0.5  # Lower confidence
                    
        elif query_type == QueryType.ANOMALY_DETECTION:
            if 'rarest' in normalized or 'rare relations' in normalized or 'unusual' in normalized or 'detect anomal' in normalized:
                # General rare relations query
                top_n_match = re.search(r'top\s+(\d+)', normalized)
                params = {
                    'find_rare_relations': True
                }
                # Only set top_n if explicitly specified in query
                if top_n_match:
                    params['top_n'] = int(top_n_match.group(1))
            elif len(groups) >= 3:
                params = {
                    'subject_concept': groups[0],
                    'relation': groups[1],
                    'object_concept': groups[2]
                }
            elif len(groups) >= 2:
                # Try to extract relation from context
                relation = self.extract_relation(normalized)
                params = {
                    'subject_concept': groups[0],
                    'relation': relation or 'on',
                    'object_concept': groups[1]
                }
                
        elif query_type == QueryType.VISUAL_ATTRIBUTE_CONSTRAINT:
            # Complex parsing for multi-attribute constraints
            attrs, concept = self.extract_attributes_and_concept(normalized)
            
            # Check for relational constraints
            relation = self.extract_relation(normalized)
            
            if relation:
                # Try to find subject and object concepts
                parts = normalized.split(relation)
                if len(parts) == 2:
                    main_attrs, main_concept = self.extract_attributes_and_concept(parts[0])
                    related_attrs, related_concept = self.extract_attributes_and_concept(parts[1])
                    params = {
                        'main_concept': main_concept,
                        'main_attributes': main_attrs,
                        'relation': relation,
                        'related_concept': related_concept,
                        'related_attributes': related_attrs
                    }
            else:
                params = {'concept': concept, 'attributes': attrs}
                
            if not params.get('concept') and not params.get('main_concept'):
                confidence_modifier = 0.5
        
        return params, confidence_modifier
    
    def get_supported_patterns(self) -> Dict[str, List[str]]:
        """Return example queries for each query type"""
        return {
            "entity_search": [
                "Find all red cars",
                "Show me white shirts",
                "List large tables",
                "Get all wooden chairs"
            ],
            "statistical_knowledge": [
                "What is the probability of finding shirt near man",
                "How likely is it to find window near building",
                "How often do we see plate with table",
                "Co-occurrence of dog and person"
            ],
            "similarity_search": [
                "Find objects similar to large green tree",
                "What looks like a red car",
                "Find things that resemble white shirt"
            ],
            "relational_path": [
                "Find path from man to shirt via wearing",
                "How is man connected to shirt",
                "Connection between plate and table",
                "Path between car and road"
            ],
            "negative_constraints": [
                "Find images with man but not woman",
                "Images containing tree but no car",
                "Show photos with dog but without cat"
            ],
            "comparative": [
                "Compare the number of chairs in kitchen and living room",
                "Is white more common on wall or bed",
                "Which has more chairs: kitchen or living room"
            ],
            "hierarchical": [
                "List all furniture",
                "Show all types of vehicles",
                "What are the electronic devices",
                "Count animals"
            ],
            "anomaly_detection": [
                "Find unusual cases of dog on table",
                "Rare instances of cat on car",
                "What are the rarest relations",
                "Top 5 rare relations"
            ],
            "visual_attribute_constraint": [
                "Find red plastic cups",
                "Tall man wearing black shirt",
                "Large wooden table with white plates"
            ]
        }


# ============================================================
# Demo and Testing
# ============================================================

def demo_parser():
    """Demo the parser with example queries"""
    parser = NaturalLanguageParser()
    
    test_queries = [
        # Entity Search
        "Find all red cars",
        "Show me white shirts",
        "Get large wooden tables",
        
        # Statistical Knowledge
        "What is the probability of finding shirt near man",
        "How often do we see window near building",
        
        # Similarity Search
        "Find objects similar to large green tree",
        "What looks like a red car",
        
        # Relational Path
        "Find path from man to shirt via wearing",
        "How is plate connected to table",
        
        # Negative Constraints
        "Find images with man but not woman",
        "Images containing tree without car",
        
        # Comparative
        "Compare the number of chairs in kitchen and living room",
        "Is white more common on wall or bed",
        
        # Hierarchical
        "List all furniture",
        "Show all types of vehicles",
        
        # Anomaly Detection
        "Find unusual cases of dog on table",
        "What are the rarest relations",
        
        # Visual-Attribute Constraint
        "Find red plastic cups",
        "Tall man wearing black shirt",
    ]
    
    print("=" * 70)
    print("NATURAL LANGUAGE PARSER DEMO")
    print("=" * 70)
    
    for query in test_queries:
        result = parser.parse(query)
        print(f"\nQuery: {query}")
        print(f"  Type: {result.query_type.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Params: {result.params}")
        print("-" * 50)


if __name__ == "__main__":
    demo_parser()
