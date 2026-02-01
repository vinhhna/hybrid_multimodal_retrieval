"""
Canonical Query Representation (CQR) for Evaluation Framework v0.1

Defines the standard query structure used across all evaluation components.
This representation is output_type-centric, not tied to specific query types.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import json


class OutputType(str, Enum):
    """Supported output types for queries."""
    RANKED_SET = "ranked_set"
    SCALAR = "scalar"
    PATH = "path"
    SUBGRAPH = "subgraph"


class ConstraintType(str, Enum):
    """Types of constraints in a query."""
    CONCEPT = "concept"
    ATTR = "attr"
    REL = "rel"


@dataclass
class Constraint:
    """
    A single constraint in a query.

    Types:
    - concept: {"type": "concept", "value": "car"}
    - attr: {"type": "attr", "value": "red"}
    - rel: {"type": "rel", "value": {"name": "on", "obj": "road"}}
    """
    type: str
    value: Union[str, Dict[str, str]]

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "value": self.value}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Constraint':
        return cls(type=d["type"], value=d["value"])

    @classmethod
    def concept(cls, name: str) -> 'Constraint':
        """Create a concept constraint."""
        return cls(type=ConstraintType.CONCEPT.value, value=name)

    @classmethod
    def attr(cls, name: str) -> 'Constraint':
        """Create an attribute constraint."""
        return cls(type=ConstraintType.ATTR.value, value=name)

    @classmethod
    def rel(cls, name: str, obj: str) -> 'Constraint':
        """Create a relation constraint."""
        return cls(type=ConstraintType.REL.value, value={"name": name, "obj": obj})

    def __eq__(self, other):
        if not isinstance(other, Constraint):
            return False
        return self.type == other.type and self.value == other.value

    def __hash__(self):
        if isinstance(self.value, dict):
            return hash((self.type, tuple(sorted(self.value.items()))))
        return hash((self.type, self.value))


@dataclass
class CQR:
    """
    Canonical Query Representation.

    A unified query format that bridges NL queries, the engine, and evaluation.
    Designed to be output_type-centric and extensible.
    """
    query_id: str
    output_type: str  # "ranked_set", "scalar", "path", "subgraph"
    op: str           # "retrieve", "stats", "path", "similarity", "compare"
    must: List[Constraint] = field(default_factory=list)
    must_not: List[Constraint] = field(default_factory=list)
    k: int = 50
    return_unit: str = "images"  # "images" or "objects"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query_id": self.query_id,
            "output_type": self.output_type,
            "op": self.op,
            "must": [c.to_dict() for c in self.must],
            "must_not": [c.to_dict() for c in self.must_not],
            "k": self.k,
            "return_unit": self.return_unit,
            "meta": self.meta
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CQR':
        """Create from dictionary."""
        return cls(
            query_id=d["query_id"],
            output_type=d["output_type"],
            op=d["op"],
            must=[Constraint.from_dict(c) for c in d.get("must", [])],
            must_not=[Constraint.from_dict(c) for c in d.get("must_not", [])],
            k=d.get("k", 50),
            return_unit=d.get("return_unit", "images"),
            meta=d.get("meta", {})
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> 'CQR':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def get_concepts(self) -> List[str]:
        """Extract all concept values from must constraints."""
        return [c.value for c in self.must if c.type == ConstraintType.CONCEPT.value]

    def get_attributes(self) -> List[str]:
        """Extract all attribute values from must constraints."""
        return [c.value for c in self.must if c.type == ConstraintType.ATTR.value]

    def get_relations(self) -> List[Dict[str, str]]:
        """Extract all relation values from must constraints."""
        return [c.value for c in self.must if c.type == ConstraintType.REL.value]

    def get_negative_concepts(self) -> List[str]:
        """Extract all concept values from must_not constraints."""
        return [c.value for c in self.must_not if c.type == ConstraintType.CONCEPT.value]

    def has_negation(self) -> bool:
        """Check if query has any must_not constraints."""
        return len(self.must_not) > 0

    def copy(self, **kwargs) -> 'CQR':
        """Create a copy with optional field overrides."""
        d = self.to_dict()
        d.update(kwargs)
        if "must" in kwargs and isinstance(kwargs["must"], list):
            if kwargs["must"] and isinstance(kwargs["must"][0], Constraint):
                d["must"] = [c.to_dict() for c in kwargs["must"]]
        if "must_not" in kwargs and isinstance(kwargs["must_not"], list):
            if kwargs["must_not"] and isinstance(kwargs["must_not"][0], Constraint):
                d["must_not"] = [c.to_dict() for c in kwargs["must_not"]]
        return CQR.from_dict(d)


@dataclass
class GoldOutput:
    """
    Gold standard output for evaluation.
    Structure varies by output_type.
    """
    output_type: str
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"output_type": self.output_type, "data": self.data}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'GoldOutput':
        return cls(output_type=d["output_type"], data=d["data"])

    @classmethod
    def ranked_set(cls, image_ids: List[str], notes: str = "") -> 'GoldOutput':
        """Create gold output for ranked_set queries."""
        return cls(
            output_type=OutputType.RANKED_SET.value,
            data={"image_ids": image_ids, "gold_size": len(image_ids), "notes": notes}
        )

    @classmethod
    def scalar(cls, value: float, numerator: int = 0, denominator: int = 0,
               notes: str = "") -> 'GoldOutput':
        """Create gold output for scalar queries."""
        return cls(
            output_type=OutputType.SCALAR.value,
            data={
                "value": value,
                "numerator": numerator,
                "denominator": denominator,
                "notes": notes
            }
        )

    @classmethod
    def path(cls, exists: bool, shortest_hops: Optional[int] = None,
             example_path: Optional[List[str]] = None, notes: str = "") -> 'GoldOutput':
        """Create gold output for path queries."""
        return cls(
            output_type=OutputType.PATH.value,
            data={
                "exists": exists,
                "shortest_hops": shortest_hops,
                "example_path": example_path,
                "notes": notes
            }
        )

    @classmethod
    def subgraph(cls, nodes: List[str], edges: List[tuple],
                 image_ids: Optional[List[str]] = None, notes: str = "") -> 'GoldOutput':
        """Create gold output for subgraph queries."""
        return cls(
            output_type=OutputType.SUBGRAPH.value,
            data={
                "nodes": nodes,
                "edges": [list(e) for e in edges],  # Convert tuples for JSON
                "image_ids": image_ids or [],
                "notes": notes
            }
        )


@dataclass
class QuerySuiteItem:
    """
    A single item in a query suite.
    Contains the gold CQR, NL templates, and gold output.
    """
    query_id: str
    cqr_gold: CQR
    nl_templates: List[str]
    gold_output: GoldOutput
    gold_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "cqr_gold": self.cqr_gold.to_dict(),
            "nl_templates": self.nl_templates,
            "gold_output": self.gold_output.to_dict(),
            "gold_summary": self.gold_summary
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QuerySuiteItem':
        return cls(
            query_id=d["query_id"],
            cqr_gold=CQR.from_dict(d["cqr_gold"]),
            nl_templates=d["nl_templates"],
            gold_output=GoldOutput.from_dict(d["gold_output"]),
            gold_summary=d.get("gold_summary", {})
        )

    def to_jsonl(self) -> str:
        """Serialize to a single JSONL line."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_jsonl(cls, line: str) -> 'QuerySuiteItem':
        """Deserialize from a JSONL line."""
        return cls.from_dict(json.loads(line.strip()))


def load_suite(filepath: str) -> List[QuerySuiteItem]:
    """Load a query suite from a JSONL file."""
    items = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(QuerySuiteItem.from_jsonl(line))
    return items


def save_suite(items: List[QuerySuiteItem], filepath: str) -> None:
    """Save a query suite to a JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(item.to_jsonl() + '\n')

