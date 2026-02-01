"""
Normalization utilities for Evaluation Framework v0.1

Handles term normalization, synonym mapping, and constraint normalization.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional, Union, Any
from functools import lru_cache


class Normalizer:
    """
    Text normalizer with synonym support.
    """

    def __init__(self, synonym_map_path: Optional[str] = None,
                 use_synonyms: bool = True,
                 lowercase: bool = True,
                 strip: bool = True):
        """
        Initialize normalizer.

        Args:
            synonym_map_path: Path to synonym_map.json
            use_synonyms: Whether to apply synonym mapping
            lowercase: Whether to lowercase terms
            strip: Whether to strip whitespace
        """
        self.use_synonyms = use_synonyms
        self.lowercase = lowercase
        self.strip = strip
        self.synonym_map: Dict[str, str] = {}

        if synonym_map_path and use_synonyms:
            self._load_synonym_map(synonym_map_path)

    def _load_synonym_map(self, path: str) -> None:
        """Load synonym mappings from JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.synonym_map = json.load(f)
            # Ensure all keys are lowercase
            self.synonym_map = {k.lower(): v.lower() for k, v in self.synonym_map.items()}
        except FileNotFoundError:
            print(f"[WARN] Synonym map not found at {path}, using empty map")
            self.synonym_map = {}
        except json.JSONDecodeError as e:
            print(f"[WARN] Failed to parse synonym map: {e}")
            self.synonym_map = {}

    def normalize_term(self, term: str) -> str:
        """
        Normalize a single term.

        Steps:
        1. Strip whitespace
        2. Lowercase
        3. Apply synonym mapping
        4. Basic cleanup (extra spaces)
        """
        if not term:
            return ""

        result = term

        if self.strip:
            result = result.strip()

        if self.lowercase:
            result = result.lower()

        # Clean up multiple spaces
        result = re.sub(r'\s+', ' ', result)

        # Apply synonym mapping
        if self.use_synonyms and result in self.synonym_map:
            result = self.synonym_map[result]

        return result

    def normalize_constraint(self, constraint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a constraint dictionary.

        Args:
            constraint: {"type": str, "value": str|dict}

        Returns:
            Normalized constraint
        """
        c_type = constraint.get("type", "")
        c_value = constraint.get("value", "")

        if c_type in ("concept", "attr"):
            # Simple string value
            return {
                "type": c_type,
                "value": self.normalize_term(c_value)
            }
        elif c_type == "rel":
            # Relation value is a dict
            if isinstance(c_value, dict):
                return {
                    "type": c_type,
                    "value": {
                        "name": self.normalize_term(c_value.get("name", "")),
                        "obj": self.normalize_term(c_value.get("obj", ""))
                    }
                }

        return constraint

    def normalize_cqr(self, cqr_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize all constraints in a CQR dictionary.
        """
        result = cqr_dict.copy()

        if "must" in result:
            result["must"] = [self.normalize_constraint(c) for c in result["must"]]

        if "must_not" in result:
            result["must_not"] = [self.normalize_constraint(c) for c in result["must_not"]]

        return result


# Global default normalizer instance
_default_normalizer: Optional[Normalizer] = None


def get_default_normalizer(synonym_map_path: Optional[str] = None) -> Normalizer:
    """Get or create the default normalizer instance."""
    global _default_normalizer
    if _default_normalizer is None:
        # Try default path
        if synonym_map_path is None:
            default_path = Path(__file__).parent.parent / "data" / "synonym_map.json"
            if default_path.exists():
                synonym_map_path = str(default_path)
        _default_normalizer = Normalizer(synonym_map_path)
    return _default_normalizer


def normalize_term(term: str, normalizer: Optional[Normalizer] = None) -> str:
    """Normalize a term using the default or provided normalizer."""
    if normalizer is None:
        normalizer = get_default_normalizer()
    return normalizer.normalize_term(term)


def normalize_constraint(constraint: Dict[str, Any],
                         normalizer: Optional[Normalizer] = None) -> Dict[str, Any]:
    """Normalize a constraint using the default or provided normalizer."""
    if normalizer is None:
        normalizer = get_default_normalizer()
    return normalizer.normalize_constraint(constraint)


def terms_match(term1: str, term2: str, normalizer: Optional[Normalizer] = None) -> bool:
    """Check if two terms match after normalization."""
    if normalizer is None:
        normalizer = get_default_normalizer()
    return normalizer.normalize_term(term1) == normalizer.normalize_term(term2)


def constraints_match(c1: Dict[str, Any], c2: Dict[str, Any],
                      normalizer: Optional[Normalizer] = None) -> bool:
    """Check if two constraints match after normalization."""
    if normalizer is None:
        normalizer = get_default_normalizer()
    n1 = normalizer.normalize_constraint(c1)
    n2 = normalizer.normalize_constraint(c2)
    return n1 == n2


# Simple lemmatization hook (placeholder for future enhancement)
def simple_lemmatize(term: str) -> str:
    """
    Very basic lemmatization.
    Just handles common plural forms.
    For production, consider using NLTK or spaCy.
    """
    term = term.lower().strip()

    # Common irregular plurals
    irregulars = {
        'people': 'person',
        'men': 'man',
        'women': 'woman',
        'children': 'child',
        'feet': 'foot',
        'teeth': 'tooth',
        'mice': 'mouse',
        'geese': 'goose',
        'leaves': 'leaf',
        'knives': 'knife',
        'wives': 'wife',
        'lives': 'life',
        'shelves': 'shelf',
    }

    if term in irregulars:
        return irregulars[term]

    # Simple rules for regular plurals
    if len(term) > 3:
        if term.endswith('ies') and len(term) > 4:
            # babies -> baby
            return term[:-3] + 'y'
        elif term.endswith('es') and term[-3] in 'sxz':
            # boxes -> box
            return term[:-2]
        elif term.endswith('es') and term[-4:-2] in ('ch', 'sh'):
            # watches -> watch
            return term[:-2]
        elif term.endswith('s') and not term.endswith('ss'):
            # cars -> car
            return term[:-1]

    return term

