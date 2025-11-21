"""
Graph package for Phase 4 (entity-centric design).

This package currently exposes:
- EntityStats
- normalize_entity
- extract_entities_from_caption
- build_entity_vocabulary
- save_entity_artifacts

All old Phase 4 graph modules (schema.py, build.py, search.py, store.py, etc.)
have been removed in the phase4-entity branch.
"""

from .entities import (
    EntityStats,
    normalize_entity,
    extract_entities_from_caption,
    build_entity_vocabulary,
    save_entity_artifacts,
)

__all__ = [
    "EntityStats",
    "normalize_entity",
    "extract_entities_from_caption",
    "build_entity_vocabulary",
    "save_entity_artifacts",
]
