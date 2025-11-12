"""Encoders module for Phase 4."""

from .clip_space import (
    l2_normalize,
    assert_clip_shape,
    to_numpy_f32,
    ensure_clip_aligned,
)

__all__ = [
    "l2_normalize",
    "assert_clip_shape",
    "to_numpy_f32",
    "ensure_clip_aligned",
]
