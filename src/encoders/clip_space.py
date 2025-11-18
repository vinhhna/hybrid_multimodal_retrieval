"""
CLIP-space Alignment Utilities
Ensures all node features are L2-normalized, 512-dim float32 vectors.
"""

import numpy as np
import torch
from typing import Union

# === CLIP-SPACE ALIGNMENT UTILS (ANCHOR) ===


def l2_normalize(
    x: Union[np.ndarray, torch.Tensor], eps: float = 1e-12
) -> Union[np.ndarray, torch.Tensor]:
    """
    L2-normalize vectors along the last dimension.
    
    Preserves container type (NumPy → NumPy, Torch → Torch).
    Output is always cast to float32.
    
    Args:
        x: Input array/tensor to normalize (any shape ending in feature dim)
        eps: Small epsilon to prevent division by zero
    
    Returns:
        L2-normalized array/tensor (same type as input) as float32.
        Container type is preserved; dtype is always float32.
    
    Example:
        >>> v = np.random.rand(3, 512).astype(np.float32)
        >>> u = l2_normalize(v)
        >>> norms = np.linalg.norm(u, axis=-1)
        >>> np.allclose(norms, 1.0, atol=1e-4)
        True
    """
    if isinstance(x, np.ndarray):
        denom = np.maximum(np.linalg.norm(x, ord=2, axis=-1, keepdims=True), eps)
        normalized = (x / denom).astype(np.float32, copy=False)
        # NaN/Inf guard after normalization
        if not np.all(np.isfinite(normalized)):
            raise ValueError(
                f"NumPy: L2-normalization produced non-finite values. "
                f"Input may contain NaN/Inf or near-zero vectors."
            )
        return normalized
    elif torch.is_tensor(x):
        denom = torch.clamp(x.norm(p=2, dim=-1, keepdim=True), min=eps)
        normalized = (x / denom).to(dtype=torch.float32)
        # NaN/Inf guard after normalization
        if not torch.all(torch.isfinite(normalized)):
            raise ValueError(
                f"Torch: L2-normalization produced non-finite values. "
                f"Input may contain NaN/Inf or near-zero vectors."
            )
        return normalized
    else:
        raise TypeError("Unsupported type for l2_normalize")


def assert_clip_shape(
    x: Union[np.ndarray, torch.Tensor], dim: int = 512
) -> None:
    """
    Assert that vector has the expected CLIP dimension.
    
    Args:
        x: Input array/tensor to check
        dim: Expected dimension (default 512 for CLIP ViT-B/32)
    
    Raises:
        ValueError: If dimension doesn't match
        TypeError: If input is neither numpy array nor torch tensor
    
    Example:
        >>> v = np.zeros((10, 512))
        >>> assert_clip_shape(v, 512)  # OK
        >>> assert_clip_shape(v, 256)  # AssertionError
    """
    if isinstance(x, np.ndarray):
        if x.shape[-1] != dim:
            raise ValueError(f"Expected dim={dim}, got {x.shape[-1]}")
    elif torch.is_tensor(x):
        if x.size(-1) != dim:
            raise ValueError(f"Expected dim={dim}, got {x.size(-1)}")
    else:
        raise TypeError("Unsupported type for assert_clip_shape")


def to_numpy_f32(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Convert input to contiguous float32 numpy array.
    
    Preserves data but converts container type to NumPy.
    Output is always float32.
    
    Args:
        x: Input array or tensor
    
    Returns:
        Contiguous float32 numpy array.
        Always returns NumPy array; dtype is always float32.
    
    Example:
        >>> t = torch.randn(3, 512)
        >>> arr = to_numpy_f32(t)
        >>> isinstance(arr, np.ndarray) and arr.dtype == np.float32
        True
    """
    if isinstance(x, np.ndarray):
        return np.ascontiguousarray(x, dtype=np.float32)
    elif torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.float32)
    else:
        raise TypeError(f"Unsupported type for to_numpy_f32: {type(x)}")


def ensure_clip_aligned(
    x: Union[np.ndarray, torch.Tensor],
    dim: int = 512,
    renorm: bool = False,
    check_unit_norm: bool = False,
    tol: float = 1e-4,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Ensure vector is properly aligned in CLIP space.
    
    Preserves container type (NumPy → NumPy, Torch → Torch).
    When renorm=True, output is cast to float32.
    
    Policy:
    - Always check shape matches dim
    - If renorm=True: normalize and cast to float32
    - If renorm=False: optionally check unit norm (when check_unit_norm=True)
    
    Args:
        x: Input array/tensor
        dim: Expected CLIP dimension (default 512)
        renorm: If True, force re-normalization and float32 cast
        check_unit_norm: If True and renorm=False, verify vectors are unit norm
        tol: Tolerance for unit norm check
    
    Returns:
        Aligned vector (same type as input).
        Container type is preserved; when renorm=True, dtype is float32.
    
    Raises:
        AssertionError: If shape doesn't match or unit norm check fails
        ValueError: If renormalization produces non-finite values
    
    Example:
        >>> v = np.random.rand(3, 512)
        >>> u = ensure_clip_aligned(v, dim=512, renorm=True)
        >>> norms = np.linalg.norm(u, axis=-1)
        >>> np.allclose(norms, 1.0, atol=1e-4)
        True
    """
    # Check shape
    assert_clip_shape(x, dim)
    
    if renorm:
        # Force normalization and float32 (includes NaN/Inf guard)
        return l2_normalize(x)
    
    if check_unit_norm:
        # Verify unit norm with improved diagnostics
        if isinstance(x, np.ndarray):
            norms = np.linalg.norm(x, axis=-1)
            deviations = np.abs(norms - 1.0)
            if not np.allclose(norms, 1.0, atol=tol):
                max_dev = deviations.max()
                mean_dev = deviations.mean()
                raise AssertionError(
                    f"NumPy: Vectors are not unit norm "
                    f"(max deviation: {max_dev:.6f}, mean deviation: {mean_dev:.6f})"
                )
        elif torch.is_tensor(x):
            norms = x.norm(p=2, dim=-1)
            if not torch.allclose(norms, torch.ones_like(norms), atol=tol):
                max_dev = (norms - 1.0).abs().max().item()
                raise AssertionError(
                    f"Vectors are not unit norm (max deviation: {max_dev:.6f})"
                )
    
    return x
