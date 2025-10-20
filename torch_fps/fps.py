from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

try:
    from . import _C  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - extension absent in pure-Python env
    _C = None


def farthest_point_sampling(
    points: Tensor,
    valid_mask: Tensor,
    K: int,
    *,
    start_idx: Optional[Tensor] = None,
    random_start: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """
    Farthest point sampling with native CPU/CUDA acceleration.

    Args:
        points:
            Float tensor with shape `[B, N, D]` (batch, points, features).
        valid_mask:
            Bool tensor with shape `[B, N]`; False marks padded / invalid points.
        K:
            Integer number of samples to draw per batch element.
            Must satisfy `K <= number of valid points` for all batches.
        start_idx:
            Optional `[B]` long tensor providing the first index per batch.
        random_start:
            If `True` (default) and `start_idx` is not supplied, draw a random
            first index from valid points.
        generator:
            Optional `torch.Generator` used for deterministic random starts.

    Returns:
        idx:
            Long tensor `[B, K]` with the selected point indices.
    """
    if points.dim() != 3:
        raise ValueError("points tensor must have shape [B, N, D]")
    if valid_mask.dim() != 2:
        raise ValueError("valid_mask tensor must have shape [B, N]")
    if points.shape[:2] != valid_mask.shape:
        raise ValueError("points and valid_mask must agree on batch & point dims")
    if K < 0:
        raise ValueError("K must be non-negative")

    if K == 0:
        B = points.shape[0]
        device = points.device
        return torch.zeros((B, 0), device=device, dtype=torch.long)

    device = points.device
    dtype = points.dtype

    if valid_mask.device != device:
        valid_mask = valid_mask.to(device)
    valid_mask = valid_mask.to(dtype=torch.bool)

    B, N, _ = points.shape
    points_c = points.contiguous()
    mask_c = valid_mask.contiguous()

    counts = mask_c.sum(dim=1, dtype=torch.long)

    # FPS is a downsampling algorithm - require K <= counts
    if K > 0:
        insufficient = counts < K
        if bool(insufficient.any()):
            raise ValueError(
                f"FPS requires K <= number of valid points. "
                f"Found batch(es) with K={K} but fewer valid points."
            )

    if start_idx is not None:
        start_idx = start_idx.to(device=device, dtype=torch.long)
        if start_idx.numel() != B:
            raise ValueError("start_idx must have shape [B]")
    else:
        if random_start:
            # Random initial point per batch (matches vectorized floor(rand * counts) approach)
            # Note: assumes valid points are densely packed at indices [0, ..., counts-1]
            if generator is not None:
                rand = torch.rand(B, device=device, dtype=dtype, generator=generator)
            else:
                rand = torch.rand(B, device=device, dtype=dtype)
            start_idx = torch.floor(rand * counts.to(dtype).clamp(min=1)).to(torch.long)
            start_idx = start_idx.masked_fill(counts == 0, 0)
        else:
            start_idx = torch.zeros(B, device=device, dtype=torch.long)

    start_idx = start_idx.masked_fill(counts == 0, 0)
    invalid_range = (start_idx < 0) | (start_idx >= N)
    if bool(invalid_range.any()):
        raise ValueError("start_idx values must be within [0, N)")

    has_valid_points = counts > 0
    if bool(has_valid_points.any()):
        current_valid = mask_c[has_valid_points, start_idx[has_valid_points]]
        if bool((~current_valid).any()):
            first_valid = torch.argmax(mask_c.long(), dim=1)
            replacement = first_valid[has_valid_points]
            start_idx = start_idx.clone()
            start_idx[has_valid_points] = torch.where(
                current_valid,
                start_idx[has_valid_points],
                replacement,
            )

    if bool(has_valid_points.any()):
        invalid_mask = ~mask_c[has_valid_points, start_idx[has_valid_points]]
        if bool(invalid_mask.any()):
            raise ValueError("start_idx must point to a valid point for batches with valid entries")

    start_idx = start_idx.contiguous()

    if _C is None:
        raise RuntimeError(
            "torch-fps extension not built. "
            "Please build the extension with: python setup.py build_ext --inplace"
        )

    idx = _C.fps_forward(points_c, mask_c, start_idx, K)
    return idx


def farthest_point_sampling_with_knn(
    points: Tensor,
    valid_mask: Tensor,
    K: int,
    k_neighbors: int,
    *,
    start_idx: Optional[Tensor] = None,
    random_start: bool = True,
    generator: Optional[torch.Generator] = None,
) -> tuple[Tensor, Tensor]:
    """
    Fused farthest point sampling + k-nearest neighbors with native CPU/CUDA acceleration.

    This function performs FPS and kNN in a single fused kernel, avoiding redundant
    distance computations. The distances computed during FPS are reused to find
    k nearest neighbors for each selected centroid.

    Args:
        points:
            Float tensor with shape `[B, N, D]` (batch, points, features).
        valid_mask:
            Bool tensor with shape `[B, N]`; False marks padded / invalid points.
        K:
            Integer number of FPS samples (centroids) to draw per batch element.
            Must satisfy `K <= number of valid points` for all batches.
        k_neighbors:
            Integer number of nearest neighbors to find for each centroid.
            Must satisfy `k_neighbors <= N`.
        start_idx:
            Optional `[B]` long tensor providing the first index per batch.
        random_start:
            If `True` (default) and `start_idx` is not supplied, draw a random
            first index from valid points.
        generator:
            Optional `torch.Generator` used for deterministic random starts.

    Returns:
        centroid_idx:
            Long tensor `[B, K]` with the selected FPS centroid indices.
        neighbor_idx:
            Long tensor `[B, K, k_neighbors]` with the k nearest neighbor indices
            for each centroid. Neighbors are sorted by distance (closest first).

    Example:
        >>> points = torch.randn(2, 100, 4, device='cuda')  # B=2, N=100, D=4
        >>> mask = torch.ones(2, 100, dtype=torch.bool, device='cuda')
        >>> centroid_idx, neighbor_idx = farthest_point_sampling_with_knn(
        ...     points, mask, K=16, k_neighbors=8
        ... )
        >>> centroid_idx.shape
        torch.Size([2, 16])
        >>> neighbor_idx.shape
        torch.Size([2, 16, 8])
    """
    if points.dim() != 3:
        raise ValueError("points tensor must have shape [B, N, D]")
    if valid_mask.dim() != 2:
        raise ValueError("valid_mask tensor must have shape [B, N]")
    if points.shape[:2] != valid_mask.shape:
        raise ValueError("points and valid_mask must agree on batch & point dims")
    if K < 0:
        raise ValueError("K must be non-negative")
    if k_neighbors <= 0:
        raise ValueError("k_neighbors must be positive")

    B, N, D = points.shape
    device = points.device
    dtype = points.dtype

    if K == 0:
        centroid_idx = torch.zeros((B, 0), device=device, dtype=torch.long)
        neighbor_idx = torch.zeros((B, 0, k_neighbors), device=device, dtype=torch.long)
        return centroid_idx, neighbor_idx

    if k_neighbors > N:
        raise ValueError(f"k_neighbors ({k_neighbors}) must be <= N ({N})")

    if valid_mask.device != device:
        valid_mask = valid_mask.to(device)
    valid_mask = valid_mask.to(dtype=torch.bool)

    points_c = points.contiguous()
    mask_c = valid_mask.contiguous()

    counts = mask_c.sum(dim=1, dtype=torch.long)

    # FPS is a downsampling algorithm - require K <= counts
    if K > 0:
        insufficient = counts < K
        if bool(insufficient.any()):
            raise ValueError(
                f"FPS requires K <= number of valid points. "
                f"Found batch(es) with K={K} but fewer valid points."
            )

    if start_idx is not None:
        start_idx = start_idx.to(device=device, dtype=torch.long)
        if start_idx.numel() != B:
            raise ValueError("start_idx must have shape [B]")
    else:
        if random_start:
            if generator is not None:
                rand = torch.rand(B, device=device, dtype=dtype, generator=generator)
            else:
                rand = torch.rand(B, device=device, dtype=dtype)
            start_idx = torch.floor(rand * counts.to(dtype).clamp(min=1)).to(torch.long)
            start_idx = start_idx.masked_fill(counts == 0, 0)
        else:
            start_idx = torch.zeros(B, device=device, dtype=torch.long)

    start_idx = start_idx.masked_fill(counts == 0, 0)
    invalid_range = (start_idx < 0) | (start_idx >= N)
    if bool(invalid_range.any()):
        raise ValueError("start_idx values must be within [0, N)")

    has_valid_points = counts > 0
    if bool(has_valid_points.any()):
        current_valid = mask_c[has_valid_points, start_idx[has_valid_points]]
        if bool((~current_valid).any()):
            first_valid = torch.argmax(mask_c.long(), dim=1)
            replacement = first_valid[has_valid_points]
            start_idx = start_idx.clone()
            start_idx[has_valid_points] = torch.where(
                current_valid,
                start_idx[has_valid_points],
                replacement,
            )

    if bool(has_valid_points.any()):
        invalid_mask = ~mask_c[has_valid_points, start_idx[has_valid_points]]
        if bool(invalid_mask.any()):
            raise ValueError("start_idx must point to a valid point for batches with valid entries")

    start_idx = start_idx.contiguous()

    if _C is None:
        raise RuntimeError(
            "torch-fps extension not built. "
            "Please build the extension with: python setup.py build_ext --inplace"
        )

    centroid_idx, neighbor_idx = _C.fps_with_knn_forward(
        points_c, mask_c, start_idx, K, k_neighbors
    )
    return centroid_idx, neighbor_idx
