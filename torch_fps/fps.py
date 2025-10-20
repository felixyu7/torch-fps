from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

try:
    from . import _C  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - extension absent in pure-Python env
    _C = None

from ._baseline import batched_fps_baseline


def farthest_point_sampling(
    points: Tensor,
    valid_mask: Tensor,
    K: int,
    *,
    start_idx: Optional[Tensor] = None,
    random_start: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Farthest point sampling with native CPU/CUDA acceleration.

    Args:
        points:
            Float tensor with shape `[B, N, D]` (batch, points, features).
        valid_mask:
            Bool tensor with shape `[B, N]`; False marks padded / invalid points.
        K:
            Integer number of samples to draw per batch element (`0 <= K <= N`).
        start_idx:
            Optional `[B]` long tensor providing the first index per batch.
        random_start:
            If `True` (default) and `start_idx` is not supplied, draw a random
            first index whenever a batch has more than `K` valid points.
        generator:
            Optional `torch.Generator` used for deterministic random starts.

    Returns:
        idx:
            Long tensor `[B, K]` with the selected point indices.
        validK:
            Bool tensor `[B, K]` indicating which positions are real selections.
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
        return (
            torch.zeros((B, 0), device=device, dtype=torch.long),
            torch.zeros((B, 0), device=device, dtype=torch.bool),
        )

    device = points.device
    dtype = points.dtype

    if valid_mask.device != device:
        valid_mask = valid_mask.to(device)
    valid_mask = valid_mask.to(dtype=torch.bool)

    B, N, _ = points.shape
    points_c = points.contiguous()
    mask_c = valid_mask.contiguous()

    counts = mask_c.sum(dim=1, dtype=torch.long)
    K_tensor = torch.as_tensor(K, device=device, dtype=counts.dtype)
    K_eff = torch.minimum(counts, K_tensor)
    validK = torch.arange(K, device=device)[None, :] < K_eff[:, None]

    if start_idx is not None:
        start_idx = start_idx.to(device=device, dtype=torch.long)
        if start_idx.numel() != B:
            raise ValueError("start_idx must have shape [B]")
    else:
        if random_start:
            rand = torch.rand(
                B, device=device, dtype=dtype, generator=generator
            )
            counts_float = counts.to(dtype=dtype)
            first = torch.floor(rand * counts_float.clamp(min=1)).to(torch.long)
            first = first.masked_fill(counts == 0, 0)
            large_batches = counts > K_tensor
            start_idx = torch.where(
                large_batches, first, torch.zeros_like(first)
            )
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

    try:
        if _C is None:
            raise RuntimeError("torch-fps extension not built")
        idx = _C.fps_forward(points_c, mask_c, start_idx, K)
    except RuntimeError as exc:
        message = str(exc)
        if _C is None or (points_c.is_cuda() and "built without CUDA" in message):
            # validK from the accelerated path remains correct
            idx, _ = batched_fps_baseline(points_c, mask_c, K, start_idx)
            return idx, validK
        raise

    return idx, validK
