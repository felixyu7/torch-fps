from __future__ import annotations

import torch
from torch import Tensor


def batched_fps_baseline(
    points: Tensor,
    valid_mask: Tensor,
    K: int,
    start_idx: Tensor,
) -> Tensor:
    """
    Baseline batched farthest point sampling using vectorized PyTorch ops.
    Mirrors the original Python implementation but with an explicit starting
    index tensor per batch to keep seed control in the caller.

    Returns:
        idx: Long tensor `[B, K]` with the selected point indices.
    """
    if points.numel() == 0:
        B = points.shape[0] if points.dim() > 0 else 0
        return torch.zeros((B, 0), device=points.device, dtype=torch.long)

    B, N, _ = points.shape
    device = points.device
    dtype = points.dtype

    counts = valid_mask.sum(dim=1)

    # Caller should ensure K <= counts for all batches
    if K > 0:
        insufficient = counts < K
        if bool(insufficient.any()):
            raise ValueError(
                f"Baseline FPS requires K <= number of valid points. "
                f"Found batch(es) with K={K} but fewer valid points."
            )

    start_idx = start_idx.clone()
    start_idx = start_idx.masked_fill(counts == 0, 0)
    invalid_range = (start_idx < 0) | (start_idx >= N)
    if bool(invalid_range.any()):
        raise ValueError("start_idx values must be within [0, N) for baseline FPS")

    has_valid_points = counts > 0
    if bool(has_valid_points.any()):
        current_valid = valid_mask[has_valid_points, start_idx[has_valid_points]]
        if bool((~current_valid).any()):
            first_valid = torch.argmax(valid_mask.long(), dim=1)
            replacement = first_valid[has_valid_points]
            start_idx[has_valid_points] = torch.where(
                current_valid,
                start_idx[has_valid_points],
                replacement,
            )

    idx = torch.zeros((B, K), device=device, dtype=torch.long)

    inf = torch.tensor(float("inf"), device=device, dtype=dtype)
    min_dists = torch.full((B, N), inf, device=device, dtype=dtype)
    min_dists.masked_fill_(~valid_mask, 0.0)

    last = start_idx
    batch_indices = torch.arange(B, device=device)

    for i in range(K):
        idx[:, i] = last

        if i + 1 >= K:
            continue

        c = points[batch_indices, last]
        d = (points - c[:, None, :]).square().sum(dim=2)
        d = d.masked_fill(~valid_mask, inf)
        min_dists = torch.minimum(min_dists, d)
        last = torch.argmax(min_dists, dim=1)

    return idx
