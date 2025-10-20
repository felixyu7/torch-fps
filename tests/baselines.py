"""
Pure PyTorch baseline implementations for FPS and FPS+kNN.

These serve as reference implementations for correctness testing.
"""

import torch
from torch import Tensor


def fps_baseline(
    points: Tensor,
    valid_mask: Tensor,
    K: int,
    start_idx: Tensor,
) -> Tensor:
    """
    Baseline batched farthest point sampling using vectorized PyTorch ops.

    Args:
        points: [B, N, D] float tensor
        valid_mask: [B, N] bool tensor
        K: Number of samples to select
        start_idx: [B] long tensor with starting indices

    Returns:
        idx: [B, K] long tensor with selected point indices
    """
    if points.numel() == 0:
        B = points.shape[0] if points.dim() > 0 else 0
        return torch.zeros((B, 0), device=points.device, dtype=torch.long)

    B, N, _ = points.shape
    device = points.device
    dtype = points.dtype

    counts = valid_mask.sum(dim=1)

    if K > 0:
        insufficient = counts < K
        if bool(insufficient.any()):
            raise ValueError(
                f"FPS requires K <= number of valid points. "
                f"Found batch(es) with K={K} but fewer valid points."
            )

    start_idx = start_idx.clone()
    start_idx = start_idx.masked_fill(counts == 0, 0)
    invalid_range = (start_idx < 0) | (start_idx >= N)
    if bool(invalid_range.any()):
        raise ValueError("start_idx values must be within [0, N)")

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
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=dtype)
    min_dists = torch.full((B, N), inf, device=device, dtype=dtype)
    min_dists.masked_fill_(~valid_mask, neg_inf)

    last = start_idx
    batch_indices = torch.arange(B, device=device)

    for i in range(K):
        idx[:, i] = last
        min_dists[batch_indices, last] = neg_inf

        if i + 1 >= K:
            continue

        c = points[batch_indices, last]
        d = (points - c[:, None, :]).square().sum(dim=2)
        d = d.masked_fill(~valid_mask, inf)
        min_dists = torch.minimum(min_dists, d)
        last = torch.argmax(min_dists, dim=1)

    return idx


def fps_with_knn_baseline(
    points: Tensor,
    valid_mask: Tensor,
    K: int,
    k_neighbors: int,
    start_idx: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Baseline implementation: separate FPS + kNN using PyTorch ops.

    This is the original two-stage approach that computes distances twice.

    Args:
        points: [B, N, D] float tensor
        valid_mask: [B, N] bool tensor
        K: Number of FPS centroids to select
        k_neighbors: Number of nearest neighbors per centroid
        start_idx: [B] long tensor with starting indices

    Returns:
        centroid_idx: [B, K] long tensor with FPS centroid indices
        neighbor_idx: [B, K, k_neighbors] long tensor with kNN indices
    """
    B, N, D = points.shape
    device = points.device

    # Step 1: FPS
    centroid_idx = fps_baseline(points, valid_mask, K, start_idx)

    # Step 2: Gather centroids
    centroids = torch.gather(
        points, 1, centroid_idx.unsqueeze(-1).expand(-1, -1, D)
    )  # [B, K, D]

    # Step 3: Compute pairwise distances from centroids to all points
    diff = centroids.unsqueeze(2) - points.unsqueeze(1)  # [B, K, N, D]
    dists = diff.square().sum(dim=-1)  # [B, K, N]

    # Mask invalid points with inf
    dists = dists.masked_fill(~valid_mask.unsqueeze(1), float('inf'))

    # Step 4: Find k nearest neighbors
    _, neighbor_idx = torch.topk(dists, k_neighbors, dim=-1, largest=False, sorted=True)

    return centroid_idx, neighbor_idx
