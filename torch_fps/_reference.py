"""Pure-PyTorch backend: any-device fallback and test oracle.

Takes contiguous inputs and a resolved [B] start index (see api.py). Same
semantics as the Triton kernels; vectorised over the batch, with a Python
loop only over the sequential K iterations. No host syncs.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def _acc(points: Tensor) -> Tensor:
    return points if points.dtype == torch.float64 else points.float()


def _fps_state(points: Tensor, mask: Tensor, start_idx: Tensor):
    n = points.shape[1]
    pts = _acc(points)
    valid = mask & points.isfinite().all(dim=-1)
    min_d = torch.where(valid, float("inf"), -float("inf")).to(pts.dtype)
    in_range = (start_idx >= 0) & (start_idx < n)
    last = torch.where(in_range, start_idx, torch.zeros_like(start_idx))
    return pts, valid, min_d, last


def fps_reference(points: Tensor, mask: Tensor, start_idx: Tensor, k: int) -> Tensor:
    B = points.shape[0]
    idx = torch.empty(B, k, device=points.device, dtype=torch.long)
    pts, valid, min_d, last = _fps_state(points, mask, start_idx)
    rows = torch.arange(B, device=points.device)
    for i in range(k):
        idx[:, i] = last
        min_d[rows, last] = -float("inf")
        if i + 1 == k:
            break
        c = pts[rows, last]
        d = (pts - c[:, None, :]).square().sum(dim=2)
        # invalid lanes keep -inf; NaN distances never overwrite
        min_d = torch.where(valid & (d < min_d), d, min_d)
        vals, nxt = min_d.max(dim=1)
        # exhausted rows (all -inf) repeat the previous selection
        last = torch.where(vals.isneginf(), last, nxt)
    return idx


def assign_reference(points: Tensor, idx: Tensor) -> Tensor:
    """[B, N] nearest centroid position in [0, K); invalid lanes unspecified.

    Accumulates per dim (no [B, N, K, D] broadcast); NaN distances become +inf.
    """
    B, N, D = points.shape
    pts = _acc(points)
    cents = torch.gather(pts, 1, idx.unsqueeze(-1).expand(-1, -1, D))
    d = pts.new_zeros(B, N, idx.size(1))
    for dim in range(D):
        diff = pts[:, :, dim].unsqueeze(-1) - cents[:, :, dim].unsqueeze(1)
        d = d + diff * diff
    d = torch.where(d == d, d, float("inf"))
    return d.argmin(dim=-1)


def fps_knn_reference(points: Tensor, mask: Tensor, start_idx: Tensor,
                      k: int, k_neighbors: int) -> Tuple[Tensor, Tensor]:
    D = points.shape[2]
    centroid_idx = fps_reference(points, mask, start_idx, k)
    pts = _acc(points)
    valid = mask & points.isfinite().all(dim=-1)
    cents = torch.gather(pts, 1, centroid_idx.unsqueeze(-1).expand(-1, -1, D))
    d = (cents.unsqueeze(2) - pts.unsqueeze(1)).square().sum(dim=-1)  # [B, K, N]
    d = torch.where(valid.unsqueeze(1), d, float("inf"))
    # stable sort: lowest index first among exact ties
    svals, sidx = d.sort(dim=-1, stable=True)
    svals, sidx = svals[..., :k_neighbors], sidx[..., :k_neighbors]
    # slots past the row's valid count are +inf: pad with the centroid
    return centroid_idx, torch.where(svals.isinf(), centroid_idx.unsqueeze(-1), sidx)


def nearest_assign_reference(p_flat: Tensor, cents: Tensor, starts: Tensor,
                             counts: Tensor, k: int) -> Tensor:
    n, D = p_flat.shape
    B = starts.numel()
    seg = torch.repeat_interleave(
        torch.arange(B, device=p_flat.device), counts, output_size=n)
    cents_per_point = cents.view(B, k, D).index_select(0, seg)
    pts = p_flat.float()
    d = pts.new_zeros(n, k)
    for dim in range(D):
        diff = pts[:, dim].unsqueeze(-1) - cents_per_point[:, :, dim]
        d = d + diff * diff
    d = torch.where(d == d, d, float("inf"))
    return d.argmin(dim=1)
