"""Public FPS API: validation, start resolution and backend dispatch.

Backends: CUDA non-float64 inputs use the Triton kernels. CPU inputs (any
supported dtype, float64 included) use the C++ kernels, compiled just-in-time
on first use (_cpu.py), for FPS and FPS + kNN; nearest-centroid assignment
and `nearest_assign` have no C++ kernel and run the reference. Everything
else (MPS, CUDA float64, CUDA without Triton, CPU without a compiler) uses the
pure-PyTorch reference.

Shared semantics: validity = `valid_mask` AND finite coordinates; distances
accumulate in float32 (float64 for double inputs); ties pick the lowest
index; rows that run out of valid points repeat the last selection.
"""
from __future__ import annotations

import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor

from . import _cpu, _reference

_triton_state: Optional[bool] = None


def _use_triton(points: Tensor) -> bool:
    global _triton_state
    if not points.is_cuda or points.dtype == torch.float64:
        return False
    if _triton_state is None:
        try:
            from . import _triton  # noqa: F401
            _triton_state = True
        except ImportError as exc:
            _triton_state = False
            warnings.warn(f"torch_fps: Triton unavailable ({exc}); using the "
                          "slower pure-PyTorch backend on CUDA.", RuntimeWarning)
    return _triton_state


def _dispatch(points: Tensor, mask: Tensor, start_idx: Tensor, k: int,
              k_neighbors: Optional[int]):
    if _use_triton(points):
        from . import _triton
        if k_neighbors is None:
            return _triton.fps(points, mask, start_idx, k)
        return _triton.fps_knn(points, mask, start_idx, k, k_neighbors)
    if points.device.type == "cpu":
        mod = _cpu.get_module()
        if mod is not None:
            if k_neighbors is None:
                return mod.fps_forward(points, mask, start_idx, k)
            return mod.fps_with_knn_forward(points, mask, start_idx, k, k_neighbors)
    if k_neighbors is None:
        return _reference.fps_reference(points, mask, start_idx, k)
    return _reference.fps_knn_reference(points, mask, start_idx, k, k_neighbors)


def _resolve_start_idx(valid: Tensor, counts: Optional[Tensor], k: int,
                       start_idx: Optional[Tensor], random_start: bool,
                       generator: Optional[torch.Generator],
                       validate: bool) -> Tensor:
    """Return a contiguous [B] long start index; syncs only when validating."""
    B, N = valid.shape
    if start_idx is None:
        if validate and bool((counts < k).any()):
            raise ValueError("FPS requires K <= number of valid points. Found "
                             f"batch(es) with K={k} but fewer valid points.")
        if not random_start:
            return valid.long().argmax(dim=1).contiguous()
        # masked-random argmax: uniform over valid points (all-invalid rows -> 0)
        scores = torch.rand(B, N, device=valid.device, generator=generator)
        return torch.where(valid, scores, float("-inf")).argmax(dim=1).contiguous()

    if start_idx.device != valid.device:
        raise ValueError("start_idx must be on the same device as points (a "
                         "cross-device copy here would silently synchronize the host)")
    start_idx = start_idx.to(dtype=torch.long)
    if start_idx.numel() != B:
        raise ValueError("start_idx must have shape [B]")
    start_idx = start_idx.reshape(B)
    if validate:
        insufficient = counts < k
        if bool((insufficient | (start_idx < 0) | (start_idx >= N)).any()):
            if bool(insufficient.any()):
                raise ValueError("FPS requires K <= number of valid points. Found "
                                 f"batch(es) with K={k} but fewer valid points.")
            raise ValueError("start_idx values must be within [0, N)")
    # a start on an invalid or out-of-range slot is repaired to the row's
    # first valid index
    safe = start_idx.clamp(0, max(N - 1, 0))
    ok = ((valid.gather(1, safe.unsqueeze(-1)).squeeze(-1) & (safe == start_idx))
          | (counts == 0))
    return torch.where(ok, start_idx, valid.long().argmax(dim=1)).contiguous()


def _prepare_and_resolve(points: Tensor, valid_mask: Tensor, k: int,
                         start_idx: Optional[Tensor], random_start: bool,
                         generator: Optional[torch.Generator],
                         precision: Optional[torch.dtype], validate: bool,
                         assume_finite: bool) -> Tuple[Tensor, Tensor, Tensor]:
    if points.dim() != 3:
        raise ValueError("points tensor must have shape [B, N, D]")
    if valid_mask.dim() != 2:
        raise ValueError("valid_mask tensor must have shape [B, N]")
    if points.shape[:2] != valid_mask.shape:
        raise ValueError("points and valid_mask must agree on batch & point dims")
    if k < 0:
        raise ValueError("K must be non-negative")
    dtype = torch.float32 if precision is None else precision
    if points.device.type == "cpu" and dtype == torch.bfloat16:
        raise ValueError("bfloat16 is not supported on CPU (use float16, "
                         "float32, or float64)")
    points = points.to(dtype=dtype).contiguous()
    mask = valid_mask.to(device=points.device, dtype=torch.bool).contiguous()
    B, N, _ = points.shape
    if k == 0:
        return points, mask, torch.zeros(B, dtype=torch.long, device=points.device)
    if N == 0:
        raise ValueError("FPS with K > 0 requires at least one point (got N=0)")
    valid = mask if assume_finite else (mask & points.isfinite().all(dim=-1))
    counts = (valid.sum(dim=1, dtype=torch.long)
              if (validate or start_idx is not None) else None)
    start = _resolve_start_idx(valid, counts, k, start_idx, random_start,
                               generator, validate)
    return points, mask, start


def farthest_point_sampling(
    points: Tensor,
    valid_mask: Tensor,
    k: int,
    *,
    start_idx: Optional[Tensor] = None,
    random_start: bool = True,
    generator: Optional[torch.Generator] = None,
    precision: Optional[torch.dtype] = None,
    validate: bool = True,
    assume_finite: bool = False,
) -> Tensor:
    """Select `k` maximally spread-out points per batch row.

    Args:
        points: Float tensor `[B, N, D]`.
        valid_mask: Bool tensor `[B, N]`; False marks padded/invalid points.
        k: Samples per row; must be <= the row's valid point count.
        start_idx: Optional `[B]` long first index per row (same device as
            `points`); an invalid slot is repaired to the first valid index.
        random_start: Draw a random valid first index when `start_idx` is None;
            otherwise start at the first valid index.
        generator: Optional generator for the random start.
        precision: Compute dtype (default float32); bfloat16 is GPU-only.
        validate: Check `k <= valid count` (one host sync). With False the
            output is padded with repeated indices instead of raising.
        assume_finite: Skip the finiteness pass; caller guarantees finite
            coordinates on mask-true points.

    Returns:
        Long tensor `[B, k]` of selected point indices.
    """
    pts, mask, start = _prepare_and_resolve(
        points, valid_mask, k, start_idx, random_start, generator,
        precision, validate, assume_finite)
    if k == 0:
        return torch.zeros((pts.shape[0], 0), device=pts.device, dtype=torch.long)
    return _dispatch(pts, mask, start, k, None)


def farthest_point_sampling_with_knn(
    points: Tensor,
    valid_mask: Tensor,
    k: int,
    k_neighbors: int,
    *,
    start_idx: Optional[Tensor] = None,
    random_start: bool = True,
    generator: Optional[torch.Generator] = None,
    precision: Optional[torch.dtype] = None,
    validate: bool = True,
    assume_finite: bool = False,
) -> Tuple[Tensor, Tensor]:
    """FPS plus each centroid's `k_neighbors` nearest valid points.

    Neighbours are closest-first and include the centroid itself; rows with
    fewer valid points pad with the centroid index. `0 < k_neighbors <= N`.
    Other arguments as in :func:`farthest_point_sampling`.

    Returns:
        `[B, k]` centroid indices and `[B, k, k_neighbors]` neighbour indices.
    """
    if k_neighbors <= 0:
        raise ValueError("k_neighbors must be positive")
    pts, mask, start = _prepare_and_resolve(
        points, valid_mask, k, start_idx, random_start, generator,
        precision, validate, assume_finite)
    B, N, _ = pts.shape
    if k == 0:
        return (torch.zeros((B, 0), device=pts.device, dtype=torch.long),
                torch.zeros((B, 0, k_neighbors), device=pts.device, dtype=torch.long))
    if k_neighbors > N:
        raise ValueError(f"k_neighbors ({k_neighbors}) must be <= N ({N})")
    return _dispatch(pts, mask, start, k, k_neighbors)


def farthest_point_sampling_with_assign(
    points: Tensor,
    valid_mask: Tensor,
    k: int,
    *,
    start_idx: Optional[Tensor] = None,
    random_start: bool = True,
    generator: Optional[torch.Generator] = None,
    precision: Optional[torch.dtype] = None,
    validate: bool = True,
    assume_finite: bool = False,
) -> Tuple[Tensor, Tensor]:
    """FPS plus nearest-centroid (Voronoi) assignment of every point.

    Ties assign to the lowest centroid position. Other arguments as in
    :func:`farthest_point_sampling`.

    Returns:
        `[B, k]` centroid indices and `[B, N]` assignments in `[0, k)` (an
        index into the centroids); values at invalid lanes are unspecified.
    """
    pts, mask, start = _prepare_and_resolve(
        points, valid_mask, k, start_idx, random_start, generator,
        precision, validate, assume_finite)
    B, N, D = pts.shape
    if k == 0:
        return (torch.zeros((B, 0), device=pts.device, dtype=torch.long),
                torch.zeros((B, N), device=pts.device, dtype=torch.long))
    if _use_triton(pts):
        from . import _triton
        if N <= _triton._single_tile_cap(_triton.SINGLE_TILE_MAX_N, D):
            return _triton.fps_assign(pts, mask, start, k)
        if k <= 512:
            # large N: tiled FPS + streaming assign, never materialises [B, N, k]
            idx = _triton.fps(pts, mask, start, k)
            cents = torch.gather(pts, 1, idx.unsqueeze(-1).expand(-1, -1, D))
            starts = torch.arange(B, device=pts.device, dtype=torch.long) * N
            counts = torch.full((B,), N, device=pts.device, dtype=torch.long)
            assign = _triton.nearest_assign(pts.reshape(B * N, D),
                                            cents.reshape(B * k, D),
                                            starts, counts, N, k)
            return idx, assign.view(B, N)
    # composed route (CPU: C++ FPS when available, reference assignment)
    idx = _dispatch(pts, mask, start, k, None)
    return idx, _reference.assign_reference(pts, idx)


def nearest_assign(
    p_flat: Tensor,
    cents: Tensor,
    starts: Tensor,
    counts: Tensor,
    n_max: int,
    k: int,
) -> Tensor:
    """Assign batch-segmented flat points to their nearest centroid.

    Deterministic (lowest centroid index on ties) and gradient-free.

    Args:
        p_flat: Float32 `[N, D]` points, rows grouped per event in order.
        cents: Float32 `[B * k, D]` centroids.
        starts: Int64 `[B]` offset of each event in `p_flat`.
        counts: Int64 `[B]` points per event.
        n_max: Largest value in `counts`, as a host int.
        k: Centroids per event.

    Returns:
        Int64 `[N]` cell indices in `[0, k)`.
    """
    p, c = p_flat.detach(), cents.detach()
    if p.numel() == 0 or starts.numel() == 0:
        return torch.empty(p.shape[0], device=p.device, dtype=torch.long)
    if _use_triton(p) and k <= 512:
        from . import _triton
        return _triton.nearest_assign(p, c, starts, counts, n_max, k)
    return _reference.nearest_assign_reference(p, c, starts, counts, k)
