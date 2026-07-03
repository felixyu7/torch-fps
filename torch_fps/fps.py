from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

_C = None
_import_error: Optional[BaseException] = None
try:
    from . import _C  # type: ignore[attr-defined]
except ImportError as _exc:  # pragma: no cover - extension absent or ABI mismatch
    _import_error = _exc


def _require_extension() -> None:
    if _C is None:
        raise RuntimeError(
            "torch-fps extension not available. "
            "Rebuild with: python setup.py build_ext --inplace"
        ) from _import_error


def _resolve_start_idx(
    points_c: Tensor,
    mask_c: Tensor,
    counts: Optional[Tensor],
    B: int,
    N: int,
    K: int,
    device: torch.device,
    start_idx: Optional[Tensor],
    random_start: bool,
    generator: Optional[torch.Generator],
    validate: bool = True,
) -> Tensor:
    """Produce a validated contiguous [B] long start_idx with minimal GPU syncs.

    Start selection honors the kernels' validity semantics: a point is
    selectable only if it is mask-true AND has no NaN coordinate. The fused
    mask costs one elementwise [B, N, D] pass — negligible next to the
    O(K*N*D) kernel. On the user-supplied path we fuse the K<=counts, range,
    and mask-validity checks into a single `.any()` sync. With
    ``validate=False`` the checks (and their host sync) are skipped entirely;
    ``counts`` may then be None on the ``start_idx is None`` path. All start
    resolution is pure-tensor: no host-device sync beyond validation.
    """
    valid = mask_c & ~points_c.isnan().any(dim=-1)

    if start_idx is None:
        if validate:
            problems = counts < K
            if bool(problems.any()):
                raise ValueError(
                    f"FPS requires K <= number of valid points. "
                    f"Found batch(es) with K={K} but fewer valid points."
                )
        if not random_start:
            # Deterministic start: first valid index per row (all-invalid rows
            # give 0; the kernel pads those rows anyway).
            return valid.long().argmax(dim=1).contiguous()
        # Random start: masked-random argmax draws uniformly over valid points
        # without multinomial's fp32 prob copy. All-invalid rows argmax to 0,
        # matching the kernel's documented pad behavior.
        scores = torch.rand(B, N, device=device, generator=generator)
        scores = torch.where(valid, scores, float("-inf"))
        return scores.argmax(dim=1).contiguous()

    # User-supplied path. Fuse all validation into a single sync. Bad
    # inputs (K>counts, out-of-range) raise; a start_idx that points to an
    # invalid slot (masked out or NaN) is silently repaired to the first
    # valid index in that row, matching the Python baseline's behavior.
    if start_idx.device != device:
        raise ValueError(
            "start_idx must be on the same device as points (a cross-device "
            "copy here would silently synchronize the host)"
        )
    start_idx = start_idx.to(dtype=torch.long)
    if start_idx.numel() != B:
        raise ValueError("start_idx must have shape [B]")
    start_idx = start_idx.reshape(B)

    if validate:
        out_of_range = (start_idx < 0) | (start_idx >= N)
        insufficient = counts < K
        problems = out_of_range | insufficient
        if bool(problems.any()):
            if bool(insufficient.any()):
                raise ValueError(
                    f"FPS requires K <= number of valid points. "
                    f"Found batch(es) with K={K} but fewer valid points."
                )
            raise ValueError("start_idx values must be within [0, N)")

    # Repair any start index that points to an invalid slot. Avoid the
    # extra sync on the common "already valid" path by skipping the repair
    # when counts>0 implies at least one valid index exists. The repair is
    # pure-tensor: no sync.
    has_valid = counts > 0
    safe_start = start_idx.clamp(0, max(N - 1, 0))
    supplied_valid = valid.gather(1, safe_start.unsqueeze(-1)).squeeze(-1)
    first_valid = torch.argmax(valid.long(), dim=1)
    repaired = torch.where(supplied_valid | ~has_valid, start_idx, first_valid)
    return repaired.contiguous()


def farthest_point_sampling(
    points: Tensor,
    valid_mask: Tensor,
    K: int,
    *,
    start_idx: Optional[Tensor] = None,
    random_start: bool = True,
    generator: Optional[torch.Generator] = None,
    precision: Optional[torch.dtype] = None,
    validate: bool = True,
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
            Must live on the same device as `points`; an index pointing at an
            invalid slot (masked out or NaN) is repaired to the row's first
            valid index.
        random_start:
            If `True` (default) and `start_idx` is not supplied, draw a random
            first index from valid points (mask-true and NaN-free). If `False`
            and `start_idx` is not supplied, the row's first valid index is
            used.
        generator:
            Optional `torch.Generator` used for deterministic random starts.
        precision:
            Optional dtype for internal computations. If None (default), uses float32 on all
            devices for numerical stability. Can override: float16, float32, float64 (CPU/GPU)
            or bfloat16 (GPU only).
        validate:
            If `True` (default), verify `K <= valid count` per batch row (and
            range-check a user-supplied `start_idx`). The check costs one
            host-device sync per call; callers that already guarantee the
            precondition (e.g. by construction of their batches) can pass
            `False` to keep the call fully asynchronous. With `validate=False`
            a violated precondition is NOT diagnosed: the kernel pads the
            output with repeated indices instead of raising.

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

    # Determine precision: default to float32 on all devices for stability
    if precision is None:
        if points.dtype != torch.float32:
            points = points.to(dtype=torch.float32)
    else:
        # Validate precision
        if device.type == 'cpu' and precision == torch.bfloat16:
            raise ValueError("bfloat16 is not supported on CPU (use float16, float32, or float64)")
        # Convert points to specified precision
        if points.dtype != precision:
            points = points.to(dtype=precision)

    if valid_mask.device != device:
        valid_mask = valid_mask.to(device)
    valid_mask = valid_mask.to(dtype=torch.bool)

    B, N, _ = points.shape
    points_c = points.contiguous()
    mask_c = valid_mask.contiguous()

    # counts feeds validation and the user-supplied start repair; skip the
    # kernel entirely when neither needs it (validate=False, start_idx=None).
    counts = (
        mask_c.sum(dim=1, dtype=torch.long)
        if (validate or start_idx is not None) else None
    )

    start_idx = _resolve_start_idx(
        points_c, mask_c, counts, B, N, K, device,
        start_idx, random_start, generator, validate,
    )

    _require_extension()
    return _C.fps_forward(points_c, mask_c, start_idx, K)


def farthest_point_sampling_with_knn(
    points: Tensor,
    valid_mask: Tensor,
    K: int,
    k_neighbors: int,
    *,
    start_idx: Optional[Tensor] = None,
    random_start: bool = True,
    generator: Optional[torch.Generator] = None,
    precision: Optional[torch.dtype] = None,
    validate: bool = True,
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
            Must live on the same device as `points`; an index pointing at an
            invalid slot (masked out or NaN) is repaired to the row's first
            valid index.
        random_start:
            If `True` (default) and `start_idx` is not supplied, draw a random
            first index from valid points (mask-true and NaN-free). If `False`
            and `start_idx` is not supplied, the row's first valid index is
            used.
        generator:
            Optional `torch.Generator` used for deterministic random starts.
        precision:
            Optional dtype for internal computations. If None (default), uses float32 on all
            devices for numerical stability. Can override: float16, float32, float64 (CPU/GPU)
            or bfloat16 (GPU only).
        validate:
            If `True` (default), verify `K <= valid count` per batch row (and
            range-check a user-supplied `start_idx`) at the cost of one
            host-device sync per call. Pass `False` when the caller guarantees
            the precondition to keep the call fully asynchronous; violations
            are then NOT diagnosed (the kernel pads with repeated indices).

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

    # Determine precision: default to float32 on all devices for stability
    if precision is None:
        if points.dtype != torch.float32:
            points = points.to(dtype=torch.float32)
    else:
        # Validate precision
        if device.type == 'cpu' and precision == torch.bfloat16:
            raise ValueError("bfloat16 is not supported on CPU (use float16, float32, or float64)")
        # Convert points to specified precision
        if points.dtype != precision:
            points = points.to(dtype=precision)

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

    counts = (
        mask_c.sum(dim=1, dtype=torch.long)
        if (validate or start_idx is not None) else None
    )

    start_idx = _resolve_start_idx(
        points_c, mask_c, counts, B, N, K, device,
        start_idx, random_start, generator, validate,
    )

    _require_extension()
    return _C.fps_with_knn_forward(points_c, mask_c, start_idx, K, k_neighbors)
