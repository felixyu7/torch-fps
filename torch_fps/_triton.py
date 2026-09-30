"""Triton CUDA kernels for FPS, fused FPS+kNN/assign, and nearest-centroid assign.

One program per batch row; the K-loop runs inside the kernel (one launch, no
host syncs). N up to SINGLE_TILE_MAX_N stays register-resident; larger N
sweeps a global fp32 scratch in TILE_N tiles. Ties go to the lowest index.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

SINGLE_TILE_MAX_N = 16384       # register-resident cap for D<=4 (scaled by D above)
KNN_SINGLE_TILE_MAX_N = 8192    # fused kernel holds one extra vector
TILE_N = 4096                   # sweep width of the tiled kernels

_INF = tl.constexpr(float("inf"))


def _num_warps(block_n: int) -> int:
    if block_n <= 2048:
        return 4
    if block_n <= 8192:
        return 8
    return 16


def _single_tile_cap(cap: int, D: int) -> int:
    # register budget ~ (D+1) vectors of BLOCK_N floats; keep parity with D=4
    return cap if D <= 4 else max(1024, cap * 5 // (D + 1))


@triton.jit
def _load_coords(p_base, inb, valid, D: tl.constexpr):
    coords = ()
    for d in tl.static_range(D):
        v = tl.load(p_base + d, mask=inb, other=0.0).to(tl.float32)
        valid = valid & (tl.abs(v) < _INF)  # finite: excludes NaN and +/-inf
        coords = coords + (v,)
    return coords, valid


@triton.jit
def _dist_to(coords, sel, D: tl.constexpr):
    # centroid coords come from the register-resident tiles; accumulate over d
    dist = tl.zeros(sel.shape, dtype=tl.float32)
    for d in tl.static_range(D):
        c = tl.sum(tl.where(sel, coords[d], 0.0))
        diff = coords[d] - c
        dist += diff * diff
    return dist


@triton.jit
def _resolve_start(start_ptr, b, N):
    start = tl.load(start_ptr + b).to(tl.int32)
    return tl.where((start >= 0) & (start < N), start, 0)


@triton.jit
def _fps_single_kernel(points_ptr, mask_ptr, start_ptr, idx_ptr, assign_ptr,
                       N, K,
                       stride_pb, stride_pn, stride_mb, stride_ib,
                       D: tl.constexpr, BLOCK_N: tl.constexpr,
                       WITH_ASSIGN: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_N)
    inb = offs < N
    p_base = points_ptr + b * stride_pb + offs * stride_pn
    m = tl.load(mask_ptr + b * stride_mb + offs, mask=inb, other=0)
    coords, valid = _load_coords(p_base, inb, (m != 0) & inb, D)

    min_d = tl.where(valid, _INF, -_INF)
    last = _resolve_start(start_ptr, b, N)
    if WITH_ASSIGN:
        # running nearest-centroid: the K-loop computes every hit's distance
        # to each selected centroid anyway, so the Voronoi assignment is free
        best_d = tl.full((BLOCK_N,), _INF, dtype=tl.float32)
        best_i = tl.zeros((BLOCK_N,), dtype=tl.int32)

    for i in range(K):
        sel = offs == last
        dist = _dist_to(coords, sel, D)
        if WITH_ASSIGN:
            # raw dist, before selection-marking; strict < keeps the earliest
            # centroid on exact ties (= argmin first-occurrence), and NaN
            # dist never updates
            upd = valid & (dist < best_d)
            best_d = tl.where(upd, dist, best_d)
            best_i = tl.where(upd, i, best_i)
        # NaN dist and -inf (invalid/selected) lanes keep their old value
        min_d = tl.where(valid & (dist < min_d), dist, min_d)
        min_d = tl.where(sel, -_INF, min_d)
        tl.store(idx_ptr + b * stride_ib + i, last.to(tl.int64))
        bv, bi = tl.max(min_d, axis=0, return_indices=True,
                        return_indices_tie_break_left=True)
        # exhausted rows (all -inf) repeat the previous selection
        last = tl.where(bv == -_INF, last, bi)

    if WITH_ASSIGN:
        tl.store(assign_ptr + b * N + offs, best_i.to(tl.int64), mask=inb)


@triton.jit
def _fps_knn_single_kernel(points_ptr, mask_ptr, start_ptr, idx_ptr, nbr_ptr,
                           N, K, k_nbrs,
                           stride_pb, stride_pn, stride_mb,
                           stride_ib, stride_nb,
                           D: tl.constexpr, BLOCK_N: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_N)
    inb = offs < N
    p_base = points_ptr + b * stride_pb + offs * stride_pn
    m = tl.load(mask_ptr + b * stride_mb + offs, mask=inb, other=0)
    coords, valid = _load_coords(p_base, inb, (m != 0) & inb, D)

    min_d = tl.where(valid, _INF, -_INF)
    last = _resolve_start(start_ptr, b, N)

    for i in range(K):
        sel = offs == last
        dist = _dist_to(coords, sel, D)
        # neighbor candidates: all valid points (self and already-selected
        # included); NaN from a pathological NaN centroid promotes to +inf
        nd = tl.where(valid, dist, _INF)
        nd = tl.where(nd == nd, nd, _INF)
        min_d = tl.where(valid & (dist < min_d), dist, min_d)
        min_d = tl.where(sel, -_INF, min_d)
        tl.store(idx_ptr + b * stride_ib + i, last.to(tl.int64))

        for nn in range(k_nbrs):
            bv, bi = tl.min(nd, axis=0, return_indices=True,
                            return_indices_tie_break_left=True)
            # no valid candidate left -> pad with the centroid itself
            chosen = tl.where(bv == _INF, last, bi)
            tl.store(nbr_ptr + b * stride_nb + i * k_nbrs + nn,
                     chosen.to(tl.int64))
            nd = tl.where(offs == bi, _INF, nd)

        bv, bi = tl.max(min_d, axis=0, return_indices=True,
                        return_indices_tie_break_left=True)
        last = tl.where(bv == -_INF, last, bi)


@triton.jit
def _centroid_coords(points_ptr, b, last, stride_pb, stride_pn, D: tl.constexpr):
    cents = ()
    for d in tl.static_range(D):
        c = tl.load(points_ptr + b * stride_pb
                    + last.to(tl.int64) * stride_pn + d).to(tl.float32)
        cents = cents + (c,)
    return cents


@triton.jit
def _tile_valid_dist(points_ptr, mask_ptr, b, offs, inb, cents,
                     stride_pb, stride_pn, stride_mb, D: tl.constexpr):
    m = tl.load(mask_ptr + b * stride_mb + offs, mask=inb, other=0)
    valid = (m != 0) & inb
    dist = tl.zeros(offs.shape, dtype=tl.float32)
    for d in tl.static_range(D):
        v = tl.load(points_ptr + b * stride_pb + offs * stride_pn + d,
                    mask=inb, other=0.0).to(tl.float32)
        valid = valid & (tl.abs(v) < _INF)
        diff = v - cents[d]
        dist += diff * diff
    return valid, dist


@triton.jit
def _fps_tiled_kernel(points_ptr, mask_ptr, start_ptr, idx_ptr, min_ptr,
                      N, K,
                      stride_pb, stride_pn, stride_mb, stride_ib,
                      D: tl.constexpr, BLOCK_N: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    mrow = min_ptr + b * N
    n_tiles = tl.cdiv(N, BLOCK_N)

    for t in range(n_tiles):
        offs = t * BLOCK_N + tl.arange(0, BLOCK_N)
        inb = offs < N
        m = tl.load(mask_ptr + b * stride_mb + offs, mask=inb, other=0)
        valid = (m != 0) & inb
        for d in tl.static_range(D):
            v = tl.load(points_ptr + b * stride_pb + offs * stride_pn + d,
                        mask=inb, other=0.0).to(tl.float32)
            valid = valid & (tl.abs(v) < _INF)
        tl.store(mrow + offs, tl.where(valid, _INF, -_INF), mask=inb)

    last = _resolve_start(start_ptr, b, N)

    for i in range(K):
        tl.store(idx_ptr + b * stride_ib + i, last.to(tl.int64))
        tl.store(mrow + last, -_INF)  # mark before the sweep: 0 < -inf is false
        cents = _centroid_coords(points_ptr, b, last, stride_pb, stride_pn, D)

        run_bv = -_INF
        run_bi = 0
        for t in range(n_tiles):
            offs = t * BLOCK_N + tl.arange(0, BLOCK_N)
            inb = offs < N
            valid, dist = _tile_valid_dist(points_ptr, mask_ptr, b, offs, inb,
                                           cents, stride_pb, stride_pn,
                                           stride_mb, D)
            old = tl.load(mrow + offs, mask=inb, other=-_INF)
            new = tl.where(valid & (dist < old), dist, old)
            tl.store(mrow + offs, new, mask=inb)
            tv, ti = tl.max(new, axis=0, return_indices=True,
                            return_indices_tie_break_left=True)
            better = tv > run_bv  # strict: earlier tile wins ties
            run_bi = tl.where(better, t * BLOCK_N + ti, run_bi)
            run_bv = tl.where(better, tv, run_bv)

        last = tl.where(run_bv == -_INF, last, run_bi)


@triton.jit
def _fps_knn_tiled_kernel(points_ptr, mask_ptr, start_ptr, idx_ptr, nbr_ptr,
                          min_ptr, curr_ptr,
                          N, K, k_nbrs,
                          stride_pb, stride_pn, stride_mb,
                          stride_ib, stride_nb,
                          D: tl.constexpr, BLOCK_N: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    mrow = min_ptr + b * N
    crow = curr_ptr + b * N
    n_tiles = tl.cdiv(N, BLOCK_N)

    for t in range(n_tiles):
        offs = t * BLOCK_N + tl.arange(0, BLOCK_N)
        inb = offs < N
        m = tl.load(mask_ptr + b * stride_mb + offs, mask=inb, other=0)
        valid = (m != 0) & inb
        for d in tl.static_range(D):
            v = tl.load(points_ptr + b * stride_pb + offs * stride_pn + d,
                        mask=inb, other=0.0).to(tl.float32)
            valid = valid & (tl.abs(v) < _INF)
        tl.store(mrow + offs, tl.where(valid, _INF, -_INF), mask=inb)

    last = _resolve_start(start_ptr, b, N)

    for i in range(K):
        tl.store(idx_ptr + b * stride_ib + i, last.to(tl.int64))
        tl.store(mrow + last, -_INF)
        cents = _centroid_coords(points_ptr, b, last, stride_pb, stride_pn, D)

        run_bv = -_INF
        run_bi = 0
        for t in range(n_tiles):
            offs = t * BLOCK_N + tl.arange(0, BLOCK_N)
            inb = offs < N
            valid, dist = _tile_valid_dist(points_ptr, mask_ptr, b, offs, inb,
                                           cents, stride_pb, stride_pn,
                                           stride_mb, D)
            nd = tl.where(valid, dist, _INF)
            nd = tl.where(nd == nd, nd, _INF)
            tl.store(crow + offs, nd, mask=inb)
            old = tl.load(mrow + offs, mask=inb, other=-_INF)
            new = tl.where(valid & (dist < old), dist, old)
            tl.store(mrow + offs, new, mask=inb)
            tv, ti = tl.max(new, axis=0, return_indices=True,
                            return_indices_tie_break_left=True)
            better = tv > run_bv
            run_bi = tl.where(better, t * BLOCK_N + ti, run_bi)
            run_bv = tl.where(better, tv, run_bv)

        for nn in range(k_nbrs):
            nb_bv = _INF
            nb_bi = 0
            for t in range(n_tiles):
                offs = t * BLOCK_N + tl.arange(0, BLOCK_N)
                inb = offs < N
                cd = tl.load(crow + offs, mask=inb, other=_INF)
                tv, ti = tl.min(cd, axis=0, return_indices=True,
                                return_indices_tie_break_left=True)
                better = tv < nb_bv  # strict: earlier tile wins ties
                nb_bi = tl.where(better, t * BLOCK_N + ti, nb_bi)
                nb_bv = tl.where(better, tv, nb_bv)
            chosen = tl.where(nb_bv == _INF, last, nb_bi)
            tl.store(nbr_ptr + b * stride_nb + i * k_nbrs + nn,
                     chosen.to(tl.int64))
            tl.store(crow + nb_bi, _INF)

        last = tl.where(run_bv == -_INF, last, run_bi)


@triton.jit
def _nearest_assign_kernel(p_ptr, c_ptr, starts_ptr, counts_ptr, out_ptr,
                           K,
                           stride_p, stride_c,
                           D: tl.constexpr, BLOCK_N: tl.constexpr,
                           BLOCK_K: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    start = tl.load(starts_ptr + b)
    count = tl.load(counts_ptr + b)

    offs_n = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    ok_n = offs_n < count
    rows = start + offs_n.to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K)
    ok_k = offs_k < K
    c_rows = (b * K + offs_k.to(tl.int64)) * stride_c

    d = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for dim in tl.static_range(D):
        p = tl.load(p_ptr + rows * stride_p + dim, mask=ok_n, other=0.0).to(tl.float32)
        c = tl.load(c_ptr + c_rows + dim, mask=ok_k, other=0.0).to(tl.float32)
        diff = p[:, None] - c[None, :]
        d += diff * diff
    # NaN centroids (reachable via degenerate Lloyd charge weights) and the
    # k >= K padding lanes must never win the argmin
    d = tl.where(d == d, d, _INF)
    d = tl.where(ok_k[None, :], d, _INF)
    a = tl.argmin(d, axis=1, tie_break_left=True)
    tl.store(out_ptr + rows, a.to(tl.int64), mask=ok_n)


def nearest_assign(p_flat: Tensor, cents: Tensor, starts: Tensor,
                   counts: Tensor, n_max: int, K: int) -> Tensor:
    """[N] int64 nearest-centroid index in [0, K) for batch-segmented flat points."""
    N, D = p_flat.shape
    B = starts.numel()
    out = torch.empty(N, device=p_flat.device, dtype=torch.long)
    if N == 0 or B == 0:
        return out
    if K > 512:
        raise ValueError(f"nearest_assign supports K <= 512, got {K}")
    p = p_flat.detach().contiguous()
    c = cents.detach().contiguous()
    BLOCK_K = max(triton.next_power_of_2(K), 16)
    # small tiles: the [BLOCK_N, BLOCK_K] fp32 distance tile must stay
    # register-resident (larger budgets spill and run 2-8x slower)
    BLOCK_N = max(16, 2048 // BLOCK_K)
    grid = (B, triton.cdiv(max(n_max, 1), BLOCK_N))
    _nearest_assign_kernel[grid](p, c, starts, counts, out, K,
                                 p.stride(0), c.stride(0),
                                 D=D, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                                 num_warps=2)
    return out


def fps(points: Tensor, mask: Tensor, start_idx: Tensor, K: int) -> Tensor:
    B, N, D = points.shape
    idx = torch.empty(B, K, device=points.device, dtype=torch.long)
    if B == 0 or K == 0:
        return idx
    if N <= _single_tile_cap(SINGLE_TILE_MAX_N, D):
        BLOCK_N = max(triton.next_power_of_2(N), 16)
        _fps_single_kernel[(B,)](points, mask, start_idx, idx, idx, N, K,
                                 points.stride(0), points.stride(1),
                                 mask.stride(0), idx.stride(0),
                                 D=D, BLOCK_N=BLOCK_N, WITH_ASSIGN=False,
                                 num_warps=_num_warps(BLOCK_N))
    else:
        min_d = torch.empty(B, N, device=points.device, dtype=torch.float32)
        _fps_tiled_kernel[(B,)](points, mask, start_idx, idx, min_d, N, K,
                                points.stride(0), points.stride(1),
                                mask.stride(0), idx.stride(0),
                                D=D, BLOCK_N=TILE_N, num_warps=8)
    return idx


def fps_assign(points: Tensor, mask: Tensor, start_idx: Tensor,
               K: int) -> tuple[Tensor, Tensor]:
    """Fused FPS + nearest-centroid assignment (single-tile N only)."""
    B, N, D = points.shape
    idx = torch.empty(B, K, device=points.device, dtype=torch.long)
    assign = torch.zeros(B, N, device=points.device, dtype=torch.long)
    if B == 0 or K == 0:
        return idx, assign
    BLOCK_N = max(triton.next_power_of_2(N), 16)
    _fps_single_kernel[(B,)](points, mask, start_idx, idx, assign, N, K,
                             points.stride(0), points.stride(1),
                             mask.stride(0), idx.stride(0),
                             D=D, BLOCK_N=BLOCK_N, WITH_ASSIGN=True,
                             num_warps=_num_warps(BLOCK_N))
    return idx, assign


def fps_knn(points: Tensor, mask: Tensor, start_idx: Tensor,
            K: int, k_neighbors: int) -> tuple[Tensor, Tensor]:
    B, N, D = points.shape
    idx = torch.empty(B, K, device=points.device, dtype=torch.long)
    nbr = torch.empty(B, K, k_neighbors, device=points.device, dtype=torch.long)
    if B == 0 or K == 0:
        return idx, nbr
    args = (points, mask, start_idx, idx, nbr, N, K, k_neighbors,
            points.stride(0), points.stride(1), mask.stride(0),
            idx.stride(0), nbr.stride(0))
    if N <= _single_tile_cap(KNN_SINGLE_TILE_MAX_N, D):
        BLOCK_N = max(triton.next_power_of_2(N), 16)
        _fps_knn_single_kernel[(B,)](*args, D=D, BLOCK_N=BLOCK_N,
                                     num_warps=_num_warps(BLOCK_N))
    else:
        min_d = torch.empty(B, N, device=points.device, dtype=torch.float32)
        curr_d = torch.empty(B, N, device=points.device, dtype=torch.float32)
        _fps_knn_tiled_kernel[(B,)](*args[:5], min_d, curr_d, *args[5:],
                                    D=D, BLOCK_N=TILE_N, num_warps=8)
    return idx, nbr
