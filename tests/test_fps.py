"""torch_fps tests: API behaviour on CPU/CUDA, Triton-vs-reference and
C++-vs-reference parity, and the CPU fallback when the C++ build fails.

Triton (and the C++ kernels, where the compiler contracts to FMA) fuse
multiply-adds, so fp32 near-ties can resolve differently from the reference.
Parity checks therefore accept a divergence only where the competing
candidates' distances agree within tolerance.
"""
import warnings

import pytest
import torch

from torch_fps import (
    farthest_point_sampling,
    farthest_point_sampling_with_assign,
    farthest_point_sampling_with_knn,
    nearest_assign,
)
from torch_fps import _cpu
from torch_fps import _reference as R

CUDA = torch.cuda.is_available()
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not CUDA, reason="CUDA required"))]
needs_cuda = pytest.mark.skipif(not CUDA, reason="CUDA required")
if CUDA:
    from torch_fps import _triton
RTOL = 1e-5


def _valid(pts, mask):
    return mask & pts.isfinite().all(-1)


def _repair_start(pts, mask, start):
    valid = _valid(pts, mask)
    ok = valid.gather(1, start.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return torch.where(ok, start, valid.long().argmax(1))


def _d2(a, b):
    return (a.double() - b.double()).square().sum(-1)


def assert_fps_close(pts, mask, got, ref):
    """Rows must match, or diverge only at a near-tie. Returns rows that match."""
    got, ref, pts, valid = got.cpu(), ref.cpu(), pts.cpu(), _valid(pts, mask).cpu()
    same = torch.ones(got.shape[0], dtype=torch.bool)
    for b in range(got.shape[0]):
        diff = (got[b] != ref[b]).nonzero()
        if diff.numel() == 0:
            continue
        j = int(diff[0])
        assert valid[b, got[b, j]], f"row {b}: selected invalid point"
        p, sel = pts[b], ref[b, :j]
        d = _d2(p[:, None], p[sel][None]).min(1).values
        dg, dr = float(d[got[b, j]]), float(d[ref[b, j]])
        assert abs(dg - dr) <= RTOL * max(dg, dr, 1e-12), f"row {b} diverges at {j}: {dg} vs {dr}"
        same[b] = False
    assert same.float().mean() >= 0.5, "too many near-tie divergences"
    return same


def assert_knn_close(pts, mask, got, ref):
    (cg, ng), (cr, nr) = [(c.cpu(), n.cpu()) for c, n in (got, ref)]
    rows = assert_fps_close(pts, mask, cg, cr)
    p = pts.cpu()
    for b in rows.nonzero().flatten().tolist():
        c = p[b, cg[b]][:, None]
        dg, dr = _d2(p[b][ng[b]], c), _d2(p[b][nr[b]], c)
        assert torch.allclose(dg, dr, rtol=RTOL, atol=1e-9), f"row {b} kNN distances differ"


def assert_assign_close(pts, mask, idx, got, ref):
    p, idx, valid = pts.cpu(), idx.cpu(), _valid(pts, mask).cpu()
    got, ref = got.cpu(), ref.cpu()
    for b in range(p.shape[0]):
        c = p[b, idx[b]]
        v = valid[b]
        dg, dr = _d2(p[b][v], c[got[b][v]]), _d2(p[b][v], c[ref[b][v]])
        assert torch.allclose(dg, dr, rtol=RTOL, atol=1e-9), f"row {b} assign distances differ"


def _ragged_nan_inputs(B, N, D, K, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    pts = torch.randn(B, N, D, device=device, generator=g)
    counts = torch.randint(K, N + 1, (B,), device=device, generator=g)
    mask = torch.arange(N, device=device)[None, :] < counts[:, None]
    pts = pts * mask[..., None]
    rows = torch.randint(0, B, (B // 4 + 1,), device=device, generator=g)
    cols = torch.randint(1, K, (B // 4 + 1,), device=device, generator=g)
    pts[rows, cols, D - 1] = float("nan")
    return pts, mask


# ---------------------------------------------------------------- parity (CUDA)

@needs_cuda
class TestTritonParity:
    @pytest.mark.parametrize("B,N,D,K", [(4, 100, 3, 16), (8, 256, 4, 64), (2, 50, 2, 10), (8, 200, 6, 32)])
    @pytest.mark.parametrize("masked", [False, True])
    def test_fps(self, B, N, D, K, masked):
        torch.manual_seed(42)
        pts = torch.randn(B, N, D, device="cuda")
        mask = (torch.rand(B, N, device="cuda") > 0.3) if masked else torch.ones(B, N, dtype=torch.bool, device="cuda")
        K = min(K, int(mask.sum(1).min()))
        start = torch.zeros(B, dtype=torch.long, device="cuda")
        idx = farthest_point_sampling(pts, mask, K, start_idx=start, random_start=False)
        ref = R.fps_reference(pts, mask, _repair_start(pts, mask, start), K)
        assert_fps_close(pts, mask, idx, ref)

    @pytest.mark.parametrize("B,N,D,K,k", [(4, 100, 4, 16, 8), (8, 512, 4, 64, 16), (2, 50, 3, 10, 5), (2, 50, 4, 10, 50)])
    def test_knn(self, B, N, D, K, k):
        torch.manual_seed(42)
        pts = torch.randn(B, N, D, device="cuda")
        mask = torch.ones(B, N, dtype=torch.bool, device="cuda")
        start = torch.zeros(B, dtype=torch.long, device="cuda")
        got = farthest_point_sampling_with_knn(pts, mask, K, k, start_idx=start, random_start=False)
        assert got[1].shape == (B, K, k)
        assert_knn_close(pts, mask, got, R.fps_knn_reference(pts, mask, start, K, k))

    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("B,N,D,K,k", [(16, 777, 4, 64, 8), (8, 300, 3, 32, 5)])
    def test_ragged_nan(self, seed, B, N, D, K, k):
        # validate=False: NaNs may push a row below K valid points (padding path)
        pts, mask = _ragged_nan_inputs(B, N, D, K, "cpu", seed)
        start = _valid(pts, mask).long().argmax(1)
        ref = R.fps_knn_reference(pts, mask, start, K, k)
        got = farthest_point_sampling_with_knn(pts.cuda(), mask.cuda(), K, k, start_idx=start.cuda(),
                                               random_start=False, validate=False)
        assert_knn_close(pts, mask, got, ref)

    @pytest.mark.parametrize("B,N,D,K,k", [(4, 512, 4, 32, 8), (4, 300, 3, 16, 4)])
    def test_tiled_kernels(self, monkeypatch, B, N, D, K, k):
        monkeypatch.setattr(_triton, "SINGLE_TILE_MAX_N", 128)
        monkeypatch.setattr(_triton, "KNN_SINGLE_TILE_MAX_N", 128)
        pts, mask = _ragged_nan_inputs(B, N, D, K, "cuda", seed=7)
        start = _valid(pts, mask).long().argmax(1)
        got = farthest_point_sampling_with_knn(pts, mask, K, k, start_idx=start, random_start=False, validate=False)
        assert_knn_close(pts, mask, got, R.fps_knn_reference(pts.cpu(), mask.cpu(), start.cpu(), K, k))
        idx, assign = farthest_point_sampling_with_assign(pts, mask, K, start_idx=start,
                                                          random_start=False, validate=False)
        assert_assign_close(pts, mask, idx, assign, R.assign_reference(pts, idx))

    @pytest.mark.parametrize("seed", range(3))
    @pytest.mark.parametrize("B,N,D,K", [(16, 777, 4, 64), (8, 300, 3, 32)])
    def test_assign(self, seed, B, N, D, K):
        pts, mask = _ragged_nan_inputs(B, N, D, K, "cpu", seed)
        start = _valid(pts, mask).long().argmax(1)
        kw = dict(start_idx=start.cuda(), random_start=False, validate=False)
        idx, assign = farthest_point_sampling_with_assign(pts.cuda(), mask.cuda(), K, **kw)
        assert torch.equal(idx, farthest_point_sampling(pts.cuda(), mask.cuda(), K, **kw))
        assert_fps_close(pts, mask, idx, R.fps_reference(pts, mask, start, K))
        v = _valid(pts, mask)
        assert assign.cpu()[v].min() >= 0 and assign.cpu()[v].max() < K
        assert_assign_close(pts, mask, idx, assign, R.assign_reference(pts.cuda(), idx))

    def test_tiled_assign_large_n(self):
        torch.manual_seed(0)
        B, D, K = 3, 4, 64
        N = _triton._single_tile_cap(_triton.SINGLE_TILE_MAX_N, D) + 5000
        pts = torch.randn(B, N, D, device="cuda")
        mask = torch.rand(B, N, device="cuda") > 0.2
        idx, assign = farthest_point_sampling_with_assign(pts, mask, K, random_start=False)
        assert assign.shape == (B, N) and int(assign[mask].max()) < K
        assert_assign_close(pts, mask, idx, assign, R.assign_reference(pts, idx))

    @pytest.mark.parametrize("seed", range(3))
    @pytest.mark.parametrize("B,K", [(8, 64), (4, 128), (3, 100), (2, 600)])
    def test_nearest_assign(self, seed, B, K):
        g = torch.Generator(device="cuda").manual_seed(seed)
        counts = torch.randint(K, 3000, (B,), device="cuda", generator=g)
        starts = torch.cumsum(counts, 0) - counts
        p = torch.randn(int(counts.sum()), 4, device="cuda", generator=g)
        cents = torch.randn(B * K, 4, device="cuda", generator=g)
        out = nearest_assign(p, cents, starts, counts, int(counts.max()), K)
        ref = R.nearest_assign_reference(p, cents, starts, counts, K)
        seg = torch.repeat_interleave(torch.arange(B, device="cuda"), counts)
        c = cents.view(B, K, 4)
        dg, dr = _d2(p, c[seg, out]), _d2(p, c[seg, ref])
        assert torch.allclose(dg, dr, rtol=RTOL, atol=1e-9)

    def test_nearest_assign_nan_centroid_and_ties(self):
        p = torch.tensor([[0., 0., 0., 0.], [1., 0., 0., 0.]], device="cuda")
        cents = torch.tensor([[float("nan")] * 4, [0.] * 4, [0.] * 4], device="cuda")
        one = torch.tensor([0], device="cuda"), torch.tensor([2], device="cuda")
        assert nearest_assign(p, cents, *one, 2, 3).tolist() == [1, 1]

    def test_cross_device(self):
        torch.manual_seed(1234)
        pts = torch.randn(4, 128, 4)
        mask = torch.ones(4, 128, dtype=torch.bool)
        mask[1, 96:] = False
        cpu = farthest_point_sampling_with_knn(pts, mask, 32, 8, random_start=False)
        gpu = farthest_point_sampling_with_knn(pts.cuda(), mask.cuda(), 32, 8, random_start=False)
        assert_knn_close(pts, mask, gpu, cpu)

    def test_inf_parity(self):
        torch.manual_seed(0)
        pts = torch.randn(2, 8, 3)
        pts[0, 3, 1] = float("inf")
        pts[1, 2, 0] = pts[1, 6, 0] = float("-inf")
        mask = torch.ones(2, 8, dtype=torch.bool)
        start = torch.zeros(2, dtype=torch.long)
        got = farthest_point_sampling_with_knn(pts.cuda(), mask.cuda(), 5, 4, start_idx=start.cuda(), random_start=False)
        assert_knn_close(pts, mask, got, R.fps_knn_reference(pts, mask, start, 5, 4))

    @pytest.mark.parametrize("low", [torch.float16, torch.bfloat16])
    def test_low_precision(self, low):
        torch.manual_seed(0)
        pts = (torch.randn(1, 500, 64) * 20).cuda()
        mask = torch.ones(1, 500, dtype=torch.bool, device="cuda")
        a = farthest_point_sampling(pts, mask, 32, random_start=False)
        b = farthest_point_sampling(pts.to(low), mask, 32, random_start=False, precision=low)
        sa, sb = set(a[0].tolist()), set(b[0].tolist())
        assert len(sa & sb) / len(sa | sb) >= 0.9

    @pytest.mark.parametrize("random_start", [False, True])
    def test_no_host_sync(self, random_start):
        torch.manual_seed(42)
        pts = torch.randn(4, 128, 4, device="cuda")
        mask = torch.ones(4, 128, dtype=torch.bool, device="cuda")
        mask[1, 96:] = False
        big = torch.randn(2, _triton._single_tile_cap(_triton.SINGLE_TILE_MAX_N, 4) + 100, 4, device="cuda")
        big_mask = torch.ones(big.shape[:2], dtype=torch.bool, device="cuda")
        kw = dict(random_start=random_start, validate=False)

        def run(k):
            farthest_point_sampling(pts, mask, k, **kw)
            farthest_point_sampling_with_knn(pts, mask, k, 4, **kw)
            farthest_point_sampling_with_assign(pts, mask, k, assume_finite=True, **kw)
            farthest_point_sampling_with_assign(big, big_mask, k, assume_finite=True, **kw)
        run(8)  # warm up CUDA state and Triton JIT
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            run(32)
        finally:
            torch.cuda.set_sync_debug_mode(0)


# ----------------------------------------------------------- parity (C++ CPU)

@pytest.fixture
def cpp():
    """The compiled CPU extension; skips when it cannot be built here."""
    mod = _cpu.get_module()
    if mod is None:
        pytest.skip("C++ CPU extension unavailable (no compiler?)")
    return mod


def _cpu_inputs(B, N, D, K, kind, seed):
    """'full': all valid; 'ragged': prefix masks with NaNs; 'random': scattered mask."""
    if kind == "ragged":
        return _ragged_nan_inputs(B, N, D, K, "cpu", seed)
    g = torch.Generator().manual_seed(seed)
    pts = torch.randn(B, N, D, generator=g)
    if kind == "full":
        return pts, torch.ones(B, N, dtype=torch.bool)
    mask = torch.rand(B, N, generator=g) > 0.4
    mask[:, :K] = True  # at least K valid per row, scattered by the permutation
    return pts, mask[:, torch.randperm(N, generator=g)]


CPU_SHAPES = [(1, 50, 3, 16), (3, 300, 4, 64), (16, 192, 4, 128), (64, 1000, 3, 32),
              (1, 3000, 4, 128)]  # B == 1 and N * D >= 8192: within-row parallel kernel


class TestCppParity:
    @pytest.mark.parametrize("B,N,D,K", CPU_SHAPES)
    @pytest.mark.parametrize("kind", ["full", "ragged", "random"])
    @pytest.mark.parametrize("start", ["random", "fixed"])
    def test_fps(self, cpp, B, N, D, K, kind, start):
        pts, mask = _cpu_inputs(B, N, D, K, kind, seed=B * 131 + N)
        if start == "random":
            kw = dict(generator=torch.Generator().manual_seed(9))
        else:  # arbitrary starts; masked / NaN slots get repaired by the API
            kw = dict(start_idx=torch.randint(0, N, (B,), generator=torch.Generator().manual_seed(9)))
        idx = farthest_point_sampling(pts, mask, K, validate=False, **kw)
        s0 = idx[:, 0] if start == "random" else _repair_start(pts, mask, kw["start_idx"])
        assert_fps_close(pts, mask, idx, R.fps_reference(pts, mask, s0, K))

    @pytest.mark.parametrize("B,N,D,K", CPU_SHAPES)
    @pytest.mark.parametrize("kind", ["full", "ragged", "random"])
    @pytest.mark.parametrize("k", [1, 8, 32])
    def test_knn(self, cpp, B, N, D, K, kind, k):
        pts, mask = _cpu_inputs(B, N, D, K, kind, seed=B * 17 + N + k)
        got = farthest_point_sampling_with_knn(pts, mask, K, k, validate=False,
                                               generator=torch.Generator().manual_seed(3))
        assert got[1].shape == (B, K, k)
        assert_knn_close(pts, mask, got, R.fps_knn_reference(pts, mask, got[0][:, 0], K, k))

    @pytest.mark.parametrize("B,N,D,K", CPU_SHAPES)
    @pytest.mark.parametrize("kind", ["full", "ragged", "random"])
    def test_assign(self, cpp, B, N, D, K, kind):
        pts, mask = _cpu_inputs(B, N, D, K, kind, seed=B + N)
        idx, assign = farthest_point_sampling_with_assign(pts, mask, K, validate=False,
                                                          generator=torch.Generator().manual_seed(5))
        assert_fps_close(pts, mask, idx, R.fps_reference(pts, mask, idx[:, 0], K))
        assert_assign_close(pts, mask, idx, assign, R.assign_reference(pts, idx))

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
    def test_dtypes(self, cpp, dtype):
        pts, mask = _ragged_nan_inputs(8, 300, 4, 32, "cpu", seed=2)
        start = _valid(pts, mask).long().argmax(1)
        pts = pts.to(dtype)
        kw = dict(start_idx=start, random_start=False, validate=False, precision=dtype)
        got = farthest_point_sampling_with_knn(pts, mask, 32, 8, **kw)
        ref = R.fps_knn_reference(pts, mask, start, 32, 8)
        if dtype == torch.float64:  # same accumulation order in fp64: exact
            assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])
        else:
            assert_knn_close(pts, mask, got, ref)

    def test_exhausted_rows_pad_like_reference(self, cpp):
        # validate=False with K > valid count, including an all-invalid row:
        # centroids repeat the last pick, and padded slots carry that pick's
        # neighbours (all-invalid rows pad everything with the start index)
        torch.manual_seed(4)
        B, N, D, K, k = 4, 40, 3, 16, 6
        pts = torch.randn(B, N, D)
        mask = torch.zeros(B, N, dtype=torch.bool)
        mask[0, :5] = True
        mask[1, 10:13] = True
        mask[2, :] = True
        start = torch.tensor([0, 11, 7, 2])
        kw = dict(start_idx=start, random_start=False, validate=False)
        got = farthest_point_sampling_with_knn(pts, mask, K, k, **kw)
        # all-invalid rows keep the supplied start (the API repairs nothing there)
        ref = R.fps_knn_reference(pts, mask, got[0][:, 0], K, k)
        assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])
        assert got[0][3].tolist() == [2] * K and got[1][3].unique().tolist() == [2]
        idx = farthest_point_sampling(pts, mask, K, **kw)
        assert torch.equal(idx, ref[0])


class TestCPUFallback:
    @pytest.fixture
    def broken_build(self, monkeypatch):
        def boom():
            raise RuntimeError("no compiler")
        monkeypatch.setattr(_cpu, "_build", boom)
        monkeypatch.setattr(_cpu, "_state", "unloaded")
        monkeypatch.setattr(_cpu, "_module", None)

    def test_fallback_matches_reference_and_warns_once(self, broken_build):
        pts, mask = _ragged_nan_inputs(8, 300, 4, 32, "cpu", seed=1)
        start = _valid(pts, mask).long().argmax(1)
        kw = dict(start_idx=start, random_start=False, validate=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            idx = farthest_point_sampling(pts, mask, 32, **kw)
            cent, nbr = farthest_point_sampling_with_knn(pts, mask, 32, 8, **kw)
            a_idx, assign = farthest_point_sampling_with_assign(pts, mask, 32, **kw)
        msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
        assert len(msgs) == 1 and "no compiler" in msgs[0], msgs
        assert _cpu._state == "failed" and _cpu.get_module() is None
        ref_c, ref_n = R.fps_knn_reference(pts, mask, start, 32, 8)
        assert torch.equal(idx, R.fps_reference(pts, mask, start, 32))
        assert torch.equal(cent, ref_c) and torch.equal(nbr, ref_n)
        assert torch.equal(a_idx, idx) and torch.equal(assign, R.assign_reference(pts, idx))

    def test_failed_state_skips_build(self, monkeypatch):
        def boom():
            raise AssertionError("must not rebuild once failed")
        monkeypatch.setattr(_cpu, "_build", boom)
        monkeypatch.setattr(_cpu, "_state", "failed")
        monkeypatch.setattr(_cpu, "_module", None)
        pts = torch.randn(2, 64, 3, dtype=torch.float64)
        mask = torch.ones(2, 64, dtype=torch.bool)
        start = torch.zeros(2, dtype=torch.long)
        idx = farthest_point_sampling(pts, mask, 16, start_idx=start, precision=torch.float64)
        assert torch.equal(idx, R.fps_reference(pts, mask, start, 16))


# ------------------------------------------------------------- API behaviour

@pytest.mark.parametrize("device", DEVICES)
class TestAPI:
    def test_determinism(self, device):
        torch.manual_seed(42)
        pts = torch.randn(4, 100, 4, device=device)
        mask = torch.ones(4, 100, dtype=torch.bool, device=device)
        gens = [torch.Generator(device=device).manual_seed(123) for _ in range(2)]
        a, b = (farthest_point_sampling_with_knn(pts, mask, 16, 8, generator=g) for g in gens)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])

    def test_duplicate_points(self, device):
        pts = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]], device=device)
        mask = torch.ones(1, 4, dtype=torch.bool, device=device)
        idx = farthest_point_sampling(pts, mask, 4, random_start=False)
        assert idx[0].tolist() == [0, 3, 2, 1]
        unique = torch.randn(4, 3)
        pts = unique[[0, 1, 2, 3, 0, 1, 2, 3, 0, 1]].unsqueeze(0).to(device)
        idx = farthest_point_sampling(pts, torch.ones(1, 10, dtype=torch.bool, device=device), 8, random_start=False)
        assert len(set(idx[0].tolist())) == 8

    def test_all_zeros_masked(self, device):
        pts = torch.zeros(1, 5, 3, device=device)
        mask = torch.tensor([[True, False, True, True, True]], device=device)
        idx = farthest_point_sampling(pts, mask, 4, random_start=False)
        assert idx[0].tolist() == [0, 2, 3, 4]

    def test_random_start_uniform(self, device):
        pts = torch.randn(1, 8, 3, device=device)
        mask = torch.tensor([[False, False, True, False, True, False, True, False]], device=device)
        counts = {2: 0, 4: 0, 6: 0}
        for seed in range(1500):
            gen = torch.Generator(device=device).manual_seed(seed)
            counts[farthest_point_sampling(pts, mask, 1, generator=gen)[0, 0].item()] += 1
        assert all(abs(v - 500) < 100 for v in counts.values()), counts

    def test_knn_short_rows_and_order(self, device):
        pts = torch.randn(1, 10, 3, device=device)
        mask = torch.tensor([[False] + [True] * 5 + [False] * 4], device=device)
        _, nbr = farthest_point_sampling_with_knn(pts, mask, 3, 8, start_idx=torch.tensor([1], device=device))
        assert bool(mask[0, nbr.flatten()].all())
        pts = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [1.0, 0.0], [3.0, 0.0]]], device=device)
        _, nbr = farthest_point_sampling_with_knn(pts, torch.ones(1, 4, dtype=torch.bool, device=device), 1, 4,
                                                  random_start=False)
        assert nbr[0, 0].tolist() == [0, 2, 1, 3]

    def test_non_finite_excluded(self, device):
        torch.manual_seed(0)
        pts = torch.randn(2, 10, 3, device=device)
        pts[0, 5] = float("nan")
        pts[0, 3, 1] = float("inf")
        pts[1, 2, 0] = float("-inf")
        mask = torch.ones(2, 10, dtype=torch.bool, device=device)
        cent, nbr = farthest_point_sampling_with_knn(pts, mask, 6, 4, random_start=False)
        for b, bad in ((0, {3, 5}), (1, {2})):
            assert not bad & set(cent[b].tolist()) and not bad & set(nbr[b].flatten().tolist())
        # 3 mask-true points, 2 finite: K=3 raises, K=2 passes
        pts = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [float("nan"), 0., 0.]]], device=device)
        mask = torch.ones(1, 3, dtype=torch.bool, device=device)
        with pytest.raises(ValueError, match="K <= number of valid points"):
            farthest_point_sampling(pts, mask, 3, random_start=False)
        assert sorted(farthest_point_sampling(pts, mask, 2, random_start=False)[0].tolist()) == [0, 1]

    def test_validate_flag(self, device):
        torch.manual_seed(42)
        pts = torch.randn(4, 128, 4, device=device)
        mask = torch.ones(4, 128, dtype=torch.bool, device=device)
        mask[1, 96:] = False
        for rs in (False, True):
            ga, gb = (torch.Generator(device=device).manual_seed(7) for _ in range(2))
            a = farthest_point_sampling(pts, mask, 32, random_start=rs, generator=ga, validate=True)
            b = farthest_point_sampling(pts, mask, 32, random_start=rs, generator=gb, validate=False)
            assert torch.equal(a, b)
        start = torch.tensor([5, 0, 63, 1], device=device)
        assert torch.equal(farthest_point_sampling(pts, mask, 16, start_idx=start),
                           farthest_point_sampling(pts, mask, 16, start_idx=start, validate=False))
        mask[0, 5:] = False
        with pytest.raises(ValueError, match="K <= number of valid points"):
            farthest_point_sampling(pts, mask, 8, random_start=False)

    def test_implicit_start(self, device):
        torch.manual_seed(42)
        pts = torch.randn(3, 64, 4, device=device)
        mask = torch.ones(3, 64, dtype=torch.bool, device=device)
        mask[0, 0] = False
        mask[2, :10] = False
        idx = farthest_point_sampling(pts, mask, 16, random_start=False)
        assert idx[:, 0].tolist() == [1, 0, 10] and bool(mask.gather(1, idx).all())
        # a start on a masked slot is repaired to the first valid index
        start = torch.tensor([0, 0, 3], device=device)
        assert torch.equal(idx, farthest_point_sampling(pts, mask, 16, start_idx=start))
        # out-of-range starts (validate=False) are repaired too, even when slot N-1 is valid
        for s in (-3, 64, 1000):
            start = torch.full((3,), s, device=device)
            assert torch.equal(idx, farthest_point_sampling(pts, mask, 16, start_idx=start, validate=False))

    def test_storage_offset(self, device):
        flat = torch.randn(4 * 128 * 4 + 1, device=device)
        pts = flat.narrow(0, 1, 4 * 128 * 4).view(4, 128, 4)
        mask = torch.ones(4, 128, dtype=torch.bool, device=device)
        a = farthest_point_sampling_with_knn(pts, mask, 16, 8, random_start=False)
        b = farthest_point_sampling_with_knn(pts.clone(), mask, 16, 8, random_start=False)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])

    def test_assign_duplicate_centroid_ties(self, device):
        pts = torch.zeros(1, 4, 2, device=device)
        mask = torch.tensor([[True, True, True, False]], device=device)
        _, assign = farthest_point_sampling_with_assign(pts, mask, 5, random_start=False, validate=False)
        assert assign[0, :3].tolist() == [0, 0, 0]

    def test_assume_finite(self, device):
        torch.manual_seed(11)
        pts = torch.randn(8, 200, 4, device=device)
        mask = torch.rand(8, 200, device=device) > 0.2
        K = int(mask.sum(1).min())
        ga, gb = (torch.Generator(device=device).manual_seed(3) for _ in range(2))
        a = farthest_point_sampling_with_assign(pts, mask, K, generator=ga)
        b = farthest_point_sampling_with_assign(pts, mask, K, generator=gb, assume_finite=True)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])

    def test_empty(self, device):
        pts = torch.randn(0, 32, 4, device=device)
        mask = torch.ones(0, 32, dtype=torch.bool, device=device)
        assert farthest_point_sampling(pts, mask, 8).shape == (0, 8)
        cent, nbr = farthest_point_sampling_with_knn(pts, mask, 8, 4)
        assert cent.shape == (0, 8) and nbr.shape == (0, 8, 4)
        for B in (0, 1, 3):
            z = torch.zeros(B, 0, 3, device=device)
            zm = torch.zeros(B, 0, dtype=torch.bool, device=device)
            assert farthest_point_sampling(z, zm, 0).shape == (B, 0)
            assert farthest_point_sampling_with_assign(z, zm, 0)[1].shape == (B, 0)
        for validate in (True, False):
            with pytest.raises(ValueError):
                farthest_point_sampling(torch.zeros(1, 0, 3, device=device),
                                        torch.zeros(1, 0, dtype=torch.bool, device=device), 2, validate=validate)
        e = torch.empty(0, 4, device=device)
        assert nearest_assign(e, torch.zeros(3, 4, device=device), torch.zeros(1, dtype=torch.long, device=device),
                              torch.zeros(1, dtype=torch.long, device=device), 0, 3).shape == (0,)
