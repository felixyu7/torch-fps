"""
Correctness tests for FPS and FPS+kNN implementations.

Validates optimized kernels against pure PyTorch baselines.
"""

import sys
from pathlib import Path

import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torch_fps import farthest_point_sampling, farthest_point_sampling_with_knn
from baselines import fps_baseline, fps_with_knn_baseline


# ============================================================================
# FPS Correctness Tests
# ============================================================================

class TestFPS:
    """Test suite for farthest point sampling."""

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize("B,N,D,K", [
        (4, 100, 3, 16),
        (8, 256, 4, 64),
        (2, 50, 2, 10),
    ])
    def test_fps_correctness(self, device, B, N, D, K):
        """Test that optimized FPS matches baseline."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)
        points = torch.randn(B, N, D, device=device)
        mask = torch.ones(B, N, dtype=torch.bool, device=device)
        start_idx = torch.zeros(B, dtype=torch.long, device=device)

        # Optimized version (use fp32 for exact comparison with baseline)
        idx_opt = farthest_point_sampling(
            points, mask, K, start_idx=start_idx, random_start=False, precision=torch.float32
        )

        # Baseline version
        idx_base = fps_baseline(points, mask, K, start_idx)

        assert torch.equal(idx_opt, idx_base), \
            f"FPS indices mismatch on {device}"

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_fps_with_masking(self, device):
        """Test FPS with variable valid point counts."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)
        B, N, D = 4, 100, 4
        points = torch.randn(B, N, D, device=device)

        # Create variable masking
        mask = torch.rand(B, N, device=device) > 0.3
        counts = mask.sum(dim=1)
        K = int(counts.min().item())

        if K == 0:
            pytest.skip("No valid points after masking")

        start_idx = torch.zeros(B, dtype=torch.long, device=device)

        idx_opt = farthest_point_sampling(
            points, mask, K, start_idx=start_idx, random_start=False, precision=torch.float32
        )
        idx_base = fps_baseline(points, mask, K, start_idx)

        assert torch.equal(idx_opt, idx_base), \
            f"FPS with masking mismatch on {device}"

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_fps_edge_cases(self, device):
        """Test edge cases: K=1, K=N."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        B, N, D = 2, 20, 3
        points = torch.randn(B, N, D, device=device)
        mask = torch.ones(B, N, dtype=torch.bool, device=device)
        start_idx = torch.zeros(B, dtype=torch.long, device=device)

        # K = 1
        idx_opt = farthest_point_sampling(points, mask, 1, start_idx=start_idx, random_start=False, precision=torch.float32)
        idx_base = fps_baseline(points, mask, 1, start_idx)
        assert torch.equal(idx_opt, idx_base)

        # K = N
        idx_opt = farthest_point_sampling(points, mask, N, start_idx=start_idx, random_start=False, precision=torch.float32)
        idx_base = fps_baseline(points, mask, N, start_idx)
        assert torch.equal(idx_opt, idx_base)


# ============================================================================
# FPS+kNN Correctness Tests
# ============================================================================

class TestFPSWithKNN:
    """Test suite for fused FPS+kNN."""

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize("B,N,D,K,k", [
        (4, 100, 4, 16, 8),
        (8, 512, 4, 64, 16),
        (2, 50, 3, 10, 5),
    ])
    def test_fused_correctness(self, device, B, N, D, K, k):
        """Test that fused FPS+kNN matches separate operations."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)
        points = torch.randn(B, N, D, device=device)
        mask = torch.ones(B, N, dtype=torch.bool, device=device)
        start_idx = torch.zeros(B, dtype=torch.long, device=device)

        # Fused version (use fp32 for exact comparison with baseline)
        centroid_idx_fused, neighbor_idx_fused = farthest_point_sampling_with_knn(
            points, mask, K, k, start_idx=start_idx, random_start=False, precision=torch.float32
        )

        # Baseline (separate FPS + kNN)
        centroid_idx_base, neighbor_idx_base = fps_with_knn_baseline(
            points, mask, K, k, start_idx
        )

        assert torch.equal(centroid_idx_fused, centroid_idx_base), \
            f"Centroid indices mismatch on {device}"
        assert torch.equal(neighbor_idx_fused, neighbor_idx_base), \
            f"Neighbor indices mismatch on {device}"

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_fused_with_masking(self, device):
        """Test fused kernel with variable valid counts."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)
        B, N, D = 4, 100, 4
        points = torch.randn(B, N, D, device=device)
        mask = torch.rand(B, N, device=device) > 0.3
        counts = mask.sum(dim=1)
        K = int(counts.min().item())

        if K == 0:
            pytest.skip("No valid points after masking")

        k = min(5, K)
        start_idx = torch.zeros(B, dtype=torch.long, device=device)

        centroid_idx_fused, neighbor_idx_fused = farthest_point_sampling_with_knn(
            points, mask, K, k, start_idx=start_idx, random_start=False, precision=torch.float32
        )
        centroid_idx_base, neighbor_idx_base = fps_with_knn_baseline(
            points, mask, K, k, start_idx
        )

        assert torch.equal(centroid_idx_fused, centroid_idx_base)
        assert torch.equal(neighbor_idx_fused, neighbor_idx_base)

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_fused_edge_cases(self, device):
        """Test edge cases: k=1, k=N."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        B, N, D, K = 2, 50, 4, 10
        points = torch.randn(B, N, D, device=device)
        mask = torch.ones(B, N, dtype=torch.bool, device=device)
        start_idx = torch.zeros(B, dtype=torch.long, device=device)

        # k = 1
        centroid_idx_fused, neighbor_idx_fused = farthest_point_sampling_with_knn(
            points, mask, K, 1, start_idx=start_idx, random_start=False, precision=torch.float32
        )
        centroid_idx_base, neighbor_idx_base = fps_with_knn_baseline(
            points, mask, K, 1, start_idx
        )
        assert torch.equal(centroid_idx_fused, centroid_idx_base)
        assert torch.equal(neighbor_idx_fused, neighbor_idx_base)
        assert neighbor_idx_fused.shape == (B, K, 1)

        # k = N (maximum neighbors)
        centroid_idx_fused, neighbor_idx_fused = farthest_point_sampling_with_knn(
            points, mask, K, N, start_idx=start_idx, random_start=False, precision=torch.float32
        )
        centroid_idx_base, neighbor_idx_base = fps_with_knn_baseline(
            points, mask, K, N, start_idx
        )
        assert torch.equal(centroid_idx_fused, centroid_idx_base)
        assert torch.equal(neighbor_idx_fused, neighbor_idx_base)
        assert neighbor_idx_fused.shape == (B, K, N)


# ============================================================================
# Determinism Tests
# ============================================================================

class TestDeterminism:
    """Test that results are deterministic with fixed seeds."""

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_fps_determinism(self, device):
        """Test FPS produces same results with same seed."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        B, N, D, K = 4, 100, 4, 16

        # Run 1
        torch.manual_seed(42)
        points = torch.randn(B, N, D, device=device)
        mask = torch.ones(B, N, dtype=torch.bool, device=device)
        gen1 = torch.Generator(device=device).manual_seed(123)
        idx1 = farthest_point_sampling(points, mask, K, generator=gen1, precision=torch.float32)

        # Run 2 (same seed)
        gen2 = torch.Generator(device=device).manual_seed(123)
        idx2 = farthest_point_sampling(points, mask, K, generator=gen2, precision=torch.float32)

        assert torch.equal(idx1, idx2), "FPS not deterministic with same seed"

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_fused_determinism(self, device):
        """Test fused kernel produces same results with same seed."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        B, N, D, K, k = 4, 100, 4, 16, 8

        # Run 1
        torch.manual_seed(42)
        points = torch.randn(B, N, D, device=device)
        mask = torch.ones(B, N, dtype=torch.bool, device=device)
        gen1 = torch.Generator(device=device).manual_seed(123)
        cent1, neigh1 = farthest_point_sampling_with_knn(points, mask, K, k, generator=gen1, precision=torch.float32)

        # Run 2 (same seed)
        gen2 = torch.Generator(device=device).manual_seed(123)
        cent2, neigh2 = farthest_point_sampling_with_knn(points, mask, K, k, generator=gen2, precision=torch.float32)

        assert torch.equal(cent1, cent2), "Centroids not deterministic"
        assert torch.equal(neigh1, neigh2), "Neighbors not deterministic"


# ============================================================================
# Regression tests for the fixes landed alongside these tests
# ============================================================================

class TestDuplicatePoints:
    """Cover the 'FPS re-picks same index on duplicates' bug."""

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_duplicate_coords_unique_indices(self, device):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        points = torch.tensor(
            [[[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]],
            device=device,
        )
        mask = torch.ones(1, 4, dtype=torch.bool, device=device)
        start = torch.zeros(1, dtype=torch.long, device=device)
        idx = farthest_point_sampling(points, mask, K=4, start_idx=start, random_start=False)
        base = fps_baseline(points, mask, 4, start)
        assert torch.equal(idx, base)
        assert len(set(idx[0].tolist())) == 4

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_k_exceeds_unique_coords(self, device):
        """10 points at 4 unique positions, K=8 — must still return 8 distinct indices."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(0)
        unique = torch.randn(4, 3)
        points = unique[[0, 1, 2, 3, 0, 1, 2, 3, 0, 1]].unsqueeze(0).to(device)
        mask = torch.ones(1, 10, dtype=torch.bool, device=device)
        start = torch.zeros(1, dtype=torch.long, device=device)
        idx = farthest_point_sampling(points, mask, K=8, start_idx=start, random_start=False)
        base = fps_baseline(points, mask, 8, start)
        assert torch.equal(idx, base)
        assert len(set(idx[0].tolist())) == 8

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_all_zeros_with_mask(self, device):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        points = torch.zeros(1, 5, 3, device=device)
        mask = torch.tensor([[True, False, True, True, True]], device=device)
        start = torch.zeros(1, dtype=torch.long, device=device)
        idx = farthest_point_sampling(points, mask, K=4, start_idx=start, random_start=False)
        base = fps_baseline(points, mask, 4, start)
        assert torch.equal(idx, base)
        # every output index must be a valid (unmasked) point
        assert all(mask[0, i].item() for i in idx[0].tolist())


class TestRandomStartDistribution:
    """Cover the 'random_start collapses to first_valid' bug."""

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_scattered_mask_uniform(self, device):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        points = torch.randn(1, 8, 3, device=device)
        mask = torch.tensor(
            [[False, False, True, False, True, False, True, False]],
            device=device,
        )
        counts = {2: 0, 4: 0, 6: 0}
        trials = 1500
        for seed in range(trials):
            gen = torch.Generator(device=device).manual_seed(seed)
            idx = farthest_point_sampling(points, mask, K=1, generator=gen)
            pick = idx[0, 0].item()
            assert pick in counts, f"random start picked invalid index {pick}"
            counts[pick] += 1
        expected = trials / 3
        # chi-square-ish tolerance: each bin within 20% of expected
        for k, v in counts.items():
            assert abs(v - expected) < 0.2 * expected, \
                f"bin {k} has {v}, expected ~{expected:.0f}"


class TestLowPrecisionParity:
    """Cover the acc_t = scalar_t accumulator bug."""

    @pytest.mark.parametrize("low_dtype", [torch.float16, torch.bfloat16])
    def test_fp16_bf16_cuda_parity(self, low_dtype):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(0)
        B, N, D, K = 1, 500, 64, 32
        pts32 = (torch.randn(B, N, D) * 20).cuda()
        mask = torch.ones(B, N, dtype=torch.bool, device='cuda')
        start = torch.zeros(B, dtype=torch.long, device='cuda')

        ref = farthest_point_sampling(
            pts32, mask, K, start_idx=start, random_start=False, precision=torch.float32
        )
        low = farthest_point_sampling(
            pts32.to(low_dtype), mask, K,
            start_idx=start, random_start=False, precision=low_dtype,
        )

        ref_set = set(ref[0].tolist())
        low_set = set(low[0].tolist())
        jaccard = len(ref_set & low_set) / len(ref_set | low_set)
        # Pre-fix: ~3% agreement on this workload. Post-fix: should be >= 90%.
        assert jaccard >= 0.9, f"{low_dtype} parity too low: jaccard={jaccard:.3f}"


class TestKNNEdgeCases:
    """Cover kNN overflow (k_neighbors > valid_count) and the sorted contract."""

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_k_neighbors_exceeds_valid_count(self, device):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(0)
        B, N, D = 1, 10, 3
        points = torch.randn(B, N, D, device=device)
        # index 0 invalid; 5 valid at [1..5]
        mask = torch.tensor(
            [[False, True, True, True, True, True, False, False, False, False]],
            device=device,
        )
        start = torch.tensor([1], dtype=torch.long, device=device)
        cent, nbr = farthest_point_sampling_with_knn(
            points, mask, K=3, k_neighbors=8,
            start_idx=start, random_start=False,
        )
        # every returned neighbor must be a valid (unmasked) point
        flat = nbr.flatten().tolist()
        for i in flat:
            assert mask[0, i].item(), f"neighbor index {i} is masked-out"

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_neighbors_sorted_by_distance(self, device):
        """Must return neighbors in closest-first order (docstring contract)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        points = torch.tensor(
            [[[0.0, 0.0], [2.0, 0.0], [1.0, 0.0], [3.0, 0.0]]],
            device=device,
        )
        mask = torch.ones(1, 4, dtype=torch.bool, device=device)
        start = torch.zeros(1, dtype=torch.long, device=device)
        cent, nbr = farthest_point_sampling_with_knn(
            points, mask, K=1, k_neighbors=4,
            start_idx=start, random_start=False,
        )
        # centroid is index 0 at (0,0); sorted distances are [0, 1, 4, 9] → [0, 2, 1, 3]
        assert nbr[0, 0].tolist() == [0, 2, 1, 3], \
            f"got {nbr[0, 0].tolist()}, expected closest-first [0, 2, 1, 3]"

        # Generalized: distances along the neighbor axis should be non-decreasing.
        cent_coords = points[0, cent[0, 0]]
        nbr_coords = points[0, nbr[0, 0]]
        dists = ((nbr_coords - cent_coords) ** 2).sum(dim=-1)
        assert torch.all(dists[1:] >= dists[:-1]), f"neighbors not sorted: {dists}"


class TestNaNHandling:
    """NaN-coordinate points must be treated as invalid and never selected."""

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_nan_point_never_selected(self, device):
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(0)
        points = torch.randn(1, 10, 3, device=device)
        points[0, 5] = float('nan')
        mask = torch.ones(1, 10, dtype=torch.bool, device=device)
        start = torch.zeros(1, dtype=torch.long, device=device)
        idx = farthest_point_sampling(points, mask, K=6, start_idx=start, random_start=False)
        assert 5 not in idx[0].tolist(), f"NaN index selected: {idx[0].tolist()}"
        assert len(set(idx[0].tolist())) == 6


class TestImportErrorChain:
    """Cover ImportError-swallowing bug — chain must preserve the original cause."""

    def test_cause_is_chained(self):
        import torch_fps.fps as fps_mod
        saved_c = fps_mod._C
        saved_err = fps_mod._import_error
        fps_mod._C = None
        fps_mod._import_error = ImportError("undefined symbol: _Z_fake_test")
        try:
            with pytest.raises(RuntimeError) as info:
                farthest_point_sampling(
                    torch.randn(1, 5, 3), torch.ones(1, 5, dtype=torch.bool), K=2
                )
            assert isinstance(info.value.__cause__, ImportError)
            assert "undefined symbol" in str(info.value.__cause__)
        finally:
            fps_mod._C = saved_c
            fps_mod._import_error = saved_err


if __name__ == "__main__":
    # Run tests without pytest
    print("Running FPS correctness tests...")

    test_fps = TestFPS()
    for device in ["cpu", "cuda"]:
        if device == "cuda" and not torch.cuda.is_available():
            print(f"  Skipping {device} (not available)")
            continue
        print(f"  Testing FPS on {device}...")
        test_fps.test_fps_correctness(device, 4, 100, 4, 16)
        test_fps.test_fps_with_masking(device)
        test_fps.test_fps_edge_cases(device)

    print("\nRunning FPS+kNN correctness tests...")
    test_fused = TestFPSWithKNN()
    for device in ["cpu", "cuda"]:
        if device == "cuda" and not torch.cuda.is_available():
            print(f"  Skipping {device} (not available)")
            continue
        print(f"  Testing FPS+kNN on {device}...")
        test_fused.test_fused_correctness(device, 4, 100, 4, 16, 8)
        test_fused.test_fused_with_masking(device)
        test_fused.test_fused_edge_cases(device)

    print("\nRunning determinism tests...")
    test_det = TestDeterminism()
    for device in ["cpu", "cuda"]:
        if device == "cuda" and not torch.cuda.is_available():
            continue
        print(f"  Testing determinism on {device}...")
        test_det.test_fps_determinism(device)
        test_det.test_fused_determinism(device)

    print("\n✓ All correctness tests passed!")
