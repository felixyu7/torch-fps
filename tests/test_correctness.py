"""
Correctness tests for FPS and FPS+kNN implementations.

Validates optimized kernels against pure PyTorch baselines.
"""

import torch
import pytest
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

        # Optimized version
        idx_opt = farthest_point_sampling(
            points, mask, K, start_idx=start_idx, random_start=False
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
            points, mask, K, start_idx=start_idx, random_start=False
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
        idx_opt = farthest_point_sampling(points, mask, 1, start_idx=start_idx, random_start=False)
        idx_base = fps_baseline(points, mask, 1, start_idx)
        assert torch.equal(idx_opt, idx_base)

        # K = N
        idx_opt = farthest_point_sampling(points, mask, N, start_idx=start_idx, random_start=False)
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

        # Fused version
        centroid_idx_fused, neighbor_idx_fused = farthest_point_sampling_with_knn(
            points, mask, K, k, start_idx=start_idx, random_start=False
        )

        # Baseline (separate FPS + kNN)
        centroid_idx_base, neighbor_idx_base = fps_with_knn_baseline(
            points, mask, K, k, start_idx
        )

        assert torch.equal(centroid_idx_fused, centroid_idx_base), \
            f"Centroid indices mismatch on {device}"
        # Sort neighbor indices before comparing (optimized version may return unsorted)
        neighbor_idx_fused_sorted = torch.sort(neighbor_idx_fused, dim=-1)[0]
        neighbor_idx_base_sorted = torch.sort(neighbor_idx_base, dim=-1)[0]
        assert torch.equal(neighbor_idx_fused_sorted, neighbor_idx_base_sorted), \
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
            points, mask, K, k, start_idx=start_idx, random_start=False
        )
        centroid_idx_base, neighbor_idx_base = fps_with_knn_baseline(
            points, mask, K, k, start_idx
        )

        assert torch.equal(centroid_idx_fused, centroid_idx_base)
        # Sort neighbor indices before comparing (optimized version may return unsorted)
        neighbor_idx_fused_sorted = torch.sort(neighbor_idx_fused, dim=-1)[0]
        neighbor_idx_base_sorted = torch.sort(neighbor_idx_base, dim=-1)[0]
        assert torch.equal(neighbor_idx_fused_sorted, neighbor_idx_base_sorted)

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
            points, mask, K, 1, start_idx=start_idx, random_start=False
        )
        centroid_idx_base, neighbor_idx_base = fps_with_knn_baseline(
            points, mask, K, 1, start_idx
        )
        assert torch.equal(centroid_idx_fused, centroid_idx_base)
        # Sort neighbor indices before comparing (optimized version may return unsorted)
        neighbor_idx_fused_sorted = torch.sort(neighbor_idx_fused, dim=-1)[0]
        neighbor_idx_base_sorted = torch.sort(neighbor_idx_base, dim=-1)[0]
        assert torch.equal(neighbor_idx_fused_sorted, neighbor_idx_base_sorted)
        assert neighbor_idx_fused.shape == (B, K, 1)

        # k = N (maximum neighbors)
        centroid_idx_fused, neighbor_idx_fused = farthest_point_sampling_with_knn(
            points, mask, K, N, start_idx=start_idx, random_start=False
        )
        centroid_idx_base, neighbor_idx_base = fps_with_knn_baseline(
            points, mask, K, N, start_idx
        )
        assert torch.equal(centroid_idx_fused, centroid_idx_base)
        # Sort neighbor indices before comparing (optimized version may return unsorted)
        neighbor_idx_fused_sorted = torch.sort(neighbor_idx_fused, dim=-1)[0]
        neighbor_idx_base_sorted = torch.sort(neighbor_idx_base, dim=-1)[0]
        assert torch.equal(neighbor_idx_fused_sorted, neighbor_idx_base_sorted)
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
        idx1 = farthest_point_sampling(points, mask, K, generator=gen1)

        # Run 2 (same seed)
        gen2 = torch.Generator(device=device).manual_seed(123)
        idx2 = farthest_point_sampling(points, mask, K, generator=gen2)

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
        cent1, neigh1 = farthest_point_sampling_with_knn(points, mask, K, k, generator=gen1)

        # Run 2 (same seed)
        gen2 = torch.Generator(device=device).manual_seed(123)
        cent2, neigh2 = farthest_point_sampling_with_knn(points, mask, K, k, generator=gen2)

        assert torch.equal(cent1, cent2), "Centroids not deterministic"
        assert torch.equal(neigh1, neigh2), "Neighbors not deterministic"


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
