"""
Performance profiling for FPS and FPS+kNN implementations.

Compares optimized kernels against baselines across various problem sizes.
"""

import time
import torch
from typing import List, Tuple
from torch_fps import farthest_point_sampling, farthest_point_sampling_with_knn
from baselines import fps_baseline, fps_with_knn_baseline


# ============================================================================
# Profiling Utilities
# ============================================================================

def benchmark_function(func, *args, warmup=3, iterations=10, sync_cuda=True):
    """
    Benchmark a function with warmup and multiple iterations.

    Args:
        func: Function to benchmark
        *args: Arguments to pass to func
        warmup: Number of warmup iterations
        iterations: Number of timed iterations
        sync_cuda: Whether to synchronize CUDA before timing

    Returns:
        Average time in milliseconds
    """
    # Warmup
    for _ in range(warmup):
        _ = func(*args)
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    # Timed runs
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        _ = func(*args)
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (elapsed / iterations) * 1000  # Convert to ms


# ============================================================================
# FPS Profiling
# ============================================================================

def profile_fps(device: str = "cuda", verbose: bool = True):
    """
    Profile FPS performance: optimized vs baseline.

    Tests various problem sizes: (B, N, K).
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, skipping CUDA profiling")
        return

    configs = [
        ("Small", 4, 100, 20),
        ("Medium", 8, 512, 64),
        ("Large", 16, 1024, 128),
        ("XLarge", 32, 2048, 256),
    ]

    if verbose:
        print(f"\n{'='*70}")
        print(f"FPS Performance Profiling ({device.upper()})")
        print(f"{'='*70}")
        print(f"{'Config':<12} {'B':>4} {'N':>6} {'K':>5} {'Baseline':>12} {'Optimized':>12} {'Speedup':>10}")
        print(f"{'-'*70}")

    results = []

    for name, B, N, K in configs:
        D = 4  # Common for spatiotemporal data
        points = torch.randn(B, N, D, device=device)
        mask = torch.ones(B, N, dtype=torch.bool, device=device)
        start_idx = torch.zeros(B, dtype=torch.long, device=device)

        # Benchmark baseline
        t_baseline = benchmark_function(
            fps_baseline, points, mask, K, start_idx,
            sync_cuda=(device == "cuda")
        )

        # Benchmark optimized
        def fps_wrapper():
            return farthest_point_sampling(points, mask, K, start_idx=start_idx, random_start=False)

        t_optimized = benchmark_function(
            fps_wrapper,
            sync_cuda=(device == "cuda")
        )

        speedup = t_baseline / t_optimized
        results.append((name, B, N, K, t_baseline, t_optimized, speedup))

        if verbose:
            print(f"{name:<12} {B:>4} {N:>6} {K:>5} "
                  f"{t_baseline:>10.2f} ms {t_optimized:>10.2f} ms {speedup:>9.2f}x")

    if verbose:
        print(f"{'='*70}\n")

    return results


# ============================================================================
# FPS+kNN Profiling
# ============================================================================

def profile_fps_with_knn(device: str = "cuda", verbose: bool = True):
    """
    Profile FPS+kNN performance: fused vs baseline (separate operations).

    Tests various problem sizes: (B, N, K, k).
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, skipping CUDA profiling")
        return

    configs = [
        ("Small", 4, 100, 16, 8),
        ("Medium", 8, 512, 64, 16),
        ("Large", 16, 1024, 128, 16),
        ("XLarge", 32, 2048, 256, 16),
    ]

    if verbose:
        print(f"\n{'='*80}")
        print(f"FPS+kNN Performance Profiling ({device.upper()})")
        print(f"{'='*80}")
        print(f"{'Config':<12} {'B':>4} {'N':>6} {'K':>5} {'k':>4} "
              f"{'Baseline':>12} {'Fused':>12} {'Speedup':>10}")
        print(f"{'-'*80}")

    results = []

    for name, B, N, K, k in configs:
        D = 4
        points = torch.randn(B, N, D, device=device)
        mask = torch.ones(B, N, dtype=torch.bool, device=device)
        start_idx = torch.zeros(B, dtype=torch.long, device=device)

        # Benchmark baseline (separate FPS + kNN)
        t_baseline = benchmark_function(
            fps_with_knn_baseline, points, mask, K, k, start_idx,
            sync_cuda=(device == "cuda")
        )

        # Benchmark fused
        def fused_wrapper():
            return farthest_point_sampling_with_knn(points, mask, K, k, start_idx=start_idx, random_start=False)

        t_fused = benchmark_function(
            fused_wrapper,
            sync_cuda=(device == "cuda")
        )

        speedup = t_baseline / t_fused
        results.append((name, B, N, K, k, t_baseline, t_fused, speedup))

        if verbose:
            print(f"{name:<12} {B:>4} {N:>6} {K:>5} {k:>4} "
                  f"{t_baseline:>10.2f} ms {t_fused:>10.2f} ms {speedup:>9.2f}x")

    if verbose:
        print(f"{'='*80}\n")

    return results


# ============================================================================
# Main Profiling Suite
# ============================================================================

def run_all_profiles():
    """Run complete profiling suite."""
    print("\n" + "="*80)
    print("TORCH-FPS PERFORMANCE PROFILING SUITE")
    print("="*80)

    # FPS profiling
    print("\n[1/4] FPS Performance (CPU)")
    profile_fps(device="cpu")

    print("\n[2/4] FPS+kNN Performance (CPU)")
    profile_fps_with_knn(device="cpu")

    if torch.cuda.is_available():
        print("\n[3/4] FPS Performance (CUDA)")
        profile_fps(device="cuda")

        print("\n[4/4] FPS+kNN Performance (CUDA)")
        profile_fps_with_knn(device="cuda")
    else:
        print("\nCUDA not available - skipping GPU profiling")

    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_all_profiles()
