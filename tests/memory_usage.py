#!/usr/bin/env python3
"""
Memory usage test for torch-fps with different dtypes.
Tests peak CUDA memory usage across fp16, bf16, fp32, and fp64.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch_fps
import gc


def measure_memory(func, *args, **kwargs):
    """Measure peak GPU memory usage for a function call."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    result = func(*args, **kwargs)
    peak_memory = torch.cuda.max_memory_allocated()

    del result
    torch.cuda.empty_cache()
    gc.collect()

    return peak_memory


def test_fps_memory(dtype, B=4, N=10000, D=4, K=128):
    """Test FPS memory usage for a given dtype."""
    device = torch.device('cuda')

    # Create inputs at fp32 (precision param controls computation dtype)
    points = torch.randn(B, N, D, device=device, dtype=torch.float32)
    mask = torch.ones(B, N, device=device, dtype=torch.bool)

    # Measure memory with specified precision
    peak_mem = measure_memory(
        torch_fps.farthest_point_sampling,
        points, mask, K, random_start=False, precision=dtype
    )

    return peak_mem


def test_fps_with_knn_memory(dtype, B=4, N=10000, D=4, K=128, k=16):
    """Test FPS+kNN memory usage for a given dtype."""
    device = torch.device('cuda')

    # Create inputs at fp32 (precision param controls computation dtype)
    points = torch.randn(B, N, D, device=device, dtype=torch.float32)
    mask = torch.ones(B, N, device=device, dtype=torch.bool)

    # Measure memory with specified precision
    peak_mem = measure_memory(
        torch_fps.farthest_point_sampling_with_knn,
        points, mask, K, k, random_start=False, precision=dtype
    )

    return peak_mem


def format_bytes(num_bytes):
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory tests")
        return

    dtypes = [
        (torch.float16, 'float16'),
        (torch.bfloat16, 'bfloat16'),
        (torch.float32, 'float32'),
        (torch.float64, 'float64'),
    ]

    # Test configuration
    B, N, D, K, k = 4, 10000, 4, 128, 16

    print("=" * 70)
    print(f"Memory Usage Test: torch-fps")
    print(f"Configuration: B={B}, N={N}, D={D}, K={K}, k={k}")
    print("=" * 70)

    # Test FPS
    print("\n1. FPS (without kNN)")
    print("-" * 70)
    print(f"{'dtype':<12} {'Peak Memory':<15} {'vs fp64':<15} {'vs fp32':<15}")
    print("-" * 70)

    fps_results = {}
    for dtype, name in dtypes:
        try:
            peak_mem = test_fps_memory(dtype, B, N, D, K)
            fps_results[name] = peak_mem

            vs_fp64 = f"-{(1 - peak_mem/fps_results.get('float64', peak_mem))*100:.1f}%" if 'float64' in fps_results else "baseline"
            vs_fp32 = f"-{(1 - peak_mem/fps_results.get('float32', peak_mem))*100:.1f}%" if 'float32' in fps_results else "baseline"

            print(f"{name:<12} {format_bytes(peak_mem):<15} {vs_fp64:<15} {vs_fp32:<15}")
        except Exception as e:
            print(f"{name:<12} ERROR: {e}")

    # Test FPS with kNN
    print("\n2. FPS with kNN (creates all_dists tensor)")
    print("-" * 70)
    print(f"{'dtype':<12} {'Peak Memory':<15} {'vs fp64':<15} {'vs fp32':<15}")
    print("-" * 70)

    knn_results = {}
    for dtype, name in dtypes:
        try:
            peak_mem = test_fps_with_knn_memory(dtype, B, N, D, K, k)
            knn_results[name] = peak_mem

            vs_fp64 = f"-{(1 - peak_mem/knn_results.get('float64', peak_mem))*100:.1f}%" if 'float64' in knn_results else "baseline"
            vs_fp32 = f"-{(1 - peak_mem/knn_results.get('float32', peak_mem))*100:.1f}%" if 'float32' in knn_results else "baseline"

            print(f"{name:<12} {format_bytes(peak_mem):<15} {vs_fp64:<15} {vs_fp32:<15}")
        except Exception as e:
            print(f"{name:<12} ERROR: {e}")

    # Summary
    if 'bfloat16' in knn_results and 'float64' in knn_results:
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        savings_vs_fp64 = (1 - knn_results['bfloat16']/knn_results['float64']) * 100
        savings_vs_fp32 = (1 - knn_results['bfloat16']/knn_results['float32']) * 100
        print(f"bfloat16 saves {savings_vs_fp64:.1f}% memory vs float64 (old default)")
        print(f"bfloat16 saves {savings_vs_fp32:.1f}% memory vs float32")
        print(f"Memory reduction: {format_bytes(knn_results['float64'])} → {format_bytes(knn_results['bfloat16'])}")


if __name__ == '__main__':
    main()
