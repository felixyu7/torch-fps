# torch-fps

Optimized standard farthest point sampling (FPS) for PyTorch written in C++.

## Install

`torch-fps` is currently published as a source distribution, so `pip` builds the
extension locally during install.

```bash
# First install a PyTorch build that matches your platform and CUDA version.
# Example for CUDA 12.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Then build torch-fps against that exact Torch install.
pip install --no-build-isolation torch-fps
```

**Note**: Ensure `gcc > 9` and `< 14`. Install can take a while because it is
built from source. `--no-build-isolation` is recommended so `pip` uses your
existing PyTorch install instead of creating a temporary build environment with
a different Torch/CUDA combination.

## Usage

```python
import torch
from torch_fps import farthest_point_sampling, farthest_point_sampling_with_knn

# Create example inputs
points = torch.randn(4, 1000, 3)     # [B, N, D] - batch of point clouds
mask = torch.ones(4, 1000, dtype=torch.bool)  # [B, N] - valid point mask
K = 512  # Number of samples per batch (must be <= number of valid points)

# Perform farthest point sampling
idx = farthest_point_sampling(points, mask, K)  # [B, K] - selected point indices

# Use indices to gather sampled points
sampled_points = points.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))  # [B, K, D]

# Fused FPS + kNN: get centroids and their k nearest neighbors in one pass
centroid_idx, neighbor_idx = farthest_point_sampling_with_knn(
    points, mask, K=512, k_neighbors=32
)  # centroid_idx: [B, K], neighbor_idx: [B, K, k_neighbors]
```

## Performance

Benchmarked on AMD Threadripper 7970X and NVIDIA RTX 5090. Values show CPU / CUDA measurements. By default uses float32; override with `precision=` parameter.
Numbers below come from the in-repo benchmark script (`python tests/profile.py`) against the local extension build.

**FPS:**

| B  | N    | K   | Baseline (ms)   | Optimized (ms) | Speedup        |
|---:|-----:|----:|----------------:|---------------:|---------------:|
| 4  | 100  | 20  | 0.45 / 1.48     | 0.05 / 0.10    | 9.00x / 14.80x |
| 8  | 512  | 64  | 2.98 / 4.31     | 0.12 / 0.16    | 24.83x / 26.94x |
| 16 | 1024 | 128 | 30.00 / 8.49    | 0.37 / 0.27    | 81.08x / 31.44x |
| 32 | 2048 | 256 | 151.86 / 16.77  | 1.43 / 1.01    | 106.20x / 16.60x |

**FPS+kNN:**

| B  | N    | K   | k  | Baseline (ms)   | Optimized (ms) | Speedup        |
|---:|-----:|----:|---:|----------------:|---------------:|---------------:|
| 4  | 100  | 16  | 8  | 0.50 / 1.28     | 0.05 / 0.21    | 10.00x / 6.10x |
| 8  | 512  | 64  | 16 | 4.97 / 4.40     | 0.17 / 1.09    | 29.24x / 4.04x |
| 16 | 1024 | 128 | 16 | 37.27 / 8.57    | 0.79 / 2.28    | 47.18x / 3.76x |
| 32 | 2048 | 256 | 16 | 172.63 / 18.11  | 2.64 / 5.15    | 65.39x / 3.52x |
