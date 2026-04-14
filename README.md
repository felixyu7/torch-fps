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
| 4  | 100  | 20  | 0.45 / 1.38     | 0.05 / 0.10    | 9.50x / 13.70x |
| 8  | 512  | 64  | 2.88 / 4.05     | 0.11 / 0.17    | 25.36x / 23.89x |
| 16 | 1024 | 128 | 29.92 / 7.81    | 0.39 / 0.31    | 77.57x / 25.42x |
| 32 | 2048 | 256 | 154.44 / 15.59  | 1.52 / 0.74    | 101.92x / 21.16x |

**FPS+kNN:**

| B  | N    | K   | k  | Baseline (ms)   | Optimized (ms) | Speedup        |
|---:|-----:|----:|---:|----------------:|---------------:|---------------:|
| 4  | 100  | 16  | 8  | 0.50 / 1.21     | 0.05 / 0.21    | 9.98x / 5.80x  |
| 8  | 512  | 64  | 16 | 4.90 / 4.11     | 0.21 / 1.13    | 23.56x / 3.65x |
| 16 | 1024 | 128 | 16 | 37.60 / 8.08    | 0.80 / 2.24    | 46.88x / 3.60x |
| 32 | 2048 | 256 | 16 | 180.33 / 16.86  | 2.57 / 4.84    | 70.24x / 3.48x |
