# torch-fps

Optimized standard farthest point sampling (FPS) for PyTorch written in C++.

```bash
pip install torch-fps
```
**Note**: Ensure gcc > 9 and < 14. Install might take a while since its building from source (to be fixed in future). 

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

**FPS:**

| B  | N    | K   | Baseline (ms)   | Optimized (ms) | Speedup        |
|---:|-----:|----:|----------------:|---------------:|---------------:|
| 4  | 100  | 20  | 0.42 / 1.39     | 0.05 / 0.24    | 8.05x / 5.83x  |
| 8  | 512  | 64  | 2.75 / 4.01     | 0.63 / 0.31    | 4.36x / 13.11x |
| 16 | 1024 | 128 | 34.52 / 7.78    | 4.54 / 0.45    | 7.61x / 17.34x |
| 32 | 2048 | 256 | 153.27 / 15.41  | 36.48 / 0.89   | 4.20x / 17.34x |

**FPS+kNN:**

| B  | N    | K   | k  | Baseline (ms)   | Optimized (ms) | Speedup        |
|---:|-----:|----:|---:|----------------:|---------------:|---------------:|
| 4  | 100  | 16  | 8  | 0.42 / 1.22     | 0.07 / 0.35    | 6.00x / 3.50x  |
| 8  | 512  | 64  | 16 | 4.14 / 4.43     | 2.01 / 1.25    | 2.06x / 3.55x  |
| 16 | 1024 | 128 | 16 | 38.37 / 8.02    | 11.69 / 2.34   | 3.28x / 3.43x  |
| 32 | 2048 | 256 | 16 | 177.50 / 16.62  | 78.47 / 4.94   | 2.26x / 3.36x  |
