# torch-fps

Optimized standard farthest point sampling (FPS) for PyTorch written in C++.

```bash
pip install torch-fps
```
**Note**: Ensure gcc > 9 and < 14.

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

Benchmarked on AMD Threadripper 7970X and NVIDIA RTX 5090. Values show CPU / CUDA measurements.

**FPS:**

| B  | N    | K   | Baseline (ms)   | Optimized (ms) | Speedup        |
|---:|-----:|----:|----------------:|---------------:|---------------:|
| 4  | 100  | 20  | 0.45 / 1.40     | 0.05 / 0.24    | 8.5x / 5.9x    |
| 8  | 512  | 64  | 2.85 / 4.04     | 0.66 / 0.31    | 4.3x / 13.0x   |
| 16 | 1024 | 128 | 33.31 / 7.78    | 4.52 / 0.59    | 7.4x / 13.1x   |
| 32 | 2048 | 256 | 158.18 / 15.56  | 33.18 / 1.66   | 4.8x / 9.4x    |

**FPS+kNN:**

| B  | N    | K   | k  | Baseline (ms)   | Optimized (ms) | Speedup        |
|---:|-----:|----:|---:|----------------:|---------------:|---------------:|
| 4  | 100  | 16  | 8  | 0.43 / 1.21     | 0.08 / 0.24    | 5.4x / 5.0x    |
| 8  | 512  | 64  | 16 | 4.98 / 4.07     | 2.16 / 0.33    | 2.3x / 12.3x   |
| 16 | 1024 | 128 | 16 | 39.00 / 7.96    | 12.00 / 0.61   | 3.3x / 13.0x   |
| 32 | 2048 | 256 | 16 | 180.14 / 16.81  | 76.60 / 1.36   | 2.4x / 12.4x   |

## Implementation

### Farthest Point Sampling
Standard greedy algorithm maintaining minimum distances to selected centroids. Each iteration selects the point farthest from all previously selected points.

- **CPU**: Sequential selection with parallel batch processing. O(K·N·D) time, O(N) space.
- **CUDA**: Cooperative parallel reduction within thread blocks. O(K·N·D) time, O(N) space per batch.

### Fused FPS + k-Nearest Neighbors
Combines FPS and kNN by reusing distance computations from the FPS phase.

- **CPU**: Incremental heap tracking during FPS. Maintains top-k neighbors per centroid using max-heaps. O(K·N·log(k)) time, O(K·k) space.
- **CUDA**: Stores all centroid distances during FPS, then applies PyTorch's optimized topk. O(K·N·D + K·N·log(k)) time, O(K·N) space per batch.

Both implementations eliminate redundant distance calculations compared to separate FPS and kNN operations.
