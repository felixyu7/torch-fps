# torch-fps

Optimized standard farthest point sampling (FPS) for PyTorch written in C++.

```bash
pip install torch-fps
```

## Usage

```python
import torch
from torch_fps import farthest_point_sampling

# Create example inputs
points = torch.randn(4, 1000, 3)     # [B, N, D] - batch of point clouds
mask = torch.ones(4, 1000, dtype=torch.bool)  # [B, N] - valid point mask
K = 512  # Number of samples per batch (must be <= number of valid points)

# Perform farthest point sampling
idx = farthest_point_sampling(points, mask, K)  # [B, K] - selected point indices

# Use indices to gather sampled points
sampled_points = points.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))  # [B, K, D]
```

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

[MIT](LICENSE)
