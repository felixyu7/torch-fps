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

[MIT](LICENSE)
