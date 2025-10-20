# torch-fps

Optimized farthest point sampling (FPS) for PyTorch. CPU path in C++, CUDA path fused. Call it with batched point clouds shaped `[B, N, D]`, boolean masks `[B, N]`, and an integer `K` samples per batch (≤ `N`).

```bash
pip install torch-fps
```

```python
from torch_fps import farthest_point_sampling

idx, valid = farthest_point_sampling(points, mask, K)
```

`points`: `[B, N, D]` float tensor, `mask`: `[B, N]` bool tensor, `K`: samples per batch.

[MIT](LICENSE)
