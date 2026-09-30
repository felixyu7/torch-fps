# torch-fps

Farthest point sampling (FPS) for PyTorch, with fused FPS + kNN and FPS +
nearest-centroid assignment. On CUDA it runs Triton kernels that are compiled
just-in-time for the local GPU. On CPU it runs C++ kernels that are compiled
just-in-time on first use. Elsewhere (and for CUDA `precision=torch.float64`)
it falls back to a pure-PyTorch implementation with the same semantics.

## Install

```bash
pip install torch-fps
```

This is a pure-Python package. Nothing is compiled at install time, and the
only dependency is `torch>=2.7`. The CUDA path also needs `triton>=3.3`, which
Linux CUDA builds of PyTorch 2.7+ already ship. If CUDA is available but
Triton is not, you get one `RuntimeWarning` and the reference backend.

A C++ compiler is optional and makes CPU calls much faster. With one on the
path, the first CPU call builds the C++ backend (FPS and FPS + kNN) through
`torch.utils.cpp_extension`, which takes about 15 s once and is then cached in
`~/.cache/torch_extensions` (or `$TORCH_EXTENSIONS_DIR`). Without a compiler
you get one `RuntimeWarning` and the pure-PyTorch backend. OpenMP is used on
Linux and Windows; on macOS it is opt-in with `TORCH_FPS_OPENMP=1`, and
`TORCH_FPS_OPENMP=0` turns it off everywhere.

## API

```python
from torch_fps import (
    farthest_point_sampling,              # -> idx [B, k]
    farthest_point_sampling_with_knn,     # -> idx [B, k], neighbours [B, k, k_neighbors]
    farthest_point_sampling_with_assign,  # -> idx [B, k], assign [B, N] in [0, k)
    nearest_assign,                       # flat segmented points -> [N] in [0, k)
)

points = torch.randn(4, 1000, 3, device="cuda")   # [B, N, D]
mask = torch.ones(4, 1000, dtype=torch.bool, device="cuda")

idx = farthest_point_sampling(points, mask, 128)
idx, nbr = farthest_point_sampling_with_knn(points, mask, 128, 16)
idx, assign = farthest_point_sampling_with_assign(points, mask, 128)
```

The three FPS functions share these keyword-only options:

| Option | Default | Effect |
|-|-|-|
| `start_idx` | `None` | Explicit `[B]` first index per row. |
| `random_start` | `True` | Random valid first index. When `False`, the first valid index is used. |
| `generator` | `None` | Generator for the random start. |
| `precision` | `None` | Input dtype. `None` means float32. Distances always accumulate in fp32, or fp64 with `precision=torch.float64`. |
| `validate` | `True` | Checks `k <= valid count`, which costs one host sync. With `False` the call is sync-free and short rows pad by repeating their last pick. |
| `assume_finite` | `False` | Skips the finiteness pass over the points. |

Other semantics:

- A point is valid when its mask entry is true and all its coordinates are finite.
- Ties go to the lowest index.
- kNN neighbours are sorted closest-first and include the centroid. Rows with fewer than `k_neighbors` valid points pad with the centroid index.

All backends agree up to fp32 rounding. Triton (and C++, where the compiler
fuses multiply-adds) can occasionally resolve exact near-ties differently.

Breaking change from 0.5: the sample count argument is now `k`, not `K`.

## Performance

Measured on an RTX 5090 with torch 2.12 + cu130 and triton 3.7, using K = 128,
k_nn = 8 and float32. The script is `benchmarks/bench_fps.py` and the raw
numbers are in `benchmarks/results_rtx5090_v3.json`.

Op-level median time in ms (extra peak memory in MB), pure-PyTorch reference
vs Triton:

| B | N | assign | knn |
|-|-|-|-|
| 128 | 192 | 10.5 (48) vs **0.13** (0.3) | 10.9 (49) vs **0.36** (1) |
| 128 | 3000 | 13.9 (753) vs **0.20** (3) | 15.9 (754) vs **0.56** (3) |
| 256 | 3000 | 18.0 (1505) vs **0.35** (6) | 22.0 (1505) vs **1.07** (6) |

The reference loops over the K samples, so its cost is mostly a fixed
per-call overhead rather than growing with N.

End to end in a Neptune model (IceCube-sized events, up to 192 pulses), the
pure-PyTorch reference makes a training step 15–42% slower, less for larger
batches and models, and inference 71–94% slower.

On CPU (Threadripper 7970X, 32 torch threads, K = 128, k_nn = 8, float32;
`bench_fps.py --sections cpu`, raw numbers in `benchmarks/results_cpu_v1.json`),
median ms for the pure-PyTorch reference vs C++:

| B | N | assign | knn |
|-|-|-|-|
| 16 | 192 | 9.0 vs **0.41** | 11.3 vs **0.11** |
| 16 | 3000 | 56.9 vs **13.6** | 87.6 vs **1.7** |
| 128 | 192 | 31.1 vs **1.4** | 40.5 vs **0.89** |
| 128 | 3000 | 328 vs **269** | 378 vs **7.5** |

C++ covers FPS and FPS + kNN. The assignment step after FPS (and
`nearest_assign`) still runs the reference, which dominates `assign` at large
B × N.
