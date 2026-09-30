"""Farthest point sampling for PyTorch: Triton on CUDA, C++ on CPU."""
from .api import (
    farthest_point_sampling,
    farthest_point_sampling_with_assign,
    farthest_point_sampling_with_knn,
    nearest_assign,
)

__version__ = "0.6.0"

__all__ = [
    "farthest_point_sampling",
    "farthest_point_sampling_with_knn",
    "farthest_point_sampling_with_assign",
    "nearest_assign",
    "__version__",
]
