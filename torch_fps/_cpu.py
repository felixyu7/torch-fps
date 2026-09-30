"""Lazy just-in-time loader for the C++ CPU kernels (FPS and fused FPS + kNN).

The sources under csrc/ compile on first CPU use via
torch.utils.cpp_extension.load and are cached in ~/.cache/torch_extensions
(or $TORCH_EXTENSIONS_DIR). Nothing is compiled at install time. Without a
working compiler, get_module() warns once and returns None, and api.py falls
back to the pure-PyTorch reference.
"""
from __future__ import annotations

import os
import sys
import threading
import warnings
from pathlib import Path

_lock = threading.Lock()
_module = None
_state = "unloaded"  # unloaded | ready | failed


def _openmp_flags():
    """(cflags, ldflags) enabling OpenMP for at::parallel_for.

    ATen's OpenMP parallel_for is header-inline, so without these flags the
    extension silently runs serially over the batch. macOS toolchains often
    lack libomp, so there it is opt-in (TORCH_FPS_OPENMP=1); TORCH_FPS_OPENMP=0
    disables it everywhere.
    """
    env = os.environ.get("TORCH_FPS_OPENMP")
    if env == "0" or (sys.platform == "darwin" and env != "1"):
        return [], []
    if sys.platform == "win32":
        return ["/openmp"], []
    if sys.platform == "darwin":
        return ["-Xpreprocessor", "-fopenmp"], ["-lomp"]
    return ["-fopenmp"], ["-fopenmp"]


def _build():
    from torch.utils.cpp_extension import load

    csrc = Path(__file__).resolve().parent / "csrc"
    omp_cflags, omp_ldflags = _openmp_flags()
    # torch adds its own -std flag (C++17 or C++20 depending on the version)
    return load(
        name="torch_fps_cpu",
        sources=[str(csrc / "fps_cpu.cpp"), str(csrc / "binding_cpu.cpp")],
        extra_cflags=(["/O2"] if sys.platform == "win32" else ["-O3"]) + omp_cflags,
        extra_ldflags=omp_ldflags,
        verbose=False,
    )


def get_module():
    """Return the compiled CPU extension, or None if it cannot be built."""
    global _module, _state
    if _state == "ready":
        return _module
    if _state == "failed":
        return None
    with _lock:
        if _state == "unloaded":
            try:
                _module = _build()
                _state = "ready"
            except Exception as exc:  # noqa: BLE001 (any build failure)
                _state = "failed"
                warnings.warn(
                    "torch_fps: the C++ CPU extension could not be built "
                    f"({exc}); falling back to the slower pure-PyTorch "
                    "backend. Install a C++ compiler for fast CPU FPS.",
                    RuntimeWarning,
                    stacklevel=3,
                )
    return _module if _state == "ready" else None
