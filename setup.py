from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup


def get_extensions():
    try:
        from torch.utils.cpp_extension import (
            BuildExtension,
            CppExtension,
            CUDAExtension,
        )
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Building torch-fps requires PyTorch to be installed as a build dependency."
        ) from exc

    sources = [
        "torch_fps/binding.cpp",
        "torch_fps/fps_cpu.cpp",
    ]

    include_dirs = ["torch_fps"]

    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"],
    }

    if sys.platform == "win32":
        extra_compile_args["cxx"].append("/EHsc")
    elif sys.platform == "darwin":
        extra_compile_args["cxx"].append("-stdlib=libc++")
        if "MACOSX_DEPLOYMENT_TARGET" not in os.environ:
            os.environ["MACOSX_DEPLOYMENT_TARGET"] = "10.14"
        try:
            sdk_path = subprocess.check_output(
                ["xcrun", "--sdk", "macosx", "--show-sdk-path"],
                text=True,
            ).strip()
            if sdk_path:
                extra_compile_args["cxx"].extend(["-isysroot", sdk_path])
                sdk_cxx = Path(sdk_path) / "usr" / "include" / "c++" / "v1"
                if sdk_cxx.exists():
                    extra_compile_args["cxx"].append(f"-I{sdk_cxx}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    extensions = []

    use_cuda = torch.utils.cpp_extension.CUDA_HOME is not None
    if use_cuda:
        cuda_sources = sources + ["torch_fps/fps_cuda.cu"]
        extra_compile_args["cxx"].append("-DWITH_CUDA")
        extra_compile_args["nvcc"] = ["-O3", "-DWITH_CUDA"]
        if sys.platform == "darwin":
            extra_compile_args["nvcc"].extend(["-Xcompiler", "-stdlib=libc++"])

        # Configure CUDA architectures for multi-GPU support
        cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
        if cuda_arch_list:
            # User specified architectures
            arch_list = cuda_arch_list.split(";") if ";" in cuda_arch_list else cuda_arch_list.split()
        else:
            # Default: support modern architectures
            arch_list = ["8.0", "8.6", "8.9", "9.0", "10.0", "12.0"]

        for arch in arch_list:
            arch_clean = arch.replace(".", "")
            # Generate both compute and SM code for maximum compatibility
            extra_compile_args["nvcc"].extend([
                f"-gencode", f"arch=compute_{arch_clean},code=sm_{arch_clean}"
            ])
        # Add PTX for forward compatibility with future architectures
        last_arch = arch_list[-1].replace(".", "")
        extra_compile_args["nvcc"].extend([
            f"-gencode", f"arch=compute_{last_arch},code=compute_{last_arch}"
        ])
        extensions.append(
            CUDAExtension(
                name="torch_fps._C",
                sources=cuda_sources,
                include_dirs=include_dirs,
                extra_compile_args=extra_compile_args,
            )
        )
    else:
        extensions.append(
            CppExtension(
                name="torch_fps._C",
                sources=sources,
                include_dirs=include_dirs,
                extra_compile_args=extra_compile_args,
            )
        )

    return extensions, BuildExtension


ext_modules, build_ext = get_extensions()

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
