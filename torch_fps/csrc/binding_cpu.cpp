// CPU-only pybind binding for the C++ FPS kernels in fps_cpu.cpp. Built
// just-in-time by torch_fps/_cpu.py; device dispatch lives in api.py.
#include <torch/extension.h>

#include <cstdint>

namespace torch_fps {

at::Tensor fps_forward_cpu(
    const at::Tensor& points,
    const at::Tensor& mask,
    const at::Tensor& start_idx,
    int64_t K);

std::tuple<at::Tensor, at::Tensor> fps_with_knn_forward_cpu(
    const at::Tensor& points,
    const at::Tensor& mask,
    const at::Tensor& start_idx,
    int64_t K,
    int64_t k_neighbors);

}  // namespace torch_fps

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Release the GIL: bodies are pure ATen (multi-ms at::parallel_for loops
    // would otherwise block all Python threads).
    m.def(
        "fps_forward",
        &torch_fps::fps_forward_cpu,
        "Farthest point sampling forward pass (CPU)",
        pybind11::arg("points"),
        pybind11::arg("mask"),
        pybind11::arg("start_idx"),
        pybind11::arg("K"),
        pybind11::call_guard<pybind11::gil_scoped_release>());

    m.def(
        "fps_with_knn_forward",
        &torch_fps::fps_with_knn_forward_cpu,
        "Fused farthest point sampling + k-nearest neighbors forward pass (CPU)",
        pybind11::arg("points"),
        pybind11::arg("mask"),
        pybind11::arg("start_idx"),
        pybind11::arg("K"),
        pybind11::arg("k_neighbors"),
        pybind11::call_guard<pybind11::gil_scoped_release>());
}
