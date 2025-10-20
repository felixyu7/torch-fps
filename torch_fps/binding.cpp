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

#ifdef WITH_CUDA
at::Tensor fps_forward_cuda(
    const at::Tensor& points,
    const at::Tensor& mask,
    const at::Tensor& start_idx,
    int64_t K);

std::tuple<at::Tensor, at::Tensor> fps_with_knn_forward_cuda(
    const at::Tensor& points,
    const at::Tensor& mask,
    const at::Tensor& start_idx,
    int64_t K,
    int64_t k_neighbors);
#endif

at::Tensor fps_forward(
    const at::Tensor& points,
    const at::Tensor& mask,
    const at::Tensor& start_idx,
    int64_t K) {
    TORCH_CHECK(points.device() == mask.device(),
                "points and mask tensors must be on the same device");
    TORCH_CHECK(points.device() == start_idx.device(),
                "start_idx tensor must be on the same device as points");

    if (points.is_cuda()) {
#ifdef WITH_CUDA
        return fps_forward_cuda(points, mask, start_idx, K);
#else
        TORCH_CHECK(false, "torch-fps was built without CUDA support");
#endif
    }

    return fps_forward_cpu(points, mask, start_idx, K);
}

std::tuple<at::Tensor, at::Tensor> fps_with_knn_forward(
    const at::Tensor& points,
    const at::Tensor& mask,
    const at::Tensor& start_idx,
    int64_t K,
    int64_t k_neighbors) {
    TORCH_CHECK(points.device() == mask.device(),
                "points and mask tensors must be on the same device");
    TORCH_CHECK(points.device() == start_idx.device(),
                "start_idx tensor must be on the same device as points");

    if (points.is_cuda()) {
#ifdef WITH_CUDA
        return fps_with_knn_forward_cuda(points, mask, start_idx, K, k_neighbors);
#else
        TORCH_CHECK(false, "torch-fps was built without CUDA support");
#endif
    }

    return fps_with_knn_forward_cpu(points, mask, start_idx, K, k_neighbors);
}

}  // namespace torch_fps

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fps_forward",
        &torch_fps::fps_forward,
        "Farthest point sampling forward pass",
        pybind11::arg("points"),
        pybind11::arg("mask"),
        pybind11::arg("start_idx"),
        pybind11::arg("K"));

    m.def(
        "fps_with_knn_forward",
        &torch_fps::fps_with_knn_forward,
        "Fused farthest point sampling + k-nearest neighbors forward pass",
        pybind11::arg("points"),
        pybind11::arg("mask"),
        pybind11::arg("start_idx"),
        pybind11::arg("K"),
        pybind11::arg("k_neighbors"));
}

