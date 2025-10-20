#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Parallel.h>
#include <torch/extension.h>

#include <c10/util/Exception.h>

#include <algorithm>
#include <limits>
#include <vector>

namespace torch_fps {
namespace {

template <typename scalar_t, typename acc_t>
void fps_kernel_cpu(
    const scalar_t* points,
    const bool* mask,
    int64_t N,
    int64_t D,
    int64_t K,
    int64_t start,
    int64_t* out_indices) {
    std::vector<acc_t> min_dists(static_cast<size_t>(N));

    int64_t valid_count = 0;
    const acc_t inf = std::numeric_limits<acc_t>::infinity();
    const acc_t neg_inf = -std::numeric_limits<acc_t>::infinity();

    for (int64_t n = 0; n < N; ++n) {
        if (mask[n]) {
            min_dists[n] = inf;
            ++valid_count;
        } else {
            min_dists[n] = acc_t(0);
        }
    }

    const int64_t effective_k = std::min<int64_t>(valid_count, K);
    int64_t last = (start >= 0 && start < N) ? start : 0;

    if (K == 0) {
        return;
    }

    if (effective_k == 0) {
        std::fill(out_indices, out_indices + K, last);
        return;
    }

    for (int64_t i = 0; i < K; ++i) {
        // Update min_dists based on distance to current centroid
        // This matches: d = (points - c[:, None, :]).square().sum(dim=2)
        //               min_dists = torch.minimum(min_dists, d)
        const scalar_t* centroid = points + last * D;

        for (int64_t n = 0; n < N; ++n) {
            if (!mask[n]) {
                continue;  // Skip distance computation for invalid points
            }

            const scalar_t* point = points + n * D;
            acc_t dist = acc_t(0);
            for (int64_t d = 0; d < D; ++d) {
                const acc_t diff =
                    static_cast<acc_t>(point[d]) - static_cast<acc_t>(centroid[d]);
                dist += diff * diff;
            }

            // Update minimum distance
            if (dist < min_dists[n]) {
                min_dists[n] = dist;
            }
        }

        // Record selection (matches: idx[:, i] = last)
        out_indices[i] = last;

        // Find next farthest point (matches: last = torch.argmax(min_dists, dim=1))
        if (i + 1 < K) {
            acc_t best_val = neg_inf;
            int64_t best_idx = 0;

            for (int64_t n = 0; n < N; ++n) {
                // argmax over all points (invalids have 0.0, selected have 0.0 after update)
                if (min_dists[n] > best_val ||
                    (min_dists[n] == best_val && n < best_idx)) {
                    best_val = min_dists[n];
                    best_idx = n;
                }
            }

            last = best_idx;
        }
    }
}

// ============================================================================
// Fused FPS + kNN kernel
// ============================================================================

template <typename scalar_t, typename acc_t>
void fps_with_knn_kernel_cpu(
    const scalar_t* points,
    const bool* mask,
    int64_t N,
    int64_t D,
    int64_t K,
    int64_t k_neighbors,
    int64_t start,
    int64_t* out_centroid_indices,
    int64_t* out_neighbor_indices) {  // [K, k_neighbors]

    std::vector<acc_t> min_dists(static_cast<size_t>(N));
    // Incremental kNN tracking: one buffer per centroid
    std::vector<std::vector<std::pair<acc_t, int64_t>>> knn_buffers(static_cast<size_t>(K));

    int64_t valid_count = 0;
    const acc_t inf = std::numeric_limits<acc_t>::infinity();
    const acc_t neg_inf = -std::numeric_limits<acc_t>::infinity();

    for (int64_t n = 0; n < N; ++n) {
        if (mask[n]) {
            min_dists[n] = inf;
            ++valid_count;
        } else {
            min_dists[n] = acc_t(0);
        }
    }

    // Pre-allocate kNN buffers
    for (int64_t i = 0; i < K; ++i) {
        knn_buffers[i].reserve(static_cast<size_t>(k_neighbors));
    }

    const int64_t effective_k = std::min<int64_t>(valid_count, K);
    int64_t last = (start >= 0 && start < N) ? start : 0;

    if (K == 0) {
        return;
    }

    if (effective_k == 0) {
        std::fill(out_centroid_indices, out_centroid_indices + K, last);
        // Fill neighbor indices with 0 (or any valid index)
        std::fill(out_neighbor_indices, out_neighbor_indices + K * k_neighbors, 0);
        return;
    }

    // FPS loop with incremental kNN tracking
    for (int64_t i = 0; i < K; ++i) {
        const scalar_t* centroid = points + last * D;
        auto& knn_buffer = knn_buffers[i];

        for (int64_t n = 0; n < N; ++n) {
            if (!mask[n]) {
                continue;  // Skip invalid points entirely
            }

            const scalar_t* point = points + n * D;
            acc_t dist = acc_t(0);
            for (int64_t d = 0; d < D; ++d) {
                const acc_t diff =
                    static_cast<acc_t>(point[d]) - static_cast<acc_t>(centroid[d]);
                dist += diff * diff;
            }

            // Incremental kNN tracking using max-heap
            // Heap property: largest distance at front
            if (knn_buffer.size() < static_cast<size_t>(k_neighbors)) {
                knn_buffer.push_back({dist, n});
                if (knn_buffer.size() == static_cast<size_t>(k_neighbors)) {
                    std::make_heap(knn_buffer.begin(), knn_buffer.end());
                }
            } else if (dist < knn_buffer.front().first) {
                // Replace the farthest neighbor with this closer one
                std::pop_heap(knn_buffer.begin(), knn_buffer.end());
                knn_buffer.back() = {dist, n};
                std::push_heap(knn_buffer.begin(), knn_buffer.end());
            }

            // Update minimum distance for FPS
            if (dist < min_dists[n]) {
                min_dists[n] = dist;
            }
        }

        // Record selection
        out_centroid_indices[i] = last;

        // Find next farthest point
        if (i + 1 < K) {
            acc_t best_val = neg_inf;
            int64_t best_idx = 0;

            for (int64_t n = 0; n < N; ++n) {
                if (min_dists[n] > best_val ||
                    (min_dists[n] == best_val && n < best_idx)) {
                    best_val = min_dists[n];
                    best_idx = n;
                }
            }

            last = best_idx;
        }
    }

    // Extract neighbor indices from kNN buffers
    for (int64_t i = 0; i < K; ++i) {
        const auto& knn_buffer = knn_buffers[i];
        const int64_t k_actual = std::min<int64_t>(
            k_neighbors,
            static_cast<int64_t>(knn_buffer.size())
        );

        // Copy neighbor indices (unsorted order is acceptable)
        for (int64_t k = 0; k < k_actual; ++k) {
            out_neighbor_indices[i * k_neighbors + k] = knn_buffer[k].second;
        }

        // Fill remaining slots if we have fewer than k_neighbors valid points
        // Use the centroid itself as fallback
        for (int64_t k = k_actual; k < k_neighbors; ++k) {
            out_neighbor_indices[i * k_neighbors + k] = out_centroid_indices[i];
        }
    }
}

}  // namespace

at::Tensor fps_forward_cpu(
    const at::Tensor& points,
    const at::Tensor& mask,
    const at::Tensor& start_idx,
    int64_t K) {
    TORCH_CHECK(points.device().is_cpu(), "points tensor must be on CPU");
    TORCH_CHECK(mask.device().is_cpu(), "mask tensor must be on CPU");
    TORCH_CHECK(start_idx.device().is_cpu(), "start_idx tensor must be on CPU");
    TORCH_CHECK(points.dim() == 3, "points tensor must have shape [B, N, D]");
    TORCH_CHECK(mask.sizes() == at::IntArrayRef({points.size(0), points.size(1)}),
                "mask tensor must have shape [B, N]");
    TORCH_CHECK(start_idx.numel() == points.size(0),
                "start_idx tensor must have shape [B]");

    TORCH_CHECK(points.scalar_type() == at::kFloat || points.scalar_type() == at::kDouble,
                "points tensor must be float32 or float64");

    TORCH_CHECK(mask.scalar_type() == at::kBool,
                "mask tensor must be boolean");
    TORCH_CHECK(K >= 0, "K must be non-negative");

    auto points_contig = points.contiguous();
    auto mask_contig = mask.contiguous();
    auto start_contig = start_idx.contiguous();

    const auto B = points_contig.size(0);
    const auto N = points_contig.size(1);
    const auto D = points_contig.size(2);

    auto idx = at::empty({B, K}, at::TensorOptions()
                                      .dtype(at::kLong)
                                      .device(points_contig.device()));

    AT_DISPATCH_FLOATING_TYPES(points_contig.scalar_type(), "fps_forward_cpu", [&] {
        using acc_t = at::acc_type<scalar_t, true>;

        const scalar_t* points_ptr = points_contig.data_ptr<scalar_t>();
        const bool* mask_ptr = mask_contig.data_ptr<bool>();
        const int64_t* start_ptr = start_contig.data_ptr<int64_t>();
        int64_t* idx_ptr = idx.data_ptr<int64_t>();

        at::parallel_for(0, B, 0, [&](int64_t begin, int64_t end) {
            for (int64_t b = begin; b < end; ++b) {
                const scalar_t* batch_points = points_ptr + b * N * D;
                const bool* batch_mask = mask_ptr + b * N;
                const int64_t start = start_ptr[b];
                int64_t* batch_idx = idx_ptr + b * K;

                fps_kernel_cpu<scalar_t, acc_t>(
                    batch_points,
                    batch_mask,
                    N,
                    D,
                    K,
                    start,
                    batch_idx);
            }
        });
    });

    return idx;
}

std::tuple<at::Tensor, at::Tensor> fps_with_knn_forward_cpu(
    const at::Tensor& points,
    const at::Tensor& mask,
    const at::Tensor& start_idx,
    int64_t K,
    int64_t k_neighbors) {
    TORCH_CHECK(points.device().is_cpu(), "points tensor must be on CPU");
    TORCH_CHECK(mask.device().is_cpu(), "mask tensor must be on CPU");
    TORCH_CHECK(start_idx.device().is_cpu(), "start_idx tensor must be on CPU");
    TORCH_CHECK(points.dim() == 3, "points tensor must have shape [B, N, D]");
    TORCH_CHECK(mask.sizes() == at::IntArrayRef({points.size(0), points.size(1)}),
                "mask tensor must have shape [B, N]");
    TORCH_CHECK(start_idx.numel() == points.size(0),
                "start_idx tensor must have shape [B]");

    TORCH_CHECK(points.scalar_type() == at::kFloat || points.scalar_type() == at::kDouble,
                "points tensor must be float32 or float64");

    TORCH_CHECK(mask.scalar_type() == at::kBool,
                "mask tensor must be boolean");
    TORCH_CHECK(K >= 0, "K must be non-negative");
    TORCH_CHECK(k_neighbors > 0, "k_neighbors must be positive");

    auto points_contig = points.contiguous();
    auto mask_contig = mask.contiguous();
    auto start_contig = start_idx.contiguous();

    const auto B = points_contig.size(0);
    const auto N = points_contig.size(1);
    const auto D = points_contig.size(2);

    TORCH_CHECK(k_neighbors <= N,
                "k_neighbors must be <= N (number of points)");

    auto centroid_idx = at::empty({B, K}, at::TensorOptions()
                                              .dtype(at::kLong)
                                              .device(points_contig.device()));

    auto neighbor_idx = at::empty({B, K, k_neighbors}, at::TensorOptions()
                                                           .dtype(at::kLong)
                                                           .device(points_contig.device()));

    AT_DISPATCH_FLOATING_TYPES(points_contig.scalar_type(), "fps_with_knn_forward_cpu", [&] {
        using acc_t = at::acc_type<scalar_t, true>;

        const scalar_t* points_ptr = points_contig.data_ptr<scalar_t>();
        const bool* mask_ptr = mask_contig.data_ptr<bool>();
        const int64_t* start_ptr = start_contig.data_ptr<int64_t>();
        int64_t* centroid_idx_ptr = centroid_idx.data_ptr<int64_t>();
        int64_t* neighbor_idx_ptr = neighbor_idx.data_ptr<int64_t>();

        at::parallel_for(0, B, 0, [&](int64_t begin, int64_t end) {
            for (int64_t b = begin; b < end; ++b) {
                const scalar_t* batch_points = points_ptr + b * N * D;
                const bool* batch_mask = mask_ptr + b * N;
                const int64_t start = start_ptr[b];
                int64_t* batch_centroid_idx = centroid_idx_ptr + b * K;
                int64_t* batch_neighbor_idx = neighbor_idx_ptr + b * K * k_neighbors;

                fps_with_knn_kernel_cpu<scalar_t, acc_t>(
                    batch_points,
                    batch_mask,
                    N,
                    D,
                    K,
                    k_neighbors,
                    start,
                    batch_centroid_idx,
                    batch_neighbor_idx);
            }
        });
    });

    return std::make_tuple(centroid_idx, neighbor_idx);
}

}  // namespace torch_fps
