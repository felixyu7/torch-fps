#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>

#include <c10/util/Exception.h>

#include <limits>
#include <type_traits>

namespace torch_fps {
namespace {

template <typename scalar_t, typename acc_t, int BLOCK_SIZE, int MAX_D>
__global__ void fps_kernel_cuda(
    const scalar_t* __restrict__ points,
    const bool* __restrict__ mask,
    const int64_t* __restrict__ start_idx,
    int64_t B,
    int64_t N,
    int64_t D,
    int64_t K,
    int64_t* __restrict__ idx,
    acc_t* __restrict__ min_dists) {
    const int b = blockIdx.x;
    if (b >= B) {
        return;
    }

    const scalar_t* batch_points = points + static_cast<int64_t>(b) * N * D;
    const bool* batch_mask = mask + static_cast<int64_t>(b) * N;
    const int64_t start = start_idx[b];
    int64_t* batch_idx = idx + static_cast<int64_t>(b) * K;
    acc_t* batch_min_dists = min_dists + static_cast<int64_t>(b) * N;

    __shared__ int64_t shared_last;
    __shared__ int64_t shared_counts[BLOCK_SIZE];
    __shared__ acc_t shared_vals[BLOCK_SIZE];
    __shared__ int64_t shared_idx[BLOCK_SIZE];

    if (threadIdx.x == 0) {
        shared_last = (start >= 0 && start < N) ? start : 0;
    }
    __syncthreads();

    const acc_t inf = std::numeric_limits<acc_t>::infinity();
    const acc_t neg_inf = -std::numeric_limits<acc_t>::infinity();

    int64_t local_count = 0;
    for (int64_t n = threadIdx.x; n < N; n += BLOCK_SIZE) {
        const bool valid = batch_mask[n];
        if (valid) {
            batch_min_dists[n] = inf;
            ++local_count;
        } else {
            batch_min_dists[n] = acc_t(0);
        }
    }

    shared_counts[threadIdx.x] = local_count;
    __syncthreads();

    for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared_counts[threadIdx.x] += shared_counts[threadIdx.x + offset];
        }
        __syncthreads();
    }

    const int64_t valid_count = shared_counts[0];
    const int64_t effective_k = valid_count < K ? valid_count : K;
    int64_t last = shared_last;

    if (K == 0) {
        return;
    }

    if (effective_k == 0) {
        if (threadIdx.x == 0) {
            for (int64_t i = 0; i < K; ++i) {
                batch_idx[i] = last;
            }
        }
        return;
    }

    for (int64_t i = 0; i < K; ++i) {
        // All threads read the current selection from shared memory
        last = shared_last;

        // Phase 1: Update min_dists based on distance to current centroid
        // This matches: d = (points - c[:, None, :]).square().sum(dim=2)
        //               min_dists = torch.minimum(min_dists, d)
        acc_t centroid_vals[MAX_D];
        for (int64_t d = 0; d < D; ++d) {
            centroid_vals[d] = static_cast<acc_t>(batch_points[last * D + d]);
        }

        for (int64_t n = threadIdx.x; n < N; n += BLOCK_SIZE) {
            if (!batch_mask[n]) {
                continue;  // Skip distance computation for invalid points
            }

            const scalar_t* point = batch_points + n * D;
            acc_t dist = acc_t(0);
            for (int64_t d = 0; d < D; ++d) {
                const acc_t diff =
                    static_cast<acc_t>(point[d]) - centroid_vals[d];
                dist += diff * diff;
            }

            // Update minimum distance
            if (dist < batch_min_dists[n]) {
                batch_min_dists[n] = dist;
            }
        }
        __syncthreads();

        // Record selection (matches: idx[:, i] = last)
        if (threadIdx.x == 0) {
            batch_idx[i] = last;
        }
        __syncthreads();

        // Find next farthest point (matches: last = torch.argmax(min_dists, dim=1))
        if (i + 1 < K) {
            // Phase 2: Each thread finds local argmax over its subset
            acc_t best_val = neg_inf;
            int64_t best_idx = 0;

            for (int64_t n = threadIdx.x; n < N; n += BLOCK_SIZE) {
                // argmax over all points (invalids have 0.0, selected have 0.0 after update)
                const acc_t val = batch_min_dists[n];
                if (val > best_val || (val == best_val && n < best_idx)) {
                    best_val = val;
                    best_idx = n;
                }
            }

            shared_vals[threadIdx.x] = best_val;
            shared_idx[threadIdx.x] = best_idx;
            __syncthreads();

            // Deterministic parallel reduction: always prefer lower index on ties
            for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
                if (threadIdx.x < offset) {
                    const acc_t other_val = shared_vals[threadIdx.x + offset];
                    const int64_t other_idx = shared_idx[threadIdx.x + offset];
                    const acc_t current_val = shared_vals[threadIdx.x];
                    const int64_t current_idx = shared_idx[threadIdx.x];

                    // Deterministic tie-breaking: prefer lower index
                    if (other_val > current_val ||
                        (other_val == current_val && other_idx < current_idx)) {
                        shared_vals[threadIdx.x] = other_val;
                        shared_idx[threadIdx.x] = other_idx;
                    }
                }
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                last = shared_idx[0];
                shared_last = last;
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// Fused FPS + kNN kernel (optimized: compute distances during FPS)
// ============================================================================

template <typename scalar_t, typename acc_t, int BLOCK_SIZE, int MAX_D>
__global__ void fps_with_knn_kernel_cuda(
    const scalar_t* __restrict__ points,
    const bool* __restrict__ mask,
    const int64_t* __restrict__ start_idx,
    int64_t B,
    int64_t N,
    int64_t D,
    int64_t K,
    int64_t* __restrict__ idx,
    acc_t* __restrict__ min_dists,
    acc_t* __restrict__ all_dists) {  // Store all centroid distances [B, K, N]
    const int b = blockIdx.x;
    if (b >= B) {
        return;
    }

    const scalar_t* batch_points = points + static_cast<int64_t>(b) * N * D;
    const bool* batch_mask = mask + static_cast<int64_t>(b) * N;
    const int64_t start = start_idx[b];
    int64_t* batch_idx = idx + static_cast<int64_t>(b) * K;
    acc_t* batch_min_dists = min_dists + static_cast<int64_t>(b) * N;
    acc_t* batch_all_dists = all_dists + static_cast<int64_t>(b) * K * N;  // [K, N]

    __shared__ int64_t shared_last;
    __shared__ int64_t shared_counts[BLOCK_SIZE];
    __shared__ acc_t shared_vals[BLOCK_SIZE];
    __shared__ int64_t shared_idx[BLOCK_SIZE];

    if (threadIdx.x == 0) {
        shared_last = (start >= 0 && start < N) ? start : 0;
    }
    __syncthreads();

    const acc_t inf = std::numeric_limits<acc_t>::infinity();
    const acc_t neg_inf = -std::numeric_limits<acc_t>::infinity();

    int64_t local_count = 0;
    for (int64_t n = threadIdx.x; n < N; n += BLOCK_SIZE) {
        const bool valid = batch_mask[n];
        if (valid) {
            batch_min_dists[n] = inf;
            ++local_count;
        } else {
            batch_min_dists[n] = acc_t(0);
        }
    }

    shared_counts[threadIdx.x] = local_count;
    __syncthreads();

    for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared_counts[threadIdx.x] += shared_counts[threadIdx.x + offset];
        }
        __syncthreads();
    }

    const int64_t valid_count = shared_counts[0];
    const int64_t effective_k = valid_count < K ? valid_count : K;
    int64_t last = shared_last;

    if (K == 0) {
        return;
    }

    if (effective_k == 0) {
        if (threadIdx.x == 0) {
            for (int64_t i = 0; i < K; ++i) {
                batch_idx[i] = last;
            }
        }
        return;
    }

    for (int64_t i = 0; i < K; ++i) {
        // All threads read the current selection from shared memory
        last = shared_last;

        // Compute distances to current centroid
        acc_t centroid_vals[MAX_D];
        for (int64_t d = 0; d < D; ++d) {
            centroid_vals[d] = static_cast<acc_t>(batch_points[last * D + d]);
        }

        for (int64_t n = threadIdx.x; n < N; n += BLOCK_SIZE) {
            acc_t dist;
            if (!batch_mask[n]) {
                // Invalid points get inf distance (will be filtered in kNN)
                dist = inf;
            } else {
                const scalar_t* point = batch_points + n * D;
                dist = acc_t(0);
                for (int64_t d = 0; d < D; ++d) {
                    const acc_t diff =
                        static_cast<acc_t>(point[d]) - centroid_vals[d];
                    dist += diff * diff;
                }
            }

            // Store distance for kNN (to be processed by topk later)
            batch_all_dists[i * N + n] = dist;

            // Update minimum distance for FPS (skip invalid points)
            if (batch_mask[n] && dist < batch_min_dists[n]) {
                batch_min_dists[n] = dist;
            }
        }
        __syncthreads();

        // Record selection (matches: idx[:, i] = last)
        if (threadIdx.x == 0) {
            batch_idx[i] = last;
        }
        __syncthreads();

        // Find next farthest point (matches: last = torch.argmax(min_dists, dim=1))
        if (i + 1 < K) {
            // Phase 2: Each thread finds local argmax over its subset
            acc_t best_val = neg_inf;
            int64_t best_idx = 0;

            for (int64_t n = threadIdx.x; n < N; n += BLOCK_SIZE) {
                const acc_t val = batch_min_dists[n];
                if (val > best_val || (val == best_val && n < best_idx)) {
                    best_val = val;
                    best_idx = n;
                }
            }

            shared_vals[threadIdx.x] = best_val;
            shared_idx[threadIdx.x] = best_idx;
            __syncthreads();

            // Deterministic parallel reduction: always prefer lower index on ties
            for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
                if (threadIdx.x < offset) {
                    const acc_t other_val = shared_vals[threadIdx.x + offset];
                    const int64_t other_idx = shared_idx[threadIdx.x + offset];
                    const acc_t current_val = shared_vals[threadIdx.x];
                    const int64_t current_idx = shared_idx[threadIdx.x];

                    // Deterministic tie-breaking: prefer lower index
                    if (other_val > current_val ||
                        (other_val == current_val && other_idx < current_idx)) {
                        shared_vals[threadIdx.x] = other_val;
                        shared_idx[threadIdx.x] = other_idx;
                    }
                }
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                last = shared_idx[0];
                shared_last = last;
            }
            __syncthreads();
        }
    }
}

}  // namespace

at::Tensor fps_forward_cuda(
    const at::Tensor& points,
    const at::Tensor& mask,
    const at::Tensor& start_idx,
    int64_t K) {
    TORCH_CHECK(points.is_cuda(), "points tensor must be on CUDA");
    TORCH_CHECK(mask.is_cuda(), "mask tensor must be on CUDA");
    TORCH_CHECK(start_idx.is_cuda(), "start_idx tensor must be on CUDA");

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

    TORCH_CHECK(D <= 16,
                "torch-fps CUDA kernel supports up to 16 feature dimensions");

    auto idx = at::empty({B, K},
                         at::TensorOptions()
                             .dtype(at::kLong)
                             .device(points_contig.device()));

    constexpr int BLOCK_SIZE = 256;
    constexpr int MAX_D = 16;
    const dim3 blocks(static_cast<unsigned int>(B));

    AT_DISPATCH_FLOATING_TYPES(points_contig.scalar_type(), "fps_forward_cuda", [&] {
        using acc_t = at::acc_type<scalar_t, true>;
        const at::ScalarType acc_scalar_type =
            std::is_same<acc_t, double>::value ? at::kDouble : at::kFloat;
        auto min_dists =
            at::empty({B, N}, points_contig.options().dtype(acc_scalar_type));
        fps_kernel_cuda<scalar_t, acc_t, BLOCK_SIZE, MAX_D>
            <<<blocks, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream()>>>(
                points_contig.data_ptr<scalar_t>(),
                mask_contig.data_ptr<bool>(),
                start_contig.data_ptr<int64_t>(),
                B,
                N,
                D,
                K,
                idx.data_ptr<int64_t>(),
                min_dists.data_ptr<acc_t>());
    });

    AT_CUDA_CHECK(cudaGetLastError());

    return idx;
}

std::tuple<at::Tensor, at::Tensor> fps_with_knn_forward_cuda(
    const at::Tensor& points,
    const at::Tensor& mask,
    const at::Tensor& start_idx,
    int64_t K,
    int64_t k_neighbors) {
    TORCH_CHECK(points.is_cuda(), "points tensor must be on CUDA");
    TORCH_CHECK(mask.is_cuda(), "mask tensor must be on CUDA");
    TORCH_CHECK(start_idx.is_cuda(), "start_idx tensor must be on CUDA");

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

    TORCH_CHECK(D <= 16,
                "torch-fps CUDA kernel supports up to 16 feature dimensions");
    TORCH_CHECK(k_neighbors <= N,
                "k_neighbors must be <= N (number of points)");

    auto centroid_idx = at::empty({B, K},
                                   at::TensorOptions()
                                       .dtype(at::kLong)
                                       .device(points_contig.device()));

    constexpr int BLOCK_SIZE = 256;
    constexpr int MAX_D = 16;
    const dim3 blocks(static_cast<unsigned int>(B));

    at::Tensor neighbor_idx;

    AT_DISPATCH_FLOATING_TYPES(points_contig.scalar_type(), "fps_with_knn_forward_cuda", [&] {
        using acc_t = at::acc_type<scalar_t, true>;
        const at::ScalarType acc_scalar_type =
            std::is_same<acc_t, double>::value ? at::kDouble : at::kFloat;

        auto min_dists = at::empty({B, N}, points_contig.options().dtype(acc_scalar_type));
        auto all_dists = at::empty({B, K, N}, points_contig.options().dtype(acc_scalar_type));

        fps_with_knn_kernel_cuda<scalar_t, acc_t, BLOCK_SIZE, MAX_D>
            <<<blocks, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream()>>>(
                points_contig.data_ptr<scalar_t>(),
                mask_contig.data_ptr<bool>(),
                start_contig.data_ptr<int64_t>(),
                B,
                N,
                D,
                K,
                centroid_idx.data_ptr<int64_t>(),
                min_dists.data_ptr<acc_t>(),
                all_dists.data_ptr<acc_t>());

        AT_CUDA_CHECK(cudaGetLastError());

        // Use PyTorch's optimized topk for kNN selection
        auto topk_result = at::topk(all_dists, k_neighbors, /*dim=*/-1, /*largest=*/false, /*sorted=*/false);
        neighbor_idx = std::get<1>(topk_result);  // [B, K, k_neighbors]
    });

    return std::make_tuple(centroid_idx, neighbor_idx);
}

}  // namespace torch_fps
