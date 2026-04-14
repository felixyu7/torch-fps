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

template <typename scalar_t, typename acc_t, int BLOCK_SIZE>
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
    extern __shared__ unsigned char shared_storage[];
    acc_t* centroid_vals = reinterpret_cast<acc_t*>(shared_storage);

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
        bool valid = batch_mask[n];
        if (valid) {
            const scalar_t* point = batch_points + n * D;
            for (int64_t d = 0; d < D; ++d) {
                if (::isnan(static_cast<float>(point[d]))) {
                    valid = false;
                    break;
                }
            }
        }
        if (valid) {
            batch_min_dists[n] = inf;
            ++local_count;
        } else {
            batch_min_dists[n] = neg_inf;
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

    for (int64_t i = 0; i < effective_k; ++i) {
        // All threads read the current selection from shared memory
        last = shared_last;

        // Load centroid coordinates into shared memory for reuse
        for (int64_t d = threadIdx.x; d < D; d += BLOCK_SIZE) {
            centroid_vals[d] = static_cast<acc_t>(batch_points[last * D + d]);
        }
        __syncthreads();

        // Phase 1: Update min_dists based on distance to current centroid
        for (int64_t n = threadIdx.x; n < N; n += BLOCK_SIZE) {
            if (batch_min_dists[n] == neg_inf) {
                continue;  // invalid, NaN, or already-selected
            }

            const scalar_t* point = batch_points + n * D;
            acc_t dist = acc_t(0);
            for (int64_t d = 0; d < D; ++d) {
                const acc_t diff =
                    static_cast<acc_t>(point[d]) - centroid_vals[d];
                dist += diff * diff;
            }

            if (dist < batch_min_dists[n]) {
                batch_min_dists[n] = dist;
            }
        }
        __syncthreads();

        // Record selection and mark it to prevent re-selection
        if (threadIdx.x == 0) {
            batch_idx[i] = last;
            batch_min_dists[last] = neg_inf;
        }
        __syncthreads();

        // Find next farthest point
        if (i + 1 < effective_k) {
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

    // Pad remaining slots with the last valid selection. Only reached when
    // effective_k < K (e.g. NaN points that the wrapper's K<=counts check
    // could not see).
    if (effective_k < K && threadIdx.x == 0) {
        const int64_t fill = batch_idx[effective_k - 1];
        for (int64_t i = effective_k; i < K; ++i) {
            batch_idx[i] = fill;
        }
    }
}

// ============================================================================
// NOTE on fused kNN v2 (single-pass top-k, reverted):
// An experimental v2 kernel using per-thread register-resident top-k with a
// block-wide pairwise merge tree was implemented and measured here. Analysis:
//  - The oblivious sort-network insertion runs MAX_K predicated ops per
//    point even with early-exit on the worst-slot check, matching v1's
//    k_neighbors argmin scans in raw work.
//  - The merge tree writes O(log(BS) * MAX_K) shared-memory entries per
//    centroid, adding overhead that scales with K.
//  - Register / shared-memory pressure forces BLOCK_SIZE ≤ 128, halving
//    occupancy vs. v1's BLOCK_SIZE=256.
//  - Net effect: ~1.5× gain in a narrow middle range (B≈16, N≈1024, k≤16)
//    and 1.3-2× regressions on huge-N and many-batch workloads. Rolled
//    back. The real single-pass win requires warp-level shuffle merges or
//    block radix sort, which is significantly more involved.
// ============================================================================

// ============================================================================
// Fused FPS + kNN kernel (optimized: compute distances during FPS)
// ============================================================================

template <typename scalar_t, typename acc_t, int BLOCK_SIZE>
__global__ void fps_with_knn_kernel_cuda(
    const scalar_t* __restrict__ points,
    const bool* __restrict__ mask,
    const int64_t* __restrict__ start_idx,
    int64_t B,
    int64_t N,
    int64_t D,
    int64_t K,
    int64_t k_neighbors,
    int64_t* __restrict__ idx,
    int64_t* __restrict__ neighbor_idx,
    acc_t* __restrict__ min_dists,
    acc_t* __restrict__ curr_dists) {
    extern __shared__ unsigned char shared_storage[];
    acc_t* centroid_vals = reinterpret_cast<acc_t*>(shared_storage);

    const int b = blockIdx.x;
    if (b >= B) {
        return;
    }

    const scalar_t* batch_points = points + static_cast<int64_t>(b) * N * D;
    const bool* batch_mask = mask + static_cast<int64_t>(b) * N;
    const int64_t start = start_idx[b];
    int64_t* batch_idx = idx + static_cast<int64_t>(b) * K;
    int64_t* batch_neighbors = neighbor_idx + static_cast<int64_t>(b) * K * k_neighbors;
    acc_t* batch_min_dists = min_dists + static_cast<int64_t>(b) * N;
    acc_t* batch_curr_dists = curr_dists + static_cast<int64_t>(b) * N;

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
        bool valid = batch_mask[n];
        if (valid) {
            const scalar_t* point = batch_points + n * D;
            for (int64_t d = 0; d < D; ++d) {
                if (::isnan(static_cast<float>(point[d]))) {
                    valid = false;
                    break;
                }
            }
        }
        if (valid) {
            batch_min_dists[n] = inf;
            ++local_count;
        } else {
            batch_min_dists[n] = neg_inf;
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
            for (int64_t i = 0; i < K * k_neighbors; ++i) {
                batch_neighbors[i] = 0;
            }
        }
        return;
    }

    for (int64_t i = 0; i < effective_k; ++i) {
        // All threads read the current selection from shared memory
        last = shared_last;

        // Load centroid coordinates into shared memory for reuse
        for (int64_t d = threadIdx.x; d < D; d += BLOCK_SIZE) {
            centroid_vals[d] = static_cast<acc_t>(batch_points[last * D + d]);
        }
        __syncthreads();

        for (int64_t n = threadIdx.x; n < N; n += BLOCK_SIZE) {
            acc_t dist;
            if (!batch_mask[n]) {
                dist = inf;
            } else {
                const scalar_t* point = batch_points + n * D;
                dist = acc_t(0);
                for (int64_t d = 0; d < D; ++d) {
                    const acc_t diff =
                        static_cast<acc_t>(point[d]) - centroid_vals[d];
                    dist += diff * diff;
                }
                // NaN-coord points produce dist=NaN; promote to inf so the
                // argmin has well-defined ordering. Self-inequality detects NaN.
                if (!(dist == dist)) {
                    dist = inf;
                }
            }

            batch_curr_dists[n] = dist;

            // FPS update: only for points that are still candidates.
            // `min_dists[n] == neg_inf` covers invalid, NaN, and already-selected.
            if (batch_min_dists[n] > neg_inf && dist < batch_min_dists[n]) {
                batch_min_dists[n] = dist;
            }
        }
        __syncthreads();

        // Record selection and mark it to prevent re-selection
        if (threadIdx.x == 0) {
            batch_idx[i] = last;
            batch_min_dists[last] = neg_inf;
        }
        __syncthreads();

        // Streaming kNN selection for the current centroid.
        // Points with `curr_dist == inf` are truly invalid (masked/NaN);
        // previously-selected centroids have a real distance and are
        // included here, matching the documented kNN contract.
        for (int64_t nn = 0; nn < k_neighbors; ++nn) {
            acc_t best_val = inf;
            int64_t best_idx = 0;

            for (int64_t n = threadIdx.x; n < N; n += BLOCK_SIZE) {
                const acc_t val = batch_curr_dists[n];
                if (val < best_val || (val == best_val && n < best_idx)) {
                    best_val = val;
                    best_idx = n;
                }
            }

            shared_vals[threadIdx.x] = best_val;
            shared_idx[threadIdx.x] = best_idx;
            __syncthreads();

            for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
                if (threadIdx.x < offset) {
                    const acc_t other_val = shared_vals[threadIdx.x + offset];
                    const int64_t other_idx = shared_idx[threadIdx.x + offset];
                    const acc_t current_val = shared_vals[threadIdx.x];
                    const int64_t current_idx = shared_idx[threadIdx.x];

                    if (other_val < current_val ||
                        (other_val == current_val && other_idx < current_idx)) {
                        shared_vals[threadIdx.x] = other_val;
                        shared_idx[threadIdx.x] = other_idx;
                    }
                }
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                if (shared_vals[0] == inf) {
                    // No valid neighbor remains — fall back to the centroid
                    // so the output contains a valid index (matches CPU).
                    batch_neighbors[i * k_neighbors + nn] = batch_idx[i];
                } else {
                    const int64_t chosen = shared_idx[0];
                    batch_neighbors[i * k_neighbors + nn] = chosen;
                    batch_curr_dists[chosen] = inf;
                }
            }
            __syncthreads();
        }

        if (i + 1 < effective_k) {
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

    // Pad remaining centroid+neighbor slots when effective_k < K.
    if (effective_k < K && threadIdx.x == 0) {
        const int64_t fill = batch_idx[effective_k - 1];
        for (int64_t i = effective_k; i < K; ++i) {
            batch_idx[i] = fill;
            for (int64_t k = 0; k < k_neighbors; ++k) {
                batch_neighbors[i * k_neighbors + k] = fill;
            }
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

    TORCH_CHECK(points.scalar_type() == at::kFloat || points.scalar_type() == at::kDouble ||
                points.scalar_type() == at::kBFloat16 || points.scalar_type() == at::kHalf,
                "points tensor must be float16, bfloat16, float32, or float64");
    TORCH_CHECK(mask.scalar_type() == at::kBool,
                "mask tensor must be boolean");

    TORCH_CHECK(K >= 0, "K must be non-negative");

    auto points_contig = points.contiguous();
    auto mask_contig = mask.contiguous();
    auto start_contig = start_idx.contiguous();

    const auto B = points_contig.size(0);
    const auto N = points_contig.size(1);
    const auto D = points_contig.size(2);

    auto idx = at::empty({B, K},
                         at::TensorOptions()
                             .dtype(at::kLong)
                             .device(points_contig.device()));

    constexpr int BLOCK_SIZE = 256;
    const dim3 blocks(static_cast<unsigned int>(B));

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, points_contig.scalar_type(), "fps_forward_cuda", [&] {
        using acc_t = at::acc_type<scalar_t, /*is_cuda=*/true>;
        const auto acc_dtype = c10::CppTypeToScalarType<acc_t>::value;
        auto min_dists =
            at::empty({B, N}, points_contig.options().dtype(acc_dtype));
        const size_t shared_mem_bytes =
            static_cast<size_t>(D) * sizeof(acc_t);
        fps_kernel_cuda<scalar_t, acc_t, BLOCK_SIZE>
            <<<blocks, BLOCK_SIZE, shared_mem_bytes, at::cuda::getCurrentCUDAStream()>>>(
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

    TORCH_CHECK(points.scalar_type() == at::kFloat || points.scalar_type() == at::kDouble ||
                points.scalar_type() == at::kBFloat16 || points.scalar_type() == at::kHalf,
                "points tensor must be float16, bfloat16, float32, or float64");
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

    auto centroid_idx = at::empty({B, K},
                                   at::TensorOptions()
                                       .dtype(at::kLong)
                                       .device(points_contig.device()));

    constexpr int BLOCK_SIZE = 256;
    const dim3 blocks(static_cast<unsigned int>(B));

    auto neighbor_idx = at::empty({B, K, k_neighbors},
                                  at::TensorOptions()
                                      .dtype(at::kLong)
                                      .device(points_contig.device()));

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, points_contig.scalar_type(), "fps_with_knn_forward_cuda", [&] {
        using acc_t = at::acc_type<scalar_t, /*is_cuda=*/true>;
        const auto acc_dtype = c10::CppTypeToScalarType<acc_t>::value;

        auto min_dists = at::empty({B, N}, points_contig.options().dtype(acc_dtype));
        auto curr_dists = at::empty({B, N}, points_contig.options().dtype(acc_dtype));

        const size_t shared_mem_bytes =
            static_cast<size_t>(D) * sizeof(acc_t);
        fps_with_knn_kernel_cuda<scalar_t, acc_t, BLOCK_SIZE>
            <<<blocks, BLOCK_SIZE, shared_mem_bytes,
               at::cuda::getCurrentCUDAStream()>>>(
                points_contig.data_ptr<scalar_t>(),
                mask_contig.data_ptr<bool>(),
                start_contig.data_ptr<int64_t>(),
                B, N, D, K, k_neighbors,
                centroid_idx.data_ptr<int64_t>(),
                neighbor_idx.data_ptr<int64_t>(),
                min_dists.data_ptr<acc_t>(),
                curr_dists.data_ptr<acc_t>());

        AT_CUDA_CHECK(cudaGetLastError());
    });

    return std::make_tuple(centroid_idx, neighbor_idx);
}

}  // namespace torch_fps
