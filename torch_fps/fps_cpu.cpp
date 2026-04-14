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

// Within-batch parallel FPS kernel: parallelizes the per-iteration distance
// update and argmax across threads. Used when B is smaller than the thread
// pool so that a single batch can still saturate the CPU.
template <typename scalar_t, typename acc_t>
void fps_kernel_cpu_inner_parallel(
    const scalar_t* points,
    const bool* mask,
    int64_t N,
    int64_t D,
    int64_t K,
    int64_t start,
    int64_t* out_indices) {
    std::vector<acc_t> min_dists(static_cast<size_t>(N));

    const acc_t inf = std::numeric_limits<acc_t>::infinity();
    const acc_t neg_inf = -std::numeric_limits<acc_t>::infinity();
    const int num_threads = std::max(1, at::get_num_threads());
    const int64_t grain = std::max<int64_t>(512, (N + num_threads - 1) / num_threads);

    // Init pass (serial — single linear scan, cheap enough).
    int64_t valid_count = 0;
    for (int64_t n = 0; n < N; ++n) {
        bool is_valid = mask[n];
        if (is_valid) {
            const scalar_t* point = points + n * D;
            for (int64_t d = 0; d < D; ++d) {
                if (std::isnan(static_cast<float>(point[d]))) {
                    is_valid = false;
                    break;
                }
            }
        }
        if (is_valid) {
            min_dists[n] = inf;
            ++valid_count;
        } else {
            min_dists[n] = neg_inf;
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

    // Thread-local argmax slots, reused across iterations. Aligned to cache
    // lines to avoid false sharing when multiple threads update neighbors.
    struct alignas(64) Local { acc_t val; int64_t idx; char pad[48]; };
    std::vector<Local> locals(static_cast<size_t>(num_threads));

    for (int64_t i = 0; i < effective_k; ++i) {
        const scalar_t* centroid = points + last * D;
        out_indices[i] = last;
        // Mark selected before the scan so it is excluded from both the
        // distance update (already at neg_inf, skipped) and the argmax.
        min_dists[last] = neg_inf;

        if (i + 1 >= effective_k) {
            break;  // no argmax needed on the final iteration
        }

        // Seed thread-local argmax slots with sentinel (idx=-1) so inactive
        // threads are naturally ignored in the final merge.
        for (int t = 0; t < num_threads; ++t) {
            locals[t].val = neg_inf;
            locals[t].idx = -1;
        }

        // Fused parallel_for: each thread updates min_dists for its range
        // AND tracks a local argmax in the same pass. Halves the
        // at::parallel_for launch overhead vs. two separate calls.
        at::parallel_for(0, N, grain, [&](int64_t begin, int64_t end) {
            const int t = at::get_thread_num();
            acc_t best_v = locals[t].val;
            int64_t best_i = locals[t].idx;
            for (int64_t n = begin; n < end; ++n) {
                acc_t md = min_dists[n];
                if (md == neg_inf) {
                    continue;
                }
                const scalar_t* point = points + n * D;
                acc_t dist = acc_t(0);
                for (int64_t d = 0; d < D; ++d) {
                    const acc_t diff =
                        static_cast<acc_t>(point[d]) - static_cast<acc_t>(centroid[d]);
                    dist += diff * diff;
                }
                if (dist < md) {
                    md = dist;
                    min_dists[n] = dist;
                }
                if (best_i < 0 || md > best_v ||
                    (md == best_v && n < best_i)) {
                    best_v = md;
                    best_i = n;
                }
            }
            locals[t].val = best_v;
            locals[t].idx = best_i;
        });

        acc_t best_v = neg_inf;
        int64_t best_i = last;
        bool have_any = false;
        for (int t = 0; t < num_threads; ++t) {
            if (locals[t].idx < 0) continue;
            if (!have_any ||
                locals[t].val > best_v ||
                (locals[t].val == best_v && locals[t].idx < best_i)) {
                best_v = locals[t].val;
                best_i = locals[t].idx;
                have_any = true;
            }
        }
        last = best_i;
    }

    for (int64_t i = effective_k; i < K; ++i) {
        out_indices[i] = out_indices[effective_k - 1];
    }
}

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
        bool is_valid = mask[n];
        if (is_valid) {
            const scalar_t* point = points + n * D;
            for (int64_t d = 0; d < D; ++d) {
                if (std::isnan(static_cast<float>(point[d]))) {
                    is_valid = false;
                    break;
                }
            }
        }
        if (is_valid) {
            min_dists[n] = inf;
            ++valid_count;
        } else {
            min_dists[n] = neg_inf;
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

    for (int64_t i = 0; i < effective_k; ++i) {
        // Update min_dists based on distance to current centroid
        // This matches: d = (points - c[:, None, :]).square().sum(dim=2)
        //               min_dists = torch.minimum(min_dists, d)
        const scalar_t* centroid = points + last * D;

        for (int64_t n = 0; n < N; ++n) {
            if (min_dists[n] == neg_inf) {
                continue;  // invalid, NaN, or already-selected
            }

            const scalar_t* point = points + n * D;
            acc_t dist = acc_t(0);
            for (int64_t d = 0; d < D; ++d) {
                const acc_t diff =
                    static_cast<acc_t>(point[d]) - static_cast<acc_t>(centroid[d]);
                dist += diff * diff;
            }

            if (dist < min_dists[n]) {
                min_dists[n] = dist;
            }
        }

        out_indices[i] = last;
        min_dists[last] = neg_inf;  // prevent re-selection

        if (i + 1 < effective_k) {
            acc_t best_val = neg_inf;
            int64_t best_idx = last;
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

    // Pad remaining slots with the last valid selection (reached only when
    // the user's K exceeds the number of non-NaN valid points, which the
    // wrapper's K <= counts check cannot see).
    for (int64_t i = effective_k; i < K; ++i) {
        out_indices[i] = out_indices[effective_k - 1];
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
    // Flat contiguous kNN buffer: K rows of up to k_neighbors (dist, idx)
    // pairs. `knn_sizes[i]` holds the current fill for row i. A single
    // allocation replaces the old vector-of-vectors.
    using KnnPair = std::pair<acc_t, int64_t>;
    std::vector<KnnPair> knn_buffer(
        static_cast<size_t>(K) * static_cast<size_t>(k_neighbors));
    std::vector<int64_t> knn_sizes(static_cast<size_t>(K), 0);
    // Tracks which points are geometrically valid (masked AND non-NaN).
    // Separate from min_dists because the kNN loop needs to include
    // already-selected centroids, whereas the FPS argmax must not.
    std::vector<char> point_valid(static_cast<size_t>(N), 0);

    int64_t valid_count = 0;
    const acc_t inf = std::numeric_limits<acc_t>::infinity();
    const acc_t neg_inf = -std::numeric_limits<acc_t>::infinity();

    for (int64_t n = 0; n < N; ++n) {
        bool is_valid = mask[n];
        if (is_valid) {
            const scalar_t* point = points + n * D;
            for (int64_t d = 0; d < D; ++d) {
                if (std::isnan(static_cast<float>(point[d]))) {
                    is_valid = false;
                    break;
                }
            }
        }
        point_valid[n] = is_valid ? 1 : 0;
        if (is_valid) {
            min_dists[n] = inf;
            ++valid_count;
        } else {
            min_dists[n] = neg_inf;
        }
    }

    const int64_t effective_k = std::min<int64_t>(valid_count, K);
    int64_t last = (start >= 0 && start < N) ? start : 0;

    if (K == 0) {
        return;
    }

    if (effective_k == 0) {
        std::fill(out_centroid_indices, out_centroid_indices + K, last);
        std::fill(out_neighbor_indices, out_neighbor_indices + K * k_neighbors, 0);
        return;
    }

    // FPS loop with incremental kNN tracking (flat buffer, K*k_neighbors).
    for (int64_t i = 0; i < effective_k; ++i) {
        const scalar_t* centroid = points + last * D;
        KnnPair* row = knn_buffer.data() + i * k_neighbors;
        int64_t& row_size = knn_sizes[i];

        for (int64_t n = 0; n < N; ++n) {
            if (!point_valid[n]) {
                continue;
            }

            const scalar_t* point = points + n * D;
            acc_t dist = acc_t(0);
            for (int64_t d = 0; d < D; ++d) {
                const acc_t diff =
                    static_cast<acc_t>(point[d]) - static_cast<acc_t>(centroid[d]);
                dist += diff * diff;
            }

            // Max-heap on the flat row: largest dist at the front.
            if (row_size < k_neighbors) {
                row[row_size++] = {dist, n};
                if (row_size == k_neighbors) {
                    std::make_heap(row, row + k_neighbors);
                }
            } else if (dist < row[0].first) {
                std::pop_heap(row, row + k_neighbors);
                row[k_neighbors - 1] = {dist, n};
                std::push_heap(row, row + k_neighbors);
            }

            if (min_dists[n] != neg_inf && dist < min_dists[n]) {
                min_dists[n] = dist;
            }
        }

        out_centroid_indices[i] = last;
        min_dists[last] = neg_inf;  // prevent re-selection

        if (i + 1 < effective_k) {
            acc_t best_val = neg_inf;
            int64_t best_idx = last;
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

    // Pad remaining centroid slots (reached only when effective_k < K).
    for (int64_t i = effective_k; i < K; ++i) {
        out_centroid_indices[i] = out_centroid_indices[effective_k - 1];
    }

    // Extract neighbor indices, sorted closest-first to match the docstring.
    for (int64_t i = 0; i < effective_k; ++i) {
        KnnPair* row = knn_buffer.data() + i * k_neighbors;
        const int64_t k_actual = knn_sizes[i];

        // std::sort gives ascending (closest-first) regardless of heap state.
        std::sort(row, row + k_actual);

        for (int64_t k = 0; k < k_actual; ++k) {
            out_neighbor_indices[i * k_neighbors + k] = row[k].second;
        }
        for (int64_t k = k_actual; k < k_neighbors; ++k) {
            out_neighbor_indices[i * k_neighbors + k] = out_centroid_indices[i];
        }
    }
    // Pad kNN rows for any padded centroid slots.
    for (int64_t i = effective_k; i < K; ++i) {
        for (int64_t k = 0; k < k_neighbors; ++k) {
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

    TORCH_CHECK(points.scalar_type() == at::kFloat || points.scalar_type() == at::kDouble ||
                points.scalar_type() == at::kHalf,
                "points tensor must be float16, float32, or float64 (bfloat16 only supported on CUDA)");

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

    // Inner parallelism only beats outer when the batch dimension itself
    // can't saturate the thread pool (B == 1) and each iteration's work is
    // big enough to amortize the at::parallel_for launch cost.
    const int num_threads = std::max(1, at::get_num_threads());
    const bool use_inner_parallel =
        (num_threads > 1) &&
        (B == 1) &&
        (N * D >= 8192);

    AT_DISPATCH_FLOATING_TYPES_AND(at::kHalf, points_contig.scalar_type(), "fps_forward_cpu", [&] {
        using acc_t = at::acc_type<scalar_t, /*is_cuda=*/true>;

        const scalar_t* points_ptr = points_contig.data_ptr<scalar_t>();
        const bool* mask_ptr = mask_contig.data_ptr<bool>();
        const int64_t* start_ptr = start_contig.data_ptr<int64_t>();
        int64_t* idx_ptr = idx.data_ptr<int64_t>();

        if (use_inner_parallel) {
            for (int64_t b = 0; b < B; ++b) {
                const scalar_t* batch_points = points_ptr + b * N * D;
                const bool* batch_mask = mask_ptr + b * N;
                const int64_t start = start_ptr[b];
                int64_t* batch_idx = idx_ptr + b * K;
                fps_kernel_cpu_inner_parallel<scalar_t, acc_t>(
                    batch_points, batch_mask, N, D, K, start, batch_idx);
            }
        } else {
            at::parallel_for(0, B, 0, [&](int64_t begin, int64_t end) {
                for (int64_t b = begin; b < end; ++b) {
                    const scalar_t* batch_points = points_ptr + b * N * D;
                    const bool* batch_mask = mask_ptr + b * N;
                    const int64_t start = start_ptr[b];
                    int64_t* batch_idx = idx_ptr + b * K;
                    fps_kernel_cpu<scalar_t, acc_t>(
                        batch_points, batch_mask, N, D, K, start, batch_idx);
                }
            });
        }
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

    TORCH_CHECK(points.scalar_type() == at::kFloat || points.scalar_type() == at::kDouble ||
                points.scalar_type() == at::kHalf,
                "points tensor must be float16, float32, or float64 (bfloat16 only supported on CUDA)");

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

    AT_DISPATCH_FLOATING_TYPES_AND(at::kHalf, points_contig.scalar_type(), "fps_with_knn_forward_cpu", [&] {
        using acc_t = at::acc_type<scalar_t, /*is_cuda=*/true>;

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
