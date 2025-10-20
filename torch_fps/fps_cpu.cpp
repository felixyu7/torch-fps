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
        out_indices[i] = last;

        if (i + 1 >= effective_k) {
            continue;
        }

        const scalar_t* centroid = points + last * D;

        acc_t best_val = -std::numeric_limits<acc_t>::infinity();
        int64_t best_idx = last;

        for (int64_t n = 0; n < N; ++n) {
            if (!mask[n]) {
                continue;
            }

            const scalar_t* point = points + n * D;

            acc_t dist = acc_t(0);
            for (int64_t d = 0; d < D; ++d) {
                const acc_t diff =
                    static_cast<acc_t>(point[d]) - static_cast<acc_t>(centroid[d]);
                dist += diff * diff;
            }

            acc_t current = min_dists[n];
            if (dist < current) {
                current = dist;
            }
            min_dists[n] = current;

            if (current > best_val ||
                (current == best_val && n < best_idx)) {
                best_val = current;
                best_idx = n;
            }
        }

        last = best_idx;
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

}  // namespace torch_fps
