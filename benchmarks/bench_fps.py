"""Benchmark graphnet's FPS backends: pure PyTorch vs torch.compile vs Triton.

Runs against graphnet's `graphnet.models.components.fps` (whatever graphnet is
importable in the active environment). Backends:

- ``reference``: graphnet's pure-PyTorch path, forced by patching
  ``fps._torch_fps`` to return None.
- ``compiled``: the same path with graphnet's reference helpers
  (``_fps_reference``, ``_sq_dist``, ``_gather_rows``) wrapped in
  ``torch.compile``.
- ``triton``: the default dispatch on CUDA, i.e. the ``torch_fps`` package.

Sections:

- op-level: ``farthest_point_sampling_with_assign`` and
  ``farthest_point_sampling_with_knn`` (k_neighbors=8), K=128, B in
  {128, 256}, N in {192, 1000, 3000}, per-event valid count ~ U[N/2, N].
- end-to-end: a full Neptune training step (forward + backward + AdamW) with
  the neptune_v5 training geometry on synthetic IceCube-like events, or an
  inference step (eval mode, ``torch.no_grad``, forward only). Each row also
  times the FPSTokenizer forward alone (same inputs, mode and precision), so
  FPS's share of the step can be read off directly.

- cpu: torch_fps's CPU backends called directly (no graphnet, no GPU
  needed): the JIT-compiled C++ kernels vs the pure-PyTorch reference
  (forced by patching ``torch_fps._cpu.get_module`` to return None), for
  assign and knn (k_neighbors=8), K=128, B in {16, 128}, N in
  {192, 1000, 3000}. Records the torch thread count and the load average.

Usage::

    python bench_fps.py [--out results.json] [--skip-compile]
        [--sections ops,e2e,cpu] [--op-batch-sizes 128,256]
        [--op-sizes 192,1000,3000]
        [--e2e-rows main,large,infer,full,bs512]
        [--cpu-batch-sizes 16,128] [--cpu-sizes 192,1000,3000]

    python bench_fps.py --sections cpu --out results_cpu_v1.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import statistics
import time
from typing import Any, Callable, Dict, Iterator, List

import os
import platform

import numpy as np
import torch
import torch.nn as nn

import torch_fps
from torch_fps import _cpu

DEV = torch.device("cuda")
_REFERENCE_HELPERS = ("_fps_reference", "_sq_dist", "_gather_rows")

# The ops/e2e sections patch graphnet's fps module; the cpu section needs
# neither graphnet nor a GPU.
_GRAPHNET_ERR = None
try:
    import graphnet
    from graphnet.models.components import fps as F
    _ORIG = {name: getattr(F, name)
             for name in ("_torch_fps", *_REFERENCE_HELPERS)}
except (ImportError, AttributeError) as exc:
    graphnet = F = None
    _ORIG = {}
    _GRAPHNET_ERR = f"{type(exc).__name__}: {exc}"
_COMPILED: Dict[str, Callable] = {}


def _compiled_fns() -> Dict[str, Callable]:
    if not _COMPILED:
        for name in _REFERENCE_HELPERS:
            _COMPILED[name] = torch.compile(_ORIG[name])
    return _COMPILED


@contextlib.contextmanager
def backend(name: str) -> Iterator[None]:
    """Route graphnet's FPS entry points to one backend."""
    try:
        if name in ("reference", "compiled"):
            F._torch_fps = lambda *args, **kwargs: None  # type: ignore
        if name == "compiled":
            for k, fn in _compiled_fns().items():
                setattr(F, k, fn)
        yield
    finally:
        for k, fn in _ORIG.items():
            setattr(F, k, fn)


# ------------------------------------------------------------------------
# Timing helpers
# ------------------------------------------------------------------------


def time_cuda(fn: Callable[[], Any], warmup: int, iters: int) -> Dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return {
        "median_ms": statistics.median(times),
        "mean_ms": statistics.fmean(times),
        "min_ms": min(times),
        "p90_ms": float(np.percentile(times, 90)),
        "iters": iters,
    }


def peak_mem(fn: Callable[[], Any]) -> Dict:
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del out
    return {"peak_extra_MB": (peak - base) / 2**20, "peak_MB": peak / 2**20}


# ------------------------------------------------------------------------
# Op-level
# ------------------------------------------------------------------------


def make_points(b: int, n: int, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    pts = torch.randn(b, n, 4, generator=g).to(DEV)
    counts = torch.randint(n // 2, n + 1, (b,), generator=g).to(DEV)
    mask = torch.arange(n, device=DEV)[None, :] < counts[:, None]
    return pts, mask


def op_call(op: str, pts, mask, k: int, knn: int):
    kw = dict(random_start=False, validate=False, assume_finite=True)
    if op == "assign":
        return F.farthest_point_sampling_with_assign(pts, mask, k, **kw)
    return F.farthest_point_sampling_with_knn(pts, mask, k, knn, **kw)


def check_equal(pts, mask, k: int, knn: int) -> Dict:
    res = {}
    with backend("triton"):
        ta = op_call("assign", pts, mask, k, knn)
        tk = op_call("knn", pts, mask, k, knn)
    with backend("reference"):
        ra = op_call("assign", pts, mask, k, knn)
        rk = op_call("knn", pts, mask, k, knn)
    res["assign_centroids_equal"] = bool(torch.equal(ta[0], ra[0]))
    res["assign_labels_equal_valid"] = bool(
        torch.equal(ta[1][mask], ra[1][mask])
    )
    res["assign_labels_mismatch_frac"] = float(
        (ta[1][mask] != ra[1][mask]).float().mean()
    )
    res["knn_centroids_equal"] = bool(torch.equal(tk[0], rk[0]))
    res["knn_neighbors_equal"] = bool(torch.equal(tk[1], rk[1]))
    res["knn_neighbors_mismatch_frac"] = float(
        (tk[1] != rk[1]).float().mean()
    )
    return res


def run_ops(
    backends: List[str],
    k: int,
    knn: int,
    iters: int,
    batch_sizes=(128, 256),
    sizes=(192, 1000, 3000),
) -> List[Dict]:
    rows = []
    for b in batch_sizes:
        for n in sizes:
            pts, mask = make_points(b, n, seed=b * 10007 + n)
            eq = check_equal(pts, mask, k, knn)
            print(f"[eq] B={b} N={n} {eq}", flush=True)
            for op in ("assign", "knn"):
                for be in backends:
                    row = dict(op=op, B=b, N=n, K=k, backend=be)
                    if op == "knn":
                        row["k_neighbors"] = knn
                    try:
                        with backend(be):
                            fn = lambda: op_call(op, pts, mask, k, knn)  # noqa
                            t0 = time.perf_counter()
                            fn()
                            torch.cuda.synchronize()
                            row["first_call_s"] = time.perf_counter() - t0
                            row.update(
                                time_cuda(fn, warmup=5, iters=iters)
                            )
                            row.update(peak_mem(fn))
                    except Exception as exc:  # noqa: BLE001
                        row["error"] = f"{type(exc).__name__}: {exc}"[:500]
                    row["equality"] = eq
                    print(
                        f"[op] {op:6s} B={b:3d} N={n:4d} {be:9s} "
                        f"med={row.get('median_ms', float('nan')):8.3f} ms "
                        f"mem={row.get('peak_extra_MB', float('nan')):8.1f} MB"
                        + (f" ERR {row['error'][:120]}" if "error" in row
                           else ""),
                        flush=True,
                    )
                    rows.append(row)
    return rows


# ------------------------------------------------------------------------
# CPU: C++ kernels vs pure-PyTorch reference (torch_fps directly)
# ------------------------------------------------------------------------


@contextlib.contextmanager
def cpu_backend(name: str) -> Iterator[None]:
    """'cpp': default CPU dispatch; 'reference': C++ module hidden."""
    orig = _cpu.get_module
    try:
        if name == "reference":
            _cpu.get_module = lambda: None  # type: ignore
        yield
    finally:
        _cpu.get_module = orig


def time_cpu(fn: Callable[[], Any], warmup: int, iters: int) -> Dict:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return {
        "median_ms": statistics.median(times),
        "mean_ms": statistics.fmean(times),
        "min_ms": min(times),
        "p90_ms": float(np.percentile(times, 90)),
        "iters": iters,
    }


def cpu_op_call(op: str, pts, mask, k: int, knn: int):
    kw = dict(random_start=False, validate=False, assume_finite=True)
    if op == "assign":
        return torch_fps.farthest_point_sampling_with_assign(pts, mask, k, **kw)
    return torch_fps.farthest_point_sampling_with_knn(pts, mask, k, knn, **kw)


def run_cpu(
    k: int,
    knn: int,
    iters: int,
    batch_sizes=(16, 128),
    sizes=(192, 1000, 3000),
) -> List[Dict]:
    if _cpu.get_module() is None:
        raise RuntimeError("C++ CPU extension unavailable; nothing to compare")
    rows = []
    for b in batch_sizes:
        for n in sizes:
            g = torch.Generator().manual_seed(b * 10007 + n)
            pts = torch.randn(b, n, 4, generator=g)
            counts = torch.randint(n // 2, n + 1, (b,), generator=g)
            mask = torch.arange(n)[None, :] < counts[:, None]
            out = {}
            for be in ("reference", "cpp"):
                with cpu_backend(be):
                    out[be] = {op: cpu_op_call(op, pts, mask, k, knn)
                               for op in ("assign", "knn")}
            ra, ca = out["reference"]["assign"], out["cpp"]["assign"]
            rk, ck = out["reference"]["knn"], out["cpp"]["knn"]
            eq = dict(
                assign_centroids_equal=bool(torch.equal(ra[0], ca[0])),
                assign_labels_mismatch_frac=float(
                    (ra[1][mask] != ca[1][mask]).float().mean()),
                knn_centroids_equal=bool(torch.equal(rk[0], ck[0])),
                knn_neighbors_mismatch_frac=float(
                    (rk[1] != ck[1]).float().mean()),
            )
            print(f"[cpu-eq] B={b} N={n} {eq}", flush=True)
            for op in ("assign", "knn"):
                for be in ("reference", "cpp"):
                    row = dict(op=op, B=b, N=n, K=k, backend=be, equality=eq)
                    if op == "knn":
                        row["k_neighbors"] = knn
                    with cpu_backend(be):
                        row.update(time_cpu(
                            lambda: cpu_op_call(op, pts, mask, k, knn),
                            warmup=2, iters=iters))
                    print(f"[cpu] {op:6s} B={b:3d} N={n:4d} {be:9s} "
                          f"med={row['median_ms']:9.3f} ms", flush=True)
                    rows.append(row)
    return rows


# ------------------------------------------------------------------------
# End-to-end Neptune training step
# ------------------------------------------------------------------------

# neptune_v5 geometry, reconstructed from ckpts_v4/neptune_v5_best_ep029.ckpt
# shapes (7,608,128 backbone params) and the run notes: IceCube86 inputs
# (7 columns), d_model 256, 8 layers, 4 heads, SwiGLU 672, 64 tokens,
# attention pooling, knn_pool=max_mean, lloyd_iters=2. The IceCube86 input
# scalings are Neptune's defaults; the charge column holds raw charge (the
# model applies log1p).
FEATURES = ["dom_x", "dom_y", "dom_z", "dom_time", "charge", "rde", "pmt_area"]
NEPTUNE_KW: Dict[str, Any] = dict(
    input_feature_names=FEATURES,
    coordinate_columns=FEATURES[:3],
    time_column="dom_time",
    charge_column="charge",
    num_patches=64,
    d_model=256,
    depth=8,
    num_heads=4,
    hidden_dim=672,
    pool_type="attention",
    tokenizer_kwargs={"knn_pool": "max_mean", "lloyd_iters": 2},
    compile_encoder=False,
)


def pulse_counts(
    n_events: int, cap: int, rng: np.random.Generator, dist: str = "lognormal"
):
    # "lognormal": tuned so the 192-capped mean is ~113 pulses/event (the
    # measured nodes_per_event of the training data at max_pulses=192);
    # ~2/3 of events exceed the 64-token budget and so go through FPS.
    # "uniform_half": U[cap/2, cap], i.e. every event is large.
    if dist == "uniform_half":
        return rng.integers(cap // 2, cap + 1, size=n_events).astype(np.int64)
    raw = rng.lognormal(mean=4.7, sigma=1.2, size=n_events)
    return np.clip(np.round(raw), 1, cap).astype(np.int64)


def make_event_batch(counts: np.ndarray, rng: np.random.Generator):
    from torch_geometric.data import Batch, Data

    data = []
    for n in counts:
        n = int(n)
        centre = rng.uniform(-0.8, 0.8, size=3)
        xyz = centre + 0.25 * rng.standard_normal((n, 3))
        t = np.sort(0.05 * rng.standard_normal(n))
        q = 10.0 ** (0.3 * rng.standard_normal(n))  # raw charge (PE)
        rde = rng.choice([1.0, 1.35], size=n) - 1.25
        aux = rng.standard_normal(n) * 0.1
        x = np.column_stack([xyz, t, q, rde, aux]).astype(np.float32)
        data.append(Data(x=torch.from_numpy(x)))
    return Batch.from_data_list(data)


# Full-size default Neptune: the constructor defaults for every model
# hyper-parameter (d_model 768, depth 12, 12 heads, SwiGLU 2048, 128 tokens,
# mean pooling, default tokenizer), with the same IceCube86 inputs.
FULL_KW: Dict[str, Any] = dict(
    input_feature_names=FEATURES,
    coordinate_columns=FEATURES[:3],
    time_column="dom_time",
    charge_column="charge",
    num_patches=128,
    d_model=768,
    depth=12,
    num_heads=12,
    hidden_dim=2048,
    compile_encoder=False,
)


class Net(nn.Module):
    def __init__(self, model_kw: Dict[str, Any]) -> None:
        super().__init__()
        from graphnet.models.transformer import Neptune

        self.backbone = Neptune(**model_kw)
        self.head = nn.Linear(self.backbone.nb_outputs, 3)

    def forward(self, data):
        return self.head(self.backbone(data))


def run_e2e(
    backends: List[str],
    batch_size: int,
    cap: int,
    precision: str,
    n_batches: int,
    steps: int,
    seed: int = 0,
    dist: str = "lognormal",
    mode: str = "train",
    model_name: str = "neptune_v5",
    model_kw: Dict[str, Any] = NEPTUNE_KW,
    tok_iters: int = 30,
) -> List[Dict]:
    rng = np.random.default_rng(seed)
    batches = []
    all_counts = []
    for _ in range(n_batches):
        c = pulse_counts(batch_size, cap, rng, dist)
        all_counts.append(c)
        batches.append(make_event_batch(c, rng).to(DEV))
    cat = np.concatenate(all_counts)
    stats = dict(
        mean_pulses=float(cat.mean()),
        median_pulses=float(np.median(cat)),
        frac_at_cap=float((cat == cap).mean()),
        frac_above_tokens=float((cat > model_kw["num_patches"]).mean()),
    )
    targets = [
        nn.functional.normalize(torch.randn(batch_size, 3, device=DEV), dim=1)
        for _ in range(n_batches)
    ]
    rows = []
    for be in backends:
        torch.manual_seed(seed)
        model = Net(model_kw).to(DEV)
        model.train(mode == "train")
        n_params = sum(p.numel() for p in model.backbone.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.02)
        amp = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if precision == "bf16"
            else contextlib.nullcontext()
        )
        it = {"i": 0}

        def step() -> None:
            i = it["i"] % n_batches
            it["i"] += 1
            opt.zero_grad(set_to_none=True)
            with amp:
                pred = model(batches[i])
            loss = (pred.float() - targets[i]).square().mean()
            loss.backward()
            opt.step()

        def infer() -> None:
            i = it["i"] % n_batches
            it["i"] += 1
            with torch.no_grad(), amp:
                model(batches[i])

        # Capture the tokenizer's inputs for every batch once, then time the
        # FPSTokenizer forward alone with the same mode and precision (grad
        # enabled in training, as in the real step).
        tok_inputs: List[Any] = []
        hook = model.backbone.tokenizer.register_forward_pre_hook(
            lambda mod, args, kwargs: tok_inputs.append(
                (tuple(a.detach() if torch.is_tensor(a) else a for a in args),
                 dict(kwargs))
            ),
            with_kwargs=True,
        )
        with torch.no_grad():
            for b in batches:
                model(b)
        hook.remove()
        tok_it = {"i": 0}

        def tok_fwd() -> None:
            i = tok_it["i"] % n_batches
            tok_it["i"] += 1
            a, kw = tok_inputs[i]
            ctx = torch.no_grad() if mode == "infer" else (
                contextlib.nullcontext()
            )
            with ctx, amp:
                model.backbone.tokenizer(*a, **kw)

        fn = step if mode == "train" else infer
        row = dict(
            model=model_name,
            mode=mode,
            backend=be,
            batch_size=batch_size,
            pulse_cap=cap,
            pulse_dist=dist,
            precision=precision,
            backbone_params=n_params,
            **stats,
        )
        try:
            with backend(be):
                t0 = time.perf_counter()
                for _ in range(n_batches if be == "compiled" else 1):
                    fn()
                torch.cuda.synchronize()
                row["first_steps_s"] = time.perf_counter() - t0
                torch.cuda.reset_peak_memory_stats()
                row.update(time_cuda(fn, warmup=n_batches, iters=steps))
                row["peak_MB"] = torch.cuda.max_memory_allocated() / 2**20
                tok = time_cuda(tok_fwd, warmup=n_batches, iters=tok_iters)
                row["tokenizer"] = tok
                row["tokenizer_frac"] = tok["median_ms"] / row["median_ms"]
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"[:800]
        print(
            f"[e2e] {model_name} {mode} bs={batch_size} cap={cap} {dist} "
            f"{precision} {be:9s} "
            f"med={row.get('median_ms', float('nan')):8.2f} ms/step "
            f"tok={row.get('tokenizer', {}).get('median_ms', float('nan')):7.2f}"
            " ms "
            f"peak={row.get('peak_MB', float('nan')):8.0f} MB "
            f"{stats}" + (f" ERR {row['error'][:200]}" if "error" in row
                          else ""),
            flush=True,
        )
        rows.append(row)
        del model, opt, tok_inputs
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_rtx5090.json")
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--e2e-steps", type=int, default=40)
    ap.add_argument("--sections", default="ops,e2e",
                    help="comma-separated subset of ops,e2e,cpu")
    ap.add_argument("--cpu-batch-sizes", default="16,128")
    ap.add_argument("--cpu-sizes", default="192,1000,3000")
    ap.add_argument("--cpu-iters", type=int, default=10)
    ap.add_argument("--op-batch-sizes", default="128,256")
    ap.add_argument("--op-sizes", default="192,1000,3000")
    ap.add_argument(
        "--e2e-rows",
        default="main,large",
        help="main: bs128 cap192 fp32+bf16; large: bs64 cap3000 fp32 "
        "(lognormal, U[1500,3000]); infer: eval/no_grad forward at bs128 "
        "cap192 fp32+bf16 and bs64 U[1500,3000] fp32; full: default "
        "full-size Neptune, bs128 cap192 bf16 training; bs512: v5 bs512 "
        "cap192 bf16 training",
    )
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    backends = ["reference", "triton"]
    if not args.skip_compile:
        backends.insert(1, "compiled")
    secs = args.sections.split(",")
    if ("ops" in secs or "e2e" in secs) and F is None:
        raise SystemExit("the ops/e2e sections need a compatible graphnet "
                         f"({_GRAPHNET_ERR})")

    try:
        import triton
        triton_version = triton.__version__
    except ImportError:
        triton_version = None

    out: Dict[str, Any] = dict(
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        torch=torch.__version__,
        triton=triton_version,
        graphnet=getattr(graphnet, "__version__", None),
        graphnet_path=getattr(graphnet, "__file__", None),
        torch_fps=torch_fps.__version__,
        torch_fps_path=torch_fps.__file__,
        neptune_kwargs=NEPTUNE_KW,
        ops=[],
        e2e=[],
    )
    if "cpu" in secs:
        cpu_name = platform.processor()
        try:
            with open("/proc/cpuinfo") as f:
                cpu_name = next(
                    line.split(":", 1)[1].strip() for line in f
                    if line.startswith("model name"))
        except (OSError, StopIteration):
            pass
        load_before = os.getloadavg()
        out["cpu_info"] = dict(
            cpu=cpu_name,
            logical_cpus=os.cpu_count(),
            torch_threads=torch.get_num_threads(),
            loadavg_before=load_before,
        )
        print(f"[cpu] {cpu_name}, torch threads={torch.get_num_threads()}, "
              f"loadavg={load_before}", flush=True)
        out["cpu"] = run_cpu(
            k=128,
            knn=8,
            iters=args.cpu_iters,
            batch_sizes=[int(v) for v in args.cpu_batch_sizes.split(",")],
            sizes=[int(v) for v in args.cpu_sizes.split(",")],
        )
        out["cpu_info"]["loadavg_after"] = os.getloadavg()
    if "ops" in secs:
        out["ops"] = run_ops(
            backends,
            k=128,
            knn=8,
            iters=args.iters,
            batch_sizes=[int(v) for v in args.op_batch_sizes.split(",")],
            sizes=[int(v) for v in args.op_sizes.split(",")],
        )
    e2e_rows = args.e2e_rows.split(",")
    if "e2e" in secs and "main" in e2e_rows:
        for prec in ("fp32", "bf16"):
            out["e2e"] += run_e2e(
                backends, 128, 192, prec, n_batches=8, steps=args.e2e_steps
            )
    if "e2e" in secs and "large" in e2e_rows:
        for dist in ("lognormal", "uniform_half"):
            out["e2e"] += run_e2e(
                backends, 64, 3000, "fp32", n_batches=8,
                steps=args.e2e_steps, dist=dist,
            )
    if "e2e" in secs and "infer" in e2e_rows:
        for prec in ("fp32", "bf16"):
            out["e2e"] += run_e2e(
                backends, 128, 192, prec, n_batches=8, steps=args.e2e_steps,
                mode="infer",
            )
        out["e2e"] += run_e2e(
            backends, 64, 3000, "fp32", n_batches=8, steps=args.e2e_steps,
            dist="uniform_half", mode="infer",
        )
    if "e2e" in secs and "full" in e2e_rows:
        out["full_kwargs"] = FULL_KW
        out["e2e"] += run_e2e(
            backends, 128, 192, "bf16", n_batches=8, steps=args.e2e_steps,
            model_name="neptune_full", model_kw=FULL_KW,
        )
    if "e2e" in secs and "bs512" in e2e_rows:
        out["e2e"] += run_e2e(
            backends, 512, 192, "bf16", n_batches=8, steps=args.e2e_steps,
        )
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
