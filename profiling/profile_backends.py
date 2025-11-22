import argparse
import gc
import os
import statistics as stats
import time
from typing import Callable, Dict, Tuple


# --- Set thread env before numpy import ---
def set_thread_env(threads: int):
    val = str(max(1, int(threads)))
    os.environ.setdefault("OMP_NUM_THREADS", val)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", val)
    os.environ.setdefault("MKL_NUM_THREADS", val)
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", val)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", val)


set_thread_env(1)

import numpy as np  # noqa: E402 (import after setting env)
import torch  # noqa: E402

from nova.src.backend.core._tensor import Tensor  # noqa: E402


def materialize(x):
    if isinstance(x, Tensor):
        return x.data
    if torch.is_tensor(x):
        return x
    return x


def bench(
    fn: Callable[[], object],
    warmup: int = 10,
    inner: int = 200,
    samples: int = 15,
) -> Tuple[float, float, float]:
    # warm-up
    for _ in range(warmup):
        out = fn()
        _ = materialize(out)

    times = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(samples):
            t0 = time.perf_counter()
            for _ in range(inner):
                out = fn()
                _ = materialize(out)
            t1 = time.perf_counter()
            times.append((t1 - t0) / inner)
    finally:
        if gc_was_enabled:
            gc.enable()

    med = stats.median(times)
    q1, q3 = stats.quantiles(times, n=4)[0], stats.quantiles(times, n=4)[2]
    iqr = q3 - q1
    mean = sum(times) / len(times)
    cv = (stats.pstdev(times) / mean) * 100.0 if mean > 0 else 0.0
    return med, iqr, cv


def main():
    parser = argparse.ArgumentParser(
        description="Profile NumPy vs Tensor vs PyTorch backends."
    )
    parser.add_argument(
        "--shape", type=int, nargs=2, default=[1000, 1000], help="Matrix shape (H W)"
    )
    parser.add_argument(
        "--dtype", type=str, default="float32", choices=["float32", "float64"]
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warm-up calls per op")
    parser.add_argument(
        "--inner", type=int, default=200, help="Inner loop iterations per sample"
    )
    parser.add_argument("--samples", type=int, default=15, help="Number of samples")
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="BLAS / Torch threads (pin for stability)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    # Reset env for BLAS-backed libs
    set_thread_env(args.threads)

    # Pin PyTorch threads (CPU)
    torch.set_num_threads(max(1, int(args.threads)))
    torch.set_num_interop_threads(1)

    dtype = np.float32 if args.dtype == "float32" else np.float64
    torch_dtype = torch.float32 if args.dtype == "float32" else torch.float64
    shape = tuple(args.shape)

    rng = np.random.default_rng(args.seed)

    # Pre-create inputs (avoid allocation churn in timed region)
    a_np = rng.random(shape, dtype=dtype)
    b_np = rng.random(shape, dtype=dtype)
    scalar_np = dtype(0.5)
    m_a_np = rng.random(shape, dtype=dtype)
    m_b_np = rng.random(shape, dtype=dtype)

    # Your tensor lib
    a_t = Tensor(a_np, requires_grad=False)
    b_t = Tensor(b_np, requires_grad=False)
    scalar_t = Tensor(scalar_np, requires_grad=False)
    m_a_t = Tensor(m_a_np, requires_grad=False)
    m_b_t = Tensor(m_b_np, requires_grad=False)

    # PyTorch CPU tensors
    a_th = torch.from_numpy(a_np.copy()).to(dtype=torch_dtype)
    b_th = torch.from_numpy(b_np.copy()).to(dtype=torch_dtype)
    scalar_th = torch.tensor(scalar_np, dtype=torch_dtype)
    m_a_th = torch.from_numpy(m_a_np.copy()).to(dtype=torch_dtype)
    m_b_th = torch.from_numpy(m_b_np.copy()).to(dtype=torch_dtype)

    # Each op: (NumPy fn, Tensor fn, Torch fn)
    operations: Dict[
        str, Tuple[Callable[[], object], Callable[[], object], Callable[[], object]]
    ] = {
        "add": (
            lambda: a_np + b_np,
            lambda: a_t + b_t,
            lambda: a_th + b_th,
        ),
        "sub": (
            lambda: a_np - b_np,
            lambda: a_t - b_t,
            lambda: a_th - b_th,
        ),
        "mul": (
            lambda: a_np * b_np,
            lambda: a_t * b_t,
            lambda: a_th * b_th,
        ),
        "div": (
            lambda: a_np / b_np,
            lambda: a_t / b_t,
            lambda: a_th / b_th,
        ),
        "matmul": (
            lambda: m_a_np @ m_b_np,
            lambda: m_a_t @ m_b_t,
            lambda: m_a_th @ m_b_th,
        ),
        "sum": (
            lambda: a_np.sum(),
            lambda: a_t.sum(),
            lambda: a_th.sum(),
        ),
        "maximum": (
            lambda: np.maximum(a_np, scalar_np),
            lambda: a_t.maximum(scalar_t),
            lambda: torch.maximum(a_th, scalar_th),
        ),
        "exp": (
            lambda: np.exp(a_np),
            lambda: a_t.exp(),
            lambda: torch.exp(a_th),
        ),
        "log": (
            lambda: np.log(a_np),
            lambda: a_t.log(),
            lambda: torch.log(a_th),
        ),
        "sqrt": (
            lambda: np.sqrt(a_np),
            lambda: a_t.sqrt(),
            lambda: torch.sqrt(a_th),
        ),
    }

    print(
        f"Profiling {len(operations)} ops on shape {shape}, dtype={args.dtype}, "
        f"samples={args.samples}, inner={args.inner}, warmup={args.warmup}, threads={args.threads}\n"
    )

    header = (
        f"{'op':10s} | "
        f"{'NumPy (med) s/run':>16s}  {'IQR':>9s}  {'CV%':>7s} | "
        f"{'Tensor (med) s/run':>18s}  {'IQR':>9s}  {'CV%':>7s} | "
        f"{'Torch (med) s/run':>17s}  {'IQR':>9s}  {'CV%':>7s} | "
        f"{'Tensor vs NumPy %':>17s}  {'Tensor vs Torch %':>17s}"
    )
    print(header)
    print("-" * len(header))

    for name, (np_fn, t_fn, th_fn) in operations.items():
        np_med, np_iqr, np_cv = bench(
            np_fn, warmup=args.warmup, inner=args.inner, samples=args.samples
        )
        t_med, t_iqr, t_cv = bench(
            t_fn, warmup=args.warmup, inner=args.inner, samples=args.samples
        )
        th_med, th_iqr, th_cv = bench(
            th_fn, warmup=args.warmup, inner=args.inner, samples=args.samples
        )

        tensor_vs_numpy = (
            ((np_med - t_med) / np_med) * 100.0 if np_med > 0 else float("nan")
        )
        tensor_vs_torch = (
            ((th_med - t_med) / th_med) * 100.0 if th_med > 0 else float("nan")
        )

        print(
            f"{name:10s} | "
            f"{np_med:16.6f}  {np_iqr:9.6f}  {np_cv:7.2f} | "
            f"{t_med:18.6f}  {t_iqr:9.6f}  {t_cv:7.2f} | "
            f"{th_med:17.6f}  {th_iqr:9.6f}  {th_cv:7.2f} | "
            f"{tensor_vs_numpy:17.2f}  {tensor_vs_torch:17.2f}"
        )

    print("\nNotes:")
    print(
        "- speed-up % columns are positive when Tensor/PyTorch are faster than NumPy (computed on medians)."
    )
    print(
        "- IQR (Q3-Q1) and CV% (stability) help spot jitter (e.g., GC, scheduler, thread startup)."
    )
    print(
        "- For apples-to-apples, keep --threads=1; then explore scaling with higher values."
    )


if __name__ == "__main__":
    main()
