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

from nova.src.backend.core._tensor import Tensor  # noqa: E402


def materialize(x):
    if hasattr(x, "data"):
        return x.data
    return x


def bench(
    fn: Callable[[], object],
    warmup: int = 10,
    inner: int = 200,
    samples: int = 15,
) -> Tuple[float, float, float]:
    # warm-up benchmarks
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
    parser = argparse.ArgumentParser(description="Profile NumPy vs Tensor backend.")
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
        "--threads", type=int, default=1, help="BLAS threads (pin for stability)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    set_thread_env(args.threads)

    dtype = np.float32 if args.dtype == "float32" else np.float64
    shape = tuple(args.shape)

    rng = np.random.default_rng(args.seed)

    # Pre-create inputs (avoid allocation churn in timed region)
    a_np = rng.random(shape, dtype=dtype)
    b_np = rng.random(shape, dtype=dtype)
    scalar_np = dtype(0.5)
    m_a_np = rng.random(shape, dtype=dtype)
    m_b_np = rng.random(shape, dtype=dtype)

    a_t = Tensor(a_np, requires_grad=False)
    b_t = Tensor(b_np, requires_grad=False)
    scalar_t = Tensor(scalar_np, requires_grad=False)
    m_a_t = Tensor(m_a_np, requires_grad=False)
    m_b_t = Tensor(m_b_np, requires_grad=False)

    operations: Dict[str, Tuple[Callable[[], object], Callable[[], object]]] = {
        "add": (lambda: a_np + b_np, lambda: a_t + b_t),
        "sub": (lambda: a_np - b_np, lambda: a_t - b_t),
        "mul": (lambda: a_np * b_np, lambda: a_t * b_t),
        "div": (lambda: a_np / b_np, lambda: a_t / b_t),
        "matmul": (lambda: m_a_np @ m_b_np, lambda: m_a_t @ m_b_t),
        "sum": (lambda: a_np.sum(), lambda: a_t.sum()),
        "maximum": (lambda: np.maximum(a_np, scalar_np), lambda: a_t.maximum(scalar_t)),
        "exp": (lambda: np.exp(a_np), lambda: a_t.exp()),
        "log": (lambda: np.log(a_np), lambda: a_t.log()),
        "sqrt": (lambda: np.sqrt(a_np), lambda: a_t.sqrt()),
    }

    print(
        f"Profiling {len(operations)} ops on shape {shape}, dtype={args.dtype}, "
        f"samples={args.samples}, inner={args.inner}, warmup={args.warmup}, threads={args.threads}\n"
    )

    header = (
        f"{'op':10s} | "
        f"{'NumPy (med) s/run':>16s}  {'IQR':>9s}  {'CV%':>7s} | "
        f"{'Tensor (med) s/run':>18s}  {'IQR':>9s}  {'CV%':>7s} | "
        f"{'speed-up %':>10s}"
    )
    print(header)
    print("-" * len(header))

    for name, (np_fn, t_fn) in operations.items():
        np_med, np_iqr, np_cv = bench(
            np_fn, warmup=args.warmup, inner=args.inner, samples=args.samples
        )
        t_med, t_iqr, t_cv = bench(
            t_fn, warmup=args.warmup, inner=args.inner, samples=args.samples
        )

        speedup_pct = (
            ((np_med - t_med) / np_med) * 100.0 if np_med > 0 else float("nan")
        )

        print(
            f"{name:10s} | "
            f"{np_med:16.6f}  {np_iqr:9.6f}  {np_cv:7.2f} | "
            f"{t_med:18.6f}  {t_iqr:9.6f}  {t_cv:7.2f} | "
            f"{speedup_pct:10.2f}"
        )

    print("\nNotes:")
    print(
        "- speed-up % is positive when Tensor is faster than NumPy (computed on medians)."
    )
    print(
        "- IQR (Q3-Q1) and CV% (stability) help spot jitter (e.g., GC, scheduler, thread startup)."
    )
    print(
        "- For apples-to-apples, keep --threads=1; then explore scaling with higher values."
    )


if __name__ == "__main__":
    main()
