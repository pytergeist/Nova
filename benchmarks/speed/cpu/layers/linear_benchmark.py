import argparse
import gc
import os
import statistics as stats
import time
from typing import Callable, Tuple

# --- Set thread env before importing NumPy / Torch / TF ---
def set_thread_env(threads: int):
    val = str(max(1, int(threads)))
    os.environ.setdefault("OMP_NUM_THREADS", val)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", val)
    os.environ.setdefault("MKL_NUM_THREADS", val)
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", val)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", val)


set_thread_env(1)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import tensorflow as tf  # noqa: E402

from nova.src.backend.core._tensor import Tensor  # noqa: E402
from nova.src.blocks.core import InputBlock  # noqa: E402
from nova.src.blocks.core.linear import Linear as NovaLinear  # noqa: E402
from nova.src.models import Model  # noqa: E402


def materialize(x):
    """Force computation / materialization for fair timing."""
    # Nova tensor
    if isinstance(x, Tensor):
        return x.data
    # PyTorch
    if torch.is_tensor(x):
        return x
    # TensorFlow
    if isinstance(x, tf.Tensor):
        return x.numpy()
    # NumPy or scalar
    return x


def bench(
        fn: Callable[[], object],
        warmup: int = 10,
        inner: int = 50,
        samples: int = 15,
) -> Tuple[float, float, float]:
    """
    Same style as your micro-benchmarks:
    - warm-up
    - multiple samples, each with 'inner' iterations
    - returns median, IQR, CV%
    """
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
        description="Profile Linear/Dense layers across Nova vs PyTorch vs Keras (CPU)."
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1024,
        help="Batch size (number of rows in input)",
    )
    parser.add_argument(
        "--in-features",
        type=int,
        default=1024,
        help="Input feature dimension",
    )
    parser.add_argument(
        "--out-features",
        type=int,
        default=1024,
        help="Output feature dimension",
    )
    parser.add_argument(
        "--dtype", type=str, default="float32", choices=["float32", "float64"]
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warm-up calls per op")
    parser.add_argument(
        "--inner", type=int, default=50, help="Inner loop iterations per sample"
    )
    parser.add_argument("--samples", type=int, default=15, help="Number of samples")
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="CPU/BLAS threads (pin for stability)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    # Reset env for BLAS-backed libs
    set_thread_env(args.threads)

    # Torch threads
    torch.set_num_threads(max(1, int(args.threads)))
    torch.set_num_interop_threads(1)

    # TF threads (force CPU-style config)
    tf.config.threading.set_intra_op_parallelism_threads(max(1, int(args.threads)))
    tf.config.threading.set_inter_op_parallelism_threads(1)
    # Optional: force CPU if GPUs are visible
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    # Dtypes
    np_dtype = np.float32 if args.dtype == "float32" else np.float64
    torch_dtype = torch.float32 if args.dtype == "float32" else torch.float64
    tf_dtype = tf.float32 if args.dtype == "float32" else tf.float64

    batch = args.batch
    in_features = args.in_features
    out_features = args.out_features

    rng = np.random.default_rng(args.seed)

    # ---- Create shared input ----
    x_np = rng.random((batch, in_features), dtype=np_dtype)

    # Nova backend tensor (runtime input)
    x_nova = Tensor(x_np, requires_grad=False)

    # PyTorch tensor
    x_th = torch.from_numpy(x_np.copy()).to(dtype=torch_dtype)

    # TF tensor
    x_tf = tf.convert_to_tensor(x_np, dtype=tf_dtype)

    # ---- Nova: build tiny 1-layer model graph (Block/Model API) ----
    # InputBlock shape uses (None, in_features) for batch-agnostic dim
    inp_block = InputBlock((None, in_features))
    # Your Linear: Linear(out_features, "random_normal", ...)
    linear_block = NovaLinear(out_features, "random_normal")
    out_block = linear_block(inp_block)
    nova_model = Model(inputs=[inp_block], outputs=[out_block])

    # ---- Nova: raw backend "linear" kernel (x @ W + b) ----
    w_np = rng.standard_normal((in_features, out_features), dtype=np_dtype)
    b_np = np.zeros((out_features,), dtype=np_dtype)

    w_nova = Tensor(w_np, requires_grad=False)
    b_nova = Tensor(b_np, requires_grad=False)

    # ---- PyTorch Linear ----
    torch_layer = torch.nn.Linear(
        in_features, out_features, bias=True, dtype=torch_dtype
    )
    torch_layer.eval()

    # ---- Keras Dense ----
    keras_layer = tf.keras.layers.Dense(
        out_features, use_bias=True, dtype=tf_dtype
    )
    # Build layer by calling once
    _ = keras_layer(x_tf)

    # ---- Define benchmark functions ----

    def fn_nova_model():
        # Run full Nova model forward (Block/Model path)
        return nova_model(x_nova)

    def fn_nova_raw():
        # Backend-only matmul + bias add
        return x_nova @ w_nova + b_nova

    @torch.no_grad()
    def fn_torch():
        return torch_layer(x_th)

    def fn_keras():
        return keras_layer(x_tf)

    print(
        f"Profiling Linear layer on CPU with shape (batch={batch}, in={in_features}, out={out_features}), "
        f"dtype={args.dtype}, samples={args.samples}, inner={args.inner}, warmup={args.warmup}, "
        f"threads={args.threads}\n"
    )

    header = (
        f"{'impl':10s} | "
        f"{'Nova/Torch/Keras (med) s/run':>30s}  {'IQR':>9s}  {'CV%':>7s} | "
        f"{'Nova vs Torch %':>15s}  {'Nova vs Keras %':>16s}"
    )
    print(header)
    print("-" * len(header))

    # Benchmark Torch and Keras once (shared)
    torch_med, torch_iqr, torch_cv = bench(
        fn_torch, warmup=args.warmup, inner=args.inner, samples=args.samples
    )
    keras_med, keras_iqr, keras_cv = bench(
        fn_keras, warmup=args.warmup, inner=args.inner, samples=args.samples
    )

    # Nova model (full framework path)
    nova_model_med, nova_model_iqr, nova_model_cv = bench(
        fn_nova_model, warmup=args.warmup, inner=args.inner, samples=args.samples
    )
    nova_model_vs_torch = (
        ((torch_med - nova_model_med) / torch_med) * 100.0
        if torch_med > 0
        else float("nan")
    )
    nova_model_vs_keras = (
        ((keras_med - nova_model_med) / keras_med) * 100.0
        if keras_med > 0
        else float("nan")
    )

    print(
        f"{'nova_model':10s} | "
        f"{nova_model_med:10.6f} / {torch_med:10.6f} / {keras_med:10.6f}  "
        f"{nova_model_iqr:9.6f}  {nova_model_cv:7.2f} | "
        f"{nova_model_vs_torch:15.2f}  {nova_model_vs_keras:16.2f}"
    )

    # Nova raw kernel (backend only)
    nova_raw_med, nova_raw_iqr, nova_raw_cv = bench(
        fn_nova_raw, warmup=args.warmup, inner=args.inner, samples=args.samples
    )
    nova_raw_vs_torch = (
        ((torch_med - nova_raw_med) / torch_med) * 100.0
        if torch_med > 0
        else float("nan")
    )
    nova_raw_vs_keras = (
        ((keras_med - nova_raw_med) / keras_med) * 100.0
        if keras_med > 0
        else float("nan")
    )

    print(
        f"{'nova_raw':10s} | "
        f"{nova_raw_med:10.6f} / {torch_med:10.6f} / {keras_med:10.6f}  "
        f"{nova_raw_iqr:9.6f}  {nova_raw_cv:7.2f} | "
        f"{nova_raw_vs_torch:15.2f}  {nova_raw_vs_keras:16.2f}"
    )

    print("\nNotes:")
    print("- 'nova_model' = full Block/Model path (InputBlock → Linear → Model(x)).")
    print("- 'nova_raw'   = backend-only matmul + bias (x @ W + b).")
    print("- Speed-up % are positive when Nova is faster (lower median).")
    print("- For apples-to-apples CPU, keep --threads=1 and ensure Torch/TF are on CPU only.")


if __name__ == "__main__":
    main()
