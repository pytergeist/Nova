import argparse
import gc
import os
import statistics as stats
import time
from typing import Callable, Tuple, Dict, Any

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

_THREADING_CONFIGURED = False

_LAYER_CACHE: Dict[Any, Dict[str, Any]] = {}


def materialize(x):
    if isinstance(x, Tensor):
        return x.data
    if torch.is_tensor(x):
        return x
    if isinstance(x, tf.Tensor):
        return x.numpy()
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


def _get_layers(
        in_features: int,
        out_features: int,
        np_dtype,
        torch_dtype,
        tf_dtype,
        seed: int,
) -> Dict[str, Any]:
    """
    Build (or fetch cached) Nova model, raw weights, Torch and Keras layers
    for the given shape + dtype combo.
    """
    key = (in_features, out_features, np_dtype, torch_dtype, tf_dtype)
    if key in _LAYER_CACHE:
        return _LAYER_CACHE[key]

    wrng = np.random.default_rng(seed + 1)

    inp_block = InputBlock((None, in_features))
    linear_block = NovaLinear(out_features, "random_normal")
    out_block = linear_block(inp_block)
    nova_model = Model(inputs=[inp_block], outputs=[out_block])

    w_np = wrng.standard_normal((in_features, out_features), dtype=np_dtype)
    b_np = np.zeros((out_features,), dtype=np_dtype)

    w_nova = Tensor(w_np, requires_grad=False)
    b_nova = Tensor(b_np, requires_grad=False)

    torch_layer = torch.nn.Linear(
        in_features, out_features, bias=True, dtype=torch_dtype
    )
    torch_layer.eval()

    keras_layer = tf.keras.layers.Dense(
        out_features, use_bias=True, dtype=tf_dtype
    )
    dummy_x = tf.zeros((1, in_features), dtype=tf_dtype)
    _ = keras_layer(dummy_x)

    layers = {
        "nova_model": nova_model,
        "w_nova": w_nova,
        "b_nova": b_nova,
        "torch_layer": torch_layer,
        "keras_layer": keras_layer,
    }
    _LAYER_CACHE[key] = layers
    return layers


def profile_linear(
        *,
        batch: int,
        in_features: int = 1024,
        out_features: int = 1024,
        dtype: str = "float32",
        warmup: int = 10,
        inner: int = 50,
        samples: int = 15,
        threads: int = 1,
        seed: int = 42,
        verbose: bool = True,
):
    """
    Run the benchmark once for a given batch size and return the medians.

    Returns a dict with median timings (seconds / forward pass).
    """

    global _THREADING_CONFIGURED

    set_thread_env(threads)

    if not _THREADING_CONFIGURED:
        torch.set_num_threads(max(1, int(threads)))
        torch.set_num_interop_threads(1)

        tf.config.threading.set_intra_op_parallelism_threads(max(1, int(threads)))
        tf.config.threading.set_inter_op_parallelism_threads(1)

        try:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

        _THREADING_CONFIGURED = True

    np_dtype = np.float32 if dtype == "float32" else np.float64
    torch_dtype = torch.float32 if dtype == "float32" else torch.float64
    tf_dtype = tf.float32 if dtype == "float32" else tf.float64

    in_features_ = in_features
    out_features_ = out_features

    layers = _get_layers(
        in_features=in_features_,
        out_features=out_features_,
        np_dtype=np_dtype,
        torch_dtype=torch_dtype,
        tf_dtype=tf_dtype,
        seed=seed,
    )
    nova_model = layers["nova_model"]
    w_nova = layers["w_nova"]
    b_nova = layers["b_nova"]
    torch_layer = layers["torch_layer"]
    keras_layer = layers["keras_layer"]

    rng = np.random.default_rng(seed)

    x_np = rng.random((batch, in_features_), dtype=np_dtype)

    x_nova = Tensor(x_np, requires_grad=False)

    x_th = torch.from_numpy(x_np.copy()).to(dtype=torch_dtype)

    # TF tensor
    x_tf = tf.convert_to_tensor(x_np, dtype=tf_dtype)

    def fn_nova_model():
        return nova_model(x_nova)

    def fn_nova_raw():
        return x_nova @ w_nova + b_nova

    @torch.no_grad()
    def fn_torch():
        return torch_layer(x_th)

    def fn_keras():
        return keras_layer(x_tf)

    if verbose:
        print(
            f"Profiling Linear layer on CPU with shape (batch={batch}, "
            f"in={in_features_}, out={out_features_}), "
            f"dtype={dtype}, samples={samples}, inner={inner}, warmup={warmup}, "
            f"threads={threads}\n"
        )

        header = (
            f"{'impl':10s} | "
            f"{'Nova/Torch/Keras (med) s/run':>30s}  {'IQR':>9s}  {'CV%':>7s} | "
            f"{'Nova vs Torch %':>15s}  {'Nova vs Keras %':>16s}"
        )
        print(header)
        print("-" * len(header))

    torch_med, torch_iqr, torch_cv = bench(
        fn_torch, warmup=warmup, inner=inner, samples=samples
    )
    keras_med, keras_iqr, keras_cv = bench(
        fn_keras, warmup=warmup, inner=inner, samples=samples
    )

    nova_model_med, nova_model_iqr, nova_model_cv = bench(
        fn_nova_model, warmup=warmup, inner=inner, samples=samples
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

    if verbose:
        print(
            f"{'nova_model':10s} | "
            f"{nova_model_med:10.6f} / {torch_med:10.6f} / {keras_med:10.6f}  "
            f"{nova_model_iqr:9.6f}  {nova_model_cv:7.2f} | "
            f"{nova_model_vs_torch:15.2f}  {nova_model_vs_keras:16.2f}"
        )

    nova_raw_med, nova_raw_iqr, nova_raw_cv = bench(
        fn_nova_raw, warmup=warmup, inner=inner, samples=samples
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

    if verbose:
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
        print("- For apples-to-apples CPU, keep threads=1 and ensure Torch/TF are on CPU only.")

    return {
        "batch": batch,
        "nova_model_med": nova_model_med,
        "nova_raw_med": nova_raw_med,
        "torch_med": torch_med,
        "keras_med": keras_med,
        "nova_model_iqr": nova_model_iqr,
        "nova_raw_iqr": nova_raw_iqr,
        "torch_iqr": torch_iqr,
        "keras_iqr": keras_iqr,
        "nova_model_cv": nova_model_cv,
        "nova_raw_cv": nova_raw_cv,
        "torch_cv": torch_cv,
        "keras_cv": keras_cv,
    }


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

    profile_linear(
        batch=args.batch,
        in_features=args.in_features,
        out_features=args.out_features,
        dtype=args.dtype,
        warmup=args.warmup,
        inner=args.inner,
        samples=args.samples,
        threads=args.threads,
        seed=args.seed,
        verbose=True,
    )


if __name__ == "__main__":
    main()
