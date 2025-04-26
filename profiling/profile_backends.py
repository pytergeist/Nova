# profile_np_vs_tensor.py
import time

import numpy as np

from nova.src.backend.core._tensor import Tensor  # adjust import path if needed


def profile(fn, repeat=10):
    fn()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - t0) / repeat


def main():
    shape = (1000, 1000)
    a_np = np.random.rand(*shape)
    b_np = np.random.rand(*shape)
    scalar = 0.5

    a_t = Tensor(a_np, requires_grad=False)
    b_t = Tensor(b_np, requires_grad=False)

    ops = {
        "addition": (lambda: a_np + b_np, lambda: a_t + b_t),
        "subtraction": (lambda: a_np - b_np, lambda: a_t - b_t),
        "multiplication": (lambda: a_np * b_np, lambda: a_t * b_t),
        "division": (lambda: a_np / b_np, lambda: a_t / b_t),
        "matmul": (lambda: a_np @ b_np, lambda: a_t @ b_t),
        "sum": (lambda: a_np.sum(), lambda: a_t.sum()),
        "maximum": (lambda: np.maximum(a_np, scalar), lambda: a_t.maximum(scalar)),
        "exp": (lambda: np.exp(a_np), lambda: a_t.exp()),
        "log": (lambda: np.log(a_np), lambda: a_t.log()),
        "sqrt": (lambda: np.sqrt(a_np), lambda: a_t.sqrt()),
    }

    print(f"Profiling NumPy vs Tensor on {shape} arrays:\n")
    for name, (fn_np, fn_t) in ops.items():
        t_np = profile(fn_np, repeat=5)
        t_t = profile(fn_t, repeat=5)
        print(
            f"{name:14s} —  NumPy: {t_np * 1e3:8.3f} ms  |  Tensor: {t_t * 1e3:8.3f} ms"
        )


if __name__ == "__main__":
    main()
