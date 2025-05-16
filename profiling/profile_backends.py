import time

import numpy as np

# import your Tensor class
from nova.src.backend.core._tensor import Tensor


def profile(fn, repeat=10):
    # warm-up
    fn()
    start = time.perf_counter()
    for _ in range(repeat):
        out = fn()
        # force materialization of the result as a numpy array,
        # so we’re not just timing Python dispatch.
        _ = out.data if hasattr(out, "data") else out
    end = time.perf_counter()
    return (end - start) / repeat


def main():
    shape = (1000, 1000)

    # 1) create raw NumPy inputs
    a_np = np.random.rand(*shape).astype(np.float64)
    b_np = np.random.rand(*shape).astype(np.float64)
    scalar_np = np.float64(0.5)
    m_a_np, m_b_np = (
        np.random.rand(*shape).astype(np.float64),
        np.random.rand(*shape).astype(np.float64),
    )

    # 2) wrap them in your Tensor
    a_t = Tensor(a_np, requires_grad=False)
    b_t = Tensor(b_np, requires_grad=False)
    scalar_t = Tensor(scalar_np, requires_grad=False)
    m_a_t = Tensor(m_a_np, requires_grad=False)
    m_b_t = Tensor(m_b_np, requires_grad=False)

    operations = {
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

    print(f"Profiling {len(operations)} ops on shape {shape}\n")
    for name, (np_fn, t_fn) in operations.items():
        t_np = profile(np_fn, repeat=10)
        t_tensor = profile(t_fn, repeat=10)
        print(f"{name:10s} | NumPy: {t_np:.6f}s/run    Tensor: {t_tensor:.6f}s/run")


if __name__ == "__main__":
    main()
