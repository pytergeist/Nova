import time
import numpy as np
from nova.src._backends._fusion import FusionBackend
from nova.src._backends._numpy import NumpyBackend


def profile_operation(backend, op_func, repeat=10):
    # Run a warmup call
    op_func(backend)
    start = time.perf_counter()
    for _ in range(repeat):
        op_func(backend)
    end = time.perf_counter()
    return (end - start) / repeat


def main():
    # Define input data
    shape = (1000, 1000)
    a = np.random.rand(*shape).astype(np.float64)
    b = np.random.rand(*shape).astype(np.float64)
    scalar = np.float32(0.5)

    # For matrix multiplication, use two square matrices.
    matmul_a = np.random.rand(*shape).astype(np.float64)
    matmul_b = np.random.rand(*shape).astype(np.float64)

    # Instantiate both backends
    fusion_backend = FusionBackend()
    numpy_backend = NumpyBackend()

    # Dictionary mapping operation names to lambda functions.
    # Each lambda takes a backend instance and calls the corresponding method.
    operations = {
        "addition": lambda be: be.add(a, b),
        "subtraction": lambda be: be.subtract(a, b),
        "multiplication": lambda be: be.multiply(a, b),
        "division": lambda be: be.divide(a, b),
        "matmul": lambda be: be.matmul(matmul_a, matmul_b),
        "sum": lambda be: be.sum(a),
        "maximum": lambda be: be.maximum(a, scalar),
        "exp": lambda be: be.exp(a),
        "log": lambda be: be.log(a),
        "sqrt": lambda be: be.sqrt(a),
    }

    print(f"Profiling {len(operations)} operations with input shape {shape}:\n")
    for op_name, op_func in operations.items():
        fusion_time = profile_operation(fusion_backend, op_func, repeat=10)
        numpy_time = profile_operation(numpy_backend, op_func, repeat=10)
        print(f"{op_name}:")
        print(f"  Fusion backend: {fusion_time:.6f} sec per run")
        print(f"  NumPy backend:  {numpy_time:.6f} sec per run\n")


if __name__ == '__main__':
    main()
