<p align="center">
  <img src="logo.png" alt="Nova Logo" width="200"/>
</p>

<p align="center">
  <img src="https://github.com/pytergeist/Nova/actions/workflows/cross-platform-build.yaml/badge.svg" />
  <img src="https://github.com/pytergeist/Nova/actions/workflows/unit-tests.yaml/badge.svg" />
  <img src="https://github.com/pytergeist/Nova/actions/workflows/integration-test.yaml/badge.svg" />
  <img src="https://github.com/pytergeist/Nova/actions/workflows/documentation.yaml/badge.svg" />
</p>

# Nova (WIP)

Nova is a high-performance deep learning framework and tensor engine built from first principles in modern C++, with a Python 3.13 frontend.

It provides a full stack for numerical computation and machine learning, including explicit memory management, a SIMD-accelerated tensor core, a deterministic automatic differentiation system, and a Python API for building and training models.

Nova is designed for researchers and engineers who care about **performance transparency**, **architectural clarity**, and **extensibility beyond conventional ML workloads**.

---

## Why Nova?

Most existing frameworks optimize for ease of use and rapid experimentation, often at the cost of opaque execution, implicit memory behavior, and tightly coupled subsystems.

Nova explores a different set of trade-offs:

- Explicit and deterministic execution
- Clear separation between tensor semantics, kernels, and autodiff
- First-class memory management
- A foundation suitable for machine learning *and* scientific computing

Nova is not intended to be a drop-in replacement for other frameworks.  
It is a research-driven system with an emphasis on correctness, performance, and long-term extensibility.

---

## Features

### Tensor Core
- Dense tensor storage with explicit layout
- NumPy-style broadcasting and reduction semantics
- Zero-copy views and slicing
- SIMD-accelerated elementwise operations
- BLAS-backed linear algebra

### Memory Management
- Pluggable allocator interfaces
- Pool-based allocation for reuse and alignment
- Deterministic ownership via RAII
- Designed for future arena and slab allocation strategies

### Automatic Differentiation
- Deterministic, SSA-style computation graphs
- Explicit forward and adjoint definitions per operation
- Topological sorting with cycle detection
- Thread-local autodiff contexts
- No hidden graph mutation during backward passes

### Python API
- Python 3.13 support
- Thin bindings over the C++ core via pybind11
- High-level abstractions for:
  - Tensors
  - Layers
  - Models
  - Optimizers
  - Loss functions
- NumPy interoperability

### Performance
- Optimized fast paths for contiguous and shape-compatible tensors
- SIMD kernels with scalar fallbacks
- Eager execution with minimal overhead
- Designed for predictable performance characteristics

---

## Example

A simple multi-layer perceptron:

```python
from nova.src.blocks.core import InputBlock, Linear
from nova.src.blocks.activations import ReLU
from nova.src.blocks.regularisation import Dropout
from nova.src.models import Model

inp = InputBlock((None, 10))
x = Linear(100, "random_normal")(inp)
x = ReLU()(x)
x = Dropout(0.5)(x)
x = Linear(10, "random_normal")(x)
out = ReLU()(x)

model = Model(inputs=[inp], outputs=[out])
```

Training with an explicit autodiff context:

```python
from nova.src.backend.core import Tensor, Grad
from nova.src.optim.sgd import SGD
from nova.src.losses import MeanSquaredError

optimizer = SGD(model.parameters(), lr=1e-3)
loss_fn = MeanSquaredError()

with Grad():
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()

optimizer.step()
```

## Installation

Nova is currently built from source.

### Requirements

- C++20-compatible compiler
- CMake ≥ 3.21
- Python 3.13 (This will soon be changed to python >= 3.11)
- pybind11
- BLAS (system or vendor-provided)

## Build (CPU backend)
Nova provides a `Makefile` that wraps CMake presets and common development configurations.

### Development build (default)
```
make dev
```
The build output will be located under:
```
build/dev/
```
A symlink to `compile_commands.json` will be created in the project root if available.

### Release build
Optimized release build with interprocedural optimisation and aggressive compiler flags:
```bash
make release
```
Build output:
```bash
build/release/
```

### CPU profiling build
Build with optimisations, debug symbols, and frame pointers enabled (useful for Instruments, perf, or sampling profilers):
```bash
make cpu-profile
```

### AddressSanitizer build
```bash
make asan
```

### Incremental rebuild
Rebuild without re-running CMake configuration:
```bash
make rebuild
```

The Python extension module is built as part of the CMake process and can be imported directly from the project root.
Packaging, wheels, and a stable installation path are under active development.

### Running tests
Nova has a split C++/python tests matrix. Fusion uses GTest, and can be run with: 
```bash
make test
```
This runs `ctest` in the active build directory.

For the python side tests, Nova uses pytest. Python tests can be run with:
```bash
pytest -v
```

## Documentation
- Architecture overview: ARCH.md
- Development roadmap: ROADMAP.md
- API and module documentation: docs/
- Additional documentation will be added as the public API stabilizes.

## Project Status
- Nova is under active development.
- The CPU backend is the primary focus
- APIs may change as the architecture evolves
- Performance tuning and correctness validation are ongoing
- Despite this, Nova is already usable for real numerical workloads and model training, and serves as a foundation for further research and experimentation.

## Design Philosophy
Nova emphasizes:
- Explicit over implicit behavior
- Deterministic execution over dynamic mutation
- Clear separation of concerns between subsystems
- Performance characteristics that are inspectable and understandable
- Complexity is not hidden, but structured.

## Contributing

Contributions are welcome.

Nova is particularly suited to contributors interested in:
- Systems programming
- Numerical computing
- Automatic differentiation
- Memory allocation strategies
- Performance engineering
- Scientific and physics-informed machine learning
- Contribution guidelines will be added in `CONTRIBUTING.md`.

## License
Nova is released under the MIT License.