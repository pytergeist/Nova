# Roadmap

This document outlines the planned development direction for Nova. The roadmap reflects engineering priorities and architectural intent rather than fixed release promises. Items may evolve as the system matures.

## Near Term Focus: stability, correctness, and developer ergonomics.

### Core Tensor and Semantics
- Harden shape inference and error reporting
- Improve handling of dynamic dimensions and batching
- Expand test coverage for broadcasting and reduction edge cases
- Validate zero-copy guarantees across tensor views

### Autodiff
- Strengthen gradient accumulation correctness
- Improve diagnostics for invalid graph construction
- Expand finite-difference gradient validation
- Refine memory reuse during backward execution

### Memory and Allocation
- Finalize allocator interfaces
- Introduce arena allocator for autodiff contexts
- Improve visibility into allocation lifetimes
- Add allocator-level diagnostics and instrumentation

### Python API
- Stabilize public-facing APIs
- Reduce exposure of internal namespaces
- Improve typing and stub coverage
- Clarify error messages and failure modes

## Medium Term Focus:  performance, extensibility, and tooling.

### Performance
- Expand SIMD kernel coverage
- Improve reduction and broadcast kernel efficiency
- Tune thread pool scheduling and work partitioning
- Explore kernel fusion opportunities where appropriate

### Graph and Execution
- Optional static graph execution mode
- Graph inspection and visualization tooling
- Selective common subexpression elimination
- Better separation of graph construction and execution phases

### Storage
- Introduce sparse storage backends
- Improve support for non-contiguous layouts
- Investigate layout-aware kernel specialization

### Tooling
- Improved profiling hooks
- Better integration with system profilers
- Enhanced logging and tracing facilities
- Build-time configuration introspection

## Physics and Scientific Computing Focus: extending Nova beyond conventional machine learning workloads.

### Physics Layer
- Differentiable ODE solvers
- Differentiable PDE solvers
- Physics solvers represented as first-class graph nodes
- Time-stepping integrated with autodiff execution

### Hybrid Workflows
- Coupled simulation and learning pipelines
- Inverse problems and parameter estimation
- Physics-informed neural networks
- Monte Carlo and sampling-based methods

## Long Term Focus: research, compilation, and heterogeneous execution.

### Devices
- GPU backend support
- Explicit device placement APIs
- Multi-device graph execution
- Device-aware memory allocation strategies ### Compilation
- Ahead-of-time graph lowering
- Kernel specialization and code generation
- JIT versus AOT execution strategies
- Intermediate representations for optimization

### Research Directions
- Alternative autodiff formulations
- Symbolic and hybrid symbolic-numeric computation
- Probabilistic programming primitives
- Differentiable programming beyond neural networks

## Non-Goals
- Full API compatibility with existing frameworks
- Implicit or opaque execution models
- Prioritizing benchmark leaderboards over clarity
- Hiding architectural complexity behind magic Nova is intended as a transparent and extensible systems platform rather than a black-box framework.
