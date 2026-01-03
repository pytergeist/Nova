# Architecture Overview
This document describes the internal architecture of Nova.
Nova is designed as a layered system with strict separation of concerns.
Each layer has a clearly defined responsibility and minimal coupling to adjacent layers. The architecture prioritizes determinism, explicit ownership, and performance transparency.

## High-Level Structure
Nova is organized into the following conceptual layers, from lowest to highest: Python API

```
│ ├── Python API
│ ├── Python bindings
│ ├── Autodiff Engine
│ ├── Operation Dispatch & Kernels
│ ├── Core Tensor Semantics
│ ├── Storage & Views
│ ├── Device Abstraction
│ └── Memory Allocation Each layer depends only on layers below it.
```
## Memory Allocation Layer

### Responsibility
The allocation layer is responsible for raw memory management, alignment, and reuse.
It provides deterministic ownership semantics and does not depend on any tensor or computation logic.

### Key Components
- AllocatorInterface
- SubAllocatorInterface
- CPUSubAllocator
- BFCPoolAllocator
- DefaultAllocator

### Design Principles
- Explicit ownership via RAII
- No global allocators
- Alignment-aware allocations
- Swappable allocation strategies

### Planned Extensions
- Arena allocator for autodiff lifetimes
- Slab allocator for physics and simulation workloads

## Device Layer
### Responsibility

The device layer abstracts execution devices and provides a foundation for future multi-device support.

### Current State
- CPU device only
- Single-device execution semantics

### Design Notes
- Device logic is intentionally minimal
- No kernel or tensor semantics live here
- Future device backends plug in without altering higher-level abstractions

## Storage Layer
### Responsibility
The storage layer represents tensor memory independently of computation, layout, or autodiff.

### Key Components
- TensorBuffer Shared, reference-counted raw byte storage with typed pointer access.
- DenseStorage Concrete dense tensor storage implementation.
- TensorView Zero-copy views into existing storage.
- StorageInterface Abstract interface for future storage types (e.g. sparse).

### Design Principles
- Zero-copy wherever possible
- Shared ownership semantics
- Storage is orthogonal to shape, layout, and operations

## Core Tensor Layer
### Responsibility
The core tensor layer defines tensor semantics: shape, layout, broadcasting, and iteration.

### Key Components
- RawTensor
- TensorDesc
- Layout metadata (contiguous vs non-contiguous)
- Broadcast iterators - Reduction iterators

### Semantics
- NumPy-style right-aligned broadcasting
- Explicit stride and layout handling
- Shape inference without implicit allocation
- No autodiff logic
- No kernel dispatch This layer defines what a tensor is, not how it is computed.

## Operation Dispatch and Kernel Layer
### Responsibility
This layer executes tensor operations efficiently using SIMD kernels, BLAS routines, and optimized dispatch paths.

### Components
- Elementwise operations
- Reductions
- Linear algebra (GEMM)
- SIMD backends (NEON with scalar fallback)
- Serial and threaded execution paths

### Dispatch Model
- Eager execution
- Hot-path optimizations for contiguous tensors and shape-compatible operands
- General fallback paths for arbitrary layouts

## Automatic Differentiation Layer
### Responsibility
The autodiff layer constructs and executes computation graphs to compute gradients deterministically.

### Core Concepts
- ADTensor Wraps a RawTensor and carries graph metadata.
- SSA-style graph Each tensor value is identified by a ValueID.
- Nodes Type-erased graph nodes parameterized by operation policies.

### Graph Structure
- Explicit produced_by and consumed_by relationships
- Deterministic graph construction
- No mutation during backward execution
- Topological sort with cycle detection

### Operation Registry Each operation defines:
- Forward computation
- Adjoint (gradient) computation
- Shape and dependency metadata Operations are registered via policy-based mechanisms.

## Autodiff Engine
### Responsibility
The autodiff engine orchestrates execution of forward and backward passes.

### Features
- Buffer ownership and reuse
- Backward execution scheduling
- Gradient accumulation
- Thread-local engine contexts

### Design Notes
- Autodiff contexts are explicit
- Nested contexts are supported
- No implicit global state

## Python Bindings
### Responsibility

The Python layer exposes Nova’s functionality while preserving the underlying execution model.

### Implementation
- pybind11 bindings
- Thin wrappers over C++ types
- Minimal operator overloading

### Philosophy
- Python reflects C++ semantics
- No hidden graph rewrites
- Debuggable and inspectable execution High-level Python abstractions are built on top of the core bindings rather than embedded inside them.

## Planned Physics Layer
### Goal
Support differentiable physics and scientific simulations as first-class citizens.

### Design Direction
- Physics solvers represented as graph nodes
- Shared autodiff and memory infrastructure
- Time-stepping and simulation loops as graph execution
- Orthogonal to ML-specific layers This is a core design goal and informs upstream architectural decisions.


## Architectural Non-Goals
- Implicit global state
- Opaque execution graphs
- Tight coupling between subsystems
- Hidden memory allocation Nova favors explicit structure over convenience abstractions
