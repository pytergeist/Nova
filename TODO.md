# Project book keeping


## TODO:
## General Layering
- Use c++23 modules to define ABI boundaries of each layer
- Introduce more concepts throughout the codebase
## CPU
### SIMD
- Numerical Stability for exp/log
- Scalar tail latency is high - tile structure with last tile padded & partially filled?
- Change SIMD dispatch to concepts + overload sets (or CPO point with tag_invoke?)
### BLAS
- Build concept model for Blas backend (similar to SIMD)
- Implement DotLikeDesc
- Implement AxpyLikeDesc
- Implement GemvLikeDesc
- Implament Transpose as a view operation and pipe it through to descs with T operator in python layer

### Autodiff
- Sparse gradients
- Multi-output model gradient accumulation

### Physics
- - Change physics SIMD dispatch to concepts + overload sets (or CPO point with tag_invoke?)