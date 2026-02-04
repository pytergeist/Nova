# Project book keeping


## TODO:
## General Layering
- Use c++23 modules to define ABI boundaries of each layer
## CPU
### SIMD
- Numerical Stability for exp/log
### BLAS
- Build concept model for Blas backend (similar to SIMD)
- Implement DotLikeDesc
- Implement AxpyLikeDesc
- Implement GemvLikeDesc
- Implament Transpose as a view operation and pipe it through to descs with T operator in python layer

### Autodiff
- Sparse gradients
- Multi-output model gradient accumulation
