#ifndef TENSOR_H
#define TENSOR_H

#include "Fusion/autodiff/AutodiffMode.hpp"
#include "Fusion/core/TensorBase.hpp"

template <typename T> using Tensor = ADTensor<T>;
template <typename T> using RawTensor = TensorBase<T>;

#endif // TENSOR_H
