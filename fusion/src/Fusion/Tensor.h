#ifndef TENSOR_H
#define TENSOR_H

#include "Fusion/autodiff/AutodiffMode.hpp"
#include "core/tensor/RawTensor.hpp"

template <typename T> using Tensor = ADTensor<T>;
template <typename T> using BaseTensor = RawTensor<T>;

#endif // TENSOR_H
