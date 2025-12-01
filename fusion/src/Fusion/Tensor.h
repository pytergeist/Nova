#ifndef TENSOR_H
#define TENSOR_H

#include "Fusion/core/TensorBase.h"
#include "Fusion/autodiff/AutodiffMode.h"

template <typename T> using Tensor = ADTensor<T>;
template <typename T> using RawTensor = ADTensor<T>;

#endif // TENSOR_H
