#ifndef TENSOR_H
#define TENSOR_H

#include "Fusion/autodiff/AutodiffMode.h"
#include "Fusion/core/TensorBase.h"

template <typename T> using Tensor = ADTensor<T>;
template <typename T> using RawTensor = TensorBase<T>;

#endif // TENSOR_H
