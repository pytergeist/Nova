#ifndef FUSION_PUBLIC_TENSOR_H
#define FUSION_PUBLIC_TENSOR_H

#include "Fusion/autodiff/ADTensor.hpp"
#include "Fusion/core/tensor/AoSoATensor.hpp"
#include "Fusion/core/tensor/DenseTensor.hpp"
#include "Fusion/core/tensor/SoATensor.hpp"
#include "Fusion/core/tensor/Tensor.hpp"

namespace fusion {

template <typename T> using ValueTensor = Tensor<T>;

template <typename T> using DifferentiableTensor = ADTensor<T>;

template <typename T> using DenseTensor = DenseTensor<T>;

} // namespace fusion

#endif // FUSION_PUBLIC_TENSOR_H