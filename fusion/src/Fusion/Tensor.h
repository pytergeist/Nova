#ifndef TENSOR_H
#define TENSOR_H


#include "Fusion/core/TensorBase.h"
#include "Fusion/autodiff/ADTensor.h"

template <typename T>
class Tensor : public ADTensor<T> {
public:
    using ADTensor<T>::ADTensor;
};


template <typename T> using RawTensor = ADTensor<T>;

#endif // TENSOR_H
