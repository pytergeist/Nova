#ifndef OP_HELPERS_H
#define OP_HELPERS_H

#include <cassert>

#include "Fusion/Tensor.h"
#include "Fusion/core/Device.h"
#include "Fusion/core/EwiseMeta.h"

template <typename T> bool grad_flow(const Tensor<T> &x, const Tensor<T> &y) {
   return x.requires_grad() || y.requires_grad();
};


template <typename T>
inline Tensor<T> init_out_from_meta(const Tensor<T> &x, const Tensor<T> &y, const BinaryEwiseMeta& m) {
    FUSION_CHECK(x.dtype() == y.dtype(), "dtypes do not match!");
    return Tensor<T>(m.out_shape, Device::CPU, x.dtype(), grad_flow(x, y));
}

template <typename T>
inline Tensor<T> init_out_from_meta(const Tensor<T> &x, const UnaryEwiseMeta& m) {
   return Tensor<T>(m.out_shape, Device::CPU, x.dtype(), x.requires_grad()); // TODO: This is LHS aligned on shape
}

#endif // OP_HELPERS_H
