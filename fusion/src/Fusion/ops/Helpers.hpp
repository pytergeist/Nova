#ifndef OP_HELPERS_H
#define OP_HELPERS_H

#include <cassert>

#include "Fusion/Tensor.h"
#include "Fusion/core/EwiseMeta.h"

template <typename T>
inline TensorBase<T> init_out_from_meta(const TensorBase<T> &x,
                                        const TensorBase<T> &y,
                                        const BinaryEwiseMeta &m) {
   FUSION_CHECK(x.dtype() == y.dtype(), "dtypes do not match!");
   FUSION_CHECK(x.device() == y.device(), "devices do not match!");
   return TensorBase<T>(m.out_shape, x.dtype(), x.device());
}

template <typename T>
inline TensorBase<T> init_out_from_meta(const TensorBase<T> &x,
                                        const UnaryEwiseMeta &m) {
   return TensorBase<T>(m.out_shape, x.dtype(), x.device());
}

#endif // OP_HELPERS_H
