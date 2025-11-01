#ifndef OP_HELPERS_H
#define OP_HELPERS_H

#include "../Tensor.h"
#include "../core/Device.h"

template <typename T> bool grad_flow(const Tensor<T> &x, const Tensor<T> &y) {
   return x.requires_grad() || y.requires_grad();
};


template <typename T>
inline Tensor<T> init_bin_out_tensor(const Tensor<T> &x, const Tensor<T> &y) {

   return Tensor<T>(x.shape(), Device::CPU, grad_flow(x, y)); // TODO: This is LHS aligned on shape
}

template <typename T>
inline Tensor<T> init_un_out_tensor(const Tensor<T> &x) {

   return Tensor<T>(x.shape(), Device::CPU, x.requires_grad()); // TODO: This is LHS aligned on shape
}

#endif // OP_HELPERS_H
