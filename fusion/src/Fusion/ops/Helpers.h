#ifndef OP_HELPERS_H
#define OP_HELPERS_H

#include <cassert>
#include "../Tensor.h"
#include "../core/Device.h"

template <typename T> bool grad_flow(const Tensor<T> &x, const Tensor<T> &y) {
   return x.requires_grad() || y.requires_grad();
};


template <typename T>
inline Tensor<T> init_bin_out_tensor(const Tensor<T> &x, const Tensor<T> &y) {
   assert((x.dtype_size() == y.dtype_size()) && "binary op: dtype sizes must match" ); // TODO: abstract into macro (change from assert)
   if ((x.shape() == y.shape()) && (x.is_contiguous() && y.is_contiguous())) { // TODO: make macro for equal shape/contig path
      return Tensor<T>(x.shape(), Device::CPU, x.dtype(), grad_flow(x, y));
   }
   auto dA = ewise::make_desc<T>(x.shape(), nullptr); // TODO: abstract this out of here or pass in plan? duplicate code
   auto dB = ewise::make_desc<T>(y.shape(), nullptr);
   auto plan_in = make_broadcast_plan({dA, dB});
   std::vector<size_t> out_shape(plan_in.out_sizes.begin(), plan_in.out_sizes.end());
   return Tensor<T>(out_shape, Device::CPU, x.dtype(), grad_flow(x, y)); // TODO: This is LHS aligned on shape
}

template <typename T>
inline Tensor<T> init_un_out_tensor(const Tensor<T> &x) {

   return Tensor<T>(x.shape(), Device::CPU, x.dtype(), x.requires_grad()); // TODO: This is LHS aligned on shape
}

#endif // OP_HELPERS_H
