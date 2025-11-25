#ifndef OPS_LINALG_H
#define OPS_LINALG_H

#include <vector>
#include <string_view>
#include "../Tensor.h"
#include "../common/Log.h"
#include "../core/ElementWise.h"
#include "Helpers.h"
#include "../cpu/blas/Gemm.h"

namespace math {
namespace linalg {
template <typename T>
inline Tensor<T> matmul(const Tensor<T> &x, const Tensor<T> &y) { // TODO: this uses vector obj copying and doesn't go through broadcast layer?
   assert((x.dtype_size() == y.dtype_size()) && "binary op: dtype sizes must match" ); // TODO: abstract into macro (change from assert)
   auto const &shapeA = x.shape();
   auto const &shapeB = y.shape();
   size_t rank = shapeA.size();
   size_t m = shapeA[rank - 2];
   size_t n = shapeB[rank - 1];
   std::vector<size_t> out_shape = shapeA;
   out_shape[rank - 1] = n;
   size_t batch = 1;
   for (size_t i = 0; i < rank - 2; ++i) {
      batch *= shapeA[i];
   }
   std::vector<T> data(batch * m * n);
   blas_ops::gemm<T>(x, shapeA, y, shapeB, data, T(1), T(0));
   return Tensor<T>(std::move(out_shape), std::move(data), x.dtype(), Device::CPU,
                    grad_flow(x, y));
}

std::string shape_str(std::vector<size_t> shape) {
   std::ostringstream oss;
   oss << '(';
   for (size_t i = 0; i < shape.size(); ++i) {
      oss << shape[i];
      if (i + 1 < shape.size())
         oss << ',';
   }
   oss << ')';
   return oss.str();
}

template <typename T>
inline Tensor<T> swapaxes(const Tensor<T> &x, int axis1, int axis2) {
   std::vector<size_t> out_shape = x.shape();
   const int nd = static_cast<int>(out_shape.size());
   if (nd < 2) {
      return Tensor<T>(out_shape, std::vector<T>(x.begin(), x.end()), x.dtype(),
                       Device::CPU, x.requires_grad());
   }
   axis1 = serial::normalise_axis(axis1, nd);
   axis2 = serial::normalise_axis(axis2, nd);
   if (axis1 == axis2) {
      return Tensor<T>(out_shape, std::vector<T>(x.begin(), x.end()), x.dtype(),
                       Device::CPU, x.requires_grad());
   }
   std::swap(out_shape[axis1], out_shape[axis2]);
   std::vector<T> out = serial::swapaxes<T>(x, x.shape(), axis1, axis2);
   return Tensor<T>(std::move(out_shape), std::move(out), x.dtype(), Device::CPU);
}
} // namespace linalg

} // namespace math

#endif // OPS_LINALG_H
