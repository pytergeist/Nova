#ifndef OPS_LINALG_HPP
#define OPS_LINALG_HPP

#include <string_view>
#include <vector>

#include "Fusion/common/Log.hpp"
#include "Fusion/core/ElementWise.hpp"
#include "Fusion/core/RawTensor.hpp"
#include "Fusion/cpu/blas/Gemm.hpp"

#include "Helpers.hpp"

namespace fusion {

namespace math {

namespace linalg {

template <typename T>
inline RawTensor<T>
matmul(const RawTensor<T> &x,
       const RawTensor<T> &y) { // TODO: this uses vector obj copying and
                                 // doesn't go through broadcast layer?
   assert((x.dtype_size() == y.dtype_size()) &&
          "binary op: dtype sizes must match"); // TODO: abstract into macro
                                                // (change from assert)
   auto const &shapeA = x.shape();
   auto const &shapeB = y.shape();

   size_t rank = shapeA.size();
   int m = int(shapeA[rank - 2]);
   int k = int(shapeA[rank - 1]);
   int n = int(shapeB[rank - 1]);

   std::vector<size_t> out_shape = shapeA;
   out_shape[rank - 1] = n;

   size_t batch = 1;
   for (size_t i = 0; i < rank - 2; ++i) {
      batch *= shapeA[i];
   }

   std::vector<T> data(size_t(batch) * m * n);

   const T *baseA = x.raw_data().template data_as<const T>();
   const T *baseB = y.raw_data().template data_as<const T>();
   T *baseC = data.data();

   blas_ops::batched_gemm<T>(baseA, baseB, baseC, m, n, k, batch, T(1), T(0));
   return RawTensor<T>(std::move(out_shape), std::move(data), x.dtype(),
                        x.device());
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
inline RawTensor<T> swapaxes(const RawTensor<T> &x, const int axis1,
                              const int axis2) {
   std::vector<size_t> out_shape = x.shape();
   const int nd = static_cast<int>(out_shape.size());
   if (nd < 2) {
      return RawTensor<T>(out_shape, std::vector<T>(x.begin(), x.end()),
                           x.dtype(), x.device());
   }
   const int naxis1 = serial::normalise_axis(axis1, nd);
   const int naxis2 = serial::normalise_axis(axis2, nd);
   if (axis1 == axis2) {
      return RawTensor<T>(out_shape, std::vector<T>(x.begin(), x.end()),
                           x.dtype(), x.device());
   }
   std::swap(out_shape[naxis1], out_shape[naxis2]);
   std::vector<T> out = serial::swapaxes<T>(x, x.shape(), naxis1, naxis2);
   return RawTensor<T>(std::move(out_shape), std::move(out), x.dtype(),
                        x.device());
}

} // namespace linalg

} // namespace math

} // namespace fusion

#endif // OPS_LINALG_HPP
