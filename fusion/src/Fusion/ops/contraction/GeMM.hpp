#ifndef FUSION_OPS_ALIGNED_GEMM_HPP
#define FUSION_OPS_ALIGNED_GEMM_HPP

#include <string_view>
#include <vector>

#include "Fusion/common/Log.hpp"
#include "Fusion/core/planning/OpContextBuilders.h"
#include "Fusion/execution/cpu/Contraction.h"
#include "Fusion/kernels/Serial.hpp"

#include "Fusion/ops/OperandValidation.h"
#include "Fusion/ops/OutputAllocation.h"

namespace fusion::ops::contraction {

template <typename T>
DenseTensor<T> matmul(const DenseTensor<T> &A, const DenseTensor<T> &B) {
   validation::validate_dense_contraction_operation<T, MatMulTag>(A, B);

   const auto &a_shape = A.shape();
   const auto &b_shape = B.shape();
   if (a_shape.size() < 2 || b_shape.size() < 2)
      throw std::runtime_error("matmul: expected rank >= 2");

   const std::size_t kA = a_shape[a_shape.size() - 1];
   const std::size_t kB = b_shape[b_shape.size() - 2];
   if (kA != kB)
      throw std::runtime_error("matmul: inner dimension mismatch");

   planning::ContractionContext ctx = planning::make_matmul_context<T>(A, B);

   DenseTensor<T> out = detail::init_out_from_meta(A, B, ctx);

   execution::cpu::contraction<T, BatchedGemmBLAS, MultiplySIMD>(A, B, ctx,
                                                                 out);

   return out;
}

template <typename T>
DenseTensor<T> swapaxes(const DenseTensor<T> &x, const int axis1,
                        const int axis2) {
   std::vector<size_t> out_shape = x.shape();
   const int nd = static_cast<int>(out_shape.size());
   if (nd < 2) {
      return DenseTensor<T>(out_shape, std::vector<T>(x.begin(), x.end()),
                            x.dtype(), x.device());
   }
   const int naxis1 = serial::normalise_axis(axis1, nd);
   const int naxis2 = serial::normalise_axis(axis2, nd);
   if (naxis1 == naxis2) {
      return DenseTensor<T>(out_shape, std::vector<T>(x.begin(), x.end()),
                            x.dtype(), x.device());
   }
   std::swap(out_shape[naxis1], out_shape[naxis2]);
   std::vector<T> out = serial::swapaxes<T>(x, x.shape(), naxis1, naxis2);
   return DenseTensor<T>(std::move(out_shape), std::move(out), x.dtype(),
                         x.device());
}

} // namespace fusion::ops::contraction

#endif // FUSION_OPS_ALIGNED_GEMM_HPP
