#ifndef FUSION_EXECUTION_CPU_CONTRACTION_H
#define FUSION_EXECUTION_CPU_CONTRACTION_H

#include <array>
#include <cstddef>
#include <cstdint>

#include "Fusion/core/iter/DenseIter.hpp"
#include "Fusion/core/opschema/OpTags.h"
#include "Fusion/core/planning/OpContext.h"
#include "Fusion/cpu/blas/BlasTraits.hpp"
#include "Fusion/cpu/simd/SimdTraits.hpp"

namespace fusion::execution::cpu {

namespace detail {

template <class OpTag, class ScalarTag> struct ContractionKernelFor;

template <> struct ContractionKernelFor<MatMulTag, MulTag> {
   using type = BatchedGemmBLAS;
   using scalar_type = MultiplySIMD;
};

template <class OpTag, class ScalarTag>
using contraction_kernel_for_t =
    typename ContractionKernelFor<OpTag, ScalarTag>::type;

template <class OpTag, class ScalarTag>
using contraction_scalar_kernel_for_t =
    typename ContractionKernelFor<OpTag, ScalarTag>::scalar_type;

template <typename T, class ScalarKernel>
void contraction_scalar_fallback(T *out, const T *lhs, const T *rhs,
                                 std::int64_t out_stride,
                                 std::int64_t lhs_stride,
                                 std::int64_t rhs_stride, std::int64_t len) {
   ScalarKernel kernel{};

   for (std::int64_t i = 0; i < len; ++i) {
      out[i * out_stride] += kernel(lhs[i * lhs_stride], rhs[i * rhs_stride]);
   }
}

template <typename T, class ScalarKernel>
void execute_contraction_segment(
    const dense::iter::DenseSegmentView<2, 1> &segment) {
   if (segment.empty()) {
      return;
   }

   T *out = segment.template output<T>(0);
   const T *lhs = segment.template input<T>(0);
   const T *rhs = segment.template input<T>(1);

   const std::int64_t out_stride =
       segment.output_stride(0).template element_stride<T>();

   const std::int64_t lhs_stride =
       segment.input_stride(0).template element_stride<T>();

   const std::int64_t rhs_stride =
       segment.input_stride(1).template element_stride<T>();

   contraction_scalar_fallback<T, ScalarKernel>(
       out, lhs, rhs, out_stride, lhs_stride, rhs_stride, segment.len);
}

} // namespace detail

template <typename T, class OpTag, class ScalarTag>
void contraction(T *out, const T *lhs, const T *rhs,
                 const planning::ContractionContext &ctx) {
   using Kernel = detail::contraction_kernel_for_t<OpTag, ScalarTag>;

   using ScalarKernel =
       detail::contraction_scalar_kernel_for_t<OpTag, ScalarTag>;

   if constexpr (blas::blas_traits<Kernel, T>::available) {
      if (ctx.plan.exec.hints.gemm_like) {
         const auto &gemm = ctx.plan.exec.hints.gemm;

         if (blas::blas_traits<Kernel, T>::can_execute(gemm)) {
            blas::blas_traits<Kernel, T>::execute(lhs, rhs, out, gemm, T{1},
                                                  T{0});

            return;
         }
      }
   }

   const dense::iter::DenseIterPlanView view =
       dense::iter::dense_iter_view(ctx.plan);

   std::array<std::byte *, 1> outputs{
       reinterpret_cast<std::byte *>(out),
   };

   std::array<const std::byte *, 2> inputs{
       reinterpret_cast<const std::byte *>(lhs),
       reinterpret_cast<const std::byte *>(rhs),
   };

   dense::iter::for_each_outer_then_inner<2, 1>(
       view, outputs, inputs,
       [](const dense::iter::DenseSegmentView<2, 1> &segment) {
          detail::execute_contraction_segment<T, ScalarKernel>(segment);
       });
}

} // namespace fusion::execution::cpu

#endif // FUSION_EXECUTION_CPU_CONTRACTION_H