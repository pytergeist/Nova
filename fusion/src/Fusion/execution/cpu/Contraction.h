#ifndef FUSION_CORE_OPS_EXECUTION_CPU_CONTRACTION_H
#define FUSION_CORE_OPS_EXECUTION_CPU_CONTRACTION_H

#include "Fusion/core/iter/DenseIter.hpp"
#include "Fusion/core/planning/OpContext.h"
#include "Fusion/cpu/simd/SimdTraits.hpp"
#include "Fusion/core/opschema/OpTags.h"

namespace fusion::execution::cpu {
namespace detail {

template <class OpTag, class ScalarTag>
struct ContractionKernelFor;

template <>
struct ContractionKernelFor<MatMulTag, MulTag> {
   using type = BatchedGemmBLAS;
   using scalar_type = MultiplySIMD;
};

template <class OpTag, class ScalarTag>
using contraction_kernel_for_t = typename ContractionKernelFor<OpTag, ScalarTag>::type;

template <class OpTag, class ScalarTag>
using contraction_scalar_kernel_for_t = typename ContractionKernelFor<OpTag, ScalarTag>::scalar_type;


template <typename T, class ScalarKernel>
void contraction_scalar_fallback(T *o, const T *a, const T *b,
                                 const int64_t &so, const int64_t &sa,
                                 const int64_t &sb, const std::size_t len) {
   ScalarKernel kernel{};
   for (int64_t i = 0; i < static_cast<int64_t>(len); ++i) {
      o[i * so] += kernel(a[i * sa], b[i * sb]);
   }
}
} // namespace detail

template <typename T, class OpTag, class ScalarTag>
void contraction(T* out, const T* lhs, const T* rhs,
                 planning::ContractionContext &meta) {

   using Kernel = detail::contraction_kernel_for_t<OpTag, ScalarTag>;
   using ScalarKernel = detail::contraction_scalar_kernel_for_t<OpTag, ScalarTag>;

   if constexpr (blas::blas_traits<Kernel, T>::available) {
      if (meta.plan.exec.hints.gemm_like) {
         const auto &g = meta.plan.exec.hints.gemm;
         if (blas::blas_traits<Kernel, T>::can_execute(g)) {
            const T *baseA = reinterpret_cast<const T *>(lhs);
            const T *baseB = reinterpret_cast<const T *>(rhs);
            T *baseC = reinterpret_cast<T *>(out);
            blas::blas_traits<Kernel, T>::execute(baseA, baseB, baseC,
                                                           g, T(1), T(0));
            return;
         }
      }
   }

   std::array<uint8_t *, 3> base = {
       reinterpret_cast<uint8_t *>(out),
       reinterpret_cast<uint8_t *>(const_cast<T *>(lhs)),
       reinterpret_cast<uint8_t *>(const_cast<T *>(rhs)),
   };

   const dense::iter::DenseIterPlanView view =
       dense::iter::dense_iter_view(meta.plan);
   for_each_outer_then_inner<3>(
       view, base, [&](dense::iter::DenseSegment<3> &segment) {
          const int64_t step = sizeof(T);
          std::int64_t const out_bytes = segment.step[0].byte_stride;
          std::int64_t const a_bytes = segment.step[1].byte_stride;
          std::int64_t const b_bytes = segment.step[2].byte_stride;

          auto *o = reinterpret_cast<T *>(segment.ptrs[0]);
          auto *a = reinterpret_cast<const T *>(segment.ptrs[1]);
          auto *b = reinterpret_cast<const T *>(segment.ptrs[2]);

          const int64_t so = out_bytes == 0 ? 0 : out_bytes / step;
          const int64_t sa = a_bytes == 0 ? 0 : a_bytes / step;
          const int64_t sb = b_bytes == 0 ? 0 : b_bytes / step;

          detail::contraction_scalar_fallback<T, ScalarKernel>(
              o, a, b, so, sa, sb, static_cast<std::size_t>(segment.len));
       });
}

} // namespace fusion::execution::cpu

#endif // FUSION_CORE_OPS_EXECUTION_CPU_CONTRACTION_H