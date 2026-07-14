#ifndef FUSION_CORE_OPS_EXECUTION_CPU_CONTRACTION_H
#define FUSION_CORE_OPS_EXECUTION_CPU_CONTRACTION_H

#include "Fusion/core/planning/OpContext.h"
#include "Fusion/cpu/simd/SimdTraits.hpp"
#include "Fusion/core/iter/DenseIter.hpp"

namespace fusion::execution::cpu {
namespace detail {
template <typename T, class Tag>
void contraction_scalar_fallback(T *o, const T *a, const T *b, const int64_t &so,
                              const int64_t &sa, const int64_t &sb,
                              const std::size_t len) {
   Tag tag{};
   for (int64_t i = 0; i < static_cast<int64_t>(len); ++i) {
      o[i * so] += tag(a[i * sa], b[i * sb]);
   }
}
} // namespace contraction

template <typename T, class BlasTag, class ScalarTag, class TensorT>
void contraction(const TensorT &A, const TensorT &B,
                     planning::ContractionContext &meta, TensorT &out_data) {

   auto *out = reinterpret_cast<T *>(out_data.get_ptr());
   std::fill(out, out + out_data.flat_size(), T{0});

   if constexpr (fusion::blas::blas_traits<BlasTag, T>::available) {
      if (meta.plan.exec.hints.gemm_like) {
         const auto &g = meta.plan.exec.hints.gemm;
         if (fusion::blas::blas_traits<BlasTag, T>::can_execute(g)) {
            const T *baseA = reinterpret_cast<const T *>(A.get_ptr());
            const T *baseB = reinterpret_cast<const T *>(B.get_ptr());
            T *baseC = reinterpret_cast<T *>(out_data.get_ptr());
            fusion::blas::blas_traits<BlasTag, T>::execute(baseA, baseB, baseC,
                                                           g, T(1), T(0));
            return;
         }
      }
   }

   std::array<uint8_t *, 3> base = {
       reinterpret_cast<uint8_t *>(out),
       reinterpret_cast<uint8_t *>(const_cast<T *>(A.get_ptr())),
       reinterpret_cast<uint8_t *>(const_cast<T *>(B.get_ptr())),
   };

   const dense::iter::DenseIterPlanView view = dense::iter::dense_iter_view(meta.plan);
   for_each_outer_then_inner<3>(view, base, [&](dense::iter::DenseSegment<3> &segment) {
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

      detail::contraction_scalar_fallback<T, ScalarTag>(
          o, a, b, so, sa, sb, static_cast<std::size_t>(segment.len));
   });
}

}

#endif // FUSION_CORE_OPS_EXECUTION_CPU_CONTRACTION_H