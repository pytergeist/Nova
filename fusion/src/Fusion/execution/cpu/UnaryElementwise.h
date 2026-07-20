#ifndef FUSION_CORE_OPS_EXECUTION_CPU_UNARY_ELEMENTWISE_H
#define FUSION_CORE_OPS_EXECUTION_CPU_UNARY_ELEMENTWISE_H

#include "Fusion/core/iter/DenseIter.hpp"
#include "Fusion/core/planning/OpContext.h"
#include "Fusion/cpu/simd/SimdTraits.hpp"

namespace fusion::execution::cpu {

namespace detail {

template <class OpTag>
struct UnaryKernelFor;

template <>
struct UnaryKernelFor<SqrtTag> {
   using type = SqrtSIMD;
};

template <>
struct UnaryKernelFor<LogTag> {
   using type = NaturalLogSIMD;
};

template <>
struct UnaryKernelFor<ExpTag> {
   using type = ExponentialSIMD;
};

template <>
struct UnaryKernelFor<ReciprocalTag> {
   using type = ReciprocalSIMD;
};

template <class OpTag>
using unary_kernel_for_t = typename UnaryKernelFor<OpTag>::type;

template <typename T, class Kernel>
void unary_scalar_fallback(T *o, const T *a, const int64_t &so,
                           const int64_t &sa, const std::size_t len) {
   Kernel kernel{};
   for (int64_t i = 0; i < len; ++i)
      o[i * so] = kernel(a[i * sa]);
}

} // namespace detail

template <typename T, class OpTag>
void unary_elementwise(T* out, const T* operand, planning::UnaryEwiseContext &ctx) {
   using Kernel = detail::unary_kernel_for_t<OpTag>;
   std::array<uint8_t *, 2> base = { // TODO: change ownership model away from array of ptrs
       reinterpret_cast<uint8_t *>(const_cast<T *>(out)),
       reinterpret_cast<uint8_t *>(const_cast<T *>(operand)),
   };

   if (ctx.fastpath) { // TODO: is contig check correct here?
      auto *o = reinterpret_cast<T *>(base[0]);
      const auto *a = reinterpret_cast<const T *>(base[1]);
      const size_t len = ctx.fast_len;
      if constexpr (simd_traits<Kernel, T>::available) {
         simd_traits<Kernel, T>::execute_contiguous(a, o, len, false);
      } else {
         detail::unary_scalar_fallback<T, Kernel>(o, a, 1, 1, len);
      }
      return;
   }
   const dense::iter::DenseIterPlanView view =
       dense::iter::dense_iter_view(ctx.plan);
   for_each_outer_then_inner<2>(
       view, base, [&](dense::iter::DenseSegment<2> &segment) {
          const std::int64_t step = sizeof(T);
          std::int64_t const out_bytes = segment.step[0].byte_stride;
          std::int64_t const a_bytes = segment.step[1].byte_stride;
          const bool out_contig = out_bytes == step;
          const bool a_unit = a_bytes == 0 || a_bytes == step;

          T *o = reinterpret_cast<T *>(segment.ptrs[0]);
          const T *a = reinterpret_cast<const T *>(segment.ptrs[1]);

          if constexpr (simd_traits<Kernel, T>::available) {

             if (out_contig && a_unit && segment.len > 0) {
                const bool a_scalar = a_bytes == 0;
                simd_traits<Kernel, T>::execute_contiguous(
                    a, o, static_cast<size_t>(segment.len), a_scalar);
                return;
             }
             const bool a_unit = a_bytes == step;
             if (out_contig && a_unit) {
                const int64_t so = 1;
                const int64_t sa = a_bytes / step;
                detail::unary_scalar_fallback<T, Kernel>(o, a, so, sa,
                                                      segment.len);
                return;
             }
          }

          const int64_t so = a_bytes / step;
          const int64_t sa = a_bytes == 0 ? 0 : a_bytes / step;
          detail::unary_scalar_fallback<T, Kernel>(o, a, so, sa, segment.len);
       });
}

} // namespace fusion::execution::cpu

#endif // FUSION_CORE_OPS_EXECUTION_CPU_UNARY_ELEMENTWISE_H