#ifndef FUSION_CORE_OPS_EXECUTION_CPU_REDUCTION_H
#define FUSION_CORE_OPS_EXECUTION_CPU_REDUCTION_H

#include "Fusion/core/iter/DenseIter.hpp"
#include "Fusion/core/planning/OpContext.h"
#include "Fusion/cpu/simd/SimdTraits.hpp"

namespace fusion::execution::cpu {
namespace detail {

template <class OpTag> struct ReductionKernelFor;

template <> struct ReductionKernelFor<SumTag> {
   using type = SumSIMD;
   template <typename T>
   static constexpr T finalise(T accumulated,
                               std::size_t /*reduce_len*/) noexcept {

      return accumulated;
   }
};

template <> struct ReductionKernelFor<MeanTag> {
   using type = SumSIMD;

   template <typename T>
   static constexpr T finalise(T accumulated, std::size_t reduce_len) noexcept {

      return accumulated / static_cast<T>(reduce_len);
   }
};

template <class OpTag>
using reduction_kernel_for_t = typename ReductionKernelFor<OpTag>::type;

template <typename T, class Kernel>
void reduction_scalar_fallback(T *o, const T *a, const int64_t &so,
                               const int64_t &sa, const std::size_t len) {
   Kernel kernel{};
   for (int64_t i = 0; i < len; ++i)
      o[i * so] += kernel(a[i * sa]);
}

} // namespace detail

template <typename T, class OpTag>
void reduction(T *out, const T *operand, const std::size_t out_size,
               planning::ReductionContext &ctx) {
   using Reduction = detail::ReductionKernelFor<OpTag>;
   using Kernel = typename Reduction::type;
   std::array<uint8_t *, 2> base = {
       reinterpret_cast<uint8_t *>(const_cast<T *>(out)),
       reinterpret_cast<uint8_t *>(const_cast<T *>(operand)),
   };

   if (ctx.fastpath) {
      auto *o = reinterpret_cast<T *>(base[0]);
      const auto *a = reinterpret_cast<const T *>(base[1]);
      const size_t len = ctx.fast_len;
      if constexpr (simd_traits<Kernel, T>::available) {
         *o += simd_traits<Kernel, T>::reduce_contiguous(a, len);
      } else {
         detail::reduction_scalar_fallback<T, Kernel>(o, a, 1, 1, len);
      }
      *o = Reduction::template finalise<T>(*o, ctx.reduce_len);
      return;
   }
   const dense::iter::DenseIterPlanView view =
       dense::iter::dense_iter_view(ctx.plan);
   for_each_outer_then_inner<2>(
       view, base, [&](dense::iter::DenseSegment<2> &segment) {
          const std::int64_t step = sizeof(T);
          std::int64_t const out_bytes = segment.step[0].byte_stride;
          std::int64_t const a_bytes = segment.step[1].byte_stride;

          T *o = reinterpret_cast<T *>(segment.ptrs[0]);
          const T *a = reinterpret_cast<const T *>(segment.ptrs[1]);

          if constexpr (simd_traits<Kernel, T>::available) {
             if (out_bytes == 0 && a_bytes == step && segment.len > 0) {
                *o += simd_traits<Kernel, T>::reduce_contiguous(
                    a, static_cast<size_t>(segment.len));
                return;
             }
          }

          const std::int64_t so = out_bytes / step;
          const std::int64_t sa = (a_bytes == 0) ? 0 : a_bytes / step;
          detail::reduction_scalar_fallback<T, Kernel>(o, a, so, sa,
                                                       segment.len);
       });
   for (std::size_t i = 0; i < out_size;
        ++i) { // TODO: find a way to remove this/manage it better
      out[i] = Reduction::template finalise<T>(out[i], ctx.reduce_len);
   }
}
} // namespace fusion::execution::cpu

#endif // FUSION_CORE_OPS_EXECUTION_CPU_REDUCTION_H