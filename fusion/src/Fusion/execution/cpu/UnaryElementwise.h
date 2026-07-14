#ifndef FUSION_CORE_OPS_EXECUTION_CPU_UNARY_ELEMENTWISE_H
#define FUSION_CORE_OPS_EXECUTION_CPU_UNARY_ELEMENTWISE_H

#include "Fusion/core/iter/DenseIter.hpp"
#include "Fusion/core/planning/OpContext.h"
#include "Fusion/cpu/simd/SimdTraits.hpp"

namespace fusion::execution::cpu {

namespace detail {

template <typename T, class Tag>
void unary_scalar_fallback(T *o, const T *a, const int64_t &so, const int64_t &sa,
                        const std::size_t len) {
   Tag tag{};
   for (int64_t i = 0; i < len; ++i)
      o[i * so] = tag(a[i * sa]);
}

} // namespace detail

template <typename T, class Tag, class TensorT>
void unary_elementwise(const TensorT &A, planning::UnaryEwiseContext &ctx,
                     TensorT &out_data) {

   std::array<uint8_t *, 2> base = {
       reinterpret_cast<uint8_t *>(out_data.get_ptr()),
       reinterpret_cast<uint8_t *>(const_cast<T *>(A.get_ptr())),
   };

   if (ctx.fastpath) { // TODO: is contig check correct here?
      auto *o = reinterpret_cast<T *>(base[0]);
      const auto *a = reinterpret_cast<const T *>(base[1]);
      const size_t len = ctx.fast_len;
      if constexpr (simd_traits<Tag, T>::available) {
         simd_traits<Tag, T>::execute_contiguous(a, o, len, false);
      } else {
         tag_fallback_unary<T, Tag>(o, a, 1, 1, len);
      }
      return;
   }
   const dense::iter::DenseIterPlanView view = dense::iter::dense_iter_view(ctx.plan);
   for_each_outer_then_inner<2>(view, base, [&](dense::iter::DenseSegment<2> &segment) {
      const std::int64_t step = sizeof(T);
      std::int64_t const out_bytes = segment.step[0].byte_stride;
      std::int64_t const a_bytes = segment.step[1].byte_stride;
      const bool out_contig = out_bytes == step;
      const bool a_unit = a_bytes == 0 || a_bytes == step;

      T *o = reinterpret_cast<T *>(segment.ptrs[0]);
      const T *a = reinterpret_cast<const T *>(segment.ptrs[1]);

      if constexpr (simd_traits<Tag, T>::available) {

         if (out_contig && a_unit && segment.len > 0) {
            const bool a_scalar = a_bytes == 0;
            simd_traits<Tag, T>::execute_contiguous(
                a, o, static_cast<size_t>(segment.len), a_scalar);
            return;
         }
         const bool a_unit = a_bytes == step;
         if (out_contig && a_unit) {
            const int64_t so = 1;
            const int64_t sa = a_bytes / step;
            detail::unary_scalar_fallback<T, Tag>(o, a, so, sa, segment.len);
            return;
         }
      }

      const int64_t so = a_bytes / step;
      const int64_t sa = a_bytes == 0 ? 0 : a_bytes / step;
      Tag tag{};
      detail::unary_scalar_fallback<T, Tag>(o, a, so, sa, segment.len);
   });
}

} // namespace fusion::execution::cpu

#endif // FUSION_CORE_OPS_EXECUTION_CPU_UNARY_ELEMENTWISE_H