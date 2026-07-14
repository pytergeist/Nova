#ifndef FUSION_CORE_OPS_EXECUTION_CPU_REDUCTION_H
#define FUSION_CORE_OPS_EXECUTION_CPU_REDUCTION_H

#include "Fusion/core/iter/DenseIter.hpp"
#include "Fusion/core/planning/OpContext.h"
#include "Fusion/cpu/simd/SimdTraits.hpp"

namespace fusion::execution::cpu {
namespace detail {

template <typename T, class Tag>
void reduction_scalar_fallback(T *o, const T *a, const int64_t &so,
                            const int64_t &sa, const std::size_t len) {
   Tag tag{};
   for (int64_t i = 0; i < len; ++i)
      o[i * so] += tag(a[i * sa]);
}


}// namespace detail

template <typename T, class Tag, class TensorT>
void reduction(const TensorT &A, planning::ReductionContext &ctx,
                   TensorT &out_data) {

   auto *out = reinterpret_cast<T *>(out_data.get_ptr());
   std::fill(out, out + out_data.flat_size(), T{0});
   std::array<uint8_t *, 2> base = {
      reinterpret_cast<uint8_t *>(const_cast<T *>(out)),
      reinterpret_cast<uint8_t *>(const_cast<T *>(A.get_ptr())),
  };

   if (ctx.fastpath) {
      auto *o = reinterpret_cast<T *>(base[0]);
      const auto *a = reinterpret_cast<const T *>(base[1]);
      const size_t len = ctx.fast_len;
      if constexpr (simd_traits<Tag, T>::available) {
         *o += simd_traits<Tag, T>::reduce_contiguous(a, len);
      } else {
         detail::reduction_scalar_fallback<T, Tag>(o, a, 1, 1, len);
      }
      return;
   }
   const dense::iter::DenseIterPlanView view = dense::iter::dense_iter_view(ctx.plan);
   for_each_outer_then_inner<2>(view, base, [&](dense::iter::DenseSegment<2> &segment) {
      const std::int64_t step = sizeof(T);
      std::int64_t const out_bytes = segment.step[0].byte_stride;
      std::int64_t const a_bytes = segment.step[1].byte_stride;

      T *o = reinterpret_cast<T *>(segment.ptrs[0]);
      const T *a = reinterpret_cast<const T *>(segment.ptrs[1]);

      if constexpr (simd_traits<Tag, T>::available) {
         if (out_bytes == 0 && a_bytes == step && segment.len > 0) {
            *o += simd_traits<Tag, T>::reduce_contiguous(
                a, static_cast<size_t>(segment.len));
            return;
         }
      }

      const std::int64_t so = out_bytes / step;
      const std::int64_t sa = (a_bytes == 0) ? 0 : a_bytes / step;
      Tag tag{};
      detail::reduction_scalar_fallback<T, Tag>(o, a, so, sa, segment.len);
   });
}
} // namespace fusion::execution::cpu

#endif // FUSION_CORE_OPS_EXECUTION_CPU_REDUCTION_H