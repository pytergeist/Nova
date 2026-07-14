#ifndef FUSION_CORE_OPS_EXECUTION_CPU_BINARY_ELEMENTWISE_H
#define FUSION_CORE_OPS_EXECUTION_CPU_BINARY_ELEMENTWISE_H

#include "Fusion/core/iter/DenseIter.hpp"
#include "Fusion/core/planning/OpContext.h"
#include "Fusion/cpu/simd/SimdTraits.hpp"

namespace fusion::execution::cpu {

namespace detail {

template <typename T, class Tag>
void binary_scalar_fallback(T *o, const T *a, const T *b, const int64_t &so,
                         const int64_t &sa, const int64_t &sb,
                         const std::size_t len) {
   Tag tag{};
   for (int64_t i = 0; i < len; ++i)
      o[i * so] = tag(a[i * sa], b[i * sb]);
}

} // namespace detail

template <typename T, class Tag, class TensorT>
void binary_elementwise(const TensorT &A, const TensorT &B,
                      const planning::BinaryEwiseContext &ctx, TensorT &out) {

   FUSION_CHECK(A.is_initialised(), "binary ewise: LHS uninitialised");
   FUSION_CHECK(B.is_initialised(), "binary ewise: RHS uninitialised");
   FUSION_CHECK(A.is_initialised() && B.is_initialised(),
                "uninitialised tensor");
   std::array<uint8_t *, 3> base = {
      reinterpret_cast<uint8_t *>(const_cast<T *>(out.get_ptr())),
      reinterpret_cast<uint8_t *>(const_cast<T *>(A.get_ptr())),
      reinterpret_cast<uint8_t *>(const_cast<T *>(B.get_ptr()))};

   if (ctx.exec == planning::BinaryExecKind::FlatContiguous) {
      auto *o = reinterpret_cast<T *>(base[0]);
      const auto *a = reinterpret_cast<const T *>(base[1]);
      const auto *b = reinterpret_cast<const T *>(base[2]);
      const size_t len = ctx.fast_len;
      if constexpr (simd_traits<Tag, T>::available) {
         simd_traits<Tag, T>::execute_contiguous(a, b, o, len, false, false);
      } else {
         tag_fallback_binary<T, Tag>(o, a, b, 1, 1, 1, len);
      }
      return;
   }
   const dense::iter::DenseIterPlanView view = dense::iter::dense_iter_view(ctx.plan);
   dense::iter::for_each_outer_then_inner<3>(view, base, [&](dense::iter::DenseSegment<3> &segment) {
      const std::int64_t step = sizeof(T);
      std::int64_t const out_bytes = segment.step[0].byte_stride;
      std::int64_t const a_bytes = segment.step[1].byte_stride;
      std::int64_t const b_bytes = segment.step[2].byte_stride;

      const bool out_contig = out_bytes == step;
      const bool a_unit = a_bytes == 0 || a_bytes == step;
      const bool b_unit = b_bytes == 0 || b_bytes == step;

      T *o = reinterpret_cast<T *>(segment.ptrs[0]);
      const T *a = reinterpret_cast<const T *>(segment.ptrs[1]);
      const T *b = reinterpret_cast<const T *>(segment.ptrs[2]);

      if constexpr (simd_traits<Tag, T>::available) {
         if (out_contig && a_unit && b_unit && segment.len > 0) {
            const bool a_scalar = a_bytes == 0;
            const bool b_scalar = b_bytes == 0;
            simd_traits<Tag, T>::execute_contiguous(
                a, b, o, static_cast<size_t>(segment.len), a_scalar, b_scalar);
            return;
         }
         if (out_contig && (a_unit || b_unit)) {
            const std::int64_t so = 1;
            const std::int64_t sa = a_bytes / step;
            const std::int64_t sb = b_bytes / step;
            detail::binary_scalar_fallback<T, Tag>(o, a, b, so, sa, sb, segment.len);
            return;
         }
      }

      const int64_t so = out_bytes / step;
      const int64_t sa = a_bytes == 0 ? 0 : a_bytes / step;
      const int64_t sb = b_bytes == 0 ? 0 : b_bytes / step;
      detail::binary_scalar_fallback<T, Tag>(o, a, b, so, sa, sb, segment.len);
   });
}
} // namespace fusion::execution::cpu

#endif // FUSION_CORE_OPS_EXECUTION_CPU_BINARY_ELEMENTWISE_H