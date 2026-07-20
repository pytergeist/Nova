#ifndef FUSION_CORE_OPS_EXECUTION_CPU_BINARY_ELEMENTWISE_H
#define FUSION_CORE_OPS_EXECUTION_CPU_BINARY_ELEMENTWISE_H

#include "Fusion/core/iter/DenseIter.hpp"
#include "Fusion/core/planning/OpContext.h"
#include "Fusion/cpu/simd/SimdTraits.hpp"
#include "Fusion/core/opschema/OpTags.h"

namespace fusion::execution::cpu {

namespace detail {

template <class OpTag>
struct BinaryKernelFor;

template <>
struct BinaryKernelFor<AddTag> {
   using type = AddSIMD;
};

template <>
struct BinaryKernelFor<SubTag> {
   using type = SubtractSIMD;
};

template <>
struct BinaryKernelFor<MulTag> {
   using type = MultiplySIMD;
};

template <>
struct BinaryKernelFor<DivTag> {
   using type = DivideSIMD;
};

template <>
struct BinaryKernelFor<PowTag> {
   using type = PowerSIMD;
};

template <>
struct BinaryKernelFor<MaximumTag> {
   using type = MaximumSIMD;
};

template <>
struct BinaryKernelFor<GreaterTag> {
   using type = GreaterThanSIMD;
};

template <>
struct BinaryKernelFor<GreaterEqualTag> {
   using type = GreaterThanEqualSIMD;
};

template <class OpTag>
using binary_kernel_for_t = typename BinaryKernelFor<OpTag>::type;

template <typename T, class Kernel>
void binary_scalar_fallback(T *o, const T *a, const T *b, const int64_t &so,
                            const int64_t &sa, const int64_t &sb,
                            const std::size_t len) {
   Kernel kernel{};
   for (int64_t i = 0; i < len; ++i)
      o[i * so] = kernel(a[i * sa], b[i * sb]);
}

} // namespace detail

template <typename T, class OpTag>
void binary_elementwise(T* out, const T* lhs, const T* rhs,
                        const planning::BinaryEwiseContext &ctx) {
   using Kernel = detail::binary_kernel_for_t<OpTag>;
   std::array<uint8_t *, 3> base = { // TODO: change ownership model from array here
       reinterpret_cast<uint8_t *>(const_cast<T *>(out)),
       reinterpret_cast<uint8_t *>(const_cast<T *>(lhs)),
       reinterpret_cast<uint8_t *>(const_cast<T *>(rhs))};

   if (ctx.exec == planning::BinaryExecKind::FlatContiguous) {
      auto *o = reinterpret_cast<T *>(base[0]);
      const auto *a = reinterpret_cast<const T *>(base[1]);
      const auto *b = reinterpret_cast<const T *>(base[2]);
      const size_t len = ctx.fast_len;
      if constexpr (simd_traits<Kernel, T>::available) {
         simd_traits<Kernel, T>::execute_contiguous(a, b, o, len, false, false);
      } else {
         detail::binary_scalar_fallback<T, Kernel>(o, a, b, 1, 1, 1, len);
      }
      return;
   }
   const dense::iter::DenseIterPlanView view =
       dense::iter::dense_iter_view(ctx.plan);
   dense::iter::for_each_outer_then_inner<3>(
       view, base, [&](dense::iter::DenseSegment<3> &segment) {
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

          if constexpr (simd_traits<Kernel, T>::available) {
             if (out_contig && a_unit && b_unit && segment.len > 0) {
                const bool a_scalar = a_bytes == 0;
                const bool b_scalar = b_bytes == 0;
                simd_traits<Kernel, T>::execute_contiguous(
                    a, b, o, static_cast<size_t>(segment.len), a_scalar,
                    b_scalar);
                return;
             }
             if (out_contig && (a_unit || b_unit)) {
                const std::int64_t so = 1;
                const std::int64_t sa = a_bytes / step;
                const std::int64_t sb = b_bytes / step;
                detail::binary_scalar_fallback<T, Kernel>(o, a, b, so, sa, sb,
                                                       segment.len);
                return;
             }
          }

          const int64_t so = out_bytes / step;
          const int64_t sa = a_bytes == 0 ? 0 : a_bytes / step;
          const int64_t sb = b_bytes == 0 ? 0 : b_bytes / step;
          detail::binary_scalar_fallback<T, Kernel>(o, a, b, so, sa, sb,
                                                 segment.len);
       });
}
} // namespace fusion::execution::cpu

#endif // FUSION_CORE_OPS_EXECUTION_CPU_BINARY_ELEMENTWISE_H