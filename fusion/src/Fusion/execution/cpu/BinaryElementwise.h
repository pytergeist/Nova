#ifndef FUSION_EXECUTION_CPU_BINARY_ELEMENTWISE_H
#define FUSION_EXECUTION_CPU_BINARY_ELEMENTWISE_H

#include <array>
#include <cstddef>
#include <cstdint>

#include "Fusion/compiler/planning/OpContext.h"
#include "Fusion/execution/iter/DenseIter.hpp"
#include "Fusion/execution/iter/DenseSegmentView.h"
#include "Fusion/kernels/simd/SimdTraits.hpp"
#include "Fusion/opschema/OpTags.h"

namespace fusion::execution::cpu {

namespace detail {
template <class OpTag> struct BinaryKernelFor;

template <> struct BinaryKernelFor<AddTag> {
   using type = AddSIMD;
};

template <> struct BinaryKernelFor<SubTag> {
   using type = SubtractSIMD;
};

template <> struct BinaryKernelFor<MulTag> {
   using type = MultiplySIMD;
};

template <> struct BinaryKernelFor<DivTag> {
   using type = DivideSIMD;
};

template <> struct BinaryKernelFor<PowTag> {
   using type = PowerSIMD;
};

template <> struct BinaryKernelFor<MaximumTag> {
   using type = MaximumSIMD;
};

template <> struct BinaryKernelFor<GreaterTag> {
   using type = GreaterThanSIMD;
};

template <> struct BinaryKernelFor<GreaterEqualTag> {
   using type = GreaterThanEqualSIMD;
};

template <class OpTag>
using binary_kernel_for_t = typename BinaryKernelFor<OpTag>::type;

template <typename T, class Kernel>
void binary_scalar_fallback(T *out, const T *lhs, const T *rhs,
                            const int64_t &out_step, const int64_t &lhs_step,
                            const int64_t &rhs_step, const std::size_t len) {
   Kernel kernel{};
   for (std::int64_t i = 0; i < len; ++i) {
      out[i * out_step] = kernel(lhs[i * lhs_step], rhs[i * rhs_step]);
   }
}

template <typename T, class Kernel>
void execute_binary_segment(
    const dense::iter::DenseSegmentView<2, 1> &segment) {
   if (segment.empty()) {
      return;
   }

   T *out = segment.template output<T>(0);

   const T *lhs = segment.template input<T>(0);

   const T *rhs = segment.template input<T>(1);

   const dense::iter::ByteStride out_stride = segment.output_stride(0);

   const dense::iter::ByteStride lhs_stride = segment.input_stride(0);

   const dense::iter::ByteStride rhs_stride = segment.input_stride(1);

   if constexpr (simd_traits<Kernel, T>::available) {
      const bool can_vectorise = out_stride.is_contiguous<T>() &&
                                 lhs_stride.is_contiguous_or_broadcast<T>() &&
                                 rhs_stride.is_contiguous_or_broadcast<T>();

      if (can_vectorise) {
         simd_traits<Kernel, T>::execute_contiguous(
             lhs, rhs, out, static_cast<std::size_t>(segment.len),
             lhs_stride.is_broadcast(), rhs_stride.is_broadcast());

         return;
      }
   }

   const std::int64_t out_step = out_stride.element_stride<T>();

   const std::int64_t lhs_step = lhs_stride.element_stride<T>();

   const std::int64_t rhs_step = rhs_stride.element_stride<T>();

   Kernel kernel{};
   detail::binary_scalar_fallback<T, Kernel>(out, lhs, rhs, out_step, lhs_step,
                                             rhs_step, segment.len);
}

template <typename T>
[[nodiscard]] dense::iter::DenseSegmentView<2, 1>
make_contiguous_binary_segment(T *out, const T *lhs, const T *rhs,
                               std::size_t len) {
   const dense::iter::ByteStride contiguous =
       dense::iter::to_byte_stride(static_cast<std::int64_t>(sizeof(T)));

   dense::iter::DenseSegmentView<2, 1> segment{};

   segment.outputs[0] = reinterpret_cast<std::byte *>(out);

   segment.inputs[0] = reinterpret_cast<const std::byte *>(lhs);

   segment.inputs[1] = reinterpret_cast<const std::byte *>(rhs);

   segment.output_byte_stride[0] = contiguous;
   segment.input_byte_stride[0] = contiguous;
   segment.input_byte_stride[1] = contiguous;

   segment.len = static_cast<std::int64_t>(len);

   return segment;
}

} // namespace detail

template <typename T, class OpTag>
void binary_elementwise(T *out, const T *lhs, const T *rhs,
                        const planning::BinaryEwiseContext &ctx) {
   using Kernel = detail::binary_kernel_for_t<OpTag>;

   using Segment = dense::iter::DenseSegmentView<2, 1>;

   const auto execute_segment = [](const Segment &segment) {
      detail::execute_binary_segment<T, Kernel>(segment);
   };

   if (ctx.exec == planning::BinaryExecKind::FlatContiguous) {
      execute_segment(
          detail::make_contiguous_binary_segment(out, lhs, rhs, ctx.fast_len));

      return;
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

   dense::iter::for_each_outer_then_inner<2, 1>(view, outputs, inputs,
                                                execute_segment);
}

} // namespace fusion::execution::cpu

#endif // FUSION_EXECUTION_CPU_BINARY_ELEMENTWISE_H