#ifndef FUSION_EXECUTION_CPU_UNARY_ELEMENTWISE_H
#define FUSION_EXECUTION_CPU_UNARY_ELEMENTWISE_H

#include <array>
#include <cstddef>
#include <cstdint>

#include "Fusion/compiler/planning/OpContext.h"
#include "Fusion/execution/iter/DenseIter.hpp"
#include "Fusion/kernels/simd/SimdTraits.hpp"
#include "Fusion/opschema/OpTags.h"

namespace fusion::execution::cpu {

namespace detail {

template <class OpTag> struct UnaryKernelFor;

template <> struct UnaryKernelFor<SqrtTag> {
   using type = SqrtSIMD;
};

template <> struct UnaryKernelFor<LogTag> {
   using type = NaturalLogSIMD;
};

template <> struct UnaryKernelFor<ExpTag> {
   using type = ExponentialSIMD;
};

template <> struct UnaryKernelFor<ReciprocalTag> {
   using type = ReciprocalSIMD;
};

template <class OpTag> using unary_kernel_for_t = UnaryKernelFor<OpTag>::type;

template <typename T, class Kernel>
void unary_scalar_fallback(T *out, const T *operand, const int64_t &out_step,
                           const int64_t &operand_step, const std::size_t len) {
   Kernel kernel{};
   for (int64_t i = 0; i < len; ++i)
      out[i * out_step] = kernel(operand[i * operand_step]);
}

template <typename T, class Kernel>
void execute_unary_segment(const dense::iter::DenseSegmentView<1, 1> &segment) {
   if (segment.empty()) {
      return;
   }

   T *out = segment.output<T>(0);
   const T *operand = segment.input<T>(0);

   const dense::iter::ByteStride out_stride = segment.output_stride(0);

   const dense::iter::ByteStride operand_stride = segment.input_stride(0);

   if constexpr (simd_traits<Kernel, T>::available) {
      const bool can_vectorise = out_stride.is_contiguous<T>() &&
                                 operand_stride.is_contiguous_or_broadcast<T>();

      if (can_vectorise) {
         simd_traits<Kernel, T>::execute_contiguous(
             operand, out, static_cast<std::size_t>(segment.len),
             operand_stride.is_broadcast());

         return;
      }
   }

   const std::int64_t out_step = out_stride.element_stride<T>();

   const std::int64_t operand_step = operand_stride.element_stride<T>();

   detail::unary_scalar_fallback<T, Kernel>(out, operand, out_step,
                                            operand_step, segment.len);
}

template <typename T>
[[nodiscard]] dense::iter::DenseSegmentView<1, 1>
make_contiguous_unary_segment(T *out, const T *operand, std::size_t len) {
   const auto contiguous =
       dense::iter::to_byte_stride(static_cast<std::int64_t>(sizeof(T)));

   dense::iter::DenseSegmentView<1, 1> segment{};

   segment.outputs[0] = reinterpret_cast<std::byte *>(out);

   segment.inputs[0] = reinterpret_cast<const std::byte *>(operand);

   segment.output_byte_stride[0] = contiguous;
   segment.input_byte_stride[0] = contiguous;

   segment.len = static_cast<std::int64_t>(len);

   return segment;
}

} // namespace detail

template <typename T, class OpTag>
void unary_elementwise(T *out, const T *operand,
                       const planning::UnaryEwiseContext &ctx) {
   using Kernel = detail::unary_kernel_for_t<OpTag>;

   using Segment = dense::iter::DenseSegmentView<1, 1>;

   const auto execute_segment = [](const Segment &segment) {
      detail::execute_unary_segment<T, Kernel>(segment);
   };

   if (ctx.fastpath) {
      execute_segment(
          detail::make_contiguous_unary_segment(out, operand, ctx.fast_len));

      return;
   }

   const dense::iter::DenseIterPlanView view =
       dense::iter::dense_iter_view(ctx.plan);

   std::array<std::byte *, 1> outputs{
       reinterpret_cast<std::byte *>(out),
   };

   std::array<const std::byte *, 1> inputs{
       reinterpret_cast<const std::byte *>(operand),
   };

   dense::iter::for_each_outer_then_inner<1, 1>(view, outputs, inputs,
                                                execute_segment);
}

} // namespace fusion::execution::cpu

#endif // FUSION_EXECUTION_CPU_UNARY_ELEMENTWISE_H