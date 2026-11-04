#ifndef FUSION_EXECUTION_CPU_REDUCTION_H
#define FUSION_EXECUTION_CPU_REDUCTION_H

#include <array>
#include <cstddef>
#include <cstdint>

#include "Fusion/compiler/planning/OpContext.h"
#include "Fusion/execution/iter/DenseIter.hpp"
#include "Fusion/kernels/simd/SimdTraits.hpp"
#include "Fusion/opschema/OpTags.h"

namespace fusion::execution::cpu {

namespace detail {

template <class OpTag> struct ReductionKernelFor;

template <> struct ReductionKernelFor<SumTag> {
   using type = SumSIMD;

   template <typename T>
   [[nodiscard]] static constexpr T
   finalise(T accumulated, std::size_t /*reduce_len*/) noexcept {
      return accumulated;
   }
};

template <> struct ReductionKernelFor<MeanTag> {
   using type = SumSIMD;

   template <typename T>
   [[nodiscard]] static constexpr T finalise(T accumulated,
                                             std::size_t reduce_len) noexcept {
      return accumulated / static_cast<T>(reduce_len);
   }
};

template <class OpTag>
using reduction_kernel_for_t = typename ReductionKernelFor<OpTag>::type;

template <typename T, class Kernel>
void reduction_scalar_fallback(T *out, const T *operand,
                               std::int64_t out_stride,
                               std::int64_t operand_stride, std::int64_t len) {
   Kernel kernel{};

   for (std::int64_t i = 0; i < len; ++i) {
      out[i * out_stride] += kernel(operand[i * operand_stride]);
   }
}

template <typename T, class Kernel>
void execute_reduction_segment(
    const dense::iter::DenseSegmentView<1, 1> &segment) {
   if (segment.empty()) {
      return;
   }

   T *out = segment.template output<T>(0);
   const T *operand = segment.template input<T>(0);

   const dense::iter::ByteStride out_stride = segment.output_stride(0);

   const dense::iter::ByteStride operand_stride = segment.input_stride(0);

   if constexpr (simd_traits<Kernel, T>::available) {
      const bool can_vectorise = out_stride.is_broadcast() &&
                                 operand_stride.template is_contiguous<T>();

      if (can_vectorise) {
         *out += simd_traits<Kernel, T>::reduce_contiguous(
             operand, static_cast<std::size_t>(segment.len));

         return;
      }
   }

   reduction_scalar_fallback<T, Kernel>(
       out, operand, out_stride.template element_stride<T>(),
       operand_stride.template element_stride<T>(), segment.len);
}

template <typename T>
[[nodiscard]] dense::iter::DenseSegmentView<1, 1>
make_contiguous_reduction_segment(T *out, const T *operand, std::size_t len) {
   dense::iter::DenseSegmentView<1, 1> segment{};

   segment.outputs[0] = reinterpret_cast<std::byte *>(out);

   segment.inputs[0] = reinterpret_cast<const std::byte *>(operand);

   // The full input is reduced into one output value.
   segment.output_byte_stride[0] = dense::iter::to_byte_stride(0);

   segment.input_byte_stride[0] =
       dense::iter::to_byte_stride(static_cast<std::int64_t>(sizeof(T)));

   segment.len = static_cast<std::int64_t>(len);

   return segment;
}

} // namespace detail

template <typename T, class OpTag>
void reduction(T *out, const T *operand, std::size_t out_size,
               const planning::ReductionContext &ctx) {
   using Reduction = detail::ReductionKernelFor<OpTag>;
   using Kernel = typename Reduction::type;
   using Segment = dense::iter::DenseSegmentView<1, 1>;

   const auto execute_segment = [](const Segment &segment) {
      detail::execute_reduction_segment<T, Kernel>(segment);
   };

   if (ctx.fastpath) {
      execute_segment(detail::make_contiguous_reduction_segment(out, operand,
                                                                ctx.fast_len));

      *out = Reduction::template finalise<T>(*out, ctx.reduce_len);

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

   for (std::size_t i = 0; i < out_size; ++i) {
      out[i] = Reduction::template finalise<T>(out[i], ctx.reduce_len);
   }
}

} // namespace fusion::execution::cpu

#endif // FUSION_EXECUTION_CPU_REDUCTION_H