#ifndef FUSION_CORE_ITER_DENSE_ITER_HPP
#define FUSION_CORE_ITER_DENSE_ITER_HPP

#include <array>
#include <cstddef>
#include <cstdint>

#include "Fusion/common/Checks.hpp"
#include "Fusion/execution/iter/DenseIterPlanView.hpp"
#include "Fusion/execution/iter/DenseSegmentView.h"

namespace fusion::dense::iter {

namespace detail {

template <std::size_t NumInputs, std::size_t NumOutputs>
void validate_dense_iter_view(const DenseIterPlanView &view) {
   constexpr std::size_t expected_operands = NumOutputs + NumInputs;

   FUSION_CHECK(view.operands.size() == expected_operands,
                "Dense iterator operand count mismatch");

   for (const auto &operand : view.operands) {
      FUSION_CHECK(operand.access == fuir::AccessKind::Affine,
                   "Dense iterator currently supports affine access only");

      FUSION_CHECK(operand.affine.byte_stride_per_loop.size() ==
                       view.loop.size(),
                   "Dense iterator stride count does not match loop rank");
   }
}

[[nodiscard]] inline ByteStride operand_stride(const DenseIterPlanView &view,
                                               std::size_t operand_index,
                                               int dim) {
   return to_byte_stride(
       view.operands[operand_index].affine.byte_stride_per_loop[dim]);
}

template <std::size_t NumInputs, std::size_t NumOutputs>
DenseSegmentView<NumInputs, NumOutputs> construct_inner_segment(
    int inner_dim, const DenseIterPlanView &view,
    const std::array<std::byte *, NumOutputs> &outputs,
    const std::array<const std::byte *, NumInputs> &inputs) {
   FUSION_CHECK(inner_dim >= 0, "Inner dimension must be non-negative");

   FUSION_CHECK(inner_dim < static_cast<int>(view.loop.size()),
                "Inner dimension is out of range");

   DenseSegmentView<NumInputs, NumOutputs> segment{};

   segment.outputs = outputs;
   segment.inputs = inputs;
   segment.len = static_cast<std::int64_t>(view.loop[inner_dim].size);

   for (std::size_t output = 0; output < NumOutputs; ++output) {
      segment.output_byte_stride[output] =
          operand_stride(view, output, inner_dim);
   }

   for (std::size_t input = 0; input < NumInputs; ++input) {
      const std::size_t operand_index = NumOutputs + input;

      segment.input_byte_stride[input] =
          operand_stride(view, operand_index, inner_dim);
   }

   return segment;
}

template <std::size_t NumInputs, std::size_t NumOutputs>
DenseSegmentView<NumInputs, NumOutputs> construct_scalar_segment(
    const std::array<std::byte *, NumOutputs> &outputs,
    const std::array<const std::byte *, NumInputs> &inputs) {
   DenseSegmentView<NumInputs, NumOutputs> segment{};

   segment.outputs = outputs;
   segment.inputs = inputs;
   segment.len = 1;

   // Scalar segments use each pointer once, so all strides remain zero.
   return segment;
}

template <std::size_t NumInputs, std::size_t NumOutputs>
void offset_pointers(const DenseIterPlanView &view, int dim, std::int64_t count,
                     std::array<std::byte *, NumOutputs> &outputs,
                     std::array<const std::byte *, NumInputs> &inputs) {
   for (std::size_t output = 0; output < NumOutputs; ++output) {
      const std::int64_t byte_offset =
          view.operands[output].affine.byte_stride_per_loop[dim] * count;

      outputs[output] += byte_offset;
   }

   for (std::size_t input = 0; input < NumInputs; ++input) {
      const std::size_t operand_index = NumOutputs + input;

      const std::int64_t byte_offset =
          view.operands[operand_index].affine.byte_stride_per_loop[dim] * count;

      inputs[input] += byte_offset;
   }
}

template <std::size_t NumInputs, std::size_t NumOutputs, typename InnerFn>
void walk(int dim, int inner_dim, const DenseIterPlanView &view,
          std::array<std::byte *, NumOutputs> &outputs,
          std::array<const std::byte *, NumInputs> &inputs, InnerFn &inner) {
   if (dim == inner_dim) {
      const auto segment = construct_inner_segment<NumInputs, NumOutputs>(
          inner_dim, view, outputs, inputs);

      inner(segment);
      return;
   }

   const fuir::LoopDim &loop_dim = view.loop[dim];
   const std::int64_t extent = static_cast<std::int64_t>(loop_dim.size);

   for (std::int64_t i = 0; i < extent; ++i) {
      walk<NumInputs, NumOutputs>(dim + 1, inner_dim, view, outputs, inputs,
                                  inner);

      offset_pointers<NumInputs, NumOutputs>(view, dim, 1, outputs, inputs);
   }

   // Restore the pointers to their values on entry to this dimension.
   offset_pointers<NumInputs, NumOutputs>(view, dim, -extent, outputs, inputs);
}

} // namespace detail

template <std::size_t NumInputs, std::size_t NumOutputs, typename FnInnermost>
void for_each_outer_then_inner(const DenseIterPlanView &view,
                               std::array<std::byte *, NumOutputs> outputs,
                               std::array<const std::byte *, NumInputs> inputs,
                               FnInnermost &&inner) {
   detail::validate_dense_iter_view<NumInputs, NumOutputs>(view);

   const int ndim = static_cast<int>(view.loop.size());

   if (ndim == 0) {
      const auto segment =
          detail::construct_scalar_segment<NumInputs, NumOutputs>(outputs,
                                                                  inputs);

      inner(segment);
      return;
   }

   detail::walk<NumInputs, NumOutputs>(0, ndim - 1, view, outputs, inputs,
                                       inner);
}

} // namespace fusion::dense::iter

#endif // FUSION_CORE_ITER_DENSE_ITER_HPP