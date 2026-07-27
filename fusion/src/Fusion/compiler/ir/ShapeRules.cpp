#include "Fusion/compiler/ir/ShapeRules.h"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "Fusion/common/error/Check.h"
#include "Fusion/compiler/ir/Builders.h"
#include "Fusion/compiler/ir/IRErrors.h"
#include "Fusion/compiler/ir/IRValidation.h"
#include "Fusion/compiler/ir/OperandConstraints.h"

namespace fusion::fuir {

namespace ferr = fusion::error;
using ferr::ErrorCategory;

std::size_t norm_axis(const std::int64_t axis, const std::size_t ndims) {
   constexpr std::string_view where = "norm_axis";

   const std::int64_t normalized =
       axis < 0 ? axis + static_cast<std::int64_t>(ndims) : axis;

   FUSION_CHECK_CODE(
       normalized >= 0 && normalized < static_cast<std::int64_t>(ndims),
       fuir_error(FuirError::InvalidAxis, ErrorCategory::InvalidArgument),
       ferr::message(where, ": fuir.shape.invalid_axis: axis ", axis,
                     " out of range for rank ", ndims));

   return static_cast<std::size_t>(normalized);
}

std::size_t broadcast_dim(const std::size_t lhs, const std::size_t rhs) {
   constexpr std::string_view where = "broadcast_dim";

   FUSION_CHECK_CODE(
       lhs == rhs || lhs == 1 || rhs == 1,
       fuir_error(FuirError::BroadcastMismatch, ErrorCategory::InvalidArgument),
       ferr::message(where, ": fuir.shape.broadcast_mismatch: dimensions ", lhs,
                     " and ", rhs, " are not broadcast-compatible"));

   if (lhs == rhs) {
      return lhs;
   }

   if (lhs == 1) {
      return rhs;
   }

   return lhs;
}

std::vector<std::size_t> out_shape_from_ir(const IndexSpaceIR &ir) {
   constexpr std::string_view where = "out_shape_from_ir";

   validation::validate_index_space_ir(ir, where);

   std::vector<std::size_t> out_shape;
   out_shape.reserve(ir.out_indices.size());

   for (const std::uint32_t id : ir.out_indices) {
      out_shape.push_back(ir.indices[id].extent);
   }

   return out_shape;
}

std::vector<std::size_t> infer_binary_contraction_out_shape_from_binding(
    const std::vector<OperandDescription> &inputs,
    const OperandLabelBinding &binding) {
   constexpr std::string_view where =
       "infer_binary_contraction_out_shape_from_binding";

   FUSION_CHECK_CODE(
       inputs.size() == 2,
       fuir_error(FuirError::DescriptorCountMismatch,
                  ErrorCategory::InvalidArgument),
       ferr::message(where,
                     ": fuir.shape.invalid_input_count: expected inputs = "
                     "{A, B}, got ",
                     inputs.size()));

   constexpr OperandGroupConstraint constraint =
       OperandGroupConstraint::HomogeneousItemSize;

   validation::validate_descs_itemsize_group(inputs, constraint, where);

   OperandDescription dummy_out;
   dummy_out.shape.assign(binding.out_labels.size(), 1);
   dummy_out.strides.assign(dummy_out.ndims(), 0);
   dummy_out.itemsize = inputs.front().itemsize;

   std::vector<OperandDescription> descs = {
       dummy_out,
       inputs.front(),
       inputs.back(),
   };

   const IndexSpaceIR ir =
       build_ir_from_label_binding(descs, binding, constraint);

   return out_shape_from_ir(ir);
}

} // namespace fusion::fuir