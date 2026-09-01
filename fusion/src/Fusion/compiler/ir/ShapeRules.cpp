#include "Fusion/compiler/ir/ShapeRules.h"

#include <cstddef>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "Fusion/common/error/Check.h"
#include "Fusion/compiler/ir/Builders.h"
#include "Fusion/compiler/ir/IRErrors.h"
#include "Fusion/compiler/ir/IRValidation.h"
#include "Fusion/compiler/ir/OperandConstraints.h"

namespace fusion::fuir {

namespace ferr = fusion::error;

using ferr::ErrorCategory;

using LabelExtentMap = std::unordered_map<Label, std::size_t>;

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
   // TODO: this is not an inference - therefore this does not belong in
   // shaperules
   constexpr std::string_view where = "out_shape_from_ir";

   validation::validate_index_space_ir(ir, where);

   const std::vector<PhysicalAxis> &out_axes = ir.physical_axes.front();

   std::vector<std::size_t> out_shape;
   out_shape.reserve(out_axes.size());

   for (const PhysicalAxis &axis : out_axes) {
      out_shape.push_back(axis.extent);
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

std::vector<std::size_t>
infer_out_shape_from_binding(const std::vector<OperandDescription> &descs,
                             const OperandLabelBinding &binding) {
   constexpr std::string_view where = "infer_out_shape_from_binding";

   FUSION_CHECK_CODE(
       descs.size() == 2,
       fuir_error(FuirError::DescriptorCountMismatch,
                  ErrorCategory::InvalidArgument),
       ferr::message(where,
                     ": fuir.shape.invalid_input_count: expected inputs = "
                     "{A, B}, got ",
                     descs.size()));

   constexpr OperandGroupConstraint constraint =
       OperandGroupConstraint::HomogeneousItemSize;

   validation::validate_descs_itemsize_group(descs, constraint, where);

   std::vector<std::size_t> out_shape;
   out_shape.reserve(binding.out_labels.size());
   for (const Label &label : binding.out_labels) {
      std::size_t logical_extent = 1;
      for (std::size_t op = 0; op < descs.size(); ++op) {
         const OperandDescription &desc = descs[op];
         // NB: This fn is currently only called with {lhs, rhs} (e.g. no output
         // operand), therefore we offset the index for the op_axis_labels by 1.
         const std::vector<Label> &labels = binding.op_axis_labels.at(op + 1);

         for (std::size_t ax = 0; ax < labels.size(); ++ax) {
            if (labels[ax] != label) {
               continue;
            }
            logical_extent = broadcast_dim(logical_extent, desc.shape.at(ax));
         }
      }
      out_shape.push_back(logical_extent);
   }
   return out_shape;
}

LabelExtentMap resolve_binary_contraction_label_extents(
    const std::vector<OperandDescription> &inputs,
    const OperandLabelBinding &binding) {
   std::unordered_set<Label> output_labels(binding.out_labels.begin(),
                                           binding.out_labels.end());

   LabelExtentMap extents_by_label;
   extents_by_label.reserve(inputs[0].ndims() + inputs[1].ndims());

   for (std::size_t input_id = 0; input_id < inputs.size(); ++input_id) {
      const OperandDescription &input = inputs[input_id];
      const std::vector<Label> &labels = binding.op_axis_labels[input_id + 1];

      for (std::size_t axis_id = 0; axis_id < labels.size(); ++axis_id) {
         const Label label = labels[axis_id];
         const std::size_t physical_extent = input.shape[axis_id];

         const auto [it, inserted] =
             extents_by_label.emplace(label, physical_extent);

         if (inserted) {
            continue;
         }

         if (output_labels.contains(label)) {
            it->second = std::max(it->second, physical_extent);
         }
      }
   }

   return extents_by_label;
}

} // namespace fusion::fuir