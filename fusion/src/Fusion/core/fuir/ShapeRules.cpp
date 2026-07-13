
#include "Fusion/core/fuir/ShapeRules.h"

#include <stdexcept>

#include "Fusion/core/fuir/OperandConstraints.h"
#include "Fusion/core/fuir/IRBuilders.h"
#include "Fusion/core/fuir/FuirValidation.h"


namespace fusion::fuir {

std::size_t norm_axis(std::int64_t ax, std::size_t ndims) {
   const std::int64_t r =
       (ax < 0) ? (ax + static_cast<std::int64_t>(ndims)) : ax;
   if (r < 0 || r >= static_cast<std::int64_t>(ndims))
      throw std::runtime_error("axis out of range");
   return static_cast<std::size_t>(r);
}

std::size_t broadcast_dim(std::size_t a, std::size_t b) {
   if (a == b)
      return a;
   if (a == 1)
      return b;
   if (b == 1)
      return a;
   throw std::runtime_error("broadcast: dimension mismatch");
}

std::vector<std::size_t> out_shape_from_ir(const IndexSpaceIR &ir) {
   std::vector<std::size_t> out_shape;
   out_shape.reserve(ir.out_indices.size());
   for (std::uint32_t id : ir.out_indices) {
      out_shape.push_back(ir.indices[id].extent);
   }
   return out_shape;
}

std::vector<std::size_t>
infer_out_shape_from_binding(const std::vector<OperandDescription> &inputs,
                             const fuir::OperandLabelBinding &binding) {
   if (inputs.size() != 2) {
      throw std::runtime_error("einsum: expected inputs = {A, B}");
   }
   // TODO: The below is hardcoded to homoItemSize
   OperandGroupConstraint constraint =
       OperandGroupConstraint::HomogeneousItemSize;
   validate_descs_itemsize_group(inputs, constraint);

   OperandDescription dummy_out;
   dummy_out.shape.assign(binding.out_labels.size(), 1);
   dummy_out.strides.assign(dummy_out.ndims(), 0);
   dummy_out.itemsize = inputs[0].itemsize;

   std::vector<OperandDescription> tmp = {dummy_out, inputs[0], inputs[1]};
   IndexSpaceIR ir = build_ir_from_label_binding(tmp, binding, constraint);
   return out_shape_from_ir(ir);
}
} // namespace fusion::fuir
