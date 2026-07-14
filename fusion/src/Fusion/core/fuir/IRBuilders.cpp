#include "Fusion/core/fuir/IRBuilders.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Fusion/common/error/Check.h"
#include "Fusion/core/fuir/FuirErrors.h"
#include "Fusion/core/fuir/FuirValidation.h"
#include "Fusion/core/fuir/ShapeRules.h"

namespace fusion::fuir {

namespace ferr = fusion::error;
using ferr::ErrorCategory;

namespace {

std::uint32_t
bind_idx_to_ir_by_label(std::unordered_map<Label, std::uint32_t>& label_to_id,
                        IndexSpaceIR& ir,
                        const Label label) {
   const auto it = label_to_id.find(label);
   if (it != label_to_id.end()) {
      return it->second;
   }

   const std::uint32_t id = static_cast<std::uint32_t>(ir.indices.size());

   IndexDef idx;
   idx.extent = 1;
   idx.label = label;
   idx.kind = IndexKind::Reduction;
   idx.axis_of_operand.assign(ir.num_operands, -1);

   ir.indices.push_back(std::move(idx));
   label_to_id.emplace(label, id);

   return id;
}

void set_out_labels_from_binding(
    const std::unordered_map<Label, std::uint32_t>& label_to_id,
    const OperandLabelBinding& bind,
    IndexSpaceIR& ir,
    const std::string_view where) {
   for (const Label label : bind.out_labels) {
      const auto it = label_to_id.find(label);

      FUSION_INTERNAL_ASSERT_CODE(
          it != label_to_id.end(),
          fuir_error(FuirError::InvalidIR, ErrorCategory::Internal),
          ferr::message(
              where,
              ": fuir.binding.internal_missing_output_label: label ",
              label,
              " passed validation but was not present in label map"));

      const std::uint32_t id = it->second;
      ir.indices[id].kind = IndexKind::Independent;
      ir.out_indices.push_back(id);
   }
}

} // namespace

IndexSpaceIR build_elementwise_ir_right_aligned(
    const std::vector<OperandDescription>& descs,
    const OperandGroupConstraint constraint) {
   constexpr std::string_view where = "build_elementwise_ir_right_aligned";

   validation::validate_descs_itemsize_group(descs, constraint, where);

   IndexSpaceIR ir;
   ir.num_operands = descs.size();
   ir.itemsize = descs.front().itemsize;

   std::size_t max_nd = 0;
   for (const OperandDescription& desc : descs) {
      max_nd = std::max(max_nd, desc.ndims());
   }

   ir.indices.resize(max_nd);
   ir.out_indices.resize(max_nd);

   for (std::size_t od = 0; od < max_nd; ++od) {
      IndexDef idx;
      idx.label = static_cast<Label>(od);
      idx.kind = IndexKind::Independent;
      idx.extent = 1;
      idx.axis_of_operand.assign(ir.num_operands, -1);

      for (std::size_t op = 0; op < ir.num_operands; ++op) {
         const OperandDescription& desc = descs[op];
         const std::size_t pad = max_nd - desc.ndims();

         if (od < pad) {
            idx.axis_of_operand[op] = -1;
            continue;
         }

         const std::size_t in_ax = od - pad;
         idx.axis_of_operand[op] = static_cast<std::int32_t>(in_ax);
         idx.extent = broadcast_dim(idx.extent, desc.shape[in_ax]);
      }

      ir.indices[od] = std::move(idx);
      ir.out_indices[od] = static_cast<std::uint32_t>(od);
   }

   validation::validate_index_space_ir(ir, where);
   return ir;
}

IndexSpaceIR build_reduction_ir(
    const std::vector<OperandDescription>& descs,
    const std::size_t axis,
    const bool keepdim,
    const OperandGroupConstraint constraint) {
   constexpr std::string_view where = "build_reduction_ir";

   validation::validate_descs_itemsize_group(descs, constraint, where);
   validation::validate_reduction_request(descs, axis, keepdim, where);

   const OperandDescription& in_desc = descs.back();
   const std::size_t in_nd = in_desc.ndims();

   IndexSpaceIR ir;
   ir.num_operands = descs.size();
   ir.itemsize = descs.front().itemsize;

   ir.indices.resize(in_nd);
   ir.out_indices.clear();

   // TODO: keepdim reductions currently exclude the reduced axis from
   // out_indices. If IndexSpaceIR::out_indices becomes the sole source of output
   // shape truth, this needs to represent kept size-1 axes explicitly.
   ir.out_indices.reserve(in_nd > 0 ? in_nd - 1 : 0);

   auto out_axis_for_in_axis = [&](const std::size_t in_ax) -> std::int32_t {
      if (keepdim) {
         return static_cast<std::int32_t>(in_ax);
      }

      if (in_ax == axis) {
         return -1;
      }

      if (in_ax < axis) {
         return static_cast<std::int32_t>(in_ax);
      }

      return static_cast<std::int32_t>(in_ax - 1);
   };

   for (std::size_t in_ax = 0; in_ax < in_nd; ++in_ax) {
      IndexDef idx;
      idx.label = static_cast<Label>(in_ax);
      idx.extent = in_desc.shape[in_ax];
      idx.kind =
          in_ax == axis ? IndexKind::Reduction : IndexKind::Independent;
      idx.axis_of_operand.assign(ir.num_operands, -1);

      idx.axis_of_operand[0] = out_axis_for_in_axis(in_ax);

      for (std::size_t op = 1; op < ir.num_operands; ++op) {
         idx.axis_of_operand[op] = static_cast<std::int32_t>(in_ax);
      }

      ir.indices[in_ax] = std::move(idx);

      if (in_ax != axis) {
         ir.out_indices.push_back(static_cast<std::uint32_t>(in_ax));
      }
   }

   validation::validate_index_space_ir(ir, where);
   return ir;
}

IndexSpaceIR build_ir_from_label_binding(
    const std::vector<OperandDescription>& descs,
    const OperandLabelBinding& bind,
    const OperandGroupConstraint constraint) {
   constexpr std::string_view where = "build_ir_from_label_binding";

   validation::validate_descs_itemsize_group(descs, constraint, where);
   validation::validate_operand_label_binding(descs, bind, where);

   IndexSpaceIR ir;
   ir.num_operands = descs.size();
   ir.itemsize = descs.front().itemsize;

   std::unordered_map<Label, std::uint32_t> label_to_id;
   label_to_id.reserve(64);

   for (std::size_t op = 0; op < descs.size(); ++op) {
      const OperandDescription& desc = descs[op];
      const std::vector<Label>& labels = bind.op_axis_labels[op];

      for (std::size_t ax = 0; ax < labels.size(); ++ax) {
         const Label label = labels[ax];

         const std::uint32_t id =
             bind_idx_to_ir_by_label(label_to_id, ir, label);

         IndexDef& idx = ir.indices[id];
         idx.axis_of_operand[op] = static_cast<std::int32_t>(ax);
         idx.extent = broadcast_dim(idx.extent, desc.shape[ax]);
      }
   }

   ir.out_indices.clear();
   ir.out_indices.reserve(bind.out_labels.size());

   set_out_labels_from_binding(label_to_id, bind, ir, where);

   validation::validate_index_space_ir(ir, where);
   return ir;
}

} // namespace fusion::fuir