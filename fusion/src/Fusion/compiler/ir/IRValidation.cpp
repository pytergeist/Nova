#include "Fusion/compiler/ir/IRValidation.h"

#include <cstddef>
#include <unordered_set>

#include "Fusion/common/error/Check.h"
#include "Fusion/compiler/ir/IRErrors.h"

// TODO: Need to move validation of the lowering pipeline out of IR validation
// and into lowering

namespace fusion::fuir::validation {

namespace ferr = fusion::error;
using ferr::ErrorCategory;

using PhysicalAxesByOperand = std::vector<std::vector<PhysicalAxis>>;

namespace detail {
void validate_physical_axis_operand_ids(
    const PhysicalAxesByOperand &physical_axes, std::string_view where) {
   for (std::size_t op = 0; op < physical_axes.size(); ++op) {
      const std::vector<PhysicalAxis> &axes_by_operand = physical_axes[op];
      for (const PhysicalAxis &physical_ax : axes_by_operand) {
         FUSION_CHECK_CODE(
             physical_ax.operand_id == static_cast<OperandId>(op),
             fuir_error(FuirError::PhysicalAxisOperandMismatch,
                        ErrorCategory::InvalidArgument),
             ferr::message(
                 where,
                 ": ir.physical_axis.operand_id_mismatch: physical axis ",
                 physical_ax.axis_id, " declares operand_id ",
                 physical_ax.operand_id,
                 ", but is stored in operand collection ", op));
      }
   }
}

void validate_physical_axis_ids(const PhysicalAxesByOperand &physical_axes,
                                std::string_view where) {
   for (std::size_t op = 0; op < physical_axes.size(); ++op) {
      const std::vector<PhysicalAxis> &axes_by_operand = physical_axes[op];
      for (std::size_t axis_id = 0; axis_id < axes_by_operand.size();
           ++axis_id) {
         const PhysicalAxis &physical_ax = axes_by_operand[axis_id];
         FUSION_CHECK_CODE(
             physical_ax.axis_id == static_cast<PhysicalAxisId>(axis_id),
             fuir_error(FuirError::InvalidPhysicalAxisId,
                        ErrorCategory::InvalidArgument),
             ferr::message(where,
                           ": ir.physical_axis.axis_id_mismatch: physical "
                           "axis declares axis_id ",
                           physical_ax.axis_id,
                           ", but is stored at axis position ", axis_id,
                           " for operand ", op));
      }
   }
}

void validate_physical_axis_extents(const PhysicalAxesByOperand &physical_axes,
                                    std::string_view where) {
   for (const std::vector<PhysicalAxis> &axes_by_operand : physical_axes) {
      for (const PhysicalAxis &physical_ax : axes_by_operand) {
         FUSION_CHECK_CODE(
             physical_ax.extent > 0,
             fuir_error(FuirError::InvalidPhysicalExtent,
                        ErrorCategory::InvalidArgument),
             ferr::message(
                 where, ": fuir.physical_axis.invalid_extent: physical axis ",
                 physical_ax.axis_id, " for operand ", physical_ax.operand_id,
                 " has extent ", physical_ax.extent,
                 "; expected an extent greater than zero"));
      }
   }
}

void validate_axis_use_physical_axis_references(
    const std::vector<OperandUse> &operand_uses,
    const PhysicalAxesByOperand &physical_axes, std::string_view where) {
   // TODO: left below check into helper
   for (const OperandUse &operand_use : operand_uses) {
      const std::size_t op = static_cast<std::size_t>(operand_use.operand_id);

      FUSION_CHECK_CODE(
          op < physical_axes.size(),
          fuir_error(FuirError::InvalidOperandUseId,
                     ErrorCategory::InvalidArgument),
          ferr::message(
              where,
              ": fuir.operand_use.invalid_operand_reference: operand_use "
              "references operand_id ",
              operand_use.operand_id, ", but the IR contains ",
              physical_axes.size(), " operand collections"));

      const std::vector<PhysicalAxis> &operand_axes = physical_axes[op];

      for (const AxisUse &axis_use : operand_use.axis_use) {
         FUSION_CHECK_CODE(
             static_cast<std::size_t>(axis_use.physical_axis_id) <
                 operand_axes.size(),
             fuir_error(FuirError::InvalidPhysicalAxisReference,
                        ErrorCategory::InvalidArgument),
             ferr::message(
                 where,
                 ": fuir.axis_use.invalid_physical_axis_reference: operand ",
                 operand_use.operand_id, " references physical_axis_id ",
                 axis_use.physical_axis_id, ", but that operand has ",
                 operand_axes.size(), " physical axes"));
      }
   }
}

void validate_axis_use_logical_axis_references(
    const std::vector<OperandUse> &operand_uses,
    const std::vector<LogicalAxis> &logical_axes, std::string_view where);

void validate_unique_physical_axis_uses(
    const std::vector<OperandUse> &operand_uses, std::string_view where);

void validate_direct_axis_uses(const std::vector<OperandUse> &operand_uses,
                               const PhysicalAxesByOperand &physical_axes,
                               const std::vector<LogicalAxis> &logical_axes,
                               std::string_view where);

void validate_broadcast_axis_uses(const std::vector<OperandUse> &operand_uses,
                                  const PhysicalAxesByOperand &physical_axes,
                                  const std::vector<LogicalAxis> &logical_axes,
                                  std::string_view where);

void validate_no_indexed_axis_uses(const std::vector<OperandUse> &operand_uses,
                                   std::string_view where);

void validate_logical_axis_participation(
    const std::vector<LogicalAxis> &logical_axes,
    const std::vector<OperandUse> &operand_uses, std::string_view where);

void validate_operand_use_operand_ids(
    const std::vector<OperandUse> &operand_uses, std::size_t num_operands,
    std::string_view where);

void validate_elementwise_index_space(
    const std::vector<LogicalAxis> &logical_axes,
    const PhysicalAxesByOperand &physical_axes,
    const std::vector<OperandUse> &operand_uses, std::string_view where);

} // namespace detail

void validate_descs_itemsize_group(const std::vector<OperandDescription> &descs,
                                   const OperandGroupConstraint constraint,
                                   const std::string_view where) {
   FUSION_CHECK_CODE(
       !descs.empty(),
       fuir_error(FuirError::EmptyOperands, ErrorCategory::InvalidArgument),
       ferr::message(where, ": fuir.desc.empty_operands: no operands"));

   const std::size_t itemsize = descs.front().itemsize;

   FUSION_CHECK_CODE(
       itemsize > 0,
       fuir_error(FuirError::ItemSizeMismatch, ErrorCategory::InvalidArgument),
       ferr::message(where,
                     ": fuir.desc.invalid_itemsize: itemsize must be > 0"));

   for (std::size_t op = 0; op < descs.size(); ++op) {
      const OperandDescription &desc = descs[op];

      FUSION_CHECK_CODE(
          desc.shape.size() == desc.strides.size(),
          fuir_error(FuirError::OperandRankMismatch,
                     ErrorCategory::InvalidArgument),
          ferr::message(where, ": fuir.desc.rank_mismatch: operand ", op,
                        " shape rank ", desc.shape.size(),
                        " does not match strides rank ", desc.strides.size()));

      if (constraint == OperandGroupConstraint::TopologyAllowed &&
          desc.type == OperandDescType::Topology) {
         continue;
      }

      FUSION_CHECK_CODE(desc.itemsize == itemsize,
                        fuir_error(FuirError::ItemSizeMismatch,
                                   ErrorCategory::InvalidArgument),
                        ferr::message(where,
                                      ": fuir.desc.itemsize_mismatch: operand ",
                                      op, " has itemsize ", desc.itemsize,
                                      " but expected ", itemsize));
   }
}

void validate_operand_label_binding(
    const std::vector<OperandDescription> &descs,
    const OperandLabelBinding &binding, const std::string_view where) {
   FUSION_CHECK_CODE(
       binding.op_axis_labels.size() == descs.size(),
       fuir_error(FuirError::BindingOperandCountMismatch,
                  ErrorCategory::InvalidArgument),
       ferr::message(where,
                     ": fuir.binding.operand_count_mismatch: binding has ",
                     binding.op_axis_labels.size(), " operands but descs has ",
                     descs.size()));

   std::unordered_set<Label> labels_seen_anywhere;
   labels_seen_anywhere.reserve(64);

   for (std::size_t op = 0; op < descs.size(); ++op) {
      const OperandDescription &desc = descs[op];
      const std::vector<Label> &labels = binding.op_axis_labels[op];

      FUSION_CHECK_CODE(
          labels.size() == desc.ndims(),
          fuir_error(FuirError::BindingAxisCountMismatch,
                     ErrorCategory::InvalidArgument),
          ferr::message(where, ": fuir.binding.axis_count_mismatch: operand ",
                        op, " has ", labels.size(), " labels but rank ",
                        desc.ndims()));

      std::unordered_set<Label> labels_seen_in_operand;
      labels_seen_in_operand.reserve(labels.size());

      for (Label label : labels) {
         FUSION_CHECK_CODE(
             labels_seen_in_operand.insert(label).second,
             fuir_error(FuirError::RepeatedOperandLabelUnsupported,
                        ErrorCategory::Unsupported),
             ferr::message(where,
                           ": fuir.binding.repeated_operand_label: label ",
                           label, " appears more than once in operand ", op,
                           " diagonal-style bindings are not supported yet"));

         labels_seen_anywhere.insert(label);
      }
   }

   for (Label label : binding.out_labels) {
      FUSION_CHECK_CODE(
          labels_seen_anywhere.contains(label),
          fuir_error(FuirError::OutputLabelMissing,
                     ErrorCategory::InvalidArgument),
          ferr::message(where,
                        ": fuir.binding.output_label_missing: output label ",
                        label, " does not appear in any operand"));
   }

   if (!descs.empty()) {
      const std::vector<Label> &output_operand_labels =
          binding.op_axis_labels.front();

      FUSION_CHECK_CODE(
          output_operand_labels.size() == binding.out_labels.size(),
          fuir_error(FuirError::OutputLabelMismatch,
                     ErrorCategory::InvalidArgument),
          ferr::message(where,
                        ": fuir.binding.output_label_count_mismatch: "
                        "operand 0 has ",
                        output_operand_labels.size(),
                        " labels but out_labels has ",
                        binding.out_labels.size()));

      for (std::size_t i = 0; i < binding.out_labels.size(); ++i) {
         FUSION_CHECK_CODE(
             output_operand_labels[i] == binding.out_labels[i],
             fuir_error(FuirError::OutputLabelMismatch,
                        ErrorCategory::InvalidArgument),
             ferr::message(where,
                           ": fuir.binding.output_label_order_mismatch: "
                           "operand 0 label at axis ",
                           i, " is ", output_operand_labels[i],
                           " but out_labels has ", binding.out_labels[i]));
      }
   }
}

void validate_reduction_request(const std::vector<OperandDescription> &descs,
                                const std::size_t axis, const bool keepdim,
                                const std::string_view where) {
   FUSION_CHECK_CODE(
       descs.size() >= 2,
       fuir_error(FuirError::DescriptorCountMismatch,
                  ErrorCategory::InvalidArgument),
       ferr::message(where,
                     ": fuir.reduction.invalid_desc_count: expected at least "
                     "{out, in}, got ",
                     descs.size()));

   const OperandDescription &out_desc = descs.front();
   const OperandDescription &in_desc = descs.back();
   const std::size_t in_nd = in_desc.ndims();

   FUSION_CHECK_CODE(
       axis < in_nd,
       fuir_error(FuirError::InvalidAxis, ErrorCategory::InvalidArgument),
       ferr::message(where, ": fuir.reduction.invalid_axis: axis ", axis,
                     " out of range for input rank ", in_nd));

   for (std::size_t op = 1; op < descs.size(); ++op) {
      FUSION_CHECK_CODE(
          descs[op].ndims() == in_nd,
          fuir_error(FuirError::OperandRankMismatch,
                     ErrorCategory::InvalidArgument),
          ferr::message(where, ": fuir.reduction.input_rank_mismatch: operand ",
                        op, " has rank ", descs[op].ndims(), " but expected ",
                        in_nd));
   }

   if (keepdim) {
      FUSION_CHECK_CODE(
          out_desc.ndims() == in_nd,
          fuir_error(FuirError::OperandRankMismatch,
                     ErrorCategory::InvalidArgument),
          ferr::message(where,
                        ": fuir.reduction.keepdim_rank_mismatch: output rank ",
                        out_desc.ndims(), " but input rank is ", in_nd));

      FUSION_CHECK_CODE(
          out_desc.shape[axis] == 1,
          fuir_error(FuirError::BroadcastMismatch,
                     ErrorCategory::InvalidArgument),
          ferr::message(where,
                        ": fuir.reduction.keepdim_axis_shape_mismatch: "
                        "output shape at reduced axis ",
                        axis, " must be 1, got ", out_desc.shape[axis]));

      for (std::size_t ax = 0; ax < in_nd; ++ax) {
         if (ax == axis) {
            continue;
         }

         FUSION_CHECK_CODE(
             out_desc.shape[ax] == in_desc.shape[ax],
             fuir_error(FuirError::BroadcastMismatch,
                        ErrorCategory::InvalidArgument),
             ferr::message(where,
                           ": fuir.reduction.output_shape_mismatch: output "
                           "axis ",
                           ax, " has dim ", out_desc.shape[ax],
                           " but input dim is ", in_desc.shape[ax]));
      }

      return;
   }

   FUSION_CHECK_CODE(
       out_desc.ndims() == in_nd - 1,
       fuir_error(FuirError::OperandRankMismatch,
                  ErrorCategory::InvalidArgument),
       ferr::message(where,
                     ": fuir.reduction.output_rank_mismatch: output rank ",
                     out_desc.ndims(), " but expected ", in_nd - 1));

   for (std::size_t in_ax = 0; in_ax < in_nd; ++in_ax) {
      if (in_ax == axis) {
         continue;
      }

      const std::size_t out_ax = in_ax < axis ? in_ax : in_ax - 1;

      FUSION_CHECK_CODE(
          out_desc.shape[out_ax] == in_desc.shape[in_ax],
          fuir_error(FuirError::BroadcastMismatch,
                     ErrorCategory::InvalidArgument),
          ferr::message(
              where, ": fuir.reduction.output_shape_mismatch: output axis ",
              out_ax, " has dim ", out_desc.shape[out_ax], " but input axis ",
              in_ax, " has dim ", in_desc.shape[in_ax]));
   }
}

void validate_index_space_ir(const IndexSpaceIR &ir,
                             const std::string_view where) {
   FUSION_CHECK_CODE(
       ir.num_operands > 0,
       fuir_error(FuirError::InvalidIR, ErrorCategory::Internal),
       ferr::message(
           where, ": fuir.ir.invalid_num_operands: num_operands must be > 0"));

   FUSION_CHECK_CODE(
       ir.itemsize > 0,
       fuir_error(FuirError::InvalidIR, ErrorCategory::Internal),
       ferr::message(where,
                     ": fuir.ir.invalid_itemsize: itemsize must be > 0"));

   for (std::size_t id = 0; id < ir.indices.size(); ++id) {
      const IndexDef &index = ir.indices[id];

      FUSION_CHECK_CODE(
          index.extent > 0,
          fuir_error(FuirError::InvalidIR, ErrorCategory::Internal),
          ferr::message(where, ": fuir.ir.invalid_index_extent: index ", id,
                        " has extent ", index.extent));

      FUSION_CHECK_CODE(
          index.axis_of_operand.size() == ir.num_operands,
          fuir_error(FuirError::InvalidIR, ErrorCategory::Internal),
          ferr::message(where, ": fuir.ir.axis_binding_rank_mismatch: index ",
                        id, " has axis_of_operand size ",
                        index.axis_of_operand.size(), " but num_operands is ",
                        ir.num_operands));
   }

   for (std::uint32_t id : ir.out_indices) {
      FUSION_CHECK_CODE(
          id < ir.indices.size(),
          fuir_error(FuirError::InvalidIndexId, ErrorCategory::Internal),
          ferr::message(where, ": fuir.ir.invalid_out_index_id: out index id ",
                        id, " but indices size is ", ir.indices.size()));

      FUSION_CHECK_CODE(
          ir.indices[id].kind == IndexKind::Independent,
          fuir_error(FuirError::InvalidIR, ErrorCategory::Internal),
          ferr::message(where,
                        ": fuir.ir.invalid_out_index_kind: out index id ", id,
                        " is not independent"));
   }
}

void validate_loop_order(const IndexSpaceIR &ir,
                         const std::vector<std::uint32_t> &loop_order,
                         const std::string_view where) {
   validate_index_space_ir(ir, where);

   for (std::size_t pos = 0; pos < loop_order.size(); ++pos) {
      const std::uint32_t id = loop_order[pos];

      FUSION_CHECK_CODE(
          id < ir.indices.size(),
          fuir_error(FuirError::InvalidIndexId, ErrorCategory::Internal),
          ferr::message(
              where, ": fuir.lowering.invalid_loop_index_id: loop_order[", pos,
              "] = ", id, " but indices size is ", ir.indices.size()));
   }
}

void validate_desc_count_matches_ir(
    const IndexSpaceIR &ir, const std::vector<OperandDescription> &descs,
    const std::string_view where) {
   FUSION_CHECK_CODE(
       descs.size() == ir.num_operands,
       fuir_error(FuirError::DescriptorCountMismatch, ErrorCategory::Internal),
       ferr::message(where, ": fuir.lowering.desc_count_mismatch: descs has ",
                     descs.size(), " operands but IR has ", ir.num_operands));
}

void validate_ir_matches_descs(const IndexSpaceIR &ir,
                               const std::vector<OperandDescription> &descs,
                               const std::string_view where) {
   validate_index_space_ir(ir, where);
   validate_desc_count_matches_ir(ir, descs, where);

   for (std::size_t index_id = 0; index_id < ir.indices.size(); ++index_id) {
      const IndexDef &index = ir.indices[index_id];

      for (std::size_t op = 0; op < ir.num_operands; ++op) {
         const std::int32_t axis = index.axis_of_operand[op];

         if (axis < 0) {
            continue;
         }

         const std::size_t axis_u = static_cast<std::size_t>(axis);

         FUSION_CHECK_CODE(
             axis_u < descs[op].ndims(),
             fuir_error(FuirError::InvalidIR, ErrorCategory::Internal),
             ferr::message(where, ": fuir.ir.invalid_axis_binding: index ",
                           index_id, " binds operand ", op, " to axis ", axis,
                           " but operand rank is ", descs[op].ndims()));
      }
   }
}

void validate_role_vector_matches_ir(const IndexSpaceIR &ir,
                                     const std::vector<IndexRole> *role_of_id,
                                     const std::string_view where) {
   if (role_of_id == nullptr) {
      return;
   }

   FUSION_CHECK_CODE(
       role_of_id->size() == ir.indices.size(),
       fuir_error(FuirError::InvalidIR, ErrorCategory::Internal),
       ferr::message(where,
                     ": fuir.lowering.role_vector_size_mismatch: role_of_id "
                     "has size ",
                     role_of_id->size(), " but IR has ", ir.indices.size(),
                     " indices"));
}

} // namespace fusion::fuir::validation