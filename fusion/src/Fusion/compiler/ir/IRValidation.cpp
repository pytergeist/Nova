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

static void validate_physical_axis_extents(const PhysicalAxesByOperand &physical_axes,
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
    const std::vector<LogicalAxis> &logical_axes, std::string_view where) {
   for (const OperandUse &operand_use : operand_uses) {

      for (const AxisUse &axis_use : operand_use.axis_use) {
         FUSION_CHECK_CODE(
             static_cast<std::size_t>(axis_use.logical_axis_id) <
                 logical_axes.size(),
             fuir_error(FuirError::InvalidLogicalAxisReference,
                        ErrorCategory::InvalidArgument),
             ferr::message(
                 where, ": fuir.axis_use.invalid_logical_axis_id: operand ",
                 operand_use.operand_id, " references logical_axis_id ",
                 axis_use.logical_axis_id, ", but that operation has ",
                 logical_axes.size(), " logical axes"));
      }
   }
}

void validate_unique_physical_axis_uses(
    const std::vector<OperandUse> &operand_uses, std::string_view where) {
   for (const OperandUse &operand_use : operand_uses) {
      std::unordered_set<PhysicalAxisId> axes_set_per_operand;
      axes_set_per_operand.reserve(operand_use.axis_use.size());
      for (const AxisUse &axis_use : operand_use.axis_use) {
         const bool inserted =
             axes_set_per_operand.insert(axis_use.physical_axis_id).second;
         FUSION_CHECK_CODE(
             inserted,
             fuir_error(FuirError::DuplicatePhysicalAxisUse,
                        ErrorCategory::InvalidArgument),
             ferr::message(
                 where,
                 ": fuir.operand_use.duplicate_physical_axis_use: operand ",
                 operand_use.operand_id, " references physical_axis_id ",
                 axis_use.physical_axis_id, " more than once"));
      }
   }
}

void validate_direct_axis_uses(const std::vector<OperandUse> &operand_uses,
                               const PhysicalAxesByOperand &physical_axes,
                               const std::vector<LogicalAxis> &logical_axes,
                               std::string_view where) {
   for (const OperandUse &operand_use : operand_uses) {
      for (const AxisUse &axis_use : operand_use.axis_use) {
         if (axis_use.access != AxisAccess::Direct) {
            continue;
         }

         const PhysicalAxis &physical_axis =
             physical_axes.at(operand_use.operand_id)
                 .at(axis_use.physical_axis_id);

         const LogicalAxis &logical_axis =
             logical_axes.at(axis_use.logical_axis_id);

         FUSION_CHECK_CODE(
             physical_axis.extent == logical_axis.extent,
             fuir_error(FuirError::DirectExtentMismatch,
                        ErrorCategory::InvalidArgument),
             ferr::message(
                 where, ": fuir.axis_use.direct_extent_mismatch: operand ",
                 operand_use.operand_id, " maps physical_axis_id ",
                 axis_use.physical_axis_id, " directly to logical_axis_id ",
                 axis_use.logical_axis_id, ", but physical extent ",
                 physical_axis.extent, " does not equal logical extent ",
                 logical_axis.extent));
      }
   }
}

void validate_broadcast_axis_uses(const std::vector<OperandUse> &operand_uses,
                                  const PhysicalAxesByOperand &physical_axes,
                                  const std::vector<LogicalAxis> &logical_axes,
                                  std::string_view where) {
   for (const OperandUse &operand_use : operand_uses) {
      for (const AxisUse &axis_use : operand_use.axis_use) {
         if (axis_use.access != AxisAccess::Broadcast) {
            continue;
         }
         const PhysicalAxis &physical_axis =
             physical_axes.at(operand_use.operand_id)
                 .at(axis_use.physical_axis_id);

         const LogicalAxis &logical_axis =
             logical_axes.at(axis_use.logical_axis_id);

         FUSION_CHECK_CODE(
             physical_axis.extent == 1 && logical_axis.extent > 1,
             fuir_error(FuirError::BroadcastExtentMismatch,
                        ErrorCategory::InvalidArgument),
             ferr::message(
                 where, ": fuir.axis_use.broadcast_extent_mismatch: operand ",
                 operand_use.operand_id, " maps physical_axis_id ",
                 axis_use.physical_axis_id, " to logical_axis_id ",
                 axis_use.logical_axis_id,
                 " with Broadcast access, but Broadcast requires physical "
                 "extent 1 "
                 "and logical extent greater than 1; got physical extent ",
                 physical_axis.extent, " and logical extent ",
                 logical_axis.extent));
      }
   }
}

void validate_no_indexed_axis_uses(const std::vector<OperandUse> &operand_uses,
                                   std::string_view where) {
   for (const OperandUse &operand_use : operand_uses) {
      for (const AxisUse &axis_use : operand_use.axis_use) {
         FUSION_CHECK_CODE(
             axis_use.access != AxisAccess::Indexed,
             fuir_error(FuirError::IndexedAccessUnsupported,
                        ErrorCategory::InvalidArgument),
             ferr::message(
                 where, ": fuir.axis_use.indexed_access_unsupported: operand ",
                 operand_use.operand_id,
                 " uses Indexed access for "
                 "physical_axis_id ",
                 axis_use.physical_axis_id, " targeting logical_axis_id ",
                 axis_use.logical_axis_id,
                 "; Indexed access is not supported by the elementwise "
                 "builder"));
      }
   }
}

void validate_logical_axis_participation(
    const std::vector<LogicalAxis> &logical_axes,
    const std::vector<OperandUse> &operand_uses, std::string_view where) {
   std::vector<bool> has_participating_axis(logical_axes.size(), false);

   for (const OperandUse &operand_use : operand_uses) {
      for (const AxisUse &axis_use : operand_use.axis_use) {
         has_participating_axis.at(axis_use.logical_axis_id) = true;
      }
   }

   for (std::size_t logical_axis_id = 0; logical_axis_id < logical_axes.size();
        ++logical_axis_id) {
      const LogicalAxis &logical_axis = logical_axes[logical_axis_id];

      FUSION_CHECK_CODE(
          has_participating_axis[logical_axis_id],
          fuir_error(FuirError::UnusedLogicalAxis,
                     ErrorCategory::InvalidArgument),
          ferr::message(where, ": fuir.logical_axis.unused: logical_axis_id ",
                        logical_axis_id, " with label ", logical_axis.label,
                        " and extent ", logical_axis.extent,
                        " is not referenced by any AxisUse"));
   }
}

void validate_operand_use_operand_ids(
    const std::vector<OperandUse> &operand_uses, std::size_t num_operands,
    std::string_view where) {
   FUSION_CHECK_CODE(
       operand_uses.size() == num_operands,
       fuir_error(FuirError::InvalidOperandUseId,
                  ErrorCategory::InvalidArgument),
       ferr::message(where, ": fuir.operand_use.count_mismatch: expected ",
                     num_operands, " OperandUse entries, but got ",
                     operand_uses.size()));

   for (std::size_t op = 0; op < operand_uses.size(); ++op) {
      const OperandUse &operand_use = operand_uses[op];

      FUSION_CHECK_CODE(
          operand_use.operand_id == static_cast<OperandId>(op),
          fuir_error(FuirError::InvalidOperandUseId,
                     ErrorCategory::InvalidArgument),
          ferr::message(
              where,
              ": fuir.operand_use.operand_id_mismatch: OperandUse declares "
              "operand_id ",
              operand_use.operand_id, ", but is stored at operand position ",
              op));
   }
}


void validate_physical_axis_collection_count(
    const PhysicalAxesByOperand &physical_axes,
    const std::size_t num_operands,
    const std::string_view where) {
   FUSION_CHECK_CODE(
       physical_axes.size() == num_operands,
       fuir_error(FuirError::PhysicalAxisCollectionCountMismatch,
                  ErrorCategory::InvalidArgument),
       ferr::message(
           where,
           ": fuir.physical_axes.operand_count_mismatch: expected ",
           num_operands, " physical-axis collections, but found ",
           physical_axes.size()));
}

void validate_logical_axis_extents(
    const std::vector<LogicalAxis> &logical_axes,
    const std::string_view where) {
   for (std::size_t logical_axis_id = 0;
        logical_axis_id < logical_axes.size(); ++logical_axis_id) {
      const LogicalAxis &logical_axis = logical_axes[logical_axis_id];

      FUSION_CHECK_CODE(
          logical_axis.extent > 0,
          fuir_error(FuirError::InvalidLogicalExtent,
                     ErrorCategory::InvalidArgument),
          ferr::message(
              where,
              ": fuir.logical_axis.invalid_extent: logical_axis_id ",
              logical_axis_id, " has extent ", logical_axis.extent,
              "; expected an extent greater than zero"));
        }
}

void validate_complete_physical_axis_uses(
    const std::vector<OperandUse> &operand_uses,
    const PhysicalAxesByOperand &physical_axes,
    const std::string_view where) {
   for (const OperandUse &operand_use : operand_uses) {
      const std::size_t operand_id =
          static_cast<std::size_t>(operand_use.operand_id);

      const std::size_t physical_axis_count =
          physical_axes.at(operand_id).size();

      FUSION_CHECK_CODE(
          operand_use.axis_use.size() == physical_axis_count,
          fuir_error(FuirError::IncompletePhysicalAxisUse,
                     ErrorCategory::InvalidArgument),
          ferr::message(
              where,
              ": fuir.operand_use.incomplete_physical_axis_use: operand ",
              operand_use.operand_id, " has ", physical_axis_count,
              " physical axes, but ", operand_use.axis_use.size(),
              " axis uses"));
   }
}

void validate_unique_logical_axis_uses(
    const std::vector<OperandUse> &operand_uses,
    const std::string_view where) {
   for (const OperandUse &operand_use : operand_uses) {
      std::unordered_set<LogicalAxisId> logical_axis_ids;
      logical_axis_ids.reserve(operand_use.axis_use.size());

      for (const AxisUse &axis_use : operand_use.axis_use) {
         const bool inserted =
             logical_axis_ids.insert(axis_use.logical_axis_id).second;

         FUSION_CHECK_CODE(
             inserted,
             fuir_error(FuirError::DuplicateLogicalAxisUse,
                        ErrorCategory::InvalidArgument),
             ferr::message(
                 where,
                 ": fuir.operand_use.duplicate_logical_axis_use: operand ",
                 operand_use.operand_id, " references logical_axis_id ",
                 axis_use.logical_axis_id, " more than once"));
      }
   }
}

void validate_unary_reduction_operand_count(
    const IndexSpaceIR &ir,
    const std::string_view where) {
   FUSION_CHECK_CODE(
       ir.num_operands == 2,
       fuir_error(FuirError::UnaryReductionOperandCountMismatch,
                  ErrorCategory::InvalidArgument),
       ferr::message(
           where,
           ": fuir.reduction.operand_count_mismatch: unary reduction "
           "requires exactly two operands—output operand 0 and input "
           "operand 1—but found ",
           ir.num_operands));
}

void validate_single_reduction_logical_axis(
    const std::vector<LogicalAxis> &logical_axes,
    const std::string_view where) {
   std::size_t reduction_axis_count = 0;

   for (const LogicalAxis &logical_axis : logical_axes) {
      if (logical_axis.kind == IndexKind::Reduction) {
         ++reduction_axis_count;
      }
   }

   FUSION_CHECK_CODE(
       reduction_axis_count == 1,
       fuir_error(FuirError::InvalidReductionAxisCount,
                  ErrorCategory::InvalidArgument),
       ferr::message(
           where,
           ": fuir.reduction.invalid_axis_count: unary single-axis "
           "reduction requires exactly one reduction logical axis, but found ",
           reduction_axis_count));
}

void validate_unary_reduction_input_mapping(
    const std::vector<LogicalAxis> &logical_axes,
    const std::vector<PhysicalAxis> &input_axes,
    const OperandUse &input_use,
    const std::string_view where) {
   FUSION_CHECK_CODE(
       input_axes.size() == logical_axes.size(),
       fuir_error(FuirError::ReductionInputMappingMismatch,
                  ErrorCategory::InvalidArgument),
       ferr::message(
           where,
           ": fuir.reduction.input_rank_mismatch: input operand ",
           input_use.operand_id, " has ", input_axes.size(),
           " physical axes, but the reduction has ", logical_axes.size(),
           " logical axes"));

   for (const AxisUse &axis_use : input_use.axis_use) {
      const LogicalAxisId expected_logical_axis_id =
          static_cast<LogicalAxisId>(axis_use.physical_axis_id);

      FUSION_CHECK_CODE(
          axis_use.logical_axis_id == expected_logical_axis_id,
          fuir_error(FuirError::ReductionInputMappingMismatch,
                     ErrorCategory::InvalidArgument),
          ferr::message(
              where,
              ": fuir.reduction.input_axis_mapping_mismatch: input operand ",
              input_use.operand_id, " physical_axis_id ",
              axis_use.physical_axis_id, " maps to logical_axis_id ",
              axis_use.logical_axis_id, ", but expected logical_axis_id ",
              expected_logical_axis_id));

      FUSION_CHECK_CODE(
          axis_use.access == AxisAccess::Direct,
          fuir_error(FuirError::ReductionInputMappingMismatch,
                     ErrorCategory::InvalidArgument),
          ferr::message(
              where,
              ": fuir.reduction.invalid_input_access: input operand ",
              input_use.operand_id, " physical_axis_id ",
              axis_use.physical_axis_id,
              " must use Direct access in a unary reduction"));
   }
}

void validate_unary_reduction_output_mapping(
    const std::vector<LogicalAxis> &logical_axes,
    const std::vector<PhysicalAxis> &output_axes,
    const OperandUse &output_use,
    const std::string_view where) {
   std::size_t reduction_axis_id = logical_axes.size();

   for (std::size_t logical_axis_id = 0;
        logical_axis_id < logical_axes.size(); ++logical_axis_id) {
      if (logical_axes[logical_axis_id].kind == IndexKind::Reduction) {
         reduction_axis_id = logical_axis_id;
         break;
      }
   }

   FUSION_CHECK_CODE(
       reduction_axis_id < logical_axes.size(),
       fuir_error(FuirError::InvalidReductionAxisCount,
                  ErrorCategory::InvalidArgument),
       ferr::message(
           where,
           ": fuir.reduction.missing_reduction_axis: no reduction logical "
           "axis was found"));

   const bool keepdim = output_axes.size() == logical_axes.size();
   const bool removes_dimension =
       logical_axes.size() == output_axes.size() + 1;

   FUSION_CHECK_CODE(
       keepdim || removes_dimension,
       fuir_error(FuirError::ReductionOutputMappingMismatch,
                  ErrorCategory::InvalidArgument),
       ferr::message(
           where,
           ": fuir.reduction.output_rank_mismatch: output operand ",
           output_use.operand_id, " has ", output_axes.size(),
           " physical axes, but expected ", logical_axes.size(),
           " for keepdim=true or ", logical_axes.size() - 1,
           " for keepdim=false"));

   for (const AxisUse &axis_use : output_use.axis_use) {
      const std::size_t physical_axis_id =
          static_cast<std::size_t>(axis_use.physical_axis_id);

      const std::size_t expected_logical_axis_id =
          keepdim
              ? physical_axis_id
              : physical_axis_id < reduction_axis_id
                    ? physical_axis_id
                    : physical_axis_id + 1;

      FUSION_CHECK_CODE(
          static_cast<std::size_t>(axis_use.logical_axis_id) ==
              expected_logical_axis_id,
          fuir_error(FuirError::ReductionOutputMappingMismatch,
                     ErrorCategory::InvalidArgument),
          ferr::message(
              where,
              ": fuir.reduction.output_axis_mapping_mismatch: output "
              "physical_axis_id ",
              axis_use.physical_axis_id, " maps to logical_axis_id ",
              axis_use.logical_axis_id, ", but expected logical_axis_id ",
              expected_logical_axis_id));

      if (expected_logical_axis_id == reduction_axis_id) {
         const PhysicalAxis &physical_axis =
             output_axes.at(physical_axis_id);
         const LogicalAxis &logical_axis =
             logical_axes.at(expected_logical_axis_id);

         FUSION_CHECK_CODE(
             physical_axis.extent == 1,
             fuir_error(FuirError::ReductionOutputMappingMismatch,
                        ErrorCategory::InvalidArgument),
             ferr::message(
                 where,
                 ": fuir.reduction.keepdim_extent_mismatch: reduced output "
                 "physical_axis_id ",
                 axis_use.physical_axis_id, " has extent ",
                 physical_axis.extent, "; expected extent 1"));

         const AxisAccess expected_access =
             logical_axis.extent > 1 ? AxisAccess::Broadcast
                                     : AxisAccess::Direct;

         FUSION_CHECK_CODE(
             axis_use.access == expected_access,
             fuir_error(FuirError::ReductionOutputMappingMismatch,
                        ErrorCategory::InvalidArgument),
             ferr::message(
                 where,
                 ": fuir.reduction.keepdim_access_mismatch: reduced output "
                 "physical_axis_id ",
                 axis_use.physical_axis_id,
                 " has the wrong access kind for logical extent ",
                 logical_axis.extent));
      } else {
         FUSION_CHECK_CODE(
             axis_use.access == AxisAccess::Direct,
             fuir_error(FuirError::ReductionOutputMappingMismatch,
                        ErrorCategory::InvalidArgument),
             ferr::message(
                 where,
                 ": fuir.reduction.invalid_output_access: independent "
                 "output physical_axis_id ",
                 axis_use.physical_axis_id,
                 " must use Direct access"));
      }
   }
}

} // namespace detail

void validate_elementwise_index_space_ir(const IndexSpaceIR &ir,
                                         const std::string_view where) {

   detail::validate_physical_axis_operand_ids(ir.physical_axes, where);
   detail::validate_physical_axis_ids(ir.physical_axes, where);
   detail::validate_physical_axis_extents(ir.physical_axes, where);

   detail::validate_operand_use_operand_ids(ir.operand_use, ir.num_operands,
                                            where);

   detail::validate_axis_use_physical_axis_references(ir.operand_use,
                                                      ir.physical_axes, where);

   detail::validate_axis_use_logical_axis_references(ir.operand_use,
                                                     ir.logical_axes, where);

   detail::validate_unique_physical_axis_uses(ir.operand_use, where);
   detail::validate_no_indexed_axis_uses(ir.operand_use, where);

   detail::validate_direct_axis_uses(ir.operand_use, ir.physical_axes,
                                     ir.logical_axes, where);

   detail::validate_broadcast_axis_uses(ir.operand_use, ir.physical_axes,
                                        ir.logical_axes, where);

   detail::validate_logical_axis_participation(ir.logical_axes, ir.operand_use,
                                               where);
}


void validate_unary_reduction_index_space_ir(
    const IndexSpaceIR &ir,
    const std::string_view where) {

   detail::validate_unary_reduction_operand_count(ir, where);

   detail::validate_physical_axis_collection_count(
       ir.physical_axes, ir.num_operands, where);

   detail::validate_physical_axis_operand_ids(ir.physical_axes, where);
   detail::validate_physical_axis_ids(ir.physical_axes, where);
   detail::validate_physical_axis_extents(ir.physical_axes, where);
   detail::validate_logical_axis_extents(ir.logical_axes, where);

   detail::validate_operand_use_operand_ids(
       ir.operand_use, ir.num_operands, where);

   detail::validate_axis_use_physical_axis_references(
       ir.operand_use, ir.physical_axes, where);

   detail::validate_axis_use_logical_axis_references(
       ir.operand_use, ir.logical_axes, where);

   detail::validate_unique_physical_axis_uses(ir.operand_use, where);
   detail::validate_unique_logical_axis_uses(ir.operand_use, where);

   detail::validate_complete_physical_axis_uses(
       ir.operand_use, ir.physical_axes, where);

   detail::validate_no_indexed_axis_uses(ir.operand_use, where);

   detail::validate_direct_axis_uses(
       ir.operand_use, ir.physical_axes, ir.logical_axes, where);

   detail::validate_broadcast_axis_uses(
       ir.operand_use, ir.physical_axes, ir.logical_axes, where);

   detail::validate_single_reduction_logical_axis(
       ir.logical_axes, where);

   detail::validate_unary_reduction_input_mapping(
       ir.logical_axes,
       ir.physical_axes.at(1),
       ir.operand_use.at(1),
       where);

   detail::validate_unary_reduction_output_mapping(
       ir.logical_axes,
       ir.physical_axes.at(0),
       ir.operand_use.at(0),
       where);

   detail::validate_logical_axis_participation(
       ir.logical_axes, ir.operand_use, where);
}

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

   for (std::size_t id = 0; id < ir.logical_axes.size(); ++id) {
      const LogicalAxis &axis = ir.logical_axes[id];

      FUSION_CHECK_CODE(
          axis.extent > 0,
          fuir_error(FuirError::InvalidIR, ErrorCategory::Internal),
          ferr::message(where, ": fuir.ir.invalid_index_extent: logical axis ", id,
                        " has extent ", axis.extent));

   }
}

void validate_loop_order(const IndexSpaceIR &ir,
                         const std::vector<std::uint32_t> &loop_order,
                         const std::string_view where) {
   validate_index_space_ir(ir, where);

   for (std::size_t pos = 0; pos < loop_order.size(); ++pos) {
      const std::uint32_t id = loop_order[pos];

      FUSION_CHECK_CODE(
          id < ir.logical_axes.size(),
          fuir_error(FuirError::InvalidIndexId, ErrorCategory::Internal),
          ferr::message(
              where, ": fuir.lowering.invalid_loop_index_id: loop_order[", pos,
              "] = ", id, " but logical_axes size is ", ir.logical_axes.size()));
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

   }


void validate_role_vector_matches_ir(const IndexSpaceIR &ir,
                                     const std::vector<IndexRole> *role_of_id,
                                     const std::string_view where) {
   if (role_of_id == nullptr) {
      return;
   }

   FUSION_CHECK_CODE(
       role_of_id->size() == ir.logical_axes.size(),
       fuir_error(FuirError::InvalidIR, ErrorCategory::Internal),
       ferr::message(where,
                     ": fuir.lowering.role_vector_size_mismatch: role_of_id "
                     "has size ",
                     role_of_id->size(), " but IR has ", ir.logical_axes.size(),
                     " logical axes"));
}

} // namespace fusion::fuir::validation