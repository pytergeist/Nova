#include "Fusion/compiler/lowering/LoweringValidation.h"

#include <cstddef>
#include <vector>

#include "Fusion/common/error/Check.h"
#include "Fusion/compiler/ir/IRErrors.h"
#include "Fusion/compiler/ir/IRValidation.h"

namespace fusion::lowering::lvalidation {

namespace ferr = fusion::error;
using ferr::ErrorCategory;

void validate_loop_order(
    const fuir::IndexSpaceIR &ir,
    const std::vector<fuir::LogicalAxisId> &loop_order,
    const std::string_view where) {
   fuir::validation::validate_index_space_ir(ir, where);

   FUSION_CHECK_CODE(
       loop_order.size() == ir.logical_axes.size(),
       fuir::fuir_error(fuir::FuirError::InvalidLoopOrder, ErrorCategory::Internal),
       ferr::message(
           where,
           ": fuir.lowering.loop_order_size_mismatch: loop_order has ",
           loop_order.size(), " entries, but the IR has ",
           ir.logical_axes.size(), " logical axes"));

   std::vector<bool> seen(ir.logical_axes.size(), false);

   for (std::size_t position = 0; position < loop_order.size(); ++position) {
      const fuir::LogicalAxisId logical_axis_id = loop_order[position];

      FUSION_CHECK_CODE(
          static_cast<std::size_t>(logical_axis_id) < ir.logical_axes.size(),
          fuir_error(fuir::FuirError::InvalidIndexId, ErrorCategory::Internal),
          ferr::message(
              where,
              ": fuir.lowering.invalid_loop_axis_id: loop_order[",
              position, "] = ", logical_axis_id,
              ", but the IR has ", ir.logical_axes.size(),
              " logical axes"));

      FUSION_CHECK_CODE(
          !seen[logical_axis_id],
          fuir_error(fuir::FuirError::InvalidLoopOrder, ErrorCategory::Internal),
          ferr::message(
              where,
              ": fuir.lowering.duplicate_loop_axis: logical_axis_id ",
              logical_axis_id, " appears more than once; duplicate at position ",
              position));

      seen[logical_axis_id] = true;
   }
}


void validate_ir_matches_descs(const fuir::IndexSpaceIR &ir,
                               const std::vector<fuir::OperandDescription> &descs,
                               const std::string_view where) {
   fuir::validation::validate_index_space_ir(ir, where);
   validate_desc_count_matches_ir(ir, descs, where);
}

void validate_role_vector_matches_ir(const fuir::IndexSpaceIR &ir,
                                     const std::vector<fuir::IndexRole> *role_of_id,
                                     const std::string_view where) {
   if (role_of_id == nullptr) {
      return;
   }

   FUSION_CHECK_CODE(
       role_of_id->size() == ir.logical_axes.size(),
       fuir_error(fuir::FuirError::InvalidIR, ErrorCategory::Internal),
       ferr::message(where,
                     ": fuir.lowering.role_vector_size_mismatch: role_of_id "
                     "has size ",
                     role_of_id->size(), " but IR has ", ir.logical_axes.size(),
                     " logical axes"));
}

void validate_desc_count_matches_ir(
    const fuir::IndexSpaceIR &ir, const std::vector<fuir::OperandDescription> &descs,
    const std::string_view where) {
   FUSION_CHECK_CODE(
       descs.size() == ir.num_operands,
       fuir_error(fuir::FuirError::DescriptorCountMismatch, ErrorCategory::Internal),
       ferr::message(where, ": fuir.lowering.desc_count_mismatch: descs has ",
                     descs.size(), " operands but IR has ", ir.num_operands));
}


} // namespace fusion::lowering::lvalidation