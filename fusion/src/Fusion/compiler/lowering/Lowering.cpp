#include "Fusion/compiler/lowering/Lowering.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Fusion/common/error/Check.h"
#include "Fusion/compiler/ir/IRErrors.h"
#include "Fusion/compiler/ir/IRValidation.h"

namespace fusion::fuir {

namespace ferr = fusion::error;
using ferr::ErrorCategory;

namespace {
std::int64_t stride_bytes_for_logical_axis(const OperandDescription &desc,
                                           const OperandUse &operand_use,
                                           const LogicalAxisId logical_axis_id,
                                           const std::size_t itemsize) {

   constexpr std::string_view where = "stride_bytes_for_logical_axis";

   for (const AxisUse &axis_use : operand_use.axis_use) {
      if (axis_use.logical_axis_id != logical_axis_id) {
         continue;
      }

      switch (axis_use.access) {
      case AxisAccess::Direct:
         return static_cast<std::int64_t>(
                    desc.strides.at(axis_use.physical_axis_id)) *
                static_cast<std::int64_t>(itemsize);

      case AxisAccess::Broadcast:
         return 0;

      case AxisAccess::Indexed:
         FUSION_CHECK_UNSUPPORTED(
             false,
             ferr::message(
                 where,
                 ": fuir.lowering.indexed_axis_use_unsupported: operand ",
                 operand_use.operand_id, " maps logical_axis_id ",
                 logical_axis_id,
                 " using indexed access, which cannot be represented by an "
                 "affine "
                 "byte stride"));
         return 0; // unreachable return for control flow analysis (TODO: make
                   // macro noreturn?)
      }
   }

   return 0;
}
} // namespace
std::vector<LoopDim>
lower_to_loops(const IndexSpaceIR &ir,
               const std::vector<OperandDescription> &descs,
               const std::vector<LogicalAxisId> &loop_order) {
   constexpr std::string_view where = "lower_to_loops";

   validation::validate_ir_matches_descs(ir, descs, where);
   validation::validate_loop_order(ir, loop_order, where);

   std::vector<LoopDim> loops;
   loops.reserve(loop_order.size());

   for (const LogicalAxisId id : loop_order) {
      const LogicalAxis &axis = ir.logical_axes.at(id);

      LoopDim ld;
      ld.size = axis.extent;
      ld.kind = axis.kind;
      ld.role = IndexRole::Batch;

      loops.push_back(std::move(ld));
   }

   return loops;
}

std::vector<LoopDim>
lower_to_loops(const IndexSpaceIR &ir,
               const std::vector<OperandDescription> &descs,
               const std::vector<LogicalAxisId> &loop_order,
               const std::vector<IndexRole> *role_of_id) {
   constexpr std::string_view where = "lower_to_loops";

   validation::validate_ir_matches_descs(ir, descs, where);
   validation::validate_loop_order(ir, loop_order, where);
   validation::validate_role_vector_matches_ir(ir, role_of_id, where);

   std::vector<LoopDim> loops;
   loops.reserve(loop_order.size());

   for (const LogicalAxisId id : loop_order) {
      const LogicalAxis &axis = ir.logical_axes.at(id);

      LoopDim ld;
      ld.size = axis.extent;
      ld.kind = axis.kind;
      ld.role = role_of_id != nullptr ? (*role_of_id)[id] : IndexRole::Batch;

      loops.push_back(std::move(ld));
   }

   return loops;
}

std::vector<OperandAccess>
lower_operand_access(const IndexSpaceIR &ir,
                     const std::vector<OperandDescription> &descs,
                     const std::vector<LogicalAxisId> &loop_order) {
   constexpr std::string_view where = "lower_operand_access";

   validation::validate_ir_matches_descs(ir, descs, where);
   validation::validate_loop_order(ir, loop_order, where);

   std::vector<OperandAccess> op_access;
   op_access.reserve(ir.num_operands);

   for (std::size_t op = 0; op < ir.num_operands; ++op) {
      OperandAccess oa;
      AffineAccess af;

      oa.operand_id = op;
      oa.layout = descs.at(op).layout;
      oa.storage = descs.at(op).storage;
      oa.update = descs.at(op).update;
      oa.access = descs.at(op).access;

      af.byte_stride_per_loop.resize(loop_order.size());

      for (std::size_t pos = 0; pos < loop_order.size(); ++pos) {
         const LogicalAxisId logical_index_id = loop_order[pos];

         af.byte_stride_per_loop.at(pos) =
             stride_bytes_for_logical_axis(descs.at(op), ir.operand_use.at(op),
                                           logical_index_id, ir.itemsize);
      }

      oa.affine = std::move(af);
      op_access.push_back(std::move(oa));
   }

   return op_access;
}

} // namespace fusion::fuir