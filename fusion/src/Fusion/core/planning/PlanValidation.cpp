#include "Fusion/core/planning/PlanValidation.h"

#include <cstddef>
#include <variant>

#include "Fusion/common/error/Check.h"
#include "Fusion/core/planning/PlanErrors.h"
#include "Fusion/core/planning/TraversalPlan.h"

namespace fusion::planning::validation {
namespace {

using error::ErrorCategory;

void validate_plan_core(const ExecutionPlan& plan,
                        const std::string_view where) {
   FUSION_CHECK_CODE(
       plan.core.itemsize > 0,
       planning_error(PlanningError::InvalidItemSize,
                      ErrorCategory::Internal),
       where,
       ": planning.execution.invalid_itemsize: itemsize must be > 0");

   FUSION_CHECK_CODE(
       plan.core.out_ndim == plan.core.out_shape.size(),
       planning_error(PlanningError::InvalidOutputRank,
                      ErrorCategory::Internal),
       where,
       ": planning.execution.output_rank_mismatch: out_ndim ",
       plan.core.out_ndim,
       " does not match out_shape rank ",
       plan.core.out_shape.size());

   FUSION_CHECK_CODE(
       plan.core.num_operands == plan.access.operands.size(),
       planning_error(PlanningError::AccessOperandCountMismatch,
                      ErrorCategory::Internal),
       where,
       ": planning.execution.operand_count_mismatch: core has ",
       plan.core.num_operands,
       " operands but access plan has ",
       plan.access.operands.size());
}

void validate_operand_access_ids(const ExecutionPlan& plan,
                                 const std::string_view where) {
   for (std::size_t i = 0; i < plan.access.operands.size(); ++i) {
      const OperandAccess& access = plan.access.operands[i];

      FUSION_CHECK_CODE(
          access.operand_id < plan.core.num_operands,
          planning_error(PlanningError::AccessOperandIdMismatch,
                         ErrorCategory::Internal),
          where,
          ": planning.execution.invalid_operand_id: access entry ",
          i,
          " has operand_id ",
          access.operand_id,
          " but num_operands is ",
          plan.core.num_operands);
   }
}

} // namespace

void validate_operand_description(const OperandDescription& desc,
                                  const std::string_view where) {
   FUSION_CHECK_CODE(
       desc.itemsize > 0,
       planning_error(PlanningError::InvalidItemSize,
                      ErrorCategory::InvalidArgument),
       where,
       ": planning.operand.invalid_itemsize: itemsize must be > 0");

   FUSION_CHECK_CODE(
       desc.strides.size() == desc.shape.size(),
       planning_error(PlanningError::ShapeRankMismatch,
                      ErrorCategory::InvalidArgument),
       where,
       ": planning.operand.rank_mismatch: strides rank ",
       desc.strides.size(),
       " does not match shape rank ",
       desc.shape.size());
}

void validate_dense_execution_plan(const ExecutionPlan& plan,
                                   const std::string_view where) {
   const auto* dense = std::get_if<DenseTraversalPlan>(&plan.traversal);

   FUSION_CHECK_CODE(
       dense != nullptr,
       planning_error(PlanningError::TraversalPayloadMismatch,
                      ErrorCategory::Internal),
       where,
       ": planning.execution.traversal_payload_mismatch: "
       "traversal_kind is Dense but traversal payload is not DenseTraversalPlan");

   for (std::size_t i = 0; i < plan.access.operands.size(); ++i) {
      const OperandAccess& access = plan.access.operands[i];

      FUSION_CHECK_CODE(
          access.access == AccessKind::Affine,
          planning_error(PlanningError::NonAffineDenseAccess,
                         ErrorCategory::Internal),
          where,
          ": planning.execution.non_affine_dense_access: dense execution "
          "requires affine access for operand ",
          i);

      FUSION_CHECK_CODE(
          access.affine.byte_stride_per_loop.size() == dense->loop.size(),
          planning_error(PlanningError::AccessRankMismatch,
                         ErrorCategory::Internal),
          where,
          ": planning.execution.access_rank_mismatch: operand ",
          i,
          " has stride rank ",
          access.affine.byte_stride_per_loop.size(),
          " but dense loop rank is ",
          dense->loop.size());
   }
}

// void validate_indexed_execution_plan(const ExecutionPlan& plan,
//                                      const std::string_view where) {
//    const auto* indexed = std::get_if<IndexedTraversalPlan>(&plan.traversal);
//
//    FUSION_CHECK_CODE(
//        indexed != nullptr,
//        planning_error(PlanningError::TraversalPayloadMismatch,
//                       ErrorCategory::Internal),
//        where,
//        ": planning.execution.traversal_payload_mismatch: "
//        "traversal_kind is Indexed but traversal payload is not IndexedTraversalPlan");
//
//    FUSION_CHECK_CODE(
//        indexed->topology.N > 0,
//        planning_error(PlanningError::InvalidPlanCore,
//                       ErrorCategory::Internal),
//        where,
//        ": planning.indexed.invalid_topology: topology.N must be > 0");
//
// }

void validate_execution_plan(const ExecutionPlan& plan,
                             const std::string_view where) {
   validate_plan_core(plan, where);
   validate_operand_access_ids(plan, where);

   switch (plan.core.traversal_kind) {
   case TraversalKind::Dense:
      validate_dense_execution_plan(plan, where);
      break;

   case TraversalKind::Indexed: // TODO: update when validate_indexed_plan is implemented
      FUSION_CHECK_CODE(
          false,
          planning_error(PlanningError::UnsupportedTraversal,
                         ErrorCategory::Internal),
          where,
          ": planning.execution.indexed_validation_not_implemented");
      break;

   default:
      FUSION_CHECK_CODE(
          false,
          planning_error(PlanningError::UnsupportedTraversal,
                         ErrorCategory::Internal),
          where,
          ": planning.execution.unsupported_traversal_kind");
   }
}

} // namespace fusion::planning::validation