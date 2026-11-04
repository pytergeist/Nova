#ifndef FUSION_CORE_PLANNING_PLANNING_ERRORS_H
#define FUSION_CORE_PLANNING_PLANNING_ERRORS_H

#include <cstdint>

#include "Fusion/common/error/ErrorCode.h"

namespace fusion::planning {

enum class PlanningError : std::uint16_t {
   InvalidOperand = 1,
   ShapeRankMismatch,
   ItemSizeMismatch,
   UnsupportedLayout,

   InvalidAxis,
   InvalidBinding,
   InvalidReduction,
   InvalidContraction,
   ShapeMismatch,
   OutputShapeMismatch,

   InvalidPlanCore,
   TraversalPayloadMismatch,
   AccessOperandCountMismatch,
   AccessOperandIdMismatch,
   AccessRankMismatch,
   NonAffineDenseAccess,
   InvalidItemSize,
   InvalidOutputRank,
   UnsupportedTraversal,
};

constexpr error::ErrorCode
planning_error(const PlanningError detail,
               const error::ErrorCategory category) noexcept {
   return error::ErrorCode{
       error::ErrorDomain::Planning,
       category,
       static_cast<std::uint16_t>(detail),
   };
}

} // namespace fusion::planning

#endif // FUSION_CORE_PLANNING_PLANNING_ERRORS_H