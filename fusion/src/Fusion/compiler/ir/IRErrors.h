#ifndef FUSION_CORE_FUIR_FUIR_ERRORS_H
#define FUSION_CORE_FUIR_FUIR_ERRORS_H

#include <cstdint>

#include "Fusion/common/error/ErrorCode.h"

namespace fusion::fuir {

enum class FuirError : std::uint8_t {
   EmptyOperands = 1,
   ItemSizeMismatch,
   OperandRankMismatch,
   InvalidAxis,
   BroadcastMismatch,

   InvalidBinding,
   BindingOperandCountMismatch,
   BindingAxisCountMismatch,
   RepeatedOperandLabelUnsupported,
   OutputLabelMissing,
   OutputLabelMismatch,

   InvalidIndexId,
   DescriptorCountMismatch,
   InvalidIR,

   // Additions for new IR model
   PhysicalAxisOperandMismatch, // used
   InvalidPhysicalAxisId, // used
   DuplicatePhysicalAxisId,
   InvalidPhysicalExtent, // used

   InvalidOperandUseId,
   InvalidPhysicalAxisReference,
   InvalidLogicalAxisId,
   DuplicatePhysicalAxisUse,

   DirectExtentMismatch,
   BroadcastExtentMismatch,
   IndexedAccessUnsupported,
   UnusedLogicalAxis,
};

constexpr error::ErrorCode
fuir_error(const FuirError detail,
           const error::ErrorCategory category) noexcept {
   return error::ErrorCode{
       .domain = error::ErrorDomain::Fuir,
       .category = category,
       .detail = static_cast<std::uint16_t>(detail),
   };
}

} // namespace fusion::fuir

#endif