#ifndef FUSION_CORE_PLANNING_OP_CONTEXT_H
#define FUSION_CORE_PLANNING_OP_CONTEXT_H

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "Fusion/core/fuir/IR.h"
#include "Fusion/core/planning/OpPlans.h"

namespace fusion::planning {

enum class BinaryExecKind : std::uint8_t {
   GenericStrided,
   FlatContiguous,
   FlatContiguousBroadcastLHS,
   FlatContiguousBroadcastRHS,
};

constexpr std::size_t kGlobalReduceAxis =
    static_cast<std::size_t>(-1);

struct BinaryEwiseContext {
   std::vector<std::size_t> out_shape{};
   std::size_t fast_len{0};

   ElementwisePlan plan{};

   OperandDescription lhs{};
   OperandDescription rhs{};
   OperandDescription out{};

   BinaryExecKind exec{BinaryExecKind::GenericStrided};
};

struct UnaryEwiseContext {
   bool fastpath{false};
   std::size_t fast_len{0};

   std::vector<std::size_t> out_shape{};

   ElementwisePlan plan{};

   OperandDescription input{};
   OperandDescription out{};
};

struct ReductionContext {
   bool fastpath{false};
   std::size_t fast_len{0};

   std::vector<std::size_t> out_shape{};

   ReductionPlan plan{};

   // TODO: Eventually remove these duplicates and use plan.reduction_axis / plan.keep_dim.
   bool keepdim{false};
   std::size_t reduction_axis{0};

   std::size_t reduce_len{0};

   OperandDescription input{};
   OperandDescription out{};
};

struct ContractionContext {
   bool fastpath{false};
   std::size_t fast_len{0};

   std::vector<std::size_t> out_shape{};

   ContractionPlan plan{};

   OperandDescription lhs{};
   OperandDescription rhs{};
   OperandDescription out{};

   OperandLabelBinding binding{};
};


inline std::string_view to_string(BinaryExecKind k) noexcept {
   switch (k) {
   case BinaryExecKind::GenericStrided:
      return "GenericStrided";
   case BinaryExecKind::FlatContiguous:
      return "FlatContiguous";
   case BinaryExecKind::FlatContiguousBroadcastLHS:
      return "FlatContiguousBroadcastLHS";
   case BinaryExecKind::FlatContiguousBroadcastRHS:
      return "FlatContiguousBroadcastRHS";
   }
   return "Unknown";
}

} // namespace fusion::planning

#endif // FUSION_CORE_PLANNING_OP_CONTEXT_H