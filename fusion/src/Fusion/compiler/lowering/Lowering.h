#ifndef FUSION_CORE_FUIR_LOWERING_H
#define FUSION_CORE_FUIR_LOWERING_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "Fusion/compiler/ir/IndexSpaceIR.h"
#include "Fusion/compiler/ir/OperandDescription.h"
#include "Fusion/core/Layout.h"

namespace fusion::lowering {

struct AffineAccess {
   std::vector<std::int64_t> byte_stride_per_loop;
};

struct BlockedAccess {};
struct IndexedAccess {};

struct LoopDim {
   std::size_t size{};
   fuir::IndexKind kind{fuir::IndexKind::Independent};
   fuir::IndexRole role{fuir::IndexRole::Batch};
};

struct OperandAccess {
   std::size_t operand_id{0};

   core::LayoutKind layout{core::LayoutKind::Dense};
   fuir::StorageKind storage{fuir::StorageKind::Owned};
   fuir::UpdateKind update{fuir::UpdateKind::ReadOnly};

   fuir::AccessKind access{fuir::AccessKind::Affine};

   AffineAccess affine{};
   BlockedAccess blocked{};
   IndexedAccess indexed{};
};

std::vector<LoopDim>
lower_to_loops(const fuir::IndexSpaceIR &ir,
               const std::vector<fuir::OperandDescription> &descs,
               const std::vector<fuir::LogicalAxisId> &loop_order);

std::vector<LoopDim>
lower_to_loops(const fuir::IndexSpaceIR &ir,
               const std::vector<fuir::OperandDescription> &descs,
               const std::vector<fuir::LogicalAxisId> &loop_order,
               const std::vector<fuir::IndexRole> *role_of_id);

std::vector<OperandAccess>
lower_operand_access(const fuir::IndexSpaceIR &ir,
                     const std::vector<fuir::OperandDescription> &descs,
                     const std::vector<fuir::LogicalAxisId> &loop_order);

} // namespace fusion::fuir

#endif