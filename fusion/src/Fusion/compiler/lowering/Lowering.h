#ifndef FUSION_CORE_FUIR_LOWERING_H
#define FUSION_CORE_FUIR_LOWERING_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "Fusion/compiler/ir/IndexSpaceIR.h"
#include "Fusion/compiler/ir/OperandDescription.h"
#include "Fusion/core/Layout.h"

namespace fusion::fuir {

struct AffineAccess {
   std::vector<std::int64_t> byte_stride_per_loop;
};

struct BlockedAccess {};
struct IndexedAccess {};

struct LoopDim {
   std::size_t size{};
   IndexKind kind{IndexKind::Independent};
   IndexRole role{IndexRole::Batch};
};

struct OperandAccess {
   std::size_t operand_id{0};

   core::LayoutKind layout{core::LayoutKind::Dense};
   StorageKind storage{StorageKind::Owned};
   UpdateKind update{UpdateKind::ReadOnly};

   AccessKind access{AccessKind::Affine};

   AffineAccess affine{};
   BlockedAccess blocked{};
   IndexedAccess indexed{};
};

std::vector<LoopDim>
lower_to_loops(const IndexSpaceIR &ir,
               const std::vector<OperandDescription> &descs,
               const std::vector<LogicalAxisId> &loop_order);

std::vector<LoopDim>
lower_to_loops(const IndexSpaceIR &ir,
               const std::vector<OperandDescription> &descs,
               const std::vector<LogicalAxisId> &loop_order,
               const std::vector<IndexRole> *role_of_id);

std::vector<OperandAccess>
lower_operand_access(const IndexSpaceIR &ir,
                     const std::vector<OperandDescription> &descs,
                     const std::vector<LogicalAxisId> &loop_order);

std::vector<IndexRole>
compute_roles_for_gemm_like(const IndexSpaceIR &ir,
                            const OperandLabelBinding &binding);

} // namespace fusion::fuir

#endif