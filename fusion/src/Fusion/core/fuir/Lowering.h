#ifndef FUSION_CORE_FUIR_LOWERING_H
#define FUSION_CORE_FUIR_LOWERING_H

#include <vector>
#include <cstddef>
#include <cstdint>

#include "Fusion/core/fuir/IndexSpaceIR.h"
#include "Fusion/core/fuir/OperandDescription.h"

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

   LayoutKind layout{LayoutKind::Dense};
   StorageKind storage{StorageKind::Owned};
   UpdateKind update{UpdateKind::ReadOnly};

   AccessKind access{AccessKind::Affine};

   AffineAccess affine{};
   BlockedAccess blocked{};
   IndexedAccess indexed{};
};

std::int64_t stride_bytes_for_binding(
    const OperandDescription& desc,
    std::int32_t axis,
    std::size_t index_extent,
    std::size_t itemsize);

std::vector<LoopDim> lower_to_loops(
    const IndexSpaceIR& ir,
    const std::vector<OperandDescription>& descs,
    const std::vector<std::uint32_t>& loop_order);

std::vector<LoopDim> lower_to_loops(
    const IndexSpaceIR& ir,
    const std::vector<OperandDescription>& descs,
    const std::vector<std::uint32_t>& loop_order,
    const std::vector<IndexRole>* role_of_id);

std::vector<OperandAccess> lower_operand_access(
    const IndexSpaceIR& ir,
    const std::vector<OperandDescription>& descs,
    const std::vector<std::uint32_t>& loop_order);

std::vector<IndexRole>
compute_roles_for_gemm_like(const IndexSpaceIR& ir,
                            const OperandLabelBinding& binding);

} // namespace fusion::fuir

#endif