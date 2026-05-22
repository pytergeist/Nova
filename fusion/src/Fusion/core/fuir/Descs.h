#ifndef FUSION_CORE_FUIR_DESCS_H
#define FUSION_CORE_FUIR_DESCS_H

#include <cstdint>
#include <vector>

enum class LayoutKind { Dense, Strided, SoA, AoSoA };
enum class AccessKind { Affine, Indexed, Segmented, Blocked };
enum class StorageKind { Owned, View };
enum class UpdateKind { ReadOnly, Overwrite, Accumulate, ScatterAdd };

enum class OperandDescType { Tensor, Topology, Index };

/// OperandDescription stores the meta-data of a single operand participating in
/// an expression.
struct OperandDescription {
   /// shape of operand tensor
   std::vector<std::size_t> shape;
   /// strides of operand tensor (if applicable)
   std::vector<std::int64_t> strides;
   /// size in bytes of operand dtype
   std::size_t itemsize;
   /// Operand runtime layout
   LayoutKind layout{LayoutKind::Dense};
   /// Operand runtime accessor pattern
   AccessKind access{AccessKind::Affine};
   StorageKind storage{StorageKind::Owned};
   UpdateKind update{UpdateKind::ReadOnly};

   /// Operand semantic type
   OperandDescType type{OperandDescType::Tensor};

   std::size_t ndims() const noexcept { return shape.size(); }
};

#endif // FUSION_CORE_FUIR_DESCS_H