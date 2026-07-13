#ifndef FUSION_CORE_FUIR_OPERAND_DESCRIPTION_H
#define FUSION_CORE_FUIR_OPERAND_DESCRIPTION_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fusion::fuir {

enum class LayoutKind : std::int8_t { Dense, Strided, SoA, AoSoA };
enum class AccessKind : std::int8_t { Affine, Indexed, Segmented, Blocked };
enum class StorageKind : std::int8_t { Owned, View };
enum class UpdateKind : std::int8_t { ReadOnly, Overwrite, Accumulate, ScatterAdd };

enum class OperandDescType : std::int8_t { Tensor, Topology, Index };

struct OperandDescription {
   std::vector<std::size_t> shape;
   std::vector<std::int64_t> strides;
   std::size_t itemsize{0};

   LayoutKind layout{LayoutKind::Dense};
   AccessKind access{AccessKind::Affine};
   StorageKind storage{StorageKind::Owned};
   UpdateKind update{UpdateKind::ReadOnly};

   OperandDescType type{OperandDescType::Tensor};

   [[nodiscard]] std::size_t ndims() const noexcept {
      return shape.size();
   }
};

} // namespace fusion::fuir

#endif // FUSION_CORE_FUIR_OPERAND_DESCRIPTION_H