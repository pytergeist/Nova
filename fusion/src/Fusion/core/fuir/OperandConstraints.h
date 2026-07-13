#ifndef FUSION_CORE_FUIR_OPERAND_CONSTRAINTS_H
#define FUSION_CORE_FUIR_OPERAND_CONSTRAINTS_H

#include <cstdint>

namespace fusion::fuir {
enum class OperandGroupConstraint: std::int8_t {
   HomogeneousItemSize,
   TopologyAllowed,
};
} // namespace fusion::fuir

#endif // FUSION_CORE_FUIR_OPERAND_CONSTRAINTS_H