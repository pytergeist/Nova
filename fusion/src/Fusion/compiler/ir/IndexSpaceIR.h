#ifndef FUSION_CORE_FUIR_INDEX_SPACE_IR_H
#define FUSION_CORE_FUIR_INDEX_SPACE_IR_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fusion::fuir {

using Label = std::size_t;
using OperandId = std::uint32_t;
using PhysicalAxisId = std::uint32_t;

enum class AxisAccess : std::uint8_t {
   Direct,
   Broadcast,
   Indexed,
};

struct PhysicalAxis {
   OperandId operand_id{0};
   PhysicalAxisId axis_id{0};
   std::size_t extent{1};
};

enum class IndexKind {
   Independent,
   Reduction,
};

using LogicalAxisId = std::uint32_t;

struct LogicalAxis {
   Label label{0};
   IndexKind kind{IndexKind::Independent};
};


struct AxisUse {
   PhysicalAxisId axis_id{0};
   LogicalAxisId logical_axis_id{0};
   AxisAccess access{AxisAccess::Direct};
};

struct OperandUse {
   OperandId operand_id{0};
   std::vector<AxisUse> axis_use;
};

enum class IndexRole {
   Batch,
   M,
   N,
   K,
};

struct IndexDef {
   Label label{0};
   std::size_t extent{1};
   IndexKind kind{IndexKind::Independent};
   std::vector<std::int32_t> axis_of_operand;
};

struct IndexSpaceIR {
   std::size_t num_operands{0};
   std::size_t itemsize{0};

   std::vector<IndexDef> indices;
   std::vector<std::uint32_t> out_indices;
};

struct OperandLabelBinding {
   std::vector<std::vector<Label>> op_axis_labels;
   std::vector<Label> out_labels;
};

} // namespace fusion::fuir

#endif