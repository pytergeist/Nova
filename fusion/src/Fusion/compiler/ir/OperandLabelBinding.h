#ifndef FUSION_COMPILER_IR_OPERAND_LABEL_BINDING_H
#define FUSION_COMPILER_IR_OPERAND_LABEL_BINDING_H

#include <vector>

#include "Fusion/compiler/ir/IndexSpaceIR.h"

namespace fusion::fuir::binding {
struct OperandLabelBinding {
   std::vector<std::vector<Label>> op_axis_labels;
   std::vector<Label> out_labels;
};
}


#endif // FUSION_COMPILER_IR_OPERAND_LABEL_BINDING_H