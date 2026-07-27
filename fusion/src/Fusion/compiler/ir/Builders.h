#ifndef FUSION_CORE_FUIR_IR_BUILDERS_H
#define FUSION_CORE_FUIR_IR_BUILDERS_H

#include <vector>

#include "Fusion/compiler/ir/IndexSpaceIR.h"
#include "Fusion/compiler/ir/OperandConstraints.h"
#include "Fusion/compiler/ir/OperandDescription.h"

namespace fusion::fuir {

IndexSpaceIR
build_elementwise_ir_right_aligned(const std::vector<OperandDescription> &descs,
                                   OperandGroupConstraint constraint);

IndexSpaceIR build_reduction_ir(const std::vector<OperandDescription> &descs,
                                std::size_t axis, bool keepdim,
                                OperandGroupConstraint constraint);

IndexSpaceIR
build_ir_from_label_binding(const std::vector<OperandDescription> &descs,
                            const OperandLabelBinding &bind,
                            OperandGroupConstraint constraint);

} // namespace fusion::fuir

#endif