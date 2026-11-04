#ifndef FUSION_PHYSICS_CORE_PAIRWISE_IR_H
#define FUSION_PHYSICS_CORE_PAIRWISE_IR_H

#include "Fusion/core/fuir/IR.h"

OperandLabelBinding make_gather_index_label_binding(std::size_t inp_nd,
                                                    std::size_t top_nd);

#endif // FUSION_PHYSICS_CORE_PAIRWISE_IR_H
