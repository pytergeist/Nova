#ifndef FUSION_PHYSICS_CORE_PAIRWISE_IR_H
#define FUSION_PHYSICS_CORE_PAIRWISE_IR_H

#include "Fusion/core/fuir/IR.h"

#include "PhysicsDescs.h"

IndexSpaceIR build_gather_and_map_ir(const std::vector<OperandDescription> &descs);

#endif // FUSION_PHYSICS_CORE_PAIRWISE_IR_H