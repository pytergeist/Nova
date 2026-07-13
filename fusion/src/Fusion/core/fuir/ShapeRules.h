#ifndef FUSION_CORE_FUIR_SHAPE_RULES_H
#define FUSION_CORE_FUIR_SHAPE_RULES_H

#include <cstddef>
#include <vector>
#include <cstdint>

#include "Fusion/core/fuir/IndexSpaceIR.h"
#include "Fusion/core/fuir/OperandDescription.h"

namespace fusion::fuir {

std::size_t norm_axis(std::int64_t ax, std::size_t ndims);

std::size_t broadcast_dim(std::size_t a, std::size_t b);

std::vector<std::size_t> out_shape_from_ir(const IndexSpaceIR& ir);

std::vector<std::size_t>
infer_out_shape_from_binding(const std::vector<OperandDescription>& inputs,
                             const OperandLabelBinding& binding);

} // namespace fusion::fuir

#endif