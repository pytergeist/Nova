#ifndef FUSION_CORE_FUIR_SHAPE_RULES_H
#define FUSION_CORE_FUIR_SHAPE_RULES_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "Fusion/compiler/ir/IndexSpaceIR.h"
#include "Fusion/compiler/ir/OperandDescription.h"

namespace fusion::fuir {

std::size_t norm_axis(std::int64_t ax, std::size_t ndims);

std::size_t broadcast_dim(std::size_t a, std::size_t b);

std::vector<std::size_t> infer_out_shape_from_ir(const IndexSpaceIR &ir);

std::vector<std::size_t> infer_binary_contraction_out_shape_from_binding(
    const std::vector<OperandDescription> &inputs,
    const OperandLabelBinding &binding);

std::vector<std::size_t>
infer_out_shape_from_binding(const std::vector<OperandDescription> &descs,
                             const OperandLabelBinding &binding);

} // namespace fusion::fuir

#endif