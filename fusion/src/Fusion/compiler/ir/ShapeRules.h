#ifndef FUSION_CORE_FUIR_SHAPE_RULES_H
#define FUSION_CORE_FUIR_SHAPE_RULES_H

#include <cstddef>
#include <unordered_map>
#include <vector>

#include "Fusion/compiler/ir/IndexSpaceIR.h"
#include "Fusion/compiler/ir/OperandDescription.h"

namespace fusion::fuir::shape {

using LabelExtentMap = std::unordered_map<Label, std::size_t>;

std::size_t norm_axis(std::int64_t ax, std::size_t ndims);

std::size_t broadcast_dim(std::size_t a, std::size_t b);

LabelExtentMap
resolve_contraction_label_extents(const std::vector<OperandDescription> &inputs,
                                  const OperandLabelBinding &binding);

std::vector<std::size_t> out_shape_from_ir(const IndexSpaceIR &ir);

std::vector<std::size_t>
infer_elementwise_out_shape(const std::vector<OperandDescription> &inputs);

std::vector<std::size_t> infer_binary_contraction_out_shape_from_binding(
    const std::vector<OperandDescription> &inputs,
    const OperandLabelBinding &binding);

} // namespace fusion::fuir::shape

#endif