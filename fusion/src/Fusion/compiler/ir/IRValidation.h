#ifndef FUSION_CORE_FUIR_FUIR_VALIDATION_H
#define FUSION_CORE_FUIR_FUIR_VALIDATION_H

#include <cstdint>
#include <string_view>
#include <vector>

#include "Fusion/compiler/ir/IndexSpaceIR.h"
#include "Fusion/compiler/ir/OperandConstraints.h"
#include "Fusion/compiler/ir/OperandDescription.h"

namespace fusion::fuir::validation {

void validate_descs_itemsize_group(
    const std::vector<OperandDescription> &descs,
    OperandGroupConstraint constraint,
    std::string_view where = "validate_descs_itemsize_group");

void validate_operand_label_binding(
    const std::vector<OperandDescription> &descs,
    const OperandLabelBinding &binding,
    std::string_view where = "validate_operand_label_binding");

void validate_reduction_request(
    const std::vector<OperandDescription> &descs, std::size_t axis,
    bool keepdim, std::string_view where = "validate_reduction_request");

void validate_index_space_ir(
    const IndexSpaceIR &ir, std::string_view where = "validate_index_space_ir");

void validate_loop_order(const IndexSpaceIR &ir,
                         const std::vector<std::uint32_t> &loop_order,
                         std::string_view where = "validate_loop_order");

void validate_desc_count_matches_ir(
    const IndexSpaceIR &ir, const std::vector<OperandDescription> &descs,
    std::string_view where = "validate_desc_count_matches_ir");

void validate_ir_matches_descs(
    const IndexSpaceIR &ir, const std::vector<OperandDescription> &descs,
    std::string_view where = "validate_ir_matches_descs");

void validate_role_vector_matches_ir(
    const IndexSpaceIR &ir, const std::vector<IndexRole> *role_of_id,
    std::string_view where = "validate_role_vector_matches_ir");

} // namespace fusion::fuir::validation

#endif // FUSION_CORE_FUIR_FUIR_VALIDATION_H