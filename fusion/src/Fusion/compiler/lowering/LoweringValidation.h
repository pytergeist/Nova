#ifndef FUSION_COMPILER_LOWERING_LOWERING_VALIDATION_H
#define FUSION_COMPILER_LOWERING_LOWERING_VALIDATION_H

#include <string_view>
#include <vector>

#include "Fusion/compiler/ir/IndexSpaceIR.h"
#include "Fusion/compiler/ir/OperandDescription.h"

namespace fusion::lowering::lvalidation {

void validate_loop_order(
    const fuir::IndexSpaceIR &ir,
    const std::vector<fuir::LogicalAxisId> &loop_order,
    std::string_view where = "validate_loop_order");

void validate_desc_count_matches_ir(
    const fuir::IndexSpaceIR &ir,
    const std::vector<fuir::OperandDescription> &descs,
    std::string_view where = "validate_desc_count_matches_ir");

void validate_ir_matches_descs(
    const fuir::IndexSpaceIR &ir,
    const std::vector<fuir::OperandDescription> &descs,
    std::string_view where = "validate_ir_matches_descs");

void validate_role_vector_matches_ir(
    const fuir::IndexSpaceIR &ir,
    const std::vector<fuir::IndexRole> *role_of_id,
    std::string_view where = "validate_role_vector_matches_ir");

} // namespace fusion::lowering::lvalidation

#endif