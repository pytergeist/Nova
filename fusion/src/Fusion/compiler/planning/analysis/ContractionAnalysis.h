#ifndef FUSION_CORE_PLANNING_ANALYSIS_CONTRACTION_ANALYSIS_H
#define FUSION_CORE_PLANNING_ANALYSIS_CONTRACTION_ANALYSIS_H

#include <optional>

#include "Fusion/compiler/planning/ExecutionPlan.h"
#include "Fusion/compiler/planning/KernelHints.h"

namespace fusion::planning::analysis {

std::optional<GemmLikeDesc>
analyse_gemm_like_contraction(const ExecutionPlan &exec);

std::optional<std::vector<fuir::IndexRole>>
compute_roles_for_gemm_like(const fuir::IndexSpaceIR &ir);

} // namespace fusion::planning::analysis

#endif // FUSION_CORE_PLANNING_ANALYSIS_CONTRACTION_ANALYSIS_H