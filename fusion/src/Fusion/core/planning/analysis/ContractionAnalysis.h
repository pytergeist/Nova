#ifndef FUSION_CORE_PLANNING_ANALYSIS_CONTRACTION_ANALYSIS_H
#define FUSION_CORE_PLANNING_ANALYSIS_CONTRACTION_ANALYSIS_H

#include <optional>

#include "Fusion/core/planning/ExecutionPlan.h"
#include "Fusion/core/planning/KernelHints.h"

namespace fusion::planning::analysis {

std::optional<GemmLikeDesc>
analyse_gemm_like_contraction(const ExecutionPlan& exec);

} // namespace fusion::planning::analysis

#endif // FUSION_CORE_PLANNING_ANALYSIS_CONTRACTION_ANALYSIS_H