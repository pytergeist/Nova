#include "Fusion/core/planning/analysis/ContractionAnalysis.h"

#include <cstdint>
#include <optional>
#include <variant>

#include "Fusion/core/planning/AccessPlan.h"
#include "Fusion/core/planning/TraversalPlan.h"



namespace fusion::planning::analysis {
namespace {

struct GemmRoleExtents {
   std::size_t batch{1};
   std::size_t M{1};
   std::size_t N{1};
   std::size_t K{1};

   int m_count{0};
   int n_count{0};
   int k_count{0};
};

GemmRoleExtents
compute_gemm_role_extents(const DenseTraversalPlan& traversal) {
   GemmRoleExtents extents{};

   for (const LoopDim& ld : traversal.loop) {
      switch (ld.role) {
      case IndexRole::Batch:
         extents.batch *= ld.size;
         break;
      case IndexRole::M:
         extents.M = ld.size;
         ++extents.m_count;
         break;
      case IndexRole::N:
         extents.N = ld.size;
         ++extents.n_count;
         break;
      case IndexRole::K:
         extents.K = ld.size;
         ++extents.k_count;
         break;
      }
   }

   return extents;
}

bool has_single_mnk(const GemmRoleExtents& extents) {
   return extents.m_count == 1 &&
          extents.n_count == 1 &&
          extents.k_count == 1;
}

bool has_three_affine_operands(const ExecutionPlan& exec) {
   if (exec.access.operands.size() != 3) {
      return false;
   }

   for (const OperandAccess& operand : exec.access.operands) {
      if (operand.access != AccessKind::Affine) {
         return false;
      }
   }

   return true;
}


bool has_valid_gemm_strides(const GemmLikeDesc& gemm) {
   return gemm.out_rs != 0 &&
          gemm.out_cs != 0 &&
          gemm.a_rs != 0 &&
          gemm.a_cs != 0 &&
          gemm.b_rs != 0 &&
          gemm.b_cs != 0;
}


bool access_rank_matches_loop_rank(const ExecutionPlan& exec,
                                   const DenseTraversalPlan& traversal) {
   const std::size_t rank = traversal.loop.size();

   for (const OperandAccess& operand : exec.access.operands) {
      if (operand.affine.byte_stride_per_loop.size() != rank) {
         return false;
      }
   }

   return true;
}

std::optional<GemmLikeDesc>
extract_gemm_desc_from_dense_plan(const ExecutionPlan& exec,
                                  const DenseTraversalPlan& traversal) {
   if (!has_three_affine_operands(exec)) {
      return std::nullopt;
   }

   if (!access_rank_matches_loop_rank(exec, traversal)) {
      return std::nullopt;
   }

   const GemmRoleExtents extents = compute_gemm_role_extents(traversal);

   if (!has_single_mnk(extents)) {
      return std::nullopt;
   }

   const std::int64_t item =
       static_cast<std::int64_t>(exec.core.itemsize);

   if (item <= 0) {
      return std::nullopt;
   }

   const std::vector<std::int64_t>& out_access =
       exec.access.operands.at(0).affine.byte_stride_per_loop;
   const std::vector<std::int64_t>& a_access =
       exec.access.operands.at(1).affine.byte_stride_per_loop;
   const std::vector<std::int64_t>& b_access =
       exec.access.operands.at(2).affine.byte_stride_per_loop;

   GemmLikeDesc gemm{};
   gemm.batch = extents.batch;
   gemm.M = extents.M;
   gemm.N = extents.N;
   gemm.K = extents.K;

   for (std::size_t pos = 0; pos < traversal.loop.size(); ++pos) {
      const LoopDim& ld = traversal.loop[pos];

      if (ld.role == IndexRole::M) {
         gemm.out_rs = out_access[pos] / item;
         gemm.a_rs = a_access[pos] / item;
      } else if (ld.role == IndexRole::N) {
         gemm.out_cs = out_access[pos] / item;
         gemm.b_cs = b_access[pos] / item;
      } else if (ld.role == IndexRole::K) {
         gemm.a_cs = a_access[pos] / item;
         gemm.b_rs = b_access[pos] / item;
      }
   }

   if (!has_valid_gemm_strides(gemm)) {
      return std::nullopt;
   }

   return gemm;
}

} // namespace

std::optional<GemmLikeDesc>
analyse_gemm_like_contraction(const ExecutionPlan& exec) {
   if (exec.core.expr != ExprKind::Contraction) {
      return std::nullopt;
   }

   if (exec.core.traversal_kind != TraversalKind::Dense) {
      return std::nullopt;
   }

   const DenseTraversalPlan* dense =
       std::get_if<DenseTraversalPlan>(&exec.traversal);

   if (dense == nullptr) {
      return std::nullopt;
   }

   return extract_gemm_desc_from_dense_plan(exec, *dense);
}

} // namespace fusion::planning::analysis