#include "Fusion/compiler/planning/analysis/ContractionAnalysis.h"

#include <cstdint>
#include <optional>
#include <variant>
#include <unordered_set>

#include "Fusion/compiler/planning/AccessPlan.h"
#include "Fusion/compiler/planning/TraversalPlan.h"

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

GemmRoleExtents compute_gemm_role_extents(const DenseTraversalPlan &traversal) {
   GemmRoleExtents extents{};

   for (const fuir::LoopDim &ld : traversal.loop) {
      switch (ld.role) {
      case fuir::IndexRole::Batch:
         extents.batch *= ld.size;
         break;
      case fuir::IndexRole::M:
         extents.M = ld.size;
         ++extents.m_count;
         break;
      case fuir::IndexRole::N:
         extents.N = ld.size;
         ++extents.n_count;
         break;
      case fuir::IndexRole::K:
         extents.K = ld.size;
         ++extents.k_count;
         break;
      }
   }

   return extents;
}

bool has_single_mnk(const GemmRoleExtents &extents) {
   return extents.m_count == 1 && extents.n_count == 1 && extents.k_count == 1;
}

bool has_three_affine_operands(const ExecutionPlan &exec) {
   if (exec.access.operands.size() != 3) {
      return false;
   }

   for (const fuir::OperandAccess &operand : exec.access.operands) {
      if (operand.access != fuir::AccessKind::Affine) {
         return false;
      }
   }

   return true;
}

bool has_valid_gemm_strides(const GemmLikeDesc &gemm) {
   return gemm.out_rs != 0 && gemm.out_cs != 0 && gemm.a_rs != 0 &&
          gemm.a_cs != 0 && gemm.b_rs != 0 && gemm.b_cs != 0;
}

bool access_rank_matches_loop_rank(const ExecutionPlan &exec,
                                   const DenseTraversalPlan &traversal) {
   const std::size_t rank = traversal.loop.size();

   for (const fuir::OperandAccess &operand : exec.access.operands) {
      if (operand.affine.byte_stride_per_loop.size() != rank) {
         return false;
      }
   }

   return true;
}

std::optional<GemmLikeDesc>
extract_gemm_desc_from_dense_plan(const ExecutionPlan &exec,
                                  const DenseTraversalPlan &traversal) {
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

   const std::int64_t item = static_cast<std::int64_t>(exec.core.itemsize);

   if (item <= 0) {
      return std::nullopt;
   }

   const std::vector<std::int64_t> &out_access =
       exec.access.operands.at(0).affine.byte_stride_per_loop;
   const std::vector<std::int64_t> &a_access =
       exec.access.operands.at(1).affine.byte_stride_per_loop;
   const std::vector<std::int64_t> &b_access =
       exec.access.operands.at(2).affine.byte_stride_per_loop;

   GemmLikeDesc gemm{};
   gemm.batch = extents.batch;
   gemm.M = extents.M;
   gemm.N = extents.N;
   gemm.K = extents.K;

   for (std::size_t pos = 0; pos < traversal.loop.size(); ++pos) {
      const fuir::LoopDim &ld = traversal.loop[pos];

      if (ld.role == fuir::IndexRole::M) {
         gemm.out_rs = out_access[pos] / item;
         gemm.a_rs = a_access[pos] / item;
      } else if (ld.role == fuir::IndexRole::N) {
         gemm.out_cs = out_access[pos] / item;
         gemm.b_cs = b_access[pos] / item;
      } else if (ld.role == fuir::IndexRole::K) {
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
analyse_gemm_like_contraction(const ExecutionPlan &exec) {
   if (exec.core.expr != ExprKind::Contraction) {
      return std::nullopt;
   }

   if (exec.core.traversal_kind != TraversalKind::Dense) {
      return std::nullopt;
   }

   const DenseTraversalPlan *dense =
       std::get_if<DenseTraversalPlan>(&exec.traversal);

   if (dense == nullptr) {
      return std::nullopt;
   }

   return extract_gemm_desc_from_dense_plan(exec, *dense);
}


std::optional<std::vector<fuir::IndexRole>>
compute_roles_for_gemm_like(const fuir::IndexSpaceIR &ir) {

   // TODO: this logic only works for binary contraction
   if (ir.num_operands != 3) {
      return std::nullopt;
   }

   const std::vector<fuir::AxisUse> &out_axis_use = ir.operand_use.at(0).axis_use;
   const std::vector<fuir::AxisUse> &a_axis_use = ir.operand_use.at(1).axis_use;
   const std::vector<fuir::AxisUse> &b_axis_use = ir.operand_use.at(2).axis_use;

   auto as_set = [](const std::vector<fuir::AxisUse> &axis_use) {
      std::unordered_set<fuir::LogicalAxisId> result;
      result.reserve(axis_use.size());

      for (const fuir::AxisUse use : axis_use) {
         result.insert(use.logical_axis_id);
      }

      return result;
   };

   const std::unordered_set<fuir::LogicalAxisId> out_index_set = as_set(out_axis_use);
   const std::unordered_set<fuir::LogicalAxisId> a_index_set = as_set(a_axis_use);
   const std::unordered_set<fuir::LogicalAxisId> b_index_set = as_set(b_axis_use);

   std::unordered_map<fuir::LogicalAxisId, fuir::IndexRole> role_of_label;
   role_of_label.reserve(64);

   std::unordered_set<fuir::LogicalAxisId> all;
   all.reserve(out_index_set.size() + a_index_set.size() + b_index_set.size());

   for (const fuir::LogicalAxisId logical_axis_id : out_index_set) {
      all.insert(logical_axis_id);
   }

   for (const fuir::LogicalAxisId logical_axis_id : a_index_set) {
      all.insert(logical_axis_id);
   }

   for (const fuir::LogicalAxisId logical_axis_id : b_index_set) {
      all.insert(logical_axis_id);
   }

   std::vector<fuir::IndexRole> role_of_id(ir.logical_axes.size(), fuir::IndexRole::Batch);

   for (const fuir::LogicalAxisId logical_axis_id : all) {
      const bool in_out = out_index_set.contains(logical_axis_id);
      const bool in_a = a_index_set.contains(logical_axis_id);
      const bool in_b = b_index_set.contains(logical_axis_id);

      fuir::IndexRole role = fuir::IndexRole::Batch;

      if (in_out && in_a && in_b) {
         role = fuir::IndexRole::Batch;
      } else if (in_out && in_a && !in_b) {
         role = fuir::IndexRole::M;
      } else if (in_out && !in_a && in_b) {
         role = fuir::IndexRole::N;
      } else if (!in_out && in_a && in_b) {
         role = fuir::IndexRole::K;
      } else {
         return std::nullopt;
      }

      const fuir::IndexKind expected_kind =
       role == fuir::IndexRole::K
           ? fuir::IndexKind::Reduction
           : fuir::IndexKind::Independent;

      if (ir.logical_axes[logical_axis_id].kind != expected_kind) {
         return std::nullopt;
      }
      role_of_id.at(logical_axis_id) = role;
   }

   return role_of_id;
}

} // namespace fusion::planning::analysis