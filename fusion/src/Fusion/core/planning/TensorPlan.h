#ifndef FUSION_CORE_PLANNING_TENSOR_PLAN_H
#define FUSION_CORE_PLANNING_TENSOR_PLAN_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "Fusion/core/fuir/IR.h"
#include "Fusion/core/topology/TopologyView.h"

enum class ExprKind : std::uint8_t {
   Elementwise,
   Reduction,
   Contraction,
   Pairwise,
   IndexedMap,
};

enum class TraversalKind : std::uint8_t {
   Dense,
   Indexed,
};

enum class IndexedFormat : std::uint8_t {
   EdgeList,
   CRS,
   BlockedCRS,
};

struct GemmLikeDesc {
   std::size_t batch{1};
   std::size_t M{1}, N{1}, K{1};

   std::int64_t out_rs{0}, out_cs{0};
   std::int64_t a_rs{0}, a_cs{0};
   std::int64_t b_rs{0}, b_cs{0};

   bool a_transpose{false};
   bool b_transpose{false};
   bool out_is_contig_mn{false};
   bool a_is_contig_mk{false};
   bool b_is_contig_kn{false};
};

struct KernelHints {
   bool all_contiguous_like{false};
   std::size_t vector_bytes{0};

   bool gemm_like{false};
   GemmLikeDesc gemm{};
};

struct DenseTraversalPlan {
   std::vector<LoopDim> loop;
};

struct IndexedTraversalPlan {
   IndexedFormat format{IndexedFormat::BlockedCRS};
   BlockedCRS blocked_crs;
};

using TraversalPlan = std::variant<DenseTraversalPlan, IndexedTraversalPlan>;

struct AccessPlan {
   std::vector<OperandAccess> operands;
};

struct PlanCore {
   static constexpr std::string_view name = "Plan Core";
   ExprKind expr;
   TraversalKind traversal;

   std::size_t itemsize{0};
   std::size_t num_operands{0};
   std::size_t out_ndim{0};
   std::vector<std::size_t> out_shape;
};

struct ExecutionPlan {
   PlanCore core;
   TraversalPlan traversal;
   AccessPlan access;
   KernelHints hints;
};

/// The execution plan for an ElementWise expression
struct ElementWisePlan {
   static constexpr std::string_view name = "Elementwise Plan";
   ExecutionPlan exec;
};

struct ReductionPlan {
   static constexpr std::string_view name = "Reduction Plan";
   ExecutionPlan exec;

   std::size_t reduction_axis{0};
   bool keep_dim{false};
};

struct ContractionPlan {
   static constexpr std::string_view name = "Contraction Plan";
   ExecutionPlan exec;
};

struct IndexedPlan {
   static constexpr std::string_view name = "Indexed Plan";
   ExecutionPlan exec;
};

ElementWisePlan
make_elementwise_plan(const std::vector<OperandDescription> &descs);

ReductionPlan make_reduction_plan(const std::vector<OperandDescription> &desc,
                                  const std::size_t axis, const bool keepdim);

ContractionPlan
make_contraction_plan_einsum(const std::vector<OperandDescription> &inputs,
                             const OperandLabelBinding &binding);

ContractionPlan
make_contraction_plan_einsum_out(const std::vector<OperandDescription> &descs,
                                 const OperandLabelBinding &binding);

IndexedPlan make_indexed_plan(const std::vector<OperandDescription> &descs,
                              const BlockedCRS &bcrs, const EdgeList &edges);

KernelHints make_kernel_hints(const std::vector<OperandDescription> &descs);

#endif // FUSION_CORE_PLANNING_TENSOR_PLAN_H
