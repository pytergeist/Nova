#ifndef BROADCAST_ITERATOR_H
#define BROADCAST_ITERATOR_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../fuir/IR.h"

/// The execution plan for a broadcast expression
struct BroadcastPlan {
   static constexpr std::string_view name = "Broadcast Plan";
   std::size_t num_operands;
   std::size_t out_ndim;
   std::vector<std::size_t> out_shape;
   std::vector<LoopDim> loop;
   std::vector<OperandAccess> op_access;

   bool all_contiguous_like{false};
   std::size_t vector_bytes{0};

   std::size_t itemsize;
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

struct ReductionPlan {
   static constexpr std::string_view name = "Reduction Plan";
   std::size_t num_operands;
   std::size_t out_ndim;
   std::vector<std::size_t> out_shape;
   std::size_t reduction_axis;
   std::vector<LoopDim> loop;
   std::vector<OperandAccess> op_access;

   bool keep_dim{false};
   bool all_contiguous_like{false}; // curr not used - evaluate
   std::size_t vector_bytes{0};

   std::size_t itemsize;
};

struct ContractionPlan {
   static constexpr std::string_view name = "Contraction Plan";
   std::size_t num_operands{0};
   std::size_t out_ndim{0};
   std::vector<std::size_t> out_shape;

   std::vector<LoopDim> loop;
   std::vector<OperandAccess> op_access;

   bool gemm_like{false};
   GemmLikeDesc gemm;

   std::size_t itemsize{0};
};

BroadcastPlan make_broadcast_plan(const std::vector<OperandDescription> &descs);

ReductionPlan make_reduction_plan(const std::vector<OperandDescription> &desc,
                                  const std::size_t axis, const bool keepdim);

ContractionPlan
make_contraction_plan_einsum(const std::vector<OperandDescription> &inputs,
                             const OperandLabelBinding &binding);

ContractionPlan
make_contraction_plan_einsum_out(const std::vector<OperandDescription> &descs,
                                 const OperandLabelBinding &binding);

#endif // BROADCAST_ITERATOR_H
