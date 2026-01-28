#ifndef BROADCAST_ITERATOR_H
#define BROADCAST_ITERATOR_H

#include <cstddef>
#include <cstdint>
#include <vector>

using Label = std::uint32_t;

enum class IndexKind { Independent, Reduction };

struct IndexDef {
   std::size_t extent{1};
   IndexKind kind{IndexKind::Independent};
   std::vector<std::int32_t> axis_of_operand;
};

struct IndexSpaceIR {
   std::size_t num_operands{0};
   std::size_t itemsize{0};

   std::vector<IndexDef> indices;
   std::vector<std::uint32_t> out_indices;
};

struct EinsumBinding {
   std::vector<std::vector<Label>> op_axis_labels;
   std::vector<Label> out_labels;
};

struct TensorDescription {
   std::size_t ndims;
   std::vector<std::size_t> shape;
   std::vector<std::int64_t> strides;
   std::size_t itemsize;
};

enum class LoopKind { Independent, Reduction };
enum class LoopRole { Batch, M, N, K };

struct LoopDim {
   std::size_t size;
   std::vector<std::int64_t> stride_bytes;
   LoopKind kind;
   LoopRole role;
};

struct AxisRef3 {
   int out = -1;
   int lhs = -1;
   int rhs = -1;
   LoopKind kind = LoopKind::Independent;
};

struct BroadcastView {
   std::size_t out_ndim = 0;
   std::vector<std::size_t> out_shape;
   std::vector<std::vector<int>> axis_map;
   std::vector<std::vector<std::int64_t>> stride_bytes;
};

struct BroadcastPlan {
   std::size_t num_operands;
   std::size_t out_ndim;
   std::vector<std::size_t> out_shape;
   std::vector<LoopDim> loop;

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
   std::size_t num_operands;
   std::size_t out_ndim;
   std::vector<std::size_t> out_shape;
   std::size_t reduction_axis;
   std::vector<LoopDim> loop;

   bool keep_dim{false};
   bool all_contiguous_like{false}; // curr not used - evaluate
   std::size_t vector_bytes{0};

   std::size_t itemsize;
};

struct ContractionAxes {
   std::int64_t lhs_operand;
   std::int64_t rhs_operand;
};

struct ContractionPlan {
   std::size_t num_operands{0};
   std::size_t out_ndim{0};
   std::vector<std::size_t> out_shape;

   std::vector<LoopDim> outer;
   std::vector<LoopDim> reduce;

   bool gemm_like{false};
   GemmLikeDesc gemm;

   std::size_t itemsize{0};
};

BroadcastPlan make_broadcast_plan(const std::vector<TensorDescription> &descs);

ReductionPlan make_reduction_plan(const std::vector<TensorDescription> &desc,
                                  const std::size_t axis, const bool keepdim);

ContractionPlan
make_contraction_plan_einsum(const std::vector<TensorDescription>& inputs,
                             const EinsumBinding& binding);

ContractionPlan
make_contraction_plan_einsum_out(const std::vector<TensorDescription>& descs,
                                 const EinsumBinding& binding);

#endif // BROADCAST_ITERATOR_H
