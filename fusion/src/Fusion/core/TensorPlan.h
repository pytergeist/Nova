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

struct LoopDim {
   std::size_t size;
   std::vector<std::int64_t> stride_bytes;
   LoopKind kind;
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

   // For each operand: output axis -> input axis, or -1 if padded
   std::vector<std::vector<int>> axis_map;

   // For each operand: stride bytes per output axis (0 if broadcasted)
   std::vector<std::vector<std::int64_t>> stride_bytes;
};

struct BroadcastPlan {
   std::size_t num_operands;
   std::size_t out_ndim;
   std::vector<std::size_t> out_shape;
   std::vector<LoopDim> loop;

   bool all_contiguous_like{false}; // curr not used - evaluate
   std::size_t vector_bytes{0};

   std::size_t itemsize;
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
   std::size_t num_operands;
   std::size_t out_ndim;
   std::vector<std::size_t> out_shape;

   ContractionAxes caxes;
   std::vector<LoopDim> loop;
   bool all_contiguous_like{false}; // curr not used - evaluate
   bool lhs_tranpose{false};
   bool rhs_tranpose{false};
   std::size_t vector_bytes{0};

   std::size_t itemsize;
};

BroadcastPlan make_broadcast_plan(const std::vector<TensorDescription> &descs);

ReductionPlan make_reduction_plan(const std::vector<TensorDescription> &desc,
                                  const std::size_t axis, const bool keepdim);

ContractionPlan
make_contraction_plan(const std::vector<TensorDescription> &descs,
                      const ContractionAxes axes);

#endif // BROADCAST_ITERATOR_H
