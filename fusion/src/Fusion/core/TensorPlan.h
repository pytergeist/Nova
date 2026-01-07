#ifndef BROADCAST_ITERATOR_H
#define BROADCAST_ITERATOR_H

#include <cstddef>
#include <cstdint>
#include <vector>

struct TensorDescription {
   std::size_t ndims;
   std::vector<std::size_t> shape;
   std::vector<std::int64_t> strides;
   std::size_t itemsize;
};

struct LoopDim {
   std::size_t size;
   std::vector<std::int64_t> stride_bytes;
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

   bool keep_dim{false};
   bool all_contiguous_like{false}; // curr not used - evaluate
   bool lhs_tranpose{false};
   bool rhs_tranpose{false};
   std::size_t vector_bytes{0};

   std::size_t itemsize;
};

BroadcastPlan make_broadcast_plan(const std::vector<TensorDescription> &descs);

ReductionPlan make_reduction_plan(const std::vector<TensorDescription> &desc,
                                  const std::size_t axis, const bool keepdim);

ContractionPlan make_contraction_plan(const std::vector<TensorDescription> &descs);

#endif // BROADCAST_ITERATOR_H
