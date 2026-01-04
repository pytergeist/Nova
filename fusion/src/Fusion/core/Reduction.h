#ifndef REDUCTION_H
#define REDUCTION_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "Broadcast.h"

struct ReductionPlan {
   std::size_t num_operands;
   std::size_t out_ndim;
   std::vector<std::size_t> out_shape;
   std::size_t reduction_axis;
   std::vector<LoopDim> loop;

   bool keep_dim{false};
   bool all_contiguous_like{false};
   std::size_t vector_bytes{0};

   std::size_t itemsize;
};

ReductionPlan make_reduction_plan(const std::vector<TensorDescription> &desc,
                                  const std::size_t axis, const bool keepdim);

#endif // REDUCTION_H
