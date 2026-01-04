#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Reduction.h"

using value_type = std::ptrdiff_t;

ReductionPlan make_reduction_plan(const std::vector<TensorDescription> &descs,
                                  std::size_t axis, bool keepdim) {
   ReductionPlan plan;
   plan.num_operands = descs.size();
   plan.itemsize = descs.back().itemsize; // assume all same dtype
   plan.reduction_axis = axis;
   plan.keep_dim = keepdim;

   const TensorDescription &out_desc = descs[0];
   const TensorDescription &in_desc = descs.back();

   const std::size_t in_ndims = in_desc.ndims;

   // ----- output shape -----
   plan.out_shape = out_desc.shape;
   plan.out_ndim = plan.out_shape.size();

   // ----- build loop dims (reduction axis last) -----
   plan.loop.resize(in_ndims);

   // permutation of logical input dims
   std::vector<std::size_t> dim_order;
   dim_order.reserve(in_ndims);
   for (std::size_t d = 0; d < in_ndims; ++d) {
      if (d != axis)
         dim_order.push_back(d);
   }
   dim_order.push_back(axis); // put reduction axis last

   // helper: map input dim -> output dim index
   auto out_dim_index = [&](std::size_t dim) -> std::size_t {
      if (keepdim) {
         // output kept same rank, but axis may be size 1
         return dim;
      } else {
         // axis is removed from out_shape
         if (dim < axis)
            return dim;
         if (dim > axis)
            return dim - 1;
         // dim == axis: has no corresponding out dim
         // (we won't use out stride for reduction axis directly)
         return static_cast<std::size_t>(-1); // sentinel, not read
      }
   };

   for (std::size_t idx = 0; idx < in_ndims; ++idx) {
      const std::size_t dim = dim_order[idx];
      LoopDim ld;

      // loop size
      if (dim == axis) {
         ld.size = in_desc.shape[dim]; // full reduction length
      } else {
         const auto od = out_dim_index(dim);
         ld.size = plan.out_shape[od];
      }

      ld.stride_bytes.resize(plan.num_operands);

      // operand 0: output
      if (dim == axis) {
         ld.stride_bytes[0] = 0; // scalar along reduction axis
      } else {
         const auto od = out_dim_index(dim);
         const auto out_stride_elems = out_desc.strides[od];
         ld.stride_bytes[0] = static_cast<std::int64_t>(out_stride_elems) *
                              static_cast<std::int64_t>(plan.itemsize);
      }

      // operand 1..: inputs (for now you really only use descs[1])
      for (std::size_t op = 1; op < plan.num_operands; ++op) {
         const auto &desc = descs[op];
         const auto stride_elems = desc.strides[dim];
         ld.stride_bytes[op] = static_cast<std::int64_t>(stride_elems) *
                               static_cast<std::int64_t>(plan.itemsize);
      }

      plan.loop[idx] = std::move(ld);
   }

   return plan;
}
