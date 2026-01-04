#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "TensorPlan.h"

using value_type = std::ptrdiff_t;

BroadcastPlan make_broadcast_plan(const std::vector<TensorDescription> &descs) {
   // Set Broadcast plan struct info (from Broadcast.h)
   BroadcastPlan plan;
   plan.num_operands = descs.size();
   plan.itemsize =
       descs[0]
           .itemsize; // TODO: this line is assuming they're all the samr DTYPE

   // find maximum rank between tensors/operands - this is so we can align axis
   // if needed i.e. t1=(5,) has rank 1 and t2(5, 5) has rank 2, find rank 2 to
   // convert t1 to (1,5) for braodcasting
   std::size_t max_ndims = 0;
   for (const auto &desc : descs) {
      max_ndims = std::max(desc.ndims, max_ndims);
   }
   plan.out_ndim = max_ndims;

   // The below code is expanding the nested vector of sizes/strides
   // such that nested vector has ndim = max_ndims. Subsequently it
   // is filling in the elements with index < pad (which defines the boundary
   // between existing and padded axis) with size 1 (e.g. axis=1) and
   // stride = 0, meaning the same element will get used as the broadcast
   // element for this axis.
   //**************************
   // If we begin with operand shape vectors s1 = (1, 1, 7) and s2 = (4, 7).
   // max_ndims = 7 and the resultant shape vector post the below operation will
   // be s1 =  (1, 1, 7), s2 = (1, 4, 7) NB: This broadcasting is based on right
   // alignment, the below code will asses whether two operand axis are
   // broadcastable starting with the right most axes and incramenting left.
   //***************************

   // TODO: renaim shape to shapes (in constitutes both shapes in a vec
   std::vector<std::vector<std::size_t>> shape(descs.size());
   std::vector<std::vector<std::int64_t>> strides(descs.size());

   for (std::size_t op = 0; op < descs.size(); ++op) {
      // left pad the axes with 1's (e.g. right axes aligment)
      auto pad = max_ndims - descs[op].ndims;
      shape[op].resize(max_ndims);
      strides[op].resize(max_ndims);

      for (std::size_t i = 0; i < pad; ++i) {
         shape[op][i] = 1;
         strides[op][i] = 0;
      }

      for (std::size_t i = 0; i < descs[op].ndims; ++i) {
         shape[op][pad + i] = descs[op].shape[i];
         strides[op][pad + i] = descs[op].strides[i];
      }
   }

   // The below routine loops through the axes (max_ndims) and operands
   // if new_dim == 1, broadcasting is allowed. If new_dim != 1 then either
   // the out_dim == 1 or out_dim == new_dim and the routine continues.
   // Otherwise a runtime error is raised as broadcasting is unachievable.
   plan.out_shape.resize(max_ndims);
   for (std::size_t dim = 0; dim < max_ndims; ++dim) {
      std::size_t out_dim = 1;
      for (std::size_t op = 0; op < plan.num_operands; ++op) {
         std::size_t new_dim = shape[op][dim];
         if (new_dim !=
             1) { // TODO: this is where the problem lies - we're setting the
            // out dim to 7 here (the correct new dim) but we have a cached 64
            // from previous loop iterations - is this a loop cycle? because 7
            // is set to the out dim and then on next iteration we pull 64 for
            // the new dim? need to lop the tensor descs
            if (out_dim != 1 && out_dim != new_dim) {
               std::cout << "out dim: " << out_dim << " new_dim: " << new_dim
                         << '\n';
               throw std::runtime_error("Broadcast sizes mismatch");
            }
            out_dim = new_dim;
         }
      }
      plan.out_shape[dim] = out_dim;
   }

   // The below routine loops through the calculated maximum ndim from
   // the tensor descriptions. On each iteration an instance of the LoopDim
   // struct is initialised, the size of the loop (e.g. the out_size dim) is set
   // and the stride_bytes is resized to the number of operands.
   // The operands are then looped over and stride_bytes are set per operand
   // with 0 stride for broadcastin if size == 0 and strides * itemsize if not.
   plan.loop.resize(max_ndims);
   for (std::size_t dim = 0; dim < max_ndims; ++dim) {
      LoopDim loop_dim;
      loop_dim.size = plan.out_shape[dim];
      loop_dim.stride_bytes.resize(plan.num_operands);
      for (std::size_t op = 0; op < plan.num_operands; ++op) {
         loop_dim.stride_bytes[op] =
             (shape[op][dim] == 1)
                 ? value_type{0}
                 : static_cast<value_type>(
                       static_cast<long long>(strides[op][dim]) *
                       static_cast<long long>(plan.itemsize));
      }
      plan.loop[dim] = std::move(loop_dim);
   }

   return plan;
}

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

   plan.out_shape = out_desc.shape;
   plan.out_ndim = plan.out_shape.size();

   plan.loop.resize(in_ndims);

   std::vector<std::size_t> dim_order;
   dim_order.reserve(in_ndims);
   for (std::size_t d = 0; d < in_ndims; ++d) {
      if (d != axis)
         dim_order.push_back(d);
   }
   dim_order.push_back(axis);

   auto out_dim_index = [&](std::size_t dim) -> std::size_t {
      if (keepdim) {
         return dim;
      } else {
         if (dim < axis)
            return dim;
         if (dim > axis)
            return dim - 1;
         return static_cast<std::size_t>(-1);
      }
   };

   for (std::size_t idx = 0; idx < in_ndims; ++idx) {
      const std::size_t dim = dim_order[idx];
      LoopDim ld;

      if (dim == axis) {
         ld.size = in_desc.shape[dim];
      } else {
         const auto od = out_dim_index(dim);
         ld.size = plan.out_shape[od];
      }

      ld.stride_bytes.resize(plan.num_operands);

      if (dim == axis) {
         ld.stride_bytes[0] = 0;
      } else {
         const auto od = out_dim_index(dim);
         const auto out_stride_elems = out_desc.strides[od];
         ld.stride_bytes[0] = static_cast<std::int64_t>(out_stride_elems) *
                              static_cast<std::int64_t>(plan.itemsize);
      }

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
