#include <vector>
#include "broadcast.h"

BroadcastPlan make_broadcast_plan(const std::vector<TensorDescription>& descs) {
  // Set Broadcast plan struct info (from broadcast.h)
  BroadcastPlan plan;
  plan.num_operands = descs.size();
  plan.itemsize = descs[0].itemsize; // TODO: this line is assuming they're all the samr DTYPE

  // find maximum rank between tensors/operands - this is so we can align axis if needed
  // i.e. t1=(5,) has rank 1 and t2(5, 5) has rank 2, find rank 2 to convert t1 to (1,5) for braodcasting
  int max_ndims = 0;
  for (auto& desc : descs) {
    max_ndims = std::max(desc.ndims, max_ndims);
  }
  plan.out_ndim = max_ndims;

  // The below code is expanding the nested vector of sizes/strides
  // such that nested vector has ndim = max_ndims. Subsequently it
  // is filling in the elements with index < pad (which defines the boundary
  // between existing and padded axis) with size 1 (e.g. axis=1) and
  // stride = 0, meaning the same element will get used as the broadcast element for this
  // axis.
  //**************************
  // If we begin with operand size vectors s1 = (1, 1, 7) and s2 = (4, 7). max_ndims = 7 and
  // the resultant size vector post the below operation will be s1 =  (1, 1, 7), s2 = (1, 4, 7)
  // NB: This broadcasting is based on right alignment, the below code will asses
  // whether two operand axis are broadcastable starting with the right most axes and
  // incramenting left.
  //***************************

  std::vector<std::vector<int64_t>> sizes(descs.size());
  std::vector<std::vector<int64_t>> strides(descs.size());

  for (int op = 0; op < descs.size(); ++op) {
    auto pad = max_ndims - descs[op].ndims;
    sizes[op].resize(max_ndims);
    strides[op].resize(max_ndims);

    for (int i = 0; i < pad; ++i) {
      sizes[op][i] = 1;
      strides[op][i] = 0;
    }

    for (int i = 0; i < descs[op].ndims; ++i) {
      sizes[op][pad + i] = descs[op].sizes[i];
      strides[op][pad + i] = descs[op].strides[i];
    }
  }


  // The below routine loops through the axes (max_ndims) and operands
  // if new_dim == 1, broadcasting is allowed. If new_dim != 1 then either
  // the out_dim == 1 or out_dim == new_dim and the routine continues. Otherwise
  // a runtime error is raised as broadcasting is unachievable.
  plan.out_sizes.resize(max_ndims);
  for (int dim = 0; dim < max_ndims; ++dim) {
    int64_t out_dim = 1;
    for (int op = 0; op < plan.num_operands; ++op) {
      auto new_dim = sizes[op][dim];
      if (new_dim != 1) {
        if (out_dim != 1 && out_dim != new_dim) {
          throw std::runtime_error("Broadcast sizes mismatch");
        }
        out_dim = new_dim;
        }
    }
  plan.out_sizes[dim] = out_dim;
  }

  // The below routine loops through the calculated maximum ndim from
  // the tensor descriptions. On each iteration an instance of the LoopDim
  // struct is initialised, the size of the loop (e.g. the out_size dim) is set
  // and the stride_bytes is resized to the number of operands.
  // The operands are then looped over and stride_bytes are set per operand
  // with 0 stride for broadcastin if size == 0 and strides * itemsize if not.
  plan.loop.resize(max_ndims); // TODO: is this correct? shouldn't this be looping over the output axes?
  for (int dim = 0; dim < max_ndims; ++dim) {
    LoopDim loop_dim;
    loop_dim.size = plan.out_sizes[dim];
    loop_dim.stride_bytes.resize(plan.num_operands);
    for (int op = 0; op < plan.num_operands; ++op) {
      loop_dim.stride_bytes[op] = (sizes[op][dim] == 1) ? 0
                             : strides[op][dim] * plan.itemsize;
    }
    plan.loop[dim] = std::move(loop_dim);

  }

  return plan;

}
