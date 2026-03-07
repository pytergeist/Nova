#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "TensorPlan.h"
#include "fuir/IR.h"

BroadcastPlan
make_broadcast_plan(const std::vector<OperandDescription> &descs) {
   IndexSpaceIR ir = build_broadcast_ir_right_aligned(descs);

   BroadcastPlan plan;
   plan.num_operands = ir.num_operands;
   plan.itemsize = ir.itemsize;

   plan.out_ndim = ir.out_indices.size();
   plan.out_shape.resize(plan.out_ndim);
   for (std::size_t i = 0; i < plan.out_ndim; ++i) {
      const std::uint32_t id = ir.out_indices[i];
      plan.out_shape[i] = ir.indices[id].extent;
   }

   const std::vector<std::uint32_t> &loop_order = ir.out_indices;

   plan.loop = lower_to_loops(ir, descs, loop_order);

   return plan;
}

ReductionPlan make_reduction_plan(const std::vector<OperandDescription> &descs,
                                  std::size_t axis, bool keepdim) {
   if (descs.size() < 2)
      throw std::runtime_error("reduction: expected at least {out, in}");

   const OperandDescription &in_desc = descs.back();
   const std::size_t in_nd = in_desc.ndims();
   const std::size_t ax = norm_axis(static_cast<std::int64_t>(axis), in_nd);

   IndexSpaceIR ir = build_reduction_ir(descs, ax, keepdim);

   ReductionPlan plan;
   plan.num_operands = descs.size();
   plan.itemsize = ir.itemsize;
   plan.keep_dim = keepdim;
   plan.reduction_axis = ax;

   plan.out_ndim = descs[0].ndims();
   plan.out_shape = descs[0].shape;

   std::vector<std::uint32_t> loop_order;
   loop_order.reserve(ir.indices.size());

   for (std::uint32_t id : ir.out_indices)
      loop_order.push_back(id);

   loop_order.push_back(static_cast<std::uint32_t>(ax));

   plan.loop = lower_to_loops(ir, descs, loop_order);

   return plan;
}

ContractionPlan
make_contraction_plan_einsum_out(const std::vector<OperandDescription> &descs,
                                 const EinsumBinding &binding) {
   if (descs.size() != 3) {
      throw std::runtime_error("einsum_out: expected descs = {out, A, B}");
   }
   validate_descs_same_itemsize(descs);

   IndexSpaceIR ir = build_ir_from_einsum_binding(descs, binding);

   const std::vector<std::size_t> expected = out_shape_from_ir(ir);
   if (descs[0].shape != expected) {
      throw std::runtime_error(
          "einsum_out: out.shape does not match inferred out shape");
   }

   const std::vector<std::uint32_t> outer_order = ir.out_indices;

   std::vector<std::uint32_t> reduce_order;
   reduce_order.reserve(ir.indices.size());
   for (std::uint32_t id = 0;
        id < static_cast<std::uint32_t>(ir.indices.size()); ++id) {
      if (ir.indices[id].kind == IndexKind::Reduction) {
         reduce_order.push_back(id);
      }
   }

   std::vector<std::uint32_t> loop_order;
   loop_order.reserve(outer_order.size() + reduce_order.size());
   loop_order.insert(loop_order.end(), outer_order.begin(), outer_order.end());
   loop_order.insert(loop_order.end(), reduce_order.begin(),
                     reduce_order.end());

   ContractionPlan plan;
   plan.num_operands = descs.size();
   plan.itemsize = ir.itemsize;

   plan.out_ndim = descs[0].ndims();
   plan.out_shape = descs[0].shape;

   const auto role_of_id = compute_roles_for_gemm_like(ir, binding);
   plan.loop = lower_to_loops(ir, descs, loop_order, &role_of_id);

   plan.gemm_like = true;
   plan.gemm = GemmLikeDesc{};

   std::size_t batch = 1, M = 1, N = 1, K = 1;
   int m_count = 0, n_count = 0, k_count = 0;

   for (const auto &ld : plan.loop) {
      switch (ld.role) {
      case IndexRole::Batch:
         batch *= ld.size;
         break;
      case IndexRole::M:
         M = ld.size;
         ++m_count;
         break;
      case IndexRole::N:
         N = ld.size;
         ++n_count;
         break;
      case IndexRole::K:
         K = ld.size;
         ++k_count;
         break;
      }
   }

   if (!(m_count == 1 && n_count == 1 && k_count == 1)) {
      plan.gemm_like = false;
      return plan;
   }

   plan.gemm.batch = batch;
   plan.gemm.M = M;
   plan.gemm.N = N;
   plan.gemm.K = K;

   const std::int64_t item = static_cast<std::int64_t>(plan.itemsize);

   std::int64_t out_m = 0, out_n = 0;
   std::int64_t a_m = 0, a_k = 0;
   std::int64_t b_k = 0, b_n = 0;

   for (const auto &ld : plan.loop) {
      if (ld.role == IndexRole::M) {
         out_m = static_cast<std::int64_t>(ld.stride_bytes[0]) / item;
         a_m = static_cast<std::int64_t>(ld.stride_bytes[1]) / item;
      } else if (ld.role == IndexRole::N) {
         out_n = static_cast<std::int64_t>(ld.stride_bytes[0]) / item;
         b_n = static_cast<std::int64_t>(ld.stride_bytes[2]) / item;
      } else if (ld.role == IndexRole::K) {
         a_k = static_cast<std::int64_t>(ld.stride_bytes[1]) / item;
         b_k = static_cast<std::int64_t>(ld.stride_bytes[2]) / item;
      }
   }

   plan.gemm.out_rs = out_m;
   plan.gemm.out_cs = out_n;

   plan.gemm.a_rs = a_m;
   plan.gemm.a_cs = a_k;

   plan.gemm.b_rs = b_k;
   plan.gemm.b_cs = b_n;

   if (plan.gemm.out_rs == 0 || plan.gemm.out_cs == 0 || plan.gemm.a_rs == 0 ||
       plan.gemm.a_cs == 0 || plan.gemm.b_rs == 0 || plan.gemm.b_cs == 0) {
      plan.gemm_like = false;
      return plan;
   }

   return plan;
}

ContractionPlan
make_contraction_plan_einsum(const std::vector<OperandDescription> &inputs,
                             const EinsumBinding &binding) {
   if (inputs.size() != 2) {
      throw std::runtime_error("einsum: expected inputs = {A, B}");
   }
   validate_descs_same_itemsize(inputs);

   OperandDescription dummy_out;
   dummy_out.shape.assign(binding.out_labels.size(), 1);
   dummy_out.strides.assign(dummy_out.ndims(), 0);
   dummy_out.itemsize = inputs[0].itemsize;

   std::vector<OperandDescription> tmp = {dummy_out, inputs[0], inputs[1]};
   IndexSpaceIR ir = build_ir_from_einsum_binding(tmp, binding);

   const std::vector<std::size_t> out_shape = out_shape_from_ir(ir);

   OperandDescription out_desc;
   out_desc.shape = out_shape;

   out_desc.strides.assign(out_desc.ndims(), 0);

   out_desc.itemsize = inputs[0].itemsize;

   std::vector<OperandDescription> descs = {out_desc, inputs[0], inputs[1]};
   return make_contraction_plan_einsum_out(descs, binding);
}
