#include "PlanBuilders.h"
#include "Fusion/core/planning/analysis/ContractionAnalysis.h"

namespace fusion::planning {
namespace {
KernelHints make_kernel_hints(const std::vector<OperandDescription> &descs) {
   KernelHints hints;
   hints.all_contiguous_like =
       std::ranges::all_of(descs, [](const OperandDescription &desc) {
          return desc.layout == LayoutKind::Dense;
       });
   return hints;
}

std::vector<std::size_t> get_output_shape_from_indices(const IndexSpaceIR &ir) {
   std::vector<std::size_t> out_shape;
   out_shape.resize(ir.out_indices.size());
   for (std::size_t i = 0; i < ir.out_indices.size(); ++i) {
      const std::uint32_t id = ir.out_indices[i];
      out_shape[i] = ir.indices[id].extent;
   }
   return out_shape;
}

PlanCore make_plan_core(const ExprKind expr, const TraversalKind traversal,
                        const IndexSpaceIR &ir,
                        std::vector<std::size_t> out_shape) {
   PlanCore plan;
   plan.expr = expr;
   plan.traversal_kind = traversal;
   plan.num_operands = ir.num_operands;
   plan.itemsize = ir.itemsize;
   plan.out_shape = std::move(out_shape);
   plan.out_ndim = plan.out_shape.size();
   return plan;
}

DenseTraversalPlan
make_dense_traversal_plan(const IndexSpaceIR &ir,
                          const std::vector<OperandDescription> &descs,
                          const std::vector<std::uint32_t> &loop_order,
                          const std::vector<IndexRole> *role_of_id = nullptr) {
   DenseTraversalPlan plan;
   plan.loop = lower_to_loops(ir, descs, loop_order,
                              role_of_id); // TODO: check fn overloads
   return plan;
};

AccessPlan make_access_plan(const IndexSpaceIR &ir,
                            const std::vector<OperandDescription> &descs,
                            const std::vector<std::uint32_t> &loop_order) {
   AccessPlan plan;
   plan.operands = lower_operand_access(ir, descs, loop_order);
   return plan;
};

ExecutionPlan
make_dense_execution_plan(const ExprKind expr, const IndexSpaceIR &ir,
                          const std::vector<OperandDescription> &descs,
                          std::vector<std::size_t> out_shape,
                          const std::vector<std::uint32_t> &loop_order,
                          const std::vector<IndexRole> *role_of_id = nullptr) {
   ExecutionPlan exec;
   exec.core =
       make_plan_core(expr, TraversalKind::Dense, ir, std::move(out_shape));
   exec.traversal =
       make_dense_traversal_plan(ir, descs, loop_order, role_of_id);
   exec.access = make_access_plan(ir, descs, loop_order);

   exec.hints = make_kernel_hints(descs);
   return exec;
}

std::vector<std::uint32_t> make_reduction_loop_order(const IndexSpaceIR &ir,
                                                     const std::size_t axis) {
   std::vector<std::uint32_t> loop_order;
   loop_order.reserve(ir.indices.size());
   for (std::uint32_t id : ir.out_indices) {
      loop_order.push_back(id);
   }
   loop_order.push_back(static_cast<std::uint32_t>(axis));
   return loop_order;
}

std::vector<std::uint32_t> make_contraction_loop_order(const IndexSpaceIR &ir) {
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
   return loop_order;
}

} // unnamed namespace

ContractionPlan
make_contraction_plan_einsum_out(const std::vector<OperandDescription> &descs,
                                 const OperandLabelBinding &binding) {
   if (descs.size() != 3) {
      throw std::runtime_error("einsum_out: expected descs = {out, A, B}");
   }

   constexpr ItemSizeGroupConstraint constraint =
       ItemSizeGroupConstraint::HomogeneousItemSize;

   IndexSpaceIR ir = build_ir_from_label_binding(descs, binding, constraint);

   const std::vector<std::size_t> expected = out_shape_from_ir(ir);
   if (descs.front().shape != expected) {
      throw std::runtime_error(
          "einsum_out: out.shape does not match inferred out shape");
   }

   const std::vector<std::uint32_t> loop_order =
       make_contraction_loop_order(ir);
   const std::vector<IndexRole> role_of_id =
       compute_roles_for_gemm_like(ir, binding);

   ContractionPlan plan;
   plan.exec = make_dense_execution_plan(ExprKind::Contraction, ir, descs,
                                         get_output_shape_from_indices(ir),
                                         loop_order, &role_of_id);
   if (std::optional<GemmLikeDesc> gemm =
           analysis::analyse_gemm_like_contraction(plan.exec)) {
      plan.exec.hints.gemm_like = true;
      plan.exec.hints.gemm = *gemm;
   }

   return plan;
}

ContractionPlan
make_contraction_plan_einsum(const std::vector<OperandDescription> &inputs,
                             const OperandLabelBinding &binding) {
   if (inputs.size() != 2) {
      throw std::runtime_error("einsum: expected inputs = {A, B}");
   }

   constexpr ItemSizeGroupConstraint constraint =
       ItemSizeGroupConstraint::HomogeneousItemSize;

   OperandDescription dummy_out;
   dummy_out.shape.assign(binding.out_labels.size(), 1);
   dummy_out.strides.assign(dummy_out.ndims(), 0);
   dummy_out.itemsize = inputs.front().itemsize;

   std::vector<OperandDescription> tmp = {dummy_out, inputs.front(),
                                          inputs.back()};
   IndexSpaceIR ir = build_ir_from_label_binding(tmp, binding, constraint);

   const std::vector<std::size_t> out_shape = out_shape_from_ir(ir);

   OperandDescription out_desc;
   out_desc.shape = out_shape;

   out_desc.strides.assign(out_desc.ndims(), 0);

   out_desc.itemsize = inputs.front().itemsize;

   std::vector<OperandDescription> descs = {out_desc, inputs.front(),
                                            inputs.back()};
   return make_contraction_plan_einsum_out(descs, binding);
}

ReductionPlan make_reduction_plan(const std::vector<OperandDescription> &descs,
                                  const std::size_t axis, const bool keepdim) {
   if (descs.size() < 2) {
      throw std::runtime_error("reduction: expected at least {out, in}");
   }

   const OperandDescription &in_desc = descs.back();
   const std::size_t ax_norm =
       norm_axis(static_cast<std::int64_t>(axis), in_desc.ndims());

   constexpr ItemSizeGroupConstraint constraint =
       ItemSizeGroupConstraint::HomogeneousItemSize;
   const IndexSpaceIR ir =
       build_reduction_ir(descs, ax_norm, keepdim, constraint);
   const std::vector<std::uint32_t> loop_order =
       make_reduction_loop_order(ir, ax_norm);

   ReductionPlan plan;
   plan.exec = make_dense_execution_plan(ExprKind::Reduction, ir, descs,
                                         descs.front().shape, loop_order);
   plan.keep_dim = keepdim;
   plan.reduction_axis = ax_norm;
   return plan;
}

ElementwisePlan
make_elementwise_plan(const std::vector<OperandDescription> &descs) {
   constexpr ItemSizeGroupConstraint constraint =
       ItemSizeGroupConstraint::HomogeneousItemSize;
   const IndexSpaceIR ir =
       build_elementwise_ir_right_aligned(descs, constraint);
   const std::vector<std::uint32_t> &loop_order = ir.out_indices;
   ElementwisePlan plan;
   plan.exec =
       make_dense_execution_plan(ExprKind::Elementwise, ir, descs,
                                 get_output_shape_from_indices(ir), loop_order);
   return plan;
}

} // namespace fusion::planning